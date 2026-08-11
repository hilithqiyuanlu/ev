"""Anchor-based hotword boosting for Qwen3-ASR decoding.

Design goal: raise recall for dictionary (lexicon) words while disturbing normal
speech as little as possible — especially common expressions that merely *share a
prefix* with a hotword.

The mechanism is deliberately conservative:

- It only ever ADDS logits. It never suppresses or penalises any token, so a
  strong acoustic candidate can never be forced out.
- It is anchored: boosting fires only when the text decoded so far already ends
  with a (strict) prefix of some hotword. Speech that does not start like a
  hotword is completely untouched.
- The boost is bounded by ``boost_max`` and scaled by the hotword weight, so it
  is a nudge, not a hard override.
- Tokens are boosted only when their decoded text continues the anchor along an
  actual hotword path (checked word-by-word through the trie). A common word
  like 「网站」next to a hotword 「网易云」never gets its 「站」boosted — only 「易」.

Because the decoder input ids for an encoder-decoder ASR model are the transcript
tokens (prompt tags are not part of it), matching is done against the raw decoder
token stream, normalized (casefold + whitespace-stripped).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize for prefix matching: NFKC, strip whitespace, casefold."""
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub("", text)
    return text.casefold()


class HotwordTrie:
    """Character-level prefix trie over hotwords, carrying per-node max weight."""

    __slots__ = ("children", "weight", "depth")

    def __init__(self, depth: int = 0) -> None:
        self.children: Dict[str, "HotwordTrie"] = {}
        self.weight: float = 0.0
        self.depth: int = depth

    def insert(self, word: str, weight: float) -> None:
        node = self
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = HotwordTrie(node.depth + 1)
                node.children[ch] = nxt
            node = nxt
            if weight > node.weight:
                node.weight = weight

    @property
    def max_depth(self) -> int:
        children = self.children.values()
        return max((1 + c.max_depth for c in children), default=0)

    def longest_suffix_prefix(
        self, text: str, min_anchor_len: int
    ) -> Tuple[str, "HotwordTrie | None"]:
        """Find the longest suffix of ``text`` that is a strict prefix of a hotword.

        A qualifying suffix must run all the way to the end of ``text`` (the
        decoded stream tail) yet still end at a node with children. This keeps
        the anchor genuinely "in progress" — once the model has committed to a
        diverging continuation, the anchor dissolves. Returns ``(text_of_suffix,
        node)`` or ``("", None)`` when nothing is safe to boost.
        """
        best_node: HotwordTrie | None = None
        n = len(text)
        # Longest suffix == smallest start index that walks into the trie all
        # the way to the end of text while remaining extendable.
        for start in range(n):
            node = self
            i = start
            while i < n:
                nxt = node.children.get(text[i])
                if nxt is None:
                    break
                node = nxt
                i += 1
            if i == n and node.depth >= min_anchor_len and node.children:
                best_node = node
                break
        if best_node is None:
            return "", None
        return text[n - best_node.depth:], best_node


def build_character_index(
    tokenizer, needed_chars: set[str]
) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """Scan the tokenizer vocab once, returning per-token text and char→token ids.

    Only tokens whose normalized decoded text starts with a needed character are
    kept, which bounds memory and (one-time) cost to the hotword character set.
    """
    text_of_id: Dict[int, str] = {}
    ids_by_char: Dict[str, List[int]] = {}
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        count = len(vocab)
    else:
        count = tokenizer.vocab_size
    for tid in range(count):
        try:
            raw = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            continue
        s = normalize_text(raw)
        if not s or s[0] not in needed_chars:
            continue
        text_of_id[tid] = s
        ids_by_char.setdefault(s[0], []).append(tid)
    return text_of_id, ids_by_char


class HotwordLogitsProcessor:
    """PyTorch logits processor adding anchored, positive-only hotword boosts.

    Attach via ``model.generate(..., logits_processor=LogitsProcessorList([...]))``.
    """

    def __init__(
        self,
        tokenizer,
        entries: List[Tuple[str, float]],
        *,
        boost_scale: float = 2.0,
        boost_max: float = 6.0,
        min_anchor_len: int = 1,
        text_of_id: Dict[int, str] | None = None,
        ids_by_char: Dict[str, List[int]] | None = None,
    ):
        self.tokenizer = tokenizer
        self.trie = HotwordTrie()
        for word, weight in entries:
            self.trie.insert(normalize_text(word), max(0.5, float(weight)))
        self.boost_scale = max(0.0, float(boost_scale))
        self.boost_max = max(0.0, float(boost_max))
        self.min_anchor_len = max(1, int(min_anchor_len))
        self.text_of_id = text_of_id or {}
        self.ids_by_char = ids_by_char or {}
        self._root_chars = set(self.trie.children.keys())
        # Suffix anchors are bounded by the longest hotword, so a short tail
        # window of tokens is enough to reconstruct them.
        self._tail_window = min(64, max(self.trie.max_depth * 2 + 6, 8))

    def _cached_index(self) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
        if not self.text_of_id:
            self.text_of_id, self.ids_by_char = build_character_index(
                self.tokenizer, self._root_chars | self._all_chars()
            )
        return self.text_of_id, self.ids_by_char

    def ensure_index(self) -> None:
        """Eagerly build (or reuse) the tokenizer index; used by adapter caching."""
        self._cached_index()

    def _all_chars(self) -> set[str]:
        chars: set[str] = set()
        stack: List[HotwordTrie] = [self.trie]
        while stack:
            node = stack.pop()
            for ch, nxt in node.children.items():
                chars.add(ch)
                stack.append(nxt)
        return chars

    def _decoded_tail(self, input_ids) -> str:
        tokens = input_ids
        if tokens.ndim > 1:
            tokens = tokens[-1]
        n = len(tokens)
        win = min(n, self._tail_window)
        try:
            ids = tokens[-win:] if win else tokens
            raw = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        except Exception:
            return ""
        return normalize_text(raw)

    def __call__(self, input_ids, scores):
        if not self.trie.children:
            return scores
        text = self._decoded_tail(input_ids)
        anchor, node = self.trie.longest_suffix_prefix(text, self.min_anchor_len)
        if node is None or not node.children:
            return scores
        text_of_id, ids_by_char = self._cached_index()
        boost = min(self.boost_max, node.weight * self.boost_scale)
        if boost <= 0:
            return scores
        for ch in node.children:
            ids = ids_by_char.get(ch)
            if not ids:
                continue
            for tid in ids:
                t = text_of_id[tid]
                if not t:
                    continue
                # Only boost tokens that actually continue the anchor along a
                # hotword path — never tokens that branch off into other words.
                sub = node
                ok = True
                for c in t:
                    nxt = sub.children.get(c)
                    if nxt is None:
                        ok = False
                        break
                    sub = nxt
                if ok:
                    scores[:, tid] = scores[:, tid] + boost
        return scores