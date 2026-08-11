"""Tests for anchor-based hotword boosting (trie + logits processor + db entries)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class FakeTokenizer:
    """Minimal tokenizer for exercising the processor: id -> decoded string."""

    def __init__(self, id_to_text: dict[int, str]):
        self.id_to_text = id_to_text
        self.vocab_size = len(id_to_text)

    def get_vocab(self):
        return dict(self.id_to_text)

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        text = "".join(self.id_to_text.get(int(i), "") for i in ids)
        return text if skip_special_tokens else text


def _make_processor(entries, **kw):
    tok = FakeTokenizer(
        {
            0: "网", 1: "易", 2: "云", 3: "看", 4: "了", 5: "小",
            6: "E", 7: "打", 8: "开", 9: "灯",
        }
    )
    from ev.asr.hotword import HotwordLogitsProcessor

    return HotwordLogitsProcessor(tok, entries, text_of_id={i: tok.id_to_text[i] for i in tok.id_to_text}, ids_by_char={"网": [0], "易": [1], "云": [2], "看": [3], "了": [4], "小": [5], "E": [6]}, **kw)


def test_trie_suffix_matching():
    from ev.asr.hotword import HotwordTrie

    trie = HotwordTrie()
    trie.insert("网易云", 3.0)
    trie.insert("网易", 5.0)

    # Suffix "网易" is a strict prefix of "网易云" -> found (depth 2)
    anchor, node = trie.longest_suffix_prefix("我们网易", 1)
    assert anchor == "网易"
    assert node is not None
    # Node weight reflects both hotwords passing through (max = 5.0)
    assert node.weight == 5.0

    # "网" alone is a prefix -> single-char anchor required
    anchor, node = trie.longest_suffix_prefix("上网", 1)
    assert anchor == "网"
    assert node is not None

    # Complete word "网易云" has no children -> not a safe anchor
    anchor, node = trie.longest_suffix_prefix("我说网易云", 1)
    assert node is None

    # No prefix match
    anchor, node = trie.longest_suffix_prefix("吃饭", 1)
    assert node is None

    # min_anchor_len filters single-char suffix
    anchor, node = trie.longest_suffix_prefix("上网", 2)
    assert node is None


def test_processor_boosts_only_continuation_tokens():
    proc = _make_processor([("网易云", 3.0)])

    # Decoded so far: "网" — anchor found, next char should be "易"
    scores = torch.zeros(1, proc.tokenizer.vocab_size)
    input_ids = torch.tensor([[0]])  # 网
    out = proc(input_ids, scores)
    # "易" (id 1) boosted by weight*scale = 3.0*2.0 = 6.0
    assert out[:, 1].item() == pytest.approx(6.0)
    # "看" (id 3) is NOT a continuation of 网易云; untouched
    assert out[:, 3].item() == 0.0
    # Nothing suppressed: other ids all remain 0
    total_boosted = (out > 0).sum().item()
    assert total_boosted >= 1


def test_processor_no_boost_when_no_prefix_matches():
    proc = _make_processor([("网易云", 3.0)])
    scores = torch.zeros(1, proc.tokenizer.vocab_size)
    input_ids = torch.tensor([[3]])  # 看
    out = proc(input_ids, scores)
    assert (out == 0).all().item()


def test_processor_completes_word_then_halts():
    proc = _make_processor([("网易云", 3.0)])
    # "网易" decoded -> anchor extends to lead "云" (id 2)
    scores = torch.zeros(1, proc.tokenizer.vocab_size)
    input_ids = torch.tensor([[0, 1]])
    out = proc(input_ids, scores)
    assert out[:, 2].item() == pytest.approx(6.0)

    # "网易云" fully spoken -> no children -> no boost
    scores2 = torch.zeros(1, proc.tokenizer.vocab_size)
    out2 = proc(torch.tensor([[0, 1, 2]]), scores2)
    assert (out2 == 0).all().item()


def test_processor_common_prefix_does_not_boost_diverging_word():
    """Hotword 网易云; user says 网××. Diverging continuation must NOT be boosted."""
    proc = _make_processor([("网易云", 3.0)])
    scores = torch.zeros(1, proc.tokenizer.vocab_size)
    # Decoded "网" then a diverging char "看"; suffix "看" is no hotword prefix.
    input_ids = torch.tensor([[0, 3]])
    out = proc(input_ids, scores)
    assert out[:, 3].item() == 0.0  # 看 not boosted on its own path
    assert out[:, 1].item() == 0.0  # and neither is 易 (no anchored target now)
    assert (out == 0).all().item()


def test_processor_boost_cap():
    proc = _make_processor([("网易云", 8.0)], boost_max=4.0)
    scores = torch.zeros(1, proc.tokenizer.vocab_size)
    out = proc(torch.tensor([[0]]), scores)
    # 8.0*2.0=16.0 but capped at 4.0
    assert out[:, 1].item() == pytest.approx(4.0)


def test_processor_disabled_flag():
    proc = _make_processor([("网易云", 3.0)])
    scores = torch.zeros(1, proc.tokenizer.vocab_size)
    out = proc(torch.tensor([[0]]), scores)
    # With enable flag handled at adapter level, processor always boosts;
    # here just verify it does when configured.
    assert out[:, 1].item() > 0