"""EV 文本唤醒匹配与 query 候选生成。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_LEADING_FILLERS = (
    "嗯", "啊", "呃", "哦", "诶", "唉", "哈", "喂", "哎", "噢",
    "那个", "这个", "就是", "然后", "所以", "对了", "哎呀", "啊对",
)

_PREFIX_GREETINGS = (
    "嗨", "喂", "哎", "诶", "噢", "哦",
)

# ASR 常把"嗨"转写成英文/拼音问候（如 "hi 小易"、"hai 小易"）。需大小写不敏感剥离，
# 但要求后随边界（空白/标点/CJK），避免误剥 "history"、"hey" 之外以同样字母开头的普通词。
_EN_PREFIX_GREETINGS = ("hi", "hai", "hay", "hey", "hei")

_SMALL_E_HOMOPHONES = (
    # Common ASR misrecognitions for "小E" - kept minimal to avoid false triggers
    # Only includes characters that:
    # 1. Paraformer frequently confuses with "E" pronunciation
    # 2. Do NOT form common everyday words when preceded by "小"
    "易", "艺",
)


@dataclass(frozen=True)
class WakeMatch:
    detected: bool
    query_text: str
    had_leading_filler: bool = False


@dataclass(frozen=True)
class QueryDecision:
    wake_detected: bool
    query_candidate: bool
    query_text: str | None
    had_leading_filler: bool = False


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"[\s\u3000]+", " ", text)
    return text


def _is_cjk_char(ch: str) -> bool:
    """Whether a single character is CJK (used for English greeting word boundary)."""
    return "一" <= ch <= "鿿"


def _strip_leading_greetings(text: str) -> tuple[str, bool]:
    """Strip leading greeting words (嗨/喂/哎 etc.) even when directly followed by wake word without space."""
    stripped = text.lstrip()
    had_greeting = False
    changed = True
    while changed:
        changed = False
        for greet in sorted(_PREFIX_GREETINGS, key=len, reverse=True):
            if stripped.startswith(greet):
                stripped = stripped[len(greet):].lstrip(" ，。！？、；：,.!?;:")
                had_greeting = True
                changed = True
                break
        if changed:
            continue
        # English/拼音 greetings: case-insensitive, require a boundary after
        # (whitespace / punctuation / CJK) so we don't clip words like "history".
        low = stripped.casefold()
        for egreet in _EN_PREFIX_GREETINGS:
            if low.startswith(egreet):
                after = stripped[len(egreet):]
                if not after or after[0].isspace() or _is_cjk_char(after[0]) or after[0] in "，。！？、；：,.!?;：":
                    stripped = after.lstrip(" ，。！？、；：,.!?;:")
                    had_greeting = True
                    changed = True
                    break
    return stripped, had_greeting


def _strip_leading_fillers(text: str) -> tuple[str, bool]:
    """Strip leading filler words, return (stripped, had_filler)."""
    stripped = text.lstrip()
    had_filler = False
    changed = True
    while changed:
        changed = False
        for filler in sorted(_LEADING_FILLERS, key=len, reverse=True):
            if stripped.startswith(filler):
                stripped = stripped[len(filler):].lstrip(" ，。！？、；：,.!?;:")
                had_filler = True
                changed = True
                break
    return stripped, had_filler


def match_wake_prefix(text: str, aliases: tuple[str, ...] = ("小E",)) -> WakeMatch:
    normalized = normalize_text(text)
    # First strip greeting prefixes (嗨/喂/哎) even when directly attached (e.g. "嗨小易")
    stripped, had_greeting = _strip_leading_greetings(normalized)
    # Then strip filler words
    stripped, had_filler = _strip_leading_fillers(stripped)
    had_leading = had_greeting or had_filler

    for alias in sorted(aliases, key=len, reverse=True):
        alias_norm = normalize_text(alias)
        alias_cf = alias_norm.casefold()
        # Special handling for "小E" / Chinese wake words: allow immediate follow-up with Chinese chars
        if alias_cf in ("小e", "小 e"):
            # Match variants:
            # 1. "小E", "小e", "小 E", "小 e" (letter e/E, optional space)
            # 2. "小易", "小艺", "小意", "小一", "小亿" (common ASR homophone errors)
            # Only matches single character after 小 - won't trigger on multi-char words like "小姨", "意思"
            homophone_class = "".join(_SMALL_E_HOMOPHONES)
            pattern = rf"^小\s*(?:e|[{homophone_class}])(?=$|[\s,，。.!！?？:：;；、\u4e00-\u9fff])"
            match = re.match(pattern, stripped, flags=re.IGNORECASE)
        elif alias_cf == "ev":
            # Original EV handling: require word boundary to avoid matching "event"
            pattern = r"^e\s*v(?=$|[\s,，。.!！?？:：;；、])"
            match = re.match(pattern, stripped, flags=re.IGNORECASE)
        else:
            # Generic wake word: require boundary
            escaped = re.escape(alias_norm)
            pattern = rf"^{escaped}(?=$|[\s,，。.!！?？:：;；、])"
            match = re.match(pattern, stripped, flags=re.IGNORECASE)
        if match:
            query = re.sub(r"^[\s,，。.!！?？:：;；、]+", "", stripped[match.end():])
            return WakeMatch(True, query, had_leading)
    return WakeMatch(False, "")


def decide_query(
    text: str,
    speaker_label: str,
    aliases: tuple[str, ...] = ("小E",),
    profile_ready: bool = True,
) -> QueryDecision:
    wake = match_wake_prefix(text, aliases)
    if not wake.detected:
        return QueryDecision(False, False, None)
    # Cold start (profile not ready): require non-trivial query after wake word
    # to reduce false triggers from random conversations mentioning "EV".
    if not profile_ready:
        query_clean = wake.query_text.strip(" ，。！？、；：,.!?;:")
        if len(query_clean) < 2:
            return QueryDecision(True, False, None, wake.had_leading_filler)
    candidate = speaker_label == "user"
    return QueryDecision(
        wake.detected,
        candidate,
        wake.query_text if candidate else None,
        wake.had_leading_filler,
    )


def decide_query_from_utterances(
    utterances: list[dict],
    aliases: tuple[str, ...] = ("小E",),
    profile_ready: bool = True,
    dominant_speaker: str = "user",
) -> QueryDecision:
    """Check utterances for wake words, gate query candidate on segment-level speaker.

    Wake-word detection is deliberately speaker-agnostic: per-utterance speaker
    labels come from noisy real-time sliding-window turns that can flip to
    "non-user" on a single bad window even when the whole segment is the user
    (segment-level fusion is far more stable). Coupling wake detection to those
    noisy turns made identical utterances randomly fail to trigger. We instead
    detect the wake word on any utterance, and gate query_candidate on the
    segment-level dominant_speaker decision.

    utterances: list of dicts with keys 'speaker' ('user'/'non-user'), 'text'
    """
    combined_query_parts = []
    wake_detected = False
    had_leading_filler = False
    found_wake_anywhere = False

    for i, utt in enumerate(utterances):
        text = utt.get("text", "")
        wake = match_wake_prefix(text, aliases)
        if wake.detected:
            wake_detected = True
            had_leading_filler = wake.had_leading_filler
            found_wake_anywhere = True
            q = wake.query_text.strip(" ，。！？、；：,.!?;:")
            if q:
                combined_query_parts.append(q)
            # Include subsequent user utterances as part of the query context
            for later_utt in utterances[i + 1:]:
                if later_utt.get("speaker") != "user":
                    break
                t = later_utt.get("text", "").strip()
                if t:
                    combined_query_parts.append(t)
            break

    if not wake_detected or not found_wake_anywhere:
        return QueryDecision(False, False, None)

    if not profile_ready:
        if not combined_query_parts or len("".join(combined_query_parts)) < 2:
            return QueryDecision(True, False, None, had_leading_filler)

    candidate = dominant_speaker == "user"
    query_text = " ".join(combined_query_parts) if combined_query_parts else ""
    return QueryDecision(
        True,
        candidate,
        query_text if query_text else None,
        had_leading_filler,
    )
