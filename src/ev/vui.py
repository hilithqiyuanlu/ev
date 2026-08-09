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
