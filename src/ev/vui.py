"""EV 文本唤醒匹配与 query 候选生成。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class WakeMatch:
    detected: bool
    query_text: str


@dataclass(frozen=True)
class QueryDecision:
    wake_detected: bool
    query_candidate: bool
    query_text: str | None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"[\s\u3000]+", " ", text)
    return text


def match_wake_prefix(text: str, aliases: tuple[str, ...] = ("EV",)) -> WakeMatch:
    normalized = normalize_text(text)
    for alias in sorted(aliases, key=len, reverse=True):
        escaped = re.escape(normalize_text(alias))
        # ASR 可能输出 "E V"，但不能把 event 等普通词当作 EV。
        if escaped.casefold() == "ev":
            pattern = r"^e\s*v(?=$|[\s,，。.!！?？:：;；、])"
        else:
            pattern = rf"^{escaped}(?=$|[\s,，。.!！?？:：;；、])"
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            query = re.sub(r"^[\s,，。.!！?？:：;；、]+", "", normalized[match.end() :])
            return WakeMatch(True, query)
    return WakeMatch(False, "")


def decide_query(
    text: str, speaker_label: str, aliases: tuple[str, ...] = ("EV",)
) -> QueryDecision:
    wake = match_wake_prefix(text, aliases)
    candidate = wake.detected and speaker_label == "user"
    return QueryDecision(wake.detected, candidate, wake.query_text if candidate else None)
