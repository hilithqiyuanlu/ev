"""Evidence-gated lexicon candidate selection for final ASR."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class LexiconEntry:
    id: str
    word: str
    weight: float
    source: str
    status: str


@dataclass(frozen=True)
class HotwordCandidate:
    entry_id: str
    word: str
    score: float
    match_type: str
    matched_text: str
    matched_chars: int


def normalize_for_matching(text: str) -> str:
    """Normalize visible word characters while dropping punctuation and spacing."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(
        ch for ch in normalized
        if not ch.isspace() and not unicodedata.category(ch).startswith(("P", "S"))
    )


def _lcs_length(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    for left_ch in left:
        current = [0]
        for index, right_ch in enumerate(right, start=1):
            if left_ch == right_ch:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _ordered_match(candidate: str, evidence: str) -> tuple[int, str]:
    window_size = min(len(evidence), len(candidate) + 2)
    if window_size <= 0:
        return 0, ""
    best_count = 0
    best_window = ""
    for start in range(max(1, len(evidence) - window_size + 1)):
        window = evidence[start:start + window_size]
        count = _lcs_length(candidate, window)
        if count > best_count:
            best_count = count
            best_window = window
    return best_count, best_window


def select_hotword_candidates(
    evidence_text: str,
    entries: list[LexiconEntry],
    *,
    limit: int = 8,
) -> list[HotwordCandidate]:
    """Select a small, deterministic set of terms supported by stream-ASR text."""
    evidence = normalize_for_matching(evidence_text)
    if len(evidence) < 2 or limit <= 0:
        return []

    selected: dict[str, tuple[HotwordCandidate, float]] = {}
    for entry in entries:
        if entry.status != "active" or entry.source == "system":
            continue
        candidate = normalize_for_matching(entry.word)
        if len(candidate) < 2:
            continue

        if candidate in evidence:
            match = HotwordCandidate(
                entry_id=entry.id,
                word=entry.word,
                score=1.0,
                match_type="exact",
                matched_text=candidate,
                matched_chars=len(candidate),
            )
        elif len(candidate) == 2:
            continue
        else:
            matched_chars, matched_text = _ordered_match(candidate, evidence)
            minimum_coverage = 0.75 if len(candidate) <= 4 else 0.60
            minimum_chars = math.ceil(len(candidate) * minimum_coverage)
            if len(candidate) >= 5:
                minimum_chars = max(3, minimum_chars)
            if matched_chars < minimum_chars:
                continue
            match = HotwordCandidate(
                entry_id=entry.id,
                word=entry.word,
                score=round(min(0.99, matched_chars / len(candidate)), 4),
                match_type="ordered",
                matched_text=matched_text,
                matched_chars=matched_chars,
            )

        existing = selected.get(candidate)
        ranked = (match, entry.weight)
        incoming_rank = (match.score, match.matched_chars, entry.weight)
        existing_rank = (
            existing[0].score, existing[0].matched_chars, existing[1]
        ) if existing else None
        if (
            existing is None
            or incoming_rank > existing_rank
            or (incoming_rank == existing_rank and match.entry_id < existing[0].entry_id)
        ):
            selected[candidate] = ranked

    ranked_candidates = sorted(
        selected.values(),
        key=lambda item: (
            -item[0].score,
            -item[0].matched_chars,
            -item[1],
            item[0].entry_id,
        ),
    )
    return [candidate for candidate, _ in ranked_candidates[:limit]]
