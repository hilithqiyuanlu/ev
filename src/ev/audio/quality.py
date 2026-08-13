"""Small, observable audio quality gate based on existing signal metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityThresholds:
    low_level_rms: float = 0.001
    low_peak_rms: float = 0.004
    low_snr_db: float = 1.0
    borderline_snr_db: float = 4.0
    min_speech_ratio: float = 0.18
    borderline_speech_ratio: float = 0.35
    min_stable_ms: int = 300


@dataclass(frozen=True)
class QualityAssessment:
    label: str
    reason: str

    @property
    def accepted(self) -> bool:
        return self.label in {"ok", "borderline"}


def assess_quality(
    *,
    avg_raw_rms: float,
    peak_raw_rms: float,
    noise_floor_rms: float,
    snr_db: float,
    speech_ratio: float,
    stream_text: str,
    stream_revision_count: int,
    stable_ms: int,
    thresholds: QualityThresholds = QualityThresholds(),
) -> QualityAssessment:
    text = stream_text.strip()
    if avg_raw_rms < thresholds.low_level_rms and peak_raw_rms < thresholds.low_peak_rms:
        return QualityAssessment("rejected_low_level", "raw level is below the usable range")
    if snr_db < thresholds.low_snr_db and speech_ratio < thresholds.borderline_speech_ratio:
        return QualityAssessment("rejected_low_snr", "signal is too close to the noise floor")
    if speech_ratio < thresholds.min_speech_ratio and not text:
        return QualityAssessment("rejected_non_voice", "insufficient speech-like audio")
    if stream_revision_count >= 3 and text and stable_ms < thresholds.min_stable_ms:
        return QualityAssessment("rejected_unstable", "streaming hypothesis never stabilized")
    if snr_db < thresholds.borderline_snr_db or speech_ratio < thresholds.borderline_speech_ratio:
        return QualityAssessment("borderline", "speech is usable but weak")
    return QualityAssessment("ok", "")
