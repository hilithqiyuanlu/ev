"""声纹 profile 构建、相似度计算与三区标签。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeakerResult:
    score: float
    label: str


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("embedding 必须是有限的非零向量")
    return vector / norm


def build_profile(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("至少需要一个 embedding")
    normalized = [normalize_embedding(item) for item in embeddings]
    dimensions = {item.shape for item in normalized}
    if len(dimensions) != 1:
        raise ValueError("embedding 维度不一致")
    return normalize_embedding(np.mean(normalized, axis=0))


def cosine_score(embedding: np.ndarray, profile: np.ndarray) -> float:
    return float(np.dot(normalize_embedding(embedding), normalize_embedding(profile)))


def classify_score(score: float, user_threshold: float, non_user_threshold: float) -> str:
    if non_user_threshold >= user_threshold:
        raise ValueError("non_user_threshold 必须小于 user_threshold")
    if score >= user_threshold:
        return "user"
    if score <= non_user_threshold:
        return "non-user"
    return "unknown"


def verify_speaker(
    embedding: np.ndarray,
    profile: np.ndarray,
    user_threshold: float,
    non_user_threshold: float,
) -> SpeakerResult:
    score = cosine_score(embedding, profile)
    return SpeakerResult(score, classify_score(score, user_threshold, non_user_threshold))
