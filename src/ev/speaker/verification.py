"""声纹 profile 构建、二分法判决、多质心评分与音频归一化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeakerResult:
    score: float
    label: str  # 'user' | 'non-user'
    best_centroid_idx: int = -1  # index of best-matching centroid (0-based, -1 if no centroids)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("embedding 必须是有限的非零向量")
    return vector / norm


def normalize_loudness(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """Simple RMS loudness normalization to reduce volume/sensitivity effects on embeddings."""
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return signal
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 1e-8 or not np.isfinite(rms):
        return signal
    gain = target_rms / rms
    # Prevent clipping: cap gain at 10x
    gain = min(gain, 10.0)
    normalized = signal * gain
    peak = float(np.max(np.abs(normalized)))
    if peak > 1.0:
        normalized = normalized / peak * 0.98
    return normalized.astype(np.float32)


def build_profile(embeddings: list[np.ndarray]) -> np.ndarray:
    """Build a single centroid from embeddings (simple mean)."""
    if not embeddings:
        raise ValueError("至少需要一个 embedding")
    normalized = [normalize_embedding(item) for item in embeddings]
    dimensions = {item.shape for item in normalized}
    if len(dimensions) != 1:
        raise ValueError("embedding 维度不一致")
    return normalize_embedding(np.mean(normalized, axis=0))


def cosine_score(embedding: np.ndarray, profile: np.ndarray) -> float:
    return float(np.dot(normalize_embedding(embedding), normalize_embedding(profile)))


def classify_score(score: float, threshold: float) -> str:
    """Binary classification: score >= threshold -> user, else non-user."""
    if score >= threshold:
        return "user"
    return "non-user"


def verify_speaker(
    embedding: np.ndarray,
    centroids: list[np.ndarray] | np.ndarray,
    threshold: float,
) -> SpeakerResult:
    """Multi-centroid verification: returns best score across all centroids.

    If centroids is empty, returns (0.0, 'non-user', -1).
    """
    # Accept single centroid (ndarray) for backward compatibility
    if isinstance(centroids, np.ndarray):
        centroids_list = [centroids]
    else:
        centroids_list = list(centroids)
    if len(centroids_list) == 0:
        return SpeakerResult(0.0, "non-user", -1)
    norm_emb = normalize_embedding(embedding)
    best_score = -1.0
    best_idx = 0
    for i, centroid in enumerate(centroids_list):
        score = float(np.dot(norm_emb, normalize_embedding(centroid)))
        if score > best_score:
            best_score = score
            best_idx = i
    label = classify_score(best_score, threshold)
    return SpeakerResult(best_score, label, best_idx)


def kmeans_cluster(
    embeddings: list[np.ndarray],
    k: int,
    max_iters: int = 20,
) -> tuple[list[np.ndarray], list[int]]:
    """Simple K-means clustering for multi-centroid building.

    Returns (centroids, labels) where labels[i] is cluster index for embeddings[i].
    Uses deterministic initialization (pick evenly spaced points by norm order).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(embeddings)
    if n == 0:
        return [], []
    normed = [normalize_embedding(e) for e in embeddings]
    if n <= k:
        # Fewer points than clusters: one centroid per point
        return [build_profile([e]) for e in normed], list(range(n))

    # Deterministic init: select k evenly spaced points sorted by vector norm
    norms = np.array([float(np.linalg.norm(e)) for e in normed])
    order = np.argsort(norms)
    indices = np.linspace(0, n - 1, k, dtype=int)
    centroids = [normed[order[i]].copy() for i in indices]

    labels = [0] * n
    for _ in range(max_iters):
        # Assign
        changed = False
        for i, emb in enumerate(normed):
            scores = [float(np.dot(emb, c)) for c in centroids]
            best = int(np.argmax(scores))
            if best != labels[i]:
                labels[i] = best
                changed = True
        if not changed:
            break
        # Update centroids
        new_centroids = []
        for c in range(k):
            cluster_points = [normed[i] for i in range(n) if labels[i] == c]
            if cluster_points:
                new_centroids.append(build_profile(cluster_points))
            else:
                # Empty cluster: reinit with farthest point
                all_sims = []
                for emb in normed:
                    min_sim = min(float(np.dot(emb, nc)) for nc in centroids if nc is not None)
                    all_sims.append(min_sim)
                farthest = int(np.argmin(all_sims))
                new_centroids.append(normed[farthest].copy())
        centroids = new_centroids
    return centroids, labels


def choose_k(n_samples: int, max_k: int = 5) -> int:
    """Choose number of clusters based on sample count.
    - 1~5 samples: k=1 (single centroid)
    - 6~10 samples: k=2
    - 11~18 samples: k=3
    - 19~26 samples: k=4
    - 27+ samples: k=5 (or max_k if lower)
    """
    if n_samples < 1:
        return 0
    if n_samples <= 5:
        return 1
    if n_samples <= 10:
        return min(2, max_k)
    if n_samples <= 18:
        return min(3, max_k)
    if n_samples <= 26:
        return min(4, max_k)
    return min(5, max_k)
