"""Automatic voice profile learning: multi-centroid + tiered sample management."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import VoiceLearningSettings
from ..store.db import Store
from .verification import (
    build_profile,
    choose_k,
    cosine_score,
    kmeans_cluster,
    normalize_embedding,
)


@dataclass
class VoiceProfileState:
    core_count: int
    cache_count: int
    sample_count: int  # total = core + cache
    centroid_count: int
    is_ready: bool  # True when core_count >= 3
    last_updated: float | None


class VoiceProfileManager:
    """Manages voice profile with multi-centroid clustering and tiered samples.

    Tiers:
      - CORE (max_core_samples): Used for building centroids. High-quality samples retained.
        Manual samples are always in CORE and never auto-evicted.
      - CACHE (max_cache_samples): Recorded but not used for centroid building.
        FIFO eviction when full.

    Multi-centroid:
      - CORE samples are K-means clustered into K centroids (K based on sample count).
      - Verification uses best score across all centroids.
    """

    def __init__(self, store: Store, settings: VoiceLearningSettings, speaker_settings=None):
        self.store = store
        self.settings = settings
        self.speaker_settings = speaker_settings
        self.auto_learn = settings.auto_learn_enabled
        self._last_collect_time: float = 0.0
        self._centroids: list[np.ndarray] = []
        self._core_sample_ids: list[str] = []
        self._core_count: int = 0
        self._cache_count: int = 0
        self._load_from_store()

    def _load_from_store(self) -> None:
        # Load core and cache separately
        core_samples = self.store.load_voice_sample_embeddings(tier="core")
        cache_count = self.store.count_voice_samples(tier="cache")
        self._core_count = len(core_samples)
        self._cache_count = cache_count
        self._core_sample_ids = [sid for sid, _, _ in core_samples]

        if core_samples:
            core_embeddings = [emb for _, emb, _ in core_samples]
            k = choose_k(len(core_embeddings), 3)
            if k == 1:
                self._centroids = [build_profile(core_embeddings)]
            else:
                self._centroids, _ = kmeans_cluster(core_embeddings, k)
            self._last_collect_time = 0.0  # allow collection after restart
        else:
            self._centroids = []

    def _is_core(self, sample_id: str) -> bool:
        row = self.store.connection.execute(
            "SELECT tier FROM speaker_samples WHERE id=?", (sample_id,)
        ).fetchone()
        return row is not None and row["tier"] == "core"

    @property
    def centroids(self) -> list[np.ndarray]:
        return self._centroids

    @property
    def centroid(self) -> np.ndarray | None:
        """Backward compatibility: return primary (first) centroid or None."""
        return self._centroids[0] if self._centroids else None

    @property
    def state(self) -> VoiceProfileState:
        last_updated = None
        row = self.store.connection.execute(
            "SELECT MAX(created_at) as ts FROM speaker_samples"
        ).fetchone()
        if row and row["ts"]:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(row["ts"])
                last_updated = dt.timestamp()
            except ValueError:
                pass
        return VoiceProfileState(
            core_count=self._core_count,
            cache_count=self._cache_count,
            sample_count=self._core_count + self._cache_count,
            centroid_count=len(self._centroids),
            is_ready=self._core_count >= 3,
            last_updated=last_updated,
        )

    def _rebuild_centroids(self) -> None:
        """Rebuild multi-centroids from current CORE samples."""
        core_data = self.store.load_voice_sample_embeddings(tier="core")
        core_embeddings = [emb for _, emb, _ in core_data]
        self._core_sample_ids = [sid for sid, _, _ in core_data]
        self._core_count = len(core_embeddings)
        if not core_embeddings:
            self._centroids = []
            return
        k = choose_k(len(core_embeddings), 3)
        if k == 1:
            self._centroids = [build_profile(core_embeddings)]
        else:
            self._centroids, _ = kmeans_cluster(core_embeddings, k)

    def should_collect(
        self,
        *,
        duration_ms: int,
        score: float,
        transcript: str,
        is_filler_only: bool,
    ) -> bool:
        if not self.auto_learn:
            return False
        # 引导门控：未完成手动引导（核心样本不足）前不自动学习
        if self._core_count < self.settings.onboarding_target:
            return False
        if duration_ms < self.settings.min_duration_ms:
            return False
        if duration_ms > self.settings.max_duration_ms:
            return False
        if not transcript or not transcript.strip():
            return False
        if is_filler_only:
            return False
        if time.monotonic() - self._last_collect_time < self.settings.min_interval_sec:
            return False
        # Collect if score >= collect_min_score (not explicit non-user)
        if score < self.settings.collect_min_score:
            return False
        return True

    def add_sample(
        self,
        *,
        embedding: np.ndarray,
        audio_path: str,
        duration_ms: int,
        score: float,
        segment_id: str | None = None,
        is_manual: bool = False,
    ) -> tuple[bool, str]:
        """Add a sample. Returns (success, tier_added_to)."""
        if not is_manual:
            if not self.should_collect(
                duration_ms=duration_ms,
                score=score,
                transcript="ok",
                is_filler_only=False,
            ):
                return False, ""

        normalized = normalize_embedding(embedding)
        tier = "cache"
        max_core = 20 if self.speaker_settings is None else self.speaker_settings.max_core_samples
        max_cache = 50 if self.speaker_settings is None else self.speaker_settings.max_cache_samples
        effective_score = 0.95 if is_manual else score

        if is_manual:
            # Manual samples always go to core
            tier = "core"
            self.store.add_voice_sample(
                segment_id=segment_id,
                audio_path=audio_path,
                duration_ms=duration_ms,
                embedding=normalized,
                score=effective_score,
                tier="core",
                is_manual=True,
            )
        else:
            # Auto: decide core vs cache
            if self._core_count < max_core:
                tier = "core"
            else:
                # Find lowest-scoring non-manual core sample
                lowest = self.store.connection.execute(
                    """
                    SELECT id, score FROM speaker_samples
                    WHERE tier='core' AND is_manual=0
                    ORDER BY score ASC LIMIT 1
                    """
                ).fetchone()
                if lowest and effective_score > lowest["score"]:
                    # Evict the lowest core to cache
                    self.store.update_voice_sample_tier(lowest["id"], "cache")
                    self._cache_count += 1
                    tier = "core"
                else:
                    tier = "cache"

            self.store.add_voice_sample(
                segment_id=segment_id,
                audio_path=audio_path,
                duration_ms=duration_ms,
                embedding=normalized,
                score=effective_score,
                tier=tier,
                is_manual=False,
            )

        # Evict oldest cache if over limit; delete evicted sample wavs from the
        # managed samples dir so disk stays consistent with the DB.
        evicted_paths = self.store.evict_oldest_cache(max_cache)
        self._unlink_evicted(evicted_paths, Path(audio_path).parent)
        self._cache_count = self.store.count_voice_samples(tier="cache")
        # Rebuild centroids from core
        self._rebuild_centroids()
        self._last_collect_time = time.monotonic()
        return True, tier

    @staticmethod
    def _unlink_evicted(evicted_paths: list[str], managed_dir) -> None:
        for raw in evicted_paths:
            path = Path(raw)
            if path.parent == managed_dir:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def promote_sample(self, sample_id: str) -> bool:
        """Promote a cache sample to core. Returns True if promoted."""
        sample = self.store.get_voice_sample(sample_id)
        if sample is None or sample["tier"] != "cache":
            return False
        max_core = 20 if self.speaker_settings is None else self.speaker_settings.max_core_samples
        max_cache = 50 if self.speaker_settings is None else self.speaker_settings.max_cache_samples

        if self._core_count >= max_core:
            # Evict lowest non-manual core
            lowest = self.store.connection.execute(
                """
                SELECT id FROM speaker_samples
                WHERE tier='core' AND is_manual=0
                ORDER BY score ASC LIMIT 1
                """
            ).fetchone()
            if lowest is None:
                return False  # All core are manual, can't evict
            self.store.update_voice_sample_tier(lowest["id"], "cache")
        self.store.update_voice_sample_tier(sample_id, "core")
        evicted_paths = self.store.evict_oldest_cache(max_cache)
        self._unlink_evicted(
            evicted_paths,
            Path(sample["audio_path"]).parent if sample.get("audio_path") else None,
        )
        self._rebuild_centroids()
        return True

    def remove_sample(self, sample_id: str) -> bool:
        sample = self.store.get_voice_sample(sample_id)
        deleted = self.store.delete_voice_sample(sample_id)
        if deleted:
            self._rebuild_centroids()
            self._cache_count = self.store.count_voice_samples(tier="cache")
        return deleted

    def reset(self) -> int:
        deleted = self.store.delete_all_voice_samples()
        self._centroids = []
        self._core_sample_ids = []
        self._core_count = 0
        self._cache_count = 0
        self._last_collect_time = 0.0
        return deleted

    def list_samples(self, tier: str | None = None, limit: int = 100) -> list[dict]:
        return self.store.list_voice_samples(tier=tier, limit=limit)

    def score(self, embedding: np.ndarray) -> float:
        """Return best score against all centroids."""
        if not self._centroids:
            return 0.0
        norm = normalize_embedding(embedding)
        return max(float(np.dot(norm, c)) for c in self._centroids)

    def set_auto_learn(self, enabled: bool) -> None:
        self.auto_learn = enabled
