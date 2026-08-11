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
        self._last_promote_time: float = 0.0
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
            k = choose_k(len(core_embeddings), self._max_centroids())
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

    def _max_centroids(self) -> int:
        return self.speaker_settings.max_centroids if self.speaker_settings is not None else 3

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
        """三档分级入库 + 簇内竞争淘汰 + 自动补位. 返回 (success, tier_added_to).

        手动样本: 永远进 core, 不受分数限制, 永不参与淘汰.
        自动样本按分数分档:
          - score >= core_score_min                          → core
          - collect_min_score <= score < core_score_min      → cache + is_diversity=1
          - score < collect_min_score                        → 不收
        核心超限: 簇内竞争, 降级最低分的非手动 core 到 cache (质量优先).
        缓存超限: FIFO 淘汰, diversity 样本优先保留 (Store 已支持).
        自动补位: add 后若某簇成员 < promote_min_members, 从缓存晋升该簇最高分样本.
        """
        if not is_manual:
            if not self.should_collect(
                duration_ms=duration_ms,
                score=score,
                transcript="ok",
                is_filler_only=False,
            ):
                return False, ""

        normalized = normalize_embedding(embedding)
        max_core = self._max_core()
        max_cache = self._max_cache()
        effective_score = 0.95 if is_manual else score

        if is_manual:
            tier = "core"
            self.store.add_voice_sample(
                segment_id=segment_id,
                audio_path=audio_path,
                duration_ms=duration_ms,
                embedding=normalized,
                score=effective_score,
                tier="core",
                is_manual=True,
                is_diversity=False,
            )
        else:
            # 三档分级: 高分进核心, 中低分进缓存并标记 diversity, 过低不收.
            # "核心未满也只收 >= core_score_min" —— 低分样本不因核心有空位就进核心.
            if score >= self.settings.core_score_min:
                tier = "core"
                is_diversity = False
            elif score >= self.settings.collect_min_score:
                tier = "cache"
                is_diversity = True
            else:
                return False, ""
            self.store.add_voice_sample(
                segment_id=segment_id,
                audio_path=audio_path,
                duration_ms=duration_ms,
                embedding=normalized,
                score=effective_score,
                tier=tier,
                is_manual=False,
                is_diversity=is_diversity,
            )

        # 核心超限: 簇内竞争, 降级最低分非手动 core (质量优先)
        self._trim_core_overflow(max_core)

        # 缓存超限: FIFO + diversity 优先保留 (Store.evict_oldest_cache 已支持)
        evicted_paths = self.store.evict_oldest_cache(max_cache)
        self._unlink_evicted(evicted_paths, Path(audio_path).parent)

        # 重建质心 (用 max_centroids 配置)
        self._rebuild_centroids()

        # 自动补位: 簇成员不足时从缓存晋升 (加冷却)
        self._auto_promote()

        self._cache_count = self.store.count_voice_samples(tier="cache")
        self._last_collect_time = time.monotonic()
        return True, tier

    def _max_core(self) -> int:
        return self.speaker_settings.max_core_samples if self.speaker_settings is not None else 20

    def _max_cache(self) -> int:
        return self.speaker_settings.max_cache_samples if self.speaker_settings is not None else 50

    def _trim_core_overflow(self, max_core: int) -> None:
        """核心超限时降级最低分的非手动 core 到 cache (质量优先的簇内竞争近似).

        每簇只需保留高分代表; 全局降级最低分, 等价于把各簇内最低分样本淘汰.
        手动样本永不参与淘汰.
        """
        core_count = self.store.count_voice_samples(tier="core")
        need = core_count - max_core
        if need <= 0:
            return
        rows = self.store.connection.execute(
            """
            SELECT id FROM speaker_samples
            WHERE tier='core' AND is_manual=0
            ORDER BY score ASC LIMIT ?
            """,
            (need,),
        ).fetchall()
        for row in rows:
            self.store.update_voice_sample_tier(row["id"], "cache")

    def _auto_promote(self) -> None:
        """自动补位: 某簇核心成员 < promote_min_members 时, 从缓存晋升该簇最高分样本.

        受 promote_cooldown_sec 冷却限制, 避免频繁抖动.
        """
        if not self._centroids:
            return
        now = time.monotonic()
        if now - self._last_promote_time < self.settings.promote_cooldown_sec:
            return
        core_data = self.store.load_voice_sample_embeddings(tier="core")
        if not core_data:
            return
        core_emb = [np.asarray(emb, dtype=np.float32) for _, emb, _ in core_data]
        cache_samples = self.store.load_voice_sample_detailed(tier="cache")

        for centroid in self._centroids:
            c = normalize_embedding(np.asarray(centroid, dtype=np.float32))
            # 该簇核心成员数 (与质心相似度 >= collect_min_score 视为同簇)
            members = sum(
                1 for e in core_emb
                if float(np.dot(normalize_embedding(e), c)) >= self.settings.collect_min_score
            )
            if members >= self.settings.promote_min_members:
                continue
            if not cache_samples:
                break
            # 缓存中与该质心最相似的样本
            best_id: str | None = None
            best_sim = -1.0
            for s in cache_samples:
                sim = float(
                    np.dot(
                        normalize_embedding(np.asarray(s["embedding"], dtype=np.float32)),
                        c,
                    )
                )
                if sim > best_sim:
                    best_sim = sim
                    best_id = s["id"]
            if best_id is not None and best_sim >= self.settings.collect_min_score:
                self.store.update_voice_sample_tier(best_id, "core")
                self._last_promote_time = now

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
        max_core = self._max_core()
        max_cache = self._max_cache()

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

    def pending_samples(self, threshold: float, limit: int = 50) -> list[dict]:
        """待人工确认的候选样本: 缓存中与所有核心质心最大相似度 < threshold 的样本.

        这类样本与用户声纹距离过大, 可能是误收录的他人/噪声, 也可能是新的
        声学变体; 由 UI 展示并让用户确认 (晋升核心) 或删除.
        """
        if not self._centroids:
            return []
        samples = self.store.load_voice_sample_detailed(tier="cache")
        if not samples:
            return []
        centroids = [
            normalize_embedding(np.asarray(c, dtype=np.float32)) for c in self._centroids
        ]
        out = []
        for s in samples:
            emb = normalize_embedding(np.asarray(s["embedding"], dtype=np.float32))
            best = max(float(np.dot(emb, c)) for c in centroids)
            if best < threshold:
                out.append({
                    "id": s["id"],
                    "audio_path": s["audio_path"],
                    "duration_ms": s["duration_ms"],
                    "score": round(float(s["score"]), 3),
                    "max_similarity": round(best, 3),
                    "tier": s["tier"],
                    "is_manual": s["is_manual"],
                    "is_diversity": s.get("is_diversity", 0),
                    "created_at": s["created_at"],
                })
        out.sort(key=lambda x: x["max_similarity"])
        return out[:limit]

    def score(self, embedding: np.ndarray) -> float:
        """Return best score against all centroids."""
        if not self._centroids:
            return 0.0
        norm = normalize_embedding(embedding)
        return max(float(np.dot(norm, c)) for c in self._centroids)

    def set_auto_learn(self, enabled: bool) -> None:
        self.auto_learn = enabled
