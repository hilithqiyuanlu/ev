"""声纹引导门控、样本音频独立管理与完整性标记的回归测试。"""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ev.config import load_settings
from ev.speaker.profile import VoiceProfileManager
from ev.store.audio import save_voice_sample
from ev.store.db import SegmentRecord, Store


def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EV_MODEL_ROOT", str(tmp_path / "models"))
    return load_settings()


def test_auto_learn_gated_until_onboarding_complete(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        vp = VoiceProfileManager(store, settings.voice_learning, settings.speaker)
        assert settings.voice_learning.onboarding_target >= 2
        # No samples at all → no auto learning
        assert vp.should_collect(
            duration_ms=2000, score=0.95, transcript="你好", is_filler_only=False
        ) is False
        # Below onboarding target → still gated even at very high confidence
        for i in range(settings.voice_learning.onboarding_target - 1):
            vp.add_sample(
                embedding=np.array([1.0, 0.01 * i], dtype=np.float32),
                audio_path="/managed/sample.wav",
                duration_ms=2000,
                score=0.95,
                segment_id=None,
                is_manual=True,
            )
        assert vp.state.core_count == settings.voice_learning.onboarding_target - 1
        assert vp.should_collect(
            duration_ms=2000, score=0.95, transcript="你好", is_filler_only=False
        ) is False
        # Complete onboarding → gated off for high-confidence samples
        vp.add_sample(
            embedding=np.array([1.0, 0.3], dtype=np.float32),
            audio_path="/managed/sample-full.wav",
            duration_ms=2000,
            score=0.95,
            segment_id=None,
            is_manual=True,
        )
        assert vp.state.core_count >= settings.voice_learning.onboarding_target
        vp._last_collect_time = 0.0
        assert vp.should_collect(
            duration_ms=2000, score=0.95, transcript="你好", is_filler_only=False
        ) is True
        # Low-confidence still rejected even after onboarding
        # (collect_min_score=0.40; 0.4 is the boundary, 0.39 is below it)
        assert vp.should_collect(
            duration_ms=2000, score=0.39, transcript="你好", is_filler_only=False
        ) is False
        # Too short still rejected
        assert vp.should_collect(
            duration_ms=200, score=0.95, transcript="你好", is_filler_only=False
        ) is False


def test_evict_oldest_cache_returns_paths_without_deleting_files(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        paths = []
        for i in range(3):
            p = save_voice_sample(
                settings.voice_samples_dir, f"cache-{i}", np.zeros(200, dtype=np.float32), 16000
            )
            paths.append(p)
            store.add_voice_sample(
                segment_id=None,
                audio_path=str(p),
                duration_ms=1000,
                embedding=np.array([1.0, i], dtype=np.float32),
                score=0.5,
                tier="cache",
            )
        evicted = store.evict_oldest_cache(max_cache=1)
        assert len(evicted) == 2
        assert {Path(p) for p in evicted} == {paths[0], paths[1]}
        assert all(Path(p).exists() for p in evicted)
        assert store.count_voice_samples(tier="cache") == 1


def test_list_voice_samples_reports_audio_available(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    real = save_voice_sample(
        settings.voice_samples_dir, "exists", np.zeros(320, dtype=np.float32), 16000
    )
    with Store(settings.db_path) as store:
        store.add_voice_sample(
            segment_id=None,
            audio_path=str(real),
            duration_ms=1000,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            score=0.9,
            tier="core",
            is_manual=True,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path=str(settings.voice_samples_dir / "ghost.wav"),
            duration_ms=1000,
            embedding=np.array([1.0, 0.1], dtype=np.float32),
            score=0.9,
            tier="core",
            is_manual=True,
        )
        rows = store.list_voice_samples()
        assert len(rows) == 2
        by_path = {Path(r["audio_path"]).name: r["audio_available"] for r in rows}
        assert by_path["exists.wav"] is True
        assert by_path["ghost.wav"] is False


def test_managed_dir_property(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    assert settings.voice_samples_dir == tmp_path / "data" / "voice-samples"
    settings.ensure_dirs()
    assert settings.voice_samples_dir.exists()


def _insert_segment(store: Store, segment_id: str, wav: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    store.insert_segment(
        SegmentRecord(
            id=segment_id,
            started_at=now,
            ended_at=now,
            duration_ms=2000,
            audio_path=str(wav),
            sample_rate=16000,
            channels=1,
            transcript_raw="你好",
            transcript_final="你好",
            speaker_label="user",
            wake_detected=False,
            query_candidate=False,
            vad_model="vad",
            asr_final_model="final",
            speaker_model="speaker",
            created_at=now,
        )
    )


def test_delete_segment_keeps_voice_sample(tmp_path, monkeypatch):
    """历史删除不应级联删除已学习的声纹样本；segment_id 置空，样本音频保留。"""
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        segment_wav = save_voice_sample(
            settings.voice_samples_dir, "seg-src", np.zeros(320, dtype=np.float32), 16000
        )
        _insert_segment(store, "seg-1", segment_wav)
        sample_wav = save_voice_sample(
            settings.voice_samples_dir, "seg-1", np.zeros(320, dtype=np.float32), 16000
        )
        sample_id = store.add_voice_sample(
            segment_id="seg-1",
            audio_path=str(sample_wav),
            duration_ms=2000,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            score=0.9,
            tier="core",
        )
        assert store.count_voice_samples() == 1

        assert store.delete_segment("seg-1") is True

        sample = store.get_voice_sample(sample_id)
        assert sample is not None, "历史删除不应删除声纹样本"
        assert sample["segment_id"] is None, "关联字段应置空而不是指向不存在的记录"
        assert sample_wav.exists(), "样本音频应保留在 voice-samples 目录"
        assert store.count_voice_samples() == 1


def test_delete_segment_keeps_sample_after_v14_migration(tmp_path, monkeypatch):
    """v13 老库（CASCADE 外键）升级后同样解耦。"""
    settings = _settings(tmp_path, monkeypatch)
    db = settings.db_path
    db.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE segments (
          id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
          duration_ms INTEGER NOT NULL, audio_path TEXT NOT NULL,
          sample_rate INTEGER NOT NULL, channels INTEGER NOT NULL,
          transcript_raw TEXT NOT NULL, transcript_final TEXT NOT NULL,
          speaker_label TEXT NOT NULL, wake_detected INTEGER NOT NULL,
          query_candidate INTEGER NOT NULL, vad_model TEXT NOT NULL,
          asr_stream_model TEXT NOT NULL DEFAULT '', asr_final_model TEXT NOT NULL,
          speaker_model TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE speaker_samples (
          id TEXT PRIMARY KEY,
          segment_id TEXT REFERENCES segments(id) ON DELETE CASCADE,
          audio_path TEXT NOT NULL, duration_ms INTEGER NOT NULL,
          embedding_blob BLOB NOT NULL, embedding_dim INTEGER NOT NULL,
          score REAL NOT NULL, tier TEXT NOT NULL DEFAULT 'core',
          is_manual INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
        );
        CREATE TABLE queries (
          id TEXT PRIMARY KEY, source TEXT NOT NULL, segment_id TEXT,
          text TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
          created_at TEXT NOT NULL
        );
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO segments VALUES ('seg-old', ?, ?, 2000, '/tmp/x.wav', 16000, 1, 't', 't', 'user', 0, 0, 'v', 's', 'f', 'sp', ?)",
        (now, now, now),
    )
    conn.execute(
        "INSERT INTO speaker_samples VALUES ('smp-old', 'seg-old', '/tmp/s.wav', 2000, ?, 2, 0.9, 'core', 0, ?)",
        (np.array([1.0, 0.0], dtype=np.float32).tobytes(), now),
    )
    conn.commit()
    conn.close()

    # Opening the store runs the v14 migration.
    with Store(db) as store:
        assert store.delete_segment("seg-old") is True
        sample = store.get_voice_sample("smp-old")
        assert sample is not None, "迁移后样本不应被级联删除"
        assert sample["segment_id"] is None


def test_add_sample_three_tier_grading(tmp_path, monkeypatch):
    """三档分级: >=core_score_min→core; collect_min~core→cache+diversity; <collect_min→不收."""
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        vp = VoiceProfileManager(store, settings.voice_learning, settings.speaker)
        # 手动样本凑够 onboarding (引导门控需要 core >= onboarding_target)
        for i in range(settings.voice_learning.onboarding_target):
            vp.add_sample(
                embedding=np.array([1.0, 0.01 * i], dtype=np.float32),
                audio_path=f"/onboard{i}.wav", duration_ms=2000,
                score=0.95, is_manual=True,
            )
        assert vp.state.core_count >= settings.voice_learning.onboarding_target
        vp._last_collect_time = 0.0

        # score >= core_score_min (0.70) → core
        ok, tier = vp.add_sample(
            embedding=np.array([1.0, 0.05], dtype=np.float32),
            audio_path="/high.wav", duration_ms=2000, score=0.80,
        )
        assert ok and tier == "core"

        # collect_min(0.40) <= score < core_score_min → cache + diversity
        vp._last_collect_time = 0.0  # 重置采集冷却
        ok, tier = vp.add_sample(
            embedding=np.array([1.0, 0.02], dtype=np.float32),
            audio_path="/mid.wav", duration_ms=2000, score=0.50,
        )
        assert ok and tier == "cache"
        mid = [s for s in store.list_voice_samples(tier="cache") if s["audio_path"] == "/mid.wav"]
        assert mid and mid[0]["is_diversity"] == 1

        # score < collect_min → 不收
        vp._last_collect_time = 0.0
        ok, tier = vp.add_sample(
            embedding=np.array([1.0, 0.01], dtype=np.float32),
            audio_path="/low.wav", duration_ms=2000, score=0.30,
        )
        assert ok is False and tier == ""


def test_core_overflow_demotes_lowest_non_manual(tmp_path, monkeypatch):
    """核心超限时簇内竞争: 降级最低分非手动 core, 手动样本永不淘汰."""
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, speaker=replace(settings.speaker, max_core_samples=3))
    with Store(settings.db_path) as store:
        store.add_voice_sample(
            segment_id=None,
            audio_path="/a.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.1], dtype=np.float32),
            score=0.90, tier="core", is_manual=False,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path="/b.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.2], dtype=np.float32),
            score=0.80, tier="core", is_manual=False,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path="/c.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.3], dtype=np.float32),
            score=0.75, tier="core", is_manual=False,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path="/m.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.4], dtype=np.float32),
            score=0.95, tier="core", is_manual=True,
        )
        vp = VoiceProfileManager(store, settings.voice_learning, settings.speaker)
        vp._trim_core_overflow(3)
        core = store.list_voice_samples(tier="core")
        assert len(core) <= 3
        assert any(s["is_manual"] == 1 for s in core), "手动样本不应被淘汰"
        cache = store.list_voice_samples(tier="cache")
        assert any(s["score"] == 0.75 for s in cache), "最低分非手动 core 应降级到 cache"


def test_auto_promote_fills_undersized_cluster(tmp_path, monkeypatch):
    """自动补位: 簇成员 < promote_min_members 时从缓存晋升该簇最高分样本."""
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        store.add_voice_sample(
            segment_id=None,
            audio_path="/core.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            score=0.95, tier="core", is_manual=True,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path="/cache.wav", duration_ms=2000,
            embedding=np.array([0.98, 0.02], dtype=np.float32),
            score=0.90, tier="cache", is_manual=False,
        )
        vp = VoiceProfileManager(store, settings.voice_learning, settings.speaker)
        vp._last_promote_time = -100.0  # 绕过冷却
        vp._auto_promote()
        core_paths = [s["audio_path"] for s in store.list_voice_samples(tier="core")]
        assert "/cache.wav" in core_paths, "缓存中与质心相似的样本应被自动晋升"


def test_pending_samples_confirm(tmp_path, monkeypatch):
    """待确认: 缓存中与质心距离过大的样本进入 pending, 确认后晋升核心."""
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        store.add_voice_sample(
            segment_id=None,
            audio_path="/core.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            score=0.95, tier="core", is_manual=True,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path="/far.wav", duration_ms=2000,
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            score=0.50, tier="cache", is_manual=False,
        )
        vp = VoiceProfileManager(store, settings.voice_learning, settings.speaker)
        pending = vp.pending_samples(0.30)
        assert len(pending) == 1
        assert pending[0]["audio_path"] == "/far.wav"
        assert vp.promote_sample(pending[0]["id"]) is True
        core_paths = [s["audio_path"] for s in store.list_voice_samples(tier="core")]
        assert "/far.wav" in core_paths


def test_pending_reject_removes_sample(tmp_path, monkeypatch):
    """待确认删除: remove_sample 从库中移除样本."""
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        store.add_voice_sample(
            segment_id=None,
            audio_path="/core.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            score=0.95, tier="core", is_manual=True,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path="/far.wav", duration_ms=2000,
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            score=0.50, tier="cache", is_manual=False,
        )
        vp = VoiceProfileManager(store, settings.voice_learning, settings.speaker)
        pending = vp.pending_samples(0.30)
        assert len(pending) == 1
        sid = pending[0]["id"]
        assert vp.remove_sample(sid) is True
        assert store.get_voice_sample(sid) is None


def test_cache_eviction_preserves_diversity(tmp_path, monkeypatch):
    """缓存 FIFO: 普通样本先淘汰, diversity 样本优先保留."""
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        store.add_voice_sample(
            segment_id=None,
            audio_path="/d1.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            score=0.50, tier="cache", is_manual=False, is_diversity=True,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path="/p1.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.1], dtype=np.float32),
            score=0.60, tier="cache", is_manual=False, is_diversity=False,
        )
        store.add_voice_sample(
            segment_id=None,
            audio_path="/p2.wav", duration_ms=2000,
            embedding=np.array([1.0, 0.2], dtype=np.float32),
            score=0.60, tier="cache", is_manual=False, is_diversity=False,
        )
        evicted = [Path(p).name for p in store.evict_oldest_cache(2)]
        assert "d1.wav" not in evicted, "diversity 样本不应先被淘汰"
        assert len(evicted) == 1


def test_choose_k_tiers():
    """choose_k 档位: ≤5→1, 6-10→2, 11-18→3, 19-26→4, ≥27→5."""
    from ev.speaker.verification import choose_k
    assert choose_k(5, 5) == 1
    assert choose_k(6, 5) == 2
    assert choose_k(10, 5) == 2
    assert choose_k(11, 5) == 3
    assert choose_k(18, 5) == 3
    assert choose_k(19, 5) == 4
    assert choose_k(26, 5) == 4
    assert choose_k(27, 5) == 5
    assert choose_k(3, 3) == 1