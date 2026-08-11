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
        assert vp.should_collect(
            duration_ms=2000, score=0.4, transcript="你好", is_filler_only=False
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
            asr_stream_model="stream",
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
          asr_stream_model TEXT NOT NULL, asr_final_model TEXT NOT NULL,
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