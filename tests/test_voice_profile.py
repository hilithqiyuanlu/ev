"""声纹引导门控、样本音频独立管理与完整性标记的回归测试。"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from ev.config import load_settings
from ev.speaker.profile import VoiceProfileManager
from ev.store.audio import save_voice_sample
from ev.store.db import Store


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