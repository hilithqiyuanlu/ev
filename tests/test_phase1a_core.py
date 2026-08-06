from datetime import datetime, timezone

import numpy as np

from ev.config import load_settings
from ev.models import verify_models
from ev.pipeline.runtime import SegmentProcessor
from ev.speaker.verification import build_profile, classify_score, verify_speaker
from ev.store.db import SegmentRecord, Store
from ev.vad.adapters import EndpointState
from ev.vui import decide_query, match_wake_prefix, normalize_text


def test_endpoint_hangover_and_flush():
    endpoint = EndpointState(hangover_frames=2)
    assert endpoint.update(False).started is False
    assert endpoint.update(True).started is True
    assert endpoint.update(False).ended is False
    assert endpoint.update(False).ended is True


def test_vui_only_sentence_prefix_and_user_gate():
    assert normalize_text(" EV， 你好 ") == "EV, 你好"
    assert match_wake_prefix("EV，打开灯").query_text == "打开灯"
    assert not match_wake_prefix("eventually open").detected
    assert decide_query("EV 打开灯", "user").query_candidate
    assert not decide_query("EV 打开灯", "non-user").query_candidate
    assert decide_query("你好 EV 打开灯", "user").wake_detected is False


def test_speaker_profile_and_three_zone():
    profile = build_profile([np.array([1.0, 0.0]), np.array([0.9, 0.1])])
    result = verify_speaker(np.array([1.0, 0.0]), profile, 0.9, 0.4)
    assert result.label == "user"
    assert classify_score(0.5, 0.9, 0.4) == "unknown"
    assert classify_score(0.2, 0.9, 0.4) == "non-user"


def test_sqlite_segment_and_profile(tmp_path):
    with Store(tmp_path / "ev.db") as store:
        now = datetime.now(timezone.utc).isoformat()
        store.insert_segment(
            SegmentRecord(
                "seg", now, now, 100, "a.wav", 16000, 1, "EV hi", "EV hi", "non-user", 0.2,
                True, False, None, "vad", "stream", "final", "speaker", now,
            )
        )
        profile = np.array([1.0, 0.0], dtype=np.float32)
        store.save_profile("user-v1", "user", "mic", "speaker", profile, 2)
        assert store.load_profile().tolist() == [1.0, 0.0]
        assert store.connection.execute("select count(*) from segments").fetchone()[0] == 1


def test_model_verify_is_offline_and_reports_missing(tmp_path):
    settings = load_settings()
    checks = verify_models(settings.models, tmp_path)
    assert len(checks) == 4
    assert all(not item.ok for item in checks)


def test_segment_processor_archives_every_segment_and_gates_query(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()

    class Stream:
        model_id = "stream"

    class Speaker:
        model_id = "speaker"

        def embed(self, audio, sample_rate):
            return np.array([1.0, 0.0], dtype=np.float32)

    class Final:
        model_id = "final"

        def transcribe(self, audio, sample_rate):
            return "EV 打开灯"

    with Store(settings.db_path) as store:
        processor = SegmentProcessor(
            settings, store, Stream(), Speaker(), settings.models.vad,
            profile=np.array([1.0, 0.0], dtype=np.float32), final_asr=Final(), output=lambda _: None,
        )
        record = processor.process(
            np.zeros(1600, dtype=np.float32),
            datetime.now(timezone.utc), datetime.now(timezone.utc), "EV 打开灯",
        )
        assert record.query_candidate is True
        assert record.audio_path.endswith(".wav")
        assert (tmp_path / "data" / "archive").exists()
