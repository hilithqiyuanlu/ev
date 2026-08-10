from datetime import datetime, timezone
import asyncio

import numpy as np

from ev.config import load_settings
from ev.asr.adapters import StreamingASRAdapter
from ev.models import verify_models
from ev.pipeline import runtime as runtime_module
from ev.pipeline.runtime import SegmentJob, SegmentProcessor, SegmentWorker
from ev.speaker.profile import VoiceProfileManager
from ev.speaker.verification import build_profile, classify_score, verify_speaker
from ev.store.db import SegmentRecord, Store
from ev.vad.adapters import EndpointState, VADAdapter
from ev.vui import decide_query, match_wake_prefix, normalize_text


class MockVoiceProfile:
    def __init__(self, centroid=None, centroids=None, sample_count=5, core_count=5, cache_count=0, is_ready=True, auto_learn=False):
        self._centroid = centroid
        self._centroids = centroids if centroids is not None else ([centroid] if centroid is not None else [])
        self._sample_count = sample_count
        self._core_count = core_count
        self._cache_count = cache_count
        self._is_ready = is_ready
        self.auto_learn = auto_learn
        self.collected = 0

    @property
    def centroid(self):
        return self._centroid

    @property
    def centroids(self):
        return self._centroids

    @property
    def state(self):
        class State:
            def __init__(self, count, core, cache, ready, centroid_count):
                self.sample_count = count
                self.core_count = core
                self.cache_count = cache
                self.is_ready = ready
                self.last_updated = None
                self.centroid_count = centroid_count
        return State(
            self._sample_count,
            self._core_count,
            self._cache_count,
            self._is_ready,
            len(self._centroids),
        )

    def should_collect(self, **kwargs):
        return False

    def add_sample(self, **kwargs):
        self.collected += 1
        return False, ""

    def set_auto_learn(self, enabled):
        self.auto_learn = enabled


def test_endpoint_hangover_and_flush():
    endpoint = EndpointState(hangover_frames=2)
    assert endpoint.update(False).started is False
    assert endpoint.update(True).started is True
    assert endpoint.update(False).ended is False
    assert endpoint.update(False).ended is True


def test_fsmn_vad_stream_boundaries_and_chunking():
    class Model:
        def __init__(self):
            self.calls = []
            self.results = [
                [{"value": [[120, -1]]}],
                [{"value": [[-1, 840]]}],
                [{"value": [[40, 220]]}],
            ]

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return self.results.pop(0)

    model = Model()
    vad = VADAdapter("vad", model=model, chunk_ms=200)
    assert vad.accept(np.zeros(1600, dtype=np.float32)) == ()
    started = vad.accept(np.zeros(1600, dtype=np.float32))
    assert started[0].started is True and started[0].ended is False
    ended = vad.accept(np.zeros(3200, dtype=np.float32))
    assert ended[0].started is False and ended[0].ended is True
    complete = vad.accept(np.zeros(3200, dtype=np.float32))
    assert complete[0].started is True and complete[0].ended is True
    assert all(call["chunk_size"] == 200 for call in model.calls)
    assert all(call["disable_pbar"] is True for call in model.calls)


def test_streaming_asr_uses_600ms_chunks_and_accumulates_partial():
    class Model:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [{"text": "你好" if len(self.calls) == 1 else "世界"}]

    model = Model()
    stream = StreamingASRAdapter("stream", model=model)
    assert stream.accept(np.zeros(4800, dtype=np.float32)) == ""
    assert stream.accept(np.zeros(4800, dtype=np.float32)) == "你好"
    assert stream.accept(np.zeros(1600, dtype=np.float32), is_final=True) == "你好世界"
    assert model.calls[0]["chunk_size"] == [0, 10, 5]
    assert model.calls[0]["encoder_chunk_look_back"] == 4
    assert model.calls[0]["decoder_chunk_look_back"] == 1
    assert model.calls[-1]["is_final"] is True


def test_vui_only_sentence_prefix_and_user_gate():
    assert normalize_text(" 小E， 你好 ") == "小E, 你好"
    assert match_wake_prefix("小E，打开灯").query_text == "打开灯"
    assert not match_wake_prefix("eventually open").detected
    assert decide_query("小E 打开灯", "user").query_candidate
    assert not decide_query("小E 打开灯", "non-user").query_candidate
    assert decide_query("你好 小E 打开灯", "user").wake_detected is False


def test_speaker_binary_classification():
    profile = build_profile([np.array([1.0, 0.0]), np.array([0.9, 0.1])])
    threshold = 0.50
    # Exact match should be user
    result = verify_speaker(np.array([1.0, 0.0]), profile, threshold)
    assert result.label == "user"
    # Orthogonal vector should be non-user
    result_ortho = verify_speaker(np.array([0.0, 1.0]), profile, threshold)
    assert result_ortho.label == "non-user"
    # Boundary: score >= threshold is user
    assert classify_score(0.50, threshold) == "user"
    assert classify_score(0.49, threshold) == "non-user"
    assert classify_score(0.2, threshold) == "non-user"


def test_sqlite_segment_and_profile(tmp_path):
    with Store(tmp_path / "ev.db") as store:
        now = datetime.now(timezone.utc).isoformat()
        store.insert_segment(
            SegmentRecord(
                "seg", now, now, 100, "a.wav", 16000, 1, "EV hi", "EV hi", "non-user", 0.2,
                True, False, None, "vad", "stream", "final", "speaker", now,
            )
        )
        assert store.connection.execute("select count(*) from segments").fetchone()[0] == 1
        assert store.connection.execute("pragma user_version").fetchone()[0] == 9
        assert store.connection.execute("select count(*) from speaker_samples").fetchone()[0] == 0


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

        def transcribe(self, audio, sample_rate, **kwargs):
            return "小E 打开灯"

    with Store(settings.db_path) as store:
        voice_profile = MockVoiceProfile(
            centroid=np.array([1.0, 0.0], dtype=np.float32),
            sample_count=5,
            is_ready=True,
        )
        processor = SegmentProcessor(
            settings, store, Stream(), Speaker(), settings.models.vad,
            voice_profile=voice_profile, final_asr=Final(), output=lambda _: None,
        )
        record = processor.process(
            np.zeros(1600, dtype=np.float32),
            datetime.now(timezone.utc), datetime.now(timezone.utc), "小E 打开灯",
        )
        assert record.query_candidate is True
        assert record.audio_path.endswith(".wav")
        assert (tmp_path / "data" / "archive").exists()
        assert store.connection.execute("select source from queries").fetchone()[0] == "voice"


def test_manual_query_history_and_atomic_segment_delete(tmp_path):
    audio = tmp_path / "segment.wav"
    audio.write_bytes(b"wav")
    now = datetime.now(timezone.utc).isoformat()
    with Store(tmp_path / "ev.db") as store:
        store.insert_segment(
            SegmentRecord(
                "delete-me", now, now, 100, str(audio), 16000, 1, "hello", "hello",
                "unknown", None, False, False, None, "vad", "stream", "final", "speaker", now,
            )
        )
        manual = store.submit_manual_query("  test query  ")
        assert manual.text == "test query"
        assert store.list_segments()[0]["id"] == "delete-me"
        assert store.delete_segment("delete-me") is True
        assert not audio.exists()
        assert store.list_queries()[0]["source"] == "manual"
        assert store.delete_query(manual.id) is True
        assert store.list_queries() == []


def test_segment_worker_processes_and_emits_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()

    class Final:
        def __init__(self, path):
            self.model_id = path

        def transcribe(self, audio, sample_rate, **kwargs):
            return "测试终稿"

    class Speaker:
        model_id = "speaker"

        def embed(self, audio, sample_rate):
            return np.array([1.0, 0.0], dtype=np.float32)

    class Stream:
        model_id = "stream"

    monkeypatch.setattr(runtime_module, "FinalASRAdapter", Final)
    events = []
    now = datetime.now(timezone.utc)
    worker = SegmentWorker(
        settings,
        {"asr_final": tmp_path / "final"},
        Stream(),
        Speaker(),
        lambda _: None,
        lambda kind, payload: events.append((kind, payload)),
    )
    worker.submit(
        SegmentJob(
            np.zeros(3200, dtype=np.float32), now, now, "测试", "worker-segment"
        )
    )
    worker.close()
    assert [kind for kind, _ in events] == [
        "segment_processing",
        "speaker_result",
        "segment_committed",
    ]
    with Store(settings.db_path) as store:
        assert store.list_segments()[0]["transcript_final"] == "测试终稿"


def test_runtime_event_order_with_background_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()

    class Boundary:
        def __init__(self, started=False, ended=False):
            self.started = started
            self.ended = ended

    class VAD:
        def __init__(self, path, **kwargs):
            self.calls = 0

        def accept(self, frame, sample_rate, is_final=False):
            self.calls += 1
            if is_final:
                return ()
            if self.calls == 2:
                return (Boundary(started=True),)
            if self.calls == 20:
                return (Boundary(ended=True),)
            return ()

        def reset(self):
            pass

    class Stream:
        def __init__(self, path):
            self.model_id = path

        def accept(self, frame, sample_rate, is_final=False):
            return "测试 partial"

        def reset(self):
            pass

    class Speaker:
        def __init__(self, path):
            self.model_id = path

        def embed(self, audio, sample_rate):
            return np.array([1.0, 0.0], dtype=np.float32)

    class Final:
        def __init__(self, path):
            self.model_id = path

        def transcribe(self, audio, sample_rate, **kwargs):
            return "测试终稿"

    class Capture:
        def __init__(self, audio, device=None, **kwargs):
            self.stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        async def frames(self):
            for _ in range(25):
                yield np.zeros(480, dtype=np.float32)

    monkeypatch.setattr(runtime_module, "VADAdapter", VAD)
    monkeypatch.setattr(runtime_module, "StreamingASRAdapter", Stream)
    monkeypatch.setattr(runtime_module, "SpeakerEmbeddingAdapter", Speaker)
    monkeypatch.setattr(runtime_module, "FinalASRAdapter", Final)
    monkeypatch.setattr(runtime_module, "AudioCapture", Capture)
    monkeypatch.setattr(
        "ev.models.require_models",
        lambda settings, root=None: {
            "vad": tmp_path / "vad",
            "asr_streaming": tmp_path / "stream",
            "asr_final": tmp_path / "final",
            "speaker": tmp_path / "speaker",
        },
    )
    events = []
    asyncio.run(
        runtime_module.transcribe_forever(
            settings,
            emit=lambda kind, payload: events.append((kind, payload)),
            output=lambda _: None,
        )
    )
    kinds = [kind for kind, _ in events]
    ordered = [
        "capture_started",
        "speech_started",
        "speech_ended",
        "segment_processing",
        "speaker_result",
        "segment_committed",
    ]
    assert [kind for kind in kinds if kind in ordered] == ordered


def test_text_cleaning_collapses_cjk_spaces_preserves_english():
    from ev.asr.adapters import _clean_text
    assert _clean_text("欢 迎 大 家") == "欢迎大家"
    assert _clean_text("hello world") == "hello world"
    assert _clean_text("今 天 good") == "今天 good"
    assert _clean_text("EV 帮 我 open the door") == "EV 帮我 open the door"
    assert _clean_text("<|zh|> 测 试 <|nospeech|>") == "测试"
    assert _clean_text("") == ""


def test_filler_only_detection():
    from ev.pipeline.runtime import _is_filler_only
    assert _is_filler_only("嗯") is True
    assert _is_filler_only("啊，呃……") is True
    assert _is_filler_only("嗯啊那个") is True
    assert _is_filler_only("") is True
    assert _is_filler_only("你好") is False
    assert _is_filler_only("嗯你好") is False
    assert _is_filler_only("好的") is False


def test_segment_processor_discards_empty_final(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()

    class Final:
        def transcribe(self, audio, sample_rate, **kwargs):
            return ""

    class Speaker:
        def embed(self, audio, sample_rate):
            return np.array([1.0], dtype=np.float32)

    class Stream:
        model_id = "stream"

    with Store(settings.db_path) as store:
        voice_profile = MockVoiceProfile(centroid=None, centroids=[], sample_count=0, core_count=0, is_ready=False)
        proc = SegmentProcessor(
            settings, store, Stream(), Speaker(), "vad-test",
            voice_profile=voice_profile, final_asr=Final(), output=lambda _: None,
        )
        audio = np.zeros(16000, dtype=np.float32)
        now = datetime.now(timezone.utc)
        result = proc.process(audio, now, now, partial="", segment_id="empty-test")
        assert result is None
        assert store.list_segments() == []


def test_segment_processor_discards_filler_only(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()

    class Final:
        def transcribe(self, audio, sample_rate, **kwargs):
            return "嗯啊"

    class Speaker:
        def embed(self, audio, sample_rate):
            return np.array([1.0], dtype=np.float32)

    class Stream:
        model_id = "stream"

    with Store(settings.db_path) as store:
        voice_profile = MockVoiceProfile(centroid=None, centroids=[], sample_count=0, core_count=0, is_ready=False)
        proc = SegmentProcessor(
            settings, store, Stream(), Speaker(), "vad-test",
            voice_profile=voice_profile, final_asr=Final(), output=lambda _: None,
        )
        audio = np.zeros(16000, dtype=np.float32)
        now = datetime.now(timezone.utc)
        result = proc.process(audio, now, now, partial="", segment_id="filler-test")
        assert result is None
        assert store.list_segments() == []


def test_segment_processor_commits_valid_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()

    class Final:
        def transcribe(self, audio, sample_rate, **kwargs):
            return "帮我打开灯"

    class Speaker:
        def embed(self, audio, sample_rate):
            return np.array([1.0], dtype=np.float32)

    class Stream:
        model_id = "stream"

    with Store(settings.db_path) as store:
        voice_profile = MockVoiceProfile(centroid=None, centroids=[], sample_count=0, core_count=0, is_ready=False)
        proc = SegmentProcessor(
            settings, store, Stream(), Speaker(), "vad-test",
            voice_profile=voice_profile, final_asr=Final(), output=lambda _: None,
        )
        audio = np.zeros(16000, dtype=np.float32)
        now = datetime.now(timezone.utc)
        result = proc.process(audio, now, now, partial="帮我", segment_id="valid-test")
        assert result is not None
        assert result.transcript_final == "帮我打开灯"
        assert len(store.list_segments()) == 1


def test_runtime_discards_too_short_segment(tmp_path, monkeypatch):
    """A VAD segment shorter than min_duration_ms should be discarded (no worker submit)."""
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()

    class Boundary:
        def __init__(self, started=False, ended=False):
            self.started = started
            self.ended = ended

    class VAD:
        def __init__(self, path, **kwargs):
            self.calls = 0
        def accept(self, frame, sample_rate, is_final=False):
            self.calls += 1
            if is_final:
                return ()
            if self.calls == 2:
                return (Boundary(started=True),)
            if self.calls == 4:
                return (Boundary(ended=True),)
            return ()
        def reset(self):
            pass

    class Stream:
        def __init__(self, path):
            self.model_id = path
        def accept(self, frame, sample_rate, is_final=False):
            return "短"
        def reset(self):
            pass

    class Speaker:
        def __init__(self, path):
            self.model_id = path
        def embed(self, audio, sample_rate):
            return np.array([1.0], dtype=np.float32)

    class Final:
        def __init__(self, path):
            self.model_id = path
        def transcribe(self, audio, sample_rate, **kwargs):
            return "短"

    class Capture:
        def __init__(self, audio, device=None, **kwargs):
            self.stopped = False
        def start(self):
            pass
        def stop(self):
            self.stopped = True
        async def frames(self):
            for _ in range(8):
                yield np.zeros(480, dtype=np.float32)

    monkeypatch.setattr(runtime_module, "VADAdapter", VAD)
    monkeypatch.setattr(runtime_module, "StreamingASRAdapter", Stream)
    monkeypatch.setattr(runtime_module, "SpeakerEmbeddingAdapter", Speaker)
    monkeypatch.setattr(runtime_module, "FinalASRAdapter", Final)
    monkeypatch.setattr(runtime_module, "AudioCapture", Capture)
    monkeypatch.setattr(
        "ev.models.require_models",
        lambda settings, root=None: {
            "vad": tmp_path / "vad",
            "asr_streaming": tmp_path / "stream",
            "asr_final": tmp_path / "final",
            "speaker": tmp_path / "speaker",
        },
    )
    events = []
    asyncio.run(
        runtime_module.transcribe_forever(
            settings,
            emit=lambda kind, payload: events.append((kind, payload)),
            output=lambda _: None,
        )
    )
    kinds = [kind for kind, _ in events]
    assert "segment_discarded" in kinds
    assert "segment_committed" not in kinds
    discarded = [p for k, p in events if k == "segment_discarded"]
    assert discarded[0]["reason"] == "too_short"


def test_correction_history_crud(tmp_path):
    from ev.store.db import Store
    from datetime import datetime, timezone
    db = tmp_path / "test.db"
    now = datetime.now(timezone.utc).isoformat()
    with Store(db) as store:
        # Insert a segment first
        store.insert_segment(SegmentRecord(
            "seg1", now, now, 1500, "/tmp/a.wav", 16000, 1,
            "我想研究强化学", "我想研究强化学", "user", 0.82,
            False, True, "我想研究强化学", "vad", "stream", "final", "speaker", now,
        ))
        # Record a correction
        corr = store.record_correction(
            segment_id="seg1",
            asr_text="我想研究强化学",
            corrected_text="我想研究强化学习",
            source="manual_edit",
            speaker_label="user",
            speaker_score=0.82,
            audio_path="/tmp/a.wav",
        )
        assert corr["source"] == "manual_edit"
        assert corr["corrected_text"] == "我想研究强化学习"
        assert store.count_corrections() == 1

        # List corrections
        corrs = store.list_corrections()
        assert len(corrs) == 1
        assert corrs[0]["asr_text"] == "我想研究强化学"

        # List with source filter
        manual = store.list_corrections(source="manual_edit")
        assert len(manual) == 1
        auto = store.list_corrections(source="implicit_repeat")
        assert len(auto) == 0

        # Update segment transcript
        updated = store.update_segment_transcript("seg1", "我想研究强化学习")
        assert updated is not None
        assert updated["transcript_final"] == "我想研究强化学习"

        # was_corrected annotation in list_segments
        segments = store.list_segments()
        assert len(segments) == 1
        assert segments[0]["was_corrected"] is True

        # Invalid source raises
        import pytest
        with pytest.raises(ValueError, match="invalid correction source"):
            store.record_correction("a", "b", "bogus")

        # Non-existent segment returns None for update
        assert store.update_segment_transcript("nonexistent", "x") is None


def test_correction_manual_add_word_signal(tmp_path):
    """Manually adding a lexicon word should be recordable as manual_add_word correction."""
    from ev.store.db import Store
    db = tmp_path / "test.db"
    with Store(db) as store:
        store.add_lexicon_word("强化学习", 3.0, source="manual")
        corr = store.record_correction(
            asr_text="",
            corrected_text="强化学习",
            source="manual_add_word",
        )
        assert corr["source"] == "manual_add_word"
        assert store.count_corrections() == 1
