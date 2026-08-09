import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.models


def _model_root() -> Path:
    value = os.environ.get("EV_MODEL_ROOT")
    if not value or os.environ.get("EV_RUN_MODEL_TESTS") != "1":
        pytest.skip("set EV_RUN_MODEL_TESTS=1 and EV_MODEL_ROOT to run local model tests")
    root = Path(value)
    if not root.is_dir():
        pytest.skip(f"model root not found: {root}")
    return root


def _audio(path: Path):
    soundfile = pytest.importorskip("soundfile")
    audio, sample_rate = soundfile.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sample_rate


def test_release_models_and_file_pipeline_are_usable(tmp_path, monkeypatch, capsys):
    from ev.asr.adapters import (
        FinalASRAdapter,
        SpeakerEmbeddingAdapter,
        StreamingASRAdapter,
    )
    from ev.config import load_settings
    from ev.pipeline.runtime import SegmentProcessor
    from ev.store.db import Store
    from ev.vad.adapters import VADAdapter

    root = _model_root()
    vad_audio, sample_rate = _audio(
        root / "ev-fsmn-vad-zh-16k" / "example" / "vad_example.wav"
    )
    vad = VADAdapter(str(root / "ev-fsmn-vad-zh-16k"))
    boundaries = []
    for offset in range(0, len(vad_audio), 480):
        boundaries.extend(vad.accept(vad_audio[offset : offset + 480], sample_rate))
    boundaries.extend(vad.accept(np.empty(0, dtype=np.float32), sample_rate, True))
    assert any(item.started for item in boundaries)
    assert any(item.ended for item in boundaries)

    asr_audio, sample_rate = _audio(
        root / "ev-paraformer-zh-streaming-16k" / "example" / "asr_example.wav"
    )
    stream = StreamingASRAdapter(str(root / "ev-paraformer-zh-streaming-16k"))
    text = ""
    for offset in range(0, len(asr_audio), 480):
        text = stream.accept(asr_audio[offset : offset + 480], sample_rate)
    text = stream.accept(np.empty(0, dtype=np.float32), sample_rate, True)
    assert text.strip()

    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()
    final = FinalASRAdapter(str(root / "ev-sensevoice-small"))
    speaker = SpeakerEmbeddingAdapter(str(root / "ev-eres2netv2-zh-16k"))
    now = datetime.now(timezone.utc)
    with Store(settings.db_path) as store:
        processor = SegmentProcessor(
            settings,
            store,
            stream,
            speaker,
            settings.models.vad,
            profile=None,
            final_asr=final,
            output=lambda _: None,
        )
        record = processor.process(asr_audio, now, now, text, "integration-segment")
        assert record.transcript_final.strip()
        assert record.speaker_label == "unknown"
        assert Path(record.audio_path).is_file()
        assert store.list_segments()[0]["id"] == "integration-segment"

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "rtf_avg" not in combined
    assert "\x1b[" not in combined
