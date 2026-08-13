from ev.audio.quality import assess_quality


def assess(**overrides):
    values = {
        "avg_raw_rms": 0.02,
        "peak_raw_rms": 0.1,
        "noise_floor_rms": 0.002,
        "snr_db": 12.0,
        "speech_ratio": 0.8,
        "stream_text": "正常语音",
        "stream_revision_count": 2,
        "stable_ms": 500,
    }
    values.update(overrides)
    return assess_quality(**values)


def test_quality_labels_signal_failures_and_borderline():
    assert assess(avg_raw_rms=0.0002, peak_raw_rms=0.001).label == "rejected_low_level"
    assert assess(snr_db=-2, speech_ratio=0.2).label == "rejected_low_snr"
    assert assess(speech_ratio=0.05, stream_text="").label == "rejected_non_voice"
    assert assess(stream_revision_count=5, stable_ms=50).label == "rejected_unstable"
    assert assess(snr_db=3.0).label == "borderline"
    assert assess().label == "ok"
