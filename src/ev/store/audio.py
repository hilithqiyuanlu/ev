"""语音段 WAV 归档。"""

from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path

import numpy as np


def archive_wav(
    archive_root: Path,
    segment_id: str,
    audio: np.ndarray,
    sample_rate: int,
    started_at: datetime,
    suffix: str = "",
) -> Path:
    day_dir = archive_root / started_at.astimezone().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{segment_id}{suffix}.wav"
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path
