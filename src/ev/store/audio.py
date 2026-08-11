"""语音段 WAV 归档与声纹样本音频管理。"""

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
    write_wav(path, audio, sample_rate)
    return path


def save_voice_sample(
    samples_root: Path,
    name: str,
    audio: np.ndarray,
    sample_rate: int,
) -> Path:
    """Write a voice-sample wav under a managed dir (decoupled from segment history)."""
    samples_root.mkdir(parents=True, exist_ok=True)
    filename = name if name.endswith(".wav") else f"{name}.wav"
    path = samples_root / filename
    write_wav(path, audio, sample_rate)
    return path


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit wav, returning (float32 samples in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=-1)
    return pcm, sample_rate
