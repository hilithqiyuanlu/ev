"""采集自检:抓 N 秒音频,实时打印电平,保存 wav 供回听(T2' 验收工具)。"""

from __future__ import annotations

import asyncio
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Settings
from .capture import AudioCapture
from .devices import resolve_device

_BAR_WIDTH = 40


def _bar(rms: float) -> str:
    level = min(1.0, rms * 10)
    filled = int(level * _BAR_WIDTH)
    return "#" * filled + "." * (_BAR_WIDTH - filled)


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


async def _collect(capture: AudioCapture, seconds: float, frame_ms: int = 30) -> np.ndarray:
    chunks: list[np.ndarray] = []
    bar_every = max(1, int(300 / frame_ms))  # 每 ~300ms 打一条电平
    capture.start()
    try:
        async for frame in capture.frames():
            chunks.append(frame)
            if len(chunks) % bar_every == 0:
                recent = np.concatenate(chunks[-bar_every:])
                rms = float(np.sqrt(np.mean(recent**2)))
                print(f"\r{_bar(rms)} {rms:.3f}", end="", flush=True)
            if len(chunks) * frame_ms >= seconds * 1000:
                break
    finally:
        capture.stop()
    print()
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def run_capture_test(settings: Settings, device: str | None, seconds: float) -> Path:
    device_idx = resolve_device(device)
    capture = AudioCapture(settings.audio, device=device_idx)
    audio = asyncio.run(_collect(capture, seconds))

    sr = settings.audio.sample_rate
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0

    out_dir = settings.data_dir / "audio-test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.wav"
    _save_wav(out_path, audio, sr)

    print(f"时长 {len(audio) / sr:.1f}s | RMS {rms:.4f} | 峰值 {peak:.3f}")
    if peak >= 0.99:
        print("WARNING: 出现削波(音量过大),请在系统设置中降低输入增益")
    elif rms < 0.001:
        print("WARNING: 信号极弱。检查设备选择,及 系统设置 → 隐私与安全性 → 麦克风 是否已授权终端")
    print(f"已保存: {out_path}")
    return out_path
