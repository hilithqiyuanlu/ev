"""音频采集:16kHz mono 定长帧,异步产出。

对齐全天候/双工:帧流以异步迭代器暴露,后续 VAD/ASR 直接消费;
设备抽象 —— 内置麦与 DJI 接收器只差一个 selector 字符串。
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

from ..config import AudioSettings


class AudioCapture:
    def __init__(
        self, audio: AudioSettings, device: int | None = None, frame_ms: int = 30
    ):
        self._audio = audio
        self._device = device
        self._frame_samples = audio.sample_rate * frame_ms // 1000
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=128)
        self._stream: sd.InputStream | None = None

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # status 溢出时丢帧优先于阻塞回调(回调线程绝不能卡)
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # 丢弃最旧的块，不能阻塞 PortAudio 回调线程。
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Empty:
                pass

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._audio.sample_rate,
            channels=self._audio.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """产出定长帧(float32, frame_ms 毫秒)。各设备 blocksize 不一,在此归一。"""
        buf = np.empty(0, dtype=np.float32)
        while True:
            block = await asyncio.to_thread(self._queue.get)
            if block is None:
                return
            buf = np.concatenate([buf, block])
            while len(buf) >= self._frame_samples:
                yield buf[: self._frame_samples]
                buf = buf[self._frame_samples :]
