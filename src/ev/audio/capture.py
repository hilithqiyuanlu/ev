"""音频采集:16kHz mono 定长帧,异步产出。

对齐全天候/双工:帧流以异步迭代器暴露,后续 VAD/ASR 直接消费;
设备抽象 —— 内置麦与 DJI 接收器只差一个 selector 字符串。

可选: 挂载 AudioPreprocessor, frames() 返回预处理后帧;
      frames_with_raw() 同时返回 (processed, raw)。
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from typing import Tuple

import numpy as np
import sounddevice as sd

from ..config import AudioSettings
from .preprocess import AudioPreprocessor, PreprocessParams


class AudioCapture:
    def __init__(
        self,
        audio: AudioSettings,
        device: int | None = None,
        frame_ms: int = 30,
        preprocessor: AudioPreprocessor | None = None,
    ):
        self._audio = audio
        self._device = device
        self._frame_ms = int(frame_ms)
        self._frame_samples = audio.sample_rate * frame_ms // 1000
        # 主队列: 归一化后的定长帧 (预处理后, 若挂载 preprocessor)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=128)
        # 原始队列: 与主队列同步, 仅当调用 frames_with_raw 时消费
        self._raw_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=128)
        self._stream: sd.InputStream | None = None
        self._preprocessor = preprocessor

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def preprocessor(self) -> AudioPreprocessor | None:
        return self._preprocessor

    @property
    def has_preprocessor(self) -> bool:
        return self._preprocessor is not None

    def _put_both(self, raw_block: np.ndarray) -> None:
        """原子写入 raw + processed 两个队列; 溢出时丢弃最旧保持同步."""
        processed_block = (
            self._preprocessor.process_frame(raw_block)
            if self._preprocessor is not None
            else raw_block
        )
        # 统一先尝试非阻塞写入, 若满则两边各丢一个最旧块
        def _put(q: queue.Queue, item: np.ndarray) -> None:
            try:
                q.put_nowait(item)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(item)
                except queue.Full:
                    pass
        _put(self._queue, processed_block)
        _put(self._raw_queue, raw_block)

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # status 溢出时丢帧优先于阻塞回调(回调线程绝不能卡)
        raw = indata[:, 0].copy().astype(np.float32, copy=False)
        self._put_both(raw)

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
        # 关闭标记: 两边队列各放一个 None (尽力)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._raw_queue.put_nowait(None)
        except queue.Full:
            pass

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """产出定长帧(float32, frame_ms 毫秒).

        若挂载了 preprocessor → 返回预处理后帧.
        各设备 blocksize 不一, 在此归一.
        """
        buf = np.empty(0, dtype=np.float32)
        while True:
            block = await asyncio.to_thread(self._queue.get)
            if block is None:
                return
            buf = np.concatenate([buf, block])
            while len(buf) >= self._frame_samples:
                yield buf[: self._frame_samples]
                buf = buf[self._frame_samples :]

    async def frames_with_raw(self) -> AsyncIterator[Tuple[np.ndarray, np.ndarray]]:
        """产出 (processed_frame, raw_frame) 定长帧对.

        processed_frame: 若挂载 preprocessor 为处理后, 否则等于 raw_frame
        raw_frame: 声卡原始 PCM 数据 (归一化到定长)
        """
        buf_p = np.empty(0, dtype=np.float32)
        buf_r = np.empty(0, dtype=np.float32)
        while True:
            block_p = await asyncio.to_thread(self._queue.get)
            # 同步取 raw; 如果取不到, 用 processed 代替做 degrade (不抛异常断流)
            try:
                block_r = self._raw_queue.get_nowait()
            except queue.Empty:
                block_r = None if block_p is None else block_p
            if block_p is None:
                return
            buf_p = np.concatenate([buf_p, block_p])
            buf_r = np.concatenate([buf_r, block_r])
            while len(buf_p) >= self._frame_samples and len(buf_r) >= self._frame_samples:
                yield buf_p[: self._frame_samples], buf_r[: self._frame_samples]
                buf_p = buf_p[self._frame_samples :]
                buf_r = buf_r[self._frame_samples :]

