"""音频采集:16kHz mono 定长帧,异步产出。

对齐全天候/双工:帧流以异步迭代器暴露,后续 VAD/ASR 直接消费;
设备抽象 —— 内置麦与 DJI 接收器只差一个 selector 字符串。

可选: 挂载 AudioPreprocessor, frames() 返回预处理后帧;
      frames_with_raw() 同时返回 (processed, raw)。
"""

from __future__ import annotations

import asyncio
import queue
import sys
from collections.abc import AsyncIterator
from typing import Tuple

import numpy as np
import sounddevice as sd

from ..config import AudioSettings
from .preprocess import AudioPreprocessor, PreprocessParams


def _safe_pa_reset() -> None:
    """当流异常后重置 PortAudio 内部状态,解决 macOS 上 -9986 Internal PortAudio error。

    PortAudio 在 macOS(CoreAudio)后端对异常终止的流可能残留内部锁/状态,
    导致下次 Pa_OpenStream 返回 -9986。 terminate+initialize 序列可清理状态。
    """
    try:
        sd._terminate()
    except Exception:
        pass
    try:
        sd._initialize()
    except Exception:
        pass


def _validate_device(device: int | None, samplerate: int, channels: int) -> tuple[int | None, str]:
    """校验设备是否支持请求的参数,返回 (可能调整后的device_index, 警告信息)。

    若指定 device 不合法或不可用,回退到系统默认设备并返回警告。
    """
    warning = ""
    if device is not None:
        try:
            dev_info = sd.query_devices(device)
        except (sd.PortAudioError, ValueError):
            warning = f"输入设备索引 {device} 不可用,已回退到系统默认麦克风"
            return None, warning
        if dev_info["max_input_channels"] < channels:
            warning = f"输入设备 {dev_info['name']!r} 支持通道数不足,已回退到系统默认麦克风"
            return None, warning
        # 检查请求的采样率是否受支持
        try:
            sd.check_input_settings(
                device=device,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
            )
        except sd.PortAudioError:
            # 设备不支持该采样率,尝试使用设备默认采样率
            default_sr = dev_info.get("default_samplerate", samplerate)
            if abs(default_sr - samplerate) > 1:
                warning = (
                    f"输入设备 {dev_info['name']!r} 不支持 {samplerate}Hz 采样率"
                    f"(默认 {default_sr:.0f}Hz),尝试仍以 {samplerate}Hz 打开"
                )
    return device, warning


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
        # blocksize: 与帧长对齐,避免 PortAudio 内部缓冲区不匹配导致 -9986
        self._blocksize = self._frame_samples
        # 主队列: 归一化后的定长帧 (预处理后, 若挂载 preprocessor)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=128)
        # 原始队列: 与主队列同步, 仅当调用 frames_with_raw 时消费
        self._raw_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=128)
        self._stream: sd.InputStream | None = None
        self._preprocessor = preprocessor
        self._started = False
        self._warning: str = ""

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def preprocessor(self) -> AudioPreprocessor | None:
        return self._preprocessor

    @property
    def has_preprocessor(self) -> bool:
        return self._preprocessor is not None

    @property
    def warning(self) -> str:
        """启动时产生的非致命警告(如设备回退),供上层日志记录。"""
        return self._warning

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
        if status:
            # xrun/overflow 等状态不致命,仅在 stderr 输出一次,避免刷屏
            if not hasattr(self, "_status_reported"):
                print(f"[audio] PortAudio status: {status}", file=sys.stderr)
                self._status_reported = True
        raw = indata[:, 0].copy().astype(np.float32, copy=False)
        self._put_both(raw)

    def start(self) -> None:
        if self._started:
            return
        sr = self._audio.sample_rate
        ch = self._audio.channels

        # 校验设备参数,必要时回退到默认设备
        device, warn = _validate_device(self._device, sr, ch)
        self._warning = warn
        if warn:
            print(f"[audio] warning: {warn}", file=sys.stderr)

        # 尝试打开流;失败时重置 PortAudio 再重试一次 (解决 macOS -9986)
        stream = None
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                stream = sd.InputStream(
                    device=device,
                    samplerate=sr,
                    channels=ch,
                    dtype="float32",
                    blocksize=self._blocksize,
                    latency="high",
                    callback=self._callback,
                )
                stream.start()
                break
            except sd.PortAudioError as exc:
                last_err = exc
                # 检测 -9986 Internal PortAudio error: args=(message_string, error_code)
                is_internal_error = (
                    (len(exc.args) >= 2 and exc.args[1] == -9986)
                    or "-9986" in str(exc)
                )
                if attempt == 0 and is_internal_error:
                    # Internal PortAudio error: 尝试重置 PA 后端再重试
                    print(f"[audio] PortAudio internal error (-9986), resetting backend...", file=sys.stderr)
                    _safe_pa_reset()
                    # 重置后重新校验设备(设备索引可能在重置后变化)
                    if self._device is not None:
                        try:
                            sd.query_devices(self._device)
                        except Exception:
                            device = None
                    continue
                raise
        if stream is None:
            raise RuntimeError(
                f"无法打开音频输入流: {last_err}.\n"
                "请检查: 1) 系统设置→隐私与安全性→麦克风 是否已授权终端/App; "
                "2) 输入设备是否被其他应用占用; "
                "3) 尝试重新选择输入设备。"
            ) from last_err
        self._stream = stream
        self._started = True

    def stop(self) -> None:
        if not self._started:
            # 确保即使未启动也发送终止信号,防止 consumer 永久阻塞
            self._put_none()
            return
        self._started = False
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._put_none()

    def _put_none(self) -> None:
        """向队列发送终止哨兵,防止 async iterator 永久阻塞。"""
        for q in (self._queue, self._raw_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(None)
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
