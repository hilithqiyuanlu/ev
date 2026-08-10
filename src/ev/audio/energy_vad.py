"""能量级 VAD: 噪声底噪追踪 + 自适应 SNR 阈值, 用作 FSMN-VAD 的兜底.

设计原则: 宁可误报不要漏报. 与 FSMN 组合时: start 用 OR, end 用 AND + hangover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyVADParams:
    """能量 VAD 参数."""

    snr_threshold_linear: float = 2.5      # RMS > floor * 2.5x (≈4dB) 视为语音
    abs_min_rms: float = 0.001             # 绝对下限, 防止静音段纯噪声触发 (~-60dBFS)
    start_frames: int = 2                  # 连续 N 帧满足才启动 (60ms @ 30ms/帧)
    hangover_frames: int = 20              # 语音消失后继续保持 M 帧 (600ms @ 30ms/帧)
    floor_track_sec: float = 3.0           # 底噪追踪窗口
    floor_min_rms: float = 1e-5            # 底噪下限 (防止太安静, SNR虚高) ~-100dBFS

    @classmethod
    def default(cls) -> "EnergyVADParams":
        return cls()


@dataclass(frozen=True)
class EnergyVADState:
    """逐帧输出的 VAD 状态, 与 vad/adapters.py 的 VADState 对齐字段."""

    speech: bool          # 当前帧是否处于语音段
    started: bool = False # 本帧是否是段起始 (speech 从 False → True)
    ended: bool = False   # 本帧是否是段结束 (speech 从 True → False)


class EnergyVAD:
    """流式逐帧能量 VAD. 维护跨帧的底噪、hangover、启动计数状态."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        params: EnergyVADParams | None = None,
    ) -> None:
        self._sr = int(sample_rate)
        self._frame_ms = int(frame_ms)
        self._params = params or EnergyVADParams.default()
        # 底噪 EMA: 向更小值缓慢跟踪
        frames_per_sec = 1000.0 / float(frame_ms)
        self._floor_alpha = 1.0 - np.exp(-1.0 / (self._params.floor_track_sec * frames_per_sec))
        self._floor_rms: float = 0.0
        self._floor_initialized: bool = False
        # 端点状态
        self._active: bool = False
        self._start_streak: int = 0  # 连续满足条件的帧数 (未 active 时统计)
        self._silence_streak: int = 0  # 连续不满足条件的帧数 (active 后统计)
        # 诊断指标
        self._last_rms: float = 0.0
        self._last_snr_linear: float = 0.0

    def reset(self) -> None:
        self._floor_rms = 0.0
        self._floor_initialized = False
        self._active = False
        self._start_streak = 0
        self._silence_streak = 0
        self._last_rms = 0.0
        self._last_snr_linear = 0.0

    # --- 诊断属性 ---

    @property
    def active(self) -> bool:
        return self._active

    @property
    def floor_rms(self) -> float:
        return self._floor_rms

    @property
    def last_rms(self) -> float:
        return self._last_rms

    @property
    def last_snr_linear(self) -> float:
        return self._last_snr_linear

    @property
    def silence_ms(self) -> float:
        """How many ms of consecutive silence while active (0 if not active or still speaking)."""
        if not self._active:
            return 0.0
        return self._silence_streak * self._frame_ms

    # --- 核心处理 ---

    def _update_floor(self, rms: float) -> None:
        if not self._floor_initialized:
            self._floor_rms = max(rms, self._params.floor_min_rms)
            self._floor_initialized = True
            return
        # 只朝更小值跟踪 (环境噪声只会缓慢下降, 快速上升通常是语音, 不计入底噪)
        if rms < self._floor_rms:
            self._floor_rms = (
                self._floor_alpha * rms
                + (1.0 - self._floor_alpha) * self._floor_rms
            )
        self._floor_rms = max(self._floor_rms, self._params.floor_min_rms)

    def _is_speech_frame(self, rms: float) -> bool:
        """单帧 SNR + 绝对下限判断."""
        if rms < self._params.abs_min_rms:
            return False
        floor = max(self._floor_rms, self._params.floor_min_rms)
        snr_linear = rms / floor if floor > 0 else 0.0
        self._last_snr_linear = snr_linear
        return snr_linear >= self._params.snr_threshold_linear

    def accept_frame(self, frame: np.ndarray) -> EnergyVADState:
        """处理一帧, 返回状态 (包含 started/ended 边沿)."""
        if frame.size == 0:
            return EnergyVADState(self._active)
        rms = float(np.sqrt(np.mean(np.square(np.asarray(frame, dtype=np.float64).reshape(-1)))))
        self._last_rms = rms
        self._update_floor(rms)
        hit = self._is_speech_frame(rms)

        started = False
        ended = False
        if not self._active:
            # 未启动: 累加 start_streak
            if hit:
                self._start_streak += 1
                if self._start_streak >= self._params.start_frames:
                    self._active = True
                    started = True
                    self._silence_streak = 0
                    self._start_streak = 0  # 启动后清零
            else:
                self._start_streak = 0
        else:
            # 已启动: 累加 silence_streak, 超过 hangover 切非 active
            if hit:
                self._silence_streak = 0
            else:
                self._silence_streak += 1
                if self._silence_streak >= self._params.hangover_frames:
                    self._active = False
                    ended = True
                    self._silence_streak = 0
                    self._start_streak = 0
        return EnergyVADState(self._active, started=started, ended=ended)

    def flush(self) -> EnergyVADState:
        """强制结束当前语音段 (用户停止/flush final)."""
        if self._active:
            self._active = False
            self._silence_streak = 0
            self._start_streak = 0
            return EnergyVADState(False, ended=True)
        return EnergyVADState(False)
