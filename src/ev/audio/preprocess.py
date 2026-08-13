"""帧级音频预处理管线: DC 移除 → 预加重 → AGC → (可选) 噪声门.

全 numpy 实现, 无额外第三方依赖. 单帧处理目标 < 0.5ms @ 16kHz/30ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreprocessParams:
    """预处理参数集合. 全部使用可配置默认值."""

    # Preemphasis
    preemphasis_coeff: float = 0.97
    # AGC
    agc_target_rms: float = 0.05
    agc_min_gain: float = 0.1       # -20dB, 防止超大音量近讲
    agc_max_gain: float = 20.0      # +26dB, 远场小声补偿
    agc_attack_ms: float = 10.0     # 增益下降 (音量变大时压缩) 快
    agc_release_ms: float = 100.0   # 增益上升 (音量变小时放大) 慢
    # NoiseGate
    noisegate_enabled: bool = True
    noisegate_snr_db: float = 3.0   # 低于此 SNR 时开始衰减
    noisegate_floor_track_sec: float = 3.0  # 底噪追踪窗口

    @classmethod
    def default(cls) -> "PreprocessParams":
        return cls()


class DCRemover:
    """一阶 IIR 高通滤除 DC 偏移 (截止 ~20Hz @ 16kHz)."""

    def __init__(self, sample_rate: int = 16000, cutoff_hz: float = 20.0) -> None:
        # α = 1 / (1 + 2π * cutoff / fs)
        self._alpha = 1.0 / (1.0 + 2.0 * np.pi * cutoff_hz / sample_rate)
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0

    def reset(self) -> None:
        self._prev_x = 0.0
        self._prev_y = 0.0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.size == 0:
            return frame
        out = np.empty_like(frame, dtype=np.float32)
        alpha = self._alpha
        px, py = self._prev_x, self._prev_y
        for i in range(frame.size):
            x = float(frame[i])
            y = x - px + alpha * py
            out[i] = y
            px, py = x, y
        self._prev_x, self._prev_y = px, py
        return out


class Preemphasis:
    """一阶 FIR 预加重, y[n] = x[n] - coeff * x[n-1]."""

    def __init__(self, coeff: float = 0.97) -> None:
        self._coeff = float(coeff)
        self._prev: float = 0.0

    def reset(self) -> None:
        self._prev = 0.0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.size == 0:
            return frame
        out = np.empty_like(frame, dtype=np.float32)
        coeff = self._coeff
        prev = self._prev
        for i in range(frame.size):
            x = float(frame[i])
            out[i] = x - coeff * prev
            prev = x
        self._prev = prev
        return out


class AGC:
    """自动增益控制: target_rms + attack/release 时间平滑.

    状态跨帧保持, 避免逐帧独立缩放造成的噪声放大突变.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_rms: float = 0.05,
        min_gain: float = 0.1,
        max_gain: float = 20.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        frame_ms: float = 30.0,
    ) -> None:
        self._target_rms = float(target_rms)
        self._min_gain = float(min_gain)
        self._max_gain = float(max_gain)
        # attack: gain 下降 (信号变大) 时用快系数
        # release: gain 上升 (信号变小) 时用慢系数
        frames_per_sec = 1000.0 / frame_ms
        self._attack_k = 1.0 - np.exp(-1.0 / (attack_ms / 1000.0 * frames_per_sec))
        self._release_k = 1.0 - np.exp(-1.0 / (release_ms / 1000.0 * frames_per_sec))
        self._current_gain: float = 1.0
        # floor 安全阈值, 防止除零 / 静音爆增益
        self._min_rms = 1e-6 * self._target_rms

    def reset(self, gain: float = 1.0) -> None:
        self._current_gain = float(gain)

    @property
    def current_gain(self) -> float:
        return self._current_gain

    @property
    def max_gain(self) -> float:
        return self._max_gain

    def set_max_gain(self, max_gain: float) -> None:
        """动态调整增益上限 (环境联动: 噪声环境下压低, 避免放大噪声)."""
        self._max_gain = float(max_gain)
        if self._current_gain > self._max_gain:
            self._current_gain = self._max_gain

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """返回 (处理后帧, 应用的平滑增益)."""
        if frame.size == 0:
            return frame, self._current_gain
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))
        if rms < self._min_rms:
            # 极静音段: 缓慢向 min_gain 回落, 不爆推
            target_gain = self._min_gain
        else:
            target_gain = self._target_rms / rms
        target_gain = max(self._min_gain, min(self._max_gain, target_gain))
        # 根据上升/下降选平滑系数
        if target_gain < self._current_gain:
            # 压缩 (attack)
            k = self._attack_k
        else:
            # 放大 (release)
            k = self._release_k
        smoothed = k * target_gain + (1.0 - k) * self._current_gain
        smoothed = max(self._min_gain, min(self._max_gain, smoothed))
        self._current_gain = smoothed
        out = (frame * smoothed).astype(np.float32)
        # 削波保护: peak 超 0.98 则缩放
        peak = float(np.max(np.abs(out)))
        if peak > 0.98:
            scale = 0.98 / peak
            out = (out * scale).astype(np.float32)
            # 同步回拉 current_gain 以防止下次继续过推
            self._current_gain = smoothed * scale
        return out, self._current_gain


# 环境联动: YAMNet 判定的「明确非语音噪声」类别 → 收紧前端降噪。
# 仅对不可能是用户语音的类别收紧, 避免误伤用户说话。
_ENV_NOISY_CATEGORIES: frozenset[str] = frozenset(
    {"typing", "background_noise", "music", "appliance"}
)
_ENV_NOISY_SNR_DB = 9.0     # 噪声环境下 NoiseGate SNR 门限 (默认 1.5dB → 9dB)
_ENV_NOISY_MAX_GAIN = 6.0   # 噪声环境下 AGC 增益上限 (默认 40x → 6x)


class NoiseGate:
    """轻度噪声门: 追踪噪声底噪, SNR 低于阈值时线性衰减."""

    def __init__(
        self,
        sample_rate: int = 16000,
        snr_db_threshold: float = 3.0,
        floor_track_sec: float = 3.0,
        frame_ms: float = 30.0,
    ) -> None:
        self._snr_db = float(snr_db_threshold)
        # SNR 阈值转线性倍数
        self._snr_linear = 10.0 ** (self._snr_db / 20.0)
        # floor EMA: 朝更小值缓慢跟踪 (floor_track_sec 时间常数)
        frames_per_sec = 1000.0 / frame_ms
        self._floor_alpha = 1.0 - np.exp(-1.0 / (floor_track_sec * frames_per_sec))
        self._floor_rms: float = 0.0
        self._initialized: bool = False

    def reset(self) -> None:
        self._floor_rms = 0.0
        self._initialized = False

    @property
    def floor_rms(self) -> float:
        return self._floor_rms

    @property
    def current_snr_db(self) -> float:
        return self._snr_db

    def set_snr_threshold_db(self, snr_db: float) -> None:
        """动态调整 SNR 门限 (环境联动: 噪声环境下收紧)."""
        self._snr_db = float(snr_db)
        self._snr_linear = 10.0 ** (self._snr_db / 20.0)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """返回 (处理后帧, 当前帧 RMS)."""
        if frame.size == 0:
            return frame, 0.0
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))
        # 初始化: 前几帧直接当底噪
        if not self._initialized:
            self._floor_rms = rms if rms > 0 else 1e-7
            self._initialized = True
            return frame, rms
        # floor 向更小值缓慢跟踪 (环境噪声只会缓慢变化, 不会突然变小)
        if rms < self._floor_rms:
            self._floor_rms = (
                self._floor_alpha * rms
                + (1.0 - self._floor_alpha) * self._floor_rms
            )
        # 计算当前 SNR (线性)
        if self._floor_rms <= 0:
            self._floor_rms = 1e-7
        snr_linear = rms / self._floor_rms
        if snr_linear >= self._snr_linear:
            # SNR 足够 -> 完全通过
            return frame, rms
        # 线性衰减: 从 gate 阈值(倍数1) 衰减到 静音 floor(倍数0.1)
        ratio = snr_linear / self._snr_linear  # 0..1
        attenuation = 0.1 + 0.9 * ratio  # 0.1..1.0
        return (frame * attenuation).astype(np.float32), rms


class AudioPreprocessor:
    """DCRemover → Preemphasis → AGC → (可选) NoiseGate 的流式管线."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        params: PreprocessParams | None = None,
    ) -> None:
        self._params = params or PreprocessParams.default()
        self._sr = int(sample_rate)
        self._frame_ms = int(frame_ms)
        p = self._params
        self._dc = DCRemover(sample_rate=self._sr)
        self._pe = Preemphasis(coeff=p.preemphasis_coeff)
        self._agc = AGC(
            sample_rate=self._sr,
            target_rms=p.agc_target_rms,
            min_gain=p.agc_min_gain,
            max_gain=p.agc_max_gain,
            attack_ms=p.agc_attack_ms,
            release_ms=p.agc_release_ms,
            frame_ms=float(frame_ms),
        )
        self._ng: NoiseGate | None = None
        if p.noisegate_enabled:
            self._ng = NoiseGate(
                sample_rate=self._sr,
                snr_db_threshold=p.noisegate_snr_db,
                floor_track_sec=p.noisegate_floor_track_sec,
                frame_ms=float(frame_ms),
            )
        # 环境联动: 记录默认参数, 便于噪声环境收紧后还原
        self._default_snr_db = float(p.noisegate_snr_db)
        self._default_max_gain = float(p.agc_max_gain)
        # 诊断: 最近应用的 AGC 增益, 便于日志
        self._last_gain: float = 1.0

    def reset(self) -> None:
        self._dc.reset()
        self._pe.reset()
        self._agc.reset()
        if self._ng is not None:
            self._ng.reset()
        self._last_gain = 1.0

    @property
    def last_gain(self) -> float:
        return self._last_gain

    @property
    def floor_rms(self) -> float:
        return self._ng.floor_rms if self._ng is not None else 0.0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """逐帧流式处理. 输入输出均为 float32 1D ndarray."""
        if frame.size == 0:
            return frame
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        x = self._dc.process_frame(x)
        x = self._pe.process_frame(x)
        x, gain = self._agc.process_frame(x)
        self._last_gain = gain
        if self._ng is not None:
            x, _ = self._ng.process_frame(x)
        return x

    def process_segment(self, audio: np.ndarray) -> np.ndarray:
        """一次性处理整段 (连续多帧 concatenated), 状态跨段保持.

        用于最终 ASR / speaker-switch check 等离线批处理场景.
        """
        if audio.size == 0:
            return audio
        samples_per_frame = self._sr * self._frame_ms // 1000
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        out_chunks: list[np.ndarray] = []
        offset = 0
        while offset < x.size:
            chunk = x[offset:offset + samples_per_frame]
            out_chunks.append(self.process_frame(chunk))
            offset += chunk.size
        if not out_chunks:
            return x
        return np.concatenate(out_chunks).astype(np.float32)

    def apply_environment(self, category: str) -> bool:
        """根据 YAMNet 环境分类动态调整前端降噪强度 (环境联动).

        明确的环境噪声类 (键盘/持续底噪/音乐/家电) 下收紧 NoiseGate 门限并压低
        AGC 增益上限, 避免噪声被放大后误触发 VAD 或污染录音; 语音/静音/人声等
        类别还原默认值. 返回是否发生变化.
        """
        noisy = category in _ENV_NOISY_CATEGORIES
        snr_db = _ENV_NOISY_SNR_DB if noisy else self._default_snr_db
        max_gain = _ENV_NOISY_MAX_GAIN if noisy else self._default_max_gain
        changed = False
        if self._ng is not None and abs(self._ng.current_snr_db - snr_db) > 1e-6:
            self._ng.set_snr_threshold_db(snr_db)
            changed = True
        if abs(self._agc.max_gain - max_gain) > 1e-6:
            self._agc.set_max_gain(max_gain)
            changed = True
        return changed
