"""段末声学特征检测 — 不依赖模型, 纯信号处理区分人声/非人声.

设计目标: 在质量门控之后做第二道过滤, 拒绝明显非人声的段
(风扇/空调稳态噪声、键盘连续敲击、金属刮擦等).

策略: 多重特征综合判断, 单一特征不轻易拒绝 (保守).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


def compute_zcr(audio: np.ndarray) -> float:
    """过零率 — 波形穿越零轴的频率.

    人声: 0.05-0.30 (声带开合有节奏)
    稳态噪声(风扇): < 0.02 (极低)
    脉冲噪声(键盘): > 0.40 (每次敲击都是高频瞬态)
    """
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(arr) < 2:
        return 0.0
    signs = np.sign(arr)
    # 忽略静音区: 振幅 < 峰值的 1% 时不计数 (防噪声地板干扰)
    threshold = np.max(np.abs(arr)) * 0.01
    mask = np.abs(arr) > threshold
    signs[~mask] = 0
    crossings = np.sum(np.abs(np.diff(signs[mask]))) / 2 if np.any(mask) else 0
    return float(crossings / max(len(arr), 1))


def compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """频谱质心 (Hz) — 频率能量的加权平均.

    人声: 300-2500 Hz (能量集中在基频+共振峰)
    摩擦/刮擦/高频噪声: >4000 Hz
    AC嗡嗡声: <200 Hz
    """
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = len(arr)
    if n < 16:
        return 0.0
    # 小窗口平均 (30ms) 减少单帧方差
    win_len = min(n, int(sr * 0.030))
    hop = max(1, win_len // 2)
    window = get_window("hann", win_len)
    centroids: list[float] = []
    for start in range(0, n - win_len + 1, hop):
        frame = arr[start:start + win_len] * window
        spec = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(win_len, 1.0 / sr)
        total = np.sum(spec)
        if total > 1e-10:
            centroids.append(float(np.sum(freqs * spec) / total))
    if not centroids:
        return 0.0
    # 取中位数 (防瞬时高频尖峰拉高均值)
    return float(np.median(centroids))


def compute_spectral_flatness(audio: np.ndarray, sr: int) -> float:
    """频谱平坦度 (0-1) — 几何均值/算术均值.

    人声: < 0.3 (谐波结构 → 频谱有峰谷)
    白噪声/稳态噪声: > 0.7 (频谱平坦)
    """
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = len(arr)
    if n < 16:
        return 0.0
    win_len = min(n, int(sr * 0.030))
    hop = max(1, win_len // 2)
    window = get_window("hann", win_len)
    flatness_vals: list[float] = []
    for start in range(0, n - win_len + 1, hop):
        frame = arr[start:start + win_len] * window
        power = np.abs(np.fft.rfft(frame)) ** 2 + 1e-12
        geo_mean = np.exp(np.mean(np.log(power)))
        arith_mean = np.mean(power)
        flatness_vals.append(float(geo_mean / arith_mean))
    if not flatness_vals:
        return 0.0
    return float(np.mean(flatness_vals))


def compute_rms_envelope_variance(audio: np.ndarray, sr: int, frame_ms: int = 50) -> float:
    """RMS 包络方差 — 幅度随时间的变化程度.

    人声: 高 (音节/停顿 → 包络起伏大)
    稳态噪声: 极低 (持续平稳)
    键盘: 中等 (每次敲击一个尖峰)
    """
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = len(arr)
    frame_len = int(sr * frame_ms / 1000)
    if frame_len < 1 or n < frame_len:
        return 0.0
    rms_vals: list[float] = []
    for start in range(0, n - frame_len + 1, frame_len):
        frame = arr[start:start + frame_len]
        rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))
    if len(rms_vals) < 2:
        return 0.0
    rms_arr = np.array(rms_vals)
    mean_rms = np.mean(rms_arr)
    if mean_rms < 1e-10:
        return 0.0
    return float(np.var(rms_arr) / (mean_rms ** 2))  # 归一化方差


def classify_voice(audio: np.ndarray, sr: int) -> dict:
    """综合判断是否为有效人声.

    Returns:
        dict with keys:
        - is_voice: bool  (False = 明显非人声)
        - zcr: float
        - centroid_hz: float
        - flatness: float
        - rms_var: float
        - reasons: list[str]  (非人声的原因)
    """
    if audio is None or not isinstance(audio, np.ndarray) or audio.size < sr * 0.1:
        return {"is_voice": True, "reasons": []}  # 太短不判断

    zcr = compute_zcr(audio)
    centroid = compute_spectral_centroid(audio, sr)
    flatness = compute_spectral_flatness(audio, sr)
    rms_var = compute_rms_envelope_variance(audio, sr)

    reasons: list[str] = []
    is_voice = True

    # 规则 1: 极低过零率 → 稳态低频噪声 (风扇/AC嗡鸣)
    #         人声最轻声也有 >0.03 的 ZCR
    if zcr < 0.015:
        reasons.append(f"zcr_too_low({zcr:.4f})")
        is_voice = False

    # 规则 2: 极高过零率 + 高频谱质心 → 脉冲性非人声 (键盘连敲/金属刮擦)
    #         键盘敲击 ZCR 可达 0.5+, 但短段可能被 OBSERVING 过滤
    if zcr > 0.45 and centroid > 3500:
        reasons.append(f"zcr_high+centroid_high({zcr:.2f},{centroid:.0f}Hz)")
        is_voice = False

    # 规则 3: 频谱极平坦 → 白噪声/粉红噪声 (风扇湍流)
    #         人声有谐波 → flatness < 0.4; 纯噪声 > 0.65
    if flatness > 0.65 and zcr < 0.15:
        reasons.append(f"flat_spectrum({flatness:.2f})")
        is_voice = False

    # 规则 4: 极低包络方差 + 频谱平坦 → 持续稳态噪声
    #         人声说话时包络必然有起伏
    if rms_var < 0.05 and flatness > 0.5:
        reasons.append(f"steady_noise(rms_var={rms_var:.3f},flatness={flatness:.2f})")
        is_voice = False

    return {
        "is_voice": is_voice,
        "zcr": zcr,
        "centroid_hz": centroid,
        "flatness": flatness,
        "rms_var": rms_var,
        "reasons": reasons,
    }
