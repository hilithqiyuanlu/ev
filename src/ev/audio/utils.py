"""音频工具函数 — 重采样等通用操作。"""

from __future__ import annotations

import math

import numpy as np


_RESAMPLE_TAPS_PER_PHASE = 12


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """使用窗函数低通滤波进行有理数重采样。

    Args:
        audio: 输入音频，shape (n_samples,) float32.
        orig_sr: 原始采样率。
        target_sr: 目标采样率。

    Returns:
        重采样后音频，同 dtype。
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("采样率必须为正数")
    if orig_sr == target_sr:
        return arr
    if arr.size < 2:
        return arr.copy()

    divisor = math.gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    target_len = max(1, round(len(arr) * target_sr / orig_sr))

    # 在升采样后的采样域构造低通 FIR。截止频率按较大的变换倍率收紧，
    # 同时抑制升采样镜像和降采样混叠；Hamming 窗避免硬截断振铃。
    max_rate = max(up, down)
    half_width = _RESAMPLE_TAPS_PER_PHASE * max_rate
    taps = np.arange(-half_width, half_width + 1, dtype=np.float64)
    cutoff = 1.0 / max_rate
    kernel = cutoff * np.sinc(cutoff * taps)
    kernel *= np.hamming(kernel.size)
    kernel *= up / np.sum(kernel)

    upsampled = np.zeros(arr.size * up, dtype=np.float64)
    upsampled[::up] = arr
    filtered = np.convolve(upsampled, kernel, mode="full")
    result = filtered[half_width::down]
    if result.size < target_len:
        result = np.pad(result, (0, target_len - result.size))
    elif result.size > target_len:
        result = result[:target_len]
    return result.astype(np.float32)
