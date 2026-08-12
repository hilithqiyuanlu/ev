"""音频工具函数 — 重采样等通用操作。"""

from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """简单线性插值重采样。

    Args:
        audio: 输入音频，shape (n_samples,) float32.
        orig_sr: 原始采样率。
        target_sr: 目标采样率。

    Returns:
        重采样后音频，同 dtype。
    """
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return arr.copy()
    duration = len(arr) / orig_sr
    target_len = max(1, int(duration * target_sr))
    indices = np.linspace(0, len(arr) - 1, target_len)
    return np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)
