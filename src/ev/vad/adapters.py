"""VAD 适配器与独立的端点状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class VADAdapter:
    model_id = ""

    def __init__(self, model_path: str, model: Any | None = None):
        self.model_id = model_path
        self.model = model
        self.cache: dict[str, Any] = {}
        if model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise RuntimeError("VAD 需要安装 FunASR，请先安装运行时依赖") from exc
            self.model = AutoModel(model=model_path, disable_update=True)

    def is_speech(self, frame: np.ndarray, sample_rate: int = 16000) -> bool:
        result = self.model.generate(
            input=frame, sampling_rate=sample_rate, cache=self.cache, is_final=False
        )
        # FunASR VAD 返回格式跨版本不同，统一取最后一个时间区间/状态。
        if not result:
            return False
        item = result[0] if isinstance(result, list) else result
        if isinstance(item, dict):
            value = item.get("value", item.get("text", item.get("vad", False)))
            if isinstance(value, list) and value:
                return value[-1][-1] != -1
            return bool(value)
        return bool(item)

    def reset(self) -> None:
        self.cache.clear()


@dataclass(frozen=True)
class VADState:
    speech: bool
    started: bool = False
    ended: bool = False


class EndpointState:
    """将逐帧 speech 状态变成带 pre-roll/hangover 的端点事件。"""

    def __init__(self, pre_roll_frames: int = 3, hangover_frames: int = 10):
        self.pre_roll_frames = max(0, pre_roll_frames)
        self.hangover_frames = max(1, hangover_frames)
        self.active = False
        self._silence = 0

    def update(self, speech: bool) -> VADState:
        if speech:
            started = not self.active
            self.active = True
            self._silence = 0
            return VADState(True, started=started)
        if not self.active:
            return VADState(False)
        self._silence += 1
        if self._silence >= self.hangover_frames:
            self.active = False
            self._silence = 0
            return VADState(False, ended=True)
        return VADState(True)

    def flush(self) -> VADState:
        if self.active:
            self.active = False
            self._silence = 0
            return VADState(False, ended=True)
        return VADState(False)
