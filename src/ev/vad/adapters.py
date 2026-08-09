"""VAD 适配器与独立的端点状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class VADAdapter:
    model_id = ""

    def __init__(self, model_path: str, model: Any | None = None, chunk_ms: int = 200):
        self.model_id = model_path
        self.model = model
        self.cache: dict[str, Any] = {}
        self.chunk_ms = chunk_ms
        self._buffer = np.empty(0, dtype=np.float32)
        if model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise RuntimeError("VAD 需要安装 FunASR，请先安装运行时依赖") from exc
            self.model = AutoModel(model=model_path, disable_update=True, disable_pbar=True)

    def accept(
        self, frame: np.ndarray, sample_rate: int = 16000, is_final: bool = False
    ) -> tuple["VADBoundary", ...]:
        """缓冲原始帧并输出 FSMN-VAD 的开始/结束边界。"""
        self._buffer = np.concatenate(
            [self._buffer, np.asarray(frame, dtype=np.float32).reshape(-1)]
        )
        chunk_samples = sample_rate * self.chunk_ms // 1000
        boundaries: list[VADBoundary] = []
        while self._buffer.size >= chunk_samples:
            chunk = self._buffer[:chunk_samples]
            self._buffer = self._buffer[chunk_samples:]
            boundaries.extend(self._generate(chunk, sample_rate, False))
        if is_final:
            chunk = self._buffer
            self._buffer = np.empty(0, dtype=np.float32)
            if chunk.size == 0:
                chunk = np.zeros(chunk_samples, dtype=np.float32)
            boundaries.extend(self._generate(chunk, sample_rate, True))
        return tuple(boundaries)

    def _generate(
        self, chunk: np.ndarray, sample_rate: int, is_final: bool
    ) -> list["VADBoundary"]:
        result = self.model.generate(
            input=chunk,
            sampling_rate=sample_rate,
            cache=self.cache,
            is_final=is_final,
            chunk_size=self.chunk_ms,
            disable_pbar=True,
        )
        item = result[0] if isinstance(result, list) and result else result
        if not isinstance(item, dict):
            return []
        value = item.get("value", [])
        if not isinstance(value, list):
            return []
        boundaries: list[VADBoundary] = []
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            start_ms, end_ms = int(pair[0]), int(pair[1])
            boundaries.append(
                VADBoundary(
                    started=start_ms >= 0,
                    ended=end_ms >= 0,
                    start_ms=start_ms if start_ms >= 0 else None,
                    end_ms=end_ms if end_ms >= 0 else None,
                )
            )
        return boundaries

    def reset(self) -> None:
        self.cache.clear()
        self._buffer = np.empty(0, dtype=np.float32)


@dataclass(frozen=True)
class VADBoundary:
    started: bool = False
    ended: bool = False
    start_ms: int | None = None
    end_ms: int | None = None


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
