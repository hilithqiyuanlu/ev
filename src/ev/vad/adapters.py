"""VAD 适配器与独立的端点状态机。

新增:
- VADAdapter 支持 fsmn_threshold 参数 (若 FunASR 接受)
- CompositeVAD: FSMN + 能量级 VAD 组合决策, start=OR end=AND+hangover
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from collections import deque

import numpy as np


class VADAdapter:
    model_id = ""

    def __init__(
        self,
        model_path: str,
        model: Any | None = None,
        chunk_ms: int = 200,
        threshold: float | None = None,
    ):
        self.model_id = model_path
        self.model = model
        self.cache: dict[str, Any] = {}
        self.chunk_ms = chunk_ms
        self.threshold: float | None = threshold  # None = 模型默认
        self._buffer = np.empty(0, dtype=np.float32)
        if model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise RuntimeError("VAD 需要安装 FunASR，请先安装运行时依赖") from exc
            self.model = AutoModel(model=model_path, disable_update=True, disable_pbar=True)

    def update_threshold(self, threshold: float | None) -> None:
        self.threshold = threshold

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
        kwargs: dict[str, Any] = {
            "input": chunk,
            "sampling_rate": sample_rate,
            "cache": self.cache,
            "is_final": is_final,
            "chunk_size": self.chunk_ms,
            "disable_pbar": True,
        }
        # 尝试传入阈值参数; 不同 FunASR 版本参数名不同, 用 try 兜底不生效时回退默认
        if self.threshold is not None:
            for key in ("speech_thresh", "threshold", "speech_threshold"):
                kwargs[key] = float(self.threshold)
        try:
            result = self.model.generate(**kwargs)
        except TypeError:
            # 模型不支持阈值参数 -> 去掉关键字重试, 不抛异常打断链路
            for key in ("speech_thresh", "threshold", "speech_threshold"):
                kwargs.pop(key, None)
            result = self.model.generate(**kwargs)
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

    def unload(self) -> None:
        """Release model resources. Safe to call multiple times."""
        import gc

        self.cache.clear()
        self._buffer = np.empty(0, dtype=np.float32)
        if self.model is not None:
            try:
                # Try to close onnxruntime sessions
                for attr_name in dir(self.model):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        attr = getattr(self.model, attr_name, None)
                        if attr is not None and hasattr(attr, "close"):
                            attr.close()
                    except Exception:
                        pass
                if hasattr(self.model, "model"):
                    try:
                        del self.model.model
                    except Exception:
                        pass
            except Exception:
                pass
            self.model = None
        gc.collect()


class CompositeVAD:
    """复合 VAD: FSMN-VAD + EnergyVAD.

    组合策略:
    - start_mode in {"or", "fsmn_only", "energy_only"}: 语音启动条件
        "or" (默认): 任意一个检测到 start → 启动 (宁可误触发不要漏触发)
    - end_mode in {"and", "fsmn_only", "energy_only"}: 语音结束条件
        "and" (默认): 两个都判定非 active (走完各自 hangover) → 结束 (保守防切断)

    对外接口与 VADAdapter.accept() 完全一致: accept(frame) -> tuple[VADBoundary]
    """

    def __init__(
        self,
        fsmn_vad: VADAdapter,
        energy_vad: Optional[Any] = None,
        start_mode: str = "or",
        end_mode: str = "and",
        sample_rate: int = 16000,
        frame_ms: int = 30,
    ) -> None:
        self.fsmn = fsmn_vad
        self.energy = energy_vad  # None 表示禁用 energy 分支, 退化为纯 FSMN
        self.start_mode = start_mode.lower()
        self.end_mode = end_mode.lower()
        self._sr = sample_rate
        self._frame_ms = frame_ms
        # 内部 state: 模拟逐帧判定 → 边沿 → VADBoundary
        self._active: bool = False
        self._fsmn_pending_start: bool = False  # 本轮 accept 周期内 FSMN 是否报告 start
        self._fsmn_pending_end: bool = False
        # 每帧累计时间戳 (ms), 用于产出边界 start_ms/end_ms
        self._elapsed_ms: int = 0
        self._start_at_ms: Optional[int] = None

    @property
    def active(self) -> bool:
        return self._active

    def _check_start(self, fsmn_started_this_round: bool, energy_active: bool) -> bool:
        mode = self.start_mode
        fsmn_says_yes = fsmn_started_this_round or self._fsmn_pending_start
        energy_says_yes = bool(self.energy is not None and self.energy.active)
        # start 边沿: 上轮非 active, 本轮判定条件满足
        if mode == "fsmn_only":
            return fsmn_says_yes
        if mode == "energy_only":
            return energy_says_yes and energy_active
        # 默认 or
        return fsmn_says_yes or (energy_says_yes and energy_active)

    def _check_end(self, fsmn_ended_this_round: bool, energy_active: bool) -> bool:
        mode = self.end_mode
        fsmn_says_end = fsmn_ended_this_round or self._fsmn_pending_end
        energy_says_active = bool(self.energy is not None and self.energy.active)
        # 注意: FSMN-VAD 的 ended 是边沿事件, 一旦报告就结束.
        # EnergyVAD 的 active=False + hangover 跑完 → 自然结束.
        if mode == "fsmn_only":
            return fsmn_says_end
        if mode == "energy_only":
            # energy_only: energy 从 True→False (ended 边沿) 即结束
            return (self.energy is not None) and (not energy_active)
        # 默认 and: 两个都结束
        fsmn_done = fsmn_says_end
        energy_done = (self.energy is None) or (not energy_active)
        return fsmn_done and energy_done

    def reset(self) -> None:
        self.fsmn.reset()
        if self.energy is not None:
            self.energy.reset()
        self._active = False
        self._fsmn_pending_start = False
        self._fsmn_pending_end = False
        self._elapsed_ms = 0
        self._start_at_ms = None

    def unload(self) -> None:
        """Release underlying model resources."""
        if hasattr(self.fsmn, "unload"):
            try:
                self.fsmn.unload()
            except Exception:
                pass
        if self.energy is not None and hasattr(self.energy, "unload"):
            try:
                self.energy.unload()
            except Exception:
                pass
        self._active = False

    def accept(
        self,
        frame: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = False,
    ) -> tuple["VADBoundary", ...]:
        """逐帧/逐块接口, 与 VADAdapter.accept() 语义一致."""
        frame_np = np.asarray(frame, dtype=np.float32).reshape(-1)
        frame_ms = (
            int(round(frame_np.size * 1000.0 / sample_rate))
            if sample_rate > 0
            else self._frame_ms
        )
        # 1) 驱动 EnergyVAD (逐帧级)
        energy_state: Optional[Any] = None
        if self.energy is not None:
            energy_state = self.energy.accept_frame(frame_np)
            energy_active = bool(energy_state.speech)
            energy_started = bool(energy_state.started)
            energy_ended = bool(energy_state.ended)
        else:
            energy_active = False
            energy_started = False
            energy_ended = False

        # 2) 驱动 FSMN-VAD (块级, 产出 0~多个边界事件)
        fsmn_boundaries = self.fsmn.accept(frame_np, sample_rate, is_final=is_final)
        # 提取本轮 FSMN 的 start/end 边沿汇总
        fsmn_has_start = any(b.started for b in fsmn_boundaries)
        fsmn_has_end = any(b.ended for b in fsmn_boundaries)
        if fsmn_has_start:
            self._fsmn_pending_start = True
        if fsmn_has_end:
            self._fsmn_pending_end = True

        # 3) flush 处理: is_final 时若 energy 还 active, 强制它 flush 出 ended
        if is_final and self.energy is not None:
            flush_state = self.energy.flush()
            if flush_state.ended:
                energy_ended = True
                energy_active = False

        # 4) 复合决策, 产出 0 或 1 个边界 (每 accept 调用最多一个 start/end)
        boundaries: list[VADBoundary] = []
        if not self._active:
            if self._check_start(fsmn_has_start, energy_active or energy_started):
                self._active = True
                self._start_at_ms = self._elapsed_ms
                self._fsmn_pending_start = False
                self._fsmn_pending_end = False
                boundaries.append(
                    VADBoundary(
                        started=True,
                        ended=False,
                        start_ms=0,  # 相对段内 0ms 起
                        end_ms=None,
                    )
                )
        else:
            # active: 检查结束
            #   对于 energy 分支: 只有当能量 VAD 报告 ended 边沿或当前已经非 active 才参与 AND
            energy_part_done = (
                (self.energy is None)
                or energy_ended
                or (not energy_active and not self.energy.active)
            )
            fsmn_part_done = fsmn_has_end or self._fsmn_pending_end
            should_end = False
            if self.end_mode == "fsmn_only":
                should_end = fsmn_part_done
            elif self.end_mode == "energy_only":
                should_end = energy_part_done and (self.energy is not None)
            else:  # and
                should_end = fsmn_part_done and energy_part_done

            if should_end:
                self._active = False
                end_ms = self._elapsed_ms + frame_ms
                boundaries.append(
                    VADBoundary(
                        started=False,
                        ended=True,
                        start_ms=None,
                        end_ms=end_ms - (self._start_at_ms or 0),
                    )
                )
                self._start_at_ms = None
                self._fsmn_pending_start = False
                self._fsmn_pending_end = False

        self._elapsed_ms += frame_ms
        return tuple(boundaries)


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
