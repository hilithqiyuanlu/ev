"""YAMNet 环境感知 — 独立于语音路径的实时环境声音分类。

设计原则:
- 环境路径与语音路径完全解耦，通过定时器独立轮询（不依赖 FSMN 触发）
- 维护 10s ring buffer，每 2s 取最新 5s 做 YAMNet 推理
- 时序聚合状态机：帧级分类 → 滑动窗口 → 持续状态
- 分类结果写入环境事件日志（不入 WAV，不入 SQLite）
"""

from __future__ import annotations

import collections
import csv
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

LOGGER = logging.getLogger(__name__)

# ── YAMNet AudioSet 类映射 (521 → ~15 有意义类别) ──────────────────────
# AudioSet 标签格式: "/m/07yv9" 等 Google Knowledge Graph ID
# 以下映射基于 AudioSet 本体论的实际 ID

_YAMNET_LABEL_TO_CATEGORY: dict[str, str] = {
    # 键盘/鼠标
    "Typing": "typing",
    "Computer keyboard": "typing",
    "Clicking": "typing",
    "Mouse": "typing",
    # 背景人声 (非当前用户说话)
    "Speech": "background_speech",
    "Conversation": "background_speech",
    "Babbling": "background_speech",
    "Child speech, kid speaking": "background_speech",
    "Narration, monologue": "background_speech",
    "Female speech, woman speaking": "background_speech",
    "Male speech, man speaking": "background_speech",
    # 音乐
    "Music": "music",
    "Singing": "music",
    "Musical instrument": "music",
    "Pop music": "music",
    "Rock music": "music",
    # 持续背景噪声
    "Traffic noise, roadway noise": "background_noise",
    "Vehicle": "background_noise",
    "Car": "background_noise",
    "Bus": "background_noise",
    "Truck": "background_noise",
    "Air conditioning": "background_noise",
    "Fan": "background_noise",
    "Water": "background_noise",
    "Rain": "background_noise",
    "Wind": "background_noise",
    "Engine": "background_noise",
    "Idling": "background_noise",
    # 警报/提醒
    "Alarm": "alert",
    "Doorbell": "alert",
    "Telephone bell ringing": "alert",
    "Smoke detector, smoke alarm": "alert",
    "Fire alarm": "alert",
    "Siren": "alert",
    "Ringtone": "alert",
    # 动物
    "Dog": "animal",
    "Bark": "animal",
    "Cat": "animal",
    "Meow": "animal",
    "Bird": "animal",
    "Bird vocalization, bird call, bird song": "animal",
    # 撞击/动静
    "Knock": "impact",
    "Crash": "impact",
    "Bang": "impact",
    "Door slam": "impact",
    "Door": "impact",
    "Thump, thud": "impact",
    # 家电
    "Vacuum cleaner": "appliance",
    "Washing machine": "appliance",
    "Microwave oven": "appliance",
    "Blender": "appliance",
    "Dishwasher": "appliance",
    # 静音/极低能量
    "Silence": "silence",
    # 其他可能有用
    "Laughter": "human_sound",
    "Cough": "human_sound",
    "Sneeze": "human_sound",
    "Breathing": "human_sound",
    "Clapping": "human_sound",
}

# 需要持续检测的最小帧数（约 3s 对应 ~6 帧 at 0.48s/帧）
_MIN_FRAMES_FOR_ACTIVE = 6
# 需要结束的最小不活跃帧数（约 2s 对应 ~4 帧）
_MIN_FRAMES_FOR_INACTIVE = 4
# 最小置信度阈值
_MIN_CONFIDENCE = 0.25


@dataclass
class EnvEvent:
    """环境事件 — 一次状态变化的记录。"""

    timestamp: float  # Unix 时间戳
    category: str  # 类别: "typing", "background_noise", "alert", ...
    confidence: float  # 平均置信度 0.0-1.0
    duration_sec: float | None  # 持续时间，None 表示点事件（瞬时）


class TemporalAggregator:
    """时序聚合状态机。

    将 YAMNet 逐帧分类结果（每 0.48s 一帧）聚合为持续环境状态。
    仅当状态变化时产出 EnvEvent。
    """

    def __init__(
        self,
        min_confidence: float = _MIN_CONFIDENCE,
        min_frames_active: int = _MIN_FRAMES_FOR_ACTIVE,
        min_frames_inactive: int = _MIN_FRAMES_FOR_INACTIVE,
    ):
        self.min_confidence = min_confidence
        self.min_frames_active = min_frames_active
        self.min_frames_inactive = min_frames_inactive

        self._current_category: str | None = None
        self._active_frames: int = 0
        self._inactive_frames: int = 0
        self._category_start_time: float | None = None
        self._confidence_sum: float = 0.0
        self._active_frame_count: int = 0

    def update(self, frame_categories: list[str], frame_confidences: list[float]) -> list[EnvEvent]:
        """处理一帧 YAMNet 分类结果。

        Args:
            frame_categories: 每帧的类别标签列表（已做类映射）。
            frame_confidences: 每帧的置信度列表。

        Returns:
            本次产出的环境事件列表（通常为空，仅在状态变化时产出一条）。
        """
        events: list[EnvEvent] = []

        # 取所有帧中出现最多的类别作为当前帧的主导类别
        if not frame_categories:
            return events

        # 简单多数投票
        cat_counts: dict[str, int] = {}
        cat_confs: dict[str, list[float]] = {}
        for cat, conf in zip(frame_categories, frame_confidences):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            cat_confs.setdefault(cat, []).append(conf)

        dominant = max(cat_counts, key=lambda k: cat_counts[k])
        conf = float(np.mean(cat_confs[dominant]))

        if conf < self.min_confidence or dominant == "silence":
            dominant = "silence"

        now = time.time()

        if dominant == self._current_category:
            self._active_frames += 1
            self._inactive_frames = 0
            self._confidence_sum += conf
            self._active_frame_count += 1
        elif self._current_category is not None:
            self._inactive_frames += 1
            if self._inactive_frames >= self.min_frames_inactive:
                # 状态结束
                avg_conf = (
                    self._confidence_sum / max(self._active_frame_count, 1)
                    if self._active_frame_count > 0
                    else conf
                )
                duration = (
                    now - self._category_start_time
                    if self._category_start_time
                    else None
                )
                events.append(
                    EnvEvent(
                        timestamp=now,
                        category=self._current_category,
                        confidence=round(avg_conf, 3),
                        duration_sec=round(duration, 1) if duration else None,
                    )
                )
                # 重置，准备新状态
                self._current_category = None
                self._active_frames = 0
                self._inactive_frames = 0
                self._confidence_sum = 0.0
                self._active_frame_count = 0
                self._category_start_time = None

        # 检查是否有新状态开始
        if self._current_category is None and dominant != "silence":
            self._active_frames += 1
            self._confidence_sum += conf
            self._active_frame_count += 1
            if self._active_frames >= self.min_frames_active:
                self._current_category = dominant
                self._category_start_time = now
                # 减去 min_frames_active 的延迟，让时间更准
                frame_duration_sec = 0.48 * self.min_frames_active
                self._category_start_time = now - frame_duration_sec

        return events

    def flush(self) -> list[EnvEvent]:
        """强制产出当前活跃状态的事件（用于停止时）。"""
        if self._current_category and self._active_frames > 0:
            avg_conf = (
                self._confidence_sum / max(self._active_frame_count, 1)
                if self._active_frame_count > 0
                else 0.0
            )
            duration = (
                time.time() - self._category_start_time
                if self._category_start_time
                else None
            )
            return [
                EnvEvent(
                    timestamp=time.time(),
                    category=self._current_category,
                    confidence=round(avg_conf, 3),
                    duration_sec=round(duration, 1) if duration else None,
                )
            ]
        return []

    def reset(self) -> None:
        self._current_category = None
        self._active_frames = 0
        self._inactive_frames = 0
        self._category_start_time = None
        self._confidence_sum = 0.0
        self._active_frame_count = 0


def load_yamnet_labels(label_path: str | Path) -> list[str]:
    """加载 YAMNet 类标签 CSV 文件。

    CSV 格式: index, mid, display_name
    返回 display_name 列表，按 index 排序。
    """
    labels: list[tuple[int, str]] = []
    with open(label_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row.get("index", 0))
            name = row.get("display_name", "")
            if name:
                labels.append((idx, name))
    labels.sort(key=lambda x: x[0])
    return [name for _, name in labels]


def build_category_map(labels: list[str]) -> dict[int, str]:
    """构建 AudioSet index → 自定义类别 的映射表。

    Args:
        labels: YAMNet 类标签列表（按 index 排序）。

    Returns:
        dict mapping AudioSet index to category string.
    """
    mapping: dict[int, str] = {}
    for idx, label in enumerate(labels):
        cat = _YAMNET_LABEL_TO_CATEGORY.get(label, "other")
        if cat != "other":
            mapping[idx] = cat
    return mapping


class EnvironmentMonitor:
    """YAMNet 驱动的环境声音监测器。

    独立于语音处理的旁路路径：
    - 维护常驻 ring buffer（由 audio capture callback 填充）
    - 定时器每 N 秒取最新 M 秒窗口做 YAMNet 推理
    - 通过状态机将帧级分类聚合为环境事件
    - 事件通过 emit_callback 输出（写入日志 + 推送到前端）
    """

    def __init__(
        self,
        model_path: str,
        label_path: str,
        sample_rate: int = 16000,
        ring_buffer_sec: float = 10.0,
        poll_interval_sec: float = 2.0,
        window_sec: float = 5.0,
    ):
        self.model_path = model_path
        self.label_path = label_path
        self.sample_rate = sample_rate
        self.ring_buffer_sec = ring_buffer_sec
        self.poll_interval_sec = poll_interval_sec
        self.window_sec = window_sec

        # YAMNet 固定输入: 0.975s @ 16kHz = 15600 samples
        self._yamnet_input_samples = 15600

        self._buffer: collections.deque[float] = collections.deque(
            maxlen=int(sample_rate * ring_buffer_sec)
        )
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._running = False
        self._aggregator = TemporalAggregator()
        self._emit_callback: Callable[[EnvEvent], None] | None = None
        self._category_map: dict[int, str] = {}
        self._interpreter: object | None = None  # tflite.Interpreter

    def _load_model(self) -> None:
        """加载 YAMNet TFLite 模型和标签。

        macOS: tflite-runtime 无预编译 wheel，回退到 tensorflow。
        """
        # 尝试加载 TFLite interpreter
        tflite = None
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore[import-untyped]
        except ImportError:
            try:
                import tensorflow.lite as tflite  # type: ignore[import-untyped]
                LOGGER.info("using tensorflow.lite (macOS fallback)")
            except ImportError:
                LOGGER.warning(
                    "tflite-runtime/tensorflow 均未安装，环境感知不可用。"
                    "安装: pip install tensorflow"
                )
                self._interpreter = None
                return

        labels = load_yamnet_labels(self.label_path)
        self._category_map = build_category_map(labels)
        LOGGER.info(
            "YAMNet labels loaded: %d total, %d mapped to categories",
            len(labels),
            len(self._category_map),
        )

        self._interpreter_module = tflite
        self._interpreter = tflite.Interpreter(model_path=self.model_path)
        self._interpreter.allocate_tensors()
        LOGGER.info("YAMNet TFLite model loaded from %s", self.model_path)

    # ── 公开接口 ──────────────────────────────────────────────────────

    def feed(self, frame: np.ndarray) -> None:
        """每帧音频数据写入 ring buffer。

        由 audio capture callback 调用（应尽量轻量）。
        """
        arr = np.asarray(frame, dtype=np.float32).reshape(-1)
        with self._lock:
            self._buffer.extend(arr.tolist())

    def start(self, emit_callback: Callable[[EnvEvent], None]) -> None:
        """启动环境监测。

        Args:
            emit_callback: 当状态变化时调用，传入产生的 EnvEvent。
        """
        self._emit_callback = emit_callback
        self._load_model()
        self._running = True
        self._schedule_next()
        LOGGER.info(
            "EnvironmentMonitor started (poll=%.1fs, window=%.1fs, buffer=%.1fs)",
            self.poll_interval_sec,
            self.window_sec,
            self.ring_buffer_sec,
        )

    def stop(self) -> None:
        """停止环境监测。"""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        # 刷出当前状态
        for event in self._aggregator.flush():
            if self._emit_callback:
                self._emit_callback(event)
        self._aggregator.reset()
        LOGGER.info("EnvironmentMonitor stopped")

    def get_current_state(self) -> dict:
        """返回当前环境状态快照（用于前端查询）。"""
        return {
            "category": self._aggregator._current_category or "silence",
            "buffer_duration_sec": (
                len(self._buffer) / self.sample_rate if self._buffer else 0
            ),
            "running": self._running,
        }

    # ── 内部 ──────────────────────────────────────────────────────────

    def _schedule_next(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self.poll_interval_sec, self._poll)
        self._timer.daemon = True
        self._timer.start()

    def _poll(self) -> None:
        """定时轮询：从 ring buffer 取窗口 → YAMNet → 状态机。"""
        if not self._running:
            return

        try:
            # 从 ring buffer 取最新 window_sec 的音频
            window_samples = int(self.window_sec * self.sample_rate)
            with self._lock:
                if len(self._buffer) < window_samples:
                    self._schedule_next()
                    return
                buf_list = list(self._buffer)
                window_data = np.array(
                    buf_list[-window_samples:], dtype=np.float32
                )
        except Exception:
            LOGGER.debug("EnvironmentMonitor poll error", exc_info=True)
            self._schedule_next()
            return

        # 极低能量时跳过推理（绝对静音）
        rms = float(np.sqrt(np.mean(np.square(window_data))))
        if rms < 1e-6:
            self._schedule_next()
            return

        try:
            frame_categories, frame_confs = self._classify_window(window_data)
            events = self._aggregator.update(frame_categories, frame_confs)
            for event in events:
                if self._emit_callback:
                    self._emit_callback(event)
        except Exception:
            LOGGER.debug("EnvironmentMonitor classification error", exc_info=True)

        self._schedule_next()

    def _classify_window(self, audio: np.ndarray) -> tuple[list[str], list[float]]:
        """对一段音频窗口做 YAMNet 分类。

        Args:
            audio: shape (window_samples,) float32.

        Returns:
            (categories, confidences) — 每帧的类别标签和置信度。
        """
        if self._interpreter is None:
            return [], []

        tflite = self._interpreter

        input_details = tflite.get_input_details()
        output_details = tflite.get_output_details()

        # YAMNet 输入: (1, 15600) float32 = 0.975s @ 16kHz
        chunk_size = self._yamnet_input_samples
        stride = chunk_size // 2  # 50% overlap

        categories: list[str] = []
        confidences: list[float] = []

        for start in range(0, len(audio) - chunk_size + 1, stride):
            chunk = audio[start : start + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            chunk_input = chunk.reshape(1, chunk_size).astype(np.float32)
            tflite.set_tensor(input_details[0]["index"], chunk_input)
            tflite.invoke()

            # scores: (1, N_frames, 521)
            scores = tflite.get_tensor(output_details[0]["index"])[0]  # (N_frames, 521)

            for frame_scores in scores:
                # frame_scores: (521,)
                best_idx = int(np.argmax(frame_scores))
                best_conf = float(np.max(frame_scores))
                cat = self._category_map.get(best_idx, "other")
                categories.append(cat)
                confidences.append(best_conf)

        return categories, confidences
