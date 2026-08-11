"""环境感知测试 — YAMNet 类映射、时序状态机逻辑（不依赖模型）。"""

import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestTemporalAggregator:
    """时序聚合状态机测试。"""

    def test_initial_state_is_none(self):
        """初始化时当前类别为 None。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator()
        assert agg._current_category is None
        assert agg._active_frames == 0

    def test_single_frame_no_trigger(self):
        """单帧不应触发状态变化（低于 min_frames_active）。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator(min_frames_active=3, min_frames_inactive=2)
        events = agg.update(["typing"], [0.8])
        assert events == []
        assert agg._current_category is None

    def test_triggers_after_min_frames(self):
        """连续同类帧达到阈值后应进入活动状态。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator(min_frames_active=3, min_frames_inactive=2)
        # 前两帧不应触发
        agg.update(["typing"], [0.7])
        agg.update(["typing"], [0.8])
        assert agg._current_category is None
        # 第三帧触发
        events = agg.update(["typing"], [0.9])
        assert len(events) == 0  # 进入状态，不产事件（事件在结束时产出）
        assert agg._current_category == "typing"

    def test_emits_event_on_state_end(self):
        """状态结束时应产出一个 EnvEvent。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator(min_frames_active=2, min_frames_inactive=2)
        # 进入 typing 状态
        agg.update(["typing"], [0.8])
        agg.update(["typing"], [0.8])
        assert agg._current_category == "typing"
        # 切换 → inactive 计数
        agg.update(["silence"], [0.1])
        events = agg.update(["silence"], [0.1])
        assert len(events) == 1
        assert events[0].category == "typing"
        assert events[0].duration_sec is not None

    def test_low_confidence_ignored(self):
        """低置信度帧应被忽略。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator(
            min_confidence=0.5, min_frames_active=3, min_frames_inactive=2
        )
        for _ in range(5):
            events = agg.update(["typing"], [0.3])
        assert events == []
        assert agg._current_category is None

    def test_silence_treated_as_inactive(self):
        """silence 类别应视为无活动状态。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator(min_frames_active=2, min_frames_inactive=1)
        # 进入音乐状态
        agg.update(["music"], [0.9])
        agg.update(["music"], [0.9])
        assert agg._current_category == "music"
        # silence → 立即结束
        events = agg.update(["silence"], [0.1])
        assert len(events) == 1
        assert events[0].category == "music"

    def test_flush_emits_current_state(self):
        """flush() 应强制产出现有状态。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator(min_frames_active=2, min_frames_inactive=3)
        agg.update(["music"], [0.9])
        agg.update(["music"], [0.9])
        assert agg._current_category == "music"
        events = agg.flush()
        assert len(events) == 1
        assert events[0].category == "music"

    def test_reset_clears_state(self):
        """reset() 应清除所有状态。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator(min_frames_active=1, min_frames_inactive=2)
        agg.update(["typing"], [0.9])
        assert agg._current_category == "typing"
        agg.reset()
        assert agg._current_category is None
        assert agg._active_frames == 0

    def test_majority_vote_selects_dominant_category(self):
        """多数投票应选择主导类别。"""
        from ev.audio.environment import TemporalAggregator

        agg = TemporalAggregator(min_frames_active=2, min_frames_inactive=2)
        # typing 3帧, music 1帧 → dominant = typing
        agg.update(["typing", "typing", "typing", "music"], [0.8, 0.8, 0.8, 0.9])
        agg.update(["typing", "typing", "typing", "music"], [0.7, 0.7, 0.7, 0.9])
        assert agg._current_category == "typing"


class TestCategoryMapping:
    """类映射表测试。"""

    def test_known_labels_mapped(self):
        """已知 AudioSet 标签应映射到有意义类别。"""
        from ev.audio.environment import build_category_map

        labels = ["Typing", "Music", "Silence", "UnknownThing", "Bark"]
        mapping = build_category_map(labels)
        assert mapping[0] == "typing"  # Typing
        assert mapping[1] == "music"  # Music
        assert mapping[2] == "silence"  # Silence
        # UnknownThing → not in mapping
        assert 3 not in mapping
        assert mapping[4] == "animal"  # Bark

    def test_unknown_labels_absent(self):
        """不在映射表中的标签应不出现在结果中。"""
        from ev.audio.environment import build_category_map

        labels = ["SomeRandomSound", "AnotherUnknown"]
        mapping = build_category_map(labels)
        assert mapping == {}


class TestEnvironmentLog:
    """环境事件日志测试。"""

    def test_append_and_query(self, tmp_path):
        """追加事件后应能查询。"""
        from ev.audio.environment import EnvEvent
        from ev.store.environment import EnvironmentLog

        log = EnvironmentLog(tmp_path)
        event = EnvEvent(
            timestamp=1754971385.0,
            category="typing",
            confidence=0.72,
            duration_sec=10.0,
        )
        log.append(event)

        results = log.query(start_time=1754970000.0, end_time=1754980000.0)
        assert len(results) == 1
        assert results[0]["category"] == "typing"
        assert results[0]["confidence"] == 0.72

    def test_query_filters_by_time(self, tmp_path):
        """时间范围外的记录应被过滤。"""
        from ev.audio.environment import EnvEvent
        from ev.store.environment import EnvironmentLog

        log = EnvironmentLog(tmp_path)
        log.append(EnvEvent(100, "typing", 0.8, 5.0))
        log.append(EnvEvent(200, "music", 0.9, 3.0))
        log.append(EnvEvent(300, "alert", 0.7, 1.0))

        results = log.query(start_time=150, end_time=250)
        assert len(results) == 1
        assert results[0]["category"] == "music"

    def test_query_summary(self, tmp_path):
        """摘要查询应返回主导类别和统计。"""
        from ev.audio.environment import EnvEvent
        from ev.store.environment import EnvironmentLog

        log = EnvironmentLog(tmp_path)
        log.append(EnvEvent(100, "typing", 0.8, 5.0))
        log.append(EnvEvent(200, "typing", 0.7, 3.0))
        log.append(EnvEvent(300, "music", 0.9, 2.0))

        summary = log.query_summary(start_time=0, end_time=500)
        assert summary["dominant_category"] == "typing"
        assert summary["event_count"] == 3
        assert 0.6 < summary["average_confidence"] < 1.0


class TestEnvironmentMonitor:
    """EnvironmentMonitor 基本结构测试（不依赖 YAMNet 模型）。"""

    def test_init_and_defaults(self, tmp_path):
        """初始化应设置默认参数。"""
        from ev.audio.environment import EnvironmentMonitor

        monitor = EnvironmentMonitor(
            model_path=str(tmp_path / "yamnet.tflite"),
            label_path=str(tmp_path / "labels.csv"),
        )
        assert monitor.sample_rate == 16000
        assert monitor.ring_buffer_sec == 10.0
        assert monitor.poll_interval_sec == 2.0
        assert monitor.window_sec == 5.0
        assert monitor._interpreter is None

    def test_feed_adds_to_buffer(self):
        """feed() 应将音频帧加入 ring buffer。"""
        from ev.audio.environment import EnvironmentMonitor

        monitor = EnvironmentMonitor(
            model_path="/nonexistent/model.tflite",
            label_path="/nonexistent/labels.csv",
        )
        frame = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        for _ in range(10):
            monitor.feed(frame)
        assert len(monitor._buffer) > 0

    def test_get_current_state(self):
        """get_current_state() 应返回运行状态。"""
        from ev.audio.environment import EnvironmentMonitor

        monitor = EnvironmentMonitor(
            model_path="/nonexistent/model.tflite",
            label_path="/nonexistent/labels.csv",
        )
        state = monitor.get_current_state()
        assert "category" in state
        assert "buffer_duration_sec" in state
        assert "running" in state
        assert state["running"] is False
