"""人声确认测试 — _check_speech_segment 辅助函数 + FSMN 人声确认逻辑。"""

import numpy as np
import pytest


def test_check_speech_segment_none_vad():
    """vad_model 为 None 时应返回 False。"""
    from ev.pipeline.runtime import _check_speech_segment

    audio = np.zeros(16000, dtype=np.float32)
    assert not _check_speech_segment(None, audio)


def test_check_speech_segment_empty_audio():
    """空音频应返回 False。"""
    from ev.pipeline.runtime import _check_speech_segment

    class MockVAD:
        def __init__(self):
            self.model = self

        def generate(self, input, chunk_size=200):
            return []

    vad = MockVAD()
    audio = np.array([], dtype=np.float32)
    assert not _check_speech_segment(vad, audio)


def test_check_speech_segment_all_speech():
    """100% 语音段应返回 True。"""
    from ev.pipeline.runtime import _check_speech_segment

    audio = np.zeros(16000, dtype=np.float32)  # 1 second

    class MockVAD:
        def __init__(self):
            self.model = self

        def generate(self, input, chunk_size=200):
            # 返回 1000ms 的语音段（全部）
            return [{"value": [[0, 1000]]}]

    vad = MockVAD()
    assert _check_speech_segment(vad, audio, speech_ratio_threshold=0.05)


def test_check_speech_segment_no_speech():
    """无语音段应返回 False（低于阈值）。"""
    from ev.pipeline.runtime import _check_speech_segment

    audio = np.zeros(32000, dtype=np.float32)  # 2 seconds

    class MockVAD:
        def __init__(self):
            self.model = self

        def generate(self, input, chunk_size=200):
            # 返回 50ms 语音 = 2.5% < 5% 阈值
            return [{"value": [[100, 150]]}]

    vad = MockVAD()
    assert not _check_speech_segment(vad, audio, speech_ratio_threshold=0.05)


def test_check_speech_segment_partial_speech():
    """50% 语音段应返回 True。"""
    from ev.pipeline.runtime import _check_speech_segment

    audio = np.zeros(16000, dtype=np.float32)  # 1 second

    class MockVAD:
        def __init__(self):
            self.model = self

        def generate(self, input, chunk_size=200):
            return [{"value": [[0, 500]]}]  # 500ms speech out of 1000ms

    vad = MockVAD()
    assert _check_speech_segment(vad, audio, speech_ratio_threshold=0.05)


def test_check_speech_segment_empty_result():
    """VAD 返回空结果应返回 False。"""
    from ev.pipeline.runtime import _check_speech_segment

    audio = np.zeros(16000, dtype=np.float32)

    class MockVAD:
        def __init__(self):
            self.model = self

        def generate(self, input, chunk_size=200):
            return []  # 空列表

    vad = MockVAD()
    assert not _check_speech_segment(vad, audio)


def test_check_speech_segment_exception():
    """VAD 抛出异常应返回 False（不崩溃）。"""
    from ev.pipeline.runtime import _check_speech_segment

    audio = np.zeros(16000, dtype=np.float32)

    class MockVAD:
        def __init__(self):
            self.model = self

        def generate(self, input, chunk_size=200):
            raise RuntimeError("mock VAD failure")

    vad = MockVAD()
    # 不应抛出异常
    assert not _check_speech_segment(vad, audio)


def test_check_speech_segment_str_result():
    """VAD 返回非列表结果应安全处理。"""
    from ev.pipeline.runtime import _check_speech_segment

    audio = np.zeros(16000, dtype=np.float32)

    class MockVAD:
        def __init__(self):
            self.model = self

        def generate(self, input, chunk_size=200):
            return "not a list"

    vad = MockVAD()
    assert not _check_speech_segment(vad, audio)


def test_check_speech_segment_with_threshold():
    """自定义阈值应正确工作。"""
    from ev.pipeline.runtime import _check_speech_segment

    audio = np.zeros(16000, dtype=np.float32)  # 1 second

    class MockVAD:
        def __init__(self):
            self.model = self

        def generate(self, input, chunk_size=200):
            # 300ms speech = 30% → passes 5% but fails 50%
            return [{"value": [[0, 300]]}]

    vad = MockVAD()
    # 默认 5% 阈值 → 通过
    assert _check_speech_segment(vad, audio, speech_ratio_threshold=0.05)
    # 50% 阈值 → 不通过
    assert not _check_speech_segment(vad, audio, speech_ratio_threshold=0.50)
