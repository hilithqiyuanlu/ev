"""FunASRNanoAdapter 热词透传测试 — 不加载真实模型，mock generate。"""

import numpy as np


class _FakeNanoModel:
    """模拟 funasr AutoModel: 记录 generate 收到的 kwargs 并返回固定文本。"""

    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def generate(self, **kwargs):
        self.last_kwargs = dict(kwargs)
        return [{"text": "测试转写"}]


def _make_adapter():
    from ev.asr.adapters import FunASRNanoAdapter

    # 跳过 __init__（会尝试加载真实模型），直接注入 mock model
    adapter = FunASRNanoAdapter.__new__(FunASRNanoAdapter)
    adapter.model = _FakeNanoModel()
    return adapter


def test_transcribe_passes_hotwords():
    """传 hotwords 时，generate 必须收到 hotwords=list[str]（模型级引导）。"""
    adapter = _make_adapter()
    audio = np.zeros(1600, dtype=np.float32)

    result = adapter.transcribe(audio, 16000, hotwords=["日照", "开放时间"])

    assert result.text == "测试转写"
    kw = adapter.model.last_kwargs
    assert kw["hotwords"] == ["日照", "开放时间"]
    assert kw["disable_pbar"] is True


def test_transcribe_omits_hotwords_when_empty():
    """不传 hotwords 时，generate 不应带 hotwords（走默认 prompt）。"""
    adapter = _make_adapter()
    audio = np.zeros(1600, dtype=np.float32)

    adapter.transcribe(audio, 16000)

    assert "hotwords" not in adapter.model.last_kwargs


def test_transcribe_hotwords_none_and_empty_list():
    """None 与空列表都不得注入 hotwords。"""
    adapter = _make_adapter()
    audio = np.zeros(1600, dtype=np.float32)

    adapter.transcribe(audio, 16000, hotwords=None)
    assert "hotwords" not in adapter.model.last_kwargs

    adapter.transcribe(audio, 16000, hotwords=[])
    assert "hotwords" not in adapter.model.last_kwargs


def test_streaming_adapter_keeps_segment_cache_and_resets_on_final():
    from ev.asr.adapters import StreamingASRAdapter

    class Model:
        def __init__(self):
            self.calls = []
            self.results = [[{"text": "你"}], [{"text": "好"}]]

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            kwargs["cache"]["used"] = True
            return self.results.pop(0)

    adapter = StreamingASRAdapter.__new__(StreamingASRAdapter)
    adapter.model = Model()
    adapter.cache = {}
    adapter._text = ""
    assert adapter.transcribe_chunk(np.zeros(3200, dtype=np.float32)).text == "你"
    assert adapter.transcribe_chunk(np.zeros(0, dtype=np.float32), is_final=True).text == "你好"
    assert adapter.cache == {}
    assert adapter._text == ""
    assert adapter.model.calls[0]["chunk_size"] == [0, 10, 5]
    assert adapter.model.calls[1]["is_final"] is True
