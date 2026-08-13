"""降噪适配器测试 — DenoiseAdapter 基本功能验证。"""

import wave

import numpy as np
import pytest


def test_resample_identity():
    """_resample 对同采样率应返回相同数据。"""
    from ev.audio.utils import resample as _resample

    audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    result = _resample(audio, 16000, 16000)
    assert result.dtype == np.float32
    assert len(result) == len(audio)
    np.testing.assert_allclose(result, audio, rtol=0.01)


def test_resample_upsample():
    """_resample 上采样 8kHz→16kHz 长度翻倍。"""
    from ev.audio.utils import resample as _resample

    audio = np.sin(np.linspace(0, 2 * np.pi, 100, dtype=np.float32))
    result = _resample(audio, 8000, 16000)
    assert len(result) == pytest.approx(len(audio) * 2, abs=2)


def test_resample_downsample():
    """_resample 下采样 48kHz→16kHz 长度缩小。"""
    from ev.audio.utils import resample as _resample

    audio = np.random.randn(4800).astype(np.float32)
    result = _resample(audio, 48000, 16000)
    assert len(result) == pytest.approx(len(audio) / 3, abs=2)


def test_resample_downsample_suppresses_out_of_band_tone():
    """48kHz 的 12kHz 分量降到 16kHz 时应被抗混叠滤波抑制。"""
    from ev.audio.utils import resample

    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 12000 * t).astype(np.float32)

    result = resample(tone, sr, 16000)

    assert float(np.sqrt(np.mean(result.astype(np.float64) ** 2))) < 0.02


def test_resample_short_audio():
    """_resample 极短音频（<2 样本）不动。"""
    from ev.audio.utils import resample as _resample

    audio = np.array([0.5], dtype=np.float32)
    result = _resample(audio, 16000, 48000)
    assert len(result) == 1


def test_denoise_adapter_init():
    """DenoiseAdapter 构造时不应立即加载模型（lazy load）。"""
    from ev.audio.denoise import DenoiseAdapter

    adapter = DenoiseAdapter()
    assert adapter._pipeline is None
    assert not adapter._load_attempted


def test_denoise_adapter_available_without_modelscope(monkeypatch):
    """modelscope 未安装时 available 应返回 False。"""
    import sys

    # 模拟 modelscope 未安装
    monkeypatch.setitem(sys.modules, "modelscope", None)
    monkeypatch.setattr("importlib.import_module", lambda n: (_ for _ in ()).throw(ImportError))

    from ev.audio.denoise import DenoiseAdapter

    adapter = DenoiseAdapter()
    assert not adapter.available
    assert adapter._load_attempted


def test_denoise_enhance_fallback():
    """降噪不可用时 enhance() 应返回原始音频。"""
    from ev.audio.denoise import DenoiseAdapter

    adapter = DenoiseAdapter()
    # 不加载模型 → available = False
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    result = adapter.enhance(audio, 16000)
    np.testing.assert_array_almost_equal(result, audio)


def test_denoise_enhance_short_segment_fallback():
    """短于 100ms 的音频不应经过降噪，直接返回原始。"""
    from ev.audio.denoise import DenoiseAdapter

    adapter = DenoiseAdapter()
    # 1599 样本 @16kHz = 99.9ms < 100ms
    audio = np.random.randn(1599).astype(np.float32)
    result = adapter.enhance(audio, 16000)
    # 即使模型可用 (实际上未安装)，也应跳过
    np.testing.assert_array_almost_equal(result, audio)


class _FakeANS:
    """模拟 ModelScope ANS pipeline: 返回 int16 LE PCM bytes (native 48k)。"""

    def __init__(self, pcm_48k: np.ndarray) -> None:
        self._pcm = pcm_48k

    def __call__(self, path) -> dict:
        return {"output_pcm": (np.clip(self._pcm, -1, 1) * 32767).astype("<i2").tobytes()}


class _InspectingANS:
    def __init__(self, output) -> None:
        self.output = output
        self.input_peak = 0.0

    def __call__(self, path) -> dict:
        with wave.open(str(path), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
        self.input_peak = float(np.max(np.abs(pcm.astype(np.float32))) / 32768.0)
        return {"output_pcm": self.output}


def test_enhance_parses_output_pcm_bytes():
    """pipeline 以 bytes (int16 PCM) 返回时, enhance() 必须解析而非原样返回。

    回归: 此前只认 np.ndarray/文件路径, bytes 走 else 分支返回原音频,
    导致降噪形同虚设 (输出 = 输入)。"""
    from ev.audio.denoise import DenoiseAdapter
    from ev.audio.utils import resample

    sr = 16000
    t = np.arange(sr) / sr
    audio = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out_48k = resample(audio, sr, 48000)

    adapter = DenoiseAdapter()
    adapter._pipeline = _FakeANS(out_48k)
    adapter._load_attempted = True

    result = adapter.enhance(audio, sr)
    assert result.shape == audio.shape
    expected = resample(out_48k, 48000, sr)
    np.testing.assert_allclose(result, expected, atol=2e-3)


def test_enhance_preserves_input_level_before_model():
    """弱输入不能在送入模型前被强制放大到满幅。"""
    from ev.audio.denoise import DenoiseAdapter

    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (0.02 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    fake = _InspectingANS(np.zeros(sr * 3, dtype="<i2"))
    adapter = DenoiseAdapter()
    adapter._pipeline = fake
    adapter._load_attempted = True

    adapter.enhance(audio, sr)

    assert fake.input_peak < 0.03


def test_enhance_normalizes_int16_ndarray_output():
    """numpy int16 输出必须解释为 PCM，而不是超出 [-1, 1] 的浮点音频。"""
    from ev.audio.denoise import DenoiseAdapter

    sr = 16000
    audio = np.full(sr, 0.1, dtype=np.float32)
    fake = _InspectingANS(np.full(sr * 3, 3277, dtype="<i2"))
    adapter = DenoiseAdapter()
    adapter._pipeline = fake
    adapter._load_attempted = True

    result = adapter.enhance(audio, sr)

    assert np.max(np.abs(result)) < 0.2
    assert float(np.mean(result)) == pytest.approx(0.1, abs=0.01)


def test_enhance_limits_abnormal_output_gain():
    """模型不能把近静音或远场片段整体放大超过 6dB。"""
    from ev.audio.denoise import DenoiseAdapter

    sr = 16000
    audio = np.full(sr, 0.005, dtype=np.float32)
    fake = _InspectingANS(np.full(sr * 3, 0.8, dtype=np.float32))
    adapter = DenoiseAdapter()
    adapter._pipeline = fake
    adapter._load_attempted = True

    result = adapter.enhance(audio, sr)

    input_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    output_rms = float(np.sqrt(np.mean(result.astype(np.float64) ** 2)))
    assert output_rms <= input_rms * 2.0 + 1e-6


def test_enhance_falls_back_on_non_finite_output():
    from ev.audio.denoise import DenoiseAdapter

    sr = 16000
    audio = np.linspace(-0.1, 0.1, sr, dtype=np.float32)
    fake = _InspectingANS(np.full(sr * 3, np.nan, dtype=np.float32))
    adapter = DenoiseAdapter()
    adapter._pipeline = fake
    adapter._load_attempted = True

    result = adapter.enhance(audio, sr)

    np.testing.assert_array_equal(result, audio)
