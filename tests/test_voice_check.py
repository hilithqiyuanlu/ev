"""voice_check 声学特征检测单元测试。"""

import numpy as np
import pytest
from ev.audio.voice_check import (
    classify_voice,
    compute_zcr,
    compute_spectral_centroid,
    compute_spectral_flatness,
    compute_rms_envelope_variance,
)


SR = 16000


def make_tone(freq: float, duration_sec: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(SR * duration_sec)) / SR
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def make_noise(duration_sec: float, amplitude: float = 0.01) -> np.ndarray:
    return (np.random.default_rng(42).normal(0, amplitude, int(SR * duration_sec))).astype(np.float32)


class TestZeroCrossingRate:
    def test_zcr_sine_200hz(self):
        """200Hz 正弦波 ZCR ≈ 400 crossings/s ÷ 16000 samples/s ≈ 0.025."""
        audio = make_tone(200, 2.0)
        zcr = compute_zcr(audio)
        assert 0.01 < zcr < 0.06, f"期望 0.025 附近, 实际 {zcr:.4f}"

    def test_zcr_sine_4000hz(self):
        """4kHz 正弦波 ZCR 应 > 0.4."""
        audio = make_tone(4000, 1.0)
        zcr = compute_zcr(audio)
        assert zcr > 0.35, f"期望 > 0.35, 实际 {zcr:.4f}"

    def test_zcr_white_noise(self):
        """白噪声 ZCR ≈ 0.5 (每样本随机穿越)."""
        audio = make_noise(2.0, amplitude=0.1)
        zcr = compute_zcr(audio)
        assert 0.35 < zcr < 0.65, f"期望 0.5 附近, 实际 {zcr:.4f}"

    def test_zcr_silence(self):
        """静音 ZCR → 0."""
        audio = np.zeros(SR, dtype=np.float32)
        assert compute_zcr(audio) == 0.0

    def test_zcr_empty(self):
        assert compute_zcr(np.array([], dtype=np.float32)) == 0.0


class TestSpectralCentroid:
    def test_centroid_200hz_tone(self):
        """200Hz 纯音 → centroid ~200Hz."""
        audio = make_tone(200, 1.0)
        c = compute_spectral_centroid(audio, SR)
        assert 100 < c < 500, f"期望 ~200Hz, 实际 {c:.0f}Hz"

    def test_centroid_white_noise(self):
        """白噪声 → centroid ~SR/3 = 5333Hz."""
        audio = make_noise(1.0)
        c = compute_spectral_centroid(audio, SR)
        assert c > 3000, f"白噪声 centroid 应 > 3000Hz, 实际 {c:.0f}Hz"


class TestSpectralFlatness:
    def test_flatness_pure_tone(self):
        """纯音 → flatness ≈ 0 (高度谐波结构)."""
        audio = make_tone(200, 1.0)
        f = compute_spectral_flatness(audio, SR)
        assert f < 0.1, f"纯音 flatness 应极低, 实际 {f:.3f}"

    def test_flatness_white_noise(self):
        """白噪声 → flatness > 0.5 (频谱平坦)."""
        audio = make_noise(1.0)
        f = compute_spectral_flatness(audio, SR)
        assert f > 0.3, f"白噪声 flatness 应较高, 实际 {f:.3f}"


class TestRMSEnvelopeVariance:
    def test_envelope_var_steady_tone(self):
        """稳态正弦波 → 归一化方差极低."""
        audio = make_tone(200, 2.0)
        v = compute_rms_envelope_variance(audio, SR)
        assert v < 0.1, f"稳态音的包络方差应极低, 实际 {v:.4f}"

    def test_envelope_var_modulated(self):
        """调制幅度 → 方差较大."""
        t = np.arange(int(SR * 2.0)) / SR
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2Hz 调制
        audio = (np.sin(2 * np.pi * 200 * t) * envelope * 0.1).astype(np.float32)
        v = compute_rms_envelope_variance(audio, SR)
        assert v > 0.05, f"调制信号包络方差应较大, 实际 {v:.4f}"


class TestClassifyVoice:
    def test_white_noise_rejected(self):
        """2s 白噪声 → 非人声."""
        audio = make_noise(2.0, amplitude=0.05)
        result = classify_voice(audio, SR)
        assert result["is_voice"] is False
        assert len(result["reasons"]) > 0

    def test_too_short_passes(self):
        """< 0.1s → 不判断, 放行."""
        audio = make_noise(0.05)
        result = classify_voice(audio, SR)
        assert result["is_voice"] is True
        assert result["reasons"] == []

    def test_empty_passes(self):
        result = classify_voice(np.array([], dtype=np.float32), SR)
        assert result["is_voice"] is True

    def test_none_passes(self):
        result = classify_voice(None, SR)
        assert result["is_voice"] is True
