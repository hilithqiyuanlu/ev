import asyncio
import time

import numpy as np
import pytest

from ev.audio.capture import AudioCapture
from ev.config import AudioSettings


def test_frame_chunking():
    """不同长度的输入块应被归一为定长帧。"""
    cap = AudioCapture(AudioSettings(sample_rate=16000), frame_ms=30)
    assert cap.frame_samples == 480
    cap._queue.put(np.full(480, 0.5, dtype=np.float32))
    cap._queue.put(np.full(500, 0.5, dtype=np.float32))  # 跨块拼出第二帧

    async def collect():
        out = []
        async for f in cap.frames():
            out.append(f)
            if len(out) == 2:
                break
        return out

    frames = asyncio.run(collect())
    assert all(len(f) == 480 for f in frames)
    assert np.allclose(frames[1][0], 0.5)


# ============================================================
# Preprocess 测试 (DC/Preemphasis/AGC/NoiseGate/Pipeline)
# ============================================================

from ev.audio.preprocess import (
    AGC,
    AudioPreprocessor,
    DCRemover,
    NoiseGate,
    Preemphasis,
    PreprocessParams,
)


SR = 16000
FRAME_MS = 30
N = SR * FRAME_MS // 1000  # 480 samples per frame


def _sine(freq_hz: float, amp: float, n: int = N, sr: int = SR, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2.0 * np.pi * freq_hz * t + phase)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


class TestDCRemover:
    def test_constant_offset_removed(self):
        dc = DCRemover(sample_rate=SR)
        frame = np.full(N, 0.5 + 0.01, dtype=np.float32)  # DC bias = 0.5, signal = 0.01
        # 跑 10 帧让滤波器收敛
        out = frame
        for _ in range(10):
            out = dc.process_frame(out)
        out_mean = float(np.mean(out))
        # DC 分量应显著减小
        assert abs(out_mean) < 0.05

    def test_silent(self):
        dc = DCRemover(sample_rate=SR)
        silence = np.zeros(N, dtype=np.float32)
        out = dc.process_frame(silence)
        assert out.shape == silence.shape


class TestPreemphasis:
    def test_boosts_high_freq(self):
        pe = Preemphasis(coeff=0.97)
        # 叠加低中频和高频: 高频幅度应被相对提升
        low = _sine(200.0, 0.1)
        high = _sine(4000.0, 0.01)
        mixed = (low + high).astype(np.float32)
        low_amp_before = float(np.max(np.abs(low)))
        high_amp_before = float(np.max(np.abs(high)))
        ratio_before = high_amp_before / low_amp_before
        out = pe.process_frame(mixed)
        # 高通: 低频衰减 (差分近似微分, 低频几乎抵消)
        out_low_amp = float(np.max(np.abs(out[2:] - 0.97 * out[:-2])))  # 粗略估计
        out_rms = _rms(out)
        # 输出非零且高频分量相对占比上升 (ratio_before 小, 处理后能量相对大)
        assert out_rms > 0.0001

    def test_memory_cross_frame(self):
        pe = Preemphasis(coeff=0.97)
        # 同一信号分成两半, 两半顺序处理 vs 一次性拼接处理 → 结果应接近
        full = _sine(1000.0, 0.1, n=N * 2)
        a = full[:N].copy()
        b = full[N:].copy()
        oa = pe.process_frame(a)
        ob = pe.process_frame(b)
        out_seq = np.concatenate([oa, ob])
        # 重置后一次性
        pe.reset()
        out_one = np.concatenate([pe.process_frame(full[:N]), pe.process_frame(full[N:])])
        np.testing.assert_allclose(out_seq, out_one, atol=1e-5)


class TestAGC:
    def test_silence_does_not_boom(self):
        agc = AGC(sample_rate=SR, target_rms=0.05)
        silence = np.zeros(N, dtype=np.float32)
        # 喂 20 帧静音
        out = silence
        for _ in range(20):
            out, g = agc.process_frame(out)
        assert agc.current_gain <= 1.0 + 1e-4
        assert _rms(out) < 1e-5

    def test_quiet_gets_amplified(self):
        # 选输入幅度使 target_gain 不超过 max_gain, 测试能真正达到 target_rms
        agc = AGC(sample_rate=SR, target_rms=0.05, max_gain=20.0)
        input_rms_target = 0.01  # 这样 target_gain = 0.05/0.01 = 5x, 远小于 20x 上限
        frame = _sine(1000.0, input_rms_target * np.sqrt(2))  # sine amp ~ rms*sqrt2
        gain = 1.0
        for _ in range(60):  # release 慢, 多跑几帧稳定
            out, gain = agc.process_frame(frame)
        out_rms = _rms(out)
        # 输出 RMS 接近 target_rms (允许 ±20%)
        assert 0.04 < out_rms < 0.065, f"AGC out_rms={out_rms:.4f} gain={gain:.2f}"
        assert 3.0 < gain < 8.0  # 约 5x

    def test_loud_gets_compressed(self):
        agc = AGC(sample_rate=SR, target_rms=0.05, min_gain=0.1)
        loud = _sine(1000.0, 0.8)  # 超大音量近讲
        for _ in range(10):
            out, g = agc.process_frame(loud)
        peak = float(np.max(np.abs(out)))
        # 必须无削波 (peak ≤ 0.99)
        assert peak <= 0.99
        # 必须压限 (gain < 1)
        assert g < 0.5

    def test_no_clipping_ever(self):
        """极端正弦: 任何时候 peak 不能超过 1."""
        agc = AGC(sample_rate=SR, target_rms=0.05)
        # 从小声瞬间跳到极大
        for amp in np.logspace(-4, -0.01, 20):  # -48dB → 0dBFS
            f = _sine(1500.0, float(amp))
            out, _ = agc.process_frame(f)
            peak = float(np.max(np.abs(out)))
            assert peak <= 1.0, f"clipped peak={peak} amp={amp}"

    def test_speed(self):
        """30ms 帧处理耗时 < 0.5ms (16x 实时) 目标."""
        agc = AGC(sample_rate=SR, target_rms=0.05)
        frame = _sine(1000.0, 0.01)
        # 预热
        for _ in range(3):
            agc.process_frame(frame)
        t0 = time.perf_counter()
        N_ITERS = 1000
        for _ in range(N_ITERS):
            agc.process_frame(frame)
        dt = time.perf_counter() - t0
        per_frame_ms = (dt / N_ITERS) * 1000.0
        assert per_frame_ms < 0.5, f"AGC too slow: {per_frame_ms:.3f}ms/frame"

    def test_far_field_whisper_gets_amplified(self):
        """远场 3m 轻中声 (RMS ~0.0025 ≈ -52dBFS) 经新参数 AGC 后应被放大到 target_rms, 增益不超过 max_gain."""
        agc = AGC(
            sample_rate=SR,
            target_rms=0.08,   # 远场目标
            max_gain=40.0,     # 远场上限 (+32dB)
            min_gain=0.1,
            attack_ms=10.0,
            release_ms=400.0,  # 远场慢 release
        )
        # 3m 轻中声典型 RMS ≈ 0.0025 (-52dBFS), sine peak = rms*sqrt(2)
        soft = _sine(1000.0, 0.0025 * np.sqrt(2))
        outs = []
        for _ in range(80):  # release 400ms, 约 13 帧/半秒, 跑 80 帧 ≈ 2.4s 让增益收敛
            out, gain = agc.process_frame(soft)
            outs.append(out)
        # 取后半段统计稳态
        tail = np.concatenate(outs[40:])
        tail_rms = _rms(tail)
        assert 0.05 < tail_rms < 0.11, f"far-field out_rms={tail_rms:.4f} gain={gain:.1f} (expected ~0.08)"
        assert 8.0 < gain < 40.0, f"AGC far-field gain={gain:.1f} (expected 10-32x)"
        # 无削波
        assert float(np.max(np.abs(tail))) <= 0.99


class TestNoiseGate:
    def test_tracks_floor_and_suppresses_low_snr(self):
        ng = NoiseGate(sample_rate=SR, snr_db_threshold=6.0, floor_track_sec=0.1)  # 短窗口方便测试
        # 先喂 10 帧底噪
        noise_amp = 0.001
        for _ in range(10):
            noise = (noise_amp * np.random.randn(N).astype(np.float32))
            _, rms = ng.process_frame(noise)
        floor = ng.floor_rms
        assert floor > 0
        # 再喂略高于底噪的语音 (SNR 3dB < 6dB 阈值 → 应衰减)
        speech = (noise_amp * 2.0 * np.random.randn(N).astype(np.float32))  # 2x ≈ 6dB? 实际 2x ≈ 6dB power? 2x amp ≈ 6dB → 刚好边界
        out, _ = ng.process_frame(speech)
        # 再喂大语音 (20x noise) → SNR >> 阈值 → 保留
        big = (noise_amp * 20.0 * np.random.randn(N).astype(np.float32))
        out_big, _ = ng.process_frame(big)
        rms_small = _rms(out)
        rms_big = _rms(out_big)
        # 低 SNR 输出显著衰减于高 SNR
        assert rms_big > rms_small * 3


class TestAudioPreprocessor:
    def test_pipeline_smoke(self):
        pp = AudioPreprocessor(sample_rate=SR, frame_ms=FRAME_MS)
        # (1) 先喂静音让 NoiseGate 的 floor 正确追踪到真实底噪 (真实环境初始化阶段总是静音)
        for _ in range(30):
            pp.process_frame(np.zeros(N, dtype=np.float32))
        # (2) 然后给可被放大的小声 (幅度使 target_gain 不超过 max_gain=20x)
        #     input RMS ≈ 0.008, 目标 target_rms=0.05, gain ≈ 6x 合理
        signal = _sine(1200.0, 0.008 * np.sqrt(2)) + 0.02  # 加少量 DC 0.02 测试 DC 移除
        outs = []
        for _ in range(60):
            outs.append(pp.process_frame(signal))
        # 丢弃前 20 帧 (AGC release 爬升过渡), 只取后 40 帧统计
        out_all = np.concatenate(outs[20:])
        out_rms = _rms(out_all)
        # DC 移除后 RMS 应该显著靠近 target_rms=0.05
        assert 0.02 < out_rms < 0.09, f"pipeline out_rms={out_rms:.4f} (expected ~0.05)"
        # 全程 peak 不超 1.0 (防削波保护工作)
        peak_overall = float(np.max(np.abs(np.concatenate(outs))))
        assert peak_overall <= 1.0

    def test_process_segment(self):
        """process_segment 与逐帧处理结果应一致."""
        pp = AudioPreprocessor(sample_rate=SR, frame_ms=FRAME_MS)
        segment = np.concatenate([_sine(800 + i * 100, 0.01 + 0.001 * i) for i in range(10)])
        # 一次性
        pp.reset()
        out_seg = pp.process_segment(segment)
        # 逐帧
        pp.reset()
        out_chunks = []
        off = 0
        while off < segment.size:
            out_chunks.append(pp.process_frame(segment[off:off + N]))
            off += N
        out_framewise = np.concatenate(out_chunks)
        # 长度相等
        assert out_seg.size == out_framewise.size == segment.size
        # 值几乎一致 (允许浮点误差)
        np.testing.assert_allclose(out_seg, out_framewise, atol=1e-5)


# ============================================================
# EnergyVAD 测试
# ============================================================

from ev.audio.energy_vad import EnergyVAD, EnergyVADParams, EnergyVADState


class TestEnergyVAD:
    def test_silence_never_starts(self):
        vad = EnergyVAD(sample_rate=SR, frame_ms=FRAME_MS,
                        params=EnergyVADParams(snr_threshold_linear=2.5, abs_min_rms=0.001,
                                               start_frames=2, hangover_frames=20))
        for _ in range(100):
            silence = np.zeros(N, dtype=np.float32)
            st = vad.accept_frame(silence)
            assert not st.speech
            assert not st.started
            assert not st.ended

    def test_continuous_speech_starts_and_ends(self):
        vad = EnergyVAD(sample_rate=SR, frame_ms=FRAME_MS,
                        params=EnergyVADParams(start_frames=2, hangover_frames=5))
        # 先静音初始化底噪
        for _ in range(20):
            vad.accept_frame(np.zeros(N, dtype=np.float32))
        # 然后连续 10 帧语音 (大声)
        starts = []
        ends = []
        for i in range(10):
            f = _sine(1000.0, 0.5)
            st = vad.accept_frame(f)
            if st.started:
                starts.append(i)
            if st.ended:
                ends.append(i)
        assert len(starts) == 1
        assert starts[0] == 1  # start_frames=2 → 第 2 帧 (idx=1) 触发 start
        # 之后静音帧, hangover_frames=5 → 静音第 5 帧后 end
        for j in range(10):
            f = np.zeros(N, dtype=np.float32)
            st = vad.accept_frame(f)
            if st.ended:
                ends.append(10 + j)
                break
        assert len(ends) == 1
        assert ends[0] == 10 + 4  # hangover=5 → 第 5 个静音帧结束 (idx=4)

    def test_short_pause_does_not_cut(self):
        """短时停顿 (< hangover) 不应切段."""
        vad = EnergyVAD(sample_rate=SR, frame_ms=FRAME_MS,
                        params=EnergyVADParams(start_frames=1, hangover_frames=10))
        # 静音 → 语音启动
        for _ in range(10):
            vad.accept_frame(np.zeros(N, dtype=np.float32))
        vad.accept_frame(_sine(1000.0, 0.5))  # started
        # 连续两段语音中间 5 帧静音 (小于 hangover=10)
        ends_before_pause = 0
        for _ in range(3):
            vad.accept_frame(_sine(1000.0, 0.5))
        for _ in range(5):
            st = vad.accept_frame(np.zeros(N, dtype=np.float32))
            if st.ended:
                ends_before_pause += 1
        assert ends_before_pause == 0
        assert vad.active is True
        # 再长静音 → 切段
        ends_after = 0
        for _ in range(20):
            st = vad.accept_frame(np.zeros(N, dtype=np.float32))
            if st.ended:
                ends_after += 1
                break
        assert ends_after == 1
        assert vad.active is False

    def test_flush_force_ends(self):
        vad = EnergyVAD(sample_rate=SR, frame_ms=FRAME_MS,
                        params=EnergyVADParams(start_frames=1, hangover_frames=20))
        for _ in range(5):
            vad.accept_frame(np.zeros(N, dtype=np.float32))
        vad.accept_frame(_sine(1000.0, 0.5))
        assert vad.active is True
        st = vad.flush()
        assert st.ended is True
        assert vad.active is False

