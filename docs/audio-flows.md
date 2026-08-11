# EV 音频处理 Flow

> 更新时间: 2026-08-11 | DB v13 | 三态状态机 + 质量门控
>
> 以 `src/ev/pipeline/runtime.py` 为准，描述麦克风 → VAD → ASR → 声纹 → 落库 的完整输入链路。

## 数据流总览

```
麦克风 16kHz mono → ② 预处理(DC→preemphasis→AGC→noise gate)
  → 双输出: processed(增强) + raw(原始)
  → ③ CompositeVAD(FSMN OR Energy) → 三态状态机
  → ④ IDLE → OBSERVING(900ms门控) → RECORDING
  → ⑤ 6触发器端点判定 → SegmentJob 入队
  → ⑥ 后台 worker: 终稿ASR → 声纹 → 唤醒词 → 双WAV落库
  → ⑦ Query决策
```

## ② 预处理

逐帧流式，跨帧保持状态。参数见 `ev.toml [preprocess]`：

| 模块 | 作用 | 关键参数 |
|------|------|----------|
| DCRemover | 一阶 IIR 高通 ~20Hz，去直流偏置 | — |
| Preemphasis | `y[n]=x[n]-0.97·x[n-1]`，补偿远场高频滚降 | coeff=0.97 |
| AGC | 动态增益，远场优化 | target=0.08, attack=10ms, release=400ms, gain=[0.1, 40.0] |
| NoiseGate | 3s EMA 底噪追踪，软门限 | SNR<1.5dB → 衰减到 10% |

## ③ CompositeVAD

start=OR / end=AND 复合策略：

| 子VAD | 类型 | 粒度 | 启动 | 结束 |
|-------|------|------|------|------|
| EnergyVAD | 能量 (raw RMS vs EMA floor) | 逐帧 30ms | 连续2帧 SNR≥1.8x, RMS≥0.0003 | hangover 20帧(600ms) |
| FSMN-VAD | 神经网络 (FunASR) | 200ms 块 | 模型输出 speech 边沿 | 模型输出 silence 边沿 |

## ④ 三态状态机

```
IDLE ──VAD start, profile就绪──→ OBSERVING (门控900ms, pre-roll 1200ms)
  │                                  │
  │  VAD start, 冷启动                │ 900ms满 → 整窗声纹打分
  │  (core<3)                        │   ≥threshold-0.06 → user
  │                                  │   <threshold-0.06 → non-user
  │                                  │
  ▼                                  ▼
RECORDING ←──────────────────────────┘
  │ · Streaming ASR (Paraformer, 600ms chunk partial)
  │ · 每600ms 滑窗 speaker turn 检测 (不对称迟滞, 只标记不截断)
  │ · 6触发器端点判定
  │
  ▼
force_segment_end → 状态复位, 重载声纹质心

OBSERVING 门内 VAD end → 静默丢弃回 IDLE (咳嗽/短促他人声, 不产生段)
```

## ⑤ 端点触发 (6 触发器, 优先级从高到低)

所有静音类触发器有最小段长门槛（防短段被过早切掉）：

| 优先级 | 触发器 | 条件 | 门槛 |
|--------|--------|------|------|
| 1 | `vad_endpoint` | CompositeVAD ended | 无 |
| 2 | `max_duration` | 段长 ≥ 20s | 无 |
| 3 | `silence_timeout` | raw RMS < 0.003 持续 1600ms | ≥ 3s |
| 4 | `relative_silence` | raw RMS < peak×30% 持续 1900ms | ≥ 6s |
| 5 | `energy_silent` | EnergyVAD 无声累计 ≥ 2100ms | ≥ 3s |
| 6 | `asr_stall` | partial 2500ms 无更新 + raw RMS < 0.003 | ≥ 1s |

## ⑥ 后台 Worker (`SegmentWorker._run()`)

单线程队列，采集/VAD 不等待。每段处理流程：

**6.0 省流** → `<800ms 且无 speaker turns 且 partial 空/语气词` → 跳过 final ASR

**6.1 质量门控** → raw 音频分帧计算 SNR（90 百分位帧 RMS，防静音稀释）/ 底噪 floor（RawNoiseTracker 跨段持久）：
- warm-up 期 (前 3s) 不拒绝
- `avg_raw_rms < min_audible_rms (0.0005)` → `rejected_low_level`
- `snr_db < min_snr_db (3.0)` → `rejected_low_snr`
- **声学特征二次检查**（仅 SNR/电平通过的段）：ZCR、频谱质心、频谱平坦度、RMS 包络方差 → `rejected_non_voice`
- 质量拒绝段跳过 final ASR，转写留空，但 WAV 仍存档 + DB 写质量元数据

**6.2 终稿 ASR**（懒加载，首个非跳过段才加载）：

| 模型 | 引擎 | 热词 | 时间戳 |
|------|------|------|--------|
| Qwen3-ASR 0.6B/1.7B [默认] | transformers (MPS/CUDA fp16) | anchor-based logits 增强 + prompt 注入 | ✅ `<\|x.xx\|>` |
| Paraformer Large | FunASR | 参数注入 | ❌ 字符比例 fallback |

- 动态 token 预算: `max(256, min(1024, duration_sec × 20))`
- GUI 热切换：卸载旧模型 → 下个段加载新模型

**6.3 段过滤** → final 为空降级用 partial → partial 仍为空 → discard (empty/filler)

**6.4 utterance 对齐** → 有时间戳按句切分，无时间戳标点切句 + 字符比例映射

**6.5 声纹融合判决**（全段 embedding 为主，threshold 0.50）：
- fullseg=user → user
- fullseg=non-user 但 turns 含 user → user（拼接污染容错）
- 两者都 non-user → non-user

**6.6 唤醒词** → 句首"小E"匹配（同音容错），须在 user 句中；唤醒后连续 user 句并入 query 上下文

**6.7 自动声纹学习** → user-only 段, ≥0.60 分, 1.5-10s, ≥30s 间隔；CORE(K-means 质心) + CACHE(FIFO)

**6.8 双 WAV 存档** → `{id}.wav` (enhanced) + `{id}.raw.wav` (raw)；segments 表含 speaker_turns/utterances/quality 字段；级联删 voice samples

## 事件协议 (stdin JSONL command → stdout JSONL event)

| 事件 | 触发时机 |
|------|----------|
| `capture_started` | 采集已启动 |
| `audio_level` | 每帧 (~30ms): rms, raw_rms, gain |
| `speech_started` | 进入 RECORDING: segment_id, speaker_label |
| `transcript_partial` | 流式 ASR 更新: segment_id, text |
| `speaker_turn_changed` | 段内说话人切换: from, to, score |
| `speech_ended` | 段结束: segment_id, trigger |
| `segment_discarded` | 段丢弃: reason (too_short/empty_or_filler) |
| `speaker_result` | 声纹融合判决: label, score |
| `segment_committed` | WAV+DB 已提交: 完整 SegmentRecord |
| `segment_failed` | 段级错误: code, message |
| `query_candidate` | 语音 query 待处理 |
| `voice_sample_added` | 声纹样本收录: tier, core_count |
| `voice_profile_ready` | 冷启动完成 (core≥3) |

## 关键配置 (`ev.toml`)

| 节 | 关键字段 | 默认值 |
|----|---------|--------|
| `[preprocess]` | agc_target_rms, agc_max_gain, noisegate_snr_db | 0.08 / 40.0 / 1.5 |
| `[vad]` | energy_snr_linear, energy_abs_min_rms, combine_start/end | 1.8 / 0.0003 / or+and |
| `[segment]` | min_duration_ms, max_duration_ms, silence_timeout_ms | 500 / 20000 / 1600 |
| `[segment]` | min_snr_db, min_audible_rms, raw_noise_warmup_sec | 3.0 / 0.0005 / 3.0 |
| `[speaker]` | threshold, max_core_samples, max_centroids | 0.40 / 20 / 3 |
| `[asr]` | hotword_boost_scale, hotword_boost_max | 2.0 / 6.0 |
| `[voice_learning]` | collect_min_score, onboarding_target | 0.60 / 5 |
