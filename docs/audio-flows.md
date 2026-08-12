# EV 音频处理 Flow

> 更新时间: 2026-08-12
>
> 本文档以 `src/ev/pipeline/runtime.py` 为准，描述麦克风输入 → ASR → 声纹 → 落库
> 的完整输入链路。声纹细节见 [voiceprint.md](./voiceprint.md)，
> 声纹学习机制见 [voice-learning.md](./voice-learning.md)。

## 实时处理链路

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ ① 硬件 & 驱动层                                                               │
│   麦克风 → macOS CoreAudio → sounddevice.InputStream                         │
│     → blocksize=480 samples (30ms), latency="high"                            │
│     → 16kHz mono PCM float32 (int16→float32, ±1.0 满量程)                    │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ② 采集回调 + 双输出 (AudioCapture._callback, audio/capture.py)               │
│   声卡原始 PCM (raw)                                                         │
│     ├─ → raw 帧队列 (供 frames_with_raw() 并行输出, 用于原始音频存档)         │
│     └─ → AudioPreprocessor (逐帧流式, 跨帧保持状态)                           │
│          ├─ DCRemover: 一阶 IIR 高通 (截止 ~20Hz) → 去除声卡直流偏置        │
│          ├─ Preemphasis: y[n] = x[n] - 0.97·x[n-1]                          │
│          ├─ AGC: 动态自动增益                                                 │
│          │    · target_rms = 0.08 (-22dBFS)                                  │
│          │    · attack = 10ms / release = 400ms                              │
│          │    · 增益区间 [0.1x (-20dB), 40.0x (+32dB)]                       │
│          └─ NoiseGate: 3s EMA 底噪追踪, SNR < 1.5dB → 软门限                  │
│     → processed 帧 (增强后) + raw 帧 (原始未处理) 双输出                      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ③ 实时帧分发 (handle_frame(), pipeline/runtime.py)                           │
│   (processed, raw) 帧对, 每帧:                                               │
│     ├─ audio_level 事件: {rms, raw_rms, gain}                                │
│     ├─ recent_frames / recent_raw_frames 滑窗 (pre-roll 40帧=1200ms)        │
│     ├─ RawNoiseTracker — 常驻 raw 底噪追踪 (跨段持久, IDLE 也在跑)           │
│     └─ 按当前状态喂给三态状态机 (见 ④)                                       │
│                                                                             │
│   CompositeVAD (vad/adapters.py), start=fsmn_only / end=AND:                │
│     ├─ EnergyVAD.accept_frame (逐帧级)                                       │
│     │    · 命中: RMS ≥ floor×1.8 AND RMS ≥ 0.0003                           │
│     │    · 启动防抖: 连续 2 帧 (60ms); 结束 hangover: 20 帧 (600ms)         │
│     └─ FSMN-VAD.accept (200ms 块级, FunASR fsmn-vad, cache 流式)            │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ④ 三态状态机 (PipelineState: IDLE / OBSERVING / RECORDING)                   │
│                                                                             │
│  IDLE (只跑 VAD, 不录音不分析):                                             │
│    VAD started 边沿 →                                                       │
│      · 声纹 profile 已就绪 (core≥3): → OBSERVING (带 pre-roll 1200ms 进入)  │
│      · 冷启动 (profile 未就绪): → 直接 RECORDING, 初始 label=user            │
│                                                                             │
│  OBSERVING (观察门, 目标 900ms, 只分析不落盘):                               │
│    进入时带入 pre-roll 1200ms; 首帧即整窗打分:                               │
│      · score ≥ threshold-0.06 → "user"                                      │
│      · score < threshold-0.06  → "non-user"                                 │
│    → RECORDING (无论 user/non-user 都入段, 初始 turn 打对应标签)            │
│    ※ 说话人切换时也会触发 OBSERVING 门控 (方案 A: 逐人切段)                  │
│                                                                             │
│  RECORDING (正式录音转写):                                                  │
│    · 帧 append (processed + raw), started_at 按已累积时长回溯对齐           │
│    · segment_id = uuid4().hex, 发 speech_started {speaker_label}            │
│    · 说话人 turn 周期检测: 每 600ms, 不对称迟滞 (认回 user 1 次,              │
│      切走 3 次, 间隔 ≥800ms)                                                │
│    · 确认切换时立即切段 (方案 A): 旧段提交, 新说话人进 OBSERVING              │
│    · 段结束判定: 见 ⑤                                                      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑤ 段结束 — 6 触发器 (优先级从高到低) → force_segment_end                    │
│   1) vad_endpoint:     CompositeVAD ended                                   │
│   2) max_duration:     段长 ≥ 20s 硬上限                                     │
│   3) silence_timeout:  raw RMS<0.003 持续 1600ms 且段长 ≥ 3000ms           │
│   4) relative_silence: raw RMS 跌破峰值30% 持续 1900ms 且段长 ≥ 6000ms     │
│   5) energy_silent:    EnergyVAD 判无声累计 ≥ 2100ms 且段长 ≥ 3000ms       │
│   6) asr_stall:        流式 partial 2500ms 无更新 且段长 ≥ 1000ms           │
│                        且当前 raw RMS < 静音阈值                            │
│   7) speaker_switch:   说话人切换确认后立即切段 (方案 A)                      │
│                                                                             │
│   force_segment_end(trigger):                                               │
│     ├─ 关闭最后一个 speaker turn                                             │
│     ├─ 发 speech_ended {segment_id, ended_at, trigger}                      │
│     ├─ < 500ms → discard (too_short)                                        │
│     └─ ≥ 500ms → SegmentJob 送入后台队列 (不阻塞采集线程)                    │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑥ 后台串行 worker (SegmentWorker → SegmentProcessor.process)                │
│    单线程队列逐段处理, 采集/VAD 不等待                                       │
│                                                                             │
│  6.1 人声确认: DFSMN 降噪 + FSMN one-shot, OR 决策                           │
│      · DenoiseAdapter (DFSMN-ANS) 对 raw 段降噪 → denoised                  │
│      · 语音帧占比 ≥5% 才算有人声; raw 与 denoised 各跑一次                    │
│      · 无人声且过 warm-up → discard (no_speech_detected)                     │
│                                                                             │
│  6.2 自动语音识别 (ASR):                                                     │
│      · 默认模型: SenseVoice Small (FunASR, 多语言, 支持中英日韩粤)            │
│      · 备选: Qwen3-ASR 1.7B (transformers, 热词增强, 时间戳)                  │
│      · 懒加载: 首个非跳过段才加载模型                                         │
│      · 运行时可通过 registry 热切换 (reload_final_asr)                       │
│      · 输入音频: 降噪后 denoised 优先, 否则用 processed                      │
│                                                                             │
│  6.3 后处理:                                                                 │
│      · 截断保护: final 长度 < partial 的 60% → 回退 partial                  │
│      · 超短段省流: <800ms 且无 speaker turns → 跳过 final ASR                │
│                                                                             │
│  6.4 段过滤:                                                                 │
│      · final 为空 → 降级 partial → 仍为空 → discard (empty)                  │
│      · 纯语气词 → discard (filler)                                           │
│                                                                             │
│  6.5 utterance 对齐 (_align_utterances):                                    │
│      · 有真实时间戳 (Qwen3) → 按句切分                                       │
│      · 无时间戳 (SenseVoice) → 标点切句 + 字符比例映射                        │
│      · 每句按时间中点落进 speaker_turns → 标 user/non-user                   │
│                                                                             │
│  6.6 声纹识别 (融合判决, 全段 embedding 为主):                              │
│      fullseg=user → user; fullseg=non-user 但 turns 含 user → user;         │
│      都 non-user → non-user; 冷启动全 user                                  │
│                                                                             │
│  6.7 唤醒词与 query 决策 (vui.py):                                          │
│      · 句首"小E": 前缀剥离, 同音容错                                         │
│      · query_candidate 由融合后的段级 dominant_speaker 门控                  │
│      · 冷启动额外要求: 唤醒词后 query ≥ 2 字                                 │
│                                                                             │
│  6.8 自动声纹学习: 三档分级 (≥0.70 core / 0.40-0.70 cache / <0.40 拒收)     │
│      (完整机制见 voice-learning.md)                                         │
│                                                                             │
│  6.9 热词词典: system > manual > auto (≤80词), 供 prompt/logits boosting    │
│                                                                             │
│  6.10 双 WAV 存档 + SQLite:                                                 │
│      · {id}.wav → 增强音频; {id}.raw.wav → raw 原始                         │
│      · 删除 segment 时 WAV 先移 trash, DB 成功后才真删                        │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑦ Query 决策                                                                │
│   · 检测到唤醒词 + 融合 label=user → query_candidate → queries 表            │
│   · non-user 或 无唤醒词 → 仅存档, 不产生 query                             │
│   · 手动输入: source="manual"                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ 旁路: 环境感知 ─────────────────────────────────────────────────────────────┐
│ EnvironmentMonitor (audio/environment.py): YAMNet tflite (AudioSet 521     │
│ 类映射 ~15 个有意义类别), 10s ring buffer, 每 2s 取最近 5s 推理,            │
│ 时序聚合成持续状态 → environment_event → EnvironmentLog (logs/ 下 jsonl).   │
│ 独立于语音路径, 不依赖 VAD 触发。                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 三态状态机转换图

```text
                 VAD started
        ┌─────────┴──────────┐
        │                    │
   profile 就绪           冷启动 (core<3)
        │                    │
        ▼                    ▼
   ┌──────────┐  门控900ms   ┌───────────┐
   │   IDLE   │ ◄──────────  │ OBSERVING │
   └──────────┘              └─────┬─────┘
        ▲                          │ 整窗打分
        │                          ▼
        │                     ┌─────────────────────────────┐
        │                     │          RECORDING          │
        │                     │  · 说话人 turn 周期检测     │
        │                     │  · 6 触发器端点判定         │
        │                     │  · 说话人切换→切段+门控     │
        │                     └─────────────┬───────────────┘
        │                                   │ force_segment_end
        └───────────────────────────────────┘
                     (状态复位, 重载质心)
```

说话人切换时额外路径: RECORDING → (切段提交) → OBSERVING → RECORDING。

## 模型槽位一览

| 槽位 | 默认模型 | 来源 | 说明 |
|---|---|---|---|
| `vad` | fsmn-vad | GitHub release | 流式端点 + 段级人声确认 |
| `speech_enhancement` | dfsmn-ans | ModelScope | 段级降噪 + 人声确认输入 |
| `asr_final` | sensevoice-small | GitHub release | 自动语音识别 (备选: qwen3-asr-1.7b) |
| `speaker` | eres2netv2 | GitHub release | 声纹 embedding |
| `environment` | yamnet | GitHub release | 环境声分类 (旁路, 实时) |

## macOS 客户端 (SwiftUI)

```text
SwiftUI 主窗口 / 菜单栏
  → Process 启动 .venv/bin/python ev engine serve
  → stdin JSONL command (start_listening / stop_listening / set_device / ...)
  → 上述 Python 音频管线
  → stdout JSONL event → AppModel @Published 状态驱动 UI
  → Python 统一完成 WAV 双存档 + SQLite + 声纹建模
```

客户端不直接调用 ASR/声纹模型，也不直接读写 SQLite。终稿、声纹、WAV 和 SQLite
在后台 worker 串行完成，采集/VAD 不等待这些计算。关闭主窗口不停止 engine；
退出应用时发送 `shutdown`。

ASR 模型可在 GUI 模型页安装/卸载 (SenseVoice Small 或 Qwen3-ASR 1.7B)，
切换后下一个段自动加载新模型。声纹录入通过独立命令
(`start_voice_enrollment` / `stop_voice_enrollment` / `capture_manual_sample`)，
另有待确认样本队列 (`list_pending_voice_samples` / `confirm_voice_sample` /
`reject_voice_sample`)。

### 事件参考

| 事件 | 作用 | 关键字段 |
|---|---|---|
| `capture_started` | 采集已实际启动 | device, sample_rate, channels |
| `audio_level` | GUI 实时输入电平 | rms, raw_rms, gain |
| `speech_started` | 进入 RECORDING | segment_id, started_at, speaker_label |
| `transcript_partial` | 实时文本 | segment_id, text |
| `speech_ended` | 段结束 | segment_id, ended_at, trigger |
| `speaker_turn_changed` | 说话人切换 (含切段标记) | segment_id, from, to, score, segment_split |
| `segment_discarded` | 段被丢弃 | segment_id, reason, duration_ms |
| `segment_processing` | 后台处理开始 | segment_id, phase, queue_depth |
| `speaker_result` | 声纹融合判决 | segment_id, label, score |
| `segment_committed` | WAV 与 SQLite 已提交 | 完整 SegmentRecord |
| `segment_failed` | 段级错误 | segment_id, code, message |
| `query_candidate` | 预留给 GUI/LLM 的 query | segment_id, source, text |
| `voice_sample_added` | 声纹样本收录 | segment_id, tier, core/cache/centroid_count |
| `voice_profile_ready` | 冷启动完成 (core≥3) | sample_count, core_count |
| `environment_event` | 环境声分类状态变化 | timestamp, category, confidence, duration_sec |
