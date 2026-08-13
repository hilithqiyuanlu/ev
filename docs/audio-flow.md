# EV 音频处理 Flow

> 更新时间: 2026-08-13
>
> 本文档以 `src/ev/pipeline/runtime.py` 为准，描述麦克风输入 → 感知 → 决策 → 理解
> 的完整输入链路。声纹细节见 [voiceprint.md](./voiceprint.md)，
> 声纹学习机制见 [voice-learning.md](./voice-learning.md)，
> 词典/热词机制见 [dictionary.md](./dictionary.md)。

## 目标架构：三层

EV 的语音输入被拆成三层，各司其职：

```text
① 实时感知层  —— 已实现 ——
   持续感知「是否有人声、是谁、什么环境」，只产出轻量信号，不产出文本。
   VAD (FSMN + EnergyVAD) · 声纹 (ERes2NetV2) · 环境 (YAMNet) · 降噪 (DFSMN-ANS)

② 终稿理解层  —— 已实现 ——
   高质量终稿转写 (SenseVoice Small)，服务 LLM：段结束后一次性产出
   准确、带标点的文本。非自回归模型，拿到完整音频才一次输出。
   由 `SegmentWorker` 懒加载，`_create_final_asr_adapter()` 按 config 自动检测
   适配器（SenseVoiceAdapter / Qwen3ASRAdapter）。

③ 业务理解层  —— Phase 3 预留 ——
   把 ② 的文本喂给业务逻辑（意图理解 / agent / 工具调用）。
```

> 关键设计原则：**ASR 是离线终稿模型，段结束后才出文本。**
> 实时信号交给感知层（VAD/声纹/环境），文本转写交给终稿层，二者分工明确。

---

## 已实现的实时链路（① 层 + 状态机 + 段处理）

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
│          └─ NoiseGate: 3s EMA 底噪追踪, SNR < 1.5dB → 软门限 (噪声环境联动↑9dB) │
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
│   6) speaker_switch:   说话人切换确认后立即切段 (方案 A)                      │
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
│        (16k↔48k 重采样; 依赖 modelscope[framework]+speechbrain, 未装则静默│
│         禁用并在启动时输出 [denoise] 提示)                                  │
│      · 语音帧占比 ≥5% 才算有人声; raw 与 denoised 各跑一次                    │
│      · 无人声且过 warm-up → discard (no_speech_detected)                     │
│                                                                             │
│  6.2 终稿 ASR: ✅ 已接入 (Fun-ASR-Nano-2512, speech-LLM)                    │
│      · SegmentWorker.final_asr_factory 钩子, service.py 注入 FunASRNanoAdapter│
│      · numpy → torch.Tensor 内存分支 generate, 不落盘临时 WAV                 │
│      · 段结束后整段转写, 无流式 partial                                       │
│      · 懒加载: 首个非跳过段才加载, 运行时热切换 (reload_final_asr)            │
│      · 输入音频: 降噪后 denoised 优先, 否则用 processed                      │
│                                                                             │
│  6.3 段过滤:                                                                 │
│      · final 为空 → discard (empty)                                          │
│      · 纯语气词 → discard (filler)                                           │
│                                                                             │
│  6.4 utterance 对齐 (_align_utterances):                                    │
│      · 有真实时间戳 → 按句切分                                               │
│      · 无时间戳 → 标点切句 + 字符比例映射                                    │
│      · 每句按时间中点落进 speaker_turns → 标 user/non-user                   │
│                                                                             │
│  6.5 声纹识别 (融合判决, 全段 embedding 为主):                              │
│      fullseg=user → user; fullseg=non-user 但 turns 含 user → user;         │
│      都 non-user → non-user; 冷启动全 user                                  │
│                                                                             │
│  6.6 唤醒词与 query 决策 (vui.py):                                          │
│      · 句首"小E": 前缀剥离, 同音容错                                         │
│      · query_candidate 由融合后的段级 dominant_speaker 门控                  │
│      · 冷启动额外要求: 唤醒词后 query ≥ 2 字                                 │
│                                                                             │
│  6.7 自动声纹学习: 三档分级 (≥0.70 core / 0.40-0.70 cache / <0.40 拒收)     │
│      (完整机制见 voice-learning.md)                                         │
│                                                                             │
│  6.8 热词词典: system > manual > auto (≤80词); 命中统计随段落库                │
│      (hotword_density / use_count); 消费未接入 — Fun-ASR-Nano 不透传热词,      │
│      原拼音后处理已移除 (优化方向见 dictionary-hotword-notes.md)               │
│                                                                             │
│  6.9 双 WAV 存档 + SQLite:                                                     │
│      · {id}.wav → 降噪后 denoised (降噪不可用=前端 processed)                  │
│      · {id}.raw.wav → raw 原始 (留作未来重处理/上下文)                         │
│      · 声纹样本 voice_samples/{id}.wav = 高置信 user 段 (decouple)             │
│      · 删除 segment 时 WAV 先移 trash, DB 成功后才真删                         │
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
│ 联动前端降噪: 噪声类 (typing/background_noise/music/appliance) 收紧         │
│  NoiseGate (SNR 1.5→9dB) 与 AGC 上限 (40→6x), 缓解噪声误触发。              │
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
| `vad` | fsmn-vad | ModelScope | 流式端点 + 段级人声确认 |
| `speech_enhancement` | dfsmn-ans | ModelScope | 段级降噪 + 人声确认输入 |
| `speaker` | eres2netv2 | GitHub release | 声纹 embedding |
| `environment` | yamnet | GitHub release | 环境声分类 (旁路, 实时) |
| `asr_final` | sensevoice-small | GitHub release | 终稿 ASR (非自回归, 默认) |

> 备选终稿 ASR：`qwen3-asr-1.7b`（ModelScope，自回归，支持词级时间戳 + 热词 logits 偏置）。
> 流式 ASR 槽位已在 v0.1.0 移除。

> 两个模型已放置于 `~/Library/Application Support/EV/models/`。注册表启动时通过
> `ModelRegistry.rescan_local()` 自动扫描 `models_root` 并注册本地已存在的模型目录
> （无需联网下载），使其出现在模型页「已安装」列表并支持卸载。

## 待接入清单

| 组件 | 当前状态 | 接入点 |
|---|---|---|
| 流式 ASR (paraformer-zh-streaming) | 槽位已注册, adapter 待接入 | `transcribe_forever` 需新增 streaming adapter, 服务 barge-in / 字幕 / `asr_stall` 端点 |
| 热词消费 (词典 → 终稿 ASR) | 已收集未消费, 原拼音后处理已移除 | `FunASRNanoAdapter.transcribe` 透传 hotwords (见 [dictionary-hotword-notes.md](./dictionary-hotword-notes.md)) |
| 业务理解层 (LLM 意图) | Phase 3 预留 | 段落库后接 query 决策下游 |

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

ASR 模型可在 GUI 模型页安装/卸载，切换后下一个段自动加载新模型。声纹录入通过
独立命令 (`start_voice_enrollment` / `stop_voice_enrollment` / `capture_manual_sample`)，
另有待确认样本队列 (`list_pending_voice_samples` / `confirm_voice_sample` /
`reject_voice_sample`)。

### 事件参考

| 事件 | 作用 | 关键字段 |
|---|---|---|
| `capture_started` | 采集已实际启动 | device, sample_rate, channels |
| `audio_level` | GUI 实时输入电平 | rms, raw_rms, gain |
| `speech_started` | 进入 RECORDING | segment_id, started_at, speaker_label |
| `speech_ended` | 段结束 | segment_id, ended_at, trigger |
| `speaker_turn_changed` | 说话人切换 (含切段标记) | segment_id, from, to, score, segment_split |
| `segment_discarded` | 段被丢弃 | segment_id, reason, duration_ms |
| `segment_processing` | 后台处理开始 | segment_id, phase, queue_depth |
| `speaker_result` | 声纹融合判决 | segment_id, label, score |
| `segment_committed` | WAV 与 SQLite 已提交 | 完整 SegmentRecord |
| `segment_failed` | 段级错误 | segment_id, code, message |
| `query_candidate` | 预留给 GUI/LLM 的 query | segment_id, source, text |
| `voice_sample_added` | 声纹样本收录 | segment_id, tier, core/cache/centroid_count |
| `voice_sample_confirmed` / `rejected` | 待确认样本处理结果 | sample_id, tier |
| `voice_profile_ready` | 冷启动完成 (core≥3) | sample_count, core_count |
| `voice_profile_reset` | 声纹档案重置 | — |
| `segment_corrected` | 手动纠错落地 | segment_id, changed, corrected_text |
| `segment_deleted` | 历史段删除 | segment_id, deleted |
| `segment_list` | 历史段列表响应 | segments |
| `lexicon_updated` | 词典增删改热更新 | word, added/updated/deleted |
| `lexicon_list` | 词典列表响应 | words |
| `engine_state` | 引擎运行状态 | state |
| `model_status` / `available_models` / `installed_models` | 模型页状态与列表 | models |
| `environment_event` | 环境声分类状态变化 | timestamp, category, confidence, duration_sec |
