# EV 音频处理 Flow

> 更新时间: 2026-08-13
>
> 本文档以 `src/ev/pipeline/runtime.py` 为准，描述麦克风输入 → 感知 → 决策 → 理解
> 的完整输入链路。声纹细节见 [voiceprint.md](./voiceprint.md)，
> 声纹学习机制见 [voice-learning.md](./voice-learning.md)，
> 词典/热词机制见 [dictionary.md](./dictionary.md)。
>
> 维护规则：凡修改音频采集、预处理、VAD、流式/终稿 ASR、质量判断、声纹、
> 环境声旁路、事件协议、数据库诊断字段或实时状态 UI，必须在同一次开发中同步
> 更新本文档，并以当前代码和测试结果为准。

## 目标架构：三层

EV 的语音输入被拆成三层，各司其职：

```text
① 实时感知层  —— 已实现 ——
   持续感知「是否有人声、是谁、什么环境」，只产出轻量信号，不产出文本。
   VAD (FSMN + EnergyVAD) · 声纹 (ERes2NetV2) · 环境 (YAMNet) · 降噪 (DFSMN-ANS)

② 流式转写层 + 终稿理解层  —— 已实现 ——
   Paraformer Streaming 在 RECORDING 内持续输出 partial 字幕；
   高质量终稿转写 (Fun-ASR-Nano-2512)，服务 LLM：段结束后一次性产出
   准确、带标点的文本。speech-LLM 自回归模型，拿到完整音频才一次输出。
   流式 ASR 与终稿 ASR 都在监听会话启动阶段预热；终稿失败时保留 partial，
   通过 `model_error` 明确通知客户端。

③ 业务理解层  —— Phase 3 预留 ——
   把 ② 的文本喂给业务逻辑（意图理解 / agent / 工具调用）。
```

> 关键设计原则：**流式 ASR 负责低延迟字幕，终稿 ASR 负责段结束后的高质量文本。**
> 环境声仍只读取 raw 音频，不经过语音降噪或任一语音 ASR。

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
│  6.1 降噪与信号统计:                                                        │
│      · DenoiseAdapter (DFSMN-ANS) 对 raw 段降噪 → denoised                  │
│        (16k↔48k 重采样; 依赖 modelscope[framework]+speechbrain, 未装则静默│
│         禁用并在启动时输出 [denoise] 提示)                                  │
│      · raw 音频计算 avg/peak RMS、noise floor 与 SNR                        │
│      · RECORDING 内累计 VAD active 帧，形成 speech_ratio                    │
│      · 环境声继续消费 raw 帧，不读取 denoised                               │
│                                                                             │
│  6.2 流式 ASR: ✅ 已接入 (Paraformer Streaming)                             │
│      · 每个语音段独立维护 cache，约每200ms发送去重 partial                   │
│      · 不注入热词，发送 revision；段结束 flush 并清理 cache                  │
│      · 记录 stream_first_partial_ms / stream_revision_count                  │
│                                                                             │
│  6.3 终稿 ASR: ✅ 已接入 (Fun-ASR-Nano-2512, speech-LLM)                    │
│      · SegmentWorker.final_asr_factory 钩子, service.py 注入 FunASRNanoAdapter│
│      · numpy → torch.Tensor 内存分支 generate, 不落盘临时 WAV                 │
│      · 监听会话启动阶段预热，段结束后整段转写                             │
│      · 记录 final_latency_ms；加载失败通过 model_error 暴露                 │
│      · 输入音频: 降噪后 denoised 优先, 否则用 processed                      │
│                                                                             │
│  6.4 质量评估与段过滤:                                                       │
│      · 基于 raw RMS / peak / noise floor / SNR / speech_ratio                 │
│      · 结合 partial 稳定性，输出 ok/borderline/rejected_*                    │
│      · 无效段仍保存 WAV 与历史记录，标红显示淘汰原因                         │
│      · 无效段不生成 Query、不计热词、不进入自动声纹学习                      │
│      · borderline 自动样本只进入 cache，手动核心样本不受影响                 │
│                                                                             │
│  6.5 utterance 对齐 (_align_utterances):                                    │
│      · 有真实时间戳 → 按句切分                                               │
│      · 无时间戳 → 标点切句 + 字符比例映射                                    │
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
│  6.9 热词词典: 证据触发                                                     │
│      · 流式 ASR 不注入热词；只使用流式文本作为首遍证据                     │
│      · 全量 active 非系统词参与保守字形/顺序检索，最多选择 8 个候选         │
│      · 无证据时终稿不传 hotwords；自动词须人工确认后才能参与                │
│      · 落库实际候选；只统计实际传入且最终文本命中的候选词                   │
│                                                                             │
│  6.10 双 WAV 存档 + SQLite:                                                │
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
│ EnvironmentMonitor (audio/environment.py): YAMNet LiteRT (AudioSet 521     │
│ 类映射 ~15 个有意义类别), 10s ring buffer, 每 2s 取最近 5s 推理。           │
│ environment_status 每轮更新实时 UI；完成区间由 environment_event 写入       │
│ EnvironmentLog (logs/ 下 jsonl)，支持按日期查询与清空。                      │
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
| `asr_final` | fun-asr-nano-2512 | ModelScope | 终稿 ASR (speech-LLM) |
| `asr_stream` | paraformer-zh-streaming | GitHub release | 流式 partial 字幕 |

> 终稿 ASR 热词只接收经过流式文本证据筛选的 active 非系统候选词；自动学习词须人工确认，流式 ASR 永不注入热词。

> 本地模型放置于 `~/Library/Application Support/EV/models/`。注册表启动时通过
> `ModelRegistry.rescan_local()` 自动扫描 `models_root` 并注册本地已存在的模型目录
> （无需联网下载），使其出现在模型页「已安装」列表并支持卸载。

## 待接入清单

| 组件 | 当前状态 | 接入点 |
|---|---|---|
| 流式 ASR (paraformer-zh-streaming) | ✅ 已接入 | `StreamingASRAdapter.transcribe_chunk()`，段内 cache + partial revision |
| 热词消费 (词典 → 终稿 ASR) | ✅ 证据触发 | 流式文本检索人工词条后，最多向终稿注入8个候选词 |
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

ASR 模型可在 GUI 模型页安装/卸载；监听会话启动时按当前槽位加载。声纹录入通过
独立命令 (`start_voice_enrollment` / `stop_voice_enrollment` / `capture_manual_sample`)，
另有待确认样本队列 (`list_pending_voice_samples` / `confirm_voice_sample` /
`reject_voice_sample`)。

### 事件参考

| 事件 | 作用 | 关键字段 |
|---|---|---|
| `capture_started` | 采集已实际启动 | device, sample_rate, channels |
| `model_loading` / `model_ready` / `model_error` | 模型槽位预热状态 | component, message |
| `pipeline_status` | 统一流水线阶段与空闲心跳 | phase, component, elapsed_ms, queue_depth, message |
| `audio_level` | GUI 实时输入电平 | rms, raw_rms, gain |
| `speech_started` | 进入 RECORDING | segment_id, started_at, speaker_label |
| `speech_ended` | 段结束 | segment_id, ended_at, trigger |
| `speaker_turn_changed` | 说话人切换 (含切段标记) | segment_id, from, to, score, segment_split |
| `segment_discarded` | 段被丢弃 | segment_id, reason, duration_ms |
| `segment_processing` | 后台处理开始 | segment_id, phase, queue_depth |
| `transcript_partial` | 流式字幕更新 | segment_id, text, revision, latency_ms |
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
| `lexicon_updated` | 词典增删改及确认/启停热更新 | word/id, added/updated/confirmed/rejected/status_updated |
| `lexicon_list` | 词典列表响应 | words |
| `engine_state` | 引擎运行状态 | state |
| `model_status` / `available_models` / `installed_models` | 模型页状态与列表 | models |
| `environment_event` | 环境声分类状态变化 | timestamp, category, confidence, duration_sec |

### 质量与数据库诊断字段

`segments` 表当前 schema version 为 v18。除原有 RMS/SNR 字段外，新增：

| 字段 | 含义 |
|---|---|
| `speech_ratio` | 当前段被 VAD 判为 speech-like 的帧比例 |
| `stream_first_partial_ms` | 从段开始到首个 partial 的耗时 |
| `stream_revision_count` | 流式假设更新次数 |
| `final_latency_ms` | 终稿 ASR 调用耗时 |

质量标签包括 `ok`、`borderline`、`rejected_low_level`、`rejected_low_snr`、
`rejected_non_voice`、`rejected_unstable` 和 `processing_error`。开发阶段无效段
仍保留 WAV 与历史记录，历史页的“清空无效”删除所有 `quality_label != ok` 的段。

### 环境声隔离约束

环境声旁路必须继续使用 `env_monitor.feed(raw)`：

- 不经过 `DenoiseAdapter`。
- 不进入 `StreamingASRAdapter` 或终稿 ASR。
- 不依赖语音 VAD 是否启动。
- 语音链路增加状态事件、质量门控或模型预热时，不改变环境事件协议和环境历史存储。
