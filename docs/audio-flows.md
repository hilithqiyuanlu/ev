# EV 音频处理 Flow

> 更新时间: 2026-08-12 (声纹学习三档分级 + Qwen3 热词增强 + 降噪/人声确认/环境感知接入中)
>
> 本文档以 `src/ev/pipeline/runtime.py` 为准, 描述麦克风输入 → ASR → 声纹 → 落库
> 的完整输入链路。声纹细节见 [voiceprint.md](./voiceprint.md),
> 声纹学习机制见 [voice-learning.md](./voice-learning.md)。
> 输出侧 (LLM/TTS/播放) 尚未实现, 见文末目标架构。

## 当前实时处理链路 (Phase 1a/1b.2)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ ① 硬件 & 驱动层 (DJI Mic Mini 自身处理)                                      │
│   DJI Mic Mini (领夹 + 接收端)                                                │
│     → 自带: 硬件降噪/硬件 AGC/无线传输纠错/限幅器                             │
│     → USB-C/Lightning → macOS CoreAudio → sounddevice.InputStream             │
│     → blocksize=480 samples (30ms), latency="high"                            │
│     → 16kHz mono PCM float32 (int16→float32, ±1.0 满量程)                   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ② 采集回调 + 双输出 (AudioCapture._callback, audio/capture.py)               │
│   声卡原始 PCM (raw)                                                         │
│     ├─ → raw 帧队列 (供 frames_with_raw() 并行输出, 用于原始音频存档)         │
│     └─ → AudioPreprocessor (逐帧流式, 跨帧保持状态, 远场友好参数)            │
│          ├─ DCRemover: 一阶 IIR 高通 (截止 ~20Hz) → 去除声卡直流偏置        │
│          ├─ Preemphasis: y[n] = x[n] - 0.97·x[n-1]                          │
│          │    → 补偿远场高频 6dB/oct 滚降, 提升辅音清晰度                    │
│          ├─ AGC: 动态自动增益 (远场优化参数)                                 │
│          │    · target_rms = 0.08  (-22dBFS)                                 │
│          │    · 压缩快 attack = 10ms / 放大慢 release = 400ms (防句中呼吸)  │
│          │    · 增益区间 [0.1x (-20dB), 40.0x (+32dB)]                       │
│          │    · 静音安全: RMS<极小值时 gain 回落至 min, 不把静音爆推成嘶嘶声 │
│          │    · 防削波: peak>0.98 时整体缩放 + 同步回拉 current_gain          │
│          └─ NoiseGate: 3s EMA 底噪追踪 (只向更小值跟踪)                      │
│               · SNR threshold = 1.5dB (远场 SNR 本就 2-5dB)                  │
│               · SNR < 1.5dB → 线性衰减到 10% (软门限, 无咔哒声)              │
│               · SNR ≥ 1.5dB → 直通                                          │
│     → processed 帧 (增强后) + raw 帧 (原始未处理) 双输出                      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ③ 实时帧分发 (handle_frame(), pipeline/runtime.py)                           │
│   (processed, raw) 帧对, 每帧:                                               │
│     ├─ audio_level 事件: {rms=processed_rms, raw_rms, gain=AGC倍数}          │
│     ├─ recent_frames / recent_raw_frames 滑窗 (pre-roll 40帧=1200ms)        │
│     ├─ RawNoiseTracker.accept_frame(raw_rms, vad.active) — 常驻 raw 底噪    │
│     │    追踪器: 跨段持久, IDLE 也在跑; 仅 VAD 非活跃且非数字静音时更新;      │
│     │    不对称 EMA (下降快 3s / 上升慢 10s); 前 3s 为 warm-up (不拒绝段)    │
│     └─ 按当前状态喂给三态状态机 (见 ④)                                       │
│                                                                             │
│   CompositeVAD (vad/adapters.py), start=OR / end=AND:                       │
│     ├─ EnergyVAD.accept_frame (逐帧级, audio/energy_vad.py)                  │
│     │    · 3s EMA 底噪追踪 (只向更小跟踪)                                    │
│     │    · 命中: RMS ≥ floor×1.8 (~2.5dB) AND RMS ≥ 0.0003 (-70dBFS)        │
│     │    · 启动防抖: 连续 2 帧 (60ms); 结束 hangover: 连续 20 帧 (600ms)    │
│     ├─ FSMN-VAD.accept (200ms 块级, FunASR fsmn-vad, cache 流式)            │
│     └─ 复合: start=OR (宁可误报勿漏报), end=AND (各自 hangover 走完才结束)  │
│       ※ 配置层注意: VADSettings dataclass 默认已改 "fsmn_only", 但           │
│         load_settings 的无 toml 回退仍是 "or" — 默认安装下有效值为 "or"      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ④ 三态状态机 (PipelineState: IDLE / OBSERVING / RECORDING) — 第二人耳模式    │
│                                                                             │
│  IDLE (只跑 VAD, 不录音不分析):                                             │
│    VAD started 边沿 →                                                       │
│      · 声纹 profile 已就绪 (core≥3): → OBSERVING (带 pre-roll 1200ms 进入)  │
│      · 冷启动 (profile 未就绪): → 直接 RECORDING, 初始 label=user (攒样本)  │
│                                                                             │
│  OBSERVING (观察门, 目标 900ms, 只分析不落盘):                               │
│    进入时带入 pre-roll 1200ms; 因 pre-roll 已超过门控值, 首帧即整窗打分:    │
│      · score ≥ threshold-0.06 → "user"  (0.06 宽限, 详见 voiceprint.md)     │
│      · score < threshold-0.06  → "non-user"                                 │
│    → RECORDING (无论 user/non-user 都入段, 初始 turn 打对应标签)            │
│    ※ 设计意图是"门内 VAD ended (<900ms 短促声) → 静默丢弃回 IDLE",         │
│      但 pre-roll 提到 1200ms 后门控首帧即满, 该路径很难再触发 —             │
│      短噪声过滤实际由 min_duration/人声确认/empty-filler 后段环节承担       │
│                                                                             │
│  RECORDING (正式录音转写):                                                  │
│    · 帧 append (processed + raw), started_at 按已累积时长回溯对齐           │
│    · segment_id = uuid4().hex, 发 speech_started {speaker_label=初始标签}   │
│    · StreamingASR (Paraformer Streaming) reset → 喂入 pre-roll+observing    │
│      帧启动, chunk_size=[0,10,5], 600ms 块增量 partial                      │
│    · 说话人 turn 周期检测 (段内实时, 只打标签不截断):                       │
│        每 600ms 检测, 滑窗=最近 600ms; 不对称迟滞 (认回 user 1 次确认,      │
│        切走 3 次, 间隔 ≥800ms) → speaker_turn_changed 事件                  │
│    · 段结束判定: 见 ⑤                                                      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑤ 段结束 — 6 触发器 (优先级从高到低, 先触发先赢) → force_segment_end        │
│   1) vad_endpoint:     CompositeVAD ended (FSMN end AND Energy hangover完) │
│   2) max_duration:     段长 ≥ 20s 硬上限 (安全网)                           │
│   3) silence_timeout:  raw RMS<0.003 持续 1600ms 且段长 ≥ 3000ms           │
│                        (绝对静音; 最小段长门槛防开口瞬间被误切)             │
│   4) relative_silence: raw RMS 跌破峰值30% 持续 1900ms 且段长 ≥ 6000ms     │
│                        (有底噪环境下"人说完了"; 只切长段)                  │
│   5) energy_silent:    EnergyVAD 判无声累计 ≥ 2100ms (=silence+500ms)      │
│                        且段长 ≥ 3000ms (hangover 耗尽后继续计时;            │
│                        专治 FSMN 卡住)                                     │
│   6) asr_stall:        流式 partial 2500ms 无更新 且段长 ≥ 1000ms          │
│                        且当前 raw RMS < 静音阈值 (仍在说话则视为模型滞后,   │
│                        不提前切)                                           │
│                                                                             │
│   force_segment_end(trigger):                                               │
│     ├─ 关闭最后一个 speaker turn; stream is_final → 最终 partial           │
│     ├─ 发 speech_ended {segment_id, ended_at, trigger}                     │
│     ├─ frames 拼接 → seg_audio (processed); raw_frames → seg_raw           │
│     ├─ < 500ms → discard (too_short), 发 segment_discarded                 │
│     ├─ ≥ 500ms → SegmentJob(audio+raw+partial+speaker_turns+               │
│     │           end_trigger+noise_floor_rms+is_warmup)                     │
│     │           送入 SegmentWorker 后台队列 (不阻塞采集线程)                │
│     └─ 状态复位 (VAD/stream reset) + 重新加载声纹质心                       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑥ 后台串行 worker (SegmentWorker → SegmentProcessor.process)                │
│    单线程队列逐段处理, 采集/VAD 不等待                                       │
│                                                                             │
│  6.0 RMS 统计 (落库元数据): avg/peak raw RMS, 底噪快照, SNR 估算            │
│                                                                             │
│  6.1 人声确认 (设计: DFSMN 降噪 + FSMN one-shot, OR 决策):                  │
│      · DenoiseAdapter (DFSMN-ANS, ModelScope 48k 原生, 内部重采样)          │
│        对 raw 段降噪 → denoised                                             │
│      · _check_speech_segment: 一次性 FSMN 跑完整段, 语音帧占比 ≥5% 才算     │
│        有人声; raw 与 denoised 各跑一次, OR 决策                            │
│      · 无人声且过 warm-up → segment_discarded reason=no_speech_detected,    │
│        不入库不跑 ASR (voice_check.py 声学规则降级为日志解释信号)           │
│      ⚠️ 接线状态: 截至本次更新, transcribe_forever 构造 worker 时未传入     │
│         denoiser/vad_model (两者=None) — 该路径会把所有段判为无人声丢弃,    │
│         属重构进行中的已知中断态, 接线: worker 构造处传 vad_model 与         │
│         speech_enhancement 槽位的 DenoiseAdapter 实例                       │
│                                                                             │
│  6.2 终稿 ASR (懒加载, 首个非跳过段才加载; 运行时热切换):                   │
│      按模型目录自动检测 (config.json 的 model_type 优先于 FunASR 配置):     │
│      ├─ Qwen3-ASR 1.7B (transformers, MPS/CUDA fp16) [默认槽位]             │
│      │   · 热词双通道:                                                      │
│      │     a) prompt 注入 ("请准确转写，注意以下词：...")                    │
│      │     b) anchor-based logits boosting: 仅当已解码末尾命中热词前缀       │
│      │        才给续写 token 加 logits (scale 2.0, 单 token 上限 4.0,       │
│      │        最小锚定 1 字); 从不抑制任何 token; system 词(小E)不参与      │
│      │   · max_new_tokens 按时长伸缩 [256, 1024] (~20 tokens/s)             │
│      │   · 逐 token avg_logprob 输出 (供幻觉检测)                           │
│      │   · 支持 <|x.xx|> 时间戳 → 句级 segments (供 utterance 对齐)         │
│      └─ SenseVoice Small (FunASR, 多语言): hotword 参数注入,                │
│          CJK 字符间空格清理 (保留英文空格); 无时间戳                        │
│      GUI 切换 asr_final 槽位 → reload_final_asr: 立即卸载旧模型,           │
│      下一个段到来时加载新模型 (省内存)                                      │
│      ※ ASR 输入音频: 降噪后 denoised 优先, 否则用 processed (接线生效后)    │
│                                                                             │
│  6.3 终稿后处理与回退:                                                      │
│      · 热词幻觉检测 (仅 Qwen3+热词+boosting 开启时):                        │
│        avg_logprob < -2.0 且 (热词覆盖率 >55% 或无功能词)                   │
│        → final 置空, 回退 partial                                          │
│      · 截断保护: final 长度 < partial 的 60% → 回退 partial (防半残终稿)   │
│      · 超短段省流: <800ms 且无 speaker turns 且 partial 空/纯语气词         │
│        → 跳过 final ASR (直接用 partial)                                   │
│                                                                             │
│  6.4 段过滤: final 为空 → 降级 partial → 仍为空 → discard (empty)           │
│      纯语气词 (17词表) → discard (filler); 统一 reason=empty_or_filler,     │
│      无 WAV 无 DB                                                           │
│                                                                             │
│  6.5 utterance 对齐 (_align_utterances):                                    │
│      · final ASR 有真实时间戳 (Qwen3) → 按句切分                            │
│      · 无时间戳 (SenseVoice) → 标点切句 + 字符比例映射 (P0 fallback)        │
│      · 每句按时间中点落进 speaker_turns → 该句标 user/non-user              │
│                                                                             │
│  6.6 声纹识别 (融合判决, 全段 embedding 为主):                              │
│      fullseg=user → user; fullseg=non-user 但 turns 含 user → user;         │
│      都 non-user → non-user; 冷启动全 user                                  │
│      (阈值默认 0.40; 细节见 voiceprint.md §3-④)                             │
│                                                                             │
│  6.7 唤醒词与 query 决策 (vui.py):                                          │
│      · 句首"小E": 前缀剥离 (嗨/喂/哎 + 语气词), 同音容错 (小易/小艺)        │
│      · 唤醒词检测不区分说话人 (per-utterance turn 噪声大),                  │
│        query_candidate 由融合后的段级 dominant_speaker 门控                 │
│      · 冷启动额外要求: 唤醒词后 query ≥ 2 字                                │
│                                                                             │
│  6.8 自动声纹学习: 三档分级 (≥0.70 core / 0.40-0.70 cache+diversity /       │
│      <0.40 拒收), onboarding 门控 (core≥5 才开闸), 簇内竞争+自动补位        │
│      (完整机制见 voice-learning.md)                                         │
│                                                                             │
│  6.9 个人词典 (热词): 字符串 (system>manual>auto, ≤80词, 无权重后缀) 供     │
│      prompt/hotword 注入; entries (manual>auto, 排除 system, 带权重) 供     │
│      logits boosting; 词典变更时 broadcast 给运行中的 worker                │
│                                                                             │
│  6.10 双 WAV 存档 + SQLite:                                                 │
│      · {id}.wav → processed (或降噪后) 增强音频; {id}.raw.wav → raw 原始    │
│      · segments 表: speaker_turns/utterances/dominant_speaker/contains_user │
│        /end_trigger/quality_label/avg_raw_rms/peak_raw_rms/noise_floor_rms/ │
│        snr_db; asr_final_model 记录实际加载的模型目录名                     │
│      · 删除 segment 时两个 WAV 先移 trash, DB 删除成功后才真删 (可回滚)     │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑦ Query 决策                                                                │
│   · 检测到唤醒词 + 融合 label=user → query_candidate=True → queries 表      │
│     (source="voice", status="pending"), 发 query_candidate 事件             │
│   · 冷启动阶段: 唤醒后 query≥2字 才产生 query                               │
│   · non-user 或 无唤醒词 → 仅存档, 不产生 query                             │
│   · 手动输入的 query 直接进 queries 表, source="manual"                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ 旁路: 环境感知 (脚手架, 未接线) ──────────────────────────────────────────┐
│ EnvironmentMonitor (audio/environment.py): YAMNet tflite (AudioSet 521     │
│ 类映射 ~15 个有意义类别: typing/background_speech/music/background_noise/  │
│ alert/animal/...), 10s ring buffer, 每 2s 取最近 5s 推理, 时序聚合成       │
│ 持续状态 → environment_event → EnvironmentLog (logs/ 下 jsonl, 不入        │
│ WAV/SQLite)。独立于语音路径, 不依赖 FSMN 触发。                            │
│ ⚠️ 当前 service.py 仅构造实例并传入 transcribe_forever, 但 runtime 未使    │
│    用该参数, .feed()/.start() 均无人调用 — 不产生任何事件                  │
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
   │   IDLE   │ ◄──────────  │ OBSERVING │  (短促声<VAD end在门内→丢弃)
   └──────────┘              └─────┬─────┘
        ▲                          │ 整窗打分
        │                          ▼
        │                     ┌─────────────────────────────┐
        │                     │          RECORDING          │
        │                     │  · 流式 ASR partial         │
        │                     │  · 600ms 滑窗 speaker turn  │
        │                     │    (不对称迟滞, 不截断)     │
        │                     │  · 6 触发器端点判定         │
        │                     └─────────────┬───────────────┘
        │                                   │ force_segment_end
        └───────────────────────────────────┘
                     (状态复位, 重载质心)
```

## 模型槽位一览 (ModelRegistry)

| 槽位 | 默认模型 | 来源 | 说明 |
|---|---|---|---|
| `vad` | fsmn-vad (ev-fsmn-vad-zh-16k) | GitHub release | 流式端点 + 段级人声确认 |
| `asr_streaming` | paraformer-zh-streaming | GitHub release | 600ms 增量 partial |
| `asr_final` | qwen3-asr-1.7B [默认] / sensevoice-small | ModelScope / GitHub | 终稿, 可热切换 |
| `speaker` | eres2netv2 (ev-eres2netv2-zh-16k) | GitHub release | 声纹 embedding |
| `speech_enhancement` | dfsmn-ans (ModelScope 48k causal) | ModelScope | 段级降噪 (未接线) |
| `environment` | yamnet (tflite, AudioSet) | GitHub release | 环境声分类 (未接线) |

注: paraformer-zh (终稿) 与 qwen3-asr-0.6b 已从 catalog 移除。

## Phase 1b 客户端 (macOS SwiftUI)

```text
SwiftUI 主窗口 / 菜单栏
  → Process 启动仓库 .venv/bin/python ev engine serve
  → stdin JSONL command (start_listening/stop_listening/set_device/...)
  → 上述 Python 音频管线
  → stdout JSONL event → AppModel @Published 状态驱动 UI
  → Python 统一完成 WAV 双存档 + SQLite + 声纹建模
```

客户端不直接调用 ASR/声纹模型，也不直接读写 SQLite。终稿、声纹、WAV 和 SQLite 在后台
worker 串行完成，采集/VAD 不等待这些计算。语音 `query_candidate` 和 GUI 手动输入都写入
`queries` 表，当前状态统一为 `pending`，留给后续 LLM 消费。关闭主窗口不停止 engine；
退出应用时发送 `shutdown`，停止采集并 flush 已结束语音段。

终稿 ASR 模型在 GUI 模型页热切换 (Qwen3-ASR 1.7B 或 SenseVoice Small)，无需重启监听。
声纹录入有独立命令 (`start_voice_enrollment` / `stop_voice_enrollment` /
`capture_manual_sample`)，另有待确认样本队列 (`list_pending_voice_samples` /
`confirm_voice_sample` / `reject_voice_sample`)。

### 调试 UI (远场调参用)

首页波形下方显示两行实时调试信息（监听中可见）:
- **`XX dB`**: 原始(raw)音频 RMS dBFS, 颜色编码:
  - 白色 (≥ -48dB): 音量足够, 应该能录上
  - 橙色 (-60 ~ -48dB): 有点轻, 可能识别不准
  - 灰色 (< -60dB): 基本没检测到人声
- **`×N.N`**: 当前 AGC 增益倍数 (远场正常说话应在 10-32x 区间)

### 最小事件集合

| 事件 | 作用 | 关键字段 |
|---|---|---|
| `capture_started` | 采集已实际启动 | device, sample_rate, channels |
| `audio_level` | GUI 实时输入电平 | rms(处理后), raw_rms(原始), gain(AGC倍数) |
| `speech_started` | 进入 RECORDING (含初始声纹标签) | segment_id, started_at, speaker_label |
| `transcript_partial` | 实时文本 | segment_id, text |
| `speech_ended` | 段结束 | segment_id, ended_at, trigger (vad_endpoint/max_duration/silence_timeout/relative_silence/energy_silent/asr_stall/stop) |
| `speaker_turn_changed` | 段内说话人切换 (只打标签, 不截断) | segment_id, from, to, score |
| `segment_discarded` | 段被丢弃 | segment_id, reason (too_short/empty_or_filler/no_speech_detected), duration_ms |
| `segment_processing` | 后台处理开始 | segment_id, phase, queue_depth |
| `speaker_result` | 声纹融合判决 | segment_id, label, score |
| `segment_committed` | WAV 与 SQLite 已提交 | 完整 SegmentRecord (含 speaker_turns/utterances/end_trigger/质量元数据) |
| `segment_failed` | 段级错误 | segment_id, code, message |
| `query_candidate` | 预留给 GUI/LLM 的 query | segment_id, source, text |
| `voice_sample_added` | 声纹样本收录 | segment_id, tier, core/cache/centroid_count, is_ready |
| `voice_profile_ready` | 冷启动完成 (core≥3) | sample_count, core_count |
| `environment_event` | 环境声分类状态变化 (未接线) | timestamp, category, confidence, duration_sec |

## 本次重构已解决 / 仍存在的架构问题

已解决 (对照旧版"已知架构缺陷"):
1. ✅ **缺少"静默观察"状态** → 三态状态机: IDLE 只跑 VAD / OBSERVING 只分析不录音 /
   RECORDING 正式录入
2. ✅ **说话人切换截断段** → 段内 turn 标记 (不截断) + 段末全段 embedding 融合判决
3. ⚠️→↩ **他人短促声音开段**: OBSERVING 门内丢弃曾解决, 但 pre-roll 提到
   1200ms 后门控首帧即满, 该路径失效 (见"仍存在"#3); 目前靠 ≥500ms 最小时长、
   人声确认 (接线后)、empty/filler 过滤兜住短噪声
4. ✅ **闭合时机单一** → 6 触发器兜底, 且静音类触发器带最小段长门槛防误切短段,
   asr_stall 需同时满足音频真静音 (说话中 ASR 滞后不提前切)
5. ✅ **"只入我一句还标错人"** → 全段融合: fullseg 与 turns 任一判 user 即 user
6. ✅ **热词幻觉** (模糊音频被热词串联) → anchor-based boosting (只正增量不抑制)
   + avg_logprob/覆盖率/功能词联合检测回退 partial
7. ✅ **终稿截断半残** → final 显著短于 partial (<60%) 时回退 partial;
   max_new_tokens 按时长伸缩

仍存在 / 进行中:
1. ⚠️ **人声确认接线中断**: worker 的 FSMN 人声确认 + DFSMN 降噪已写好,
   但 transcribe_forever 构造 worker 时未传 vad_model/denoiser (=None),
   当前工作区状态下所有段会在 warm-up 后被 no_speech_detected 丢弃 ——
   重构进行中的 P0 级断点
2. ⚠️ **环境感知未接线**: EnvironmentMonitor 已构造但无人 feed/start,
   不产生 environment_event
3. ⚠️ **OBSERVING 门控失效**: pre-roll 1200ms > 门控 900ms, 门控首帧即触发,
   "门内 VAD 结束即丢弃短促声"的设计路径不再生效 (短噪声后移由
   min_duration/人声确认/empty-filler 过滤); 要么调大门控要么接受现状
4. **VAD 仍是"哑巴守门人"**: FSMN 只分人声/非人声, EnergyVAD 只看能量,
   都不区分说话人 → 持续的他人讲话仍会开段 (靠事后打标签补救)
5. **段内混合语音物理全录**: turn 只是事后标签, 信号层面不分离;
   non-user 音频照样存档 (隐私/存储角度可再收紧)
6. **OBSERVING 门控不拦长段**: 非用户讲话仍会开段录入,
   靠融合判决保证不产生 query、不参与学习
7. **utterance 对齐精度依赖 final ASR**: Qwen3 有真实时间戳较准;
   SenseVoice 无时间戳时退化为字符比例映射 (P0)
8. **配置双默认不一致**: VADSettings dataclass 默认 combine_start_mode="fsmn_only",
   但 load_settings 无 toml 回退为 "or", 有效值以后者为准 (ev.toml 已删除)

### 分层演进路线

| 阶段 | 能力 | 技术方案 | 状态 |
|---|---|---|---|
| P0 | 说话人门控逻辑修复 | OBSERVING 门控 + turn 标记 + 融合判决 | ✅ 已完成 |
| P1 | 智能VAD/人声确认 | FSMN 段级人声确认 (raw∨降噪, ≥5%占比) + DFSMN 降噪 | 🚧 代码就绪, 接线中断 |
| P2 | 帧级说话人标记 | 600ms 滑动窗口 embedding + USER/NON_USER 实时标签 | ✅ 已完成 |
| P3 | 音频事件检测(SED) | YAMNet (AudioSet 521→~15类), 独立旁路轮询 | 🚧 脚手架已建, 未接线 |
| P4 | 语义VAD | ASR partial + 意图分类, 判断"是不是在跟我说话" | 待做 |
| P5 | 多人分离(Diarization) | pyannote/ERes2Net 聚类, 多人对话区分说话人A/B/C | 待做 |
| P6 | 音源分离 | Conv-TasNet/Demucs (可选), 信号层面人声/噪音/多说话人分离 | 待做 |

## 最终全天候双工链路 (目标架构)

```text
                           +---------- 播放参考 ----------+
                           |                              |
麦克风持续采集 -> 隐私状态 -> AEC/输入预处理 -> AudioFrame |
                                            |             |
                       +--------------------+--------+    |
                       |                    |        |    |
                       v                    v        v    |
                智能VAD+SED         唤醒词KWS   说话人分离 |
              (人声/环境声分类)                  (实时Diar)|
                       +----------+---------+             |
                                  v                       |
                         状态协调器 (idle/observing/recording)
                         +--------+---------+              |
                         v                  v              |
                     LLM/工具            记忆系统           |
                         |                                 |
                         v                                 |
                      流式 TTS -> 音频输出 -----------------+
                         ^
                         |
播放期间用户说话 -> barge-in -> 停止播放并取消旧生成 -> 新输入
```

最终形态要求采集不中断、外放有 AEC、打断能取消整条输出链路、激活与存储策略分离，
并有明确的工作/静默/禁用状态。当前阶段已实现: 连续帧、三态状态机、VAD 事件、
partial/final、声纹 turn 标签与融合判决、`segment_id` 和双WAV可回放存档;
本阶段不实现 AEC、LLM、TTS 或 barge-in。
