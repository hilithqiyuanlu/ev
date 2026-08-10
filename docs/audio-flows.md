# EV 音频处理 Flow

> 更新时间: 2026-08-10 (三态状态机 / "第二人耳模式" 重构后, Phase 1a/1b.2)
>
> 本文档以 `src/ev/pipeline/runtime.py` 为准, 描述麦克风输入 → ASR → 声纹 → 落库
> 的完整输入链路。输出侧 (LLM/TTS/播放) 尚未实现, 见文末目标架构。

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
│     │    (GUI 调试显示: raw dBFS + AGC 增益)                                 │
│     ├─ recent_frames / recent_raw_frames 滑窗 (pre-roll 20帧=600ms)         │
│     └─ 按当前状态喂给三态状态机 (见 ④)                                       │
│                                                                             │
│   CompositeVAD (vad/adapters.py), start=OR / end=AND:                       │
│     ├─ EnergyVAD.accept_frame (逐帧级, audio/energy_vad.py)                  │
│     │    · 3s EMA 底噪追踪 (只向更小跟踪)                                    │
│     │    · 命中: RMS ≥ floor×1.8 (~2.5dB) AND RMS ≥ 0.0003 (-70dBFS)        │
│     │    · 启动防抖: 连续 2 帧 (60ms); 结束 hangover: 连续 20 帧 (600ms)    │
│     ├─ FSMN-VAD.accept (200ms 块级, FunASR fsmn-vad, cache 流式)            │
│     └─ 复合: start=OR (宁可误报勿漏报), end=AND (各自 hangover 走完才结束)  │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ④ 三态状态机 (PipelineState: IDLE / OBSERVING / RECORDING) — 第二人耳模式    │
│                                                                             │
│  IDLE (只跑 VAD, 不录音不分析):                                             │
│    VAD started 边沿 →                                                       │
│      · 声纹 profile 已就绪 (core≥3): → OBSERVING (带 pre-roll 600ms 进入)   │
│      · 冷启动 (profile 未就绪): → 直接 RECORDING, 初始 label=user (攒样本)  │
│                                                                             │
│  OBSERVING (观察门, 900ms, 只分析不落盘):                                   │
│    继续累积帧 + 喂 VAD:                                                     │
│      · VAD ended (门内声音 <900ms 就消失, 咳嗽/敲击/短促他人声)              │
│        → 静默丢弃回 IDLE (不产生段, 不发任何事件)                           │
│      · 累积满 900ms → 整窗打分 (_score_window):                             │
│          窗口级 normalize_loudness → ERes2NetV2 embedding                   │
│          → 多质心余弦 best score                                            │
│          · score ≥ threshold-0.06 → "user"                                  │
│            (0.06 宽限: 短窗 embedding 噪声大, 段末全段评分会纠正误判)       │
│          · score < threshold-0.06  → "non-user"                             │
│        → RECORDING (无论 user/non-user 都入段, 初始 turn 打对应标签,        │
│           供段末融合判决; non-user 段不产生 query、不参与声纹学习)          │
│                                                                             │
│  RECORDING (正式录音转写):                                                  │
│    · 帧 append (processed + raw), started_at 按已累积时长回溯对齐           │
│    · segment_id = uuid4().hex, 发 speech_started {speaker_label=初始标签}   │
│    · StreamingASR reset → 一次性喂入 pre-roll+observing 帧启动              │
│       (Paraformer Streaming, chunk_size=[0,10,5], 600ms 块增量 partial,     │
│        encoder_look_back=4, decoder_look_back=1)                            │
│    · 说话人 turn 周期检测 (段内实时, 只打标签不截断):                       │
│        · 每 600ms 检测一次, 滑窗=最近 600ms processed 帧                    │
│        · normalize_loudness → embedding → 多质心 best score                 │
│        · 不对称迟滞: non-user→user 1次确认 (快速认回)                       │
│                      user→non-user 连续3次确认 (防单窗噪声误翻)             │
│        · 两次切换最小间隔 800ms                                             │
│        · 确认切换 → 关闭当前 turn / 开新 turn (边界对齐窗中点)              │
│          → 发 speaker_turn_changed {from, to, score}                        │
│    · 段结束判定: 见 ⑤                                                      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑤ 段结束 — 6 触发器 (优先级从高到低, 先触发先赢) → force_segment_end        │
│   1) vad_endpoint:     CompositeVAD ended (FSMN end AND Energy hangover完) │
│   2) max_duration:     段长 ≥ 20s 硬上限 (安全网)                           │
│   3) silence_timeout:  raw RMS < 0.003 (~-50dBFS) 持续 1200ms (绝对静音)   │
│   4) relative_silence: 峰值>-50dBFS 后 raw RMS 跌破峰值30% 持续 1500ms     │
│                        (有底噪环境下"人说完了"; 用 raw 防 AGC 失真)         │
│   5) energy_silent:    EnergyVAD 判无声累计 ≥ 1400ms                        │
│                        (hangover 600ms 耗尽后继续计时; 专治 FSMN 卡住)      │
│   6) asr_stall:        流式 partial 2500ms 无更新 且段长 ≥ 1000ms          │
│                                                                             │
│   force_segment_end(trigger):                                               │
│     ├─ 关闭最后一个 speaker turn                                            │
│     ├─ stream.accept(is_final=True) → 最终 partial                         │
│     ├─ 发 speech_ended {segment_id, ended_at, trigger}                     │
│     ├─ frames 拼接 → seg_audio (processed); raw_frames → seg_raw           │
│     ├─ < 500ms → discard (too_short, 咳嗽/敲击/噪声), 发 segment_discarded │
│     ├─ ≥ 500ms → SegmentJob(audio+raw+partial+speaker_turns)               │
│     │           送入 SegmentWorker 后台队列 (不阻塞采集线程)                │
│     └─ 状态复位 (VAD/stream reset) + 重新加载声纹质心 (拣到新样本)          │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑥ 后台串行 worker (SegmentWorker → SegmentProcessor.process)                │
│    单线程队列逐段处理, 采集/VAD 不等待                                       │
│                                                                             │
│  6.0 超短段省流: <800ms 且无 speaker turns 且 partial 为空/纯语气词         │
│      → 跳过 final ASR (直接用 partial 判空/语气词后丢弃)                    │
│                                                                             │
│  6.1 终稿 ASR (懒加载, 首个非跳过段才加载; 可插拔, 运行时热切换):           │
│      按模型目录自动检测 (config.json 的 model_type 优先于 FunASR 配置):     │
│      ├─ Qwen3-ASR 0.6B/1.7B (transformers, MPS/CUDA fp16) [默认槽位]        │
│      │   · 热词: prompt 注入 ("请准确转写，注意以下词：...")                │
│      │   · 输出 language <lang><asr_text>... 解析; 贪心解码 max 256 tokens │
│      │   · 支持 <|x.xx|> 时间戳 → 句级 segments (供 utterance 对齐)        │
│      └─ Paraformer Large (FunASR): hotword 参数注入, use_itn,              │
│          CJK 字符间空格清理 (保留英文空格); 无时间戳                        │
│      GUI 切换 asr_final 槽位 → reload_final_asr: 立即卸载旧模型,           │
│      下一个段到来时加载新模型 (省内存)                                      │
│                                                                             │
│  6.2 段过滤: final 为空 → 降级用 partial → 仍为空 → discard (empty)         │
│      纯语气词 (嗯/啊/呃/那个/就是等, 17词表) → discard (filler)             │
│      (丢弃即无 WAV 无 DB, 发 segment_discarded reason=empty_or_filler)      │
│                                                                             │
│  6.3 utterance 对齐 (_align_utterances):                                    │
│      · final ASR 有真实时间戳 (Qwen3) → 按句切分, 直接用语速边界            │
│      · 无时间戳 (Paraformer) → 标点切句 + 字符比例映射 (P0 fallback)        │
│      · 每句按时间中点落进 speaker_turns → 该句标 user/non-user              │
│                                                                             │
│  6.4 声纹识别 (融合判决, 全段 embedding 为主):                              │
│      · audio_for_embedding = normalize_loudness(seg_audio)                  │
│        (段级 RMS 校准到 0.05, 增益封顶 10x; 与帧级 AGC 互补不冲突)          │
│      · 全段 embedding → 多质心 best score → fullseg label (threshold 0.50) │
│      · 融合策略:                                                           │
│          fullseg=user                    → user                             │
│          fullseg=non-user 但 turns 含 user → user (段中切换/拼接污染)       │
│          两者都 non-user                 → non-user                         │
│      · 冷启动 (core<3): 全部判 user 用于学习                                │
│      · dominant_speaker 与融合标签对齐后写库                                │
│                                                                             │
│  6.5 唤醒词检测 (utterance 级, vui.py):                                     │
│      · 句首"小E"匹配: 前缀剥离 (嗨/喂/哎/诶/噢/哦 + 嗯啊那个等语气词),      │
│        同音容错 (小易/小艺), 单字限定防误触发 ("小姨"/"意思" 不命中)        │
│      · decide_query_from_utterances: 唤醒词必须在 user 句中才有效;          │
│        唤醒句之后连续的 user 句并入 query_text 上下文                       │
│      · 冷启动额外要求: 唤醒词后 query ≥ 2 字 (防闲聊提到"小E"误触发)        │
│                                                                             │
│  6.6 自动声纹学习:                                                          │
│      · 仅 user-only 段 (无任何 non-user turn, 全段 embedding 不被污染)      │
│      · 门槛: 1.5-10s / 非语气词 / ≥30s 间隔 / score ≥ 0.40                 │
│      · 冷启动: 全收 (score 记 0.8)                                          │
│      · CORE 层 (最多20条) → K-means 质心 (1-5样本→1, 6-10→2, 11+→3)        │
│      · CACHE 层 (最多50条) → 仅记录不建模, FIFO 淘汰                        │
│      · 手动样本 (录入命令) 永远在 CORE, 不被自动淘汰                        │
│                                                                             │
│  6.7 个人词典 (热词): system(小E 5.0) > manual > auto 优先级,              │
│      最多 80 词拼成空格分隔字符串 (无权重后缀, 防 FunASR 解析失败),         │
│      词典变更时 broadcast 给运行中的 worker                                 │
│                                                                             │
│  6.8 双 WAV 存档 + SQLite:                                                  │
│      · {id}.wav     → processed 增强音频 (回放/ASR重转写用, 默认)           │
│      · {id}.raw.wav → raw 原始音频 (为后续人声增强/环境声分析/SE保留)       │
│      · segments 表: speaker_turns / utterances / dominant_speaker /        │
│        contains_user / source_type 等字段                                   │
│      · 删除 segment 时两个 WAV 先移 trash, DB 删除成功后才真删              │
│        (失败回滚), 级联删关联 voice samples                                 │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⑦ Query 决策                                                                │
│   · user 句中检测到唤醒词 → query_candidate=True → 写 queries 表            │
│     (source="voice", status="pending"), 发 query_candidate 事件             │
│   · 冷启动阶段全部走 user 流程收集样本, 唤醒后 query≥2字 才产生 query       │
│   · non-user 或 无唤醒词 → 仅存档, 不产生 query                             │
│   · 手动输入的 query 直接进 queries 表, source="manual"                     │
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

终稿 ASR 模型可在 GUI 模型页热切换 (Qwen3-ASR 0.6B/1.7B 或 Paraformer)，无需重启监听。
声纹录入有独立命令 (`start_voice_enrollment` / `stop_voice_enrollment` /
`capture_manual_sample`)，复用同一 speaker embedding 模型，手动样本永远在 CORE 层。

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
| `speech_ended` | 段结束 | segment_id, ended_at, trigger (vad_endpoint/max_duration/silence_timeout/relative_silence/energy_silent/asr_stall) |
| `speaker_turn_changed` | 段内说话人切换 (只打标签, 不截断) | segment_id, from, to, score |
| `segment_discarded` | 段被丢弃 | segment_id, reason (too_short/empty_or_filler), duration_ms |
| `segment_processing` | 后台处理开始 | segment_id, phase, queue_depth |
| `speaker_result` | 声纹融合判决 | segment_id, label, score |
| `segment_committed` | WAV 与 SQLite 已提交 | 完整 SegmentRecord (含 speaker_turns/utterances/dominant_speaker/contains_user) |
| `segment_failed` | 段级错误 | segment_id, code, message |
| `query_candidate` | 预留给 GUI/LLM 的 query | segment_id, source, text |
| `voice_sample_added` | 声纹样本收录 | segment_id, tier, core_count, cache_count, centroid_count, is_ready |
| `voice_profile_ready` | 冷启动完成 (core≥3) | sample_count, core_count |

## 本次重构已解决 / 仍存在的架构问题

已解决 (对照旧版"已知架构缺陷"):
1. ✅ **缺少"静默观察"状态** → 三态状态机: IDLE 只跑 VAD / OBSERVING 只分析不录音 /
   RECORDING 正式录入
2. ✅ **说话人切换截断段** → 旧版连续3次低分 force_segment_end 且切完立刻又开新段;
   现改为段内 turn 标记 (不截断) + 段末全段 embedding 融合判决
3. ✅ **他人短促声音开段** → OBSERVING 门内 VAD 结束 (<900ms) 直接静默丢弃
4. ✅ **闭合时机单一** (只依赖 FSMN 报 end, 卡住就不收尾) → 6 触发器兜底
   (绝对静音/相对静音/EnergyVAD无声/ASR停滞/20s硬上限)
5. ✅ **"只入我一句还标错人"** → 全段融合策略: fullseg 与 turns 任一判 user 即 user

仍存在 (下一阶段重点):
1. **VAD 仍是"哑巴守门人"**: FSMN 只分人声/非人声, EnergyVAD 只看能量,
   都不区分说话人 → 敲门声/持续的他人讲话仍会开段 (靠事后打标签补救)
2. **段内混合语音物理全录**: turn 只是事后标签, 信号层面不分离;
   non-user 音频照样存档 (隐私/存储角度可再收紧)
3. **OBSERVING 门控不拦长段**: ≥900ms 的非用户讲话仍会开段录入,
   靠融合判决保证不产生 query、不参与学习
4. **utterance 对齐精度依赖 final ASR**: Qwen3 有真实时间戳较准;
   Paraformer 无时间戳时退化为字符比例映射 (P0)

### 分层演进路线

| 阶段 | 能力 | 技术方案 | 状态 |
|---|---|---|---|
| P0 | 说话人门控逻辑修复 | OBSERVING 门控 + turn 标记 + 融合判决 | ✅ 已完成 |
| P1 | 智能VAD替换能量VAD | Silero VAD (400KB, CPU实时) 替代EnergyVAD | 待做 |
| P2 | 帧级说话人标记 | 600ms 滑动窗口 embedding + USER/NON_USER 实时标签 | ✅ 已完成 |
| P3 | 音频事件检测(SED) | PANNs小模型/MobileNet变种, 分类敲门/咀嚼/走路/键盘等 | 待做 |
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
