# EV 音频 Flow

## Phase 1a 输入链路

```text
麦克风
  -> AudioCapture(16 kHz mono 定长帧)
  -> FSMN-VAD(pre-roll + hangover)
  -> 所有人声段 WAV 归档
  -> Paraformer Streaming(partial,实时显示/EV 早期匹配)
  -> SpeechEnded
       +-> SenseVoiceSmall(final,ITN)
       +-> ERes2NetV2(profile cosine score -> user/non-user/unknown)
  -> SQLite segment
  -> 句首 EV 匹配
  -> EV + user -> QueryCandidate(query_text)
```

`partial` 不作为最终记录；终稿以 SenseVoiceSmall 结果为准。非用户和 unknown
仍保存 WAV 与 SQLite，只是不产生 `QueryCandidate`。当前 query 仅限同一 VAD 段，
不维持多轮激活状态。

## Phase 1b 客户端链路

```text
SwiftUI 主窗口 / 菜单栏
  -> Process 启动仓库 .venv/bin/python
  -> stdin JSONL command -> ev engine serve
  -> Phase 1a 音频管线
  -> stdout JSONL event -> 实时状态、partial、历史和 Query 队列
  -> Python 统一写入 WAV + SQLite
```

客户端不直接调用 FunASR，也不直接读写 SQLite。语音 `query_candidate` 和 GUI 手动
输入都写入 `queries` 表，当前状态统一为 `pending`，留给后续 LLM 消费。关闭主窗口
不停止 engine；退出应用时发送 `shutdown`，停止采集并 flush 已结束语音段。

最小事件集合：

| 事件 | 作用 |
|---|---|
| `audio_level` | GUI 实时输入电平 |
| `speech_started` | 创建段并启动流式 ASR |
| `transcript_partial` | 实时文本和 EV 候选检测 |
| `speech_ended` | 触发终稿、声纹和持久化 |
| `speaker_result` | 分数及三区标签 |
| `segment_committed` | WAV 与 SQLite 已提交 |
| `query_candidate` | 预留给 GUI/LLM 的 query 接口 |

## 最终全天候双工链路

```text
                           +---------- 播放参考 ----------+
                           |                              |
麦克风持续采集 -> 隐私状态 -> AEC/输入预处理 -> AudioFrame |
                                            |             |
                       +--------------------+--------+    |
                       |                    |        |    |
                       v                    v        v    |
                  VAD + 流式 ASR       EV/KWS    环境事件 |
                       +----------+---------+             |
                                  v                       |
                         激活策略 + 用户声纹               |
                                  v                       |
                            对话状态协调器                  |
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
并有明确的工作/静默/禁用状态。Phase 1a 的连续帧、VAD 事件、partial/final、声纹标签、
`segment_id` 和可回放存档均可复用；本阶段不实现 AEC、LLM、TTS 或 barge-in。
