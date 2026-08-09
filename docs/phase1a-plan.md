# EV Phase 1a:VUI-ASR 输入闭环

## 范围

Phase 1a 只完成稳定、可回放、可测量的语音输入链路。暂不实现 NLP、LLM、
TTS、GUI、独立 KWS、多轮激活和多说话人聚类。GUI 与 LLM 通过事件接口接入。

```text
麦克风 -> FSMN-VAD -> 所有人声段 WAV -> Paraformer Streaming partial
       -> ERes2NetV2 用户声纹 -> SenseVoiceSmall final -> SQLite
       -> EV 句首匹配 -> EV + user 标记 query_candidate
```

所有 VAD 人声段都归档。声纹只决定标签和是否可形成 query，不决定是否保存。

## 模型与运行时

首版使用 FunASR Python 和本地模型目录，运行时不会让 FunASR 静默联网，也暂不转换
ONNX。模型由显式的 `ev models download` 或 macOS 客户端从固定 Release 下载并校验；
MacBook Pro M4 以 CPU 可运行作为基线，MPS 后续单独验证。

| 环节 | ModelScope 模型 | 本地目录 |
|---|---|---|
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | `ev-fsmn-vad-zh-16k` |
| 流式 ASR | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | `ev-paraformer-zh-streaming-16k` |
| 终稿 ASR | `iic/SenseVoiceSmall` | `ev-sensevoice-small` |
| 用户声纹 | `iic/speech_eres2netv2_sv_zh-cn_16k-common` | `ev-eres2netv2-zh-16k` |

FSMN-VAD、Paraformer 和 ERes2NetV2 在进程启动时加载并常驻。SenseVoiceSmall
首次生成终稿时懒加载，之后保持常驻。只有实际内存超限时才增加卸载策略。

模型包发布于 [models-v0.1.0](https://github.com/hilithqiyuanlu/ev/releases/tag/models-v0.1.0)：

```text
ev-fsmn-vad-zh-16k.tar.gz
ev-paraformer-zh-streaming-16k.tar.gz
ev-sensevoice-small.tar.gz
ev-eres2netv2-zh-16k.tar.gz
SHA256SUMS.txt
```

下载后在 Release 文件所在目录执行 `shasum -a 256 -c SHA256SUMS.txt`。由于 tar 包
内部没有顶层目录，请分别解压到对应目录：

```bash
mkdir -p data/models/ev-fsmn-vad-zh-16k data/models/ev-paraformer-zh-streaming-16k \
  data/models/ev-sensevoice-small data/models/ev-eres2netv2-zh-16k
tar -xzf ev-fsmn-vad-zh-16k.tar.gz -C data/models/ev-fsmn-vad-zh-16k
tar -xzf ev-paraformer-zh-streaming-16k.tar.gz -C data/models/ev-paraformer-zh-streaming-16k
tar -xzf ev-sensevoice-small.tar.gz -C data/models/ev-sensevoice-small
tar -xzf ev-eres2netv2-zh-16k.tar.gz -C data/models/ev-eres2netv2-zh-16k
```

`ev models verify` 会检查配置、非空权重、词表和必要配套文件。

## 处理规则

1. `AudioCapture` 持续输出 16 kHz mono 定长帧。
2. 30 ms 帧在 VAD 适配层缓冲为 200 ms chunk；保留约 600 ms pre-roll，避免切掉首字。
3. `SpeechStarted` 后以 600 ms chunk 调用 Paraformer，固定使用 `[0,10,5]` 与 4/1
   encoder/decoder look-back；partial 按 chunk 累积。
4. VAD 结束或停止监听时以 `is_final=true` flush 流式 ASR。
5. `SpeechEnded` 后将完整语音段交给串行后台 worker；采集和 VAD 继续运行。
6. worker 由 SenseVoiceSmall 开启 ITN 生成 final，再运行 ERes2NetV2、保存 WAV 和 SQLite。
7. 每段均生成 `segment_id`；任一后台阶段失败都发出 `segment_failed`。

用户注册命令为 `ev voice enroll --device <selector> --segments 8`。注册只计算并
合并 8-12 段、每段约 3-5 秒的归一化 embedding，不微调模型。profile 保存模型、
设备、样本数和版本。在线分数采用可配置三区：

```text
score >= user_threshold      user
score <= non_user_threshold  non-user
其他                         unknown
```

VUI 对文本做大小写、空白和标点标准化，只检查语音段开头的 `EV` 或配置别名。
同一段去除 EV 前缀后，只有 `wake_detected && speaker_label == user` 才设置
`query_candidate=true`。本阶段不保持多轮激活状态。

## 存储

音频写入 `data/archive/YYYY-MM-DD/<segment_id>.wav`。SQLite 的 `segments` 保存
时间、音频路径、原始/最终文本、声纹标签与分数、EV/query 状态及各模型标识；
`speaker_profiles` 保存设备绑定的用户 profile embedding。音频不写入数据库 BLOB。

## CLI

```text
ev models download [--model-root PATH]
ev models verify [--model-root PATH]
ev voice enroll [--device SELECTOR] [--segments 8] [--model-root PATH]
ev transcribe [--device SELECTOR] [--model-root PATH]
ev engine serve
```

## 开发顺序与验收

- [x] 工程脚手架与音频采集
- [x] 模型配置、目录验证器和运行时适配层
- [x] VAD 分段、pre-roll 和 hangover
- [x] 流式/终稿 ASR 接口
- [x] 声纹 enrollment、profile 与三区判断
- [x] WAV + SQLite 持久化
- [x] EV 句首匹配与 `QueryCandidate` 事件
- [x] 四模型本地加载、真实麦克风 partial/final、停止 flush、WAV 和 SQLite
- [x] 8 段声纹 profile 重启持久化
- [ ] 现场完成 `EV + user` / `EV + non-user` query 门控验收
- [ ] 声纹阈值标定
- [ ] 延迟、内存、字错率与误触发性能测试

首个 partial p50 目标为 300 ms，端点后 final p50 目标为 800 ms。两秒以上用户
语音漏检率初始目标 5%，非用户误判初始目标 1%，EV 联合误触发目标 0.1 次/小时。
这些是测量目标，不在真实数据验证前承诺达到 GPT Live 或豆包体验。
