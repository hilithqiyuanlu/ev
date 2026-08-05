# EV

个人语音助手。**Always-on,本地部署,不上云**。北极星:像电影里那样的贴身 AI 伙伴——听得清、记得住、插得上话、知道什么时候该闭嘴。

## 硬件

| 设备 | 角色 |
|---|---|
| MacBook Pro M4 16GB | 常开前端:拾音、VAD、ASR、交互界面 |
| Win 台式机(5060Ti 16GB / 32GB RAM) | 推理后端:本地 LLM、后期 TTS |
| DJI Mic Mini | 近场拾音,机内降噪(接收器 USB-C 连接) |
| iPhone 16 Pro Max | (后期)Phase 1c 移动采集前端:采集+回传,不做计算 |
| 智能眼镜 | (后期)Phase 2 视觉采集,到时复用成熟产品 |
| 可穿戴(Apple Watch / Garmin / Oura / 肌电) | (后期)Phase 3 生物信号:心率/HRV/血氧/体温 |

## 分层架构

| 层 | 职责 | 现状 |
|---|---|---|
| 硬件层 | 全天候麦克风(领夹)→ 智能眼镜 CV → 可穿戴生物信号 | DJI Mic Mini(暂用内置麦开发) |
| 感知层 | 声音(VAD/ASR/标点/声纹/环境音)、视觉、生物信号 | Phase 1a:声音感知开发中 |
| 认知层 | 对话状态机(级联→半级联→端到端)、记忆(工作/情景/语义)、推理(演绎/归纳/溯因)、情绪模型(多模态) | NLP 本地双档排期中 |
| 行为层 | 响应生成、流式 TTS、主动行为(预警/静默) | Phase 1b 起 |

层间只传流式事件(事件总线),与前端/后端分离一致:任何一层可独立替换 —— 研究消融的地基。

## 路线图

- **Phase 1a(当前)**:ASR 转写 + 用户声纹门控(NLP 本地双档后置)→ [docs/phase1a-plan.md](docs/phase1a-plan.md)
- Phase 1b:TTS 播报(CosyVoice2 流式,克隆 EV 声线)
- Phase 1c:全天候(KWS 唤醒)+ barge-in 打断
- Phase 2:环境感知(环境音 + CV,智能眼镜/麦克风阵列/人在传感器)
- Phase 3:身体感知(可穿戴生物信号 + 多模态情绪)

## 原则

1. 工程尽量外包给成熟方案:sherpa-onnx(VAD/ASR/标点/声纹)、Ollama(本地 LLM)、CosyVoice2(TTS,后期)。
2. 全链路延迟从第一天起可测量,先测量后优化。
3. 原始数据(VAD 门控的语音段音频 + 逐字稿)append-only 留存,模型升级后可重挖。
4. 数据与推理全本地,不上云;备份走本地/自托管(NAS 等)。

## 开发

uv + Python 3.11(src 布局)。

```sh
uv sync                            # 建环境
uv run pytest                      # 测试
uv run python -m ev info           # 配置/路径
uv run python -m ev audio devices  # 输入设备枚举
uv run python -m ev audio test     # 采集自检(保存 wav 供回听)
```

任务清单与进度见 [docs/phase1a-plan.md](docs/phase1a-plan.md);研究定位见 [docs/research.md](docs/research.md)。
