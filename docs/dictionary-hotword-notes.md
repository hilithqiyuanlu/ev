# 词典/热词优化方向备忘（后置待办）

> 记录：2026-08-13。本轮词典开发后置，此笔记供后续轮次重看，不展开实现。
> 详细现状见 [dictionary.md](./dictionary.md)（其中 SenseVoice/Qwen3 部分已过时，终稿已换 Fun-ASR-Nano-2512）。

## 一句话现状

终稿 ASR 已换成 Fun-ASR-Nano-2512，但热词**零生效**：

- `FunASRNanoAdapter.transcribe()` 调 `model.generate(input=tensor)`，**不传热词**；
- 原 `asr/hotword.py`（含 `postprocess_hotwords` 拼音纠错）**已删**（只剩 .pyc 残留）；
- 热词当前只被收集 + 统计（`hotword_density`/`hotword_hits` 覆盖率），对转写结果无影响。

## 讨论定下的方向（同事「三级增强」裁到两级）

- **第一级 hotword 透传（必做）**：`transcribe(audio, sr)` 加 `hotwords` 参数 → 透传 `generate`。Fun-ASR-Nano 唯一值得做的路径，收益最大、改动最小。
- **第二级 logits 锚定（不做）**：Fun-ASR-Nano 经 funasr 调用是黑盒 `generate`，不暴露逐 token logits，无法低成本迁移；「权重作用于 logits」在 prompt 注入方式下不成立。
- **第三级 拼音兜底（暂缓）**：原 `postprocess_hotwords` 已删、需重写；等第一级跑通看效果再决定。

## 待查证（下轮开工前先做）

1. Fun-ASR-Nano `generate` 的热词传参方式：官方 `hotword=` 参数 vs 拼 prompt 前缀（读已下载模型推理代码 + funasr 接口）。
2. 权重（0.5–10）在第一级怎么落地：若只认字符串列表不认权重，权重退化为「排序 + 统计」，需决定保留程度。

## 复用不动的部分

`lexicon` 表、三种来源（system/manual/auto）、热更新 `_broadcast_hotwords`、`hotword_density`/`use_count` 统计——全模型无关，保留。
