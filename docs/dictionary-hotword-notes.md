# 词典/热词优化开发记录

> 更新：2026-08-13。详细实际行为见 [dictionary.md](./dictionary.md)。

## 现状（2026-08-13 更新）

- 终稿 ASR = Fun-ASR-Nano-2512（speech-LLM，解码器 Qwen3-0.6B）。
- 已改为证据触发：流式 ASR 无热词，使用流式文本检索 active 非系统词条，终稿最多接收 8 个候选。
- 候选选择位于独立 `ev.lexicon` 模块，采用 NFKC、完整命中和长度分级的有序字符覆盖规则。
- SQLite v19 分离 `source` 与 `status`；自动词待人工确认，确认后仍保留 auto 来源。
- 每段保存实际候选、最终命中和覆盖率；`use_count` 只统计实际传入且命中的段数。

## 已核实：Fun-ASR-Nano 热词接口

- 官方 README 用法：`generate(input=..., hotwords=["词1","词2"], language="中文", itn=True)`。
- 参数名是 `hotwords`（**复数**，注意 `generate` docstring 误写单数 `hotword`，model 层实际读 `kwargs.get("hotwords")`）。
- **权重不参与解码**：`get_prompt` 只收纯字符串列表，权重退化为排序（weight DESC 靠前）。

## 本轮测试

- 候选归一化、长度阈值、排序、去重、上限 8 和第 81 个词可检索。
- pending/disabled/system 隔离、auto 确认保留来源、v18 到 v19 迁移。
- 无证据不传热词、候选与命中集合约束、每段计数去重。
- 引擎确认、拒绝、启停命令和 macOS 数据解析/命令发送。

## 后续方向

1. **第三级拼音兜底（暂缓）**：funasr 原生支持 `postprocess_hotwords` 参数（文本级纠错，`apply_postprocess_hotwords_to_results` 在 `generate` 内自动应用）。原 `asr/hotword.py` 已删。等第一级实测看命中率再决定。
2. **language 提示**：README 示例传 `language="中文"`；当前未传（`get_prompt` 默认走「语音转写」分支）。如需方言/中文优化可加。
3. **不做的第二级**：logits 锚定 — Fun-ASR-Nano 黑盒 `generate`，不暴露逐 token logits。

4. **效果评估**：建立带无热词对照集，区分自然命中与热词带来的真实改善。
