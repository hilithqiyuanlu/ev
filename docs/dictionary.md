# EV 词典/热词系统

> 状态:基于当前工作区(含未提交改动)逐一核实。上次同步:2025-08-12

## 1. 词典是什么

SQLite `lexicon` 表(Store 管理):

| 字段 | 说明 |
|---|---|
| `word` | 词条(唯一,`INSERT OR REPLACE` 按 word 去重) |
| `weight` | 0.5–10,手动默认 3.0、自动 1.5、系统词固定 5.0 |
| `source` | `system` / `manual` / `auto` |
| `use_count` | 命中计数(当前无写入路径) |
| `created_at` / `updated_at` | 时间戳 |

三种来源:
- **system**:系统词(如唤醒词 小E),`_seed_system_words` 注入,固定 5.0,优先级最高
- **manual**:macOS 词典页手动添加
- **auto**:自动学习(当前**已禁用**,`learn_high_frequency_words` 直接 `return 0`,无调用方)

## 2. 数据流(按当前实际代码)

```
lexicon 表
  ├─ get_hotwords_string()   → SegmentWorker → SenseVoiceAdapter.transcribe(hotword=…)
  └─ get_hotword_entries()   → SegmentWorker → 仅 Qwen3 路径(Qwen3ASRAdapter)消费
```

Worker 启动(`runtime.py:731`):同时取 `get_hotwords_string()` 和 `get_hotword_entries()` 注入 `SegmentProcessor`;词典增删改经 `engine/service._broadcast_hotwords()` → `SegmentWorker.update_hotwords()` 热更新,runtime 无需重启。

- `get_hotwords_string()`(db.py:933):空格分词、**不带权重**(`:3.0` 后缀会破坏 FunASR 的 seg_tokenize 使其整体变 `<unk>`)、含 system/manual/auto,按 system>manual>auto 排序、weight DESC,上限 80。
- `get_hotword_entries()`(db.py:956):`(word, weight)` 列表,**排除 system 词**(唤醒词由 vui.py 规则匹配,不注入 decode bias),manual>auto、weight DESC,上限 80。

## 3. 终稿 ASR 的两条消费路径

当前配置(`config.py` 未提交改动):`default_slots["asr_final"] = "sensevoice-small"`,models-v0.1.0.json 的 `asr_final` 资产即 SenseVoice。模型槽位由 `model_registry` 运行时决定,`_final()` 按 adapter 类型分发:

### 路径 A:FunASR(SenseVoice,当前默认)
`adapter.transcribe(audio, rate, hotword=self.hotwords)` → `kwargs["hotword"] = hotword` → `AutoModel.generate(**kwargs)`。

**关键核实(代码级)**:SenseVoice 的 hotword 参数**完全不生效**:
- FunASR `auto_model.py` 只对带 `hotword_as_bias` 方法的模型(bias 注入)消费 hotword;逐文件核实,`paraformer/model.py`, `sense_voice/model.py`, `contextual_paraformer` 中**没有** `def hotword_as_bias`(该方法是早前版本 API,当前安装的 FunASR 已移除)。
- `SenseVoiceSmall.inference` 签名 `(self, data_in, data_lengths, key, tokenizer, frontend, kwargs)`,函数体内 `hotword` 出现次数 = 0 —— 传入的 hotword 落在 `kwargs` 里被直接忽略,既不参与解码也不报错。
- 结论:**SenseVoice 路径下加词典词对转写结果零影响**,且全程无声。这解释了"加了词像没加"。

### 路径 B:Qwen3(引擎仍支持,需手动切槽位)
`Qwen3ASRAdapter.transcribe` 接收 `hotword`(prompt)与 `hotword_entries`(logits),两条增强:
1. **Prompt 注入**:仅注入 `hotword` 字符串(来自 `get_hotwords_string()`,**含 system 词**)。配置的 `hotword_inject_max_words=30` **从未被读取**(qwen3_adapter 里该字段 0 引用)→ 广播的 80 词全量进 prompt。
2. **锚定式 logits 增强**(`asr/hotword.py`):仅当已解码文本尾部命中词典前缀时,对续写该词 token 加 `weight × boost_scale`,封顶 `boost_max=4.0`;只加不抑制。启用 by `AsrSettings.hotword_boosting_enabled`。

## 4. 幻觉保护(已移除)

上一版有"热词过载检测"(avg_logprob<-2.0 且热词密度>55% → 整段拒收),**当前工作区已删除**(git diff 确认 `-` 行),`_compute_hotword_coverage` 成为未被调用的死代码。Qwen3 的 logits 上限也从 6 降到 4(config 内注释: 防弱信号幻觉)。

## 5. 当前可用性问题清单

### P0 — SenseVoice 下词典完全不生效
默认就是 SenseVoice(未提交改动大背景),hotword kwarg 在 FunASR 侧被静默忽略。用户加词、清空、调权重,对识别零影响且无任何日志提示。

### P0 — 双路径不一致,调参者极易踩空
Qwen3 的参数(prompt 注入、logits boost_scale/max/min_anchor、hotword_entries 增强)在 SenseVoice 上全部无效;而 SenseVoice 是当前唯一实际加载的终稿。用户照着 Qwen3 思路调,白调。

### P1 — Qwen3 路径机制本身残缺
- `hotword_inject_max_words=30` 死配置(从未读取)。
- prompt 注入用含 system 词的 `get_hotwords_string()`,而 logits 增强用排除 system 词的 `get_hotword_entries()` —— 同一份词典在 Qwen3 路径下被两种不一致的规则消费。

### P1 — 无命中统计,召回不可观测
- segments 表无 hotword 命中/覆盖字段;
- `use_count` 无任何写入路径;
- `_compute_hotword_coverage` 已无人调用。用户无法知道哪个词命中/失败。

### P1 — 纠错学习是假功能
- 引擎 `_learn_corrections` 返回空(`added:0`);
- Swift `learnCapsule`(从纠错学习按钮)在 `LexiconView.headerBar` 中**未被引用** → 死代码;但 `learnCorrections()` handlers 仍在 AppModel/service 里保留。

## 6. 建议方向

1. **明确选定终稿路径再实现**。要么把默认切回支持 bias 的 Paraformer/重新引入 SenseVoice 的 hotword 支持(hotword 需走 `postprocess_hotwords` 这类文本级修正或自定义 bias),要么默认 Qwen3 并修 Qwen3 路径。
2. **让 SenseVoice 的词典"至少做点事"**:可用 FunASR 的 `postprocess_hotwords`(文本级后处理替换,与解码热词不同)做近似兜底,或在前端把词典词显著化。
3. **修 Qwen3 死配置**:让 `hotword_inject_max_words` 真正生效;统一 prompt 注入与 logits 增强的取词规则。
4. **可观测性**:给 segments 加 hotword 命中/覆盖字段,UI 展示每个词命中率,形成「加词 → 验证 → 调整」闭环。
5. **清理死代码**:移除或接线 learnCapsule;删除/实现 `_compute_hotword_coverage`。