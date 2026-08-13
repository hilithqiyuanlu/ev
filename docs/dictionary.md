# 词典与热词系统

> 当前实现：2026-08-13，SQLite schema v19。

## 识别流程

流式 ASR 始终无热词运行。段结束后，系统使用最终流式文本作为首遍证据，
从所有已确认且启用的非系统词条中筛选候选，最多向终稿 ASR 传入 8 个词。
没有可靠证据时，终稿 ASR 不携带 `hotwords`。

```text
流式 ASR（无热词）
  -> 流式文本证据
  -> 全量 active 非系统词条检索
  -> 最多 8 个候选
  -> Fun-ASR-Nano 终稿
  -> 候选与命中统计
```

## 候选规则

匹配前对证据和词条执行 Unicode NFKC、大小写归一化，并移除空白、标点和符号。
归一化后不足 2 个字符的证据或词条不参与匹配。

- 完整子串命中：分数为 1.0。
- 2 字词：只接受完整命中。
- 3 至 4 字词：滑动窗口内有序字符覆盖率至少 75%。
- 5 字及以上：至少匹配 3 个有序字符，且覆盖率至少 60%。
- 权重仅在证据同分时排序，不能降低证据阈值。
- 排序依次比较证据分数、匹配字符数、权重、词条 ID，结果稳定且最多 8 个。

当前不使用拼音、语义向量或终稿文本后处理。

## 词条状态

`source` 表示来源，`status` 表示是否允许参与候选检索，两者相互独立。

| 字段 | 值 | 含义 |
|---|---|---|
| `source` | `manual` / `auto` / `system` | 人工添加、纠错学习、系统内置 |
| `status` | `pending` / `active` / `disabled` | 待确认、已启用、已停用 |
| `confirmed_at` | ISO 时间或空 | 人工确认时间 |

人工词创建后直接 active；自动学习词创建后为 pending。确认自动词只改为 active，
仍保留 `source=auto`。拒绝单个自动词会将其设为 disabled，以避免反复建议；
“清空自动词”会永久删除全部 auto 词。system 词不参与终稿热词候选。

v18 升级 v19 时，旧 manual/system 回填 active，旧 auto 回填 pending；旧 manual 的
`confirmed_at` 使用 `created_at`。

## 统计口径

- `segments.hotword_candidates`：本段实际传入终稿 ASR 的候选及证据 JSON。
- `segments.hotword_hits`：上述候选中最终文本实际出现的词。
- `segments.hotword_density`：最终文本中被实际候选命中覆盖的归一化字符比例。
- `lexicon.use_count`：词条“实际传入且最终命中”的段数，同一段最多增加一次。

`use_count` 是相关性观测，不代表热词造成了识别改善，也不是纠正成功次数。
质量拒绝段不计热词命中。

## 引擎协议

- `list_lexicon`
- `add_lexicon_word`
- `update_lexicon_word`，旧 `promote_to_manual=true` 兼容映射为确认，不再改变来源
- `confirm_lexicon_word {id}`
- `reject_lexicon_word {id}`
- `set_lexicon_word_status {id,status}`，status 只接受 active 或 disabled
- `delete_lexicon_word`
- `clear_auto_lexicon`
- `learn_corrections`

词典变更会向运行中的段落 worker 热更新全部 active 非系统词条。每段候选仍由该段
流式文本实时决定，不存在全量热词 prompt。

## 已知限制与后续事项

- 字形/顺序证据无法覆盖完全同音但字形完全不同的首遍错误。
- `use_count` 不能单独用于评估热词的因果收益；后续需要带对照的识别评测。
- 拼音兜底、语义检索和 `postprocess_hotwords` 暂不实现，先观测误触发率和召回率。
