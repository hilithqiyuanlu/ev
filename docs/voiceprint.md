# EV 声纹 (Voiceprint) Flow

> 更新时间: 2026-08-12
> 代码依据: `src/ev/speaker/profile.py`、`src/ev/speaker/verification.py`、
> `src/ev/pipeline/runtime.py`、`src/ev/engine/service.py`、`src/ev/store/db.py`。
> 整体音频链路见 [audio-flows.md](./audio-flows.md), 本文档只深入声纹部分。

## 0. 总览: 声纹在链路中的 5 个触点

```text
                        声纹模型: ERes2NetV2 (FunASR, 16kHz mono)
                        槽位: speaker (registry: eres2netv2)
                        输出: spk_embedding float32 向量 (L2 归一化后余弦比对)
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  ① 实时门控打分          ② 段内 turn 标记          ③ 段末融合判决 │
   │  OBSERVING 900ms 整窗    RECORDING 600ms 滑窗     全段 embedding  │
   │  → 初始 turn label      → speaker_turns[]        → speaker_label │
   │  (threshold-0.06 宽限)   (threshold 严格)         (threshold 严格)│
   │                                                                  │
   │  ④ 段末自动学习                     ⑤ 离线手动管理                 │
   │  三档分级 core/cache                enrollment / manual sample   │
   │  + 簇内竞争 + 自动补位              + pending 确认 + 增删/重置     │
   └──────────────────────────────────────────────────────────────────┘
```

## 1. 模型与 embedding

- **模型**: ERes2NetV2 (`iic/speech_eres2netv2_sv_zh-cn_16k-common`, FunASR AutoModel),
  注册表 key `eres2netv2`, 本地目录 `ev-eres2netv2-zh-16k`, 约 69MB。
- **调用**: `SpeakerEmbeddingAdapter.embed(audio, sample_rate=16000)` → `spk_embedding`
  (float32 一维向量)。
- **加载时机**: 随监听管线启动加载 (`transcribe_forever`), 实时路径与后台 worker
  共享同一实例; 手动录入命令单独临时加载一份。
- **比对**: `normalize_embedding` (L2 归一化) 后点积 = 余弦相似度;
  多质心取 **best score** (`verify_speaker`)。

### 打分前的响度归一化 (`normalize_loudness`)

embedding 对远近/音量敏感, 所有打分路径 (门控/滑窗/段末/学习) 统一先做段级响度校准:

- 目标 RMS = 0.05, 增益封顶 10x, 防削波 peak ≤ 0.98;
- 与采集链路的帧级 AGC 互补不冲突 (AGC 稳实时响度, 段级校准消除段间音量差);
- 开关: `speaker.loudness_normalize` (默认 true)。

## 2. Profile 数据结构

### SQLite `speaker_samples` 表

| 字段 | 说明 |
|---|---|
| `id` | uuid hex |
| `segment_id` | 来源段 (自动样本), `REFERENCES segments(id) ON DELETE CASCADE`; 手动样本为 NULL |
| `audio_path` | 样本 wav 路径 (托管目录, 见下) |
| `duration_ms` | 时长 |
| `embedding_blob` / `embedding_dim` | little-endian float32 二进制 + 维度 |
| `score` | 收录时与质心的余弦分 (手动样本记 0.95) |
| `tier` | `core` / `cache` |
| `is_manual` | 手动样本=1: 永远 core, 永不参与淘汰/降级 |
| `is_diversity` | 多样性标记 (中分自动样本=1, cache 淘汰时优先保留) |
| `created_at` | ISO 时间戳 |

### 样本音频托管目录 `data/voice-samples/`

- 自动收集: 段音频先 `save_voice_sample()` 拷贝为 `{segment_id}.wav` —— 与 segment
  存档解耦 (清空历史记录不影响样本), 且可被"重新学习"再次 embed;
- 手动录入: `archive_wav()` 写入 `voice-samples/YYYY-MM-DD/{manual|enroll}-*.wav`;
- 淘汰/删除/拒绝样本时, 仅当 wav 在托管目录内才删文件 (`_unlink_evicted` /
  `_unlink_sample_audio`)。

### 两层结构 + 多质心 (VoiceProfileManager)

| 层 | 容量 | 用途 | 淘汰 |
|---|---|---|---|
| CORE | 30 (`speaker.max_core_samples`) | K-means 建质心, 判决依据 | 超限时全局最低分**非手动**样本降级到 cache |
| CACHE | 100 (`speaker.max_cache_samples`) | 仅记录不建模 (候选池) | FIFO, 先淘普通样本, 不够再淘 diversity 样本 |

**质心数量** (`choose_k`, 按 core 样本数):

| core 样本数 | 1-5 | 6-10 | 11-18 | 19-26 | 27+ |
|---|---|---|---|---|---|
| 质心数 k | 1 | 2 | 3 | 4 | 5 (上限 `max_centroids`) |

- K-means 确定性初始化 (按向量 norm 排序等距取点), 均值建质心, 空簇用最远点重初始化;
- 多质心意义: 覆盖用户不同声学状态 (近讲/远场/感冒/环境差异), 判决取 best score;
- `is_ready = core_count >= 3` —— 这是"profile 就绪"的判据 (区别于自动学习开闸, 见 §4);
- 重建时机: 启动加载 / add_sample / promote / remove / reset 之后;
  实时路径 (runtime) 在每个段结束时 `load_centroids()` 重新加载, 保证新样本生效。

## 3. 实时打分 (采集线程, runtime.py)

统一打分函数 `_score_window(buf)`: 响度归一化 → embed → 与所有质心余弦取 max。
无 profile 时返回 1.0 (冷启动视同 user)。

### ① OBSERVING 门控 (段前, 900ms)

- VAD start 且 profile 就绪 → 进 OBSERVING (带入 pre-roll 1200ms 帧);
  门控目标 900ms, 因 pre-roll (1200ms) 已超过门控值, 进 OBSERVING 首帧即打分,
  实际打分窗口 ≈1.2s (门内 VAD-end 丢弃路径因而很难再触发, 见 §9);
- 整窗打分, 判定阈值 = `threshold - 0.06` (**宽限 margin**):
  短窗 embedding 噪声大, 宁可初始错标 user (段末全段评分会纠正),
  也不能把用户自己的声音在开口瞬间错标 non-user;
- 结果只决定**初始 turn label**; 无论 user/non-user 都会进入 RECORDING 入段
  (non-user 段事后不产生 query、不参与学习);
- 门内 VAD 结束 (<900ms 的短促声) → 静默丢弃, 不打分不开段;
- profile 未就绪 (冷启动) → 跳过门控, 直接 RECORDING 且 label=user。

### ② RECORDING 段内 turn 标记 (600ms 滑窗)

- 每 600ms 检测一次, 滑窗 = 最近 600ms processed 帧, 严格 `threshold` 判定;
- **不对称迟滞** (防单窗噪声误翻):
  - non-user → user: 1 次确认 (快速认回用户);
  - user → non-user: 连续 3 次确认;
  - 两次切换最小间隔 800ms;
- 确认切换 → 关闭当前 turn / 开新 turn, 边界对齐**滑窗中点**,
  发 `speaker_turn_changed` 事件;
- turn 只打标签**不截断段** (区别于旧版的 force_segment_end)。

### ③ 段末全段判决 + 融合 (worker 线程)

- 整段音频 → 响度归一化 → embed → `verify_speaker` → fullseg label/score;
- **融合策略** (全段 embedding 远比 600ms 滑窗稳定, 作为主信号):

| fullseg | turns 含 user | 最终 label |
|---|---|---|
| user | 任意 | **user** |
| non-user | 是 | **user** (段中切换/拼接污染) |
| non-user | 否 | **non-user** |

- 冷启动 (无质心或未就绪): label=user, score=None;
- `dominant_speaker` 与融合后的 label 对齐写库 (DB/UI 反映纠正后的判决);
- query 门控: `decide_query_from_utterances` 对唤醒词检测本身**不区分说话人**
  (per-utterance turn 标签噪声大, 曾导致相同语句随机不触发), 而是用融合后的
  段级 label (`dominant_speaker`) 决定 `query_candidate`。

### 阈值 (`speaker.threshold`, 默认 0.40)

- 三处共用: 门控 (宽限 0.06) / 滑窗 turn (严格) / 段末 (严格);
- GUI 可在 0.1-0.9 调整; `set_thresholds` 命令运行时生效, 无需重启:
  实时路径走 `shared_threshold` 跨线程字典, 段末走 `processor.threshold`。

## 4. 自动学习 (段末, worker 内)

### 收集门槛 (两层把关)

**runtime 前置** (SegmentProcessor.process):
- 融合 label = user 且 score 非 None (即 profile 已就绪);
- 段内**无任何 non-user turn** (多人段全段 embedding 被污染, 不学习);
- score ≥ `collect_min_score` (0.40)。

**VoiceProfileManager.should_collect**:
- `auto_learn` 开启;
- **onboarding 门控**: `core_count >= onboarding_target` (5) —— 手动引导完成前
  不自动学习 (即冷启动阶段不再自动收集, 靠手动录入建 profile);
- 时长 1.5s-10s, 非空、非纯语气词, 距上次收录 ≥ 30s。

### 三档分级入库 (add_sample)

| score | 去向 | 说明 |
|---|---|---|
| ≥ `core_score_min` (0.70) | **CORE** | 高置信, 参与建模; "核心未满"也不会放宽此线 |
| [0.40, 0.70) | **CACHE** + `is_diversity=1` | 中分多样性样本, 淘汰时优先保留 |
| < 0.40 | 拒收 | — |

手动样本: 永远 CORE, score 记 0.95, 不受分数/门控限制。

### 容量管理与自动补位

- **CORE 超限**: 全局最低分的非手动 core 降级到 cache (质量优先的簇内竞争近似,
  手动样本豁免);
- **CACHE 超限**: FIFO —— 先淘 `is_diversity=0`, 仍超限再淘 diversity 样本;
  托管目录内的 wav 同步删除;
- **自动补位** (`_auto_promote`): 某质心簇的 core 成员数 (与该质心相似度 ≥ 0.40 计)
  < `promote_min_members` (2) 时, 从 cache 晋升与该质心最相似且 sim ≥ 0.40 的样本;
  冷却 `promote_cooldown_sec` (60s) 防抖;
- 每次变动后重建质心, 并发射 `voice_sample_added`; 若 profile 首次就绪
  发射 `voice_profile_ready`。

## 5. 待人工确认 (pending confirmation)

- **判定**: `pending_samples()` —— cache 中与**所有**质心的最大相似度
  < `pending_distance_threshold` (0.30) 的样本;
- **语义**: 距用户声纹过远的收录, 可能是误收的他人/噪声, 也可能是新的声学变体,
  交由用户裁决;
- **操作**: confirm → `promote_sample` 晋升 core (core 满则降级最低分非手动样本);
  reject → 删除记录并删托管 wav;
- 命令: `list_pending_voice_samples` / `confirm_voice_sample` / `reject_voice_sample`;
- 事件: `pending_voice_samples` / `voice_sample_confirmed` / `voice_sample_rejected`。

## 6. 手动录入 (onboarding)

| 命令 | 行为 |
|---|---|
| `start_voice_enrollment` | 开始自由时长录入; 实时电平经 `voice_enroll_status` 回报; 若正在监听会先停监听 |
| `stop_voice_enrollment` | 结束录入 (≥0.5s), 整段 embed, score=0.95, is_manual → CORE |
| `capture_manual_sample` | 定长补录 (1.5-10s, 默认 3s), 同上; `manual_sample_status` 回报 |

- 录入样本 wav 存 `voice-samples/YYYY-MM-DD/{enroll,manual}-*.wav`;
- 手动样本永远 CORE 且不被自动淘汰 —— profile 的"锚";
- 典型引导流程: 录入 ≥5 条 → core≥5 → 判决就绪 (core≥3 时已可用) 且自动学习开闸。

## 7. Engine 命令 / 事件速查

### 命令 (stdin JSONL)

| 命令 | 作用 | 主要响应事件 |
|---|---|---|
| `list_voice_samples` | 样本列表 (可按 tier 过滤) | `voice_samples` |
| `delete_voice_sample` | 删除样本 (含托管 wav) | `voice_sample_deleted` |
| `promote_voice_sample` | cache → core 手动晋升 | `voice_sample_promoted` |
| `reset_voice_profile` | 清空全部样本 (含 speaker_profiles 行) | `voice_profile_reset` |
| `set_voice_learning` | 自动学习开关 | `profile_status` |
| `capture_manual_sample` | 定长手动补录 | `manual_sample_status` |
| `start/stop_voice_enrollment` | 自由时长录入 | `voice_enroll_status` |
| `list_pending_voice_samples` | 待确认样本列表 | `pending_voice_samples` |
| `confirm_voice_sample` | 确认为本人 (晋升 core) | `voice_sample_confirmed` |
| `reject_voice_sample` | 确认非本人 (删除) | `voice_sample_rejected` |
| `set_thresholds` | 运行时调判决阈值 (0.1-0.9) | `command_result` |

### 运行时事件

| 事件 | 时机 | 关键字段 |
|---|---|---|
| `speaker_turn_changed` | 段内说话人切换确认 | segment_id, from, to, score |
| `speaker_result` | 段末融合判决 | segment_id, label, score |
| `voice_sample_added` | 自动/手动样本入库 | segment_id, tier, core/cache/centroid_count, is_ready |
| `voice_profile_ready` | 首次 core≥3 | sample_count, core_count |
| `profile_status` | profile 变动后广播 | exists, is_ready, core_count, cache_count, auto_learn |

## 8. 参数速查 (config.py 默认值, ev.toml 已并入代码默认)

| 参数 | 默认 | 说明 |
|---|---|---|
| `speaker.threshold` | 0.40 | 判决阈值 (门控宽限 0.06) |
| `speaker.max_core_samples` | 30 | CORE 容量 |
| `speaker.max_cache_samples` | 100 | CACHE 容量 |
| `speaker.max_centroids` | 5 | 质心数上限 |
| `speaker.loudness_normalize` | true | 打分前响度校准 |
| `voice_learning.collect_min_score` | 0.40 | 收录最低分 |
| `voice_learning.core_score_min` | 0.70 | 进 CORE 的分数线 |
| `voice_learning.pending_distance_threshold` | 0.30 | 待确认判定距离 |
| `voice_learning.onboarding_target` | 5 | 自动学习开闸所需 core 数 |
| `voice_learning.promote_min_members` | 2 | 簇内补位触发线 |
| `voice_learning.promote_cooldown_sec` | 60 | 补位冷却 |
| `voice_learning.min/max_duration_ms` | 1500 / 10000 | 收录时长窗口 |
| `voice_learning.min_interval_sec` | 30 | 两次收录最小间隔 |
| `voice_learning.auto_learn_enabled` | true | 自动学习总开关 |
| 实时: 门控 / 滑窗 / 确认 / 间隔 | 900ms / 600ms / 1或3次 / 800ms | runtime.py 常量 |

## 9. 设计权衡与已知边界

1. **滑窗 embedding 噪声大** → 600ms turn 只做标签, 最终判决以全段融合为准;
   门控给 0.06 宽限同理 (开口瞬间 embedding 最不稳)。
2. **多人段不学习**: 含 non-user turn 的段, 全段 embedding 已被污染, 跳过收录。
3. **门控不拦长段**: ≥900ms 的非用户讲话仍会开段录入, 靠融合判决保证
   不产生 query、不参与学习 (隐私/存储层面待后续收紧)。
4. **segment 删除的级联**: DB 行级联删除 (`ON DELETE CASCADE`), 但自动样本在
   托管目录的 wav **不会**随之清理 (孤儿文件); 只有 delete/reject/淘汰路径会删文件。
5. **阈值全局共用**: 门控/滑窗/段末用同一 threshold, 调参时三处同时受影响。
6. **质心数上限实际为 5**: `choose_k` 按 core 样本数分档 (27+ 才到 5),
   `max_centroids` 只是钳制上限。
7. **冷启动行为**: core<3 时所有段判 user (score=None)、不自动学习;
   query 需唤醒词且唤醒后 ≥2 字。
8. **OBSERVING 门控已被 pre-roll 覆盖**: pre-roll 1200ms > 门控 900ms,
   门控在首帧即触发 (打分窗口 ≈1.2s), "门内 VAD 结束即丢弃"的短噪声过滤
   实际不再生效 —— 短噪声改由 min_duration(500ms)/人声确认/empty-filler
   等后段环节过滤。
