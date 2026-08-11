# 声纹自动学习（Voice Learning）设计

> 更新时间: 2026-08-11

声纹样本的分级采集、簇内竞争淘汰、自动补位与人工确认机制。

## 样本分级（add_sample 三档）

自动采集的样本按声纹打分 `score` 三档入库：

| 分数区间 | 去向 | 说明 |
|---|---|---|
| `score >= core_score_min` (0.70) | **core** | 高置信，参与质心建模 |
| `collect_min_score` (0.40) ≤ score < 0.70 | **cache + is_diversity=1** | 中低分，可能是特殊状态/环境变体，标记 diversity 供人工确认或保留 |
| `score < collect_min_score` (0.40) | 不收 | 太低置信，直接丢弃 |

- 手动样本永远进 core，不受分数限制，**永不参与淘汰**。
- "核心未满也只收 ≥ core_score_min" —— 低分样本不因核心有空位就进核心。

> 背景: 之前 `collect_min_score=0.60` 导致正常说话的 0.40~0.60 段被"非采集"挡掉、
> 永不入库 → 核心样本永远停在引导门控（onboarding_target=5）附近。降到 0.40 后
> 这部分进入缓存，再由三档分级 + 人工确认决定去向。

## 核心淘汰（簇内竞争）

`_trim_core_overflow`：core 总数超过 `max_core_samples` 时，降级分数最低的
**非手动** core 样本到 cache。全局降级最低分 ≈ 各簇内保留高分代表（质量优先）。

## 缓存淘汰（FIFO + diversity 保护）

`Store.evict_oldest_cache`：普通缓存样本（is_diversity=0）先淘汰，只有仍超限才
淘汰 diversity 样本。diversity 样本优先保留。

## 自动补位（_auto_promote）

某簇核心成员数 < `promote_min_members` (2) 时，从缓存晋升该簇最高分样本到 core。
受 `promote_cooldown_sec` (60s) 冷却限制，避免频繁抖动。在 `add_sample` 后统一触发。

## 待确认样本（pending）

`VoiceProfileManager.pending_samples(threshold)`：缓存中与所有核心质心**最大相似度
< pending_distance_threshold** (0.30) 的样本进入待确认列表（可能是误收录的他人/噪声，
也可能是新的声学变体）。UI 提供：
- **确认** → `promote_sample` 晋升核心
- **删除** → `remove_sample` 移除

引擎命令：`list_pending_voice_samples` / `confirm_voice_sample` / `reject_voice_sample`。

## 配置项（config.py VoiceLearningSettings / SpeakerSettings）

```toml
[speaker]
max_core_samples = 30
max_cache_samples = 100
max_centroids = 5

[voice_learning]
collect_min_score = 0.40
core_score_min = 0.70
pending_distance_threshold = 0.30
promote_min_members = 2
promote_cooldown_sec = 60.0
```

`choose_k` 档位（verification.py）：≤5→1、6-10→2、11-18→3、19-26→4、≥27→5。

## 已知边界

- `_trim_core_overflow` 是全局降级最低分，并非严格按"argmax 最近质心定簇"逐簇竞争；
  在手动样本占比高时，自动样本可能因竞争不过手动而被降级，属预期（手动永不淘汰）。
- pending 判定基于与质心的余弦相似度，未做阈值校准；不同说话距离/口音下边界值需实测。
- 自动补位仅在下一次 `add_sample` 时触发，长时间无新样本时不会自愈空簇。
