"""环境事件日志 — 纯文本 JSONL 存储，不入 SQLite，不入 WAV。

每个环境事件仅存储时间戳、类别、置信度、持续时间。
按日期分文件，支持时间范围查询。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


_ENV_LOG_PREFIX = "env"


class EnvironmentLog:
    """环境事件日志。

    格式 (每行):
        {"ts": 1754971385.123, "category": "typing",
         "confidence": 0.72, "duration_sec": 23.5}

    文件:
        data/logs/env-2026-08-11.jsonl
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _filename(self, date_str: str | None = None) -> str:
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"{_ENV_LOG_PREFIX}-{date_str}.jsonl"

    def _path(self, date_str: str | None = None) -> Path:
        return self.log_dir / self._filename(date_str)

    # ── 写入 ──────────────────────────────────────────────────────────

    def append(self, event) -> None:
        """追加一条环境事件。

        Args:
            event: EnvEvent 实例 (来自 ev.audio.environment)。
        """
        line = json.dumps(
            {
                "ts": event.timestamp,
                "category": event.category,
                "confidence": event.confidence,
                "duration_sec": event.duration_sec,
            },
            ensure_ascii=False,
        )
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── 查询 ──────────────────────────────────────────────────────────

    def query(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict]:
        """按时间范围查询环境事件。

        Args:
            start_time: Unix 时间戳下限（含）。
            end_time: Unix 时间戳上限（含）。

        Returns:
            事件列表，按时间升序排列。
        """
        results: list[dict] = []
        # 扫描可能相关的日期文件（简化：扫描 log_dir 下所有 env-*.jsonl）
        for path in sorted(self.log_dir.glob(f"{_ENV_LOG_PREFIX}-*.jsonl")):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        ts = record.get("ts")
                        if ts is None:
                            continue
                        if start_time is not None and ts < start_time:
                            continue
                        if end_time is not None and ts > end_time:
                            continue
                        results.append(record)
            except OSError:
                continue

        results.sort(key=lambda r: r.get("ts", 0))
        return results

    def query_summary(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, float | list[str]]:
        """按时间范围查询环境摘要（用于与语音段关联）。

        Returns:
            {"dominant_category": "typing", "categories": [...],
             "average_confidence": 0.68, "event_count": 3}
        """
        records = self.query(start_time, end_time)
        if not records:
            return {}

        categories = [r.get("category", "unknown") for r in records]
        confidences = [r.get("confidence", 0) for r in records]

        # 主导类别 = 出现最多
        cat_counts: dict[str, int] = {}
        for c in categories:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        dominant = max(cat_counts, key=lambda k: cat_counts[k])

        return {
            "dominant_category": dominant,
            "categories": list(dict.fromkeys(categories)),  # 保持顺序去重
            "average_confidence": (
                round(sum(confidences) / len(confidences), 3)
                if confidences
                else 0.0
            ),
            "event_count": len(records),
        }
