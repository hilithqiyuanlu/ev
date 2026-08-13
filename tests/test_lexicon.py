from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from ev.lexicon import LexiconEntry, normalize_for_matching, select_hotword_candidates
from ev.store.db import Store


def entry(word: str, *, id: str | None = None, weight: float = 3.0,
          source: str = "manual", status: str = "active") -> LexiconEntry:
    return LexiconEntry(id=id or word, word=word, weight=weight, source=source, status=status)


def words(evidence: str, entries: list[LexiconEntry], limit: int = 8) -> list[str]:
    return [item.word for item in select_hotword_candidates(evidence, entries, limit=limit)]


def test_normalization_and_empty_evidence():
    assert normalize_for_matching("Ｖｉｂｅ， Coding!") == "vibecoding"
    assert words("", [entry("网易云")]) == []
    assert words("网", [entry("网易云")]) == []


def test_two_character_terms_require_exact_match():
    assert words("我想找张山", [entry("张三")]) == []
    assert words("联系张三", [entry("张三")]) == ["张三"]


def test_ordered_matching_uses_length_specific_thresholds():
    short = select_hotword_candidates("请打开网易音乐云", [entry("网易云")])
    assert short[0].match_type == "ordered"
    assert short[0].matched_chars == 3
    assert words("研究强化习", [entry("强化学习")]) == ["强化学习"]
    assert words("研究强学", [entry("强化学习")]) == []


def test_filters_status_source_deduplicates_and_limits_stably():
    entries = [
        entry(f"测试词{i}", id=f"id-{i:02}", weight=float(i)) for i in range(10)
    ] + [
        entry("测试词0", id="pending", status="pending"),
        entry("测试词0", id="disabled", status="disabled"),
        entry("测试词0", id="system", source="system"),
    ]
    selected = select_hotword_candidates(" ".join(item.word for item in entries), entries)
    assert len(selected) == 8
    assert [item.entry_id for item in selected] == [f"id-{i:02}" for i in range(9, 1, -1)]


def test_normalized_duplicate_terms_select_only_best_entry():
    selected = select_hotword_candidates("VIBE CODING", [
        entry("Vibe Coding", id="lower", weight=2.0),
        entry("ｖｉｂｅ coding", id="higher", weight=4.0),
    ])
    assert [item.entry_id for item in selected] == ["higher"]


def test_all_active_terms_are_searchable_beyond_eighty(tmp_path):
    with Store(tmp_path / "ev.db") as store:
        for index in range(81):
            store.add_lexicon_word(f"候选词{index:02}", weight=10 - index / 100)
        entries = store.get_active_lexicon_entries()
        assert len(entries) == 81
        selected = words("请使用候选词80", entries)
        assert "候选词80" in selected


def test_auto_term_confirmation_preserves_source_and_status(tmp_path):
    with Store(tmp_path / "ev.db") as store:
        item = store.add_lexicon_word("自动候选", source="auto")
        assert item["status"] == "pending"
        assert store.get_active_lexicon_entries() == []
        assert store.confirm_lexicon_word(item["id"])
        confirmed = next(row for row in store.list_lexicon() if row["id"] == item["id"])
        assert confirmed["source"] == "auto"
        assert confirmed["status"] == "active"
        assert confirmed["confirmed_at"]
        assert [value.word for value in store.get_active_lexicon_entries()] == ["自动候选"]
        assert store.reject_lexicon_word(item["id"])
        assert store.get_active_lexicon_entries() == []
        assert store.set_lexicon_word_status(item["id"], "active")
        assert [value.word for value in store.get_active_lexicon_entries()] == ["自动候选"]


def test_enabling_rejected_pending_auto_term_records_confirmation(tmp_path):
    with Store(tmp_path / "ev.db") as store:
        item = store.add_lexicon_word("待确认词", source="auto")
        assert store.reject_lexicon_word(item["id"])
        assert store.set_lexicon_word_status(item["id"], "active")
        row = next(value for value in store.list_lexicon() if value["id"] == item["id"])
        assert row["status"] == "active"
        assert row["confirmed_at"]


def test_v18_to_v19_migration(tmp_path):
    db = tmp_path / "legacy.db"
    now = datetime.now(timezone.utc).isoformat()
    connection = sqlite3.connect(db)
    connection.executescript("""
        CREATE TABLE lexicon (
          id TEXT PRIMARY KEY, word TEXT NOT NULL UNIQUE, weight REAL NOT NULL,
          source TEXT NOT NULL, use_count INTEGER NOT NULL, created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE segments (
          id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
          duration_ms INTEGER NOT NULL, audio_path TEXT NOT NULL, sample_rate INTEGER NOT NULL,
          channels INTEGER NOT NULL, transcript_raw TEXT NOT NULL, transcript_final TEXT NOT NULL,
          speaker_label TEXT NOT NULL, wake_detected INTEGER NOT NULL, query_candidate INTEGER NOT NULL,
          vad_model TEXT NOT NULL, asr_final_model TEXT NOT NULL, speaker_model TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        PRAGMA user_version=18;
    """)
    connection.executemany(
        "INSERT INTO lexicon VALUES (?, ?, 3.0, ?, 0, ?, ?)",
        [("m", "人工词", "manual", now, now), ("a", "自动词", "auto", now, now),
         ("s", "系统词", "system", now, now)],
    )
    connection.commit()
    connection.close()

    with Store(db) as store:
        rows = {row["id"]: row for row in store.list_lexicon()}
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 19
        assert rows["m"]["status"] == "active" and rows["m"]["confirmed_at"] == now
        assert rows["a"]["status"] == "pending" and rows["a"]["confirmed_at"] is None
        assert rows["s"]["status"] == "active" and rows["s"]["confirmed_at"] is None
        assert "hotword_candidates" in {
            row["name"] for row in store.connection.execute("PRAGMA table_info(segments)")
        }


def test_record_hotword_hits_counts_each_word_once(tmp_path):
    with Store(tmp_path / "ev.db") as store:
        store.add_lexicon_word("网易云")
        store.record_hotword_hits(["网易云", "网易云", "不存在"])
        item = next(row for row in store.list_lexicon() if row["word"] == "网易云")
        assert item["use_count"] == 1
