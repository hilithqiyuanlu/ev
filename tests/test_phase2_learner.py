"""Phase 2b: Conservative Error-driven Lexicon Learning tests.

The new learner is intentionally simple and predictable:
- Takes the EXACT difference region between ASR and correction (no n-gram guessing)
- Adds on first correction (min_corrections=1)
- Filters: small edits (<10% diff), complete rewrites (<30% similarity),
  common words, stopwords, single chars, words already present in ASR.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_learner.db"


def _add_segment(store, seg_id: str, text: str) -> None:
    from ev.store.db import SegmentRecord
    now = datetime.now(timezone.utc).isoformat()
    store.insert_segment(SegmentRecord(
        seg_id, now, now, 1500, "/tmp/a.wav", 16000, 1,
        text, text, "user", 0.85,
        False, False, None, "vad", "stream", "final", "speaker", now,
    ))


def test_single_correction_adds_word(db_path: Path):
    """A single correction (replacement error) should add the word immediately."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "我喜欢外部扣鼎")
        store.record_correction(
            segment_id="seg1",
            asr_text="我喜欢外部扣鼎",
            corrected_text="我喜欢强化学习",
            source="manual_edit",
        )
        added = store.learn_from_corrections()
        assert "强化学习" in added
        row = store.connection.execute(
            "SELECT * FROM lexicon WHERE word=?", ("强化学习",)
        ).fetchone()
        assert row is not None
        assert row["source"] == "auto"
        assert abs(float(row["weight"]) - 2.0) < 0.01
        assert int(row["use_count"]) == 1


def test_single_insertion_char_filtered(db_path: Path):
    """Inserting a single character (e.g. adding one missing char) should be filtered
    because the exact diff is only 1 char - not a useful hotword."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "我想研究强化学")
        store.record_correction(
            segment_id="seg1",
            asr_text="我想研究强化学",
            corrected_text="我想研究强化学习",
            source="manual_edit",
        )
        added = store.learn_from_corrections()
        assert added == []
        unapplied = store.connection.execute(
            "SELECT COUNT(*) as c FROM correction_history WHERE is_applied=0"
        ).fetchone()["c"]
        assert unapplied == 0


def test_small_edit_skipped(db_path: Path):
    """Small edits (adding a particle like 啊/了) should be skipped (>90% similarity)."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "今天天气真好")
        store.record_correction(
            segment_id="seg1",
            asr_text="今天天气真好",
            corrected_text="今天天气真好啊",
            source="manual_edit",
        )
        added = store.learn_from_corrections()
        assert added == []


def test_complete_rewrite_skipped(db_path: Path):
    """Complete rewrites (<30% similarity) should be skipped."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "我想吃饭")
        store.record_correction(
            segment_id="seg1",
            asr_text="我想吃饭",
            corrected_text="明天下午三点开会讨论项目进度",
            source="manual_edit",
        )
        added = store.learn_from_corrections()
        assert added == []


def test_english_phrase_supported(db_path: Path):
    """English phrases with spaces like 'vibe coding' should be learnable."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "我喜欢外部扣鼎")
        store.record_correction(
            segment_id="seg1",
            asr_text="我喜欢外部扣鼎",
            corrected_text="我喜欢vibe coding",
            source="manual_edit",
        )
        added = store.learn_from_corrections()
        assert "vibe coding" in added
        row = store.connection.execute(
            "SELECT * FROM lexicon WHERE word=?", ("vibe coding",)
        ).fetchone()
        assert row is not None
        assert row["source"] == "auto"
        assert abs(float(row["weight"]) - 2.0) < 0.01


def test_corrections_not_reprocessed(db_path: Path):
    """After learning, corrections are marked applied and not reprocessed."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "我喜欢外部扣鼎")
        store.record_correction(
            segment_id="seg1",
            asr_text="我喜欢外部扣鼎",
            corrected_text="我喜欢强化学习",
            source="manual_edit",
        )
        first_added = store.learn_from_corrections()
        assert "强化学习" in first_added
        second_added = store.learn_from_corrections()
        assert second_added == []


def test_existing_manual_word_not_modified(db_path: Path):
    """If a word already exists as manual/system, don't change its weight."""
    from ev.store.db import Store
    with Store(db_path) as store:
        store.add_lexicon_word("强化学习", weight=3.0, source="manual")
        _add_segment(store, "seg1", "我喜欢外部扣鼎")
        store.record_correction(
            segment_id="seg1",
            asr_text="我喜欢外部扣鼎",
            corrected_text="我喜欢强化学习",
            source="manual_edit",
        )
        added = store.learn_from_corrections()
        assert "强化学习" not in added
        row = store.connection.execute(
            "SELECT weight FROM lexicon WHERE word=?", ("强化学习",)
        ).fetchone()
        assert abs(float(row["weight"]) - 3.0) < 0.01


def test_hotwords_reflect_learned_words(db_path: Path):
    """After learning, get_hotwords_string should include learned words with 2.0 weight."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "我喜欢外部扣鼎")
        store.record_correction(
            segment_id="seg1",
            asr_text="我喜欢外部扣鼎",
            corrected_text="我喜欢强化学习",
            source="manual_edit",
        )
        store.learn_from_corrections()
        hotwords = store.get_hotwords_string()
        assert "强化学习:2" in hotwords or "强化学习:2.0" in hotwords


def test_common_word_filtered(db_path: Path):
    """Common words/particles should be filtered even if they appear in diff region."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "然后我们继续")
        store.record_correction(
            segment_id="seg1",
            asr_text="然后我们继续",
            corrected_text="但是我们继续",
            source="manual_edit",
        )
        added = store.learn_from_corrections()
        assert added == []


def test_high_freq_learning_disabled(db_path: Path):
    """High-frequency word learning is deprecated and should always return 0."""
    from ev.store.db import Store
    with Store(db_path) as store:
        _add_segment(store, "seg1", "今天我们继续研究这个问题")
        _add_segment(store, "seg2", "今天我们继续讨论那个项目")
        added = store.learn_high_frequency_words()
        assert added == 0


def test_v9_migration_cleans_old_high_freq_words(db_path: Path):
    """Migration v9 should delete auto words with weight < 2.0 (old high-frequency garbage)."""
    from ev.store.db import Store
    with Store(db_path) as store:
        import uuid
        now = datetime.now(timezone.utc).isoformat()
        store.connection.execute(
            "INSERT INTO lexicon (id, word, weight, source, use_count, created_at, updated_at) VALUES (?, ?, 1.5, 'auto', 5, ?, ?)",
            (uuid.uuid4().hex, "继续", now, now),
        )
        store.connection.execute(
            "INSERT INTO lexicon (id, word, weight, source, use_count, created_at, updated_at) VALUES (?, ?, 1.5, 'auto', 3, ?, ?)",
            (uuid.uuid4().hex, "研究", now, now),
        )
        store.connection.execute(
            "INSERT INTO lexicon (id, word, weight, source, use_count, created_at, updated_at) VALUES (?, ?, 2.0, 'auto', 2, ?, ?)",
            (uuid.uuid4().hex, "强化学习", now, now),
        )
        store.connection.commit()
        store.connection.close()
    with Store(db_path) as store2:
        rows = store2.connection.execute("SELECT word, weight FROM lexicon WHERE source='auto'").fetchall()
        words = {r["word"]: float(r["weight"]) for r in rows}
        assert "继续" not in words
        assert "研究" not in words
        assert "强化学习" in words
        assert abs(words["强化学习"] - 2.0) < 0.01
