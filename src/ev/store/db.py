"""EV SQLite schema、迁移与原子存储操作。"""

from __future__ import annotations

import re
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Chinese stopwords: common function words, particles, pronouns that should not be auto-learned
_STOPWORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "怎么", "如何", "为什么",
    "哪里", "哪个", "谁", "吗", "呢", "吧", "啊", "哦", "嗯", "呃", "诶", "唉", "哈",
    "喂", "哎", "噢", "呀", "哇", "啦", "哟", "呗", "噻", "咯", "嗬", "嗯啊",
    "那个", "这个", "就是", "然后", "所以", "因为", "但是", "不过", "其实", "可能",
    "应该", "可以", "已经", "还是", "只是", "不是", "没", "被", "把", "让", "给",
    "从", "向", "对", "与", "及", "等", "而", "且", "或", "但", "之", "其", "此",
    "以", "于", "为", "将", "能", "会", "要", "想", "觉得", "知道", "时候", "现在",
    "今天", "明天", "昨天", "这里", "那里", "这样", "那样", "这么", "那么", "怎么",
    "一下", "一点", "一些", "有些", "某个", "某种", "东西", "事情", "问题",
})

_PUNCT_RE = re.compile(r"[][\s，。！？、；：""''（）【】…—().,!?;:+=~`@#$%^&*|\\/<>-]+")


@dataclass(frozen=True)
class SegmentRecord:
    id: str
    started_at: str
    ended_at: str
    duration_ms: int
    audio_path: str
    sample_rate: int
    channels: int
    transcript_raw: str
    transcript_final: str
    speaker_label: str
    wake_detected: bool
    query_candidate: bool
    vad_model: str
    asr_stream_model: str
    asr_final_model: str
    speaker_model: str
    created_at: str
    raw_audio_path: str | None = None
    speaker_score: float | None = None
    query_text: str | None = None
    speaker_turns: str | None = None
    utterances: str | None = None
    source_type: str = "voice"
    dominant_speaker: str | None = None
    contains_user: bool = True
    end_trigger: str | None = None
    # 音频质量元数据 (v13)
    quality_label: str = "ok"
    avg_raw_rms: float | None = None
    peak_raw_rms: float | None = None
    noise_floor_rms: float | None = None
    snr_db: float | None = None


@dataclass(frozen=True)
class QueryRecord:
    id: str
    source: str
    segment_id: str | None
    text: str
    status: str
    created_at: str


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=5.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS segments (
              id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
              duration_ms INTEGER NOT NULL, audio_path TEXT NOT NULL,
              raw_audio_path TEXT,
              sample_rate INTEGER NOT NULL, channels INTEGER NOT NULL,
              transcript_raw TEXT NOT NULL, transcript_final TEXT NOT NULL,
              speaker_label TEXT NOT NULL, speaker_score REAL,
              wake_detected INTEGER NOT NULL, query_candidate INTEGER NOT NULL,
              query_text TEXT, vad_model TEXT NOT NULL, asr_stream_model TEXT NOT NULL,
              asr_final_model TEXT NOT NULL, speaker_model TEXT NOT NULL,
              created_at TEXT NOT NULL,
              speaker_turns TEXT, utterances TEXT,
              source_type TEXT NOT NULL DEFAULT 'voice',
              dominant_speaker TEXT, contains_user INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS speaker_profiles (
              id TEXT PRIMARY KEY, label TEXT NOT NULL, device_selector TEXT,
              model_id TEXT NOT NULL, embedding_blob BLOB NOT NULL,
              embedding_dim INTEGER NOT NULL, embedding_dtype TEXT NOT NULL,
              sample_count INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS speaker_samples (
              id TEXT PRIMARY KEY,
              segment_id TEXT REFERENCES segments(id) ON DELETE SET NULL,
              audio_path TEXT NOT NULL,
              duration_ms INTEGER NOT NULL,
              embedding_blob BLOB NOT NULL,
              embedding_dim INTEGER NOT NULL,
              score REAL NOT NULL,
              tier TEXT NOT NULL DEFAULT 'core' CHECK(tier IN ('core','cache')),
              is_manual INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS queries (
              id TEXT PRIMARY KEY,
              source TEXT NOT NULL CHECK(source IN ('voice','manual')),
              segment_id TEXT REFERENCES segments(id) ON DELETE CASCADE,
              text TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending',
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lexicon (
              id TEXT PRIMARY KEY,
              word TEXT NOT NULL UNIQUE,
              weight REAL NOT NULL DEFAULT 2.0,
              source TEXT NOT NULL CHECK(source IN ('manual','auto','system')),
              use_count INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_segments_started_at ON segments(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(speaker_label);
            CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_speaker_samples_created ON speaker_samples(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_lexicon_source ON lexicon(source);
            CREATE INDEX IF NOT EXISTS idx_lexicon_word ON lexicon(word);
            """
        )
        self._migrate_samples_v4()
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_speaker_samples_tier ON speaker_samples(tier)")
        self._migrate_binary_classification_v5()
        self._migrate_remove_unknown_v6()
        self._migrate_lexicon_v7()
        self._migrate_corrections_v8()
        self._migrate_cleanup_high_freq_v9()
        self._migrate_raw_audio_path_v10()
        self._migrate_speaker_turns_v11()
        self._migrate_end_trigger_v12()
        self._migrate_quality_v13()
        self._migrate_samples_fk_v14()
        self.connection.execute("PRAGMA user_version=14")
        self._migrate_legacy_profile()
        self._seed_system_words()
        self.connection.commit()

    def _migrate_samples_v4(self) -> None:
        """Add tier/is_manual columns to existing speaker_samples table (v3->v4)."""
        cols = {row["name"] for row in self.connection.execute("PRAGMA table_info(speaker_samples)").fetchall()}
        if "tier" not in cols:
            self.connection.execute("ALTER TABLE speaker_samples ADD COLUMN tier TEXT NOT NULL DEFAULT 'core'")
        if "is_manual" not in cols:
            self.connection.execute("ALTER TABLE speaker_samples ADD COLUMN is_manual INTEGER NOT NULL DEFAULT 0")
        # Ensure check constraint by updating any invalid values
        self.connection.execute("UPDATE speaker_samples SET tier='core' WHERE tier NOT IN ('core','cache')")

    def _migrate_samples_fk_v14(self) -> None:
        """Rebuild speaker_samples so deleting a history segment NULLs sample.segment_id
        instead of cascading the sample away (v13->v14). Sample wavs live in the managed
        voice-samples dir and are only removed via explicit sample deletion/reset."""
        fks = self.connection.execute("PRAGMA foreign_key_list(speaker_samples)").fetchall()
        if not any(row["on_delete"] == "CASCADE" for row in fks):
            return
        with self.connection:
            self.connection.execute("PRAGMA foreign_keys=OFF")
            try:
                self.connection.execute(
                    """
                    CREATE TABLE speaker_samples_new (
                      id TEXT PRIMARY KEY,
                      segment_id TEXT REFERENCES segments(id) ON DELETE SET NULL,
                      audio_path TEXT NOT NULL,
                      duration_ms INTEGER NOT NULL,
                      embedding_blob BLOB NOT NULL,
                      embedding_dim INTEGER NOT NULL,
                      score REAL NOT NULL,
                      tier TEXT NOT NULL DEFAULT 'core' CHECK(tier IN ('core','cache')),
                      is_manual INTEGER NOT NULL DEFAULT 0,
                      created_at TEXT NOT NULL
                    )
                    """
                )
                self.connection.execute(
                    """
                    INSERT INTO speaker_samples_new
                      (id, segment_id, audio_path, duration_ms, embedding_blob, embedding_dim, score, tier, is_manual, created_at)
                    SELECT id, segment_id, audio_path, duration_ms, embedding_blob, embedding_dim, score, tier, is_manual, created_at
                    FROM speaker_samples
                    """
                )
                self.connection.execute("DROP TABLE speaker_samples")
                self.connection.execute("ALTER TABLE speaker_samples_new RENAME TO speaker_samples")
            finally:
                self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_speaker_samples_created ON speaker_samples(created_at DESC)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_speaker_samples_tier ON speaker_samples(tier)"
            )

    def _migrate_binary_classification_v5(self) -> None:
        """Migrate from three-class (user/uncertain/non-user) to binary (user/non-user).
        - Existing 'uncertain' segments are reclassified as 'non-user' (safe default: better to reject than falsely accept)
        """
        self.connection.execute("UPDATE segments SET speaker_label='non-user' WHERE speaker_label='uncertain'")

    def _migrate_remove_unknown_v6(self) -> None:
        """Remove 'unknown' label - cold-start segments are treated as 'user'."""
        self.connection.execute("UPDATE segments SET speaker_label='user' WHERE speaker_label='unknown'")

    def _migrate_lexicon_v7(self) -> None:
        """Create lexicon table if it doesn't exist (v7)."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lexicon (
              id TEXT PRIMARY KEY,
              word TEXT NOT NULL UNIQUE,
              weight REAL NOT NULL DEFAULT 2.0,
              source TEXT NOT NULL CHECK(source IN ('manual','auto','system')),
              use_count INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_lexicon_source ON lexicon(source)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_lexicon_word ON lexicon(word)")

    def _migrate_corrections_v8(self) -> None:
        """Create correction_history table (v8) for ASR error tracking."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS correction_history (
              id TEXT PRIMARY KEY,
              segment_id TEXT REFERENCES segments(id) ON DELETE SET NULL,
              asr_text TEXT NOT NULL,
              corrected_text TEXT NOT NULL,
              source TEXT NOT NULL CHECK(source IN ('manual_edit','manual_add_word','implicit_repeat')),
              context TEXT,
              speaker_label TEXT,
              speaker_score REAL,
              audio_path TEXT,
              is_applied INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_corrections_source ON correction_history(source)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_corrections_applied ON correction_history(is_applied)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_corrections_created ON correction_history(created_at)")

    def _migrate_cleanup_high_freq_v9(self) -> None:
        """Clean up legacy high-frequency auto-learned words (weight=1.5) from v7/v8.
        These were added by the naive high-frequency word learner which produced garbage.
        Only keep auto words from correction learning (weight >= 2.0).
        """
        self.connection.execute("DELETE FROM lexicon WHERE source='auto' AND weight < 2.0")

    def _migrate_raw_audio_path_v10(self) -> None:
        """Add raw_audio_path column to segments table (v9->v10) for dual-wav archive."""
        cols = {row["name"] for row in self.connection.execute("PRAGMA table_info(segments)").fetchall()}
        if "raw_audio_path" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN raw_audio_path TEXT")

    def _migrate_speaker_turns_v11(self) -> None:
        """Add speaker_turns/utterances/source_type/dominant_speaker/contains_user columns (v10->v11)
        for multi-speaker diarization support ("second ear" feature)."""
        cols = {row["name"] for row in self.connection.execute("PRAGMA table_info(segments)").fetchall()}
        if "speaker_turns" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN speaker_turns TEXT")
        if "utterances" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN utterances TEXT")
        if "source_type" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN source_type TEXT NOT NULL DEFAULT 'voice'")
        else:
            self.connection.execute("UPDATE segments SET source_type='voice' WHERE source_type IS NULL OR source_type=''")
        if "dominant_speaker" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN dominant_speaker TEXT")
        if "contains_user" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN contains_user INTEGER NOT NULL DEFAULT 1")
        # Backfill: for existing segments, dominant_speaker = speaker_label, contains_user = 1
        self.connection.execute(
            "UPDATE segments SET dominant_speaker=speaker_label WHERE dominant_speaker IS NULL"
        )
        self.connection.execute(
            "UPDATE segments SET contains_user=1 WHERE contains_user IS NULL"
        )

    def _seed_system_words(self) -> None:
        """Seed built-in system words (wake words) that cannot be deleted."""
        now = datetime.now(timezone.utc).isoformat()
        system_words = [
            ("小E", 5.0),
            ("小e", 5.0),
        ]
        for word, weight in system_words:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO lexicon (id, word, weight, source, use_count, created_at, updated_at)
                VALUES (?, ?, ?, 'system', 0, ?, ?)
                """,
                (uuid.uuid4().hex, word, weight, now, now),
            )

    def _migrate_end_trigger_v12(self) -> None:
        """Add end_trigger column to segments table (v11->v12).

        Records which endpoint trigger closed each segment
        (vad_endpoint/max_duration/silence_timeout/relative_silence/energy_silent/
        asr_stall/stop) so endpoint misbehavior can be diagnosed from the DB alone.
        """
        cols = {row["name"] for row in self.connection.execute("PRAGMA table_info(segments)").fetchall()}
        if "end_trigger" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN end_trigger TEXT")

    def _migrate_quality_v13(self) -> None:
        """Add audio quality columns to segments table (v12->v13).

        Stores per-segment audio quality metrics computed from raw (unprocessed)
        audio for post-hoc analysis and quality-gating threshold tuning.
        """
        cols = {row["name"] for row in self.connection.execute("PRAGMA table_info(segments)").fetchall()}
        if "quality_label" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN quality_label TEXT NOT NULL DEFAULT 'ok'")
        if "avg_raw_rms" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN avg_raw_rms REAL")
        if "peak_raw_rms" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN peak_raw_rms REAL")
        if "noise_floor_rms" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN noise_floor_rms REAL")
        if "snr_db" not in cols:
            self.connection.execute("ALTER TABLE segments ADD COLUMN snr_db REAL")

    def _migrate_legacy_profile(self) -> None:
        """Migrate legacy single-embedding profile to a sample entry if samples table is empty."""
        sample_count = self.connection.execute(
            "SELECT COUNT(*) as c FROM speaker_samples"
        ).fetchone()["c"]
        if sample_count > 0:
            return
        legacy = self.connection.execute(
            "SELECT embedding_blob, embedding_dim FROM speaker_profiles WHERE id='user-v1'"
        ).fetchone()
        if legacy is None:
            return
        embedding = np.frombuffer(
            legacy["embedding_blob"], dtype="<f4", count=legacy["embedding_dim"]
        ).copy()
        sample_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            INSERT INTO speaker_samples
              (id, segment_id, audio_path, duration_ms, embedding_blob, embedding_dim, score, tier, is_manual, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'core', 1, ?)
            """,
            (sample_id, None, "", 0, embedding.tobytes(), embedding.size, 1.0, now),
        )

    def insert_segment(self, record: SegmentRecord) -> None:
        data = asdict(record)
        columns = ", ".join(data)
        placeholders = ", ".join(f":{key}" for key in data)
        with self.connection:
            self.connection.execute(
                f"INSERT INTO segments ({columns}) VALUES ({placeholders})", data
            )
            if record.query_candidate and record.query_text:
                query = QueryRecord(
                    id=uuid.uuid4().hex,
                    source="voice",
                    segment_id=record.id,
                    text=record.query_text,
                    status="pending",
                    created_at=record.created_at,
                )
                self._insert_query(query)

    def submit_manual_query(self, text: str) -> QueryRecord:
        value = text.strip()
        if not value:
            raise ValueError("query 不能为空")
        query = QueryRecord(
            id=uuid.uuid4().hex,
            source="manual",
            segment_id=None,
            text=value,
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.connection:
            self._insert_query(query)
        return query

    def _insert_query(self, record: QueryRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO queries (id,source,segment_id,text,status,created_at)
            VALUES (:id,:source,:segment_id,:text,:status,:created_at)
            """,
            asdict(record),
        )

    def list_segments(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        speaker_label: str | None = None,
        query_only: bool = False,
        date_prefix: str | None = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: dict[str, object] = {
            "limit": min(max(int(limit), 1), 500),
            "offset": max(int(offset), 0),
        }
        if speaker_label:
            # speaker_label filter uses new columns for multi-speaker segments, falls back for old data
            if speaker_label == "user":
                # Show segments that contain any user speech
                clauses.append("(contains_user=1 OR (dominant_speaker IS NULL AND speaker_label='user'))")
            elif speaker_label == "non-user":
                # Show pure non-user segments (no user speech at all)
                clauses.append("(contains_user=0 OR (dominant_speaker IS NULL AND speaker_label='non-user'))")
            else:
                clauses.append("(dominant_speaker=:speaker_label OR (dominant_speaker IS NULL AND speaker_label=:speaker_label))")
                params["speaker_label"] = speaker_label
        if query_only:
            clauses.append("query_candidate=1")
        if date_prefix:
            clauses.append("started_at LIKE :date_prefix")
            params["date_prefix"] = f"{date_prefix}%"
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM segments{where} ORDER BY started_at DESC LIMIT :limit OFFSET :offset",
            params,
        ).fetchall()
        results = [dict(row) for row in rows]
        # Annotate with correction status
        if results:
            ids = [r["id"] for r in results]
            placeholders = ",".join("?" * len(ids))
            corrected_ids = {
                row["segment_id"]
                for row in self.connection.execute(
                    f"SELECT DISTINCT segment_id FROM correction_history WHERE segment_id IN ({placeholders}) AND source='manual_edit'",
                    ids,
                ).fetchall()
                if row["segment_id"]
            }
            for r in results:
                r["was_corrected"] = r["id"] in corrected_ids
        return results

    def list_queries(self, limit: int = 100) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM queries ORDER BY created_at DESC LIMIT ?",
            (min(max(int(limit), 1), 500),),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_query(self, query_id: str) -> bool:
        row = self.connection.execute(
            "SELECT source, segment_id FROM queries WHERE id=?", (query_id,)
        ).fetchone()
        if row is None:
            return False
        segment_id = row["segment_id"] if row["source"] == "voice" else None
        with self.connection:
            self.connection.execute("DELETE FROM queries WHERE id=?", (query_id,))
            if segment_id:
                self.connection.execute(
                    "UPDATE segments SET query_candidate=0, query_text='' WHERE id=?",
                    (segment_id,),
                )
        return True

    def delete_all_queries(self) -> int:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM queries")
        return cursor.rowcount

    def delete_segment(self, segment_id: str) -> bool:
        row = self.connection.execute(
            "SELECT audio_path, raw_audio_path FROM segments WHERE id=?", (segment_id,)
        ).fetchone()
        if row is None:
            return False
        audio_path = Path(row["audio_path"])
        raw_path = Path(row["raw_audio_path"]) if row["raw_audio_path"] else None
        trash_path = self._move_to_trash(audio_path)
        raw_trash_path = self._move_to_trash(raw_path) if raw_path is not None else None
        try:
            with self.connection:
                self.connection.execute("DELETE FROM segments WHERE id=?", (segment_id,))
        except Exception:
            if trash_path is not None and trash_path.exists():
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                trash_path.replace(audio_path)
            if raw_trash_path is not None and raw_trash_path.exists() and raw_path is not None:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_trash_path.replace(raw_path)
            raise
        if trash_path is not None:
            trash_path.unlink(missing_ok=True)
        if raw_trash_path is not None:
            raw_trash_path.unlink(missing_ok=True)
        return True

    def delete_all_segments(self) -> int:
        rows = self.connection.execute("SELECT id FROM segments").fetchall()
        deleted = 0
        for row in rows:
            if self.delete_segment(row["id"]):
                deleted += 1
        return deleted

    @staticmethod
    def _move_to_trash(audio_path: Path) -> Path | None:
        if not audio_path.exists():
            return None
        trash_dir = audio_path.parent / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        trash_path = trash_dir / f"{uuid.uuid4().hex}-{audio_path.name}"
        audio_path.replace(trash_path)
        return trash_path

    def save_profile(
        self,
        profile_id: str,
        label: str,
        device_selector: str | None,
        model_id: str,
        embedding: np.ndarray,
        sample_count: int,
    ) -> None:
        vector = np.asarray(embedding, dtype="<f4").reshape(-1)
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO speaker_profiles
                  (id,label,device_selector,model_id,embedding_blob,embedding_dim,
                   embedding_dtype,sample_count,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  label=excluded.label, device_selector=excluded.device_selector,
                  model_id=excluded.model_id, embedding_blob=excluded.embedding_blob,
                  embedding_dim=excluded.embedding_dim, embedding_dtype=excluded.embedding_dtype,
                  sample_count=excluded.sample_count, updated_at=excluded.updated_at
                """,
                (
                    profile_id, label, device_selector, model_id, vector.tobytes(),
                    vector.size, "float32", sample_count, now, now,
                ),
            )

    def load_profile(self, profile_id: str = "user-v1") -> np.ndarray | None:
        embeddings = self.load_voice_sample_embeddings()
        if not embeddings:
            return None
        from ..speaker.verification import build_profile
        return build_profile(embeddings)

    def profile_status(self, profile_id: str = "user-v1") -> dict:
        core_count = self.count_voice_samples(tier="core")
        cache_count = self.count_voice_samples(tier="cache")
        sample_count = core_count + cache_count
        updated_row = self.connection.execute(
            "SELECT MAX(created_at) as updated FROM speaker_samples"
        ).fetchone()
        updated_at = updated_row["updated"] if updated_row else None
        from ..speaker.verification import choose_k
        centroid_count = choose_k(core_count, 3) if core_count > 0 else 0
        if sample_count == 0:
            return {
                "exists": False,
                "is_ready": False,
                "sample_count": 0,
                "core_count": 0,
                "cache_count": 0,
                "centroid_count": 0,
                "updated_at": None,
                "last_updated": None,
            }
        return {
            "exists": core_count >= 3,
            "is_ready": core_count >= 3,
            "sample_count": sample_count,
            "core_count": core_count,
            "cache_count": cache_count,
            "centroid_count": centroid_count,
            "updated_at": updated_at,
            "last_updated": updated_at,
        }

    def add_voice_sample(
        self,
        *,
        segment_id: str | None,
        audio_path: str,
        duration_ms: int,
        embedding: np.ndarray,
        score: float,
        tier: str = "core",
        is_manual: bool = False,
    ) -> str:
        vector = np.asarray(embedding, dtype="<f4").reshape(-1)
        sample_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO speaker_samples
                  (id, segment_id, audio_path, duration_ms, embedding_blob, embedding_dim, score, tier, is_manual, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sample_id, segment_id, audio_path, duration_ms, vector.tobytes(),
                 vector.size, score, tier, 1 if is_manual else 0, now),
            )
        return sample_id

    def list_voice_samples(self, tier: str | None = None, limit: int = 100) -> list[dict]:
        if tier not in (None, "core", "cache"):
            raise ValueError(f"invalid tier: {tier}")
        sql = """
            SELECT id, segment_id, audio_path, duration_ms, score, tier, is_manual, created_at
            FROM speaker_samples
        """
        params: list = []
        if tier:
            sql += " WHERE tier = ?"
            params.append(tier)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(min(max(int(limit), 1), 200))
        rows = self.connection.execute(sql, params).fetchall()
        out = []
        for row in rows:
            sample = dict(row)
            sample["audio_available"] = bool(sample["audio_path"]) and sample["audio_path"] is not None and self._path_exists(sample["audio_path"])
            out.append(sample)
        return out

    @staticmethod
    def _path_exists(path: str | None) -> bool:
        if not path:
            return False
        from pathlib import Path
        return Path(path).exists()

    def list_all_voice_sample_paths(self) -> list[str]:
        rows = self.connection.execute("SELECT audio_path FROM speaker_samples").fetchall()
        return [row["audio_path"] for row in rows if row["audio_path"]]

    def update_voice_sample_embedding(self, sample_id: str, embedding: np.ndarray) -> bool:
        vector = np.asarray(embedding, dtype="<f4").reshape(-1)
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE speaker_samples SET embedding_blob=?, embedding_dim=? WHERE id=?",
                (vector.tobytes(), vector.size, sample_id),
            )
            return cursor.rowcount > 0

    def update_voice_sample_audio_path(self, sample_id: str, audio_path: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE speaker_samples SET audio_path=? WHERE id=?", (audio_path, sample_id)
            )
            return cursor.rowcount > 0

    def count_voice_samples(self, tier: str | None = None) -> int:
        if tier:
            row = self.connection.execute(
                "SELECT COUNT(*) as c FROM speaker_samples WHERE tier=?", (tier,)
            ).fetchone()
        else:
            row = self.connection.execute("SELECT COUNT(*) as c FROM speaker_samples").fetchone()
        return int(row["c"])

    def load_voice_sample_embeddings(self, tier: str = "core") -> list[tuple[str, np.ndarray, float]]:
        """Return list of (sample_id, embedding, score) for samples in given tier."""
        if tier not in ("core", "cache", None):
            raise ValueError(f"invalid tier: {tier}")
        if tier:
            rows = self.connection.execute(
                "SELECT id, embedding_blob, embedding_dim, score FROM speaker_samples WHERE tier=? ORDER BY created_at ASC",
                (tier,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT id, embedding_blob, embedding_dim, score FROM speaker_samples ORDER BY created_at ASC"
            ).fetchall()
        return [
            (
                row["id"],
                np.frombuffer(row["embedding_blob"], dtype="<f4", count=row["embedding_dim"]).copy(),
                float(row["score"]),
            )
            for row in rows
        ]

    def get_voice_sample(self, sample_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM speaker_samples WHERE id=?", (sample_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_voice_sample_tier(self, sample_id: str, tier: str) -> bool:
        if tier not in ("core", "cache"):
            raise ValueError(f"invalid tier: {tier}")
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE speaker_samples SET tier=? WHERE id=?", (tier, sample_id)
            )
            return cursor.rowcount > 0

    def delete_voice_sample(self, sample_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM speaker_samples WHERE id=?", (sample_id,)
            )
            return cursor.rowcount > 0

    def evict_oldest_cache(self, max_cache: int) -> list[str]:
        """Evict oldest cache samples to keep total cache count <= max_cache.

        Returns the list of audio_paths of the evicted samples (files are NOT
        removed here; callers decide based on path ownership).
        """
        rows = self.connection.execute(
            """
            SELECT id, audio_path FROM speaker_samples
            WHERE tier='cache'
            ORDER BY created_at ASC
            LIMIT MAX(0, (SELECT COUNT(*) FROM speaker_samples WHERE tier='cache') - ?)
            """,
            (max_cache,),
        ).fetchall()
        paths = [row["audio_path"] for row in rows if row["audio_path"]]
        ids = [row["id"] for row in rows]
        if not ids:
            return []
        with self.connection:
            placeholders = ",".join("?" for _ in ids)
            self.connection.execute(
                f"DELETE FROM speaker_samples WHERE id IN ({placeholders})", ids
            )
        return paths

    def delete_all_voice_samples(self) -> int:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM speaker_samples")
            deleted = cursor.rowcount
            self.connection.execute("DELETE FROM speaker_profiles WHERE id='user-v1'")
        return deleted

    # --- Lexicon (hotwords) ---

    def list_lexicon(self) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM lexicon ORDER BY CASE source WHEN 'system' THEN 0 WHEN 'manual' THEN 1 ELSE 2 END, weight DESC, use_count DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def add_lexicon_word(self, word: str, weight: float = 3.0, source: str = "manual") -> dict:
        cleaned = word.strip()
        if not cleaned:
            raise ValueError("词语不能为空")
        if source not in ("manual", "auto", "system"):
            raise ValueError(f"invalid source: {source}")
        w = max(0.5, min(10.0, float(weight)))
        now = datetime.now(timezone.utc).isoformat()
        word_id = uuid.uuid4().hex
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO lexicon (id, word, weight, source, use_count, created_at, updated_at)
                VALUES (
                    COALESCE((SELECT id FROM lexicon WHERE word=?), ?),
                    ?, ?, ?,
                    COALESCE((SELECT use_count FROM lexicon WHERE word=?), 0),
                    COALESCE((SELECT created_at FROM lexicon WHERE word=?), ?),
                    ?
                )
                """,
                (cleaned, word_id, cleaned, w, source, cleaned, cleaned, now, now),
            )
        row = self.connection.execute("SELECT * FROM lexicon WHERE word=?", (cleaned,)).fetchone()
        return dict(row) if row else {}

    def update_lexicon_word(self, word_id: str, word: str | None = None, weight: float | None = None, promote_to_manual: bool = False) -> bool:
        existing = self.connection.execute("SELECT * FROM lexicon WHERE id=?", (word_id,)).fetchone()
        if existing is None:
            return False
        updates = []
        params: list = []
        if word is not None:
            cleaned = word.strip()
            if cleaned:
                updates.append("word=?")
                params.append(cleaned)
        if weight is not None:
            w = max(0.5, min(10.0, float(weight)))
            updates.append("weight=?")
            params.append(w)
        if promote_to_manual and existing["source"] != "system":
            updates.append("source='manual'")
            updates.append("weight=?")
            params.append(max(float(existing["weight"]), 3.0))
        if not updates:
            return False
        updates.append("updated_at=?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(word_id)
        with self.connection:
            cursor = self.connection.execute(
                f"UPDATE lexicon SET {', '.join(updates)} WHERE id=?",
                params,
            )
            return cursor.rowcount > 0

    def delete_lexicon_word(self, word_id: str) -> bool:
        """Delete a lexicon entry. System words cannot be deleted."""
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM lexicon WHERE id=? AND source!='system'", (word_id,)
            )
            return cursor.rowcount > 0

    def clear_auto_words(self) -> int:
        """Delete all auto-learned words. Returns count deleted."""
        with self.connection:
            cursor = self.connection.execute("DELETE FROM lexicon WHERE source='auto'")
            return cursor.rowcount

    def get_hotwords_string(self, max_words: int = 80) -> str:
        """Build hotwords string for FunASR model-level boosting.

        Returns space-separated word list WITHOUT weights, e.g. "word1 word2 word3".
        Weight suffixes (":3.0") break FunASR's seg_tokenize, turning the whole
        hotword into <unk> and defeating the boosting mechanism entirely.
        Priority ordering (system > manual > auto, weight DESC) is preserved.
        """
        rows = self.connection.execute(
            """
            SELECT word, weight FROM lexicon
            ORDER BY CASE source WHEN 'system' THEN 0 WHEN 'manual' THEN 1 ELSE 2 END, weight DESC
            LIMIT ?
            """,
            (max_words,),
        ).fetchall()
        parts = []
        for row in rows:
            w = str(row["word"]).strip()
            if w and _PUNCT_RE.sub("", w):
                parts.append(w)
        return " ".join(parts)

    def get_hotword_entries(self, max_words: int = 80) -> list[tuple[str, float]]:
        """Return (word, weight) pairs for anchor-based logits boosting.

        System words (e.g. 小E) are excluded: wake-word matching is rule-based in
        vui.py, and injecting the wake word into decode bias would nudge mundane
        speech toward spurious wake words. Ordering: manual > auto, weight DESC.
        """
        rows = self.connection.execute(
            """
            SELECT word, weight FROM lexicon
            WHERE source != 'system'
            ORDER BY CASE source WHEN 'manual' THEN 0 ELSE 1 END, weight DESC
            LIMIT ?
            """,
            (max_words,),
        ).fetchall()
        entries = []
        for row in rows:
            w = str(row["word"]).strip()
            if w and _PUNCT_RE.sub("", w):
                entries.append((w, float(row["weight"])))
        return entries

    def learn_high_frequency_words(self, min_count: int = 2, max_auto_words: int = 100) -> int:
        """Deprecated: High-frequency word learning is disabled.
        It produced too many garbage common words. Only correction-driven learning is used now.
        """
        return 0

    def record_correction(
        self,
        asr_text: str,
        corrected_text: str,
        source: str,
        segment_id: str | None = None,
        context: str | None = None,
        speaker_label: str | None = None,
        speaker_score: float | None = None,
        audio_path: str | None = None,
    ) -> dict:
        """Record an ASR correction event. This is append-only research data."""
        if source not in ("manual_edit", "manual_add_word", "implicit_repeat"):
            raise ValueError(f"invalid correction source: {source}")
        now = datetime.now(timezone.utc).isoformat()
        cid = uuid.uuid4().hex
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO correction_history
                    (id, segment_id, asr_text, corrected_text, source, context,
                     speaker_label, speaker_score, audio_path, is_applied, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (cid, segment_id, asr_text, corrected_text, source, context,
                 speaker_label, speaker_score, audio_path, now),
            )
        row = self.connection.execute("SELECT * FROM correction_history WHERE id=?", (cid,)).fetchone()
        return dict(row) if row else {}

    def list_corrections(
        self,
        limit: int = 100,
        offset: int = 0,
        source: str | None = None,
        is_applied: bool | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM correction_history WHERE 1=1"
        params: list = []
        if source is not None:
            sql += " AND source=?"
            params.append(source)
        if is_applied is not None:
            sql += " AND is_applied=?"
            params.append(1 if is_applied else 0)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.connection.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count_corrections(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) as c FROM correction_history").fetchone()
        return int(row["c"]) if row else 0

    def learn_from_corrections(self, min_corrections: int = 1, max_auto_words: int = 100) -> list[str]:
        """Auto-learning from corrections is DISABLED pending LLM-based semantic analysis.
        Correction history is still recorded and marked as applied, but no words are
        automatically added to the lexicon. Users must manually add words via the UI.
        Returns empty list always.
        """
        with self.connection:
            self.connection.execute("UPDATE correction_history SET is_applied=1 WHERE is_applied=0")
        return []

    def update_segment_transcript(self, segment_id: str, corrected_text: str) -> dict | None:
        """Update a segment's transcript_final and return the updated row, or None if not found."""
        row = self.connection.execute(
            "SELECT * FROM segments WHERE id=?", (segment_id,)
        ).fetchone()
        if row is None:
            return None
        with self.connection:
            self.connection.execute(
                "UPDATE segments SET transcript_final=? WHERE id=?",
                (corrected_text, segment_id),
            )
        row = self.connection.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
