"""EV SQLite schema、迁移与原子存储操作。"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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
    speaker_score: float | None
    wake_detected: bool
    query_candidate: bool
    query_text: str | None
    vad_model: str
    asr_stream_model: str
    asr_final_model: str
    speaker_model: str
    created_at: str


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
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS segments (
              id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
              duration_ms INTEGER NOT NULL, audio_path TEXT NOT NULL,
              sample_rate INTEGER NOT NULL, channels INTEGER NOT NULL,
              transcript_raw TEXT NOT NULL, transcript_final TEXT NOT NULL,
              speaker_label TEXT NOT NULL, speaker_score REAL,
              wake_detected INTEGER NOT NULL, query_candidate INTEGER NOT NULL,
              query_text TEXT, vad_model TEXT NOT NULL, asr_stream_model TEXT NOT NULL,
              asr_final_model TEXT NOT NULL, speaker_model TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS speaker_profiles (
              id TEXT PRIMARY KEY, label TEXT NOT NULL, device_selector TEXT,
              model_id TEXT NOT NULL, embedding_blob BLOB NOT NULL,
              embedding_dim INTEGER NOT NULL, embedding_dtype TEXT NOT NULL,
              sample_count INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS speaker_samples (
              id TEXT PRIMARY KEY,
              segment_id TEXT REFERENCES segments(id) ON DELETE CASCADE,
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
            CREATE INDEX IF NOT EXISTS idx_segments_started_at ON segments(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(speaker_label);
            CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_speaker_samples_created ON speaker_samples(created_at DESC);
            """
        )
        self._migrate_samples_v4()
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_speaker_samples_tier ON speaker_samples(tier)")
        self._migrate_binary_classification_v5()
        self._migrate_remove_unknown_v6()
        self.connection.execute("PRAGMA user_version=6")
        self._migrate_legacy_profile()
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

    def _migrate_binary_classification_v5(self) -> None:
        """Migrate from three-class (user/uncertain/non-user) to binary (user/non-user).
        - Existing 'uncertain' segments are reclassified as 'non-user' (safe default: better to reject than falsely accept)
        """
        self.connection.execute("UPDATE segments SET speaker_label='non-user' WHERE speaker_label='uncertain'")

    def _migrate_remove_unknown_v6(self) -> None:
        """Remove 'unknown' label - cold-start segments are treated as 'user'."""
        self.connection.execute("UPDATE segments SET speaker_label='user' WHERE speaker_label='unknown'")

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
            clauses.append("speaker_label=:speaker_label")
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
        return [dict(row) for row in rows]

    def list_queries(self, limit: int = 100) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM queries ORDER BY created_at DESC LIMIT ?",
            (min(max(int(limit), 1), 500),),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_query(self, query_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM queries WHERE id=?", (query_id,))
        return cursor.rowcount > 0

    def delete_all_queries(self) -> int:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM queries")
        return cursor.rowcount

    def delete_segment(self, segment_id: str) -> bool:
        row = self.connection.execute(
            "SELECT audio_path FROM segments WHERE id=?", (segment_id,)
        ).fetchone()
        if row is None:
            return False
        audio_path = Path(row["audio_path"])
        trash_path = self._move_to_trash(audio_path)
        try:
            with self.connection:
                self.connection.execute("DELETE FROM segments WHERE id=?", (segment_id,))
        except Exception:
            if trash_path is not None and trash_path.exists():
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                trash_path.replace(audio_path)
            raise
        if trash_path is not None:
            trash_path.unlink(missing_ok=True)
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
        return [dict(row) for row in rows]

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

    def evict_oldest_cache(self, max_cache: int) -> int:
        """Evict oldest cache samples to keep total cache count <= max_cache. Returns deleted count."""
        with self.connection:
            cursor = self.connection.execute(
                """
                DELETE FROM speaker_samples
                WHERE tier='cache' AND id IN (
                    SELECT id FROM speaker_samples
                    WHERE tier='cache'
                    ORDER BY created_at ASC
                    LIMIT MAX(0, (SELECT COUNT(*) FROM speaker_samples WHERE tier='cache') - ?)
                )
                """,
                (max_cache,),
            )
            return cursor.rowcount

    def delete_all_voice_samples(self) -> int:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM speaker_samples")
            deleted = cursor.rowcount
            self.connection.execute("DELETE FROM speaker_profiles WHERE id='user-v1'")
        return deleted

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
