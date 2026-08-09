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
            PRAGMA user_version=2;
            """
        )
        self.connection.commit()

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
        row = self.connection.execute(
            "SELECT embedding_blob, embedding_dim FROM speaker_profiles WHERE id=?",
            (profile_id,),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["embedding_blob"], dtype="<f4", count=row["embedding_dim"]).copy()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
