"""Phase 1a SQLite schema 与原子写入。"""

from __future__ import annotations

import sqlite3
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


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
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
