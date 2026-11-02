"""Simple durable session metadata store for the phase-1, single-tenant server.

This is not an inference queue or a multi-user account store.
"""
from pathlib import Path
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone

from .schemas import SessionCreate, SessionRecord


class SessionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT,
                    language TEXT,
                    tone TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, body: SessionCreate) -> SessionRecord:
        record = SessionRecord(
            id=uuid4().hex,
            mode=body.mode,
            status="created",
            title=body.title,
            language=body.language,
            tone=body.tone,
            created_at=datetime.now(timezone.utc),
        )
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO sessions
                    (id, mode, status, title, language, tone, created_at)
                    VALUES (:id, :mode, :status, :title, :language, :tone, :created_at)""",
                record.model_dump(mode="json"),
            )
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return SessionRecord.model_validate(dict(row)) if row else None
