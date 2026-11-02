"""Resumable meeting upload ledger with durable byte offsets and crash repair."""
from pathlib import Path
import os
import sqlite3
import threading
from uuid import uuid4

from .meeting_store import now

MAX_PART = 4 * 1024 * 1024

class UploadError(Exception):
    pass

class UploadConflict(UploadError):
    def __init__(self, offset: int):
        self.offset = offset

class UploadStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.root = db_path.parent / "meetings" / "incoming"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS meeting_uploads (
                id TEXT PRIMARY KEY, client_ref TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL, total_bytes INTEGER NOT NULL,
                received_bytes INTEGER NOT NULL DEFAULT 0,
                language TEXT, title TEXT, meeting_date TEXT, kind TEXT NOT NULL DEFAULT 'meeting',
                asr_model_id TEXT,
                job_id TEXT, created_at TEXT NOT NULL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(meeting_uploads)")}
            for column in ("title", "meeting_date", "kind", "asr_model_id"):
                if column not in columns:
                    db.execute(f"ALTER TABLE meeting_uploads ADD COLUMN {column} TEXT" +
                               (" NOT NULL DEFAULT 'meeting'" if column == "kind" else ""))

    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def get(self, upload_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM meeting_uploads WHERE id=?", (upload_id,)).fetchone()
        return dict(row) if row else None

    def start(self, client_ref: str, total_bytes: int, language: str | None,
              title: str | None = None, meeting_date: str | None = None,
              kind: str = "meeting", asr_model_id: str | None = None) -> dict:
        with self.lock:
            with self.connect() as db:
                prior = db.execute("SELECT * FROM meeting_uploads WHERE client_ref=?", (client_ref,)).fetchone()
                if prior:
                    if prior["total_bytes"] != total_bytes:
                        raise UploadError("Upload reference already exists with another file size")
                    return dict(prior)
                uid = uuid4().hex
                path = self.root / f"{uid}.part"
                path.touch(exist_ok=False)
                db.execute("""INSERT INTO meeting_uploads
                   (id,client_ref,path,total_bytes,language,title,meeting_date,kind,asr_model_id,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",(uid,client_ref,str(path),total_bytes,language,
                                              title,meeting_date,kind,asr_model_id,now()))
            return self.get(uid)

    def append(self, uid: str, offset: int, data: bytes) -> dict:
        if not data or len(data) > MAX_PART:
            raise UploadError("Part must contain 1 to 4 MiB")
        with self.lock:
            row=self.get(uid)
            if row is None:
                raise FileNotFoundError()
            if row["job_id"] or row["received_bytes"] != offset:
                raise UploadConflict(row["received_bytes"])
            if offset + len(data) > row["total_bytes"]:
                raise UploadError("Upload exceeds announced size")
            path = Path(row["path"])
            with path.open("r+b") as file:
                # After a crash, an fsynced block might be longer than the
                # recorded offset. Truncate to the durable SQLite checkpoint.
                file.truncate(offset)
                file.seek(offset)
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            with self.connect() as db:
                db.execute("""UPDATE meeting_uploads SET received_bytes=?
                    WHERE id=?""",(offset+len(data),uid))
            return self.get(uid)

    def completed(self, uid: str, job_id: str, final_path: Path) -> dict:
        with self.lock:
            with self.connect() as db:
                db.execute("""UPDATE meeting_uploads SET job_id=?,path=? WHERE id=?""",
                           (job_id,str(final_path),uid))
            return self.get(uid)
