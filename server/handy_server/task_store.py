"""Durable, explicit, evidence-backed task management for meeting action items.

Import never creates a task without matching original recorded utterances.
Tasks belong to the current single-tenant API and require its bearer token.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .notes_store import NoteConflict
from .schemas import MeetingAnalysis


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS action_tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                assignee TEXT,
                due_date TEXT,
                status TEXT NOT NULL DEFAULT 'todo'
                    CHECK(status IN ('todo','in_progress','done','cancelled')),
                source_meeting_id TEXT,
                source_index INTEGER,
                source_ids TEXT NOT NULL DEFAULT '[]',
                source_quotes TEXT NOT NULL DEFAULT '[]',
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_meeting_id,source_index)
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS task_status_date
                ON action_tasks(status,due_date,updated_at)""")

    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _read(row):
        result = dict(row)
        result["source_ids"] = json.loads(result["source_ids"])
        result["source_quotes"] = json.loads(result["source_quotes"])
        return result

    def get(self, task_id: str):
        with self.connect() as db:
            item = db.execute("SELECT * FROM action_tasks WHERE id=?", (task_id,)).fetchone()
        return self._read(item) if item else None

    def list(self, status: str | None = None, offset: int = 0, limit: int = 50):
        with self.connect() as db:
            where = "WHERE status=?" if status else ""
            args = [status] if status else []
            rows = db.execute(f"""SELECT * FROM action_tasks {where}
                ORDER BY CASE WHEN due_date IS NULL THEN 1 ELSE 0 END,
                         due_date ASC,updated_at DESC LIMIT ? OFFSET ?""",
                (*args,limit,offset)).fetchall()
            total = db.execute(f"SELECT COUNT(*) FROM action_tasks {where}", args).fetchone()[0]
        return {"items":[self._read(item) for item in rows],"total":total}

    def create(self, title: str, assignee: str | None, due_date: str | None):
        task_id = uuid4().hex
        stamp = now()
        with self.connect() as db:
            db.execute("""INSERT INTO action_tasks
                (id,title,assignee,due_date,created_at,updated_at)
                VALUES(?,?,?,?,?,?)""",
                (task_id,title,assignee,due_date,stamp,stamp))
        return self.get(task_id)

    def update(self, task_id: str, version: int, title: str, assignee: str | None,
               due_date: str | None, status: str):
        with self.connect() as db:
            row = db.execute("SELECT version FROM action_tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                return None
            if row["version"] != version:
                raise NoteConflict("Task was updated on another device")
            changed = db.execute("""UPDATE action_tasks
                SET title=?,assignee=?,due_date=?,status=?,version=version+1,updated_at=?
                WHERE id=? AND version=?""",
                (title,assignee,due_date,status,now(),task_id,version))
            if changed.rowcount != 1:
                raise NoteConflict("Task was updated on another device")
        return self.get(task_id)

    def delete(self, task_id: str) -> bool:
        with self.connect() as db:
            return bool(db.execute("DELETE FROM action_tasks WHERE id=?", (task_id,)).rowcount)

    def import_meeting(self, meeting_id: str) -> dict:
        # One transaction makes repeated clicks and concurrent imports idempotent.
        with self.connect() as db:
            job = db.execute(
                "SELECT status,insights FROM meeting_jobs WHERE id=?", (meeting_id,)
            ).fetchone()
            if job is None:
                raise KeyError("Meeting not found")
            if job["status"] != "completed" or not job["insights"]:
                raise ValueError("Meeting analysis is not complete")
            insights = MeetingAnalysis.model_validate_json(job["insights"])
            created = 0
            for idx, task in enumerate(insights.tasks):
                if not task.source_ids:
                    continue
                cited = list(dict.fromkeys(task.source_ids))
                if len(cited) > 20:
                    continue
                marks = ",".join("?" for _ in cited)
                quoted = db.execute(
                    "SELECT id,text FROM meeting_segments WHERE job_id=? AND id IN (" +
                    marks + ")", (meeting_id, *cited)
                ).fetchall()
                if len(quoted) != len(cited):
                    continue
                quotes = [{"id":row["id"],"text":row["text"][:1200]} for row in quoted]
                stamp = now()
                inserted = db.execute("""INSERT OR IGNORE INTO action_tasks
                    (id,title,assignee,due_date,source_meeting_id,source_index,
                     source_ids,source_quotes,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",(
                    uuid4().hex,task.text[:500],task.assignee[:120] if task.assignee else None,
                    task.due_date[:32] if task.due_date else None,
                    meeting_id,idx,json.dumps(cited,ensure_ascii=False),
                    json.dumps(quotes,ensure_ascii=False),stamp,stamp
                ))
                created += inserted.rowcount
        return {"created":created,"tasks":self.by_meeting(meeting_id)}

    def by_meeting(self, meeting_id: str):
        with self.connect() as db:
            items = db.execute("""SELECT * FROM action_tasks WHERE source_meeting_id=?
                ORDER BY source_index""",(meeting_id,)).fetchall()
        return [self._read(row) for row in items]
