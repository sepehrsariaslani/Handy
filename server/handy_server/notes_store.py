"""Single-tenant durable notes with provenance, optimistic edits and FTS search.

Audio and AI analysis are separate: creating/editing/searching notes needs no GPU.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .schemas import NotesAnalysis


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NoteConflict(Exception):
    pass


class NoteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_meeting_id TEXT,
                analysis_json TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS note_sources (
                note_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (note_id, source_id)
            )""")
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
                USING fts5(note_id UNINDEXED, title, content, tokenize='unicode61')""")

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def automatic_sources(content: str) -> list[dict]:
        # Source IDs are stable until the user changes the underlying content.
        chunks = [line.strip() for line in content.splitlines() if line.strip()]
        if not chunks and content.strip():
            chunks = [content.strip()]
        return [{"id": f"n{index:06d}", "text": text[:1200]}
                for index, text in enumerate(chunks[:1000], 1)]

    def create(self, title: str, content: str, meeting_id: str | None = None,
               sources: list[dict] | None = None) -> dict:
        note_id = uuid4().hex
        timestamp = utc_now()
        sources = sources if sources is not None else self.automatic_sources(content)
        with self.connect() as db:
            db.execute("""INSERT INTO notes
                (id,title,content,source_meeting_id,version,created_at,updated_at)
                VALUES(?,?,?,?,1,?,?)""", (note_id, title, content, meeting_id, timestamp, timestamp))
            db.executemany("""INSERT INTO note_sources(note_id,source_id,text)
                VALUES(?,?,?)""", [(note_id, row["id"], row["text"]) for row in sources])
            db.execute("INSERT INTO notes_fts(note_id,title,content) VALUES(?,?,?)",
                       (note_id, title, content))
        return self.get(note_id)

    def get(self, note_id: str) -> dict | None:
        with self.connect() as db:
            note = db.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
            if note is None:
                return None
            sources = db.execute("""SELECT source_id AS id,text FROM note_sources
                WHERE note_id=? ORDER BY source_id""", (note_id,)).fetchall()
        result = dict(note)
        result["analysis"] = (json.loads(result.pop("analysis_json"))
                              if result.get("analysis_json") else None)
        result["sources"] = [dict(row) for row in sources]
        return result

    def list(self, query: str = "", offset: int = 0, limit: int = 25) -> dict:
        with self.connect() as db:
            term = query.strip()
            if term:
                # Quote the term as a literal FTS phrase, not arbitrary FTS syntax.
                tokens = re.findall(r"[\w\u0600-\u06ff]+", term, re.UNICODE)[:8]
                if not tokens:
                    return {"items": [], "total": 0}
                expression = " OR ".join('"' + token.replace('"', '') + '"' for token in tokens)
                items = db.execute("""SELECT n.id,n.title,n.source_meeting_id,
                    n.version,n.created_at,n.updated_at, substr(n.content,1,240) AS preview
                    FROM notes n JOIN notes_fts f ON f.note_id=n.id
                    WHERE notes_fts MATCH ? ORDER BY rank, n.updated_at DESC LIMIT ? OFFSET ?""",
                    (expression, limit, offset)).fetchall()
                count = db.execute("""SELECT count(*) FROM notes_fts
                    WHERE notes_fts MATCH ?""", (expression,)).fetchone()[0]
            else:
                items = db.execute("""SELECT id,title,source_meeting_id,version,
                    created_at,updated_at,substr(content,1,240) AS preview
                    FROM notes ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                    (limit, offset)).fetchall()
                count = db.execute("SELECT count(*) FROM notes").fetchone()[0]
        return {"items": [dict(row) for row in items], "total": count}

    def update(self, note_id: str, version: int, title: str, content: str) -> dict | None:
        with self.connect() as db:
            old = db.execute("SELECT version FROM notes WHERE id=?", (note_id,)).fetchone()
            if old is None:
                return None
            if old["version"] != version:
                raise NoteConflict("Note changed on another device")
            db.execute("""UPDATE notes SET title=?,content=?,analysis_json=NULL,
                version=version+1,updated_at=? WHERE id=? AND version=?""",
                (title, content, utc_now(), note_id, version))
            db.execute("DELETE FROM note_sources WHERE note_id=?", (note_id,))
            sources = self.automatic_sources(content)
            db.executemany("""INSERT INTO note_sources(note_id,source_id,text)
                VALUES(?,?,?)""", [(note_id, row["id"], row["text"]) for row in sources])
            db.execute("DELETE FROM notes_fts WHERE note_id=?", (note_id,))
            db.execute("INSERT INTO notes_fts(note_id,title,content) VALUES(?,?,?)",
                       (note_id, title, content))
        return self.get(note_id)

    def set_analysis(self, note_id: str, version: int, result: NotesAnalysis) -> dict:
        with self.connect() as db:
            cursor = db.execute("""UPDATE notes SET analysis_json=?,updated_at=?
                WHERE id=? AND version=?""",
                (json.dumps(result.model_dump(), ensure_ascii=False), utc_now(),
                 note_id, version))
            if cursor.rowcount != 1:
                raise NoteConflict("Note changed while AI was processing it")
        return self.get(note_id)

    def delete(self, note_id: str) -> bool:
        with self.connect() as db:
            exists = db.execute("SELECT 1 FROM notes WHERE id=?", (note_id,)).fetchone()
            if exists is None:
                return False
            db.execute("DELETE FROM note_sources WHERE note_id=?", (note_id,))
            db.execute("DELETE FROM notes_fts WHERE note_id=?", (note_id,))
            db.execute("DELETE FROM notes WHERE id=?", (note_id,))
        return True
