"""Persistent bounded multi-turn conversations, one conversation per explicit session."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class ConversationConflict(Exception):
    pass


class ConversationStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memory_conversations(
                id TEXT PRIMARY KEY,title TEXT NOT NULL,
                created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
                turn_count INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS memory_turns(
                id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,question TEXT NOT NULL,
                answer TEXT,status TEXT NOT NULL,
                citations TEXT NOT NULL,created_at TEXT NOT NULL,
                UNIQUE(conversation_id,sequence),
                FOREIGN KEY(conversation_id) REFERENCES memory_conversations(id))""")
            db.execute("""CREATE INDEX IF NOT EXISTS memory_turns_history
                ON memory_turns(conversation_id,sequence)""")

    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def create(self, title: str = "New conversation"):
        cid = uuid4().hex
        stamp = now()
        with self.connect() as db:
            db.execute("""INSERT INTO memory_conversations(id,title,created_at,updated_at)
                VALUES(?,?,?,?)""",(cid,title[:180],stamp,stamp))
        return self.get(cid)

    def get(self, cid: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM memory_conversations WHERE id=?",(cid,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 50):
        with self.connect() as db:
            rows = db.execute("""SELECT * FROM memory_conversations
                ORDER BY updated_at DESC LIMIT ?""",(limit,)).fetchall()
        return [dict(row) for row in rows]

    def turns(self, cid: str, limit: int = 50):
        with self.connect() as db:
            rows = db.execute("""SELECT * FROM (
                SELECT * FROM memory_turns WHERE conversation_id=?
                ORDER BY sequence DESC LIMIT ?) ORDER BY sequence""",(cid,limit)).fetchall()
        result=[]
        for row in rows:
            entry=dict(row)
            entry["citations"]=json.loads(entry["citations"])
            result.append(entry)
        return result

    def append(self, cid: str, expected_turns: int, question: str, answer: str | None,
               status: str, citations: list[str]):
        with self.connect() as db:
            changed = db.execute("""UPDATE memory_conversations
                SET turn_count=turn_count+1,updated_at=?,title=CASE WHEN turn_count=0
                   THEN ? ELSE title END
                WHERE id=? AND turn_count=?""",(
                    now(),question[:100],cid,expected_turns
                ))
            if changed.rowcount != 1:
                raise ConversationConflict("Conversation changed during answer generation")
            db.execute("""INSERT INTO memory_turns
                (id,conversation_id,sequence,question,answer,status,citations,created_at)
                VALUES(?,?,?,?,?,?,?,?)""",(
                    uuid4().hex,cid,expected_turns+1,question,answer,status,
                    json.dumps(citations,ensure_ascii=False),now()
                ))
        return self.get(cid)

    def delete(self, cid: str):
        with self.connect() as db:
            db.execute("DELETE FROM memory_turns WHERE conversation_id=?",(cid,))
            return bool(db.execute("DELETE FROM memory_conversations WHERE id=?",(cid,)).rowcount)
