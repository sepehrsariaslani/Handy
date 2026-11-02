"""Crash-resilient, bounded PCM chunk ledger for opt-in streaming sessions."""
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
from datetime import datetime, timezone

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
MAX_CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 6
MAX_SESSION_SECONDS = 20 * 3600


class ChunkConflict(Exception):
    pass


class ChunkOutOfOrder(Exception):
    def __init__(self, expected: int):
        self.expected = expected


class ChunkLimit(Exception):
    pass


class StreamStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.root = db_path.parent / "stream_audio"
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS stream_chunks (
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                path TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                committed INTEGER NOT NULL DEFAULT 0,
                raw_text TEXT,
                clean_text TEXT,
                language TEXT,
                warnings_json TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, sequence)
            )""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def state(self, session_id: str, after: int = -1, limit: int = 100) -> dict:
        with self._connect() as db:
            rows = db.execute("""SELECT sequence, start_ms, duration_ms, raw_text,
                                clean_text, language, warnings_json
                                FROM stream_chunks WHERE session_id=? AND
                                committed=1 AND sequence>? ORDER BY sequence LIMIT ?""",
                              (session_id, after, min(200, max(1, limit)))).fetchall()
            all_committed = db.execute("""SELECT COUNT(*) FROM stream_chunks
                WHERE session_id=? AND committed=1""", (session_id,)).fetchone()[0]
            pending = db.execute("""SELECT MIN(sequence) FROM stream_chunks
                WHERE session_id=? AND committed=0""", (session_id,)).fetchone()[0]
            count = db.execute("SELECT COUNT(*) FROM stream_chunks WHERE session_id=?",
                               (session_id,)).fetchone()[0]
        return {
            "next_sequence": int(pending) if pending is not None else int(count),
            "pending_sequence": int(pending) if pending is not None else None,
            "committed_count": int(all_committed),
            "results": [self._result(row) for row in rows],
        }

    @staticmethod
    def _result(row: sqlite3.Row) -> dict:
        import json
        seq = int(row["sequence"])
        return {
            "type": "final",
            "sequence": seq,
            "segment": {
                "id": f"c{seq:08d}",
                "start": row["start_ms"] / 1000,
                "end": (row["start_ms"] + row["duration_ms"]) / 1000,
                "speaker": None,
                "raw_text": row["raw_text"] or "",
                "clean_text": row["clean_text"],
                "final": True,
            },
            "language": row["language"],
            "warnings": json.loads(row["warnings_json"] or "[]"),
        }

    def find(self, session_id: str, seq: int) -> sqlite3.Row | None:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM stream_chunks WHERE session_id=? AND sequence=?",
                (session_id, seq),
            ).fetchone()

    def accept(self, session_id: str, seq: int, pcm: bytes) -> tuple[str, dict | None]:
        if seq < 0 or seq >= 20000 or len(pcm) < 3200 or len(pcm) > MAX_CHUNK_BYTES or len(pcm) % 2:
            raise ChunkLimit("A chunk must hold 0.1-6.0 seconds of mono PCM16 at 16 kHz")
        fingerprint = hashlib.sha256(pcm).hexdigest()
        with self._connect() as db:
            previous = db.execute("SELECT * FROM stream_chunks WHERE session_id=? AND sequence=?",
                                  (session_id, seq)).fetchone()
            if previous:
                if previous["sha256"] != fingerprint:
                    raise ChunkConflict("Same sequence already has different audio")
                return "duplicate", self._result(previous) if previous["committed"] else None
            last = db.execute("""SELECT sequence, start_ms, duration_ms
                FROM stream_chunks WHERE session_id=? ORDER BY sequence DESC LIMIT 1""",
                (session_id,)).fetchone()
            expected = 0 if last is None else int(last["sequence"]) + 1
            if seq != expected:
                raise ChunkOutOfOrder(expected)
            start_ms = 0 if last is None else int(last["start_ms"]) + int(last["duration_ms"])
            duration_ms = round(len(pcm) * 1000 / (SAMPLE_RATE * BYTES_PER_SAMPLE))
            if start_ms + duration_ms > MAX_SESSION_SECONDS * 1000:
                raise ChunkLimit("This stream exceeds the 20-hour session limit")
            folder = self.root / session_id
            folder.mkdir(exist_ok=True)
            target = folder / f"{seq:08d}.pcm"
            fd, temp = tempfile.mkstemp(prefix=".chunk-", dir=folder)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(pcm)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temp, target)
                db.execute("""INSERT INTO stream_chunks
                    (session_id,sequence,sha256,path,start_ms,duration_ms,committed,created_at)
                    VALUES (?,?,?,?,?,?,0,?)""",
                    (session_id, seq, fingerprint, str(target), start_ms, duration_ms,
                     datetime.now(timezone.utc).isoformat()))
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        return "accepted", None

    def finish(self, session_id: str, seq: int, raw: str, clean: str | None,
               language: str | None, warnings: list[str]) -> dict:
        import json
        with self._connect() as db:
            db.execute("""UPDATE stream_chunks SET committed=1, raw_text=?, clean_text=?,
                language=?, warnings_json=? WHERE session_id=? AND sequence=?""",
                       (raw, clean, language, json.dumps(warnings), session_id, seq))
        item = self.find(session_id, seq)
        if item is None:
            raise RuntimeError("Committed chunk missing")
        return self._result(item)

    def previous_audio(self, session_id: str, seq: int, max_samples: int = 9600) -> bytes:
        if seq == 0:
            return b""
        prior = self.find(session_id, seq - 1)
        if prior is None:
            return b""
        with open(prior["path"], "rb") as reader:
            reader.seek(max(0, os.path.getsize(prior["path"]) - max_samples * 2))
            return reader.read()

    def raw_previous(self, session_id: str, seq: int) -> str:
        if seq == 0:
            return ""
        prior = self.find(session_id, seq - 1)
        return prior["raw_text"] or "" if prior and prior["committed"] else ""

    def audio(self, session_id: str, seq: int) -> bytes:
        item = self.find(session_id, seq)
        if item is None:
            raise FileNotFoundError()
        return Path(item["path"]).read_bytes()

    def complete(self, session_id: str) -> bool:
        with self._connect() as db:
            pending = db.execute("SELECT COUNT(*) FROM stream_chunks WHERE session_id=? AND committed=0",
                                 (session_id,)).fetchone()[0]
        return pending == 0
