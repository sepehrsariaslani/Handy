"""Durable, single-process meeting jobs and incremental transcript checkpoints."""
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .schemas import MeetingAnalysis

def now():
    return datetime.now(timezone.utc).isoformat()

class MeetingStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.root = db_path.parent / "meetings"
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS meeting_jobs (
                id TEXT PRIMARY KEY, client_ref TEXT UNIQUE, audio_path TEXT NOT NULL,
                status TEXT NOT NULL, stage TEXT NOT NULL, error TEXT,
                duration REAL NOT NULL, next_sample INTEGER NOT NULL DEFAULT 0,
                language TEXT, insights TEXT, title TEXT, meeting_date TEXT,
                asr_model_id TEXT, kind TEXT NOT NULL DEFAULT 'meeting', generated_note_id TEXT,
                 created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(meeting_jobs)")}
            for column in ("title", "meeting_date", "kind", "generated_note_id", "asr_model_id"):
                if column not in columns:
                    default = " NOT NULL DEFAULT 'meeting'" if column == "kind" else ""
                    db.execute(f"ALTER TABLE meeting_jobs ADD COLUMN {column} TEXT" + default)
            db.execute("""CREATE TABLE IF NOT EXISTS meeting_cleaned_blocks (
                job_id TEXT NOT NULL, block_index INTEGER NOT NULL,
                text TEXT NOT NULL, PRIMARY KEY(job_id, block_index)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS meeting_segments (
                job_id TEXT NOT NULL, id TEXT NOT NULL,
                speaker TEXT NOT NULL, start REAL NOT NULL,
                end REAL NOT NULL, text TEXT NOT NULL,
                PRIMARY KEY(job_id, id),
                FOREIGN KEY(job_id) REFERENCES meeting_jobs(id)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS meeting_speaker_names (
                job_id TEXT NOT NULL, speaker TEXT NOT NULL,
                name TEXT NOT NULL, PRIMARY KEY(job_id, speaker),
                FOREIGN KEY(job_id) REFERENCES meeting_jobs(id)
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS meeting_segment_seek
                ON meeting_segments(job_id, start, id)""")
            # Unfinished jobs survive app restarts; no in-memory queue is authoritative.
            db.execute("UPDATE meeting_jobs SET status='queued', stage='resuming' WHERE status='processing'")

    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def create(self, path: Path, duration: float, client_ref: str | None, language: str | None,
               title: str | None = None, meeting_date: str | None = None,
               kind: str = "meeting", asr_model_id: str | None = None) -> dict:
        job_id = uuid4().hex
        with self.connect() as db:
            if client_ref:
                prior = db.execute("SELECT * FROM meeting_jobs WHERE client_ref=?", (client_ref,)).fetchone()
                if prior:
                    return dict(prior)
            db.execute("""INSERT INTO meeting_jobs
                (id, client_ref, audio_path, status, stage, duration, next_sample, language,
                 title, meeting_date, kind, asr_model_id, created_at, updated_at)
                 VALUES(?,?,?,'queued','queued',?,0,?,?,?,?,?,?,?)""",
                (job_id, client_ref, str(path), duration, language,
                 title, meeting_date, kind, asr_model_id, now(), now()))
        return self.get(job_id)

    def by_client_ref(self, value: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM meeting_jobs WHERE client_ref=?", (value,)).fetchone()
        return dict(row) if row else None

    def get(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM meeting_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def claim(self) -> dict | None:
        with self.connect() as db:
            row = db.execute("""SELECT * FROM meeting_jobs WHERE status='queued'
                ORDER BY created_at LIMIT 1""").fetchone()
            if not row:
                return None
            result = db.execute("""UPDATE meeting_jobs SET status='processing', stage='transcribing',
                error=NULL, updated_at=? WHERE id=? AND status='queued'""", (now(), row["id"]))
            if result.rowcount != 1:
                return None
        return self.get(row["id"])

    def stage(self, job_id: str, stage: str):
        with self.connect() as db:
            db.execute("UPDATE meeting_jobs SET stage=?,updated_at=? WHERE id=?",
                       (stage, now(), job_id))

    def checkpoint(self, job_id: str, next_sample: int, segments: list[dict]):
        # Offset and segments are committed in one transaction: crash-safe resume.
        with self.connect() as db:
            for entry in segments:
                db.execute("""INSERT OR REPLACE INTO meeting_segments
                  (job_id,id,speaker,start,end,text) VALUES (?,?,?,?,?,?)""",
                  (job_id, entry["id"], entry["speaker"], entry["start"], entry["end"], entry["text"]))
            db.execute("""UPDATE meeting_jobs SET next_sample=?,stage='transcribing',
                updated_at=? WHERE id=?""",(next_sample, now(), job_id))

    def update_speakers(self, job_id: str, updates: list[tuple[str, str]]):
        with self.connect() as db:
            db.executemany("""UPDATE meeting_segments SET speaker=?
                WHERE job_id=? AND id=?""", ((speaker,job_id,id) for id,speaker in updates))

    def replace_aligned(self, job_id: str, replacements: list[tuple[str, list[dict]]]):
        """Atomically split original ASR blocks, preserving words and stable IDs."""
        with self.connect() as db:
            for original_id, parts in replacements:
                if len(parts) < 2:
                    continue
                original = db.execute(
                    "SELECT text FROM meeting_segments WHERE job_id=? AND id=?",
                    (job_id, original_id),
                ).fetchone()
                if not original or " ".join(p["text"] for p in parts) != " ".join(original["text"].split()):
                    raise ValueError("Aligned words must preserve every original token")
                db.execute("DELETE FROM meeting_segments WHERE job_id=? AND id=?",
                           (job_id, original_id))
                for part in parts:
                    db.execute("""INSERT INTO meeting_segments
                        (job_id,id,speaker,start,end,text) VALUES(?,?,?,?,?,?)""",
                        (job_id, part["id"], part["speaker"], part["start"],
                         part["end"], part["text"]))

    def speaker_names(self, job_id: str) -> dict[str, str]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT speaker,name FROM meeting_speaker_names WHERE job_id=?",
                (job_id,),
            ).fetchall()
        return {r["speaker"]: r["name"] for r in rows}

    def rename_speaker(self, job_id: str, speaker: str, name: str):
        with self.connect() as db:
            job = db.execute("SELECT status FROM meeting_jobs WHERE id=?",
                             (job_id,)).fetchone()
            if not job or job["status"] != "completed":
                raise ValueError("Meeting must be completed")
            if not db.execute(
                "SELECT 1 FROM meeting_segments WHERE job_id=? AND speaker=? LIMIT 1",
                (job_id, speaker),
            ).fetchone():
                raise ValueError("Speaker not found in meeting")
            if name:
                db.execute("""INSERT INTO meeting_speaker_names(job_id,speaker,name)
                    VALUES(?,?,?) ON CONFLICT(job_id,speaker) DO UPDATE SET name=excluded.name""",
                    (job_id, speaker, name))
            else:
                db.execute("DELETE FROM meeting_speaker_names WHERE job_id=? AND speaker=?",
                           (job_id, speaker))

    def correct_speaker(self, job_id: str, segment_id: str, speaker: str):
        with self.connect() as db:
            job = db.execute("SELECT status FROM meeting_jobs WHERE id=?",
                             (job_id,)).fetchone()
            if not job or job["status"] != "completed":
                raise ValueError("Meeting must be completed")
            result = db.execute(
                "UPDATE meeting_segments SET speaker=? WHERE job_id=? AND id=?",
                (speaker, job_id, segment_id),
            )
            if result.rowcount != 1:
                raise KeyError("Transcript segment not found")

    def segments(self, job_id: str, after: int=0, limit: int=150) -> list[dict]:
        with self.connect() as db:
            rows=db.execute("""SELECT id,speaker,start,end,text FROM meeting_segments
               WHERE job_id=? ORDER BY start,id LIMIT ? OFFSET ?""",
               (job_id, limit, after)).fetchall()
        return [dict(row) for row in rows]

    def count(self, job_id: str) -> int:
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM meeting_segments WHERE job_id=?", (job_id,)).fetchone()[0]

    def save_cleaned(self, job_id: str, parts: list[str]):
        """Persist cleaned transcript before optional Qwen meeting analysis.

        An unavailable analyzer must never hide successful Whisper output.
        """
        with self.connect() as db:
            for index, part in enumerate(parts):
                db.execute("""INSERT OR REPLACE INTO meeting_cleaned_blocks
                    (job_id,block_index,text) VALUES(?,?,?)""",
                    (job_id, index, part))

    def cleaned(self, job_id: str, offset: int = 0, limit: int = 10) -> dict:
        with self.connect() as db:
            records = db.execute("""SELECT text FROM meeting_cleaned_blocks
                WHERE job_id=? ORDER BY block_index LIMIT ? OFFSET ?""",
                (job_id, limit, offset)).fetchall()
            total = db.execute("SELECT COUNT(*) FROM meeting_cleaned_blocks WHERE job_id=?",
                               (job_id,)).fetchone()[0]
        return {"items": [row["text"] for row in records], "offset": offset, "total": total}

    def complete(self, job_id: str, insights: MeetingAnalysis | None,
                 cleaned_blocks: list[str] | None = None):
        # One transaction commits the completed job, its cleaned transcript,
        # automatic notes, and any evidence-backed meeting tasks together.
        from .task_store import TaskStore
        from .notes_store import NoteStore
        tasks = TaskStore(self.db_path)
        notes = NoteStore(self.db_path)
        payload = json.dumps(insights.model_dump(), ensure_ascii=False) if insights else None
        with self.connect() as db:
            row = db.execute("SELECT * FROM meeting_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("Meeting not found")
            cleaned_blocks = cleaned_blocks or []
            suggestion = (insights.title or "").strip() if insights else ""
            summary = (insights.summary or "").strip() if insights else ""
            first_text = next((part.strip() for part in cleaned_blocks if part.strip()), "")
            title = row["title"] or suggestion[:180] or (
                first_text.split("\n")[0][:90] if row["kind"] == "note" else ""
            ) or summary.split("\n")[0][:120] or "Meeting"
            proposed_date = (insights.meeting_date or "").strip() if insights else ""
            try:
                inferred_date = date.fromisoformat(proposed_date).isoformat()
            except ValueError:
                inferred_date = row["created_at"][:10]
            date_value = row["meeting_date"] or inferred_date
            for idx, part in enumerate(cleaned_blocks):
                db.execute("""INSERT OR REPLACE INTO meeting_cleaned_blocks
                    (job_id,block_index,text) VALUES(?,?,?)""", (job_id, idx, part))
            note_id = row["generated_note_id"]
            if row["kind"] == "note" and first_text and not note_id:
                # Preserve the entire transcription in several editable notes
                # rather than silently truncating recordings longer than 12k.
                full_text = "\n\n".join(part for part in cleaned_blocks if part.strip())
                pieces = [full_text[idx:idx+11000] for idx in range(0,len(full_text),11000)]
                for idx, part in enumerate(pieces):
                    suffix = f" ({idx+1}/{len(pieces)})" if len(pieces) > 1 else ""
                    new_note = notes.create(title[:180-len(suffix)] + suffix,
                                            part, meeting_id=job_id, connection=db)
                    if idx == 0:
                        note_id = new_note["id"]
            db.execute("""UPDATE meeting_jobs SET status='completed',stage='completed',
                title=?,meeting_date=?,insights=?,generated_note_id=?,
                error=NULL,updated_at=? WHERE id=?""",
                (title,date_value,payload,note_id,now(),job_id))
            if row["kind"] == "meeting" and insights:
                tasks.import_meeting(job_id, connection=db)

    def fail(self, job_id: str, reason: str):
        with self.connect() as db:
            db.execute("""UPDATE meeting_jobs SET status='failed',stage='failed',
               error=?,updated_at=? WHERE id=?""",(reason[:220],now(),job_id))

    def retry(self, job_id: str) -> dict:
        with self.connect() as db:
            result=db.execute("""UPDATE meeting_jobs SET status='queued',stage='queued',
                error=NULL,updated_at=? WHERE id=? AND status='failed'""",(now(),job_id))
            if result.rowcount!=1:
                raise ValueError("Only failed jobs can be retried")
        return self.get(job_id)

    def response(self, job_id: str, limit: int = 150) -> dict | None:
        job=self.get(job_id)
        if job is None:
            return None
        cleaned_page = self.cleaned(job_id, limit=1)
        cleaned = cleaned_page["items"]
        return {
            "id": job_id, "title": job["title"], "meeting_date": job["meeting_date"],
            "kind": job["kind"], "generated_note_id": job["generated_note_id"],
            "asr_model_id": job["asr_model_id"],
            "clean_text": cleaned[0] if cleaned else None,
            "cleaned_block_count": cleaned_page["total"],
            "status": job["status"], "stage": job["stage"],
            "error": job["error"], "duration": job["duration"],
            "processed_seconds": min(job["duration"], job["next_sample"]/16000),
            "transcript_count": self.count(job_id),
            "transcript": self.segments(job_id,limit=limit),
            "speaker_names": self.speaker_names(job_id),
            "insights": json.loads(job["insights"]) if job["insights"] else None,
        }
