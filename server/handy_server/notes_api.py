"""Authenticated notes, search, edit, analysis and explicit meeting-to-note import."""
from datetime import datetime
import json
import re

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .notes_analysis import NoteAnalyzer
from .notes_store import NoteConflict, NoteStore
from .schemas import NotesAnalysis
from .workers import WorkerFailure, WorkerUnavailable

ID = re.compile(r"^[0-9a-f]{32}$")


class NoteCreate(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    content: str = Field(min_length=1, max_length=12000)


class NoteUpdate(NoteCreate):
    version: int = Field(ge=1)


def attach_notes_routes(app: FastAPI, config, meetings, runtime, require_auth):
    store = NoteStore(config.db_path)
    analyzer = NoteAnalyzer(config, runtime.llm)
    protected = [Depends(require_auth)]

    @app.get("/v1/notes", dependencies=protected)
    def search_notes(q: str = "", offset: int = 0, limit: int = 25):
        if len(q) > 160 or offset < 0 or limit < 1 or limit > 100:
            raise HTTPException(status_code=422, detail="Invalid search or page")
        return store.list(q, offset, limit)

    @app.post("/v1/notes", dependencies=protected, status_code=201)
    def create_note(body: NoteCreate):
        return store.create(body.title.strip(), body.content)

    @app.post("/v1/notes/from-meeting/{job_id}", dependencies=protected, status_code=201)
    def import_meeting(job_id: str):
        if not ID.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="Meeting not found")
        job = meetings.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Meeting not found")
        if job["status"] != "completed" or not job["insights"]:
            raise HTTPException(status_code=409, detail="Meeting analysis not completed")
        from .schemas import MeetingAnalysis
        insights = MeetingAnalysis.model_validate(json.loads(job["insights"]))
        text = ["خلاصه جلسه", insights.summary]
        known_ids = set()
        for heading, rows in (
            ("تصمیم‌ها", insights.decisions),
            ("وظایف", insights.tasks),
            ("سؤالات باز", insights.open_questions),
        ):
            text.append(heading)
            for item in rows:
                sources = ", ".join(item.source_ids[:8])
                known_ids.update(item.source_ids[:8])
                detail = item.text
                if item.assignee:
                    detail += " | مسئول: " + item.assignee
                if item.due_date:
                    detail += " | موعد: " + item.due_date
                text.append("- " + detail + (" [" + sources + "]" if sources else ""))
        content = "\n".join(text)
        if len(content) > 12000:
            raise HTTPException(status_code=413, detail="Meeting summary too large to import")
        source_rows = []
        if known_ids:
            # Query only cited utterances, not every transcript page in a 20h meeting.
            with meetings.connect() as db:
                for batch_start in range(0, len(known_ids), 200):
                    ids = sorted(known_ids)[batch_start:batch_start+200]
                    marks = ",".join("?" for _ in ids)
                    records = db.execute(
                        "SELECT id,text FROM meeting_segments WHERE job_id=? AND id IN (" + marks + ")",
                        [job_id, *ids],
                    ).fetchall()
                    source_rows.extend(dict(row) for row in records)
        title = "جلسه " + job["created_at"][:10]
        return store.create(title, content, meeting_id=job_id, sources=source_rows)

    def required(note_id: str):
        if not ID.fullmatch(note_id):
            raise HTTPException(status_code=404, detail="Note not found")
        note = store.get(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="Note not found")
        return note

    @app.get("/v1/notes/{note_id}", dependencies=protected)
    def get_note(note_id: str):
        return required(note_id)

    @app.put("/v1/notes/{note_id}", dependencies=protected)
    def edit_note(note_id: str, body: NoteUpdate):
        required(note_id)
        try:
            return store.update(note_id, body.version, body.title.strip(), body.content)
        except NoteConflict:
            raise HTTPException(status_code=409, detail="Note was modified elsewhere") from None

    @app.delete("/v1/notes/{note_id}", dependencies=protected, status_code=204)
    def delete_note(note_id: str):
        required(note_id)
        store.delete(note_id)
        return None

    @app.post("/v1/notes/{note_id}/analyze", dependencies=protected)
    async def analyze_note(note_id: str):
        note = required(note_id)
        try:
            result = await analyzer.analyze(note)
            return store.set_analysis(note_id, note["version"], result)
        except WorkerUnavailable:
            raise HTTPException(status_code=503, detail="Qwen worker is unavailable") from None
        except WorkerFailure:
            raise HTTPException(status_code=502, detail="Qwen worker failed") from None
        except NoteConflict:
            raise HTTPException(status_code=409, detail="Note was edited during analysis") from None
        except ValueError:
            raise HTTPException(status_code=422, detail="Note too large for analysis") from None
