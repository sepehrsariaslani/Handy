"""Authenticated resumable meeting upload endpoints (4 MiB parts)."""
import asyncio
from datetime import date
from typing import Literal
from pathlib import Path
import os
import re

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .meeting_processing import validated_duration
from .upload_store import UploadStore, UploadConflict, UploadError, MAX_PART
from .workers import WorkerUnavailable

ID = re.compile(r"^[a-f0-9]{32}$")
REF = re.compile(r"^[A-Za-z0-9-]{1,80}$")

class BeginUpload(BaseModel):
    meeting_id: str = Field(min_length=1, max_length=80)
    total_bytes: int = Field(gt=44)
    language: str | None = Field(default=None, max_length=35)
    kind: Literal["meeting", "note"] = "meeting"
    asr_model_id: str | None = Field(default=None, min_length=1, max_length=512)
    title: str | None = Field(default=None, max_length=180)
    meeting_date: date | None = None

def attach_upload_routes(app: FastAPI, config, meetings, require_auth, runtime=None):
    uploads = UploadStore(Path(config.db_path))
    auth = [Depends(require_auth)]

    @app.post("/v1/meetings/uploads", dependencies=auth)
    def begin_upload(body: BeginUpload):
        if not REF.fullmatch(body.meeting_id):
            raise HTTPException(status_code=422, detail="Invalid meeting reference")
        if body.total_bytes > config.max_meeting_bytes:
            raise HTTPException(status_code=413, detail="Meeting exceeds limit")
        if body.asr_model_id and runtime is not None:
            try:
                runtime.validate_asr_model(body.asr_model_id)
            except WorkerUnavailable as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
        existing = meetings.by_client_ref(body.meeting_id)
        if existing:
            return {"id": None, "offset": body.total_bytes,
                    "total_bytes": body.total_bytes, "job_id": existing["id"]}
        try:
            row = uploads.start(body.meeting_id, body.total_bytes, body.language,
                                body.title, body.meeting_date.isoformat() if body.meeting_date else None,
                                body.kind, body.asr_model_id)
        except UploadError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return {"id": row["id"], "offset": row["received_bytes"],
                "total_bytes": row["total_bytes"], "job_id": row["job_id"]}

    @app.get("/v1/meetings/uploads/{uid}", dependencies=auth)
    def upload_state(uid: str):
        if not ID.fullmatch(uid) or (row := uploads.get(uid)) is None:
            raise HTTPException(status_code=404, detail="Upload not found")
        return {"id": uid, "offset": row["received_bytes"],
                "total_bytes": row["total_bytes"], "job_id": row["job_id"]}

    @app.put("/v1/meetings/uploads/{uid}", dependencies=auth)
    async def upload_part(
        uid: str, request: Request,
        offset: int = Header(alias="X-Upload-Offset", ge=0),
    ):
        if not ID.fullmatch(uid) or uploads.get(uid) is None:
            raise HTTPException(status_code=404, detail="Upload not found")
        payload = bytearray()
        async for part in request.stream():
            payload.extend(part)
            if len(payload) > MAX_PART:
                raise HTTPException(status_code=413, detail="Part exceeds 4 MiB")
        try:
            row = await asyncio.to_thread(uploads.append, uid, offset, bytes(payload))
        except UploadConflict as exc:
            raise HTTPException(status_code=409, detail={"next_offset": exc.offset}) from None
        except UploadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return {"offset": row["received_bytes"]}

    @app.post("/v1/meetings/uploads/{uid}/finish", dependencies=auth)
    def finalize_upload(uid: str):
        if not ID.fullmatch(uid):
            raise HTTPException(status_code=404, detail="Upload not found")
        # Single-process atomicity between the last PUT and finalize.
        with uploads.lock:
            row = uploads.get(uid)
            if row is None:
                raise HTTPException(status_code=404, detail="Upload not found")
            if row["job_id"]:
                return meetings.response(row["job_id"])
            if row["received_bytes"] != row["total_bytes"]:
                raise HTTPException(status_code=409, detail="Upload not complete")
            source = Path(row["path"])
            destination = meetings.root / (uid + ".wav")
            # Repair a crash between file rename and DB commit.
            if not source.exists() and destination.exists():
                source = destination
            if not source.exists():
                raise HTTPException(status_code=409, detail="Upload file missing")
            try:
                duration = validated_duration(source, config.max_meeting_seconds)
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid WAV PCM16 16kHz") from None
            if source != destination:
                os.replace(source, destination)
            created = meetings.create(
                destination, duration, row["client_ref"], row["language"],
                row["title"], row["meeting_date"], row["kind"], row["asr_model_id"]
            )
            uploads.completed(uid, created["id"], destination)
            return meetings.response(created["id"])
