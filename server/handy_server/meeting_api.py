"""Authenticated, disk-streamed meeting upload and progress/pagination API."""
from pathlib import Path
from datetime import date
from typing import Literal
import re
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import Field
from .schemas import StrictModel

from .meeting_processing import validated_duration
from .schemas import MeetingJob
from .workers import WorkerUnavailable

REF = re.compile(r"^[A-Za-z0-9-]{1,80}$")
ID = re.compile(r"^[a-f0-9]{32}$")
SPEAKER = re.compile(r"^(SPEAKER_[0-9]{2,}|MANUAL_[0-9]{2,}|Unknown)$")
SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class SpeakerNamePatch(StrictModel):
    name: str = Field(max_length=80)


class SegmentSpeakerPatch(StrictModel):
    speaker: str = Field(min_length=1, max_length=80)


def attach_meeting_routes(app: FastAPI, config, store, processor, require_auth):
    authorized = [Depends(require_auth)]

    @app.post("/v1/meetings", response_model=MeetingJob,
              status_code=202, dependencies=authorized)
    async def submit_meeting(
        audio: UploadFile = File(...),
        meeting_id: str | None = Form(default=None),
        language: str | None = Form(default=None),
        kind: Literal["meeting", "note"] = Form(default="meeting"),
        asr_model_id: str | None = Form(default=None, min_length=1, max_length=512),
        title: str | None = Form(default=None, max_length=180),
        meeting_date: date | None = Form(default=None),
    ):
        if asr_model_id is not None:
            try:
                processor.runtime.validate_asr_model(asr_model_id)
            except WorkerUnavailable as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
        if meeting_id is not None and not REF.fullmatch(meeting_id):
            raise HTTPException(status_code=422, detail="Invalid meeting reference")
        if language is not None and len(language) > 35:
            raise HTTPException(status_code=422, detail="Invalid language code")
        if meeting_id:
            prior = await __import__("asyncio").to_thread(store.by_client_ref, meeting_id)
            if prior is not None:
                await audio.close()
                return store.response(prior["id"])
        if not await processor.runtime.asr_ready():
            raise HTTPException(status_code=503, detail="Whisper Large V3 is not ready")
        if Path(audio.filename or "").suffix.lower() != ".wav":
            raise HTTPException(status_code=415, detail="Meeting audio must be WAV PCM16")
        location = store.root / (uuid4().hex + ".wav")
        length = 0
        try:
            with location.open("xb") as file:
                while data := await audio.read(1024 * 1024):
                    length += len(data)
                    if length > config.max_meeting_bytes:
                        raise HTTPException(status_code=413, detail="Meeting exceeds upload limit")
                    file.write(data)
                file.flush()
            try:
                duration = await __import__("asyncio").to_thread(
                    validated_duration, location, config.max_meeting_seconds
                )
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid 16kHz mono PCM16 meeting WAV") from None
            created = await __import__("asyncio").to_thread(
                store.create, location, duration, meeting_id, language,
                title, meeting_date.isoformat() if meeting_date else None, kind, asr_model_id
            )
            if created["audio_path"] != str(location):
                location.unlink(missing_ok=True)
            return store.response(created["id"])
        except BaseException:
            # Do not retain partial audio after network cancellation or error.
            # The original recording remains in Handy's on-device directory.
            if location.exists() and store.by_client_ref(meeting_id) is None:
                location.unlink(missing_ok=True)
            raise
        finally:
            await audio.close()

    @app.get("/v1/meetings/{job_id}", response_model=MeetingJob, dependencies=authorized)
    def get_meeting_job(job_id: str):
        if not ID.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="Meeting not found")
        result = store.response(job_id)
        if not result:
            raise HTTPException(status_code=404, detail="Meeting not found")
        return result

    @app.get("/v1/meetings/{job_id}/transcript", dependencies=authorized)
    def get_transcript(job_id: str, offset: int = 0, limit: int = 150):
        if not ID.fullmatch(job_id) or store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Meeting not found")
        if offset < 0 or limit < 1 or limit > 500:
            raise HTTPException(status_code=422, detail="Invalid pagination")
        return {
            "items": store.segments(job_id, offset, limit),
            "offset": offset,
            "total": store.count(job_id),
        }

    @app.get("/v1/meetings/{job_id}/cleaned", dependencies=authorized)
    def get_cleaned_transcript(job_id: str, offset: int = 0, limit: int = 10):
        if not ID.fullmatch(job_id) or store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Meeting not found")
        if offset < 0 or not 1 <= limit <= 20:
            raise HTTPException(status_code=422, detail="Invalid pagination")
        return store.cleaned(job_id, offset, limit)

    @app.put("/v1/meetings/{job_id}/speakers/{speaker}", dependencies=authorized)
    def update_speaker_name(job_id: str, speaker: str, body: SpeakerNamePatch):
        if not ID.fullmatch(job_id) or store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Meeting not found")
        if not SPEAKER.fullmatch(speaker) or speaker == "Unknown":
            raise HTTPException(status_code=422, detail="Invalid speaker ID")
        name = body.name.strip()
        if any(ord(ch) < 32 for ch in name):
            raise HTTPException(status_code=422, detail="Invalid speaker name")
        try:
            store.rename_speaker(job_id, speaker, name)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return {"speaker_names": store.speaker_names(job_id)}

    @app.put("/v1/meetings/{job_id}/transcript/{segment_id}/speaker",
             dependencies=authorized)
    def correct_segment_speaker(job_id: str, segment_id: str,
                                body: SegmentSpeakerPatch):
        if not ID.fullmatch(job_id) or store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Meeting not found")
        if not SEGMENT.fullmatch(segment_id) or not SPEAKER.fullmatch(body.speaker):
            raise HTTPException(status_code=422, detail="Invalid segment or speaker ID")
        try:
            store.correct_speaker(job_id, segment_id, body.speaker)
        except KeyError:
            raise HTTPException(status_code=404, detail="Segment not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return {"id": segment_id, "speaker": body.speaker}

    @app.post("/v1/meetings/{job_id}/retry", response_model=MeetingJob,
              dependencies=authorized)
    def retry_meeting_job(job_id: str):
        if not ID.fullmatch(job_id) or store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Meeting not found")
        try:
            store.retry(job_id)
        except ValueError:
            raise HTTPException(status_code=409, detail="Only failed meetings may be retried") from None
        return store.response(job_id)
