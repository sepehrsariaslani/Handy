"""Versioned shared API contracts; mirror in src/lib/ai/contracts.ts."""
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Mode(str, Enum):
    dictation = "dictation"
    meeting = "meeting"
    note_taking = "note_taking"


class Tone(str, Enum):
    verbatim = "verbatim"
    professional = "professional"
    formal = "formal"
    casual = "casual"
    friendly = "friendly"


class SessionCreate(StrictModel):
    mode: Mode
    title: str | None = Field(default=None, max_length=180)
    language: str | None = Field(default=None, max_length=35)
    tone: Tone = Tone.verbatim


class SessionRecord(StrictModel):
    id: str
    mode: Mode
    status: Literal["created", "recording", "processing", "completed", "failed"]
    title: str | None
    language: str | None
    tone: Tone
    created_at: datetime


class TranscriptSegment(StrictModel):
    id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str | None = None
    raw_text: str
    clean_text: str | None = None
    final: bool = False

    @model_validator(mode="after")
    def valid_times(self):
        if self.end < self.start:
            raise ValueError("Segment end precedes start")
        return self


class Evidence(StrictModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)
    assignee: str | None = None
    due_date: str | None = None


class MeetingAnalysis(StrictModel):
    summary: str
    decisions: list[Evidence] = Field(default_factory=list)
    tasks: list[Evidence] = Field(default_factory=list)
    open_questions: list[Evidence] = Field(default_factory=list)


class NotesAnalysis(StrictModel):
    title: str
    sections: list[dict[str, str]] = Field(default_factory=list)
    key_points: list[Evidence] = Field(default_factory=list)
    review_questions: list[str] = Field(default_factory=list)


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    api_version: Literal["v1"] = "v1"


class CapabilitiesResponse(StrictModel):
    api_version: Literal["v1"] = "v1"
    modes: list[Mode] = Field(default_factory=lambda: list(Mode))
    processing: dict[str, bool] = Field(
        default_factory=lambda: {
            "asr": False,
            "llm": False,
            "speaker_diarization": False,
            "streaming": False,
        }
    )
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "asr": "whisper-large-v3",
            "llm": "qwen3.5-9b",
        }
    )


class ModelStatus(StrictModel):
    asr_model: str
    asr_ready: bool
    asr_device: str
    llm_model: str
    llm_ready: bool


class CleanupRequest(StrictModel):
    text: str = Field(min_length=1, max_length=50000)
    tone: Tone = Tone.professional
    language: str | None = Field(default=None, max_length=35)
    protected_terms: list[str] = Field(default_factory=list, max_length=40)


class CleanupResponse(StrictModel):
    raw_text: str
    clean_text: str
    tone: Tone
    model: str
    review_required: bool = False
    warnings: list[str] = Field(default_factory=list)


class TranscriptionResponse(StrictModel):
    raw_text: str
    clean_text: str | None
    language: str | None
    model: str
    segments: list[TranscriptSegment]
    cleanup: CleanupResponse | None = None
    elapsed_ms: int = Field(ge=0)


class MeetingUtterance(StrictModel):
    id: str
    speaker: str = "Unknown"
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class MeetingJob(StrictModel):
    id: str
    status: Literal["queued", "processing", "completed", "failed"]
    stage: str | None = None
    error: str | None = None
    duration: float = Field(ge=0)
    processed_seconds: float = Field(ge=0)
    transcript_count: int = Field(ge=0)
    transcript: list[MeetingUtterance] | None = None
    insights: MeetingAnalysis | None = None
