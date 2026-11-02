"""Versioned shared API contracts; see src/lib/ai/contracts.ts."""
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
