"""Phase 7: memory search and cited question-answering from prior records."""
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .memory_answer import MemoryAnswerer
from .memory_store import MemoryStore, infer_utc_dates
from .workers import WorkerFailure, WorkerUnavailable


class MemoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=500)
    scope: Literal["all", "notes", "meetings"] = "all"
    since: date | None = None
    until: date | None = None

    @model_validator(mode="after")
    def valid_range(self):
        if self.since and self.until and self.since > self.until:
            raise ValueError("End date precedes start date")
        return self


class MemorySource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["note", "meeting"]
    document_id: str
    source_id: str
    title: str
    quote: str
    created_at: str
    speaker: str | None = None
    start: float | None = None


class MemoryResponse(BaseModel):
    question: str
    scope: Literal["all", "notes", "meetings"]
    status: Literal["answered", "search_only", "not_found"]
    answer: str | None
    citations: list[str]
    sources: list[MemorySource]
    warning: str | None = None
    applied_since: date | None = None
    applied_until: date | None = None


def attach_memory_routes(app: FastAPI, config, models, require_auth) -> None:
    store = MemoryStore(config.db_path)
    answerer = MemoryAnswerer(config, models.llm)
    protected = [Depends(require_auth)]

    @app.post("/v1/memory/search", dependencies=protected, response_model=MemoryResponse)
    def search_memory(body: MemoryQuery) -> MemoryResponse:
        since, until = infer_utc_dates(body.question, body.since, body.until)
        sources = store.search(body.question, body.scope, since, until)
        return MemoryResponse(
            question=body.question, scope=body.scope,
            status="search_only" if sources else "not_found",
            answer=None, citations=[], sources=sources,
            applied_since=since, applied_until=until,
        )

    @app.post("/v1/memory/ask", dependencies=protected, response_model=MemoryResponse)
    async def ask_memory(body: MemoryQuery) -> MemoryResponse:
        # Retrieval is always first. No available evidence -> no model call.
        since, until = infer_utc_dates(body.question, body.since, body.until)
        sources = store.search(body.question, body.scope, since, until)
        if not sources:
            return MemoryResponse(
                question=body.question, scope=body.scope,
                status="not_found", answer=None, citations=[], sources=[],
                applied_since=since, applied_until=until,
            )
        try:
            answer, cited = await answerer.answer(body.question, sources)
        except WorkerUnavailable:
            return MemoryResponse(
                question=body.question, scope=body.scope,
                status="search_only", answer=None, citations=[], sources=sources,
                warning="Qwen is not configured. Displaying source excerpts only.",
                applied_since=since, applied_until=until,
            )
        except WorkerFailure:
            raise HTTPException(
                status_code=502,
                detail="The AI answer failed; source search remains available",
            ) from None
        if not answer:
            return MemoryResponse(
                question=body.question, scope=body.scope,
                status="not_found", answer=None, citations=[], sources=sources,
                warning="No answer with verifiable source IDs was returned.",
                applied_since=since, applied_until=until,
            )
        # Cite only the evidence actually used by the answer, but make the
        # retrieved alternatives visible to help users check contradictions.
        return MemoryResponse(
            question=body.question, scope=body.scope,
            status="answered", answer=answer,
            citations=cited, sources=sources,
            applied_since=since, applied_until=until,
        )
