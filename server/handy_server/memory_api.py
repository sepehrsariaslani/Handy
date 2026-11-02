"""Phase 7: memory search and cited question-answering from prior records."""
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .memory_answer import MemoryAnswerer
from .memory_store import MemoryStore
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


def attach_memory_routes(app: FastAPI, config, models, require_auth) -> None:
    store = MemoryStore(config.db_path)
    answerer = MemoryAnswerer(config, models.llm)
    protected = [Depends(require_auth)]

    @app.post("/v1/memory/search", dependencies=protected, response_model=MemoryResponse)
    def search_memory(body: MemoryQuery) -> MemoryResponse:
        sources = store.search(
            body.question, body.scope, body.since, body.until,
        )
        return MemoryResponse(
            question=body.question, scope=body.scope,
            status="search_only" if sources else "not_found",
            answer=None, citations=[], sources=sources,
        )

    @app.post("/v1/memory/ask", dependencies=protected, response_model=MemoryResponse)
    async def ask_memory(body: MemoryQuery) -> MemoryResponse:
        # Retrieval is always first. No available evidence -> no model call.
        sources = store.search(
            body.question, body.scope, body.since, body.until,
        )
        if not sources:
            return MemoryResponse(
                question=body.question, scope=body.scope,
                status="not_found", answer=None, citations=[], sources=[],
            )
        try:
            answer, cited = await answerer.answer(body.question, sources)
        except WorkerUnavailable:
            return MemoryResponse(
                question=body.question, scope=body.scope,
                status="search_only", answer=None, citations=[], sources=sources,
                warning="Qwen is not configured. Displaying source excerpts only.",
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
            )
        # Cite only the evidence actually used by the answer, but make the
        # retrieved alternatives visible to help users check contradictions.
        return MemoryResponse(
            question=body.question, scope=body.scope,
            status="answered", answer=answer,
            citations=cited, sources=sources,
        )
