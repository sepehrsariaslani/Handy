"""Phase 8: hybrid semantic retrieval, durable conversations and cited answers."""
import asyncio
from datetime import date
import re
import sqlite3
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conversation_store import ConversationStore, ConversationConflict
from .memory_answer import MemoryAnswerer
from .memory_store import MemoryStore, infer_utc_dates, tokens
from .semantic_index import SemanticIndex, SemanticUnavailable
from .workers import WorkerFailure, WorkerUnavailable


ID = re.compile(r"^[0-9a-f]{32}$")


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
    retrieval: Literal["lexical", "hybrid"] = "lexical"


class NewConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="New conversation", min_length=1, max_length=180)


class ConversationTurn(MemoryQuery):
    expected_turns: int = Field(ge=0)


def attach_memory_routes(app: FastAPI, config, models, require_auth) -> None:
    store = MemoryStore(config.db_path)
    answerer = MemoryAnswerer(config, models.llm)
    conversations = ConversationStore(config.db_path)
    semantic = SemanticIndex(
        config.db_path, config.semantic_enabled, config.semantic_model,
    )
    protected = [Depends(require_auth)]

    async def retrieve(question, scope, since, until, history):
        # Context from previous user questions resolves "what about its date?"
        # without allowing the previous AI answer to supply new evidence.
        retrieval_text = question
        if history:
            last = history[-1]["question"]
            if len(tokens(question)) < 4:
                retrieval_text += " " + last[:350]
        lexical = await asyncio.to_thread(
            store.search, retrieval_text, scope, since, until, 12,
        )
        mode = "lexical"
        if semantic.enabled:
            try:
                ranked = await asyncio.to_thread(semantic.search, retrieval_text, 12)
                # Cross-check each stored vector's document is still visible.
                verified = await asyncio.to_thread(
                    store.resolve, [key for key, _ in ranked],
                    scope, since, until,
                )
                mode = "hybrid"
                # Keep some lexical hits and some semantic results; the latter
                # can recover paraphrases that share no exact query token.
                ids = set()
                combined = []
                for hit in lexical[:6] + verified[:8] + lexical[6:]:
                    if hit["id"] not in ids:
                        ids.add(hit["id"])
                        combined.append(hit)
                    if len(combined) >= 12:
                        break
                return combined, mode
            except SemanticUnavailable:
                pass
        return lexical, mode

    async def evaluate(body: MemoryQuery, history=None, search_only=False):
        history = history or []
        since, until = infer_utc_dates(body.question, body.since, body.until)
        sources, mode = await retrieve(
            body.question, body.scope, since, until, history,
        )
        metadata = dict(
            question=body.question, scope=body.scope,
            sources=sources, applied_since=since, applied_until=until,
            retrieval=mode,
        )
        if not sources:
            return MemoryResponse(status="not_found",answer=None,citations=[],**metadata)
        if search_only:
            return MemoryResponse(status="search_only",answer=None,citations=[],**metadata)
        try:
            answer, cited = await answerer.answer(
                body.question, sources, history=history[-3:],
            )
        except WorkerUnavailable:
            return MemoryResponse(
                status="search_only", answer=None, citations=[],
                warning="Qwen is not configured. Displaying evidence only.",
                **metadata,
            )
        except WorkerFailure:
            raise HTTPException(
                status_code=502, detail="AI answer failed; source search is available"
            ) from None
        if not answer:
            return MemoryResponse(
                status="not_found", answer=None, citations=[],
                warning="No answer with verifiable sources was returned.",
                **metadata,
            )
        return MemoryResponse(status="answered",answer=answer,citations=cited,**metadata)

    @app.post("/v1/memory/search", dependencies=protected, response_model=MemoryResponse)
    async def search_memory(body: MemoryQuery):
        return await evaluate(body, search_only=True)

    @app.post("/v1/memory/ask", dependencies=protected, response_model=MemoryResponse)
    async def ask_memory(body: MemoryQuery):
        return await evaluate(body)

    @app.get("/v1/memory/semantic/status", dependencies=protected)
    def semantic_status():
        with sqlite3.connect(config.db_path) as db:
            count = db.execute("""SELECT COUNT(*) FROM semantic_vectors
                WHERE model=?""",(semantic.model_name,)).fetchone()[0]
        return {"enabled":semantic.enabled,"model":semantic.model_name,"indexed":count}

    @app.post("/v1/memory/semantic/sync", dependencies=protected)
    async def sync_semantic(limit: int = 100):
        if not semantic.enabled:
            raise HTTPException(status_code=409,detail="Semantic indexing disabled")
        if not 1 <= limit <= 200:
            raise HTTPException(status_code=422,detail="Invalid batch limit")
        try:
            return await asyncio.to_thread(semantic.sync, limit)
        except SemanticUnavailable:
            raise HTTPException(
                status_code=503,detail="Optional CPU embedding model is unavailable"
            ) from None

    @app.post("/v1/memory/conversations", dependencies=protected, status_code=201)
    def create_conversation(body: NewConversation):
        return conversations.create(body.title.strip())

    @app.get("/v1/memory/conversations", dependencies=protected)
    def list_conversations():
        return {"items":conversations.list()}

    def require_conversation(conversation_id):
        if not ID.fullmatch(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        value = conversations.get(conversation_id)
        if value is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return value

    @app.get("/v1/memory/conversations/{conversation_id}", dependencies=protected)
    def get_conversation(conversation_id: str):
        meta=require_conversation(conversation_id)
        return {"conversation":meta,"turns":conversations.turns(conversation_id)}

    @app.delete("/v1/memory/conversations/{conversation_id}",
                dependencies=protected,status_code=204)
    def delete_conversation(conversation_id: str):
        require_conversation(conversation_id)
        conversations.delete(conversation_id)

    @app.post("/v1/memory/conversations/{conversation_id}/turns",
              dependencies=protected)
    async def add_conversation_turn(conversation_id: str, body: ConversationTurn):
        meta = require_conversation(conversation_id)
        if body.expected_turns != meta["turn_count"]:
            raise HTTPException(status_code=409,detail="Conversation changed on another device")
        history = conversations.turns(conversation_id,limit=3)
        result = await evaluate(body,history=history)
        try:
            latest = conversations.append(
                conversation_id,body.expected_turns,body.question,
                result.answer,result.status,result.citations,
            )
        except ConversationConflict:
            raise HTTPException(status_code=409,detail="Conversation changed during reply") from None
        return {"conversation":latest,"result":result}
