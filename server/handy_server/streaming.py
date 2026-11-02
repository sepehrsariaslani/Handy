"""Authenticated WebSocket PCM streaming with resumable, idempotent commits."""
import asyncio
import hmac
import json
from pathlib import Path
import re
import tempfile
import wave

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .schemas import CleanupRequest, Mode, Tone
from .stream_store import (
    ChunkConflict, ChunkLimit, ChunkOutOfOrder, StreamStore, SAMPLE_RATE,
)
from .workers import WorkerFailure, WorkerUnavailable

SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
TOKEN_TIMEOUT_SECONDS = 6
# Only 0.6 s of raw audio is repeated as ASR context; no repeated transmission.
OVERLAP_SAMPLES = 9600
WORD_SPLIT = re.compile(r"\s+")


def deduplicate(previous: str, new: str) -> str:
    """Remove longest exact normalized suffix-prefix match after ASR overlap.

    This is a conservative best-effort strategy, not forced alignment.
    """
    from unicodedata import normalize
    def clean(value: str) -> str:
        return normalize("NFKC", value).strip(".,،.!؟?؛:;").casefold()
    left = previous.split()
    right = new.split()
    match_len = 0
    for amount in range(1, min(len(left), len(right), 15) + 1):
        if list(map(clean, left[-amount:])) == list(map(clean, right[:amount])):
            match_len = amount
    return " ".join(right[match_len:]).strip()


async def process_chunk(session_id: str, sequence: int, store: StreamStore,
                        runtime, language: str | None, cleanup: bool, tone: Tone) -> dict:
    pcm = await asyncio.to_thread(store.audio, session_id, sequence)
    overlap = await asyncio.to_thread(store.previous_audio, session_id, sequence, OVERLAP_SAMPLES)
    # Every chunk becomes an independently decodable WAV; no unbounded PCM buffering.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with wave.open(str(tmp_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(overlap + pcm)
        transcript, detected = await runtime.transcribe(tmp_path, language)
        raw = " ".join(s.raw_text for s in transcript if s.raw_text).strip()
        if overlap:
            prev_raw = await asyncio.to_thread(store.raw_previous, session_id, sequence)
            raw = deduplicate(prev_raw, raw)
        # Send raw candidate immediately; Qwen can run later.
        return {"raw_text": raw, "language": detected or language}
    finally:
        tmp_path.unlink(missing_ok=True)


def attach_stream_routes(app: FastAPI, config, sessions, runtime) -> None:
    store = StreamStore(Path(config.db_path))
    locks: dict[str, asyncio.Lock] = {}

    def safe_session(session_id: str):
        if not SESSION_ID.fullmatch(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        entry = sessions.get(session_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return entry

    # Authorization for this HTTP route is provided by the existing dependency.
    # This helper is called only by a route with dependencies=[Depends(require_auth)].
    def register_state_route(require_auth):
        from fastapi import Depends
        @app.get("/v1/sessions/{session_id}/stream-state", dependencies=[Depends(require_auth)])
        def state(session_id: str, after_sequence: int = -1, limit: int = 100):
            safe_session(session_id)
            if after_sequence < -1 or limit < 1 or limit > 200:
                raise HTTPException(status_code=422, detail="Invalid pagination")
            return store.state(session_id, after_sequence, limit)

    app.state.register_stream_state = register_state_route

    @app.websocket("/v1/stream/{session_id}")
    async def stream(websocket: WebSocket, session_id: str):
        if not SESSION_ID.fullmatch(session_id):
            await websocket.close(code=4404)
            return
        origin = websocket.headers.get("origin")
        if origin and origin not in config.cors_origins:
            await websocket.close(code=4403)
            return
        # Token must be in the first JSON message; NEVER in a URL or logs.
        await websocket.accept()
        try:
            hello = await asyncio.wait_for(websocket.receive_json(), TOKEN_TIMEOUT_SECONDS)
            if not isinstance(hello, dict) or hello.get("type") != "hello":
                await websocket.close(code=4400)
                return
            token = hello.get("token")
            if not isinstance(token, str) or not hmac.compare_digest(token, config.api_key):
                await websocket.close(code=4401)
                return
            session = sessions.get(session_id)
            if session is None or session.status == "completed":
                await websocket.close(code=4404)
                return
            if not await runtime.asr_ready():
                await websocket.send_json({"type": "error", "reason": "asr_unavailable"})
                await websocket.close(code=1013)
                return
            cleanup = hello.get("cleanup") is True
            language = hello.get("language")
            if language is not None and (not isinstance(language, str) or len(language) > 35):
                await websocket.close(code=4400)
                return
            try:
                tone = Tone(hello.get("tone", session.tone.value))
            except ValueError:
                await websocket.close(code=4400)
                return
            lock = locks.setdefault(session_id, asyncio.Lock())
            if lock.locked():
                await websocket.send_json({"type": "error", "reason": "session_in_use"})
                await websocket.close(code=4409)
                return
            async with lock:
                snapshot = store.state(session_id, max(-1, int(hello.get("last_seen", -1))), 200)
                await websocket.send_json({"type": "ready", **snapshot})
                while True:
                    message = await websocket.receive_json()
                    if not isinstance(message, dict):
                        await websocket.close(code=4400)
                        return
                    if message.get("type") == "finish":
                        if store.complete(session_id):
                            sessions.set_status(session_id, "completed")
                            await websocket.send_json({"type": "complete"})
                            await websocket.close(code=1000)
                            return
                        await websocket.send_json({"type": "error", "reason": "unfinished_chunks"})
                        continue
                    if message.get("type") != "chunk":
                        await websocket.send_json({"type": "error", "reason": "unknown_message"})
                        continue
                    seq = message.get("sequence")
                    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0 or seq > 19999:
                        await websocket.send_json({"type": "error", "reason": "invalid_sequence"})
                        continue
                    if message.get("encoding") != "pcm_s16le" or message.get("sample_rate") != SAMPLE_RATE:
                        await websocket.send_json({"type": "error", "reason": "unsupported_audio"})
                        continue
                    pcm = await websocket.receive_bytes()
                    try:
                        result, committed = await asyncio.to_thread(store.accept, session_id, seq, pcm)
                    except ChunkConflict:
                        await websocket.send_json({"type": "error", "reason": "sequence_conflict", "sequence": seq})
                        continue
                    except ChunkOutOfOrder as exc:
                        await websocket.send_json({"type": "resync", "next_sequence": exc.expected})
                        continue
                    except ChunkLimit:
                        await websocket.send_json({"type": "error", "reason": "chunk_limit", "sequence": seq})
                        continue
                    await websocket.send_json({"type": "ack", "sequence": seq, "duplicate": result == "duplicate"})
                    if committed:
                        await websocket.send_json(committed)
                        continue
                    sessions.set_status(session_id, "recording")
                    try:
                        response = await process_chunk(session_id, seq, store, runtime, language, cleanup, tone)
                        await websocket.send_json({
                            "type": "partial",
                            "sequence": seq,
                            "raw_text": response["raw_text"],
                        })
                        cleaned = None
                        warnings = []
                        if cleanup and response["raw_text"]:
                            try:
                                edited = await runtime.cleanup(CleanupRequest(
                                    text=response["raw_text"], tone=tone,
                                    language=response["language"],
                                ))
                                cleaned = edited.clean_text
                                warnings = edited.warnings
                            except (WorkerFailure, WorkerUnavailable, ValueError):
                                warnings = ["Qwen unavailable; original text kept"]
                        final = await asyncio.to_thread(
                            store.finish, session_id, seq,
                            response["raw_text"], cleaned, response["language"], warnings,
                        )
                        await websocket.send_json(final)
                    except (WorkerUnavailable, WorkerFailure):
                        # The chunk remains on disk as uncommitted; client retries on reconnect.
                        await websocket.send_json({"type": "error", "reason": "inference_failed", "sequence": seq})
                        await websocket.close(code=1013)
                        return
        except (WebSocketDisconnect, asyncio.TimeoutError):
            return
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            await websocket.close(code=4400)
