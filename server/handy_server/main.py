"""Authenticated Handy AI API: offline metadata plus optional self-hosted GPU workers.

Phase 2 provides bounded, non-streaming transcription and text cleanup. Meeting
analysis and incremental streaming intentionally remain separate phases.
"""
from contextlib import asynccontextmanager
import hmac
from pathlib import Path
import tempfile
from time import monotonic

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings
from .schemas import (
    CapabilitiesResponse,
    CleanupRequest,
    CleanupResponse,
    HealthResponse,
    ModelStatus,
    SessionCreate,
    SessionRecord,
    Tone,
    TranscriptionResponse,
)
from .store import SessionStore
from .workers import InferenceRuntime, WorkerFailure, WorkerUnavailable

bearer = HTTPBearer(auto_error=False)
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".mp4"}


def create_app(
    settings: Settings | None = None,
    runtime: InferenceRuntime | None = None,
) -> FastAPI:
    config = settings or Settings.from_env()
    if len(config.api_key) < 24:
        raise ValueError("API key must contain at least 24 characters")
    store = SessionStore(Path(config.db_path))
    models = runtime or InferenceRuntime(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Do not download/load a GPU model at module import time.
        import asyncio
        await asyncio.to_thread(models.initialize)
        yield

    app = FastAPI(
        title="Handy AI API",
        version="0.2.0",
        description="Whisper Large V3 + Qwen3.5-9B (phase 2)",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def require_auth(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if credentials is None or not hmac.compare_digest(
            credentials.credentials, config.api_key
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    protected = [Depends(require_auth)]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/capabilities", response_model=CapabilitiesResponse, dependencies=protected)
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            processing={
                "asr": await models.asr_ready(),
                "llm": await models.llm_ready(),
                "speaker_diarization": False,
                "streaming": False,
            }
        )

    @app.get("/v1/models/status", response_model=ModelStatus, dependencies=protected)
    async def model_status() -> ModelStatus:
        return ModelStatus(
            asr_model=config.asr_model,
            asr_ready=await models.asr_ready(),
            asr_device=config.asr_device,
            llm_model=config.qwen_model,
            llm_ready=await models.llm_ready(),
        )

    @app.post("/v1/sessions", status_code=201, response_model=SessionRecord, dependencies=protected)
    def create_session(body: SessionCreate) -> SessionRecord:
        return store.create(body)

    @app.get("/v1/sessions/{session_id}", response_model=SessionRecord, dependencies=protected)
    def get_session(session_id: str) -> SessionRecord:
        record = store.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return record

    @app.post("/v1/text/cleanup", response_model=CleanupResponse, dependencies=protected)
    async def text_cleanup(body: CleanupRequest) -> CleanupResponse:
        if len(body.text) > config.max_text_chars:
            raise HTTPException(status_code=413, detail="Text exceeds maximum size")
        try:
            return await models.cleanup(body)
        except WorkerUnavailable:
            raise HTTPException(status_code=503, detail="Qwen worker is unavailable") from None
        except WorkerFailure:
            raise HTTPException(status_code=502, detail="Qwen worker failed") from None
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid cleanup request") from None

    @app.post("/v1/transcriptions", response_model=TranscriptionResponse, dependencies=protected)
    async def transcribe(
        audio: UploadFile = File(...),
        language: str | None = Form(default=None, max_length=35),
        cleanup: bool = Form(default=False),
        tone: Tone = Form(default=Tone.professional),
    ) -> TranscriptionResponse:
        if not await models.asr_ready():
            raise HTTPException(status_code=503, detail="Whisper worker is unavailable")
        if cleanup and not await models.llm_ready():
            raise HTTPException(status_code=503, detail="Qwen worker is unavailable")
        suffix = Path(audio.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Unsupported audio file extension")

        filename: Path | None = None
        started = monotonic()
        try:
            # Stream to disk with a hard byte cap: never buffer the whole audio in RAM.
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                filename = Path(tmp.name)
                size = 0
                while data := await audio.read(1024 * 1024):
                    size += len(data)
                    if size > config.max_audio_bytes:
                        raise HTTPException(status_code=413, detail="Audio exceeds maximum upload size")
                    tmp.write(data)
            if size == 0:
                raise HTTPException(status_code=422, detail="Empty audio file")
            segments, detected_language = await models.transcribe(filename, language)
            raw_text = " ".join(s.raw_text for s in segments).strip()
            edited = None
            if cleanup and raw_text:
                if len(raw_text) > config.max_text_chars:
                    raise HTTPException(status_code=413, detail="Transcript too long for cleanup")
                edited = await models.cleanup(
                    CleanupRequest(text=raw_text, tone=tone, language=detected_language or language)
                )
            return TranscriptionResponse(
                raw_text=raw_text,
                clean_text=edited.clean_text if edited else None,
                language=detected_language or language,
                model="whisper-large-v3",
                segments=segments,
                cleanup=edited,
                elapsed_ms=int((monotonic() - started) * 1000),
            )
        except WorkerUnavailable:
            raise HTTPException(status_code=503, detail="Inference worker unavailable") from None
        except WorkerFailure:
            raise HTTPException(status_code=502, detail="Inference worker failed") from None
        finally:
            await audio.close()
            if filename is not None:
                filename.unlink(missing_ok=True)

    @app.post("/v1/sessions/{session_id}/chunks", dependencies=protected)
    def audio_chunk(session_id: str) -> None:
        if store.get(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        raise HTTPException(status_code=501, detail="Streaming inference is planned for phase 3")

    @app.post("/v1/meetings", dependencies=protected)
    def submit_meeting() -> None:
        raise HTTPException(status_code=501, detail="Meeting processing is planned for phase 5")

    @app.get("/v1/meetings/{job_id}", dependencies=protected)
    def get_meeting_job(job_id: str) -> None:
        raise HTTPException(status_code=501, detail="Meeting workers are not configured")

    @app.post("/v1/meetings/{job_id}/retry", dependencies=protected)
    def retry_meeting_job(job_id: str) -> None:
        raise HTTPException(status_code=501, detail="Meeting workers are not configured")

    return app


app = create_app()
