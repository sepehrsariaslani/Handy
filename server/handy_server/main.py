"""Versioned API shell for the three Handy modes.

No raw audio or transcription is accepted until phase-2/3 workers are installed.
"""
import hmac
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings
from .schemas import (
    CapabilitiesResponse,
    HealthResponse,
    SessionCreate,
    SessionRecord,
)
from .store import SessionStore

bearer = HTTPBearer(auto_error=False)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    if len(config.api_key) < 24:
        raise ValueError("API key must contain at least 24 characters")
    store = SessionStore(Path(config.db_path))
    app = FastAPI(title="Handy AI API", version="0.1.0", description="Phase-1 API skeleton")
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
        if (
            credentials is None
            or not hmac.compare_digest(credentials.credentials, config.api_key)
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
    def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse()

    @app.post("/v1/sessions", status_code=201, response_model=SessionRecord, dependencies=protected)
    def create_session(body: SessionCreate) -> SessionRecord:
        return store.create(body)

    @app.get("/v1/sessions/{session_id}", response_model=SessionRecord, dependencies=protected)
    def get_session(session_id: str) -> SessionRecord:
        record = store.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return record

    @app.post("/v1/sessions/{session_id}/chunks", dependencies=protected)
    def audio_chunk(session_id: str) -> None:
        if store.get(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        raise HTTPException(status_code=501, detail="Streaming inference is planned for phase 3")

    @app.post("/v1/meetings", dependencies=protected)
    def submit_meeting() -> None:
        # Deliberately do not consume/upload a potentially multi-hour recording.
        raise HTTPException(status_code=501, detail="Meeting processing is planned for phase 5")

    @app.get("/v1/meetings/{job_id}", dependencies=protected)
    def get_meeting_job(job_id: str) -> None:
        raise HTTPException(status_code=501, detail="Meeting workers are not configured")

    @app.post("/v1/meetings/{job_id}/retry", dependencies=protected)
    def retry_meeting_job(job_id: str) -> None:
        raise HTTPException(status_code=501, detail="Meeting workers are not configured")

    return app


app = create_app()
