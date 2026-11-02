"""Environment-only settings; secrets are never committed or returned to clients."""
from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit


def enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean")
    return value in {"true", "1", "yes"}


def validate_llm_url(value: str) -> str:
    parsed = urlsplit(value)
    # qwen is an internal service name in the example Docker Compose network.
    local_names = {"localhost", "127.0.0.1", "::1", "qwen"}
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in local_names
    ):
        raise ValueError("Qwen endpoint requires HTTPS outside loopback/Compose")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Qwen URL cannot contain credentials or tokens")
    if parsed.path.rstrip("/") != "/v1":
        raise ValueError("Qwen URL must end in /v1")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    api_key: str
    db_path: Path
    cors_origins: tuple[str, ...] = (
        "http://localhost:1420",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
    )
    asr_enabled: bool = False
    asr_model: str = "large-v3"
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    llm_enabled: bool = False
    qwen_base_url: str = "http://127.0.0.1:8000/v1"
    qwen_model: str = "Qwen/Qwen3.5-9B"
    qwen_api_key: str = "EMPTY"
    max_audio_bytes: int = 25 * 1024 * 1024
    max_text_chars: int = 12000
    max_meeting_bytes: int = 2_400_000_000
    max_meeting_seconds: int = 20 * 3600
    diarization_enabled: bool = False
    hf_token: str | None = None
    diarization_max_hours: float = 2.0
    semantic_enabled: bool = False
    semantic_model: str = "intfloat/multilingual-e5-small"

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.getenv("HANDY_API_KEY", "")
        if len(key) < 24 or key == "replace-with-long-random-secret":
            raise RuntimeError("Set HANDY_API_KEY to a random secret of at least 24 characters")
        raw_origins = os.getenv("HANDY_CORS_ORIGINS")
        origins = (
            tuple(s.strip() for s in raw_origins.split(",") if s.strip())
            if raw_origins is not None
            else cls.cors_origins
        )
        if "*" in origins:
            raise RuntimeError("Wildcard CORS origins are not allowed")
        max_audio_bytes = int(os.getenv("HANDY_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
        if not 1024 <= max_audio_bytes <= 250 * 1024 * 1024:
            raise ValueError("HANDY_MAX_AUDIO_BYTES outside supported range")
        settings = cls(
            api_key=key,
            db_path=Path(os.getenv("HANDY_DB_PATH", "./data/handy.sqlite3")),
            cors_origins=origins,
            asr_enabled=enabled("HANDY_ASR_ENABLED"),
            asr_model=os.getenv("HANDY_ASR_MODEL", "large-v3"),
            asr_device=os.getenv("HANDY_ASR_DEVICE", "cuda"),
            asr_compute_type=os.getenv("HANDY_ASR_COMPUTE_TYPE", "float16"),
            llm_enabled=enabled("HANDY_LLM_ENABLED"),
            qwen_base_url=os.getenv("HANDY_QWEN_BASE_URL", "http://127.0.0.1:8000/v1"),
            qwen_model=os.getenv("HANDY_QWEN_MODEL", "Qwen/Qwen3.5-9B"),
            qwen_api_key=os.getenv("HANDY_QWEN_API_KEY", "EMPTY"),
            max_audio_bytes=max_audio_bytes,
            max_text_chars=int(os.getenv("HANDY_MAX_TEXT_CHARS", "12000")),
            max_meeting_bytes=int(os.getenv("HANDY_MAX_MEETING_BYTES", "2400000000")),
            max_meeting_seconds=int(os.getenv("HANDY_MAX_MEETING_SECONDS", "72000")),
            diarization_enabled=enabled("HANDY_DIARIZATION_ENABLED"),
            hf_token=os.getenv("HANDY_HF_TOKEN") or None,
            diarization_max_hours=float(os.getenv("HANDY_DIARIZATION_MAX_HOURS", "2")),
            semantic_enabled=enabled("HANDY_SEMANTIC_ENABLED"),
            semantic_model=os.getenv("HANDY_SEMANTIC_MODEL", "intfloat/multilingual-e5-small"),
        )
        if settings.asr_model != "large-v3":
            raise ValueError("Phase 2 ASR is pinned to Whisper large-v3")
        if settings.asr_device not in {"cuda", "cpu", "auto"}:
            raise ValueError("Invalid HANDY_ASR_DEVICE")
        if settings.max_text_chars < 100 or settings.max_text_chars > 50000:
            raise ValueError("Invalid HANDY_MAX_TEXT_CHARS")
        if not 1 <= settings.max_meeting_seconds <= 20 * 3600:
            raise ValueError('Invalid meeting duration limit')
        if not 1024 <= settings.max_meeting_bytes <= 3_000_000_000:
            raise ValueError('Invalid meeting byte limit')
        if not 0.1 <= settings.diarization_max_hours <= 20:
            raise ValueError('Invalid diarization hour cap')
        if not settings.semantic_model or len(settings.semantic_model) > 120:
            raise ValueError("Invalid semantic model name")
        validate_llm_url(settings.qwen_base_url)
        return settings
