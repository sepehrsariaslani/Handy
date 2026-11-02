"""Server configuration, loaded from the environment (never from the repository)."""
from dataclasses import dataclass
import os
from pathlib import Path


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
        return cls(
            api_key=key,
            db_path=Path(os.getenv("HANDY_DB_PATH", "./data/handy.sqlite3")),
            cors_origins=origins,
        )
