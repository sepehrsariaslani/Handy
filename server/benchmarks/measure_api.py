"""Reproducible local API overhead baseline. Does NOT benchmark Whisper or Qwen.

Run after installing server dev dependencies:
  cd server && python benchmarks/measure_api.py
"""
from pathlib import Path
import json
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
import platform
import sys

# Run from the server directory (editable install: pip install -e '.[dev]').
from fastapi.testclient import TestClient

from handy_server.config import Settings
from handy_server.main import create_app


def measure(client: TestClient, method: str, path: str, token: str | None = None) -> dict:
    latencies = []
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for _ in range(10):
        assert client.request(method, path, headers=headers).status_code == 200
    for _ in range(100):
        start = perf_counter()
        response = client.request(method, path, headers=headers)
        latencies.append((perf_counter() - start) * 1000)
        assert response.status_code == 200
    latencies.sort()
    return {
        "requests": len(latencies),
        "median_ms": round(median(latencies), 3),
        "p95_ms": round(latencies[94], 3),
        "max_ms": round(latencies[-1], 3),
    }


def main() -> None:
    key = "benchmark-local-test-only-token-0123456789"
    with TemporaryDirectory() as directory:
        app = create_app(Settings(api_key=key, db_path=Path(directory) / "db.sqlite3"))
        with TestClient(app) as client:
            output = {
                "python": sys.version.split()[0],
                "system": platform.platform(),
                "scope": "FastAPI/TestClient overhead, no network/GPU/audio inference",
                "health": measure(client, "GET", "/health"),
                "capabilities": measure(client, "GET", "/v1/capabilities", key),
            }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
