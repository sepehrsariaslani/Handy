from pathlib import Path
import os

# The production app module rejects startup without a secret; test collection
# sets an explicit test-only value before importing its Uvicorn entrypoint.
os.environ.setdefault("HANDY_API_KEY", "test-only-token-not-for-production-1234567890")

from fastapi.testclient import TestClient
import pytest

from handy_server.config import Settings
from handy_server.main import create_app

TOKEN = "test-only-token-not-for-production-1234567890"


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(Settings(api_key=TOKEN, db_path=tmp_path / "sessions.db"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_health_is_not_a_model_readiness_claim(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "api_version": "v1"}


def test_authorization_required(client: TestClient):
    assert client.get("/v1/capabilities").status_code == 401
    assert client.post("/v1/sessions", json={"mode": "dictation"}).status_code == 401
    assert client.get("/v1/capabilities", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_modes_and_workers_are_declared_but_not_faked(client: TestClient, auth: dict[str, str]):
    payload = client.get("/v1/capabilities", headers=auth).json()
    assert payload["modes"] == ["dictation", "meeting", "note_taking"]
    assert payload["models"]["asr"] == "whisper-large-v3"
    assert payload["models"]["llm"] == "qwen3.5-9b"
    assert not any(payload["processing"].values())


@pytest.mark.parametrize("mode", ["dictation", "meeting", "note_taking"])
def test_create_get_and_persist_session(client: TestClient, auth: dict[str, str], mode: str, tmp_path: Path):
    created = client.post(
        "/v1/sessions",
        headers=auth,
        json={"mode": mode, "title": "Sample", "language": "fa", "tone": "formal"},
    )
    assert created.status_code == 201
    data = created.json()
    assert data["mode"] == mode
    assert data["status"] == "created"
    assert client.get(f"/v1/sessions/{data['id']}", headers=auth).json() == data
    reopened = TestClient(create_app(Settings(api_key=TOKEN, db_path=tmp_path / "sessions.db")))
    assert reopened.get(f"/v1/sessions/{data['id']}", headers=auth).json() == data


def test_rejects_unknown_modes_and_fields(client: TestClient, auth: dict[str, str]):
    for body in ({"mode": "unknown"}, {"mode": "meeting", "admin": True}):
        assert client.post("/v1/sessions", headers=auth, json=body).status_code == 422


def test_missing_session_and_inference_not_implemented(client: TestClient, auth: dict[str, str]):
    assert client.get("/v1/sessions/missing", headers=auth).status_code == 404
    session = client.post("/v1/sessions", headers=auth, json={"mode": "meeting"}).json()
    assert client.post(f"/v1/sessions/{session['id']}/chunks", headers=auth).status_code == 501
    assert client.post("/v1/meetings", headers=auth).status_code == 422
    assert client.get("/v1/meetings/f" + "0" * 31, headers=auth).status_code == 404
    assert client.post("/v1/meetings/f" + "0" * 31 + "/retry", headers=auth).status_code == 404


def test_cors_is_restricted(client: TestClient):
    allowed = client.options(
        "/v1/sessions",
        headers={"Origin": "http://localhost:1420", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:1420"
    denied = client.options(
        "/v1/sessions",
        headers={"Origin": "https://malicious.example", "Access-Control-Request-Method": "POST"},
    )
    assert denied.headers.get("access-control-allow-origin") is None
