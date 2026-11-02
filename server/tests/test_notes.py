"""Phase 6 notes API: CPU-only CRUD/search, Qwen citations and meeting import."""
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("HANDY_API_KEY", "test-only-token-not-for-production-1234567890")

import httpx
import pytest
from fastapi.testclient import TestClient

from handy_server.config import Settings
from handy_server.main import create_app
from handy_server.meeting_store import MeetingStore
from handy_server.schemas import MeetingAnalysis
from handy_server.workers import InferenceRuntime, QwenWorker

KEY = "test-only-token-not-for-production-1234567890"
AUTH = {"Authorization": "Bearer " + KEY}


class DummyAsr:
    ready = False
    def initialize(self):
        pass


def qwen_request(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.5-9B"}]})
    body = json.loads(request.content)
    assert "enable_thinking" in body["chat_template_kwargs"]
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
        "title": "incorrect-model-title",
        "sections": [{"heading": "خلاصه", "content": "گزارش آماده شود."}],
        "key_points": [
            {"text": "علی گزارش را آماده کند", "source_ids": ["n000001"],
             "assignee": "علی", "due_date": None},
            {"text": "ادعای جعلی", "source_ids": ["n999999"],
             "assignee": None, "due_date": None},
        ],
        "review_questions": ["چه زمانی تکمیل شود؟"],
    }, ensure_ascii=False)}}]})


@pytest.fixture
def disabled(tmp_path):
    config = Settings(api_key=KEY, db_path=tmp_path / "notes.sqlite")
    with TestClient(create_app(config)) as client:
        yield client, config


@pytest.fixture
def enabled(tmp_path):
    config = Settings(api_key=KEY, db_path=tmp_path / "notes.sqlite",
                      llm_enabled=True)
    llm = QwenWorker(config, transport=httpx.MockTransport(qwen_request))
    with TestClient(create_app(config, InferenceRuntime(config, DummyAsr(), llm))) as client:
        yield client, config


def create(client, title="گزارش هفتگی", content="علی گزارش را آماده کند.\nجلسه روز پنجشنبه"):
    response = client.post("/v1/notes", headers=AUTH,
                           json={"title": title, "content": content})
    assert response.status_code == 201, response.text
    return response.json()


def test_note_crud_and_persian_search_without_gpu(disabled):
    client, config = disabled
    assert client.get("/v1/notes").status_code == 401
    one = create(client)
    two = create(client, title="خرید مواد", content="قهوه عربیکا و شیر")
    assert len(one["id"]) == 32
    assert one["version"] == 1
    assert [s["id"] for s in one["sources"]] == ["n000001", "n000002"]
    assert client.get("/v1/notes", headers=AUTH).json()["total"] == 2
    assert client.get("/v1/notes?q=%D9%82%D9%87%D9%88%D9%87", headers=AUTH).json()["items"][0]["id"] == two["id"]
    assert client.get("/v1/notes?q=گزارش", headers=AUTH).json()["items"][0]["id"] == one["id"]
    assert client.get("/v1/notes?q=%22%20OR%20*", headers=AUTH).status_code == 200
    assert client.get("/v1/notes?limit=200", headers=AUTH).status_code == 422
    assert client.post("/v1/notes/" + one["id"] + "/analyze",
                       headers=AUTH).status_code == 503
    altered = client.put("/v1/notes/" + one["id"], headers=AUTH, json={
        "title": "جدید", "content": "قرارداد با تاریخ جدید", "version": 1,
    })
    assert altered.status_code == 200
    assert altered.json()["version"] == 2
    assert client.put("/v1/notes/" + one["id"], headers=AUTH, json={
        "title": "بازنویسی تکراری", "content": "بی‌اثر", "version": 1,
    }).status_code == 409
    assert client.get("/v1/notes?q=گزارش", headers=AUTH).json()["total"] == 0
    assert client.get("/v1/notes?q=قرارداد", headers=AUTH).json()["total"] == 1
    assert client.delete("/v1/notes/" + one["id"], headers=AUTH).status_code == 204
    assert client.get("/v1/notes/" + one["id"], headers=AUTH).status_code == 404
    assert client.get("/v1/notes?q=قرارداد", headers=AUTH).json()["total"] == 0


def test_qwen_grounds_notes_and_rejects_invented_evidence(enabled):
    client, config = enabled
    note = create(client)
    response = client.post("/v1/notes/" + note["id"] + "/analyze", headers=AUTH)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["analysis"]["title"] == "گزارش هفتگی"
    assert result["analysis"]["key_points"][0]["source_ids"] == ["n000001"]
    assert len(result["analysis"]["key_points"]) == 1
    assert result["analysis"]["review_questions"]
    assert client.get("/v1/notes/" + note["id"], headers=AUTH).json()["analysis"] == result["analysis"]
    updated = client.put("/v1/notes/" + note["id"], headers=AUTH, json={
        "title": "ویرایش", "content": "محتوای جدید", "version": 1,
    })
    assert updated.json()["analysis"] is None


def test_explicit_meeting_import_preserves_source_ids(disabled):
    client, config = disabled
    meetings = MeetingStore(config.db_path)
    meeting = meetings.create(Path("placeholder.wav"), 5.0, "test-import", "fa")
    meetings.checkpoint(meeting["id"], 80000, [{
        "id": "u00000001", "speaker": "SPEAKER_00",
        "start": 0, "end": 3, "text": "علی گزارش را تا جمعه آماده می‌کند",
    }])
    insight = MeetingAnalysis.model_validate({
        "summary": "علی مسئول تهیه گزارش است",
        "tasks": [{"text": "تهیه گزارش", "source_ids": ["u00000001"],
                   "assignee": "علی", "due_date": None}],
    })
    meetings.complete(meeting["id"], insight)
    response = client.post("/v1/notes/from-meeting/" + meeting["id"], headers=AUTH)
    assert response.status_code == 201, response.text
    note = response.json()
    assert note["source_meeting_id"] == meeting["id"]
    assert note["sources"] == [{"id": "u00000001",
                                "text": "علی گزارش را تا جمعه آماده می‌کند"}]
    assert "تهیه گزارش" in note["content"]
    assert client.get("/v1/notes?q=تهیه", headers=AUTH).json()["total"] == 1


def test_import_requires_completed_meeting(disabled):
    client, config = disabled
    meeting = MeetingStore(config.db_path).create(Path("placeholder.wav"), 1, None, "fa")
    assert client.post("/v1/notes/from-meeting/" + meeting["id"],
                       headers=AUTH).status_code == 409
    assert client.post("/v1/notes/from-meeting/" + "0"*32, headers=AUTH).status_code == 404
    assert client.post("/v1/notes", headers=AUTH,
                       json={"title": "x", "content": ""}).status_code == 422
    assert client.put("/v1/notes/" + "0"*32, headers=AUTH,
                      json={"title":"a","content":"b","version":1}).status_code == 404


def test_notes_persist_after_restart(tmp_path):
    config = Settings(api_key=KEY, db_path=tmp_path / "repeat.sqlite")
    with TestClient(create_app(config)) as first:
        note = create(first)
    with TestClient(create_app(config)) as second:
        persisted = second.get("/v1/notes/" + note["id"], headers=AUTH)
        assert persisted.status_code == 200
        assert persisted.json()["content"] == note["content"]
        assert second.get("/v1/notes?q=هفتگی", headers=AUTH).json()["total"] == 1
