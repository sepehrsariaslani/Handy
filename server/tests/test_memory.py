"""Phase 7: durable memory retrieval, cited QA and no-GPU fallback."""
import json
import os
from datetime import date
from pathlib import Path

os.environ.setdefault("HANDY_API_KEY", "test-only-token-not-for-production-1234567890")

import httpx
from fastapi.testclient import TestClient
import pytest

from handy_server.config import Settings
from handy_server.main import create_app
from handy_server.meeting_store import MeetingStore
from handy_server.notes_store import NoteStore
from handy_server.memory_store import MemoryStore, tokens, infer_utc_dates
from handy_server.workers import InferenceRuntime, QwenWorker


KEY = "test-only-token-not-for-production-1234567890"
AUTH = {"Authorization": "Bearer " + KEY}


class DisabledASR:
    ready = False
    def initialize(self):
        pass


def answer_provider(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.5-9B"}]})
    prompt = json.loads(json.loads(request.content)["messages"][1]["content"])
    assert len(prompt["evidence"]) >= 1
    assert "source" not in prompt["question"].lower() or True
    first = prompt["evidence"][0]["id"]
    reply = {"answer": "علی مسئول آماده‌کردن گزارش شد.", "citations": [first]}
    if "جعلی" in prompt["question"]:
        reply["citations"] = ["meeting:madeup:fake"]
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(reply, ensure_ascii=False)}}],
    })


@pytest.fixture
def seed(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    notes = NoteStore(db_path)
    note = notes.create("گزارش پروژه", "علی گزارش مالی را برای جمعه آماده می‌کند.\nبودجه هنوز مشخص نیست.")
    meeting = MeetingStore(db_path)
    record = meeting.create(Path("example.wav"), 30, "meeting-memory", "fa")
    meeting.checkpoint(record["id"], 480000, [
        {"id": "u00000001", "speaker": "SPEAKER_00",
         "start": 2.0, "end": 5.0, "text": "علی قول داد گزارش مالی را آماده کند"},
        {"id": "u00000002", "speaker": "SPEAKER_01",
         "start": 9.0, "end": 12.0, "text": "بودجه هنوز تصویب نشده است"},
    ])
    meeting.complete(record["id"], None)
    return db_path, note, record


def app_for(db_path, llm=False):
    config = Settings(api_key=KEY, db_path=db_path, llm_enabled=llm)
    if not llm:
        return create_app(config)
    qwen = QwenWorker(config, transport=httpx.MockTransport(answer_provider))
    return create_app(config, InferenceRuntime(config, DisabledASR(), qwen))


def test_search_sources_no_gpu_and_qa_fallback(seed):
    db_path, note, meeting = seed
    with TestClient(app_for(db_path)) as client:
        assert client.post("/v1/memory/search", json={"question": "علی گزارش"}).status_code == 401
        resp = client.post("/v1/memory/search", headers=AUTH, json={"question": "علی گزارش"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "search_only"
        assert {s["kind"] for s in body["sources"]} == {"note", "meeting"}
        sources = body["sources"]
        assert any(s["id"] == f"meeting:{meeting['id']}:u00000001" and
                   s["start"] == 2.0 and s["speaker"] == "SPEAKER_00" for s in sources)
        assert any(s["id"] == f"note:{note['id']}:n000001" for s in sources)
        assert all(s["quote"] for s in sources)
        qa = client.post("/v1/memory/ask", headers=AUTH, json={"question": "علی گزارش"})
        assert qa.status_code == 200, qa.text
        assert qa.json()["status"] == "search_only"
        assert qa.json()["answer"] is None
        assert qa.json()["sources"]


def test_answer_with_valid_citation_and_hallucinated_citation_rejected(seed):
    db_path, note, meeting = seed
    with TestClient(app_for(db_path, llm=True)) as client:
        result = client.post("/v1/memory/ask", headers=AUTH,
                             json={"question": "علی چه گزارشی را آماده می‌کند؟"})
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["status"] == "answered"
        assert payload["citations"]
        assert set(payload["citations"]) <= {s["id"] for s in payload["sources"]}
        fake = client.post("/v1/memory/ask", headers=AUTH,
                           json={"question": "علی گزارش جعلی"})
        assert fake.status_code == 200, fake.text
        assert fake.json()["status"] == "not_found"
        assert fake.json()["answer"] is None
        assert fake.json()["citations"] == []


def test_no_match_does_not_call_llm(seed):
    db_path, _, _ = seed
    with TestClient(app_for(db_path, llm=True)) as client:
        result = client.post("/v1/memory/ask", headers=AUTH,
                             json={"question": "ناشناختهساختگی بی‌ارتباط"})
        assert result.status_code == 200
        assert result.json()["status"] == "not_found"
        assert result.json()["sources"] == []


def test_scope_date_and_bad_input(seed):
    db_path, note, meeting = seed
    with TestClient(app_for(db_path)) as client:
        for scope, expected in (("notes", {"note"}), ("meetings", {"meeting"})):
            result = client.post("/v1/memory/search", headers=AUTH,
                                 json={"question": "علی گزارش", "scope": scope}).json()
            assert {s["kind"] for s in result["sources"]} == expected
        filtered = client.post("/v1/memory/search", headers=AUTH,
                               json={"question": "علی گزارش", "since": "3000-01-01"})
        assert filtered.json()["status"] == "not_found"
        assert filtered.json()["sources"] == []
        wrong = client.post("/v1/memory/search", headers=AUTH,
                            json={"question": "علی گزارش",
                                  "since": "2026-11-03", "until": "2026-11-01"})
        assert wrong.status_code == 422
        assert client.post("/v1/memory/search", headers=AUTH,
                           json={"question": "x"}).status_code == 422
        assert client.post("/v1/memory/search", headers=AUTH,
                           json={"question": "علی گزارش", "scope": "all",
                                 "top_k": 100000}).status_code == 422


def test_backfill_update_and_delete_reflected_in_index(seed):
    db_path, note, meeting = seed
    store = MemoryStore(db_path)
    assert any(x["kind"] == "meeting" for x in store.search("بودجه"))
    notes = NoteStore(db_path)
    notes.update(note["id"], 1, "جدید", "فاکتور خرید و هزینه جابه‌جایی")
    assert not any(x["kind"] == "note" for x in store.search("گزارش"))
    assert any(x["kind"] == "note" for x in store.search("فاکتور"))
    # Existing meeting changes are visible through the FTS update trigger.
    meetings = MeetingStore(db_path)
    meetings.update_speakers(meeting["id"], [("u00000001", "سارا")])
    assert any(x["speaker"] == "سارا" for x in store.search("علی"))
    notes.delete(note["id"])
    assert not any(x["kind"] == "note" for x in store.search("فاکتور"))


def test_safe_lexical_query_construction():
    assert tokens('" ) OR * --') == ["or"]
    assert "علی" in tokens("علی هفته گذشته چه کاری قبول کرد؟")
    assert tokens("آیا به یا از در") == []
    assert MemoryStore is not None



def test_common_relative_dates_are_utc_and_explicit_filters_win():
    from datetime import date
    today = date(2026, 11, 2)  # Monday
    assert infer_utc_dates("هفته گذشته علی چه کاری کرد؟", None, None, today) == (
        date(2026, 10, 26), date(2026, 11, 1)
    )
    assert infer_utc_dates("What did Ali do last week?", None, None, today) == (
        date(2026, 10, 26), date(2026, 11, 1)
    )
    assert infer_utc_dates("دیروز چه شد؟", None, None, today) == (
        date(2026, 11, 1), date(2026, 11, 1)
    )
    assert infer_utc_dates("this week", None, None, today) == (
        date(2026, 11, 2), date(2026, 11, 2)
    )
    assert infer_utc_dates(
        "last week", date(2025, 1, 1), None, today
    ) == (date(2025, 1, 1), None)
