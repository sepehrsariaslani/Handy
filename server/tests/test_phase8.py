"""Phase 8: tasks, durable multi-turn conversations and optional CPU semantic index."""
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HANDY_API_KEY", "test-only-token-not-for-production-1234567890")

import httpx
import pytest
from fastapi.testclient import TestClient

from handy_server.config import Settings
from handy_server.main import create_app
from handy_server.meeting_store import MeetingStore
from handy_server.notes_store import NoteStore
from handy_server.semantic_index import SemanticIndex
from handy_server.schemas import MeetingAnalysis
from handy_server.workers import InferenceRuntime, QwenWorker

KEY="test-only-token-not-for-production-1234567890"
AUTH={"Authorization":"Bearer "+KEY}


class NoASR:
    ready=False
    def initialize(self): pass


def seed(db_path):
    notes=NoteStore(db_path)
    note=notes.create("گزارش تولید","چیدمان صندلی‌ها تا جمعه آماده می‌شود")
    meetings=MeetingStore(db_path)
    job=meetings.create(Path("fake.wav"),30,"phase8-sample","fa")
    meetings.checkpoint(job["id"],16000,[{
        "id":"u00000001","speaker":"علی",
        "start":2.0,"end":6.0,"text":"من گزارش کارگاه را تا جمعه تحویل می‌دهم"
    }])
    meetings.complete(job["id"],MeetingAnalysis.model_validate({
        "summary":"علی گزارش را تهیه می‌کند",
        "tasks":[
            {"text":"آماده‌سازی گزارش کارگاه","source_ids":["u00000001"],
             "assignee":"علی","due_date":"2026-11-06"},
            {"text":"وظیفه با منبع ساختگی","source_ids":["u99999999"]},
        ],
    }))
    return note,job


def response_factory(log):
    def handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200,json={"data":[{"id":"Qwen/Qwen3.5-9B"}]})
        sent=json.loads(json.loads(request.content)["messages"][1]["content"])
        log.append(sent)
        sources=sent["evidence"]
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps({
            "answer":"علی گزارش کارگاه را برای جمعه آماده می‌کند.",
            "citations":[sources[0]["id"]],
        },ensure_ascii=False)}}]})
    return handler


@pytest.fixture()
def setup(tmp_path):
    db_path=tmp_path/"phase8.db"
    note,job=seed(db_path)
    config=Settings(api_key=KEY,db_path=db_path,llm_enabled=True)
    events=[]
    worker=QwenWorker(config,transport=httpx.MockTransport(response_factory(events)))
    app=create_app(config,InferenceRuntime(config,NoASR(),worker))
    with TestClient(app) as client:
        yield client,db_path,note,job,events


def test_manual_tasks_and_optimistic_version_conflict(setup):
    client,_,_,_,_=setup
    assert client.get("/v1/tasks").status_code==401
    result=client.post("/v1/tasks",headers=AUTH,json={
        "title":"تکمیل خروجی","assignee":"سارا","due_date":"2026-12-20"
    })
    assert result.status_code==201,result.text
    item=result.json()
    assert item["version"]==1 and item["status"]=="todo"
    assert client.get("/v1/tasks",headers=AUTH).json()["total"]==1
    changed=client.put("/v1/tasks/"+item["id"],headers=AUTH,json={
        "title":"خروجی نهایی","assignee":"سارا","due_date":None,
        "status":"in_progress","version":1,
    })
    assert changed.status_code==200,changed.text
    assert changed.json()["version"]==2
    assert client.put("/v1/tasks/"+item["id"],headers=AUTH,json={
        "title":"قدیمی","assignee":None,"due_date":None,
        "status":"done","version":1,
    }).status_code==409
    assert client.get("/v1/tasks?status=in_progress",headers=AUTH).json()["total"]==1
    assert client.get("/v1/tasks?status=invalid",headers=AUTH).status_code==422
    assert client.delete("/v1/tasks/"+item["id"],headers=AUTH).status_code==204
    assert client.get("/v1/tasks/"+item["id"],headers=AUTH).status_code==404


def test_meeting_import_requires_explicit_action_and_is_idempotent(setup):
    client,_,_,job,_=setup
    assert client.get("/v1/tasks",headers=AUTH).json()["total"]==0
    first=client.post("/v1/tasks/from-meeting/"+job["id"],headers=AUTH)
    assert first.status_code==200,first.text
    body=first.json()
    assert body["created"]==1
    assert len(body["tasks"])==1
    task=body["tasks"][0]
    assert task["source_quotes"][0]["text"]=="من گزارش کارگاه را تا جمعه تحویل می‌دهم"
    assert task["source_ids"]==["u00000001"]
    assert task["assignee"]=="علی" and task["due_date"]=="2026-11-06"
    twice=client.post("/v1/tasks/from-meeting/"+job["id"],headers=AUTH).json()
    assert twice["created"]==0 and twice["tasks"][0]["id"]==task["id"]
    assert client.get("/v1/tasks",headers=AUTH).json()["total"]==1


def test_multi_turn_conversation_with_citations_and_restart(setup):
    client,db_path,_,_,events=setup
    created=client.post("/v1/memory/conversations",headers=AUTH,json={
        "title":"جلسه کارگاه",
    })
    assert created.status_code==201,created.text
    convo=created.json()
    cid=convo["id"]
    first=client.post(f"/v1/memory/conversations/{cid}/turns",headers=AUTH,json={
        "question":"علی گزارش کارگاه را چه زمانی می‌دهد؟",
        "expected_turns":0,
    })
    assert first.status_code==200,first.text
    assert first.json()["conversation"]["turn_count"]==1
    assert first.json()["result"]["citations"]
    assert events[-1]["previous_turns"]==[]
    follow=client.post(f"/v1/memory/conversations/{cid}/turns",headers=AUTH,json={
        "question":"چه روزی؟","expected_turns":1,
    })
    assert follow.status_code==200,follow.text
    assert follow.json()["conversation"]["turn_count"]==2
    assert events[-1]["previous_turns"][0]["question"].startswith("علی")
    assert client.post(f"/v1/memory/conversations/{cid}/turns",headers=AUTH,json={
        "question":"بازهم چه روزی؟","expected_turns":1,
    }).status_code==409
    detail=client.get(f"/v1/memory/conversations/{cid}",headers=AUTH).json()
    assert len(detail["turns"])==2
    assert detail["turns"][0]["citations"]
    assert client.get("/v1/memory/conversations",headers=AUTH).json()["items"][0]["id"]==cid
    assert client.delete(f"/v1/memory/conversations/{cid}",headers=AUTH).status_code==204
    assert client.get(f"/v1/memory/conversations/{cid}",headers=AUTH).status_code==404
    # SQLite persists between app restarts; deleted conversation cannot return.
    with TestClient(create_app(Settings(api_key=KEY,db_path=db_path))) as another:
        assert another.get(f"/v1/memory/conversations/{cid}",headers=AUTH).status_code==404


def fake_embed(self,texts,query=False):
    output=[]
    for text in texts:
        if "دکوراسیون" in text or "چیدمان" in text:
            output.append([1.0,0.0])
        elif "گزارش" in text:
            output.append([0.0,1.0])
        else:
            output.append([0.6,0.8])
    return output


def test_semantic_recovers_paraphrase_and_revalidates_deleted_notes(tmp_path):
    db_path=tmp_path/"semantic.db"
    note,_=seed(db_path)
    config=Settings(api_key=KEY,db_path=db_path,semantic_enabled=True)
    with patch.object(SemanticIndex,"_encode",fake_embed):
        with TestClient(create_app(config)) as client:
            assert client.get("/v1/memory/semantic/status",headers=AUTH).json()["enabled"]
            assert client.post("/v1/memory/semantic/sync",headers=AUTH,
                               params={"limit":0}).status_code==422
            index=client.post("/v1/memory/semantic/sync",headers=AUTH)
            assert index.status_code==200,index.text
            assert index.json()["indexed"]==2
            result=client.post("/v1/memory/search",headers=AUTH,json={
                "question":"دکوراسیون","scope":"notes",
            })
            assert result.status_code==200,result.text
            payload=result.json()
            assert payload["retrieval"]=="hybrid"
            assert payload["sources"][0]["document_id"]==note["id"]
            # A deleted note cannot leak from its old embedding vector.
            assert client.delete("/v1/notes/"+note["id"],headers=AUTH).status_code==204
            second=client.post("/v1/memory/search",headers=AUTH,json={
                "question":"دکوراسیون","scope":"notes",
            }).json()
            assert second["sources"]==[]


def test_semantic_disabled_uses_lexical_without_gpu(setup):
    client,_,_,_,_=setup
    assert client.get("/v1/memory/semantic/status",headers=AUTH).json()["enabled"] is False
    assert client.post("/v1/memory/semantic/sync",headers=AUTH).status_code==409
    found=client.post("/v1/memory/search",headers=AUTH,json={"question":"گزارش"}).json()
    assert found["retrieval"]=="lexical"
