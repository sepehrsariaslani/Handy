"""Resumable streaming protocol tests, without microphone or GPU hardware."""
import os
from pathlib import Path

os.environ.setdefault("HANDY_API_KEY", "test-only-token-not-for-production-1234567890")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import pytest

from handy_server.config import Settings
from handy_server.main import create_app
from handy_server.schemas import CleanupRequest, CleanupResponse, TranscriptSegment
from handy_server.streaming import deduplicate
from handy_server.workers import InferenceRuntime

KEY = "test-only-token-not-for-production-1234567890"
AUTH = {"Authorization": "Bearer " + KEY}
PCM = b"\x00\x00" * 16000  # one second, 16kHz, mono PCM16


class StreamingASR:
    ready = True

    def __init__(self):
        self.calls = 0
        self.fail_once = False

    def initialize(self):
        pass

    def transcribe_file(self, path: Path, language):
        import wave
        self.calls += 1
        if self.fail_once:
            from handy_server.workers import WorkerFailure
            self.fail_once = False
            raise WorkerFailure("Synthetic inference interruption")
        with wave.open(str(path), "rb") as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
        return [TranscriptSegment(id="t1", start=0, end=1,
                                  raw_text="این یک جمله آزمایشی است", final=True)], "fa"


class StreamingLLM:
    async def ready(self):
        return True

    async def cleanup(self, req: CleanupRequest):
        return CleanupResponse(raw_text=req.text, clean_text=req.text + ".",
                               tone=req.tone, model="Qwen/Qwen3.5-9B")


@pytest.fixture()
def fixture(tmp_path):
    config = Settings(api_key=KEY, db_path=tmp_path / "db.sqlite3")
    asr = StreamingASR()
    runtime = InferenceRuntime(config, asr=asr, llm=StreamingLLM())
    with TestClient(create_app(config, runtime=runtime)) as client:
        session = client.post("/v1/sessions", json={"mode": "dictation"}, headers=AUTH).json()
        yield client, session["id"], asr


def connect(client, sid, *, last_seen=-1, cleanup=True):
    websocket = client.websocket_connect("/v1/stream/" + sid, headers={"origin": "http://localhost:1420"})
    ws = websocket.__enter__()
    ws.send_json({"type": "hello", "token": KEY, "language": "fa",
                  "cleanup": cleanup, "tone": "professional", "last_seen": last_seen})
    assert ws.receive_json()["type"] == "ready"
    return websocket, ws


def chunk(ws, seq, pcm=PCM):
    ws.send_json({"type": "chunk", "sequence": seq,
                  "encoding": "pcm_s16le", "sample_rate": 16000})
    ws.send_bytes(pcm)


def test_websocket_partial_and_final_flow(fixture):
    client, sid, asr = fixture
    context, ws = connect(client, sid)
    try:
        chunk(ws, 0)
        assert ws.receive_json() == {"type": "ack", "sequence": 0, "duplicate": False}
        candidate = ws.receive_json()
        assert candidate["type"] == "partial"
        assert candidate["raw_text"] == "این یک جمله آزمایشی است"
        final = ws.receive_json()
        assert final["type"] == "final"
        assert final["segment"]["clean_text"] == "این یک جمله آزمایشی است."
        assert final["segment"]["raw_text"] == candidate["raw_text"]
        assert final["segment"]["start"] == 0
        assert final["segment"]["end"] == 1
        assert asr.calls == 1
        ws.send_json({"type": "finish"})
        assert ws.receive_json() == {"type": "complete"}
        assert client.get("/v1/sessions/" + sid, headers=AUTH).json()["status"] == "completed"
    finally:
        context.__exit__(None, None, None)


def test_reconnect_replay_and_duplicate_does_not_reinfer(fixture):
    client, sid, asr = fixture
    context, ws = connect(client, sid)
    try:
        chunk(ws, 0)
        assert ws.receive_json()["type"] == "ack"
        assert ws.receive_json()["type"] == "partial"
        assert ws.receive_json()["type"] == "final"
    finally:
        context.__exit__(None, None, None)
    with client.websocket_connect("/v1/stream/" + sid) as resumed:
        resumed.send_json({"type": "hello", "token": KEY, "last_seen": -1})
        state = resumed.receive_json()
        assert state["next_sequence"] == 1
        assert state["committed_count"] == 1
        assert state["results"][0]["segment"]["id"] == "c00000000"
        chunk(resumed, 0)
        assert resumed.receive_json() == {"type": "ack", "sequence": 0, "duplicate": True}
        assert resumed.receive_json()["type"] == "final"
        assert asr.calls == 1
        chunk(resumed, 2)
        assert resumed.receive_json() == {"type": "resync", "next_sequence": 1}
        chunk(resumed, 1)
        assert resumed.receive_json()["type"] == "ack"
        assert resumed.receive_json()["type"] == "partial"
        final = resumed.receive_json()
        assert final["segment"]["raw_text"] == ""  # 0.6s overlap deduplicated
        assert final["segment"]["start"] == 1
        assert final["segment"]["end"] == 2
        assert asr.calls == 2
        resumed.send_json({"type": "finish"})
        assert resumed.receive_json()["type"] == "complete"


def test_conflict_and_oversized_audio(fixture):
    client, sid, asr = fixture
    with client.websocket_connect("/v1/stream/" + sid) as ws:
        ws.send_json({"type": "hello", "token": KEY})
        assert ws.receive_json()["type"] == "ready"
        chunk(ws, 0, b"\x00\x00" * 500)  # too short
        assert ws.receive_json()["reason"] == "chunk_limit"
        chunk(ws, 0, PCM)
        for key in ("ack", "partial", "final"):
            assert ws.receive_json()["type"] == key
        chunk(ws, 0, b"\x01\x01" * 16000)
        assert ws.receive_json()["reason"] == "sequence_conflict"
        chunk(ws, 1, b"\x00\x00" * 97000)  # > 6 seconds
        assert ws.receive_json()["reason"] == "chunk_limit"
    assert asr.calls == 1


def test_http_state_pagination_and_auth(fixture):
    client, sid, _ = fixture
    assert client.get("/v1/sessions/" + sid + "/stream-state").status_code == 401
    assert client.get("/v1/sessions/" + sid + "/stream-state?limit=300",
                      headers=AUTH).status_code == 422
    state = client.get("/v1/sessions/" + sid + "/stream-state", headers=AUTH).json()
    assert state["next_sequence"] == 0
    assert state["results"] == []


def test_unknown_session_or_origin_rejected(fixture):
    client, sid, _ = fixture
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/stream/" + sid,
                                      headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/stream/" + "0" * 32) as ws:
            ws.send_json({"type": "hello", "token": KEY})
            ws.receive_json()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/stream/" + sid) as ws:
            ws.send_json({"type": "hello", "token": "wrong"})
            ws.receive_json()


def test_deduplication_is_conservative():
    assert deduplicate("سلام خوبی", "خوبی امروز چطوری") == "امروز چطوری"
    assert deduplicate("جلسه ساعت ۱۲", "ساعت 12 شروع شد") == "شروع شد"
    assert deduplicate("بودجه ۱۲۵۰ تومان", "هزینه ۱۲۶۰ تومان") == "هزینه ۱۲۶۰ تومان"
    assert deduplicate("تست", "تست") == ""


def test_durable_uncommitted_audio_retries_after_server_restart(tmp_path):
    config = Settings(api_key=KEY, db_path=tmp_path / "persistent.db")
    failing = StreamingASR()
    failing.fail_once = True
    runtime = InferenceRuntime(config, asr=failing, llm=StreamingLLM())
    with TestClient(create_app(config, runtime=runtime)) as first:
        session_id = first.post("/v1/sessions", json={"mode": "dictation"}, headers=AUTH).json()["id"]
        with first.websocket_connect("/v1/stream/" + session_id) as ws:
            ws.send_json({"type": "hello", "token": KEY})
            assert ws.receive_json()["type"] == "ready"
            chunk(ws, 0)
            assert ws.receive_json()["type"] == "ack"
            assert ws.receive_json()["reason"] == "inference_failed"
        state = first.get("/v1/sessions/" + session_id + "/stream-state", headers=AUTH).json()
        assert state["pending_sequence"] == 0
        assert state["next_sequence"] == 0
        assert not state["results"]
    # A completely new app process can pick up the persisted PCM chunk.
    working = StreamingASR()
    with TestClient(create_app(config, runtime=InferenceRuntime(config, asr=working, llm=StreamingLLM()))) as second:
        with second.websocket_connect("/v1/stream/" + session_id) as resumed:
            resumed.send_json({"type": "hello", "token": KEY})
            assert resumed.receive_json()["next_sequence"] == 0
            chunk(resumed, 0)
            assert resumed.receive_json() == {"type": "ack", "sequence": 0, "duplicate": True}
            assert resumed.receive_json()["type"] == "partial"
            assert resumed.receive_json()["type"] == "final"
            assert working.calls == 1
            resumed.send_json({"type": "finish"})
            assert resumed.receive_json()["type"] == "complete"
