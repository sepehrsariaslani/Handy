"""Phase 5: disk-streamed upload, real processing contract, durable retry, evidence."""
import io
import json
import os
import time
import wave
from pathlib import Path

os.environ.setdefault("HANDY_API_KEY", "test-only-token-not-for-production-1234567890")

import httpx
from fastapi.testclient import TestClient
import pytest

from handy_server.config import Settings
from handy_server.main import create_app
from handy_server.meeting_store import MeetingStore
from handy_server.meeting_processing import validated_duration
from handy_server.diarization import label_segments
from handy_server.schemas import TranscriptSegment
from handy_server.workers import InferenceRuntime, QwenWorker

KEY = "test-only-token-not-for-production-1234567890"
HEADERS = {"Authorization": "Bearer " + KEY}


def wav_data(seconds: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setframerate(16000)
        wav.setsampwidth(2)
        # Keep tests small; zero/silence is accepted by the fake ASR.
        wav.writeframes(b"\0\0" * 16000 * seconds)
    return buffer.getvalue()


class MockASR:
    ready = True
    def __init__(self):
        self.calls = []
        self.fail = False
    def initialize(self):
        pass
    def transcribe_file(self, file: Path, language: str | None):
        with wave.open(str(file), "rb") as wav:
            seconds = wav.getnframes() / 16000
            assert 0 < seconds <= 30
        self.calls.append(seconds)
        if self.fail:
            from handy_server.workers import WorkerFailure
            self.fail = False
            raise WorkerFailure("Synthetic GPU crash")
        return [TranscriptSegment(id="tmp", start=0, end=seconds,
                                  raw_text="تصمیم: علی گزارش را آماده کند.",
                                  final=True)], "fa"


def handler(req: httpx.Request):
    if req.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.5-9B"}]})
    body = json.loads(req.content)
    prompt = body["messages"][1]["content"]
    # Source IDs must originate in the supplied prompt; the model adapter filters hallucinated IDs.
    import re
    ids = list(dict.fromkeys(re.findall(r"u[0-9]{8}", prompt)))
    return httpx.Response(200, json={"choices": [{"message": {
        "content": json.dumps({
            "summary": "علی مسئول تهیه گزارش شد.",
            "decisions": [{"text": "تهیه گزارش", "source_ids": ids[:1],
                           "assignee": None, "due_date": None}],
            "tasks": [{"text": "گزارش تهیه شود", "source_ids": ids[:1],
                       "assignee": "علی", "due_date": None}],
            "open_questions": [{"text": "بدون منبع", "source_ids": ["u99999999"],
                                "assignee": None, "due_date": None}],
        }, ensure_ascii=False),
    }}]})


@pytest.fixture()
def engine(tmp_path):
    settings = Settings(
        api_key=KEY, db_path=tmp_path / "meeting.db",
        llm_enabled=True, max_meeting_bytes=3000000,
    )
    asr = MockASR()
    llm = QwenWorker(settings, transport=httpx.MockTransport(handler))
    runtime = InferenceRuntime(settings, asr=asr, llm=llm)
    with TestClient(create_app(settings, runtime=runtime)) as client:
        yield client, asr, settings


def wait_for(client, job_id, target, timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.get("/v1/meetings/" + job_id, headers=HEADERS)
        assert result.status_code == 200
        if result.json()["status"] in {"completed", "failed"}:
            assert result.json()["status"] == target, result.json()
            return result.json()
        time.sleep(0.02)
    pytest.fail("Meeting processing did not finish")


def test_meeting_upload_and_evidence(engine):
    client, asr, settings = engine
    data = wav_data(31)
    created = client.post(
        "/v1/meetings", headers=HEADERS,
        files={"audio": ("recording.wav", data, "audio/wav")},
        data={"meeting_id": "meeting-2026-1", "language": "fa"},
    )
    assert created.status_code == 202, created.text
    job = created.json()
    assert len(job["id"]) == 32
    result = wait_for(client, job["id"], "completed")
    assert result["transcript_count"] == 2
    assert result["processed_seconds"] == 31
    assert asr.calls == [30, 1]
    assert result["transcript"][0]["speaker"] == "Unknown"
    assert result["transcript"][0]["id"] == "u00000001"
    assert result["transcript"][1]["start"] == 30
    assert result["insights"]["tasks"][0]["assignee"] == "علی"
    assert result["insights"]["tasks"][0]["source_ids"] == ["u00000001"]
    assert result["insights"]["open_questions"] == []
    page = client.get(
        "/v1/meetings/" + job["id"] + "/transcript?offset=1&limit=1",
        headers=HEADERS,
    ).json()
    assert page["total"] == 2
    assert page["items"][0]["id"] == "u00000002"
    # Duplicate uploads with the same client_ref must reuse the prior job.
    again = client.post("/v1/meetings", headers=HEADERS,
                        files={"audio": ("recording.wav", data, "audio/wav")},
                        data={"meeting_id": "meeting-2026-1"})
    assert again.json()["id"] == job["id"]


def test_rejects_bad_audio_and_auth(engine):
    client, _, _ = engine
    assert client.post("/v1/meetings").status_code == 401
    invalid = client.post("/v1/meetings", headers=HEADERS,
                          files={"audio": ("bad.wav", b"not-a-wave", "audio/wav")})
    assert invalid.status_code == 422
    wrong = client.post("/v1/meetings", headers=HEADERS,
                        files={"audio": ("bad.mp3", b"test", "audio/mpeg")})
    assert wrong.status_code == 415
    assert client.get("/v1/meetings/" + "e" * 32, headers=HEADERS).status_code == 404
    assert client.get("/v1/meetings/" + "f" * 32 + "/transcript?limit=900", headers=HEADERS).status_code == 404


def test_resume_from_saved_checkpoint(tmp_path):
    file = tmp_path / "session.wav"
    file.write_bytes(wav_data(31))
    store = MeetingStore(tmp_path / "db.sqlite")
    job = store.create(file, 31.0, "resumed", "fa")
    store.checkpoint(job["id"], 30*16000, [{
        "id": "u00000001", "speaker": "Unknown", "start": 0,
        "end": 30, "text": "تا دقیقه اول",
    }])
    # Reopen store to simulate process restart. The next worker must start
    # from frame 480000, not rerun the first 30 seconds.
    settings = Settings(api_key=KEY, db_path=tmp_path / "db.sqlite", llm_enabled=True)
    asr = MockASR()
    runtime = InferenceRuntime(settings, asr=asr,
                               llm=QwenWorker(settings, transport=httpx.MockTransport(handler)))
    with TestClient(create_app(settings, runtime=runtime)) as client:
        result = wait_for(client, job["id"], "completed")
        assert asr.calls == [1]
        assert result["transcript_count"] == 2


def test_failed_inference_can_retry(engine):
    client, asr, settings = engine
    asr.fail = True
    result = client.post("/v1/meetings", headers=HEADERS,
                         files={"audio": ("a.wav", wav_data(), "audio/wav")})
    job_id = result.json()["id"]
    result = wait_for(client, job_id, "failed")
    assert "WorkerFailure" in result["error"]
    retry = client.post("/v1/meetings/" + job_id + "/retry", headers=HEADERS)
    assert retry.status_code == 200
    assert wait_for(client, job_id, "completed")["transcript_count"] == 1
    assert client.post("/v1/meetings/" + job_id + "/retry", headers=HEADERS).status_code == 409


def test_meeting_length_and_speaker_alignment(tmp_path):
    path = tmp_path / "m.wav"
    path.write_bytes(wav_data(3))
    assert validated_duration(path) == 3
    with pytest.raises(ValueError):
        validated_duration(path, max_seconds=2)
    utterances = [
        {"id": "u00000001", "start": 0, "end": 2, "text": "a"},
        {"id": "u00000002", "start": 2, "end": 3, "text": "b"},
        {"id": "u00000003", "start": 4, "end": 5, "text": "c"},
    ]
    turns = [(0, 1.5, "SPEAKER_00"), (1.5, 3, "SPEAKER_01")]
    assert label_segments(utterances, turns) == [
        ("u00000001", "SPEAKER_00"),
        ("u00000002", "SPEAKER_01"),
        ("u00000003", "Unknown"),
    ]
