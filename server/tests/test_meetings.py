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
            assert 0 < seconds <= 120
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
    assert result["transcript_count"] == 1
    assert result["processed_seconds"] == 31
    assert asr.calls == [31]
    assert result["transcript"][0]["speaker"] == "Unknown"
    assert result["transcript"][0]["id"] == "u00000001"
    assert result["insights"]["tasks"][0]["assignee"] == "علی"
    assert result["insights"]["tasks"][0]["source_ids"] == ["u00000001"]
    saved = client.get("/v1/tasks", headers=HEADERS).json()
    assert saved["total"] == 1
    assert saved["items"][0]["source_meeting_id"] == job["id"]
    assert saved["items"][0]["source_quotes"][0]["id"] == "u00000001"
    assert saved["items"][0]["source_meeting_title"] == result["title"]
    assert result["insights"]["open_questions"] == []
    page = client.get(
        "/v1/meetings/" + job["id"] + "/transcript?offset=0&limit=1",
        headers=HEADERS,
    ).json()
    assert page["total"] == 1
    assert page["items"][0]["id"] == "u00000001"
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


def test_hierarchical_summary_retains_all_source_evidence(tmp_path):
    """Evidence must not disappear while summaries are reduced for long meetings."""
    import asyncio
    from handy_server.meeting_analysis import QwenMeetingAnalyzer

    settings = Settings(api_key=KEY, db_path=tmp_path / "a.db", llm_enabled=True)
    llm = QwenWorker(settings, transport=httpx.MockTransport(handler))
    analyzer = QwenMeetingAnalyzer(settings, llm)
    entries = [
        {
            "id": f"u{i:08d}",
            "speaker": "Unknown",
            "start": float(i),
            "end": float(i + 1),
            "text": "تصمیم: علی گزارش را آماده کند.",
        }
        for i in range(1, 100)
    ]
    result = asyncio.run(analyzer.analyze(entries))
    # At least two independent source IDs must survive a reduce stage.
    ids = {ref for task in result.tasks for ref in task.source_ids}
    assert len(ids) >= 2
    assert result.summary
    assert all(ref in {row["id"] for row in entries} for ref in ids)

def test_note_capture_uses_same_whisper_pipeline_and_persists_automatically(engine):
    client, asr, _ = engine
    response = client.post(
        "/v1/meetings", headers=HEADERS,
        files={"audio": ("note.wav", wav_data(3), "audio/wav")},
        data={"meeting_id": "voice-note-2026-test", "kind": "note", "title": "یادداشت من"},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    result = wait_for(client, job_id, "completed")
    assert result["kind"] == "note"
    assert result["generated_note_id"]
    assert result["clean_text"]
    note = client.get("/v1/notes/by-recording/" + job_id, headers=HEADERS)
    assert note.status_code == 200, note.text
    assert note.json()["title"] == "یادداشت من"
    assert "تصمیم" in note.json()["content"]
    assert note.json()["source_meeting_id"] == job_id
    assert client.get("/v1/tasks", headers=HEADERS).json()["total"] == 0
    assert client.get("/v1/meetings/" + job_id + "/cleaned",
                      headers=HEADERS).json()["total"] == 1
    # Idempotent retry/duplicate upload must not generate more notes.
    again = client.post(
        "/v1/meetings", headers=HEADERS,
        files={"audio": ("note.wav", wav_data(3), "audio/wav")},
        data={"meeting_id": "voice-note-2026-test", "kind": "note"},
    )
    assert again.json()["id"] == job_id
    assert client.get("/v1/notes", headers=HEADERS).json()["total"] == 1


def test_whisper_transcript_survives_unavailable_qwen(tmp_path):
    """Optional summarization outage must not erase a successful ASR transcript."""
    settings = Settings(api_key=KEY, db_path=tmp_path / "offline-llm.db",
                        llm_enabled=False, max_meeting_bytes=3000000)
    runtime = InferenceRuntime(settings, asr=MockASR(), llm=QwenWorker(settings))
    with TestClient(create_app(settings, runtime=runtime)) as client:
        submitted = client.post(
            "/v1/meetings", headers=HEADERS,
            files={"audio": ("fa.wav", wav_data(2), "audio/wav")},
            data={"meeting_id": "offline-qwen", "language": "fa"},
        )
        assert submitted.status_code == 202, submitted.text
        result = wait_for(client, submitted.json()["id"], "failed")
        assert result["transcript_count"] == 1
        assert "تصمیم" in result["clean_text"]
        blocks = client.get(
            "/v1/meetings/" + result["id"] + "/cleaned", headers=HEADERS,
        ).json()
        assert blocks["total"] == 1
        assert "تصمیم" in blocks["items"][0]


def test_ollama_meeting_analysis_retries_without_json_grammar_on_501(tmp_path):
    """Keep using the same Qwen model when MLX lacks constrained JSON mode."""
    import asyncio
    calls = []
    def backend(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "qwen3.5:9b"}]})
        payload = json.loads(request.content)
        calls.append(payload)
        if "response_format" in payload:
            return httpx.Response(501, json={"error": "structured output is unavailable"})
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"title":null,"meeting_date":null,"summary":"جلسه فارسی",'
            '"decisions":[],"tasks":[],"open_questions":[]}'}}]})

    settings = Settings(
        api_key=KEY, db_path=tmp_path / "llm.db",
        qwen_provider="ollama", qwen_base_url="http://127.0.0.1:11434/v1",
        qwen_model="qwen3.5:9b", llm_enabled=True,
    )
    worker = QwenWorker(settings, transport=httpx.MockTransport(backend))
    result = asyncio.run(worker.structured(
        [{"role": "system", "content": "JSON only"},
         {"role": "user", "content": "A Persian meeting"}],
        max_tokens=2048, operation="meeting_analysis",
    ))
    assert result["summary"] == "جلسه فارسی"
    assert len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    assert all(item["model"] == "qwen3.5:9b" for item in calls)
    assert calls[0]["messages"] == calls[1]["messages"]


def test_ollama_non_501_failure_never_retries_without_validation(tmp_path):
    import asyncio
    calls = []
    def backend(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "qwen3.5:9b"}]})
        calls.append(json.loads(request.content))
        return httpx.Response(429)

    settings = Settings(
        api_key=KEY, db_path=tmp_path / "error.db",
        qwen_provider="ollama", qwen_base_url="http://127.0.0.1:11434/v1",
        qwen_model="qwen3.5:9b", llm_enabled=True,
    )
    worker = QwenWorker(settings, transport=httpx.MockTransport(backend))
    from handy_server.workers import WorkerFailure
    with pytest.raises(WorkerFailure):
        asyncio.run(worker.structured(
            [{"role": "system", "content": "JSON only"}],
            max_tokens=100, operation="meeting_analysis",
        ))
    assert len(calls) == 1
