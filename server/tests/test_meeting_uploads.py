"""Phase 5 resumable-upload tests, auth, offsets, persistence and finalize."""
import io
import os
import wave

os.environ.setdefault("HANDY_API_KEY", "test-only-token-not-for-production-1234567890")
from fastapi.testclient import TestClient
from handy_server.config import Settings
from handy_server.main import create_app

TOKEN = "test-only-token-not-for-production-1234567890"
AUTH = {"Authorization": "Bearer " + TOKEN}


def sample(seconds=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setframerate(16000)
        file.setsampwidth(2)
        file.writeframes(b"\x01\x00" * (16000 * seconds))
    return buffer.getvalue()


class DummyAsr:
    ready = True
    def initialize(self): pass
    def transcribe_file(self, path, language):
        from handy_server.schemas import TranscriptSegment
        return [TranscriptSegment(id="s1", start=0, end=1,
                                  raw_text="آزمایش", final=True)], "fa"


class DummyLlm:
    async def ready(self): return True


def test_resume_upload_and_deduplicate(tmp_path):
    from handy_server.workers import InferenceRuntime
    from handy_server.schemas import MeetingAnalysis
    from handy_server.meeting_analysis import QwenMeetingAnalyzer

    # Keep the job queued while checking the byte offset. Once submitted,
    # async worker may process; upload correctness is tested independently.
    config = Settings(api_key=TOKEN, db_path=tmp_path / "jobs.sqlite3",
                      max_meeting_bytes=3000000)
    runtime = InferenceRuntime(config, asr=DummyAsr(), llm=DummyLlm())
    data = sample(2)
    with TestClient(create_app(config, runtime)) as client:
        start = client.post("/v1/meetings/uploads", headers=AUTH, json={
            "meeting_id": "my-meeting-123", "total_bytes": len(data), "language": "fa",
        })
        assert start.status_code == 200
        uid = start.json()["id"]
        assert len(uid) == 32
        assert start.json()["offset"] == 0
        half = len(data) // 2
        first = client.put(
            "/v1/meetings/uploads/" + uid,
            headers={**AUTH, "X-Upload-Offset": "0"},
            content=data[:half],
        )
        assert first.status_code == 200, first.text
        assert first.json()["offset"] == half
        wrong_offset = client.put(
            "/v1/meetings/uploads/" + uid,
            headers={**AUTH, "X-Upload-Offset": "0"},
            content=b"duplicated",
        )
        assert wrong_offset.status_code == 409
        assert wrong_offset.json()["detail"]["next_offset"] == half
        status = client.get("/v1/meetings/uploads/" + uid, headers=AUTH)
        assert status.json()["offset"] == half
        resumed = client.post("/v1/meetings/uploads", headers=AUTH, json={
            "meeting_id": "my-meeting-123", "total_bytes": len(data),
        })
        assert resumed.json()["id"] == uid
        assert resumed.json()["offset"] == half
        assert client.post("/v1/meetings/uploads/" + uid + "/finish",
                           headers=AUTH).status_code == 409
        second = client.put(
            "/v1/meetings/uploads/" + uid,
            headers={**AUTH, "X-Upload-Offset": str(half)},
            content=data[half:],
        )
        assert second.json()["offset"] == len(data)
        completed = client.post(
            "/v1/meetings/uploads/" + uid + "/finish", headers=AUTH
        )
        assert completed.status_code == 200, completed.text
        job_id = completed.json()["id"]
        assert len(job_id) == 32
        again = client.post(
            "/v1/meetings/uploads/" + uid + "/finish", headers=AUTH
        )
        assert again.json()["id"] == job_id
        assert client.post("/v1/meetings/uploads", headers=AUTH, json={
            "meeting_id": "my-meeting-123", "total_bytes": len(data)
        }).json()["job_id"] == job_id


def test_upload_survives_server_restart(tmp_path):
    config = Settings(api_key=TOKEN, db_path=tmp_path / "restart.db")
    first_half = sample()[:13000]
    with TestClient(create_app(config)) as app:
        upload = app.post("/v1/meetings/uploads", headers=AUTH, json={
            "meeting_id": "restart-test", "total_bytes": len(sample()),
        }).json()
        uid=upload["id"]
        uploaded=app.put("/v1/meetings/uploads/" + uid,
                         headers={**AUTH, "X-Upload-Offset": "0"}, content=first_half)
        assert uploaded.json()["offset"] == len(first_half)
    with TestClient(create_app(config)) as resumed:
        state=resumed.get("/v1/meetings/uploads/" + uid, headers=AUTH).json()
        assert state["offset"] == len(first_half)


def test_upload_requires_auth_and_bounds(tmp_path):
    config = Settings(api_key=TOKEN, db_path=tmp_path / "bounded.db",
                      max_meeting_bytes=2048)
    with TestClient(create_app(config)) as client:
        assert client.post("/v1/meetings/uploads", json={
            "meeting_id": "x", "total_bytes": 100,
        }).status_code == 401
        assert client.post("/v1/meetings/uploads", json={
            "meeting_id": "../traverse", "total_bytes": 100,
        }, headers=AUTH).status_code == 422
        assert client.post("/v1/meetings/uploads", json={
            "meeting_id": "x", "total_bytes": 2049,
        }, headers=AUTH).status_code == 413
        uid = client.post("/v1/meetings/uploads", headers=AUTH, json={
            "meeting_id": "small", "total_bytes": 500,
        }).json()["id"]
        assert client.put("/v1/meetings/uploads/" + uid,
                          headers={**AUTH, "X-Upload-Offset": "0"},
                          content=b"x" * 501).status_code == 422
        assert client.put("/v1/meetings/uploads/" + uid,
                          headers={**AUTH, "X-Upload-Offset": "0"},
                          content=b"x" * 500).status_code == 200
        assert client.post("/v1/meetings/uploads/" + uid + "/finish",
                           headers=AUTH).status_code == 422
        assert client.get("/v1/meetings/uploads/" + uid).status_code == 401
