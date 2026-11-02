"""Selected desktop GGUF model is pinned per Note/Meeting, never guessed or silently replaced."""
import io
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import wave

os.environ.setdefault("HANDY_API_KEY", "test-only-token-not-for-production-1234567890")

from fastapi.testclient import TestClient
from handy_server.config import Settings
from handy_server.main import create_app
from handy_server.schemas import TranscriptSegment
from handy_server.workers import GgufWhisperWorker, InferenceRuntime

QWEN_ID = "handy-computer/qwen3-asr-1.7b-gguf/qwen3-asr-1.7b-Q5_K_M.gguf"
WHISPER_ID = "handy-computer/whisper-large-v3-gguf/whisper-large-v3-Q5_K_M.gguf"


def registry_name(model_id):
    return "asr-" + hashlib.sha256(model_id.encode("utf-8")).hexdigest() + ".gguf"


TOKEN = "test-only-token-not-for-production-1234567890"
AUTH = {"Authorization": "Bearer " + TOKEN}


def wav(seconds=2):
    data = io.BytesIO()
    with wave.open(data, "wb") as output:
        output.setnchannels(1)
        output.setframerate(16000)
        output.setsampwidth(2)
        output.writeframes(b"\x01\x00" * 16000 * seconds)
    return data.getvalue()


def setup(tmp_path):
    directory = tmp_path / "selected"
    directory.mkdir()
    source = tmp_path / "downloaded-qwen3-asr-1.7b.gguf"
    source.write_bytes(b"model placeholder")
    (directory / registry_name(QWEN_ID)).symlink_to(source)
    (directory / registry_name(WHISPER_ID)).symlink_to(source)
    default = tmp_path / "whisper-large-v3.gguf"
    default.write_bytes(b"default model placeholder")
    executable = tmp_path / "transcribe-cli"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    settings = Settings(
        api_key=TOKEN, db_path=tmp_path / "db.sqlite",
        asr_enabled=True, asr_backend="gguf_cli",
        gguf_model_path=default, gguf_model_dir=directory,
        transcribe_cli_path=executable, llm_enabled=False,
    )
    return settings


def test_selected_model_survives_upload_and_reaches_asr(tmp_path):
    settings = setup(tmp_path)

    class StubGguf(GgufWhisperWorker):
        def __init__(self, config):
            super().__init__(config)
            self.calls = []

        def _run_cli(self, audio_path, language, timeout=300, model_id=None):
            self.calls.append((model_id, language))
            return ([TranscriptSegment(
                id="s1", start=0, end=2, raw_text="جلسه فارسی",
                final=True,
            )], language)

    asr = StubGguf(settings)
    runtime = InferenceRuntime(settings, asr=asr)
    with TestClient(create_app(settings, runtime=runtime)) as client:
        data = wav()
        begin = client.post("/v1/meetings/uploads", headers=AUTH, json={
            "meeting_id": "qwen-session",
            "kind": "note",
            "language": "fa",
            "asr_model_id": QWEN_ID,
            "total_bytes": len(data),
        })
        assert begin.status_code == 200, begin.text
        uid = begin.json()["id"]
        sent = client.put("/v1/meetings/uploads/" + uid, headers={
            **AUTH, "X-Upload-Offset": "0",
        }, content=data)
        assert sent.status_code == 200, sent.text
        response = client.post("/v1/meetings/uploads/" + uid + "/finish", headers=AUTH)
        assert response.status_code == 200, response.text
        from test_meetings import wait_for
        completed = wait_for(client, response.json()["id"], "completed")
        assert completed["asr_model_id"] == QWEN_ID
        assert (QWEN_ID, "fa") in asr.calls
        assert completed["generated_note_id"] is not None
        assert completed["clean_text"] == "جلسه فارسی"


def test_missing_or_unregistered_model_rejected_before_audio_upload(tmp_path):
    settings = setup(tmp_path)
    class StubGguf(GgufWhisperWorker):
        def _run_cli(self, audio_path, language, timeout=300, model_id=None):
            return [], language

    asr = StubGguf(settings)
    with TestClient(create_app(settings, runtime=InferenceRuntime(settings, asr=asr))) as client:
        result = client.post("/v1/meetings/uploads", headers=AUTH, json={
            "meeting_id": "invalid-model",
            "total_bytes": len(wav()),
            "asr_model_id": "missing-model",
        })
        assert result.status_code == 422
        assert "not registered" in result.json()["detail"]
        traversal = client.post("/v1/meetings/uploads", headers=AUTH, json={
            "meeting_id": "traversal",
            "total_bytes": len(wav()),
            "asr_model_id": "../../etc/passwd",
        })
        assert traversal.status_code == 422


def test_qwen3_asr_has_text_without_timestamps_and_uses_auto_language(tmp_path, monkeypatch):
    settings = setup(tmp_path)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(wav())
    command_used = []
    def mock_run(command, **kwargs):
        command_used.extend(command)
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "file": str(audio),
            "text": "سلام، قرار جلسه فردا است.",
            "segments": [],
        }))
    monkeypatch.setattr("handy_server.workers.subprocess.run", mock_run)
    asr = GgufWhisperWorker(settings)
    asr._ready = True
    segments, _ = asr.transcribe_file(audio, "fa", model_id=QWEN_ID)
    assert "--language" not in command_used
    assert command_used[command_used.index("--timestamps") + 1] == "none"
    assert segments[0].raw_text == "سلام، قرار جلسه فردا است."
    assert segments[0].start == 0
    assert segments[0].end == 2


def test_selected_model_requires_gguf_runtime(tmp_path):
    settings = Settings(api_key=TOKEN, db_path=tmp_path / "unavailable.db")
    class MockOther:
        ready = True
        def initialize(self): pass
        def transcribe_file(self, path, language): return [], language
    from handy_server.workers import WorkerUnavailable
    runtime = InferenceRuntime(settings, asr=MockOther())
    import pytest
    with pytest.raises(WorkerUnavailable):
        runtime.validate_asr_model("qwen3-asr-1.7b")


def test_whisper_catalog_id_with_slashes_is_valid_and_keeps_language(tmp_path, monkeypatch):
    settings = setup(tmp_path)
    audio = tmp_path / "voice.wav"
    audio.write_bytes(wav())
    command_used = []

    def mock_run(command, **kwargs):
        command_used.extend(command)
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "file": str(audio),
            "text": "جلسه فردا",
            "segments": [{"text": "جلسه فردا", "t0_ms": 0, "t1_ms": 1500}],
        }))

    monkeypatch.setattr("handy_server.workers.subprocess.run", mock_run)
    asr = GgufWhisperWorker(settings)
    asr._ready = True
    assert asr.selected_path(WHISPER_ID).name == registry_name(WHISPER_ID)
    segments, language = asr.transcribe_file(audio, "fa", model_id=WHISPER_ID)
    assert language == "fa"
    assert segments[0].raw_text == "جلسه فردا"
    assert command_used[command_used.index("--language") + 1] == "fa"
    assert command_used[command_used.index("--timestamps") + 1] == "segment"
    assert command_used[command_used.index("-m") + 1] == str(
        settings.gguf_model_dir / registry_name(WHISPER_ID)
    )


def test_slash_ids_cannot_escape_local_asr_registry(tmp_path):
    settings = setup(tmp_path)
    asr = GgufWhisperWorker(settings)
    from handy_server.workers import WorkerUnavailable
    import pytest

    for candidate in ("../etc/passwd", "/tmp/something", "repo/../../file.gguf"):
        with pytest.raises(WorkerUnavailable):
            asr.selected_path(candidate)
    assert asr.selected_path(WHISPER_ID).is_file()
    assert asr.selected_path(QWEN_ID).is_file()
    assert registry_name(WHISPER_ID) != registry_name(QWEN_ID)


def test_native_handy_reuses_desktop_transcription_manager(tmp_path, monkeypatch):
    """The Meeting path executes the SAME headless native pipeline as dictation."""
    from handy_server.workers import NativeHandyWorker
    settings = setup(tmp_path)
    executable = tmp_path / "handy"
    executable.write_text("#!/bin/sh\\nexit 0\\n")
    executable.chmod(0o700)
    from dataclasses import replace
    settings = replace(settings, asr_backend="native_handy", native_handy_path=executable)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(wav())
    invoked = []

    def native(command, **kwargs):
        invoked.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "model": WHISPER_ID,
            "text": "این همان متن اصلاح شده توسط موتور اصلی Handy است.",
            "bound_backend": "metal",
        }), stderr="")

    monkeypatch.setattr("handy_server.workers.subprocess.run", native)
    worker = NativeHandyWorker(settings)
    worker.initialize()
    assert worker.ready
    segments, reported_language = worker.transcribe_file(audio, "fa", WHISPER_ID)
    assert reported_language is None
    assert segments[0].raw_text == "این همان متن اصلاح شده توسط موتور اصلی Handy است."
    assert segments[0].start == 0
    assert segments[0].end == 2
    assert invoked == [[
        str(executable), "--transcribe-file", str(audio),
        "--model", WHISPER_ID, "--json",
    ]]


def test_native_asr_missing_executable_never_silently_falls_back(tmp_path):
    from handy_server.workers import NativeHandyWorker, WorkerUnavailable
    from dataclasses import replace
    import pytest
    settings = replace(
        setup(tmp_path), asr_backend="native_handy",
        native_handy_path=tmp_path / "not-compiled-yet",
    )
    worker = NativeHandyWorker(settings)
    worker.initialize()
    audio = tmp_path / "speech.wav"
    audio.write_bytes(wav())
    with pytest.raises(WorkerUnavailable, match="not compiled"):
        worker.transcribe_file(audio, "fa", WHISPER_ID)


def test_native_asr_rejects_wrong_model_response(tmp_path, monkeypatch):
    from handy_server.workers import NativeHandyWorker, WorkerFailure
    from dataclasses import replace
    import pytest
    settings = setup(tmp_path)
    binary = tmp_path / "handy"
    binary.write_text("#!/bin/sh\\nexit 0\\n")
    binary.chmod(0o700)
    settings = replace(settings, asr_backend="native_handy", native_handy_path=binary)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(wav())
    monkeypatch.setattr(
        "handy_server.workers.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"model": QWEN_ID, "text": "wrong model"}),
            stderr="",
        ),
    )
    worker = NativeHandyWorker(settings)
    worker.initialize()
    with pytest.raises(WorkerFailure, match="invalid transcript"):
        worker.transcribe_file(audio, "fa", WHISPER_ID)
