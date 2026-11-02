# Handy Meeting Service (opt-in)

This adds an **independent, self-hosted** transcription/diarization/meeting-notes service for the Handy desktop app. It does not change Handy's existing short dictation workflow. Audio is uploaded **only after the user explicitly clicks Process**. Processing is on your own server; no provider API is required.

## Models

- **Speech:** faster-whisper with `large-v3` by default.
- **Speaker diarization:** `pyannote/speaker-diarization-community-1` with its gated model license and Hugging Face token.
- **Notes:** Ollama `qwen3.5:9b`. Install/run Ollama on the processing host and pull `qwen3.5:9b` separately. The model is **not** bundled with Handy.

## Setup (Python 3.11+, suitable GPU recommended)

```sh
cd services/meeting
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3.5:9b
# Visit https://huggingface.co/pyannote/speaker-diarization-community-1 and accept its conditions.
export HF_TOKEN=your_hf_token
export MEETING_API_KEY=choose-a-long-random-secret
uvicorn app:app --host 127.0.0.1 --port 8765
```

Use `http://127.0.0.1:8765` in the Meeting tab for a local service. For a different machine, expose the service **only behind HTTPS** with authentication; use the same `MEETING_API_KEY` in Handy. Do not expose Ollama's port publicly. To change the storage directory, set `MEETING_DATA_DIR`. The WAV audio and JSON results stay on the server until manually deleted. Protect/rotate those files according to your retention policy.

## Usage

1. Inform meeting participants and get consent before recording.
2. Select **Meeting** in Handy, Start, then Stop. A 16 kHz mono PCM WAV is written to your local app data directory under `meetings/`; memory consumption is bounded by disk I/O instead of meeting length.
3. Enter the processing server URL and (if set) bearer token, click **Process**. The server queues the job and records its state to disk, recovering queued/interrupted jobs after a restart.
4. Handy polls for the word-timed, speaker-labelled transcript, summary, decisions, action items, and open questions. Every extracted task is a **proposal** with references to exact transcript utterances; no task is dispatched automatically.

## Limits / production considerations

- The current service has **one in-process worker** (durable job metadata and retry after restart, but not a distributed task queue). A 20-hour file is ~2.3 GB as WAV, and requires significant time, storage, compute, and upload bandwidth. Max upload defaults to 3 GiB. Full-file pyannote diarization is computationally expensive and may need a high-memory GPU or a scalable segmentation/identity-stitching service for very long recordings.
- Recording writes WAV to disk with periodic flushes; a sudden power loss may require WAV header recovery. Disk full and capture failures are surfaced on Stop. Keep the app running until Stop.
- The desktop uses the OS default microphone in this first version. Use dedicated audio input if needed; desktop system sound/remote speakers are **not** captured automatically.
- The model assigns provisional labels such as `SPEAKER_00`, **not personal identities**. A user must verify speakers, assignments, and deadlines before relying on output.
- Language auto-detection in mixed-language sessions, overlapping speakers, and timestamps can be inaccurate.
- No mobile client, always-on daily capture, global long-term memory search, or automatic todo sync is shipped by this change.
- This is **not** a deployed service; run it on your own server and check the server logs. The processing endpoint is only invoked by explicit user action.
