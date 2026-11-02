# Handy AI Server — Phases 2–3

Handy now has an authenticated non-streaming transcription API backed by
**Whisper Large V3** (faster-whisper/CTranslate2), plus a text cleanup API
backed by **Qwen3.5-9B** through a private OpenAI-compatible vLLM server.

The existing desktop **offline dictation remains the default** until
incremental streaming is built in phase 3. Full meeting analysis and notes are
not part of phase 2. Both workers are **disabled by default** and readiness
reflects actual model availability, not a fake success.

## Install API and Whisper on a GPU host

Requirements: Python 3.11+, GPU and current NVIDIA driver, CUDA 12 and
cuDNN 9 (or set CPU fallback). From the repository root:

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,asr]'
export HANDY_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export HANDY_ASR_ENABLED=true
export HANDY_ASR_DEVICE=cuda
export HANDY_ASR_COMPUTE_TYPE=float16
```

The model will download and load on startup. On CPU, choose
`HANDY_ASR_DEVICE=cpu` and `HANDY_ASR_COMPUTE_TYPE=int8`;
large-v3 on CPU can be very slow.

## Install Qwen3.5-9B separately

Use a vLLM version supporting Qwen3.5, ideally following the model's official
installation recipe. In a **separate environment** install vLLM, then run:

```bash
cd server
bash scripts/run_qwen.sh
```

The Qwen server listens only on `127.0.0.1:8000`. On the Handy API terminal:

```bash
export HANDY_LLM_ENABLED=true
export HANDY_QWEN_BASE_URL=http://127.0.0.1:8000/v1
export HANDY_QWEN_MODEL=Qwen/Qwen3.5-9B
uvicorn handy_server.main:app --host 127.0.0.1 --port 8765
```

Running both models on one 24 GB GPU may fail due to VRAM limits.
Use separate GPUs or hosts for production, or size and benchmark the models.

An optional `Dockerfile.gpu` is included for the Handy API/Whisper service.
It requires NVIDIA Container Toolkit and an explicit private network path to
the Qwen service. The Qwen host-loopback address is not reachable inside an
ordinary Docker container; configure `HANDY_QWEN_BASE_URL` appropriately.

## Endpoints

- `GET /health` — liveness only.
- `GET /v1/capabilities` — authenticated actual model readiness.
- `GET /v1/models/status` — Whisper/Qwen state and model names.
- `POST /v1/text/cleanup` — JSON with `text`, `tone` (verbatim,
  professional, formal, casual, friendly), `language` and
  `protected_terms`. Returns raw and clean variants.
- `POST /v1/transcriptions` — multipart short audio plus optional
  `language`, `cleanup` and `tone`. Returns timestamps and clean text.
- `POST /v1/sessions` and `GET /v1/sessions/{id}` — session metadata
  retained from phase 1.
- Streaming audio chunks and full meeting jobs continue to return HTTP
  501 until their later phases.

Only WAV, MP3, FLAC, M4A, OGG, WEBM and MP4 are accepted. Default upload
limit is 25 MiB. This endpoint is for short files, **not a 20-hour recording**.

## Verify using real models

```bash
curl -H "Authorization: Bearer $HANDY_API_KEY" \
  http://127.0.0.1:8765/v1/models/status

curl -X POST http://127.0.0.1:8765/v1/text/cleanup \
  -H "Authorization: Bearer $HANDY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"خب هزینه ۱۵۰۰ تومان شد.","tone":"formal","language":"fa"}'

curl -X POST http://127.0.0.1:8765/v1/transcriptions \
  -H "Authorization: Bearer $HANDY_API_KEY" \
  -F "audio=@/path/to/test.wav" -F "language=fa" \
  -F "cleanup=true" -F "tone=professional"

python scripts/smoke_models.py --audio /path/to/test.wav --language fa
```

Use a recording whose participants consented to processing.

## Safety and deployment constraints

- A shared bearer API key is **single-tenant only**. Before public deployment
  add accounts, per-user authorization, quotas, HTTPS, upload body limits
  at the reverse proxy, rate limits and retention policies.
- Only localhost HTTP is allowed in desktop settings; remote endpoints
  must be HTTPS. Keep vLLM private; never expose its port publicly.
- Audio uploads go to a bounded temporary file and are deleted after
  inference; the server does not save the audio or transcript.
- Raw transcript is always retained in API output. Numeric differences and
  lost explicitly protected terms cause safe fallback to raw text. This
  is not proof of complete semantic accuracy; users should review edits.
- CI uses mock workers and does **not** download multi-GB models or
  prove real GPU performance. See `docs/ai-phase2-architecture.md`.

## Tests and synthetic benchmark

```bash
pip install -e '.[dev]'
pytest -q
python benchmarks/measure_api.py
```

## Phase 3 — live, resumable dictation

The Handy sidebar now has an **opt-in Live** page. Enter the server URL and
the single-tenant bearer token, agree to microphone processing, and start.
The desktop's original global shortcut and offline transcription remain
unchanged. This is not yet system-wide live typing/paste (phase 4).

The client captures microphone audio with an AudioWorklet (ScriptProcessor
fallback), downsamples to 16 kHz mono PCM16 and sends _new_, non-overlapping
1–5-second chunks at voice pauses or the five-second boundary. The server
prepends up to 0.6 seconds of _previous_ persisted PCM as recognition context,
then removes matching repeated words before finalizing the segment. Partial
raw text is emitted when Whisper finishes a chunk, followed by a final
committed segment after optional Qwen editing. This is chunked Whisper, not
true token-level Whisper streaming.

### WebSocket v1 contract

1. Create an authenticated dictation session with `POST /v1/sessions`.
2. Open `ws(s)://HOST/v1/stream/{session_id}`. For remote endpoints use WSS.
3. Send `{"type":"hello","token":"...","language":"fa","tone":"professional","cleanup":true,"last_seen":-1}`
   as the **first WebSocket JSON message**. Never put the token in a URL.
4. Server sends `{"type":"ready","next_sequence":0,"pending_sequence":null,"committed_count":0,"results":[]}`.
5. Send JSON metadata `{"type":"chunk","sequence":0,"encoding":"pcm_s16le","sample_rate":16000}`
   immediately followed by one **binary** PCM chunk (mono signed 16-bit little-endian).
   The server replies with an `ack` once the bytes are durable, a `partial`
   when raw ASR finishes, then a `final` after the commit. Never discard your
   local chunk on ACK alone.
6. After a disconnect, reopen the same session and retransmit unfinalized
   sequences. Identical retransmits are idempotent; changed bytes with the
   same sequence are rejected. Out-of-order chunks receive `resync`.
7. On stop, send `{"type":"finish"}` **after every locally queued chunk is final**.
   The server transitions the session to `completed`.
8. `GET /v1/sessions/{id}/stream-state?after_sequence=-1&limit=100`
   returns paginated final results and the first pending sequence.

Accepted audio is fsynced as PCM files beside the SQLite database; final
segments (including raw and edited text) are persisted in SQLite. Each chunk
is between 0.1–6 seconds; cumulative PCM duration is limited to 20 hours
per session. There is currently no automated garbage collection/retention.
Plan disk headroom of at least 2.3 GB per 20-hour session, plus temporary
files and SQLite. **WebView/IndexedDB storage quotas may prevent a reliable
20-hour continuous unattended recording.** Production long-form sessions
should use the native Rust recorder and resumable upload pipeline.

### Before deploying

- HTTPS/WSS and an authenticated private server (do not expose Uvicorn
  plaintext or vLLM).
- Whisper Large V3 on the GPU; `/v1/capabilities.processing.streaming`
  becomes true **only when ASR is actually loaded**.
- Check WebView microphone permissions on the target macOS/Windows build.
- Verify reconnection, device sleep, disk pressure and low-bandwidth behavior
  on a real client. CI uses mocks; real audio accuracy and 20-hour endurance
  have not been hardware-tested.
- Multi-worker FastAPI deployment needs a shared distributed session lock;
  the current WebSocket per-session lock is **per-process**, and this release
  supports only a single API process per server.

Run `cd server && python -m pytest -q` for protocol regression tests.

## Phase 5: Meeting Mode

The server supports durable meeting jobs, 30-second windowed transcription,
optional pyannote speaker diarization, grounded Qwen summaries and resumable
4 MiB chunk uploads. Full operational instructions, permissions, security
limits and API routes: [Phase 5 Meeting Mode](../docs/ai-phase5-meeting.md).
