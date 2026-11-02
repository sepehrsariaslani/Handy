# Handy AI Server — Phases 2–3

Handy now has an authenticated non-streaming transcription API backed by
**Whisper Large V3** (faster-whisper/CTranslate2), plus a text cleanup API
backed by **Qwen3.5-9B** through a private OpenAI-compatible vLLM server.

The existing desktop **offline dictation remains the default** until
incremental streaming is built in phase 3. Full meeting analysis and notes are
not part of phase 2. Both workers are **disabled by default** and readiness
reflects actual model availability, not a fake success.


## Switch the desktop-selected GGUF model in Meeting and Notes (macOS development)

When launching with `bun run dev:mac`, the app and local API share
`server/data/asr-models`. For each *new* Meeting or Note, the recorder
pins Handy's selected and fully downloaded model ID and registers a symlink
to the existing GGUF file. No additional model download or copy is required.
The resumable upload pins that ID and the server runs the same selected
transcribe.cpp GGUF, instead of always using `whisper-large-v3-Q5_K_M.gguf`.
The chosen model appears under the Meeting details and in backend ASR logs.
Change the model in Handy settings before starting another recording.

Qwen3-ASR-1.7B GGUF is supported by transcribe.cpp and natively detects
Persian. It does **not** supply speaker labels or timestamps, so a recording
will appear as a single time-spanning utterance if the CLI returns only text.
Whisper retains segment timestamps. A selected GGUF that cannot run with the
installed `transcribe-cli` produces a visible job failure; it never silently
falls back to another model. Update the local transcribe.cpp checkout and
rebuild its CLI if a newly downloaded model is not recognized.

Only the local Mac development server configured with
`HANDY_GGUF_MODEL_DIR` supports this automatic selected-model registry.
Existing remote GPU servers still use their explicitly configured model.
Legacy uploads without an `asr_model_id` continue to use the configured
default model. Changing the selected model does not retroactively
retranscribe an existing recording. Do not expose the local model registry
directory as a network writeable share.

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

## Run Qwen locally with Ollama (macOS and CPU development)

The OpenAI-compatible Ollama API can serve the same Qwen adapter without vLLM.
Keep Ollama running, pull `qwen3.5:9b`, and start the Handy API with:

```bash
export HANDY_LLM_ENABLED=true
export HANDY_QWEN_BASE_URL=http://127.0.0.1:11434/v1
export HANDY_QWEN_MODEL=qwen3.5:9b
export HANDY_QWEN_API_KEY=ollama
# Optional explicit override if Ollama runs on a port other than 11434:
# export HANDY_QWEN_PROVIDER=ollama
```

The worker recognizes localhost port 11434 automatically. It sends
`reasoning_effort: "none"` to Ollama's OpenAI-compatible chat-completions
endpoint instead of vLLM's `chat_template_kwargs`. Production vLLM
deployments retain their original request format; use
`HANDY_QWEN_PROVIDER=vllm` to force that format. Backend restart is required
after environment changes. `/v1/models/status` confirms reachability but
does **not** prove inference; test `/v1/text/cleanup` separately. A cold
9B model on a memory-limited Mac may be slow or fail to allocate memory.

## Reuse an existing Whisper Large V3 GGUF on macOS (no Hugging Face download)

Handy desktop's GGUF model can be reused for Meeting Mode through an opt-in
transcribe.cpp CLI backend. This does not download CTranslate2 weights.

Build the official transcribe.cpp command-line tool once:

```bash
cd ~/Projects/Handy/server
brew install cmake
mkdir -p tools models
git clone --depth 1 https://github.com/handy-computer/transcribe.cpp.git tools/transcribe.cpp
cmake -S tools/transcribe.cpp -B tools/transcribe.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/transcribe.cpp/build --target transcribe-cli -j 4
```

Locate the already downloaded model in the shared HF cache and link it,
without duplicating the model data:

```bash
cd ~/Projects/Handy/server
CACHE="$HOME/.cache/huggingface/hub/models--handy-computer--whisper-large-v3-gguf/snapshots"
MODEL="$(find -L "$CACHE" -name 'whisper-large-v3-Q5_K_M.gguf' -type f -print -quit)"
test -n "$MODEL" && test -s "$MODEL"
if [ ! -e models/whisper-large-v3-Q5_K_M.gguf ] && [ ! -L models/whisper-large-v3-Q5_K_M.gguf ]; then
  ln -s "$MODEL" models/whisper-large-v3-Q5_K_M.gguf
fi
ls -lhL models/whisper-large-v3-Q5_K_M.gguf
```

Start only one Uvicorn process on 127.0.0.1:8765. Keep existing
HANDY_API_KEY and HANDY_DB_PATH from the local server setup:

```bash
source .venv/bin/activate
export HANDY_ASR_ENABLED=true
export HANDY_ASR_BACKEND=gguf_cli
export HANDY_GGUF_MODEL_PATH="$PWD/models/whisper-large-v3-Q5_K_M.gguf"
export HANDY_TRANSCRIBE_CLI_PATH="$PWD/tools/transcribe.cpp/build/bin/transcribe-cli"
export HANDY_GGUF_COMPUTE_BACKEND=auto
export HANDY_LLM_ENABLED=true
export HANDY_QWEN_PROVIDER=ollama
export HANDY_QWEN_BASE_URL=http://127.0.0.1:11434/v1
export HANDY_QWEN_MODEL=qwen3.5:9b
export HANDY_QWEN_API_KEY=ollama
python -m uvicorn handy_server.main:app --host 127.0.0.1 --port 8765 --workers 1
```

The server verifies the executable and model with a synthetic short WAV
on startup before reporting ready status; there is no HF request.
Test /v1/models/status and a short recorded meeting to confirm actual inference.
The CLI must support --batch, --batch-jsonl and --timestamps segment.

This opt-in integration starts a fresh CLI per 30-second meeting window and
reloads the memory-mapped GGUF on each call. It is suitable for initial
end-to-end testing, not yet a persistent inference worker or a proven
20-hour unattended session. Meeting WAV is 16 kHz mono PCM16. Other formats
may need conversion externally. Diarization remains a separate optional model.
The faster-whisper backend remains the default for a GPU server.
The model in server/models is a symlink; do not commit weight files to Git.

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

## Phase 6: Notes and knowledge

CPU-only searchable notes, optional Qwen analysis and meeting imports.
See [Phase 6](../docs/ai-phase6-notes.md) for the API and deployment.

## Phase 7: Memory and evidence-grounded questions

CPU-only full-text retrieval across notes and completed meetings, with
optional Qwen answers and exact source citations. See
[Phase 7 Memory](../docs/ai-phase7-memory.md) for usage and safety limits.

## Phase 8: conversations, tasks and semantic memory

Tasks with transcript evidence, multi-turn memory and optional CPU embeddings.
See [Phase 8 Workflows](../docs/ai-phase8-workflows.md) for setup and limitations.
