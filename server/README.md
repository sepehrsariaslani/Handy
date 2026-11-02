# Handy AI Server — Phase 2

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
