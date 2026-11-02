# Handy AI Server — Phase 1

This is a **functional, authenticated API foundation**, not yet an ASR/LLM server. The three session modes are defined and persisted in SQLite; inference/upload/streaming endpoints explicitly return HTTP 501 until their workers are implemented. Nothing is silently sent to a cloud provider.

## Development

Python 3.11+:

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export HANDY_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export HANDY_DB_PATH=./data/handy.sqlite3
uvicorn handy_server.main:app --host 127.0.0.1 --port 8765
```

For Windows use `.venv\Scripts\activate`. Connect the existing Handy Meeting Mode server field to `http://127.0.0.1:8765` and provide the same API token. Remote hosts **must** use HTTPS (TLS proxy); never publish Uvicorn's plain HTTP port to the internet.

Check readiness and API contract:

```bash
curl http://127.0.0.1:8765/health
curl -H "Authorization: Bearer $HANDY_API_KEY" http://127.0.0.1:8765/v1/capabilities
```

`/health` only checks HTTP availability. `/v1/capabilities.processing` is the source of truth for readiness; all workers are false in phase 1.

Run tests and synthetic API overhead measurement:

```bash
pytest -q
python benchmarks/measure_api.py
```

### Security and operational boundaries

- Required API token, constant-time comparison, no wildcard CORS; `HANDY_CORS_ORIGINS` must explicitly include the production web frontend origin.
- Token in the desktop Meeting UI is volatile/in-memory; only non-secret server URL is saved locally.
- SQLite stores **session metadata only**. No recording, transcript or task is uploaded/persisted by this server yet.
- This is a single-tenant prototype. Before multi-user/public deployment, add per-user identities, authorization, quotas, durable job workers, encrypted object storage, retention and deletion policies.
- Do not send multi-hour recordings to the phase-1 stub; server endpoints deliberately return 501.

See `docs/ai-phase1-architecture.md` for the shared versioned contract and migration milestones.
