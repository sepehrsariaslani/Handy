# Phase 8 — Semantic memory, follow-up conversations and meeting tasks

Phase 8 extends the self-hosted Handy desktop app without requiring a hosted AI service or forcing GPU inference.

## Features

**Tasks** in the Handy sidebar:

- Create, edit, list, filter and delete tasks, set an assignee and a due date, mark To do / In progress / Done / Cancelled.
- From a completed meeting, press **Import meeting tasks**. This is an explicit action; the server never silently creates assignments.
- Imported tasks retain their original transcript utterance IDs and a preview of the cited speech. Tasks with missing or invented sources are skipped.
- Importing the same meeting again will not duplicate previously imported tasks. Updated tasks use optimistic versions to prevent accidental overwrites.
- The app does not yet assign tasks in Google Calendar or external TODO apps or send notifications.

**Multi-turn memory** in the Memory sidebar:

- Create/select/delete a stored conversation and continue it with follow-up questions, such as "What did Ali promise?" followed by "By what date?".
- The last few actual user questions help retrieve sources for ambiguous follow-ups. Bounded prior answers are passed to Qwen as background, but every **new** answer must cite currently retrieved source records.
- The current result and its evidence remain visible with the citation IDs, speaker, timecode and meeting/note date. Stored conversations persist in SQLite after a server restart.
- Optimistic turn numbers return HTTP 409 if two clients answer concurrently. Deleting a conversation erases its stored turn history, but not any original meetings/notes.

**Optional semantic retrieval**:

- Uses `intfloat/multilingual-e5-small`, a multilingual embedding model in `sentence-transformers`, with **CPU execution**. No GPU is required for creating or searching embeddings.
- Explicitly index at most **100 snippets per click** in Memory (the API accepts 1–200). Long archives can be processed in multiple batches.
- Embeddings live in SQLite. At query time, matching sources are **revalidated** against the original notes and completed transcripts, so previously deleted notes or unfinished recordings cannot be returned from stale vectors.
- Hybrid retrieval merges keyword (FTS5) and embedding candidates, and gracefully falls back to lexical search if the optional model is unavailable.
- The reference implementation scans up to 20,000 stored vectors in Python. It is for a private/small deployment, **not** yet an approximate-nearest-neighbour vector service for huge customer datasets.
- The embedding model will download on its **first explicitly requested indexing/search operation**, not at API startup. Pre-download the model into a trusted cache for offline deployments.

## Installation and environment

The original server runs without installing any new model. For optional embeddings, install:

```bash
cd server
python -m pip install -e '.[semantic]'
```

Then set:

```bash
HANDY_SEMANTIC_ENABLED=true
HANDY_SEMANTIC_MODEL=intfloat/multilingual-e5-small
```

Restart the single-worker Handy API and open Memory -> **Index next 100 snippets**. Repeat until the index reports no new documents. This indexing step is CPU-bound and should be scheduled outside peak usage for long archives. New transcripts and edited notes require incremental reindexing.

For notes, tasks, conversations and lexical memory *without* any AI model:

```bash
HANDY_ASR_ENABLED=false
HANDY_LLM_ENABLED=false
HANDY_SEMANTIC_ENABLED=false
```

Qwen3.5-9B is still optional and independently configured for natural-language answers; otherwise Handy displays retrieved excerpts only. No Qwen is required to create, search and update tasks.

## API

All requests use `Authorization: Bearer HANDY_API_KEY`.

- `GET /v1/tasks?status=todo`
- `POST /v1/tasks` (title, optional assignee/date)
- `GET /v1/tasks/{id}`
- `PUT /v1/tasks/{id}` (title, assignee, date, status, expected version)
- `DELETE /v1/tasks/{id}`
- `POST /v1/tasks/from-meeting/{job_id}`
- `GET /v1/memory/conversations`
- `POST /v1/memory/conversations` (title)
- `GET /v1/memory/conversations/{id}`
- `DELETE /v1/memory/conversations/{id}`
- `POST /v1/memory/conversations/{id}/turns` (question, scope, expected_turns)
- `GET /v1/memory/semantic/status`
- `POST /v1/memory/semantic/sync?limit=100`

Earlier `POST /v1/memory/search` and `POST /v1/memory/ask` still work with no session.

## Privacy and limitations

All phases currently use a **single tenant shared bearer key**. Do not give that key to unrelated users. Real public SaaS needs per-account authorization, partitioned databases and vector indexes, rate limits, usage metering, cancellation and retention/deletion policies. Cached AI responses and task source excerpts are sensitive. Use TLS, private network access, backups and access controls.

Embedding similarity is not proof of relevance. Qwen can still misread source language; exact citation validation verifies the source ID exists, not that its answer is factually correct. Review commitments, assignees and deadlines before acting. No automatic emails or notifications are sent. Semantic retrieval has not yet been benchmarked with 20-hour recordings and many concurrent clients.

## Verification

```bash
cd server && python -m pytest -q
cd .. && bun run lint && bun run format:check && bun run build
```

CI covers mocked Qwen and ASR, fake CPU embeddings, task import idempotence, concurrency conflicts, persistence, notes deleted after indexing and conversation follow-ups. A production CPU/GPU load test remains separate.
