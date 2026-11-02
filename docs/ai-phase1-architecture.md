# Handy AI architecture — phase 1

## Existing desktop pipeline (preserved)

Global shortcut / overlay → `AudioRecordingManager` (`src-tauri/src/managers/audio.rs`) → VAD and local recorder → `transcription_coordinator.rs` → `managers/transcription.rs` → local `transcribe-cpp` (Whisper/Parakeet) → post-processing `llm_client.rs` → clipboard/typing via `actions.rs` → history. The existing local dictation path is **not redirected** to the new server in phase 1.

The pre-existing Meeting Mode is in `src-tauri/src/commands/meeting.rs` and `src/components/meeting/MeetingMode.tsx`. It captures mono 16-kHz WAV straight to disk with periodic flushing, supports interrupted recording recovery, and has upload/poll hooks for `v1/meetings`. Phase 1 retains those existing behaviors and centralizes its non-secret server URL preference.

## Modes and boundaries

| Mode          | Purpose                                                        | Future worker                | Phase-1 behavior                                                    |
| ------------- | -------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------- |
| `dictation`   | Incremental text injection with tone control                   | Whisper + Qwen               | Session metadata contract                                           |
| `meeting`     | Speaker transcript, decisions, action items and evidence links | Whisper + diarization + Qwen | Existing local recorder; remote processing deliberately unavailable |
| `note_taking` | Lecture chapters, highlights, questions and citations          | Whisper + Qwen               | Session metadata contract                                           |

All three modes share `SessionCreate`, `SessionRecord`, `TranscriptSegment` (raw + optional cleaned, stable IDs, timestamps), and source-backed evidence. TS mirror: `src/lib/ai/contracts.ts`; Python source: `server/handy_server/schemas.py`. Contract version: `v1`.

## HTTP contract

- `GET /health`: liveness **only**. No auth; does not imply model readiness.
- `GET /v1/capabilities`: bearer auth; supported modes and *actual* worker readiness, initially all false.
- `POST /v1/sessions`: bearer auth; create session metadata, status `created`; supports mode, title, language, tone.
- `GET /v1/sessions/{id}`: bearer auth; load session metadata.
- `POST /v1/sessions/{id}/chunks`: **501** until phase 3, must eventually use sequence IDs, timestamps, idempotency keys, ACK/resume, backpressure, bounded audio buffers and overlap deduplication.
- `POST /v1/meetings`, `GET /v1/meetings/{id}`, `POST /v1/meetings/{id}/retry`: **501** until phase 5; reserved to match the existing Rust client contract.

Phase-1 persistence is SQLite metadata, not recordings. No model processes, queues or unsafe claims of completed inference. Do not store access tokens in URLs, browser storage or logs; only a non-secret server URL persists. HTTPS is mandatory for remote connections, with loopback HTTP for local development only.

## Follow-on integration design

1. Phase 2: GPU worker adapters, Whisper Large V3 & Qwen3.5-9B, error/retry and model readiness reporting.
2. Phase 3: idempotent streaming session/chunk protocol (5-s target, dynamic VAD) and rollback-safe partial/final text events. Keep old local dictation until the server path passes regression and latency tests.
3. Phase 5: change current Meeting Mode's whole-file upload to resumable chunked object storage; don't buffer 20h recordings in memory. Add diarization and evidence-backed analysis.
4. Phase 6: `note_taking` analysis, hierarchical long-form summaries and slide/notes exports.
5. Before public service: accounts/tenant isolation, authentication and authorization, rate limits, encrypted storage and data retention policies. The current single shared API key is for one trusted deployment, *not* a public multi-user service.

## Compatibility and baseline

Existing Tauri setup, model selector, shortcuts, transcription, paste and history files are intentionally untouched. API tests assert auth, CORS, all three modes, durable metadata and honest 501 behavior. The server has a reproducible HTTP/TestClient overhead benchmark in `server/benchmarks/measure_api.py`.

For a real ASR baseline, record machine/GPU, model, language, clip duration, VAD settings, first text latency, real-time factor, WER/CER and RAM/VRAM. **None of these measurements has been collected in phase 1**; do not report invented latency or accuracy numbers. Compare against the current offline desktop path before switching any default.
