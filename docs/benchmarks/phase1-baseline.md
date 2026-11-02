# Phase 1 baseline checklist (2026-11-02)

No measured hardware results are available from the repository-only review. Never label fabricated numbers as measurements.

| Area                                | Before phase 1            | After phase 1                                  |
| ----------------------------------- | ------------------------- | ---------------------------------------------- |
| Local dictation/VAD/typing pipeline | Existing Handy, unchanged | Must remain unchanged                          |
| Meeting recording/recovery          | Existing Tauri feature    | Must remain unchanged                          |
| Remote mode contracts               | No common contract        | `/v1/sessions`, TS/Python schemas              |
| Server readiness                    | No common health API      | `/health`, authenticated `/v1/capabilities`    |
| Server API request latency          | Not measured              | CI runs 100-request p50/p95 benchmark          |
| ASR WER/CER, RTF                    | Not measured              | Not measured — requires model and real audio   |
| 20-hour capture durability          | Not measured              | Not measured — requires desktop endurance test |

Run `cd server && pytest -q && python benchmarks/measure_api.py` to produce a real local API overhead baseline. CI publishes the machine-scoped JSON as `phase1-api-baseline`. These timing numbers measure FastAPI's in-process HTTP overhead, **not** network latency or inference. To capture ASR baseline use 10-30 minutes of consented test audio including Persian, mixed-language names, silence and long speech; compare transcription accuracy, memory, dropped audio frames and perceived typing latency on identical hardware.

Regression checklist before enabling any server processing: record/stop/cancel, push-to-talk/toggle, model switching, offline mode, clipboard paste, history, recording recovery and macOS/Windows permission prompts.
