# Phase 2 architecture and acceptance criteria

## Included

- Whisper Large V3 as the self-hosted non-streaming server ASR, loaded once
  at application startup when explicitly enabled.
- Qwen3.5-9B behind an OpenAI-compatible vLLM endpoint with explicit
  `enable_thinking=false` and model readiness validation.
- Authenticated text cleanup and bounded audio file transcription, with
  separate raw/clean text and timed segments.
- Persian, English and code-switched editing prompts and five user tones.
- Fallback to raw text when the Qwen result changes digits, drops a protected
  term, returns empty text or leaks reasoning markup.
- GPU-free tests, model-status monitoring, a real-model smoke test script,
  and a shared TypeScript API client.

## Later phases, not falsely claimed as ready

- Phase 3: five-second live audio chunks, WebSocket and stable partial text,
  with server transcription wired to the main global dictation route.
- Phase 5: long meeting upload, diarization, meeting summaries and tasks.
- Phase 6: lecture note-taking, lesson organization and exports.
- Actual server deployment, GPU benchmarks and real Persian WER require
  hardware and recordings unavailable to GitHub Actions.

## Before declaring production readiness

1. Start API without models: check status reports false and routes return 503.
2. Install ASR and Qwen on an NVIDIA GPU host; confirm status reports true.
3. Test real consented audio in Persian, English and mixed-language samples.
4. Measure transcript accuracy, latency, real-time factor and GPU VRAM.
5. Compare raw and edited text for negation, dates, numbers, quantities,
   business terms and meaningful repetitions.
6. Verify invalid file type and too-large uploads fail safely.
7. Verify interruptions and Qwen outages return an error without losing raw
   audio in the desktop app.
8. Confirm original local Handy dictation still records and pastes correctly.

Passing mocked API tests is not the same as installing or validating GPU
models in a live environment.
