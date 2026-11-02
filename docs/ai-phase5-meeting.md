# Phase 5: long-form meeting transcription and grounded analysis

The existing Handy Meeting Mode recorder writes a 16 kHz mono 16-bit PCM WAV
directly to disk; it can preserve recordings lasting up to 20 hours. Phase 5
adds a real server-side processing pipeline, resumable uploads, optional
speaker diarization, cited summaries, decisions and action items.

## Operational flow

1. Get recording consent from all participants. Record in Meeting Mode.
2. Stop the recording and choose **Process**. Handy preserves the original WAV
   locally even if upload/processing fails.
3. The Rust client requests a resumable upload identified by the meeting's
   local ID. The server returns the previously committed byte offset.
4. The client PUTs the file in **4 MiB** chunks, then calls **finish**. When
   network transfer fails or is interrupted, repeating Process continues from
   the recorded byte offset. Requests and responses use bearer authentication.
5. The server validates WAV headers, channel count, sample rate and <=20-hour
   duration; it queues a SQLite-backed meeting job.
6. A single API worker processes one **30-second** window with Whisper Large
   V3 at a time. Each window's transcript, absolute timestamps and the next
   audio sample are committed together. After a crash, restarting the service
   resumes from its last committed checkpoint.
7. If optional pyannote Community-1 is installed, configured and loaded,
   speaker segments are assigned to utterances by temporal overlap. The
   pipeline processes the whole audio file and is **disabled for recordings
   longer than HANDY_DIARIZATION_MAX_HOURS** (default two hours) to avoid
   memory exhaustion. If unavailable, speakers are labelled **Unknown**,
   never invented. Speaker names remain editable in the UI.
8. Qwen3.5-9B generates evidence-linked summaries, decisions, tasks and open
   questions. The model processes batches and hierarchical summaries; invalid
   source IDs are removed. Model output still needs user review.
9. Handy polls job status, displays the current stage and processed time, and
   fetches long transcripts in pages (150 utterances). Clicking evidence
   sources retrieves their original utterance without fetching the whole file.

## GPU installation

Use the earlier Phase-2 server setup for Whisper Large V3 and Qwen3.5-9B,
including the server bearer secret. Qwen is needed for completed meeting
insights. For optional speaker diarization, install in the server's Python
environment:

\`\`\`bash
pip install -e '.[asr,diarization]'
\`\`\`

Accept the model license and obtain a Hugging Face access token for the
pyannote Community-1 model:

- https://huggingface.co/pyannote/speaker-diarization-community-1

Configure:

\`\`\`bash
export HANDY_DIARIZATION_ENABLED=true
export HANDY_HF_TOKEN=your_hf_token
export HANDY_DIARIZATION_MAX_HOURS=2
export HANDY_MAX_MEETING_SECONDS=72000
export HANDY_MAX_MEETING_BYTES=2400000000
\`\`\`

Pyannote 4 additionally requires FFmpeg available on the host. Keep Hugging
Face access tokens and Handy API keys outside source control, never expose
Qwen/vLLM to the public network.

**Run a single API process:** \`uvicorn handy_server.main:app --workers 1\`.
The meeting queue, upload locks and the existing streaming session locks are
single-process and are not safe to scale across multiple Uvicorn workers yet.
Do not treat a 20-hour audio upload as a single FastAPI request: use the
resumable API below. Reverse proxies must permit 4 MiB parts and long
requests for the direct legacy whole-file endpoint.

## HTTP API

All endpoints require \`Authorization: Bearer HANDY_API_KEY\`.

- \`POST /v1/meetings/uploads\`: JSON \`{"meeting_id":"client-stable-id","total_bytes":640044,"language":"fa"}\`.
  Returns \`id\`, \`offset\`, \`total_bytes\`, \`job_id\` (when finalized).
- \`GET /v1/meetings/uploads/{id}\`: durable current upload offset.
- \`PUT /v1/meetings/uploads/{id}\`: binary body up to 4 MiB with
  \`X-Upload-Offset\`; returns \`{"offset":next_offset}\`. Error 409 includes
  the authoritative offset if a request is stale.
- \`POST /v1/meetings/uploads/{id}/finish\`: validates complete WAV and creates
  an idempotent asynchronous meeting job. Repeating finish returns the same job.
- \`POST /v1/meetings\`: legacy authenticated whole-file multipart upload
  remains supported (nonresumable).
- \`GET /v1/meetings/{job_id}\`: status, stage, processed seconds, segment count,
  first 150 utterances and available insights.
- \`GET /v1/meetings/{job_id}/transcript?offset=0&limit=150\`: paginated
  utterances with stable IDs and timestamps.
- \`POST /v1/meetings/{job_id}/retry\`: requeues only failed jobs, preserving
  already committed Whisper transcription windows.

## Security and limitations

- **Single tenant**: the bearer key grants access to all meetings and should
  never be shared between unrelated end users. Production multi-user service
  needs per-user authorization, quotas and data isolation.
- The server retains uploaded audio, transcripts and insights on disk; a
  configurable deletion/retention service has not been added. Limit filesystem
  access, encrypt storage where appropriate and delete recordings only under
  a defined retention policy.
- 20-hour recordings require roughly 2.3 GB for one PCM16 16 kHz WAV, plus
  SQLite data, upload spool and temporary inference windows. Ensure sufficient
  free disk and filesystem file-size limits. No 20-hour hardware endurance
  benchmark has yet been run.
- Speaker diarization for >2h is **not** guaranteed: the configurable default
  skips the pyannote whole-file pipeline. Genuine identity persistence over
  arbitrarily long recordings needs a scalable diarization architecture.
- LLM extraction is inherently probabilistic. Evidence IDs validate that a
  source exists, not that a paraphrase is true. Review owners, dates,
  decisions, and negations before publishing/exporting.
- Jobs are durable in SQLite but processing is performed by the single
  Uvicorn process; use a dedicated worker queue and distributed locking
  before deploying across multiple instances.
- This phase does not create external TODO tasks or mobile background recording.
  Meeting Mode remains a separate desktop feature.

## Verification

\`\`\`bash
cd server
pip install -e '.[dev]'
pytest -q
\`\`\`

CI exercises mocked ASR/LLM, uploads and crash recovery. Real GPU validation
requires a consented test recording with multiple speakers in Persian/English,
then checking ASR accuracy, diarization errors, timestamp alignment,
evidence grounding, processing time, and recovery after restart.
