# Phase 7 — Grounded memory and questions about past meetings

This phase adds **Memory** to Handy: search your previous Meeting Mode transcripts
and saved Notes, or ask a natural-language question about them with Qwen. Every
answer includes exact identifiers of the original sources; if the model cannot
return citations to retrieved evidence, the answer is withheld.

## User workflow

1. Open **Memory** in the Handy sidebar.
2. Enter the private server URL and bearer API key; the key is in UI memory
   only, not browser storage.
3. Ask a question in Persian or English, e.g.:
   *"علی در جلسه هفته گذشته چه کاری را قبول کرده بود؟"*
4. Optionally choose meetings, notes, or both. Optional UTC creation-date
   filters override any relative dates in the question.
5. Press **Ask with Qwen** to retrieve relevant snippets and generate a
   source-cited answer. **Find sources without AI** searches using CPU alone.
6. Read the cited snippets, including meeting date, speaker and timestamp.
   The UI can jump to the exact retrieved evidence record.

Recognized date phrases (interpreted relative to the **server's UTC date**):
- Persian: "هفته گذشته", "هفته قبل", "این هفته", "دیروز", "امروز".
- English: "last week", "this week", "yesterday", "today".
- Weeks mean Monday through Sunday, and the period used is displayed to
  the user. Other relative expressions do not receive automatic filters.

## API and index

All endpoints require `Authorization: Bearer HANDY_API_KEY`.

- `POST /v1/memory/search`: CPU-only lexical FTS5 search.
- `POST /v1/memory/ask`: the same retrieval, followed by optional private Qwen.

JSON request:

```json
{
  "question": "علی هفته گذشته چه کاری را قبول کرد؟",
  "scope": "all",
  "since": null,
  "until": null
}
```

The result contains `status`, `answer`, `citations`, `sources`,
`applied_since`, `applied_until`, and optional `warning`. Status is
`answered`, `search_only` or `not_found`.

Notes are searched with the existing `notes_fts` index. The server creates
`meeting_memory_fts` for transcription segments, backfills old completed
recordings, and uses SQLite triggers to keep the index updated as new
transcriptions arrive or speakers change. Source IDs use stable patterns:
- `meeting:{job_id}:{utterance_id}`
- `note:{note_id}:{source_id}`

Source snippets are sent to Qwen as *untrusted data* with a bounded context.
The response must be a JSON object with an answer and exact source IDs.
Unknown/hallucinated IDs are discarded. A response with no valid citations
is **not displayed as an answer**. The raw meeting audio is never sent to
Qwen by this feature.

## No GPU for search

The source search is SQLite FTS5 and needs no GPU or embeddings. Qwen is
optional: if unavailable, the server returns the matching source snippets
as `search_only`. Running Qwen on CPU is possible using a compatible
quantized inference endpoint, but response speed depends on model size and
hardware. Whisper, Qwen and pyannote remain independently configured.

## Known limits and safety

- This is **lexical retrieval**, not semantic vector search: paraphrases,
  homonyms, Persian spelling variants, and indirect references can be missed.
  The OR-term search may also retrieve less relevant sources.
- A citation confirms the excerpt exists; it **does not automatically verify
  the semantic truth** of every claim Qwen makes. Users must check important
  commitments, due dates and names against the originals.
- Meeting dates are the session creation date, not the exact timestamp of a
  midnight-spanning utterance. Relative periods use the server's UTC date,
  not an inferred client timezone.
- No persistent chat history or cross-turn conversation context yet. Each
  query stands alone, so restate references in follow-up questions.
- This remains a **single-tenant service**: one bearer key can read all
  transcripts and notes. For a consumer product, multi-user isolation,
  consent, token management, rate limits, encrypted storage, deletion/retention
  and auditing must be implemented before deployment to unrelated customers.
- Search reads local server records and makes **no** third-party request
  unless the user chooses AI answering with their configured private Qwen.

## Tests

```bash
cd server && python -m pytest -q
bun run lint
bun run format:check
bun run build
```

Automated tests cover stored notes and transcripts, citations, invalid
model responses, model-unavailable fallback, FTS updates, UTC date ranges
and restart backfill. Real hardware, concurrent users and GPU performance
are not verified by these tests.
