"""Bounded-memory meeting transcription pipeline, safe restart, and background worker."""
import asyncio
import logging
from time import monotonic
from pathlib import Path
import tempfile
import wave

from .diarization import Diarizer, label_segments
from .meeting_store import MeetingStore
from .meeting_analysis import QwenMeetingAnalyzer, meeting_text_parts
from .workers import WorkerUnavailable, WorkerFailure
from .schemas import CleanupRequest, Tone, MeetingAnalysis

LOG = logging.getLogger(__name__)
RATE = 16000
WINDOW_SECONDS = 120  # Match the normal full-file ASR context more closely.

def validated_duration(path: Path, max_seconds: int = 20 * 3600) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                    wav.getcomptype()) != (1, 2, RATE, "NONE"):
                raise ValueError("Meeting WAV must be mono 16kHz PCM16")
            duration = wav.getnframes() / RATE
            if duration <= 0 or duration > max_seconds:
                raise ValueError("Meeting length must be between 0 and 20 hours")
            if path.stat().st_size < wav.getnframes() * 2 + 44:
                raise ValueError("Truncated meeting WAV")
            return duration
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid meeting WAV") from exc

def wav_window(path: Path, start_frame: int, frames: int) -> tuple[Path, int]:
    with wave.open(str(path), "rb") as reader:
        reader.setpos(start_frame)
        samples = reader.readframes(frames)
    if not samples:
        raise ValueError("Missing audio at checkpoint")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        destination = Path(tmp.name)
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setframerate(RATE)
        writer.setsampwidth(2)
        writer.writeframes(samples)
    return destination, len(samples) // 2

class MeetingProcessor:
    def __init__(self, store: MeetingStore, runtime, diarizer: Diarizer, analyzer=None):
        self.store = store
        self.runtime = runtime
        self.diarizer = diarizer
        self.analyzer = analyzer or QwenMeetingAnalyzer(runtime.settings, runtime.llm)
        self._stop = asyncio.Event()

    def stop(self):
        self._stop.set()

    async def serve(self):
        # One worker per API process. Set uvicorn --workers 1 for this release.
        while not self._stop.is_set():
            job = await asyncio.to_thread(self.store.claim)
            if job:
                try:
                    await self.process(job)
                except (WorkerUnavailable, WorkerFailure, ValueError, OSError) as exc:
                    LOG.warning("meeting.failed job=%s error_type=%s", job["id"], type(exc).__name__)
                    await asyncio.to_thread(self.store.fail, job["id"],
                                            f"{type(exc).__name__}: {str(exc)[:120]}")
                except Exception as exc:
                    LOG.exception("meeting.unexpected_failure job=%s", job["id"])
                    await asyncio.to_thread(self.store.fail, job["id"],
                                            "Processing failed (" + type(exc).__name__ + ")")
            else:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

    async def process(self, job: dict):
        job_id = job["id"]
        started = monotonic()
        LOG.info("meeting.started job=%s kind=%s duration_seconds=%.1f", job_id, job.get("kind", "meeting"), job["duration"])
        audio_path = Path(job["audio_path"])
        if not await self.runtime.asr_ready():
            raise WorkerUnavailable("Whisper Large V3 not loaded")
        duration = await asyncio.to_thread(validated_duration, audio_path)
        limit_frames = round(duration * RATE)
        position = int(job["next_sample"])
        if position > limit_frames or position < 0:
            raise ValueError("Invalid persisted transcription checkpoint")
        counter = await asyncio.to_thread(self.store.count, job_id)
        while position < limit_frames:
            window, actual = await asyncio.to_thread(
                wav_window, audio_path, position, min(RATE * WINDOW_SECONDS, limit_frames-position)
            )
            window_started = monotonic()
            LOG.info(
                "meeting.asr.start job=%s offset_seconds=%.1f duration_seconds=%.1f requested_language=%s backend=%s model=%s",
                job_id, position / RATE, actual / RATE, job["language"] or "auto",
                self.runtime.settings.asr_backend, job.get("asr_model_id") or "default",
            )
            try:
                segments, language = await self.runtime.transcribe(
                    window, job["language"], job.get("asr_model_id")
                )
            finally:
                window.unlink(missing_ok=True)
            LOG.info(
                "meeting.asr.done job=%s segments=%d model_reported_language=%s elapsed_ms=%d",
                job_id, len(segments), language or "unknown",
                int((monotonic() - window_started) * 1000),
            )
            start_offset = position / RATE
            utterances = []
            for segment in segments:
                if not segment.raw_text.strip():
                    continue
                counter += 1
                utterances.append({
                    "id": f"u{counter:08d}",
                    "speaker": "Unknown",
                    "start": round(start_offset + segment.start, 3),
                    "end": round(min(duration, start_offset + segment.end), 3),
                    "text": segment.raw_text.strip(),
                })
            position += actual
            await asyncio.to_thread(self.store.checkpoint, job_id, position, utterances)
            if self._stop.is_set():
                return

        if self.diarizer.ready:
            await asyncio.to_thread(self.store.stage, job_id, "diarizing")
            # The pyannote pipeline is intentionally not run on 20-hour files:
            # it loads full audio and could exhaust GPU or system memory.
            try:
                turns = await asyncio.to_thread(self.diarizer.turns, audio_path, duration)
            except Exception as exc:
                # ASR and meeting insights must survive a diarizer outage.
                # Do not misattribute an unrecognized speaker.
                LOG.warning("Diarization unavailable (%s); speaker labels remain Unknown",
                            type(exc).__name__)
                turns = []
            if turns:
                offset = 0
                while True:
                    page = await asyncio.to_thread(self.store.segments, job_id, offset, 500)
                    if not page:
                        break
                    names = label_segments(page, turns)
                    await asyncio.to_thread(self.store.update_speakers, job_id, names)
                    offset += len(page)

        utterances = []
        offset = 0
        while True:
            page = await asyncio.to_thread(self.store.segments, job_id, offset, 500)
            if not page:
                break
            utterances.extend(page)
            offset += len(page)

        # Use the SAME Whisper runtime as /v1/transcriptions, then Qwen's
        # shared text cleanup after Stop. Never use low-context live chunks.
        await asyncio.to_thread(self.store.stage, job_id, "polishing")
        chunks = []
        current = []
        length = 0
        cap = min(5500, max(300, self.runtime.settings.max_text_chars))
        for row in utterances:
            # Native Handy returns one utterance per 120s window. A single
            # utterance may exceed Qwen's cleanup context budget, so split
            # *within* that text rather than silently dropping its ending.
            for line in meeting_text_parts(row["text"], max_chars=cap - 1):
                if current and length + len(line) + 1 > cap:
                    chunks.append("\n".join(current))
                    current = []
                    length = 0
                current.append(line)
                length += len(line) + 1
        if current:
            chunks.append("\n".join(current))

        polished = []
        can_clean = await self.runtime.llm_ready()
        LOG.info("meeting.polishing job=%s blocks=%d qwen_ready=%s", job_id, len(chunks), can_clean)
        for index, text in enumerate(chunks):
            if can_clean:
                try:
                    edited = await self.runtime.cleanup(CleanupRequest(
                        text=text,
                        tone=Tone.professional,
                        language=job.get("language") or (
                            "fa" if sum("\u0600" <= ch <= "\u06ff" for ch in text) > len(text) // 6 else None
                        ),
                    ))
                    polished.append(edited.clean_text)
                    LOG.info(
                        "meeting.cleanup job=%s block=%d changed=%s review_required=%s warnings=%d",
                        job_id, index, edited.clean_text != text,
                        edited.review_required, len(edited.warnings),
                    )
                except (WorkerFailure, WorkerUnavailable, ValueError) as exc:
                    LOG.warning("meeting.cleanup_fallback job=%s block=%d error=%s",
                                job_id, index, type(exc).__name__)
                    polished.append(text)
            else:
                polished.append(text)

        # Keep the full polished text available even if downstream insights
        # fail (e.g. Ollama out-of-memory or an invalid JSON response).
        await asyncio.to_thread(self.store.save_cleaned, job_id, polished)

        if job.get("kind", "meeting") == "note":
            # Notes are saved automatically, including after closing the UI.
            # A missing Qwen service must not lose the Whisper transcript.
            insights = None
        else:
            await asyncio.to_thread(self.store.stage, job_id, "summarizing")
            insights = await self.analyzer.analyze(utterances)
        await asyncio.to_thread(self.store.complete, job_id, insights, polished)
        LOG.info("meeting.completed job=%s segments=%d polished_blocks=%d elapsed_ms=%d",
                 job_id, len(utterances), len(polished), int((monotonic()-started)*1000))
