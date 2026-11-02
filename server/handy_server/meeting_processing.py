"""Bounded-memory meeting transcription pipeline, safe restart, and background worker."""
import asyncio
from pathlib import Path
import tempfile
import wave

from .diarization import Diarizer, label_segments
from .meeting_store import MeetingStore
from .meeting_analysis import QwenMeetingAnalyzer
from .workers import WorkerUnavailable, WorkerFailure

RATE = 16000
WINDOW_SECONDS = 30

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
                    await asyncio.to_thread(self.store.fail, job["id"],
                                            f"{type(exc).__name__}: {str(exc)[:120]}")
                except Exception as exc:
                    await asyncio.to_thread(self.store.fail, job["id"],
                                            "Processing failed (" + type(exc).__name__ + ")")
            else:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

    async def process(self, job: dict):
        job_id = job["id"]
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
            try:
                segments, language = await self.runtime.transcribe(window, job["language"])
            finally:
                window.unlink(missing_ok=True)
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
            turns = await asyncio.to_thread(self.diarizer.turns, audio_path, duration)
            if turns:
                offset = 0
                while True:
                    page = await asyncio.to_thread(self.store.segments, job_id, offset, 500)
                    if not page:
                        break
                    names = label_segments(page, turns)
                    await asyncio.to_thread(self.store.update_speakers, job_id, names)
                    offset += len(page)

        await asyncio.to_thread(self.store.stage, job_id, "summarizing")
        utterances = []
        offset = 0
        while True:
            page = await asyncio.to_thread(self.store.segments, job_id, offset, 500)
            if not page:
                break
            utterances.extend(page)
            offset += len(page)
        insights = await self.analyzer.analyze(utterances)
        await asyncio.to_thread(self.store.complete, job_id, insights)
