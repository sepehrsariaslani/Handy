"""Memory-bounded speaker diarization with conservative cross-window identity.

Speaker IDs describe acoustic clusters, not people's actual identities.
For long recordings each slice is processed independently; pyannote's
speaker embeddings are matched conservatively, never by local label alone.
"""
from pathlib import Path
import math
import tempfile
import threading
import wave


RATE = 16000
WINDOW_SECONDS = 900
OVERLAP_SECONDS = 10
# Require strong evidence before claiming the same speaker across windows.
COSINE_MIN = 0.78
COSINE_MARGIN = 0.07


def _raw_turns(output):
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = output.speaker_diarization
    if hasattr(annotation, "itertracks"):
        turns = [(float(span.start), float(span.end), str(speaker))
                 for span, _, speaker in annotation.itertracks(yield_label=True)]
    else:
        turns = [(float(span.start), float(span.end), str(speaker))
                 for span, speaker in annotation]
    return annotation, sorted(turns, key=lambda row: (row[0], row[1]))


def _normalized(vec):
    import numpy as np
    arr = np.asarray(vec, dtype="float32")
    norm = float(np.linalg.norm(arr))
    return arr / norm if math.isfinite(norm) and norm > 1e-8 else None


class _SpeakerRegistry:
    """Ephemeral per-meeting voice centroids. Never stores voiceprints on disk."""

    def __init__(self):
        self.vectors = {}
        self.next_id = 0

    def match(self, labels, vectors):
        candidates = []
        if vectors is not None:
            import numpy as np
            for label, vector in vectors.items():
                v = _normalized(vector)
                if v is None:
                    continue
                sims = sorted(
                    ((float(np.dot(v, known)), key)
                     for key, known in self.vectors.items()
                     if len(known) == len(v)),
                    reverse=True,
                )
                if sims:
                    best, target = sims[0]
                    second = sims[1][0] if len(sims) > 1 else -1.0
                    if best >= COSINE_MIN and best - second >= COSINE_MARGIN:
                        candidates.append((best, label, target))
        mapping = {}
        claimed = set()
        for _, label, target in sorted(candidates, reverse=True):
            if label not in mapping and target not in claimed:
                mapping[label] = target
                claimed.add(target)
        for label in labels:
            vector = _normalized(vectors[label]) if vectors is not None and label in vectors else None
            if label not in mapping:
                mapping[label] = f"SPEAKER_{self.next_id:02d}"
                self.next_id += 1
            key = mapping[label]
            if vector is not None:
                old = self.vectors.get(key)
                self.vectors[key] = _normalized(0.85 * old + 0.15 * vector) if old is not None else vector
        return mapping


def _embedding_map(output, annotation):
    embeddings = getattr(output, "speaker_embeddings", None)
    if embeddings is None or not hasattr(annotation, "labels"):
        return None
    labels = list(annotation.labels())
    if len(labels) != len(embeddings):
        return None
    return {str(label): vec for label, vec in zip(labels, embeddings)}


def _wav_slice(source: Path, start: int, end: int) -> Path:
    with wave.open(str(source), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("Diarization expects 16kHz mono PCM16 WAV")
        reader.setpos(start)
        audio = reader.readframes(end - start)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        path = Path(temp.name)
    try:
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(RATE)
            writer.writeframes(audio)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


class Diarizer:
    def __init__(self, enabled: bool = False, token: str | None = None,
                 max_hours: float = 2.0):
        self.enabled = enabled
        self.token = token
        self.max_hours = max_hours
        self.pipeline = None
        self.error: str | None = None
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def initialize(self):
        if not self.enabled:
            return
        try:
            from pyannote.audio import Pipeline
            import torch
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=self.token
            )
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            self.pipeline = pipeline
        except Exception as exc:
            self.error = type(exc).__name__
            self.pipeline = None

    def turns(self, path: Path, duration: float) -> list[tuple[float, float, str]]:
        if not self.ready:
            return []
        # No 20-hour waveform is ever handed to pyannote in one call.
        window = max(30.0, min(float(self.max_hours) * 3600, WINDOW_SECONDS))
        overlap = min(OVERLAP_SECONDS, window / 5)
        registry = _SpeakerRegistry()
        turns = []
        completed = 0.0
        with self.lock:
            while completed < duration:
                start = max(0.0, completed - overlap) if completed else 0.0
                end = min(duration, start + window)
                # Avoid a tiny final slice by closing exactly at the recording end.
                if end <= completed:
                    raise RuntimeError("Non-advancing diarization window")
                whole = start == 0 and end == duration
                sliced = None
                try:
                    sliced = path if whole else _wav_slice(
                        path, round(start * RATE), round(end * RATE)
                    )
                    output = self.pipeline(str(sliced))
                    annotation, local_turns = _raw_turns(output)
                    labels = sorted({speaker for _, _, speaker in local_turns})
                    mapping = registry.match(labels, _embedding_map(output, annotation))
                    for first, last, speaker in local_turns:
                        # Commit each region only once. Overlap is context, not output.
                        lo = max(completed, start + first)
                        hi = min(duration, end, start + last)
                        if hi > lo:
                            turns.append((round(lo, 3), round(hi, 3), mapping[speaker]))
                finally:
                    if sliced is not None and sliced != path:
                        sliced.unlink(missing_ok=True)
                completed = end
        return sorted(turns, key=lambda row: (row[0], row[1]))


def speaker_at(start: float, end: float, turns: list[tuple[float, float, str]]) -> str:
    scores = {}
    for first, last, speaker in turns:
        overlap = max(0.0, min(end, last) - max(start, first))
        if overlap:
            scores[speaker] = scores.get(speaker, 0.0) + overlap
    if not scores:
        return "Unknown"
    ordered = sorted(scores.items(), key=lambda row: row[1], reverse=True)
    # A large ASR window may contain multiple speakers. Do not invent certainty.
    if len(ordered) > 1 and (
        ordered[0][1] < 0.65 * sum(scores.values())
        or (end - start >= 8.0 and ordered[1][1] >= 1.0)
    ):
        # A 40-second block with ten seconds of another voice must not be
        # attributed entirely to the dominant speaker.
        return "Unknown"
    return ordered[0][0]


def label_segments(segments: list[dict], turns: list[tuple[float, float, str]]) -> list[tuple[str, str]]:
    return [(segment["id"], speaker_at(segment["start"], segment["end"], turns))
            for segment in segments]
