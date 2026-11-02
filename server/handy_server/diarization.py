"""Optional pyannote Community-1 speaker diarization, never fabricated."""
from pathlib import Path
import threading

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
        # pyannote loads audio for end-to-end diarization. Bound its input to
        # prevent 20-hour OOM; transcription still proceeds for long meetings.
        if duration > self.max_hours * 3600:
            return []
        with self.lock:
            output = self.pipeline(str(path))
            annotation = getattr(output, "exclusive_speaker_diarization", None)
            if annotation is None:
                annotation = output.speaker_diarization
            if hasattr(annotation, "itertracks"):
                result = [(float(span.start), float(span.end), str(speaker))
                          for span, _, speaker in annotation.itertracks(yield_label=True)]
            else:
                result = [(float(span.start), float(span.end), str(speaker))
                          for span, speaker in annotation]
        return result


def label_segments(segments: list[dict], turns: list[tuple[float, float, str]]) -> list[tuple[str, str]]:
    output = []
    for segment in segments:
        scores: dict[str, float] = {}
        for start, end, speaker in turns:
            overlap = max(0.0, min(segment["end"], end) - max(segment["start"], start))
            if overlap > 0:
                scores[speaker] = scores.get(speaker, 0.0) + overlap
        best = max(scores, key=scores.get) if scores else "Unknown"
        output.append((segment["id"], best))
    return output
