"""Optional bounded WhisperX word alignment for diarized meeting transcripts.

No guessed word timestamps: if alignment is absent/incomplete, retain the
original ASR block and let the speaker remain Unknown when ambiguous.
"""
import logging
from pathlib import Path
import threading
import wave

from .diarization import RATE, speaker_at

LOG = logging.getLogger(__name__)
MAX_ALIGN_SECONDS = 45


def split_on_aligned_words(segment: dict, word_segments: list[dict], turns: list[tuple]) -> list[dict] | None:
    """Split only when *every* original whitespace token has a valid time.

    WhisperX returns its own tokenization. Require token-for-token equivalence
    to preserve every word, including the final sentence and punctuation.
    """
    tokens = segment["text"].split()
    words = [w for w in word_segments if (w.get("word") or "").strip()]
    if len(tokens) != len(words) or not tokens:
        return None

    import re
    normalize = lambda v: "".join(re.findall(r"\w+", v.casefold().replace("ي", "ی").replace("ك", "ک")))
    if any(normalize(t) != normalize(w["word"]) for t, w in zip(tokens, words)):
        return None
    start = float(segment["start"])
    end = float(segment["end"])
    marks = []
    previous = start
    for token, word in zip(tokens, words):
        a, b = word.get("start"), word.get("end")
        if a is None or b is None:
            return None
        a, b = float(a), float(b)
        if not (start - 0.02 <= a <= b <= end + 0.02) or a < previous - 0.05:
            return None
        previous = b
        marks.append((token, max(start, a), min(end, b), speaker_at(a, b, turns)))

    runs = []
    for token, a, b, speaker in marks:
        if runs and runs[-1]["speaker"] == speaker:
            runs[-1]["text"] += " " + token
            runs[-1]["end"] = b
        else:
            runs.append({"speaker": speaker, "start": a, "end": b, "text": token})
    if len(runs) <= 1:
        return None
    return [
        {"id": f'{segment["id"]}-w{i:03d}', "speaker": entry["speaker"],
         "start": round(entry["start"], 3), "end": round(entry["end"], 3),
         "text": entry["text"]}
        for i, entry in enumerate(runs, 1)
    ]


class WordAligner:
    def __init__(self, enabled: bool = False, device: str = "cpu"):
        self.enabled = enabled
        self.device = device
        self.models = {}
        self.lock = threading.Lock()
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self.enabled

    def split(self, path: Path, segment: dict, turns: list[tuple], language: str | None):
        duration = float(segment["end"]) - float(segment["start"])
        if not self.enabled or not 0.1 <= duration <= MAX_ALIGN_SECONDS:
            return None
        # Native long-form blocks (often 600s) are intentionally NOT assigned
        # fabricated word timings. They need timestamp-capable ASR in the future.
        code = (language or "fa").split("-")[0].lower()
        if code not in {"fa", "en"}:
            return None
        with self.lock:
            try:
                import numpy as np
                import whisperx
                if code not in self.models:
                    self.models[code] = whisperx.load_align_model(
                        language_code=code, device=self.device
                    )
                model, metadata = self.models[code]
                with wave.open(str(path), "rb") as reader:
                    reader.setpos(round(segment["start"] * RATE))
                    raw = reader.readframes(round(duration * RATE))
                waveform = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
                if not len(waveform):
                    return None
                aligned = whisperx.align(
                    [{"text": segment["text"], "start": 0.0,
                      "end": len(waveform) / RATE}],
                    model, metadata, waveform, self.device,
                    return_char_alignments=False,
                )
                words = [
                    {**word,
                     "start": word["start"] + segment["start"],
                     "end": word["end"] + segment["start"]}
                    if "start" in word and "end" in word else word
                    for word in aligned.get("word_segments", [])
                ]
                return split_on_aligned_words(segment, words, turns)
            except Exception as exc:
                # Transcription stays available even if a device/model fails.
                self.error = type(exc).__name__
                LOG.warning("meeting.word_alignment_unavailable reason=%s", self.error)
                return None
