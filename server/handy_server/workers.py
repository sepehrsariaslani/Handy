"""Inference adapters. Optional GPU dependencies load only when explicitly enabled."""
import asyncio
from pathlib import Path
import threading
from time import monotonic

import httpx

from .config import Settings, validate_llm_url
from .fidelity import protect_source
from .prompts import build_messages
from .schemas import CleanupRequest, CleanupResponse, TranscriptSegment, TranscriptionResponse


class WorkerUnavailable(RuntimeError):
    """A requested model is not ready."""


class WorkerFailure(RuntimeError):
    """A configured worker failed processing (no transcript should leak)."""


class WhisperWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def initialize(self) -> None:
        if not self.settings.asr_enabled:
            return
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                "large-v3",
                device=self.settings.asr_device,
                compute_type=self.settings.asr_compute_type,
            )
        except Exception as exc:
            self._load_error = type(exc).__name__
            self._model = None

    def transcribe_file(self, path: Path, language: str | None = None) -> tuple[list[TranscriptSegment], str | None]:
        if self._model is None:
            raise WorkerUnavailable("Whisper large-v3 is not loaded")
        try:
            # CTranslate2 model contexts are shared: one request at a time.
            with self._lock:
                segments, info = self._model.transcribe(
                    str(path),
                    language=language or None,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                # Iterating the generator must occur under the lock.
                entries = [
                    TranscriptSegment(
                        id=f"s{index:06d}",
                        start=float(seg.start),
                        end=float(seg.end),
                        speaker=None,
                        raw_text=seg.text.strip(),
                        clean_text=None,
                        final=True,
                    )
                    for index, seg in enumerate(segments, 1)
                    if seg.text.strip()
                ]
                return entries, getattr(info, "language", None)
        except Exception as exc:
            raise WorkerFailure(f"Whisper inference failed ({type(exc).__name__})") from None


class QwenWorker:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self._transport = transport
        self._ready_cached = False
        self._checked_at = 0.0

    async def ready(self) -> bool:
        if not self.settings.llm_enabled:
            return False
        if monotonic() - self._checked_at < 10:
            return self._ready_cached
        try:
            base = validate_llm_url(self.settings.qwen_base_url)
            async with httpx.AsyncClient(
                transport=self._transport, timeout=5.0, follow_redirects=False
            ) as client:
                response = await client.get(
                    base + "/models",
                    headers={"Authorization": f"Bearer {self.settings.qwen_api_key}"},
                )
                response.raise_for_status()
                models = response.json().get("data", [])
                ok = any(item.get("id") == self.settings.qwen_model for item in models)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError):
            ok = False
        self._ready_cached = ok
        self._checked_at = monotonic()
        return ok

    async def cleanup(self, body: CleanupRequest) -> CleanupResponse:
        if len(body.text) > self.settings.max_text_chars:
            raise ValueError("Text exceeds the configured length limit")
        if any(len(term) > 200 for term in body.protected_terms):
            raise ValueError("Protected term is too long")
        if not await self.ready():
            raise WorkerUnavailable("Qwen3.5-9B is not ready")
        payload = {
            "model": self.settings.qwen_model,
            "messages": build_messages(body.text, body.tone, body.language, body.protected_terms),
            "temperature": 0.1,
            "max_tokens": 4096,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        base = validate_llm_url(self.settings.qwen_base_url)
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=httpx.Timeout(90.0, connect=5.0), follow_redirects=False
            ) as client:
                response = await client.post(
                    base + "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.settings.qwen_api_key}"},
                )
                response.raise_for_status()
                edited = response.json()["choices"][0]["message"]["content"]
                if not isinstance(edited, str):
                    raise ValueError("Invalid model response")
        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
            raise WorkerFailure(f"Qwen inference failed ({type(exc).__name__})") from None
        warnings = protect_source(body.text, edited, body.protected_terms)
        # Reasoning must never appear in the displayed transcript.
        if "<think>" in edited or "</think>" in edited:
            warnings.append("Unexpected reasoning output; original text returned")
        return CleanupResponse(
            raw_text=body.text,
            clean_text=body.text if warnings else edited.strip(),
            tone=body.tone,
            model=self.settings.qwen_model,
            review_required=bool(warnings),
            warnings=warnings,
        )


class InferenceRuntime:
    def __init__(self, settings: Settings, asr: WhisperWorker | None = None, llm: QwenWorker | None = None):
        self.settings = settings
        self.asr = asr or WhisperWorker(settings)
        self.llm = llm or QwenWorker(settings)
        self._asr_queue = asyncio.Semaphore(1)
        self._llm_queue = asyncio.Semaphore(2)

    def initialize(self) -> None:
        self.asr.initialize()

    async def asr_ready(self) -> bool:
        return self.asr.ready

    async def llm_ready(self) -> bool:
        return await self.llm.ready()

    async def transcribe(self, path: Path, language: str | None = None) -> tuple[list[TranscriptSegment], str | None]:
        if not self.asr.ready:
            raise WorkerUnavailable("Whisper large-v3 is not ready")
        async with self._asr_queue:
            return await asyncio.to_thread(self.asr.transcribe_file, path, language)

    async def cleanup(self, body: CleanupRequest) -> CleanupResponse:
        async with self._llm_queue:
            return await self.llm.cleanup(body)
