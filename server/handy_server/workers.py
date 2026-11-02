"""Inference adapters. Optional GPU dependencies load only when explicitly enabled."""
import asyncio
import json
import os
import hashlib
from pathlib import Path
import subprocess
import tempfile
import wave
import logging
import threading
from urllib.parse import urlsplit
from time import monotonic

import httpx

from .config import Settings, validate_llm_url
from .fidelity import protect_source
from .prompts import build_messages
from .schemas import CleanupRequest, CleanupResponse, TranscriptSegment, TranscriptionResponse

logger = logging.getLogger(__name__)


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
                    # Keep context across Whisper's internal windows for connected speech.
                    condition_on_previous_text=True,
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


class GgufWhisperWorker:
    """Opt-in local GGUF Whisper adapter using transcribe.cpp's transcribe-cli.

    This reuses the same GGUF as the Handy desktop, without converting or
    downloading a CTranslate2 model. Only a local executable is launched;
    caller-supplied audio is passed as a filename, never through a shell.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ready = False
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        if not self.settings.asr_enabled:
            return
        model = self.settings.gguf_model_path
        binary = self.settings.transcribe_cli_path
        if (model is None or binary is None or not model.is_file() or
                not binary.is_file() or not os.access(binary, os.X_OK)):
            self._load_error = "MissingModelOrExecutable"
            logger.warning("GGUF Whisper model or executable is missing")
            return
        # Verify the model actually loads and transcribes a small local WAV
        # before advertising ASR readiness; no HF request is made.
        try:
            with tempfile.TemporaryDirectory(prefix="handy-gguf-probe-") as temp:
                silence = Path(temp) / "probe.wav"
                with wave.open(str(silence), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(16000)
                    writer.writeframes(b"\0" * 16000)  # 0.5 s silence
                self._run_cli(silence, None, timeout=180)
            self._ready = True
        except (WorkerFailure, OSError, ValueError) as exc:
            self._load_error = type(exc).__name__
            logger.warning("GGUF Whisper initialization failed (%s)", self._load_error)
            self._ready = False

    def selected_path(self, model_id: str | None) -> Path:
        """Resolve only a registered GGUF, never a caller-supplied filesystem path."""
        if not model_id:
            if self.settings.gguf_model_path is None:
                raise WorkerUnavailable("Default GGUF model is not configured")
            return self.settings.gguf_model_path
        # This identifier is a catalog ID, commonly
        # "handy-computer/repo-name/quantization.gguf". Never insert it into a
        # path directly. Rust registers the same SHA-256 filename on record.
        if not isinstance(model_id, str) or not 1 <= len(model_id) <= 512:
            raise WorkerUnavailable("Invalid selected ASR model ID")
        directory = self.settings.gguf_model_dir
        if directory is None:
            raise WorkerUnavailable("This server does not support local model switching")
        safe_name = "asr-" + hashlib.sha256(model_id.encode("utf-8")).hexdigest() + ".gguf"
        target = directory / safe_name
        if not target.is_file():
            raise WorkerUnavailable(
                f"Selected model '{model_id}' is not registered on the local server"
            )
        return target

    def _run_cli(self, audio_path: Path, language: str | None,
                 timeout: int = 300, model_id: str | None = None
                 ) -> tuple[list[TranscriptSegment], str | None]:
        selected = self.selected_path(model_id)
        # The registry filename is opaque; model family comes from catalog ID.
        is_qwen_asr = "qwen3-asr" in (model_id or selected.name).lower()
        with tempfile.TemporaryDirectory(prefix="handy-gguf-batch-") as temp:
            inputs = Path(temp) / "audio.list"
            inputs.write_text(str(audio_path.resolve()) + "\n", encoding="utf-8")
            command = [
                str(self.settings.transcribe_cli_path), "-q",
                "-m", str(selected),
                "--backend", self.settings.gguf_compute_backend,
                "--timestamps", "none" if is_qwen_asr else "segment",
                "--batch", str(inputs), "--batch-jsonl",
            ]
            # Qwen3-ASR detects language natively and has no forced-language
            # or timestamp capability. Whisper retains explicit language.
            if language and not is_qwen_asr:
                command.extend(["--language", language])
            logger.info("asr.cli.start model=%s family=%s requested_language=%s",
                        model_id or selected.name, "qwen3-asr" if is_qwen_asr else "gguf",
                        language or "auto")
            started = monotonic()
            try:
                proc = subprocess.run(
                    command, capture_output=True, text=True, check=False,
                    timeout=timeout, stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("GGUF CLI execution failed (%s)", type(exc).__name__)
                raise WorkerFailure("GGUF Whisper execution failed") from None
            if proc.returncode != 0:
                logger.warning("GGUF CLI exited with status %s model=%s elapsed_ms=%d",
                               proc.returncode, model_id or selected.name,
                               int((monotonic() - started) * 1000))
                raise WorkerFailure("Selected GGUF ASR model inference failed")
            logger.info("asr.cli.completed model=%s elapsed_ms=%d",
                        model_id or selected.name, int((monotonic() - started) * 1000))
            try:
                records = [json.loads(line) for line in proc.stdout.splitlines()
                           if line.strip()]
                results = [row for row in records if row.get("file") is not None]
                if len(results) != 1 or results[0].get("error"):
                    raise ValueError("Unexpected transcribe-cli result")
                result = results[0]
                text = result["text"]
                if not isinstance(text, str):
                    raise ValueError("Invalid transcript")
                rows = result.get("segments") or []
                if not isinstance(rows, list):
                    raise ValueError("Invalid timestamps")
                segments = [
                    TranscriptSegment(
                        id=f"s{index:06d}",
                        start=max(0.0, float(row["t0_ms"]) / 1000),
                        end=max(0.0, float(row["t1_ms"]) / 1000),
                        speaker=None,
                        raw_text=row["text"].strip(),
                        clean_text=None,
                        final=True,
                    )
                    for index, row in enumerate(rows, 1)
                    if isinstance(row.get("text"), str) and row["text"].strip()
                ]
                # If timestamps are not returned, retain the text rather than
                # silently losing the recording from the meeting pipeline.
                if not segments and text.strip():
                    with wave.open(str(audio_path), "rb") as reader:
                        duration = reader.getnframes() / reader.getframerate()
                    segments = [TranscriptSegment(
                        id="s000001", start=0, end=duration, speaker=None,
                        raw_text=text.strip(), clean_text=None, final=True,
                    )]
                reported = result.get("language")
                if not isinstance(reported, str):
                    # Qwen3-ASR performs auto-detection; do not misrepresent a
                    # user language preference as a model-detected language.
                    reported = None if is_qwen_asr else language
                return segments, reported
            except (ValueError, TypeError, KeyError, AttributeError, wave.Error,
                    json.JSONDecodeError):
                logger.warning("GGUF CLI returned an invalid result")
                raise WorkerFailure("Invalid GGUF Whisper result") from None

    def transcribe_file(self, path: Path, language: str | None = None,
                        model_id: str | None = None) -> tuple[list[TranscriptSegment], str | None]:
        if not self._ready:
            raise WorkerUnavailable("GGUF ASR is not ready")
        # Validate selection before starting an expensive subprocess.
        self.selected_path(model_id)
        # Each CLI call loads the memory-mapped model, so long meetings will
        # be slower than a persistent in-process backend on a small Mac.
        with self._lock:
            return self._run_cli(path, language, model_id=model_id)


class NativeHandyWorker(GgufWhisperWorker):
    """Use the exact TranscriptionManager behind desktop dictation.

    This calls Handy's dedicated, single-instance-safe --transcribe-file mode.
    It loads the recording's pinned model, runs the regular long-form, language
    and text normalization pipeline, then returns its finalized text. The
    Python service still checkpoints the audio and runs Qwen/diarization.
    Only the macOS local developer process opts into this backend.
    """

    def initialize(self) -> None:
        # Dev starts the API BEFORE cargo finishes compiling the native app.
        # Validate the actual executable at inference time instead.
        self._ready = bool(self.settings.asr_enabled and self.settings.native_handy_path)

    def _run_cli(self, audio_path: Path, language: str | None,
                 timeout: int = 300, model_id: str | None = None
                 ) -> tuple[list[TranscriptSegment], str | None]:
        if not model_id:
            # Pre-upgrade recordings have no pinned model. Preserve their
            # original backend, rather than silently changing historic runs.
            logger.info("asr.native.legacy_job using_original_gguf_backend")
            return super()._run_cli(audio_path, language, timeout, model_id=None)
        self.selected_path(model_id)
        binary = self.settings.native_handy_path
        if binary is None or not binary.is_file() or not os.access(binary, os.X_OK):
            raise WorkerUnavailable(
                "Handy native transcription is not compiled yet. "
                "Start the desktop with bun run dev:mac before processing."
            )
        if not model_id:
            raise WorkerUnavailable(
                "Native transcription needs a model pinned at recording start"
            )
        command = [
            str(binary), "--transcribe-file", str(audio_path),
            "--model", model_id, "--json",
        ]
        started = monotonic()
        logger.info(
            "asr.native.start model=%s language_source=desktop_settings requested_language=%s",
            model_id, language or "auto",
        )
        try:
            # Audio windows are bounded by MeetingProcessor, so this process
            # never loads a complete 20-hour recording into memory.
            with wave.open(str(audio_path), "rb") as reader:
                duration = reader.getnframes() / reader.getframerate()
            result = subprocess.run(
                command, capture_output=True, text=True, check=False,
                timeout=max(timeout, round(duration * 12)),
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired, wave.Error) as exc:
            logger.warning(
                "asr.native.failed model=%s error=%s", model_id, type(exc).__name__
            )
            raise WorkerFailure("Native Handy transcription could not run") from None
        if result.returncode != 0:
            # Never log stderr: model output or user data can appear there.
            logger.warning(
                "asr.native.failed model=%s exit_code=%s elapsed_ms=%d",
                model_id, result.returncode, int((monotonic() - started) * 1000),
            )
            raise WorkerFailure(
                "Native Handy transcription failed; check desktop model and permissions"
            )
        try:
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict) or payload.get("model") != model_id:
                raise ValueError("Wrong native model response")
            content = payload["text"]
            if not isinstance(content, str):
                raise ValueError("Invalid native transcript")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise WorkerFailure("Native Handy returned an invalid transcript") from None
        logger.info(
            "asr.native.completed model=%s characters=%d elapsed_ms=%d",
            model_id, len(content), int((monotonic() - started) * 1000),
        )
        if not content.strip():
            return [], None
        return [TranscriptSegment(
            id="s000001", start=0.0, end=duration, speaker=None,
            raw_text=content.strip(), clean_text=None, final=True,
        )], None


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
                
            if not ok:
                logger.warning("qwen.model_not_found provider=%s requested=%s available_count=%d",
                               self.settings.qwen_provider, self.settings.qwen_model, len(models))
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError) as exc:
            logger.warning("qwen.unavailable provider=%s error=%s", self.settings.qwen_provider, type(exc).__name__)
            ok = False
        self._ready_cached = ok
        self._checked_at = monotonic()
        return ok

    def _is_ollama(self) -> bool:
        if self.settings.qwen_provider != "auto":
            return self.settings.qwen_provider == "ollama"
        # The official Ollama localhost API uses port 11434; allow an explicit
        # override for non-standard ports via HANDY_QWEN_PROVIDER.
        return urlsplit(self.settings.qwen_base_url).port == 11434

    async def structured(self, messages: list[dict], *, max_tokens: int, operation: str) -> dict:
        """One provider-aware JSON endpoint for meeting and note analysis.

        Logging intentionally excludes audio, prompts, model output and credentials.
        """
        if not await self.ready():
            raise WorkerUnavailable("Qwen model is not ready; check provider and model ID")
        ollama = self._is_ollama()
        payload = {
            "model": self.settings.qwen_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if ollama:
            payload["reasoning_effort"] = "none"
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        started = monotonic()
        provider = "ollama" if ollama else "vllm"
        logger.info("qwen.start operation=%s provider=%s model=%s", operation, provider, self.settings.qwen_model)
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(240.0 if ollama else 120.0, connect=5.0),
                follow_redirects=False,
            ) as client:
                endpoint = validate_llm_url(self.settings.qwen_base_url) + "/chat/completions"
                headers = {"Authorization": "Bearer " + self.settings.qwen_api_key}
                response = await client.post(endpoint, headers=headers, json=payload)
                if ollama and response.status_code == 501:
                    # Some Ollama MLX distributions have no XGrammar grammar
                    # library: response_format=json_object returns 501 while
                    # regular chat still works. Use the same Qwen model without
                    # grammar constraints and validate its JSON locally.
                    logger.warning(
                        "qwen.structured_output_unavailable operation=%s provider=ollama "
                        "http_status=501 retry_without_response_format=true",
                        operation,
                    )
                    plain_payload = {key: value for key, value in payload.items()
                                     if key != "response_format"}
                    response = await client.post(
                        endpoint, headers=headers, json=plain_payload
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if not isinstance(content, str) or "<think>" in content or "</think>" in content:
                    raise ValueError("Invalid model output")
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError("Expected JSON object")
        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
            response = getattr(exc, "response", None)
            logger.warning(
                "qwen.failed operation=%s provider=%s error=%s http_status=%s elapsed_ms=%d",
                operation, provider, type(exc).__name__,
                getattr(response, "status_code", None), int((monotonic()-started)*1000),
            )
            raise WorkerFailure(f"Qwen {operation} failed ({type(exc).__name__})") from None
        logger.info(
            "qwen.completed operation=%s provider=%s elapsed_ms=%d",
            operation, provider, int((monotonic()-started)*1000),
        )
        return result

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
        }
        ollama = self._is_ollama()
        if ollama:
            # Ollama's OpenAI-compatible endpoint ignores vLLM's
            # chat_template_kwargs. Use its documented thinking control.
            payload["reasoning_effort"] = "none"
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        base = validate_llm_url(self.settings.qwen_base_url)
        started = monotonic()
        logger.info("qwen.start operation=cleanup provider=%s model=%s chars=%d",
                    "ollama" if ollama else "vllm", self.settings.qwen_model, len(body.text))
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=httpx.Timeout(240.0 if ollama else 90.0, connect=5.0), follow_redirects=False
            ) as client:
                response = await client.post(
                    base + "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.settings.qwen_api_key}"},
                )
                response.raise_for_status()
                edited = response.json()["choices"][0]["message"]["content"]
                if not isinstance(edited, str) or not edited.strip():
                    raise ValueError("Empty or invalid model response")
        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
            response = getattr(exc, "response", None)
            logger.warning(
                "Qwen inference failed (provider=%s, error=%s, status=%s)",
                "ollama" if ollama else "vllm",
                type(exc).__name__,
                getattr(response, "status_code", None),
            )
            raise WorkerFailure(f"Qwen inference failed ({type(exc).__name__})") from None
        logger.info("qwen.completed operation=cleanup provider=%s elapsed_ms=%d",
                    "ollama" if ollama else "vllm", int((monotonic()-started)*1000))
        warnings = protect_source(body.text, edited, body.protected_terms)
        if edited.lstrip().startswith("{") and edited.rstrip().endswith("}"):
            warnings.append("Unexpected JSON in edited transcript; original text returned")
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
        self.asr = asr or (NativeHandyWorker(settings) if settings.asr_backend == "native_handy"
                          else GgufWhisperWorker(settings) if settings.asr_backend == "gguf_cli"
                          else WhisperWorker(settings))
        self.llm = llm or QwenWorker(settings)
        self._asr_queue = asyncio.Semaphore(1)
        self._llm_queue = asyncio.Semaphore(2)

    def initialize(self) -> None:
        self.asr.initialize()

    async def asr_ready(self) -> bool:
        return self.asr.ready

    async def llm_ready(self) -> bool:
        return await self.llm.ready()

    def validate_asr_model(self, model_id: str | None) -> None:
        if model_id:
            if not isinstance(self.asr, GgufWhisperWorker):
                raise WorkerUnavailable("Selected GGUF models need the local transcribe.cpp backend")
            self.asr.selected_path(model_id)

    async def transcribe(self, path: Path, language: str | None = None,
                         model_id: str | None = None) -> tuple[list[TranscriptSegment], str | None]:
        if not self.asr.ready:
            raise WorkerUnavailable("Selected ASR model is not ready")
        self.validate_asr_model(model_id)
        async with self._asr_queue:
            if isinstance(self.asr, GgufWhisperWorker):
                return await asyncio.to_thread(self.asr.transcribe_file, path, language, model_id)
            return await asyncio.to_thread(self.asr.transcribe_file, path, language)

    async def cleanup(self, body: CleanupRequest) -> CleanupResponse:
        async with self._llm_queue:
            return await self.llm.cleanup(body)
