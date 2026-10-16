"""Speech-to-text with faster-whisper on the CPU (int8). No network access.

Used by the Transcription tab, the Personalize tab, and the model pipeline's
evaluation script, so all three decode exactly the same way.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core import ffmpeg
from app.core.persian_text import clean_transcript
from app.core.personal import Personalization
from app.core.stt_models import model_options

log = logging.getLogger(__name__)

# Never let Hugging Face libraries phone home (models are loaded from local folders only).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_CPU_THREADS = 4

DECODE_OPTIONS = dict(
    language="fa",                       # always Persian, no auto-detection
    task="transcribe",
    vad_filter=True,                     # skip silence
    condition_on_previous_text=False,    # avoids repeated / skipped sentences on long audio
    beam_size=5,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),   # fallback when a window decodes badly
)


class Cancelled(Exception):
    pass


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    segments: list[Segment]
    raw_text: str              # model output, only basic clean-up
    text: str                  # after personalization rules (if any)
    audio_seconds: float
    processing_seconds: float
    personalized: bool = False
    warnings: list[str] = field(default_factory=list)


class Transcriber:
    """Holds one loaded model. Load once, transcribe many files."""

    def __init__(self, model_dir: str | Path, cpu_threads: int = DEFAULT_CPU_THREADS):
        from faster_whisper import WhisperModel   # imported late: slow import, not needed for converting

        self.model_dir = Path(model_dir)
        self.options = model_options(self.model_dir)
        t0 = time.perf_counter()
        self.model = WhisperModel(str(self.model_dir), device="cpu", compute_type="int8",
                                  cpu_threads=max(1, int(cpu_threads)), local_files_only=True)
        self.load_seconds = time.perf_counter() - t0
        log.info("Loaded model %s in %.1fs (cpu_threads=%s)", self.model_dir, self.load_seconds, cpu_threads)

    def count_tokens(self, text: str) -> int:
        return len(self.model.hf_tokenizer.encode(" " + text, add_special_tokens=False).ids)

    def transcribe(self, audio_path: str | Path,
                   personalization: Personalization | None = None,
                   on_progress: Callable[[float, float], None] | None = None,
                   cancel_event: threading.Event | None = None) -> TranscriptResult:
        """on_progress(fraction 0-1, audio seconds done). Cancel is checked between segments."""
        t0 = time.perf_counter()
        samples = ffmpeg.decode_to_pcm16k(str(audio_path))
        duration = len(samples) / 16000
        options = dict(DECODE_OPTIONS)
        if self.options.get("without_timestamps"):
            options["without_timestamps"] = True
        if personalization and personalization.hotwords:
            options["hotwords"] = personalization.hotwords
        segments_iter, _info = self.model.transcribe(samples, **options)
        segments: list[Segment] = []
        for seg in segments_iter:          # the actual work happens while iterating
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            text = seg.text.strip()
            if text:
                segments.append(Segment(seg.start, seg.end, clean_transcript(text)))
            if on_progress and duration:
                on_progress(min(1.0, seg.end / duration), min(seg.end, duration))
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        raw_text = "\n".join(s.text for s in segments)
        text = personalization.apply(raw_text) if personalization else raw_text
        if personalization and personalization.rules:
            for s in segments:
                s.text = personalization.apply(s.text)
        if on_progress:
            on_progress(1.0, duration)
        result = TranscriptResult(segments, raw_text, text, duration, time.perf_counter() - t0,
                                  personalized=personalization is not None)
        if duration > 5 and not segments:
            result.warnings.append("No speech was recognized.")
        return result
