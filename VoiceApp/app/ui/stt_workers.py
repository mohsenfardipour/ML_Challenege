"""Background workers for speech-to-text, so the window never freezes.

The model is loaded once per batch and reused for every file.
Cancel is checked between parts of a file (every few seconds of audio).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.core.personal import PersonalStore, Personalization, Rule, apply_rules, suggest_rules
from app.core.persian_text import clean_transcript
from app.core.text_align import score
from app.core.transcriber import Cancelled, Transcriber, TranscriptResult

log = logging.getLogger(__name__)


def _load_model(model_dir: Path, threads: int, fallback: Path | None, status) -> tuple[Transcriber, str]:
    """Load the chosen model; if it fails and the base model exists, use that instead."""
    status(f"Loading model ({model_dir.name})… this takes 5–30 seconds")
    try:
        return Transcriber(model_dir, threads), ""
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not load model %s", model_dir)
        if not fallback:
            raise RuntimeError(f"Could not load the model in {model_dir}: {exc}") from exc
        status("The Persian model could not be loaded – using the base model instead…")
        return Transcriber(fallback, threads), (f"The model in {model_dir} could not be loaded ({exc}); "
                                                f"the base large-v3-turbo model was used instead.")


@dataclass
class TranscribeJob:
    row: int
    audio: Path


class TranscriptionWorker(QObject):
    status = Signal(str)
    file_started = Signal(int)
    file_progress = Signal(int, float, float)      # row, fraction, audio seconds done
    file_finished = Signal(int, object, str)       # row, TranscriptResult or None, error message
    batch_finished = Signal(int, int, bool, str)   # ok, failed, cancelled, warning

    def __init__(self, jobs: list[TranscribeJob], model_dir: Path, threads: int,
                 fallback_dir: Path | None, personal_root: Path | None, use_hints: bool = True):
        super().__init__()
        self.jobs, self.model_dir, self.threads = jobs, model_dir, threads
        self.fallback_dir, self.personal_root, self.use_hints = fallback_dir, personal_root, use_hints
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    @Slot()
    def run(self):
        ok = failed = 0
        warning = ""
        try:
            transcriber, warning = _load_model(self.model_dir, self.threads, self.fallback_dir, self.status.emit)
        except Exception as exc:  # noqa: BLE001
            for job in self.jobs:
                self.file_finished.emit(job.row, None, str(exc))
            self.batch_finished.emit(0, len(self.jobs), False, str(exc))
            return
        personalization = None
        if self.personal_root:
            personalization = PersonalStore(self.personal_root).personalization(transcriber.count_tokens)
            if not self.use_hints:
                personalization.hotwords = None
            log.info("Personalization: hotwords=%r, %d rules", personalization.hotwords, len(personalization.rules))
        self.status.emit("")
        for job in self.jobs:
            if self.cancel_event.is_set():
                break
            self.file_started.emit(job.row)
            try:
                result = transcriber.transcribe(
                    job.audio, personalization,
                    on_progress=lambda f, done, row=job.row: self.file_progress.emit(row, f, done),
                    cancel_event=self.cancel_event)
                ok += 1
                self.file_finished.emit(job.row, result, "")
            except Cancelled:
                self.file_finished.emit(job.row, None, "Cancelled")
            except Exception as exc:  # noqa: BLE001
                log.exception("Transcription failed for %s", job.audio)
                failed += 1
                self.file_finished.emit(job.row, None, _friendly(exc))
        self.batch_finished.emit(ok, failed, self.cancel_event.is_set(), warning)


def _friendly(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if "invalid data" in low or "error opening input" in low or "matches no streams" in low:
        return "The file is damaged or has no audio track."
    if "memory" in low:
        return "Not enough memory. Close other programs or use the fast model."
    return text.splitlines()[-1][:200] if text else exc.__class__.__name__


# ---------------------------------------------------------------- personalize tab
class SampleAnalysisWorker(QObject):
    """Transcribe each personal sample (no personalization) and cache the model output.
    The tab then suggests vocabulary and correction rules from the differences."""
    status = Signal(str)
    progress = Signal(int, int)        # done, total
    finished = Signal(bool, str)       # ok, message

    def __init__(self, store: PersonalStore, model_key: str, model_dir: Path, threads: int,
                 fallback_dir: Path | None, only_missing: bool = True):
        super().__init__()
        self.store, self.model_key, self.model_dir = store, model_key, model_dir
        self.threads, self.fallback_dir, self.only_missing = threads, fallback_dir, only_missing
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    @Slot()
    def run(self):
        samples = self.store.list_samples()
        todo = [s for s in samples if not (self.only_missing and s.model_output(self.model_key) is not None)]
        if not todo:
            self.finished.emit(True, "All samples were already analyzed with this model.")
            return
        try:
            transcriber, _ = _load_model(self.model_dir, self.threads, self.fallback_dir, self.status.emit)
            for i, sample in enumerate(todo):
                if self.cancel_event.is_set():
                    raise Cancelled()
                self.status.emit(f"Transcribing sample {i + 1} of {len(todo)}: {sample.source_name}")
                result = transcriber.transcribe(sample.audio, None, cancel_event=self.cancel_event)
                sample.save_model_output(self.model_key, result.raw_text)
                self.progress.emit(i + 1, len(todo))
        except Cancelled:
            self.finished.emit(False, "Cancelled.")
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Sample analysis failed")
            self.finished.emit(False, _friendly(exc))
            return
        self.finished.emit(True, f"Analyzed {len(todo)} sample(s).")


@dataclass
class EvalRow:
    model_key: str
    plain_wer: float
    plain_cer: float
    pers_wer: float
    pers_cer: float
    loo_wer: float | None       # rules learned only from the *other* samples (honest estimate)
    speed_ratio: float
    words: int


class EvaluationWorker(QObject):
    """For each model: transcribe every sample without and with personalization; report WER/CER."""
    status = Signal(str)
    progress = Signal(int, int)
    finished = Signal(bool, str, object)     # ok, message, list[EvalRow]

    def __init__(self, store: PersonalStore, models: list[tuple[str, Path]], threads: int, use_hints: bool = True):
        super().__init__()
        self.store, self.models, self.threads, self.use_hints = store, models, threads, use_hints
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    @Slot()
    def run(self):
        samples = self.store.list_samples()
        vocabulary = self.store.load_vocabulary()
        approved = [r for r in self.store.load_rules() if r.approved]
        manual = [r for r in approved if r.source == "manual"]
        rows: list[EvalRow] = []
        total_steps = len(self.models) * len(samples) * 2
        step = 0
        try:
            for key, model_dir in self.models:
                self.status.emit(f"Loading {key}…")
                transcriber = Transcriber(model_dir, self.threads)
                from app.core.personal import build_hotwords
                hints = build_hotwords(vocabulary, transcriber.count_tokens) if self.use_hints else None
                pers = Personalization(hints, approved)
                plain_texts, pers_texts, refs = [], [], []
                audio_s = proc_s = 0.0
                for sample in samples:
                    for mode in ("plain", "personalized"):
                        if self.cancel_event.is_set():
                            raise Cancelled()
                        self.status.emit(f"{key}: {sample.source_name} ({mode})")
                        t0 = time.perf_counter()
                        res: TranscriptResult = transcriber.transcribe(
                            sample.audio, pers if mode == "personalized" else None, cancel_event=self.cancel_event)
                        if mode == "plain":
                            plain_texts.append(res.raw_text)
                            sample.save_model_output(key, res.raw_text)
                            audio_s += res.audio_seconds
                            proc_s += time.perf_counter() - t0
                        else:
                            pers_texts.append(res.text)
                        step += 1
                        self.progress.emit(step, total_steps)
                    refs.append(sample.transcript)
                rows.append(self._score(key, refs, plain_texts, pers_texts, manual, proc_s / max(audio_s, 0.1)))
                del transcriber
        except Cancelled:
            self.finished.emit(False, "Cancelled.", rows)
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Evaluation failed")
            self.finished.emit(False, _friendly(exc), rows)
            return
        self.finished.emit(True, "Done.", rows)

    @staticmethod
    def _score(key, refs, plain, pers, manual_rules: list[Rule], speed) -> EvalRow:
        def corpus(hyps):
            joined_ref, joined_hyp = "\n".join(refs), "\n".join(hyps)
            return score(joined_ref, joined_hyp)

        p, q = corpus(plain), corpus(pers)
        loo = None
        if len(refs) >= 3:
            # Honest check: for each sample, use only rules learned from the other samples.
            loo_hyps = []
            for i in range(len(refs)):
                others = [(refs[j], plain[j]) for j in range(len(refs)) if j != i]
                rules = [r for r in suggest_rules(others) if r.safe] + manual_rules
                for r in rules:
                    r.approved = True
                loo_hyps.append(clean_transcript(apply_rules(plain[i], rules)))
            loo = corpus(loo_hyps).wer
        return EvalRow(key, p.wer, p.cer, q.wer, q.cer, loo, speed, p.ref_words)
