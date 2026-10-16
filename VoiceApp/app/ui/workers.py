"""Background worker: runs conversions in a separate thread so the window never freezes."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.core.converter import ConversionResult, convert_file
from app.core.formats import OutputFormat

log = logging.getLogger(__name__)


@dataclass
class ConversionJob:
    row: int          # row in the file table, used to send updates back
    source: Path
    out_folder: Path


class ConversionWorker(QObject):
    """Converts jobs one after another.

    Files are converted one at a time: voice messages take well under a second
    each, and running ffmpeg sequentially keeps a low-power laptop responsive.
    """

    file_started = Signal(int)                 # row
    file_progress = Signal(int, float)         # row, 0.0-1.0
    file_finished = Signal(int, object)        # row, ConversionResult
    batch_finished = Signal(int, int, bool)    # succeeded, failed, cancelled

    def __init__(self, jobs: list[ConversionJob], fmt: OutputFormat, bitrate_kbps: int):
        super().__init__()
        self._jobs = jobs
        self._fmt = fmt
        self._bitrate = bitrate_kbps
        self._cancel = threading.Event()

    def cancel(self) -> None:
        # Called from the GUI thread; the Event is thread-safe.
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        ok = failed = 0
        for job in self._jobs:
            if self._cancel.is_set():
                break
            self.file_started.emit(job.row)
            try:
                result = convert_file(
                    job.source, self._fmt, self._bitrate, job.out_folder,
                    on_progress=lambda p, row=job.row: self.file_progress.emit(row, p),
                    cancel_event=self._cancel,
                )
            except Exception as exc:  # noqa: BLE001 - never let the thread die silently
                log.exception("Unexpected error converting %s", job.source)
                result = ConversionResult(False, message=f"Unexpected error: {exc}")
            if result.ok:
                ok += 1
            elif not result.cancelled:
                failed += 1
            self.file_finished.emit(job.row, result)
        self.batch_finished.emit(ok, failed, self._cancel.is_set())
