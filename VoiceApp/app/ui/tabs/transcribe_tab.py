"""The "Transcribe" tab: Persian speech-to-text, fully offline."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication, QTextOption
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.core import ffmpeg
from app.core.exporters import export_docx, export_srt, export_txt
from app.core.formats import file_dialog_filter, is_supported_input
from app.core.naming import unique_output_path
from app.core.personal import PersonalStore
from app.core.stt_models import get_model
from app.paths import personal_dir
from app.settings import Settings
from app.ui.stt_common import (
    JobGuard, choose_local_model, fallback_dir, format_duration, model_dir_for, model_label_with_status,
    run_download, selectable_models,
)
from app.ui.stt_workers import TranscribeJob, TranscriptionWorker

log = logging.getLogger(__name__)

COL_FILE, COL_LENGTH, COL_STATUS = range(3)
MODEL_LOAD_ESTIMATE_S = 20


def persian_font(size: int = 12) -> QFont:
    families = set(QFontDatabase.families())
    for name in ("Vazirmatn", "Vazir", "Tahoma", "Segoe UI"):
        if name in families:
            return QFont(name, size)
    return QFont("", size)


class FileRow:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.duration = ffmpeg.probe(str(path)).duration or 0.0
        except Exception:  # noqa: BLE001
            self.duration = 0.0
        self.result = None            # TranscriptResult
        self.text = ""                # possibly edited by the user
        self.status = "Waiting"


class DropTable(QTableWidget):
    def __init__(self, on_drop):
        super().__init__(0, 3)
        self._on_drop = on_drop
        self.setHorizontalHeaderLabels(["File", "Length", "Status"])
        h = self.horizontalHeader()
        h.setSectionResizeMode(COL_FILE, QHeaderView.Stretch)
        h.setSectionResizeMode(COL_LENGTH, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_STATUS, QHeaderView.Interactive)
        self.setColumnWidth(COL_STATUS, 220)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)

    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._on_drop([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
        else:
            super().dropEvent(e)


class TranscribeTab(QWidget):
    TAB_TITLE = "Transcribe (Persian)"

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.rows: list[FileRow] = []
        self._thread: QThread | None = None
        self._worker: TranscriptionWorker | None = None
        self._batch: list[int] = []
        self._batch_audio = 0.0
        self._audio_done_before = 0.0      # finished files in this batch
        self._current_audio_done = 0.0
        self._started_at = 0.0
        self._first_file_at = 0.0
        self._loading_text = False
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_time_labels)
        self._build_ui()
        self._refresh_models()
        self._update_buttons()

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        self.add_btn = QPushButton("Add files…")
        self.remove_btn = QPushButton("Remove")
        self.clear_btn = QPushButton("Clear list")
        self.add_btn.clicked.connect(self._choose_files)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear)
        top = QHBoxLayout()
        for b in (self.add_btn, self.remove_btn, self.clear_btn):
            top.addWidget(b)
        top.addStretch()

        self.table = DropTable(self.add_paths)
        self.table.itemSelectionChanged.connect(self._show_selected)

        # model settings
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.download_btn = QPushButton("Download model")
        self.local_btn = QPushButton("Select local model folder…")
        self.download_btn.clicked.connect(self._download_model)
        self.local_btn.clicked.connect(self._select_local_model)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, max(2, os.cpu_count() or 8))
        self.threads_spin.setValue(self.settings.cpu_threads)
        self.threads_spin.setToolTip("CPU threads used by the model. 4 is best for most laptops.")
        self.threads_spin.valueChanged.connect(lambda v: setattr(self.settings, "cpu_threads", v))
        self.personal_check = QCheckBox()
        self.personal_check.setChecked(self.settings.personalization_enabled)
        self.personal_check.toggled.connect(lambda v: setattr(self.settings, "personalization_enabled", v))
        self.model_info = QLabel()
        self.model_info.setStyleSheet("color: gray;")
        self.model_info.setWordWrap(True)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model:"))
        row1.addWidget(self.model_combo, 1)
        row1.addWidget(self.download_btn)
        row1.addWidget(self.local_btn)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("CPU threads:"))
        row2.addWidget(self.threads_spin)
        row2.addSpacing(20)
        row2.addWidget(self.personal_check)
        row2.addStretch()
        box = QGroupBox("Settings")
        box_layout = QVBoxLayout(box)
        box_layout.addLayout(row1)
        box_layout.addLayout(row2)
        box_layout.addWidget(self.model_info)

        # progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.time_label = QLabel("")
        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.setMinimumWidth(120)
        self.cancel_btn = QPushButton("Cancel")
        self.transcribe_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.cancel)
        prow = QHBoxLayout()
        prow.addWidget(self.time_label, 1)
        prow.addWidget(self.cancel_btn)
        prow.addWidget(self.transcribe_btn)

        # transcript editor
        self.editor = QPlainTextEdit()
        self.editor.setFont(persian_font(13))
        self.editor.setLayoutDirection(Qt.RightToLeft)
        option = QTextOption()
        option.setTextDirection(Qt.RightToLeft)
        option.setAlignment(Qt.AlignRight)
        option.setWrapMode(QTextOption.WordWrap)
        self.editor.document().setDefaultTextOption(option)
        self.editor.setPlaceholderText("متن پیاده‌شده اینجا نمایش داده می‌شود. می‌توانید آن را ویرایش کنید.")
        self.editor.textChanged.connect(self._text_edited)
        self.copy_btn = QPushButton("Copy")
        self.txt_btn = QPushButton("Save TXT…")
        self.docx_btn = QPushButton("Save DOCX…")
        self.srt_btn = QPushButton("Save SRT…")
        self.all_btn = QPushButton("Save all…")
        self.srt_check = QCheckBox("include SRT")
        self.srt_check.setChecked(self.settings.export_srt)
        self.srt_check.toggled.connect(lambda v: setattr(self.settings, "export_srt", v))
        self.sample_btn = QPushButton("Save as personal sample")
        self.sample_btn.setToolTip("After correcting the text, save this voice + text so the app learns your words.")
        self.copy_btn.clicked.connect(self._copy)
        self.txt_btn.clicked.connect(lambda: self._save_one("txt"))
        self.docx_btn.clicked.connect(lambda: self._save_one("docx"))
        self.srt_btn.clicked.connect(lambda: self._save_one("srt"))
        self.all_btn.clicked.connect(self._save_all)
        self.sample_btn.clicked.connect(self._save_as_sample)
        erow = QHBoxLayout()
        for w in (self.copy_btn, self.txt_btn, self.docx_btn, self.srt_btn, self.all_btn, self.srt_check):
            erow.addWidget(w)
        erow.addStretch()
        erow.addWidget(self.sample_btn)
        editor_box = QWidget()
        el = QVBoxLayout(editor_box)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(self.editor, 1)
        el.addLayout(erow)

        upper = QWidget()
        ul = QVBoxLayout(upper)
        ul.setContentsMargins(0, 0, 0, 0)
        ul.addLayout(top)
        ul.addWidget(self.table, 1)
        ul.addWidget(box)
        ul.addWidget(self.progress)
        ul.addLayout(prow)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(upper)
        splitter.addWidget(editor_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # ---------------------------------------------------------------- models
    def _refresh_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for m in selectable_models():
            self.model_combo.addItem(model_label_with_status(self.settings, m), m.key)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(self.settings.stt_model)))
        self.model_combo.blockSignals(False)
        self._model_changed()

    def _model_changed(self):
        key = self.model_combo.currentData()
        if key:
            self.settings.stt_model = key
        folder = model_dir_for(self.settings, self.settings.stt_model)
        model = get_model(self.settings.stt_model)
        if folder:
            self.model_info.setText(f"Model folder: {folder}")
        else:
            self.model_info.setText(f"Not installed. Click “Download model” ({model.size_gb:.1f} GB, from GitHub), "
                                    f"or “Select local model folder…” if you downloaded {model.zip_name} yourself.")
        self._update_estimate()
        self._update_buttons()

    def _download_model(self):
        model = get_model(self.settings.stt_model)
        answer = QMessageBox.question(
            self, "Download model",
            f"Download “{model.label}” (about {model.size_gb:.1f} GB) from GitHub?\n\n"
            "This is the only time VoiceApp uses the internet. If it is interrupted, "
            "click Download again and it continues where it stopped.")
        if answer == QMessageBox.Yes and run_download(self, model):
            self._refresh_models()

    def _select_local_model(self):
        if choose_local_model(self, self.settings, get_model(self.settings.stt_model)):
            self._refresh_models()

    def refresh_personalization_label(self):
        store = PersonalStore(personal_dir())
        words = len(store.load_vocabulary())
        rules = sum(r.approved for r in store.load_rules())
        hints = f"{words} hint words" if self.settings.use_vocabulary_hints else "hints off"
        self.personal_check.setText(f"Use my personalization ({rules} correction rules, {hints})")

    def showEvent(self, event):
        self.refresh_personalization_label()
        if not self.is_busy():
            self._refresh_models()          # the Personalize tab may have changed the default model
        super().showEvent(event)

    # ---------------------------------------------------------------- files
    def _choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose voice messages", self.settings.stt_last_dir,
                                                file_dialog_filter())
        if files:
            self.settings.stt_last_dir = str(Path(files[0]).parent)
            self.add_paths(files)

    def add_paths(self, paths):
        if self.is_busy():
            return
        known = {os.path.normcase(str(r.path.resolve())) for r in self.rows}
        for p in paths:
            path = Path(p)
            files = sorted(f for f in path.rglob("*") if f.is_file()) if path.is_dir() else [path]
            for f in files:
                key = os.path.normcase(str(f.resolve()))
                if f.is_file() and is_supported_input(f.name) and key not in known:
                    known.add(key)
                    self._append(FileRow(f))
        self._update_estimate()
        self._update_buttons()
        if self.rows and not self.table.selectedItems():
            self.table.selectRow(0)

    def _append(self, row: FileRow):
        i = self.table.rowCount()
        self.table.insertRow(i)
        item = QTableWidgetItem(row.path.name)
        item.setToolTip(str(row.path))
        self.table.setItem(i, COL_FILE, item)
        self.table.setItem(i, COL_LENGTH, QTableWidgetItem(format_duration(row.duration) if row.duration else "?"))
        self.table.setItem(i, COL_STATUS, QTableWidgetItem(row.status))
        self.rows.append(row)

    def _remove_selected(self):
        r = self._selected_row()
        if r is not None and not self.is_busy():
            self.table.removeRow(r)
            del self.rows[r]
            self._update_estimate()
            self._show_selected()
            self._update_buttons()

    def _clear(self):
        if self.is_busy():
            return
        self.table.setRowCount(0)
        self.rows.clear()
        self.progress.setValue(0)
        self.time_label.setText("")
        self._show_selected()
        self._update_estimate()
        self._update_buttons()

    def _selected_row(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else None

    def _set_status(self, i: int, text: str, color=None):
        self.rows[i].status = text
        item = self.table.item(i, COL_STATUS)
        item.setText(text)
        item.setToolTip(text)
        item.setForeground(color if color is not None else self.palette().text())

    # ---------------------------------------------------------------- estimate
    def _pending_rows(self) -> list[int]:
        return [i for i, r in enumerate(self.rows) if r.result is None]

    def _update_estimate(self):
        if self.is_busy():
            return
        pending = self._pending_rows() or list(range(len(self.rows)))
        audio = sum(self.rows[i].duration for i in pending)
        if not audio:
            self.time_label.setText("Add voice messages (drag and drop works too).")
            return
        factor = get_model(self.settings.stt_model).speed_factor
        estimate = audio * factor + MODEL_LOAD_ESTIMATE_S
        self.time_label.setText(f"{len(pending)} file(s), {format_duration(audio)} of audio — "
                                f"estimated time about {format_duration(estimate)}")

    def _update_time_labels(self):
        if not self.is_busy():
            return
        elapsed = time.monotonic() - self._started_at
        done = self._audio_done_before + self._current_audio_done
        text = f"Elapsed {format_duration(elapsed)}"
        if done > 2 and self._first_file_at:
            rate = (time.monotonic() - self._first_file_at) / done          # measured on this PC
        else:
            rate = get_model(self.settings.stt_model).speed_factor
        remaining = max(0.0, self._batch_audio - done) * rate
        if not self._first_file_at:
            remaining += max(0.0, MODEL_LOAD_ESTIMATE_S - elapsed)
        text += f"  ·  remaining about {format_duration(remaining)}"
        if self._batch_audio:
            self.progress.setValue(int(min(1.0, done / self._batch_audio) * 1000))
        self.time_label.setText(text)

    # ---------------------------------------------------------------- run
    def is_busy(self) -> bool:
        return self._thread is not None

    def start(self):
        if self.is_busy() or not self.rows:
            return
        model_dir = model_dir_for(self.settings, self.settings.stt_model)
        if not model_dir:
            QMessageBox.information(self, "Model needed",
                                    "The selected model is not installed yet. Click “Download model” or "
                                    "“Select local model folder…”.")
            return
        rows = self._pending_rows() or list(range(len(self.rows)))
        if not JobGuard.acquire("transcription", self):
            return
        jobs = [TranscribeJob(i, self.rows[i].path) for i in rows]
        for i in rows:
            self.rows[i].result = None
            self._set_status(i, "Waiting")
        self._batch = rows
        self._batch_audio = sum(self.rows[i].duration for i in rows)
        self._audio_done_before = self._current_audio_done = 0.0
        self._started_at = time.monotonic()
        self._first_file_at = 0.0
        self.progress.setValue(0)
        use_personal = self.personal_check.isChecked()
        log.info("Transcribing %d files with %s (threads=%d, personalization=%s)",
                 len(jobs), model_dir, self.settings.cpu_threads, use_personal)
        self._thread = QThread(self)
        self._worker = TranscriptionWorker(jobs, model_dir, self.settings.cpu_threads,
                                           fallback_dir(self.settings, self.settings.stt_model),
                                           personal_dir() if use_personal else None,
                                           self.settings.use_vocabulary_hints)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status.connect(self._on_status)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_finished.connect(self._on_file_finished)
        self._worker.batch_finished.connect(self._on_batch_finished)
        self._worker.batch_finished.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()
        self._timer.start()
        self._update_buttons()

    def cancel(self):
        if self._worker:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.time_label.setText("Cancelling… (finishes the current part of the audio first)")

    def _on_status(self, text: str):
        if text:
            self.time_label.setText(text)

    def _on_file_started(self, i: int):
        if not self._first_file_at:
            self._first_file_at = time.monotonic()
        self._current_audio_done = 0.0
        self._set_status(i, "Transcribing… 0%")
        self.table.selectRow(i)

    def _on_file_progress(self, i: int, fraction: float, audio_done: float):
        self._current_audio_done = audio_done
        self._set_status(i, f"Transcribing… {int(fraction * 100)}%")
        self._update_time_labels()

    def _on_file_finished(self, i: int, result, error: str):
        row = self.rows[i]
        self._audio_done_before += row.duration
        self._current_audio_done = 0.0
        if result is not None:
            row.result = result
            row.text = result.text
            secs = result.processing_seconds
            status = f"Done ({format_duration(secs)})"
            if result.warnings:
                status += " – " + " ".join(result.warnings)
            self._set_status(i, status, Qt.darkGreen if not result.warnings else Qt.darkYellow)
        elif error == "Cancelled":
            self._set_status(i, "Cancelled", Qt.gray)
        else:
            self._set_status(i, f"Error: {error}", Qt.red)
        if self._selected_row() == i:
            self._show_selected()

    def _on_batch_finished(self, ok: int, failed: int, cancelled: bool, warning: str):
        elapsed = time.monotonic() - self._started_at
        msg = f"{ok} transcribed in {format_duration(elapsed)}"
        if failed:
            msg += f", {failed} failed (hover over the status)"
        if cancelled:
            msg += ", cancelled"
        self.time_label.setText(msg + ".")
        if not cancelled and not failed:
            self.progress.setValue(1000)
        if warning:
            QMessageBox.warning(self, "Model problem", warning)

    def _on_thread_finished(self):
        self._timer.stop()
        self._thread.deleteLater()
        self._worker.deleteLater()
        self._thread = self._worker = None
        JobGuard.release()
        for i in self._batch:
            if self.rows[i].status == "Waiting":
                self._set_status(i, "Cancelled", Qt.gray)
        self._update_buttons()

    # ---------------------------------------------------------------- editor & export
    def _show_selected(self):
        i = self._selected_row()
        self._loading_text = True
        self.editor.setPlainText(self.rows[i].text if i is not None and i < len(self.rows) else "")
        self._loading_text = False
        self._update_buttons()

    def _text_edited(self):
        if self._loading_text:
            return
        i = self._selected_row()
        if i is not None and self.rows[i].result is not None:
            self.rows[i].text = self.editor.toPlainText()

    def _current(self) -> FileRow | None:
        i = self._selected_row()
        if i is None or self.rows[i].result is None:
            return None
        return self.rows[i]

    def _copy(self):
        QGuiApplication.clipboard().setText(self.editor.toPlainText())
        self.time_label.setText("Copied to clipboard.")

    def _default_folder(self, row: FileRow) -> Path:
        folder = Path(self.settings.stt_output_dir) if self.settings.stt_output_dir else row.path.parent
        return folder if folder.is_dir() else row.path.parent

    def _save_one(self, kind: str):
        row = self._current()
        if not row:
            return
        default = unique_output_path(self._default_folder(row), row.path.stem, "." + kind)
        filters = {"txt": "Text (*.txt)", "docx": "Word document (*.docx)", "srt": "Subtitles (*.srt)"}
        path, _ = QFileDialog.getSaveFileName(self, f"Save {kind.upper()}", str(default), filters[kind])
        if not path:
            return
        try:
            self._write(kind, Path(path), row)
            self.settings.stt_output_dir = str(Path(path).parent)
            self.time_label.setText(f"Saved {path}")
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))

    def _write(self, kind: str, path: Path, row: FileRow):
        if kind == "txt":
            export_txt(path, row.text)
        elif kind == "docx":
            export_docx(path, row.text, title=row.path.stem)
        else:
            export_srt(path, row.result.segments)

    def _save_all(self):
        done = [r for r in self.rows if r.result is not None]
        if not done:
            return
        start = self.settings.stt_output_dir or str(done[0].path.parent)
        folder = QFileDialog.getExistingDirectory(self, "Save all transcripts to…", start)
        if not folder:
            return
        kinds = ["txt", "docx"] + (["srt"] if self.srt_check.isChecked() else [])
        count = 0
        try:
            for row in done:
                for kind in kinds:
                    self._write(kind, unique_output_path(Path(folder), row.path.stem, "." + kind), row)
                    count += 1
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
        self.settings.stt_output_dir = folder
        self.time_label.setText(f"Saved {count} file(s) to {folder} (existing files were not overwritten).")

    def _save_as_sample(self):
        row = self._current()
        if not row:
            return
        text = self.editor.toPlainText().strip()
        if not text:
            return
        answer = QMessageBox.question(
            self, "Save as personal sample",
            "Is this text now exactly what was said?\n\nThe voice file (a copy) and this text will be "
            "added to your personal samples, so VoiceApp can learn your words and common mistakes.")
        if answer != QMessageBox.Yes:
            return
        try:
            PersonalStore(personal_dir()).add_sample(row.path, text)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save sample", str(exc))
            return
        self.time_label.setText("Saved as a personal sample. Open the Personalize tab to update suggestions.")

    # ---------------------------------------------------------------- state
    def _update_buttons(self):
        busy = self.is_busy()
        has_rows = bool(self.rows)
        installed = model_dir_for(self.settings, self.settings.stt_model) is not None
        for w in (self.add_btn, self.clear_btn, self.model_combo, self.download_btn, self.local_btn,
                  self.threads_spin, self.personal_check):
            w.setEnabled(not busy)
        self.download_btn.setText("Download again" if installed else "Download model")
        self.remove_btn.setEnabled(not busy and self._selected_row() is not None)
        self.transcribe_btn.setEnabled(not busy and has_rows and installed)
        self.cancel_btn.setEnabled(busy)
        current = self._current() is not None
        for w in (self.copy_btn, self.txt_btn, self.docx_btn, self.sample_btn):
            w.setEnabled(current)
        self.srt_btn.setEnabled(current and bool(self._current() and self._current().result.segments))
        self.all_btn.setEnabled(any(r.result is not None for r in self.rows))
        self.editor.setReadOnly(not current)
        self.table.setAcceptDrops(not busy)

    def shutdown(self):
        if self._worker:
            self._worker.cancel()
        if self._thread:
            self._thread.quit()
            self._thread.wait(30_000)
