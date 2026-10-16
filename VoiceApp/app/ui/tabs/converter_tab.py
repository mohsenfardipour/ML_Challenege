"""The "Convert audio" tab."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.core.converter import ConversionResult, output_folder_for
from app.core.formats import (
    BITRATES_KBPS, OUTPUT_FORMATS, file_dialog_filter, get_format, is_supported_input,
)
from app.paths import IS_WINDOWS
from app.settings import Settings
from app.ui.workers import ConversionJob, ConversionWorker

log = logging.getLogger(__name__)

COL_FILE, COL_STATUS, COL_OUTPUT = range(3)

STATUS_WAITING = "Waiting"
STATUS_DONE = "Done"


class FileTable(QTableWidget):
    """File list that accepts files and folders dropped from Explorer."""

    def __init__(self, on_paths_dropped):
        super().__init__(0, 3)
        self._on_paths_dropped = on_paths_dropped
        self.setHorizontalHeaderLabels(["File", "Status", "Output"])
        header = self.horizontalHeader()
        header.setSectionResizeMode(COL_FILE, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.Interactive)
        header.setSectionResizeMode(COL_OUTPUT, QHeaderView.Stretch)
        self.setColumnWidth(COL_STATUS, 230)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setWordWrap(False)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            event.acceptProposedAction()
            self._on_paths_dropped(paths)
        else:
            super().dropEvent(event)


class ConverterTab(QWidget):
    TAB_TITLE = "Convert audio"

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._sources: list[Path] = []          # one per table row
        self._thread: QThread | None = None
        self._worker: ConversionWorker | None = None
        self._batch_rows: list[int] = []
        self._finished_in_batch = 0
        self._last_output: Path | None = None
        self._loading = False
        self._build_ui()
        self._load_settings()
        self._update_buttons()

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        self.add_files_btn = QPushButton("Add files…")
        self.add_folder_btn = QPushButton("Add folder…")
        self.remove_btn = QPushButton("Remove selected")
        self.clear_btn = QPushButton("Clear list")
        self.add_files_btn.clicked.connect(self._choose_files)
        self.add_folder_btn.clicked.connect(self._choose_folder)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear_list)

        top = QHBoxLayout()
        for btn in (self.add_files_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
            top.addWidget(btn)
        top.addStretch()

        self.table = FileTable(self.add_paths)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.hint = QLabel("Drag and drop audio files or folders here, or click “Add files…”.")
        self.hint.setStyleSheet("color: gray;")

        # Settings box
        self.format_combo = QComboBox()
        for fmt in OUTPUT_FORMATS:
            self.format_combo.addItem(fmt.label, fmt.key)
        self.bitrate_combo = QComboBox()
        for kbps in BITRATES_KBPS:
            self.bitrate_combo.addItem(f"{kbps} kbps", kbps)

        self.same_dir_radio = QRadioButton("Same folder as each source file")
        self.custom_dir_radio = QRadioButton("This folder:")
        self.custom_dir_edit = QLineEdit()
        self.custom_dir_edit.setPlaceholderText("Choose an output folder…")
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._choose_output_dir)
        self.same_dir_radio.toggled.connect(self._update_buttons)
        for widget in (self.format_combo, self.bitrate_combo):
            widget.currentIndexChanged.connect(self._settings_changed)
        self.same_dir_radio.toggled.connect(self._settings_changed)
        self.custom_dir_edit.editingFinished.connect(self._settings_changed)

        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_dir_radio)
        custom_row.addWidget(self.custom_dir_edit, 1)
        custom_row.addWidget(self.browse_btn)
        out_box = QVBoxLayout()
        out_box.addWidget(self.same_dir_radio)
        out_box.addLayout(custom_row)

        form = QFormLayout()
        form.addRow("Output format:", self.format_combo)
        form.addRow("Bitrate:", self.bitrate_combo)
        form.addRow("Save to:", out_box)
        settings_box = QGroupBox("Settings")
        settings_box.setLayout(form)

        # Bottom bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.summary = QLabel("")
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.setDefault(True)
        self.convert_btn.setMinimumWidth(110)
        self.cancel_btn = QPushButton("Cancel")
        self.open_btn = QPushButton("Open output folder")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.open_btn.clicked.connect(self._open_output_folder)

        bottom = QHBoxLayout()
        bottom.addWidget(self.summary, 1)
        bottom.addWidget(self.open_btn)
        bottom.addWidget(self.cancel_btn)
        bottom.addWidget(self.convert_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.hint)
        layout.addWidget(settings_box)
        layout.addWidget(self.progress)
        layout.addLayout(bottom)

    # -------------------------------------------------------------- settings
    def _load_settings(self):
        s = self.settings
        self._loading = True
        self.format_combo.setCurrentIndex(max(0, self.format_combo.findData(s.output_format)))
        self.bitrate_combo.setCurrentIndex(max(0, self.bitrate_combo.findData(s.bitrate_kbps)))
        self.custom_dir_edit.setText(s.custom_output_dir)
        (self.custom_dir_radio if s.use_custom_output_dir else self.same_dir_radio).setChecked(True)
        self._loading = False

    def _settings_changed(self):
        if self._loading:
            return
        s = self.settings
        s.output_format = self.format_combo.currentData()
        s.bitrate_kbps = self.bitrate_combo.currentData()
        s.use_custom_output_dir = self.custom_dir_radio.isChecked()
        s.custom_output_dir = self.custom_dir_edit.text().strip()

    # ------------------------------------------------------------ file list
    def _choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose audio files", self.settings.last_input_dir, file_dialog_filter())
        if files:
            self.settings.last_input_dir = str(Path(files[0]).parent)
            self.add_paths(files)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder with audio files", self.settings.last_input_dir)
        if folder:
            self.settings.last_input_dir = folder
            self.add_paths([folder])

    def add_paths(self, paths: list[str]):
        """Add files; folders are searched (including sub-folders) for audio files."""
        if self.is_busy():
            return
        found: list[Path] = []
        skipped = 0
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for root, _dirs, names in os.walk(path):
                    for name in sorted(names):
                        if is_supported_input(name) and ".voiceapp.part" not in name:
                            found.append(Path(root) / name)
            elif path.is_file() and is_supported_input(path.name):
                found.append(path)
            else:
                skipped += 1
        known = {os.path.normcase(str(s.resolve())) for s in self._sources}
        added = 0
        for path in found:
            key = os.path.normcase(str(path.resolve()))
            if key in known:
                continue
            known.add(key)
            self._append_row(path)
            added += 1
        msg = f"Added {added} file(s)."
        if skipped:
            msg += f" Skipped {skipped} unsupported item(s)."
        self.summary.setText(msg)
        log.info("Added %d files, skipped %d", added, skipped)
        self._update_buttons()

    def _append_row(self, path: Path):
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(path.name)
        name_item.setToolTip(str(path))
        self.table.setItem(row, COL_FILE, name_item)
        self.table.setItem(row, COL_STATUS, QTableWidgetItem(STATUS_WAITING))
        self.table.setItem(row, COL_OUTPUT, QTableWidgetItem(""))
        self._sources.append(path)

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            del self._sources[row]
        self._update_buttons()

    def _clear_list(self):
        self.table.setRowCount(0)
        self._sources.clear()
        self.progress.setValue(0)
        self.summary.setText("")
        self._update_buttons()

    def _choose_output_dir(self):
        start = self.custom_dir_edit.text() or self.settings.last_input_dir
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if folder:
            self.custom_dir_edit.setText(str(Path(folder)))
            self.custom_dir_radio.setChecked(True)
            self._settings_changed()

    # ------------------------------------------------------------ conversion
    def is_busy(self) -> bool:
        return self._thread is not None

    def start_conversion(self):
        if self.is_busy() or not self._sources:
            return
        self._settings_changed()
        custom = None
        if self.settings.use_custom_output_dir:
            custom = self.settings.custom_output_dir
            if not custom:
                QMessageBox.warning(self, "No output folder",
                                    "Please choose an output folder, or select "
                                    "“Same folder as each source file”.")
                return

        # Convert everything not yet done; if all are done, convert all again.
        rows = [r for r in range(len(self._sources)) if not self._status(r).startswith(STATUS_DONE)]
        if not rows:
            rows = list(range(len(self._sources)))
        jobs = [ConversionJob(r, self._sources[r], output_folder_for(self._sources[r], custom))
                for r in rows]
        for r in rows:
            self._set_status(r, STATUS_WAITING)
            self.table.item(r, COL_OUTPUT).setText("")

        fmt = get_format(self.settings.output_format)
        log.info("Starting batch: %d files, %s, %d kbps", len(jobs), fmt.key, self.settings.bitrate_kbps)
        self._batch_rows = rows
        self._finished_in_batch = 0
        self.progress.setValue(0)
        self.summary.setText(f"Converting {len(jobs)} file(s)…")

        self._thread = QThread(self)
        self._worker = ConversionWorker(jobs, fmt, self.settings.bitrate_kbps)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_finished.connect(self._on_file_finished)
        self._worker.batch_finished.connect(self._on_batch_finished)
        self._worker.batch_finished.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()
        self._update_buttons()

    def cancel_conversion(self):
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.summary.setText("Cancelling…")

    def _on_file_started(self, row: int):
        self._set_status(row, "Converting…")
        self.table.scrollToItem(self.table.item(row, COL_FILE))

    def _on_file_progress(self, row: int, fraction: float):
        self._set_status(row, f"Converting… {int(fraction * 100)}%")
        self._update_overall(fraction)

    def _on_file_finished(self, row: int, result: ConversionResult):
        self._finished_in_batch += 1
        status_item = self.table.item(row, COL_STATUS)
        if result.ok:
            if result.warning:
                status_item.setText(f"{STATUS_DONE} (warning)")
                status_item.setForeground(Qt.darkYellow)
                status_item.setToolTip("Warning: " + result.warning)
            else:
                status_item.setText(STATUS_DONE)
                status_item.setForeground(Qt.darkGreen)
                status_item.setToolTip("Converted successfully")
            out_item = self.table.item(row, COL_OUTPUT)
            out_item.setText(result.output_path.name)
            out_item.setToolTip(str(result.output_path))
            self._last_output = result.output_path
        elif result.cancelled:
            status_item.setText("Cancelled")
            status_item.setForeground(Qt.gray)
        else:
            status_item.setText(f"Error: {result.message}")
            status_item.setToolTip(result.message)
            status_item.setForeground(Qt.red)
        self._update_overall(0.0)

    def _on_batch_finished(self, ok: int, failed: int, cancelled: bool):
        parts = [f"{ok} converted"]
        if failed:
            parts.append(f"{failed} failed (hover over the status to see why)")
        if cancelled:
            parts.append("cancelled")
        self.summary.setText(", ".join(parts) + ".")
        if not cancelled:
            self.progress.setValue(self.progress.maximum())
        log.info("Batch finished: ok=%d failed=%d cancelled=%s", ok, failed, cancelled)

    def _on_thread_finished(self):
        self._thread.deleteLater()
        self._worker.deleteLater()
        self._thread = None
        self._worker = None
        # Rows left "Waiting" after a cancel were never started.
        for r in self._batch_rows:
            if self._status(r) == STATUS_WAITING:
                self._set_status(r, "Cancelled")
        self._update_buttons()

    def _update_overall(self, current_fraction: float):
        total = len(self._batch_rows) or 1
        value = (self._finished_in_batch + current_fraction) / total
        self.progress.setValue(int(value * self.progress.maximum()))

    def _status(self, row: int) -> str:
        return self.table.item(row, COL_STATUS).text()

    def _set_status(self, row: int, text: str):
        item = self.table.item(row, COL_STATUS)
        item.setText(text)
        item.setForeground(self.palette().text())
        item.setToolTip("")

    # ------------------------------------------------------------ misc
    def _open_output_folder(self):
        if self._last_output and self._last_output.exists():
            if IS_WINDOWS:
                # Opens Explorer with the converted file selected.
                subprocess.Popen(["explorer", "/select,", str(self._last_output)])
                return
            folder = self._last_output.parent
        elif self.settings.use_custom_output_dir and self.settings.custom_output_dir:
            folder = Path(self.settings.custom_output_dir)
        elif self._sources:
            folder = self._sources[0].parent
        else:
            return
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _update_buttons(self):
        busy = self.is_busy()
        has_files = bool(self._sources)
        for w in (self.add_files_btn, self.add_folder_btn, self.clear_btn, self.format_combo,
                  self.bitrate_combo, self.same_dir_radio, self.custom_dir_radio):
            w.setEnabled(not busy)
        self.table.setAcceptDrops(not busy)
        self.remove_btn.setEnabled(not busy and bool(self.table.selectedIndexes()))
        custom_enabled = not busy and self.custom_dir_radio.isChecked()
        self.custom_dir_edit.setEnabled(custom_enabled)
        self.browse_btn.setEnabled(not busy)
        self.convert_btn.setEnabled(not busy and has_files)
        self.cancel_btn.setEnabled(busy)
        self.open_btn.setEnabled(has_files or self._last_output is not None)
        self.hint.setVisible(not has_files)

    def shutdown(self):
        """Stop a running conversion and wait for it (called when the window closes)."""
        self._settings_changed()
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10_000)
