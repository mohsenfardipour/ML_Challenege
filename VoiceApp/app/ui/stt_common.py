"""Helpers shared by the Transcribe and Personalize tabs: model status, download dialog,
and a guard so only one heavy speech-to-text job runs at a time."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QEventLoop, QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QWidget

from app.core.model_download import DownloadCancelled, DownloadError, download_model, install_zip
from app.core.stt_models import FALLBACK_MODEL_KEY, STT_MODELS, SttModel, find_model_dir, get_model, resolve_model_dir
from app.paths import models_dir
from app.settings import Settings

log = logging.getLogger(__name__)


class JobGuard:
    """Only one model may be running (transcribing, testing or downloading) at a time."""
    owner: str | None = None

    @classmethod
    def acquire(cls, name: str, parent: QWidget) -> bool:
        if cls.owner is not None:
            QMessageBox.information(parent, "Please wait", f"Another job is running ({cls.owner}). "
                                    "Wait for it to finish or cancel it first.")
            return False
        cls.owner = name
        return True

    @classmethod
    def release(cls) -> None:
        cls.owner = None


def format_duration(seconds: float) -> str:
    seconds = int(round(max(0, seconds)))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def model_dir_for(settings: Settings, key: str) -> Path | None:
    return resolve_model_dir(key, settings.model_dirs)


def model_label_with_status(settings: Settings, model: SttModel) -> str:
    installed = model_dir_for(settings, model.key) is not None
    return f"{model.label}  —  {'installed' if installed else f'not installed ({model.size_gb:.1f} GB)'}"


def selectable_models() -> list[SttModel]:
    return [m for m in STT_MODELS if not m.fallback_only] + [m for m in STT_MODELS if m.fallback_only]


def fallback_dir(settings: Settings, key: str) -> Path | None:
    """Base large-v3-turbo, used only when a Persian model fails to load."""
    if key == FALLBACK_MODEL_KEY:
        return None
    return model_dir_for(settings, FALLBACK_MODEL_KEY)


# ---------------------------------------------------------------- download
class _DownloadWorker(QObject):
    progress = Signal(object, object, str)     # done, total, phase  (object: may exceed 32-bit int)
    finished = Signal(bool, str)               # ok, message or folder

    def __init__(self, model: SttModel, zip_file: Path | None, cancel_event):
        super().__init__()
        self.model, self.zip_file, self.cancel_event = model, zip_file, cancel_event

    @Slot()
    def run(self):
        try:
            if self.zip_file:
                folder = install_zip(self.zip_file, lambda d, t, p: self.progress.emit(d, t, p), self.cancel_event)
            else:
                folder = download_model(self.model, lambda d, t, p: self.progress.emit(d, t, p), self.cancel_event)
            self.finished.emit(True, str(folder))
        except DownloadCancelled:
            self.finished.emit(False, "Cancelled. Click Download again later to continue where it stopped.")
        except DownloadError as exc:
            self.finished.emit(False, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("Model download failed")
            self.finished.emit(False, f"Unexpected error: {exc}")


def run_download(parent: QWidget, model: SttModel, zip_file: Path | None = None) -> Path | None:
    """Download (or unzip a local zip) with a modal progress dialog. Returns the model folder."""
    import threading

    if not JobGuard.acquire("model download", parent):
        return None
    cancel_event = threading.Event()
    dialog = QProgressDialog(f"Downloading {model.label}…", "Cancel", 0, 1000, parent)
    dialog.setWindowTitle("Model download")
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setMinimumWidth(460)
    thread = QThread(parent)
    worker = _DownloadWorker(model, zip_file, cancel_event)
    worker.moveToThread(thread)
    outcome: dict = {}

    def on_progress(done, total, phase):
        mb, total_mb = done / 1e6, total / 1e6
        if phase == "download":
            dialog.setLabelText(f"Downloading {model.label}\n{mb:,.0f} MB of {total_mb:,.0f} MB"
                                if total else f"Downloading {model.label}\n{mb:,.0f} MB")
        else:
            dialog.setLabelText(f"Unpacking {model.label}…")
        if total:
            dialog.setValue(int(done / total * 1000))

    def on_finished(ok, message):
        outcome["ok"], outcome["message"] = ok, message
        thread.quit()

    thread.started.connect(worker.run)
    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    dialog.canceled.connect(cancel_event.set)
    loop = QEventLoop()
    thread.finished.connect(loop.quit)
    thread.start()
    dialog.show()
    loop.exec()                       # keeps the window responsive until the worker is done
    dialog.close()
    worker.deleteLater()
    thread.deleteLater()
    JobGuard.release()
    if outcome.get("ok"):
        QMessageBox.information(parent, "Model ready", f"{model.label} is installed in:\n{outcome['message']}")
        return Path(outcome["message"])
    QMessageBox.warning(parent, "Model not installed", outcome.get("message", "Unknown error"))
    return None


def choose_local_model(parent: QWidget, settings: Settings, model: SttModel) -> Path | None:
    """Let the user pick an unzipped model folder or a downloaded zip."""
    box = QMessageBox(parent)
    box.setWindowTitle("Install model manually")
    box.setText(f"Where is the model for “{model.label}”?\n\n"
                f"Expected file name: {model.zip_name}\n"
                f"Standard folder: {models_dir() / model.key}")
    folder_btn = box.addButton("Unzipped folder…", QMessageBox.AcceptRole)
    zip_btn = box.addButton("Zip file…", QMessageBox.AcceptRole)
    box.addButton(QMessageBox.Cancel)
    box.exec()
    if box.clickedButton() is folder_btn:
        folder = QFileDialog.getExistingDirectory(parent, "Select the model folder", str(models_dir()))
        if not folder:
            return None
        found = find_model_dir(folder)
        if not found:
            QMessageBox.warning(parent, "Not a model folder",
                                "This folder does not contain model.bin, config.json, tokenizer.json "
                                "and vocabulary.json. Select the folder that contains these files.")
            return None
        settings.model_dirs[model.key] = str(found)
        return found
    if box.clickedButton() is zip_btn:
        path, _ = QFileDialog.getOpenFileName(parent, "Select the model zip", "", "Zip files (*.zip)")
        if path:
            folder = run_download(parent, model, Path(path))
            if folder:
                settings.model_dirs.pop(model.key, None)
            return folder
    return None


def model_key_list() -> list[str]:
    return [m.key for m in selectable_models()]


__all__ = ["JobGuard", "format_duration", "model_dir_for", "model_label_with_status", "selectable_models",
           "fallback_dir", "run_download", "choose_local_model", "get_model", "model_key_list"]
