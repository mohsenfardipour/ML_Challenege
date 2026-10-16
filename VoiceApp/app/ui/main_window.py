"""Main window: one tab per feature (see app/ui/tabs/__init__.py)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget

from app import APP_NAME, APP_VERSION
from app.core.ffmpeg import FFmpegNotFound, ffmpeg_path
from app.logging_setup import log_file
from app.settings import Settings, save_settings
from app.ui.tabs import TABS

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(900, 620)

        self.tabs = QTabWidget()
        for tab_class in TABS:
            self.tabs.addTab(tab_class(settings, self), tab_class.TAB_TITLE)
        self.tabs.setCurrentIndex(min(settings.last_tab, self.tabs.count() - 1))
        self.setCentralWidget(self.tabs)

        status = QLabel(f"Log file: {log_file()}")
        status.setStyleSheet("color: gray;")
        self.statusBar().addWidget(status)

        if settings.window_geometry:
            self.restoreGeometry(QByteArray.fromHex(settings.window_geometry.encode()))

    def check_ffmpeg(self):
        try:
            log.info("Using FFmpeg: %s", ffmpeg_path())
        except FFmpegNotFound as exc:
            log.error(str(exc))
            QMessageBox.critical(self, "FFmpeg missing", str(exc))

    def _tab_widgets(self):
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def closeEvent(self, event):
        busy = [t for t in self._tab_widgets() if getattr(t, "is_busy", lambda: False)()]
        if busy:
            answer = QMessageBox.question(
                self, "Work in progress",
                "A conversion is still running. Stop it and quit?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        for tab in self._tab_widgets():
            if hasattr(tab, "shutdown"):
                tab.shutdown()
        self.settings.last_tab = self.tabs.currentIndex()
        self.settings.window_geometry = bytes(self.saveGeometry().toHex()).decode()
        save_settings(self.settings)
        log.info("Closing")
        event.accept()
