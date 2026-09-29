"""مرتب‌کننده‌ی خودکار پوشه‌ی Downloads — رابط گرافیکی (PySide6)."""

import sys
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import organizer


def make_icon() -> QIcon:
    """آیکون ساده را با کد می‌کشیم تا به فایل عکس جداگانه نیاز نباشد."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#2b7de9"))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 12, 12)
    painter.setPen(QColor("white"))
    painter.setFont(QFont("Arial", 30, QFont.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "D")
    painter.end()
    return QIcon(pixmap)


class AutoSortHandler(FileSystemEventHandler):
    """وقتی فایل جدیدی در پوشه ظاهر شد، آن را مرتب می‌کند.

    watchdog این کلاس را در یک ترد جدا صدا می‌زند؛ پس با Signal به پنجره خبر می‌دهیم.
    """

    def __init__(self, root: Path, log_callback, moved_callback):
        super().__init__()
        self.root = root
        self.log_callback = log_callback
        self.moved_callback = moved_callback
        self._lock = threading.Lock()
        self._busy = set()  # فایل‌هایی که همین الان در حال بررسی‌اند

    def on_created(self, event):
        if not event.is_directory:
            self._handle(Path(event.src_path))

    def on_moved(self, event):  # مثلاً وقتی a.crdownload به a.zip تغییر نام می‌دهد
        if not event.is_directory:
            self._handle(Path(event.dest_path))

    def _handle(self, path: Path):
        if path.parent != self.root or organizer.should_skip(path):
            return
        with self._lock:
            if path in self._busy:
                return
            self._busy.add(path)
        try:
            if not organizer.wait_until_stable(path):
                return
            result = organizer.move_file(path, self.root)
            if result:
                src, dst = result
                self.log_callback(f"[خودکار] {src.name}  ←  {dst.parent.name}/{dst.name}")
                self.moved_callback([result])
        except OSError as error:
            self.log_callback(f"[خودکار] خطا در {path.name}: {error}")
        finally:
            with self._lock:
                self._busy.discard(path)


class Bridge(QObject):
    """پلی بین تردِ watchdog و پنجره (Qt فقط اجازه می‌دهد پنجره در تردِ اصلی عوض شود)."""

    log = Signal(str)
    moved = Signal(list)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("مرتب‌کننده‌ی Downloads")
        self.setWindowIcon(make_icon())
        self.resize(640, 460)
        self.setLayoutDirection(Qt.RightToLeft)

        self.history = []  # هر عضو = لیست جابه‌جایی‌های یک مرتب‌سازی
        self.observer = None
        self.bridge = Bridge()
        self.bridge.log.connect(self.add_log)
        self.bridge.moved.connect(self.record_moves)

        self._build_ui()
        self._build_tray()

    # ---------- ساخت رابط ----------
    def _build_ui(self):
        self.folder_edit = QLineEdit(str(organizer.get_downloads_folder()))
        self.folder_edit.setLayoutDirection(Qt.LeftToRight)
        browse_btn = QPushButton("انتخاب پوشه...")
        browse_btn.clicked.connect(self.choose_folder)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("پوشه:"))
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse_btn)

        self.sort_btn = QPushButton("مرتب کن")
        self.sort_btn.clicked.connect(self.sort_now)
        self.undo_btn = QPushButton("برگرداندن (Undo)")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.undo_last)

        button_row = QHBoxLayout()
        button_row.addWidget(self.sort_btn)
        button_row.addWidget(self.undo_btn)

        self.auto_check = QCheckBox("مرتب‌سازی خودکار (پوشه را زیر نظر بگیر)")
        self.auto_check.toggled.connect(self.toggle_auto)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addLayout(folder_row)
        layout.addLayout(button_row)
        layout.addWidget(self.auto_check)
        layout.addWidget(QLabel("گزارش کارها:"))
        layout.addWidget(self.log_view, 1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _build_tray(self):
        self.tray = QSystemTrayIcon(make_icon(), self)
        self.tray.setToolTip("مرتب‌کننده‌ی Downloads")

        menu = QMenu()
        show_action = QAction("نمایش پنجره", self)
        show_action.triggered.connect(self.show_window)
        sort_action = QAction("مرتب کن", self)
        sort_action.triggered.connect(self.sort_now)
        self.tray_auto = QAction("مرتب‌سازی خودکار", self, checkable=True)
        self.tray_auto.toggled.connect(self.auto_check.setChecked)
        self.auto_check.toggled.connect(self.tray_auto.setChecked)
        quit_action = QAction("خروج", self)
        quit_action.triggered.connect(self.quit_app)

        for action in (show_action, sort_action, self.tray_auto):
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_clicked)
        self.tray.show()

    # ---------- کارهای دکمه‌ها ----------
    def current_folder(self) -> Path:
        return Path(self.folder_edit.text().strip())

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "یک پوشه انتخاب کنید", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            if self.auto_check.isChecked():  # نظارت را روی پوشه‌ی جدید ببر
                self.stop_watching()
                self.start_watching()

    def add_log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"{stamp}  {text}")

    def record_moves(self, moves: list):
        if moves:
            self.history.append(moves)
            self.undo_btn.setEnabled(True)

    def sort_now(self):
        folder = self.current_folder()
        if not folder.is_dir():
            QMessageBox.warning(self, "خطا", "این پوشه وجود ندارد.")
            return
        moves = organizer.organize_folder(folder, log=self.add_log)
        self.record_moves(moves)

    def undo_last(self):
        if not self.history:
            return
        moves = self.history.pop()
        organizer.undo_moves(moves, log=self.add_log)
        self.undo_btn.setEnabled(bool(self.history))

    # ---------- مرتب‌سازی خودکار ----------
    def toggle_auto(self, enabled: bool):
        if enabled:
            self.start_watching()
        else:
            self.stop_watching()

    def start_watching(self):
        folder = self.current_folder()
        if not folder.is_dir():
            QMessageBox.warning(self, "خطا", "این پوشه وجود ندارد.")
            self.auto_check.setChecked(False)
            return
        handler = AutoSortHandler(folder, self.bridge.log.emit, self.bridge.moved.emit)
        self.observer = Observer()
        self.observer.schedule(handler, str(folder), recursive=False)
        self.observer.start()
        self.add_log(f"نظارت روشن شد: {folder}")

    def stop_watching(self):
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=3)
            self.observer = None
            self.add_log("نظارت خاموش شد.")

    # ---------- System Tray و بستن پنجره ----------
    def show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_clicked(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.show_window()

    def closeEvent(self, event):
        """با بستن پنجره برنامه بسته نمی‌شود؛ فقط پنهان می‌شود."""
        event.ignore()
        self.hide()
        self.tray.showMessage(
            "مرتب‌کننده‌ی Downloads",
            "برنامه در پس‌زمینه در حال اجراست. برای خروج از منوی آیکون استفاده کنید.",
            QSystemTrayIcon.Information,
            3000,
        )

    def quit_app(self):
        self.stop_watching()
        self.tray.hide()
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # پنجره بسته شد، برنامه بماند
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
