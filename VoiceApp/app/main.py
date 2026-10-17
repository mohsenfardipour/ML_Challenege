"""Program entry point.

    python -m app.main                       start the GUI
    python -m app.main --selftest IN OUT     convert every file in IN to all formats,
                                             check the results, write OUT/selftest_report.txt
    python -m app.main --smoke-test          open the window, close it after 2 s (for build checks)
    python -m app.main --stt-selftest MODEL_DIR AUDIO OUT [REFERENCE.txt]
                                             transcribe AUDIO with a real model, write OUT/stt_selftest_report.txt

The same options work with the packaged VoiceApp.exe.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from app import APP_NAME
from app.logging_setup import setup_logging

log = logging.getLogger(__name__)


def run_gui(smoke_test: bool = False) -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app.settings import load_settings
    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    if sys.platform.startswith("win"):
        # Segoe UI contains Persian/Arabic glyphs and is on every Windows 10/11 PC.
        app.setFont(QFont("Segoe UI", 10))

    def excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.critical("Unhandled error:\n%s", text)
        QMessageBox.critical(None, "Unexpected error",
                             f"Something went wrong:\n\n{exc}\n\nDetails are in the log file.")
    sys.excepthook = excepthook

    window = MainWindow(load_settings())
    window.show()
    if smoke_test:
        QTimer.singleShot(2000, window.close)
    else:
        QTimer.singleShot(0, window.check_ffmpeg)
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.add_argument("--selftest", nargs=2, metavar=("INPUT_DIR", "OUTPUT_DIR"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stt-selftest", nargs="+", metavar="ARG")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        if args.stt_selftest or args.selftest:
            # Command-line modes: report errors in the log and the exit code. Raising would make
            # the packaged exe (no console) show an error dialog that waits forever for a click.
            try:
                if args.stt_selftest:
                    from app.selftest import run_stt_selftest
                    a = args.stt_selftest
                    return run_stt_selftest(Path(a[0]), Path(a[1]), Path(a[2]), Path(a[3]) if len(a) > 3 else None)
                from app.selftest import run_selftest
                return run_selftest(Path(args.selftest[0]), Path(args.selftest[1]))
            except Exception:
                log.critical("Self-test crashed:\n%s", traceback.format_exc())
                return 2
        return run_gui(smoke_test=args.smoke_test)
    except Exception:
        log.critical("Fatal error:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
