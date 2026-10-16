"""Log file for troubleshooting: data/logs/voiceapp.log (rotated, max ~3 MB)."""

from __future__ import annotations

import logging
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app import APP_NAME, APP_VERSION
from app.paths import data_dir


def log_file() -> Path:
    return data_dir() / "logs" / "voiceapp.log"


def setup_logging() -> Path:
    path = log_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if sys.stderr is not None:   # no console in the packaged exe
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(console)
    logging.getLogger(__name__).info(
        "%s %s starting (Python %s, %s)", APP_NAME, APP_VERSION,
        platform.python_version(), platform.platform())
    return path
