"""Where things live on disk: the app folder, the data folder, and FFmpeg.

All settings and logs are stored in a "data" folder next to the app
(e.g. D:\\VoiceApp\\data), so nothing is written to drive C: unless the app
folder itself is read-only (then %LOCALAPPDATA%\\VoiceApp is used).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""


def is_frozen() -> bool:
    """True when running as the packaged PyInstaller exe."""
    return getattr(sys, "frozen", False)


def app_dir() -> Path:
    """Folder that contains VoiceApp.exe (packaged) or the project root (dev)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _is_writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


_data_dir: Path | None = None


def data_dir() -> Path:
    """Folder for settings.json and logs/."""
    global _data_dir
    if _data_dir is None:
        override = os.environ.get("VOICEAPP_DATA_DIR")
        candidates = [Path(override)] if override else []
        candidates.append(app_dir() / "data")
        local = os.environ.get("LOCALAPPDATA")
        candidates.append(Path(local) / "VoiceApp" if local else Path.home() / ".voiceapp")
        for folder in candidates:
            if _is_writable(folder):
                _data_dir = folder
                break
        else:
            _data_dir = candidates[-1]
    return _data_dir


def _ffmpeg_search_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("VOICEAPP_FFMPEG_DIR"):
        dirs.append(Path(os.environ["VOICEAPP_FFMPEG_DIR"]))
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        dirs.append(Path(sys._MEIPASS) / "ffmpeg")  # PyInstaller's _internal folder
    dirs.append(app_dir() / "ffmpeg")
    return dirs


def find_tool(name: str) -> str | None:
    """Find 'ffmpeg' or 'ffprobe'. Bundled copy first, then the system PATH."""
    for folder in _ffmpeg_search_dirs():
        candidate = folder / (name + EXE_SUFFIX)
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)
