"""Download a model zip from the project's GitHub Release and unzip it into models/.

This is the only network access in the whole app, and only when you click
"Download model". Interrupted downloads continue where they stopped.
"""

from __future__ import annotations

import logging
import shutil
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from app import APP_NAME, APP_VERSION
from app.core.stt_models import SttModel, find_model_dir
from app.paths import models_dir

log = logging.getLogger(__name__)

CHUNK = 1024 * 512


class DownloadCancelled(Exception):
    pass


class DownloadError(Exception):
    pass


def download_model(model: SttModel,
                   on_progress: Callable[[int, int, str], None] | None = None,
                   cancel_event: threading.Event | None = None,
                   url: str | None = None) -> Path:
    """Returns the model folder. on_progress(done_bytes, total_bytes, phase)."""
    url = url or model.download_url
    target_root = models_dir()
    target_root.mkdir(parents=True, exist_ok=True)
    temp_dir = target_root / ".download"
    temp_dir.mkdir(exist_ok=True)
    part = temp_dir / (model.zip_name + ".part")

    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"})
    if done:
        request.add_header("Range", f"bytes={done}-")
    log.info("Downloading %s (resume from %d bytes)", url, done)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:          # already complete
            response = None
        elif exc.code == 404:
            raise DownloadError("The model file was not found on GitHub (404). "
                                "Use “Select local model folder” and the manual steps in START_HERE.md.") from exc
        else:
            raise DownloadError(f"Download failed: HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DownloadError(f"Cannot connect to GitHub: {getattr(exc, 'reason', exc)}. "
                            "Check your internet/VPN, or install the model manually.") from exc

    if response is not None:
        with response:
            if done and response.status != 206:      # server ignored Range: start over
                done = 0
            mode = "ab" if done else "wb"
            length = int(response.headers.get("Content-Length") or 0)
            total = done + length if length else 0
            with open(part, mode) as out:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise DownloadCancelled()
                    try:
                        chunk = response.read(CHUNK)
                    except OSError as exc:
                        raise DownloadError(f"The connection was interrupted ({exc}). "
                                            "Click Download again to continue where it stopped.") from exc
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, total, "download")
            if total and done < total:
                raise DownloadError("The download stopped early. Click Download again to continue.")

    return install_zip(part, on_progress, cancel_event)


def install_zip(zip_path: Path,
                on_progress: Callable[[int, int, str], None] | None = None,
                cancel_event: threading.Event | None = None) -> Path:
    """Check and unzip a model zip into models/. Also used for a manually downloaded zip."""
    target_root = models_dir()
    staging = target_root / ".download" / "unpack"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = z.infolist()
            total = sum(m.file_size for m in members) or 1
            done = 0
            for m in members:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled()
                name = Path(m.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise DownloadError("The zip file contains unsafe paths.")
                z.extract(m, staging)       # CRC is checked while extracting
                done += m.file_size
                if on_progress:
                    on_progress(done, total, "unzip")
    except zipfile.BadZipFile as exc:
        zip_path.unlink(missing_ok=True)
        raise DownloadError("The downloaded file is damaged. Click Download again.") from exc
    found = find_model_dir(staging)
    if not found:
        raise DownloadError("The zip does not contain a valid model (model.bin, tokenizer.json …).")
    final = target_root / found.name
    if final.exists():
        backup = target_root / ".download" / (found.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        final.rename(backup)
    found.rename(final)
    shutil.rmtree(staging, ignore_errors=True)
    if zip_path.name.endswith(".part"):
        zip_path.unlink(missing_ok=True)
    log.info("Model installed in %s", final)
    return final
