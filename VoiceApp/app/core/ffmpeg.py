"""Thin wrapper around the ffmpeg command-line program.

Only ffmpeg.exe is needed at runtime (ffprobe is not bundled, to keep the app
small): the duration is read from the "Duration:" line ffmpeg prints.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable

from app.paths import IS_WINDOWS, find_tool

log = logging.getLogger(__name__)

# Hide the black console window that would pop up for every ffmpeg call.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Audio:")


class FFmpegNotFound(Exception):
    pass


class Cancelled(Exception):
    pass


def ffmpeg_path() -> str:
    path = find_tool("ffmpeg")
    if not path:
        raise FFmpegNotFound(
            "FFmpeg was not found. Put ffmpeg.exe in the 'ffmpeg' folder next to the app "
            "(run setup.bat to download it automatically)."
        )
    return path


@dataclass
class ProbeInfo:
    duration: float | None   # seconds, None if unknown
    has_audio: bool
    raw_output: str


def probe(path: str) -> ProbeInfo:
    """Read duration and check that the file contains an audio stream."""
    cmd = [ffmpeg_path(), "-hide_banner", "-nostdin", "-i", path]
    # ffmpeg exits with an error here because no output is given - that is expected.
    # stdin must be redirected too: the packaged exe has no console handles.
    proc = subprocess.run(
        cmd, capture_output=True, stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW, timeout=60,
    )
    text = proc.stderr.decode("utf-8", errors="replace")
    duration = None
    match = _DURATION_RE.search(text)
    if match:
        h, m, s = match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)
        if duration <= 0:
            duration = None
    return ProbeInfo(duration, bool(_AUDIO_STREAM_RE.search(text)), text)


def build_convert_command(src: str, dst_temp: str, codec: str, muxer: str,
                          bitrate_kbps: int, extra_args: tuple[str, ...] = ()) -> list[str]:
    return [
        ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "error",
        "-y",                      # dst_temp is our own temporary file, safe to replace
        "-i", src,
        "-map", "0:a:0",           # first audio track only
        "-vn", "-sn", "-dn",       # drop video / cover art / subtitles / data
        "-map_metadata", "0",
        "-c:a", codec, "-b:a", f"{bitrate_kbps}k",
        *extra_args,
        "-progress", "pipe:1", "-nostats",
        "-f", muxer, dst_temp,
    ]


def run_with_progress(cmd: list[str], duration: float | None,
                      on_progress: Callable[[float], None] | None,
                      cancel_event: threading.Event | None) -> None:
    """Run ffmpeg, reporting progress 0.0-1.0. Raises RuntimeError on failure."""
    log.debug("Running: %s", subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        creationflags=_NO_WINDOW,
    )
    stderr_chunks: list[bytes] = []
    # Read stderr in a separate thread so ffmpeg can never block on a full pipe.
    reader = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    reader.start()
    cancelled = False
    try:
        for raw in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            line = raw.decode("utf-8", errors="replace").strip()
            # out_time_us / out_time_ms are both microseconds in ffmpeg's output.
            if on_progress and duration and line.startswith(("out_time_us=", "out_time_ms=")):
                try:
                    seconds = int(line.split("=", 1)[1]) / 1_000_000
                    on_progress(max(0.0, min(1.0, seconds / duration)))
                except ValueError:
                    pass
    finally:
        if cancelled:
            proc.kill()
        proc.wait()
        reader.join(timeout=5)
    if cancelled:
        raise Cancelled()
    if proc.returncode != 0:
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"ffmpeg exited with code {proc.returncode}")
    if on_progress:
        on_progress(1.0)
