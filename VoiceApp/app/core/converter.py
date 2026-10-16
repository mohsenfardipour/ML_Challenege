"""Convert one audio file. Used by the GUI worker and by the --selftest mode.

Safety rules:
  * The source file is only ever read.
  * ffmpeg writes to a hidden temporary file first; only a complete result is
    renamed to the final name, and that name is always a new, unused one.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core import ffmpeg
from app.core.formats import OutputFormat
from app.core.naming import move_to_unique_name

log = logging.getLogger(__name__)

# The result may differ a little from the source (codec padding); more than this
# means something went wrong.
DURATION_TOLERANCE_SEC = 1.0
DURATION_TOLERANCE_RATIO = 0.02


@dataclass
class ConversionResult:
    ok: bool
    output_path: Path | None = None
    message: str = ""          # short human-readable text for the UI
    warning: str = ""          # set when ok but something looks odd
    cancelled: bool = False


def friendly_error(raw: str) -> str:
    """Turn ffmpeg's technical error output into a sentence a user understands."""
    text = raw.lower()
    rules = [
        ("no space left", "The disk is full. Free some space or choose another output folder."),
        ("permission denied", "Permission denied. The output folder may be read-only, or the file is open in another program."),
        ("matches no streams", "This file has no audio track."),
        ("does not contain any stream", "This file has no audio track."),
        ("invalid data found", "The file is damaged or is not a supported audio file."),
        ("could not find codec parameters", "The file is damaged or is not a supported audio file."),
        ("end of file", "The file is empty or cut off (incomplete download?)."),
        ("no such file", "The file could not be found. Was it moved or deleted?"),
        ("unknown encoder", "This FFmpeg build is missing a required encoder. Re-run setup.bat."),
    ]
    for needle, message in rules:
        if needle in text:
            return message
    last_line = raw.strip().splitlines()[-1] if raw.strip() else "unknown error"
    return f"Conversion failed: {last_line[:200]}"


def output_folder_for(src: Path, custom_folder: str | None) -> Path:
    return Path(custom_folder) if custom_folder else src.parent


def convert_file(src: Path, fmt: OutputFormat, bitrate_kbps: int,
                 out_folder: Path,
                 on_progress: Callable[[float], None] | None = None,
                 cancel_event: threading.Event | None = None) -> ConversionResult:
    src = Path(src)
    out_folder = Path(out_folder)
    log.info("Converting %s -> %s (%s, %d kbps)", src, out_folder, fmt.key, bitrate_kbps)

    if not src.is_file():
        return ConversionResult(False, message="The file could not be found. Was it moved or deleted?")
    if src.stat().st_size == 0:
        return ConversionResult(False, message="The file is empty (0 bytes).")
    try:
        out_folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("Cannot create output folder %s: %s", out_folder, exc)
        return ConversionResult(False, message=f"Cannot create the output folder: {out_folder}")

    try:
        info = ffmpeg.probe(str(src))
    except ffmpeg.FFmpegNotFound as exc:
        return ConversionResult(False, message=str(exc))
    except Exception as exc:  # noqa: BLE001 - report anything to the user
        log.exception("Probe failed for %s", src)
        return ConversionResult(False, message=f"Could not read the file: {exc}")
    if not info.has_audio:
        log.warning("No audio stream found in %s:\n%s", src, info.raw_output)
        raw = info.raw_output.lower()
        if "error opening input" in raw or "invalid data" in raw:
            return ConversionResult(False, message="The file is damaged or is not a supported audio file.")
        return ConversionResult(False, message="This file has no audio track, or it is not an audio file.")

    temp_path = out_folder / f".{src.stem}.{uuid.uuid4().hex[:8]}.voiceapp.part"
    try:
        cmd = ffmpeg.build_convert_command(
            str(src), str(temp_path), fmt.codec, fmt.muxer, bitrate_kbps, fmt.extra_args)
        ffmpeg.run_with_progress(cmd, info.duration, on_progress, cancel_event)
        final_path = move_to_unique_name(temp_path, out_folder, src.stem, fmt.extension)
    except ffmpeg.Cancelled:
        _remove_quietly(temp_path)
        log.info("Cancelled: %s", src)
        return ConversionResult(False, message="Cancelled", cancelled=True)
    except ffmpeg.FFmpegNotFound as exc:
        _remove_quietly(temp_path)
        return ConversionResult(False, message=str(exc))
    except RuntimeError as exc:
        _remove_quietly(temp_path)
        log.error("ffmpeg failed for %s:\n%s", src, exc)
        return ConversionResult(False, message=friendly_error(str(exc)))
    except OSError as exc:
        _remove_quietly(temp_path)
        log.exception("File error for %s", src)
        return ConversionResult(False, message=friendly_error(str(exc)))

    warning = _check_duration(info.duration, final_path)
    log.info("Done: %s%s", final_path, f" (warning: {warning})" if warning else "")
    return ConversionResult(True, final_path, "Done", warning)


def _check_duration(src_duration: float | None, output: Path) -> str:
    if not src_duration:
        return ""
    try:
        out_duration = ffmpeg.probe(str(output)).duration
    except Exception:  # noqa: BLE001
        return ""
    if out_duration is None:
        return ""
    diff = abs(out_duration - src_duration)
    if diff > max(DURATION_TOLERANCE_SEC, src_duration * DURATION_TOLERANCE_RATIO):
        return (f"length differs from the source ({out_duration:.1f}s vs {src_duration:.1f}s); "
                "the source may be damaged")
    return ""


def _remove_quietly(path: Path) -> None:
    try:
        if path.exists():
            os.remove(path)
    except OSError:
        log.warning("Could not delete temporary file %s", path)
