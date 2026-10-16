"""Supported input extensions and output formats."""

from __future__ import annotations

from dataclasses import dataclass

INPUT_EXTENSIONS = (
    ".ogg", ".opus", ".oga", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".webm",
)

BITRATES_KBPS = (64, 96, 128, 160, 192, 256, 320)
DEFAULT_BITRATE_KBPS = 128


@dataclass(frozen=True)
class OutputFormat:
    key: str            # stored in settings
    label: str          # shown in the UI
    extension: str      # file extension of the result
    codec: str          # FFmpeg audio encoder
    muxer: str          # FFmpeg container format (-f)
    extra_args: tuple[str, ...] = ()


OUTPUT_FORMATS = (
    OutputFormat("mp3", "MP3", ".mp3", "libmp3lame", "mp3", ("-id3v2_version", "3")),
    OutputFormat("mp4", "MP4 (audio only, AAC)", ".mp4", "aac", "mp4", ("-movflags", "+faststart")),
    OutputFormat("m4a", "M4A (AAC)", ".m4a", "aac", "ipod", ("-movflags", "+faststart")),
)

DEFAULT_FORMAT_KEY = "mp3"


def get_format(key: str) -> OutputFormat:
    for fmt in OUTPUT_FORMATS:
        if fmt.key == key:
            return fmt
    raise ValueError(f"Unknown output format: {key}")


def is_supported_input(path: str) -> bool:
    return path.lower().endswith(INPUT_EXTENSIONS)


def file_dialog_filter() -> str:
    patterns = " ".join("*" + ext for ext in INPUT_EXTENSIONS)
    return f"Audio files ({patterns});;All files (*.*)"
