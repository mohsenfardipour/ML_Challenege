"""Remembers the last-used settings in data/settings.json."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields

from app.core.formats import BITRATES_KBPS, DEFAULT_BITRATE_KBPS, DEFAULT_FORMAT_KEY, OUTPUT_FORMATS
from app.core.stt_models import DEFAULT_MODEL_KEY, STT_MODELS
from app.paths import data_dir

log = logging.getLogger(__name__)


@dataclass
class Settings:
    output_format: str = DEFAULT_FORMAT_KEY
    bitrate_kbps: int = DEFAULT_BITRATE_KBPS
    use_custom_output_dir: bool = False
    custom_output_dir: str = ""
    last_input_dir: str = ""
    window_geometry: str = ""     # Qt window geometry, hex encoded
    last_tab: int = 0
    # speech-to-text
    stt_model: str = DEFAULT_MODEL_KEY
    cpu_threads: int = 4          # i3-1215U: 4 is a good balance; more threads gain little on E-cores
    model_dirs: dict = field(default_factory=dict)   # model key -> folder chosen with "Select local model folder"
    personalization_enabled: bool = True
    use_vocabulary_hints: bool = True   # give the vocabulary to the model as "hotwords" (rules are separate)
    export_srt: bool = False
    stt_last_dir: str = ""
    stt_output_dir: str = ""

    def validate(self) -> "Settings":
        if self.output_format not in {f.key for f in OUTPUT_FORMATS}:
            self.output_format = DEFAULT_FORMAT_KEY
        if self.bitrate_kbps not in BITRATES_KBPS:
            self.bitrate_kbps = DEFAULT_BITRATE_KBPS
        if self.stt_model not in {m.key for m in STT_MODELS}:
            self.stt_model = DEFAULT_MODEL_KEY
        if not isinstance(self.cpu_threads, int) or not 1 <= self.cpu_threads <= (os.cpu_count() or 8) * 2:
            self.cpu_threads = 4
        if not isinstance(self.model_dirs, dict):
            self.model_dirs = {}
        return self


def settings_file():
    return data_dir() / "settings.json"


def load_settings() -> Settings:
    path = settings_file()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Settings)}
        return Settings(**{k: v for k, v in raw.items() if k in known}).validate()
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError, TypeError) as exc:
        log.warning("Ignoring unreadable settings file %s: %s", path, exc)
        return Settings()


def save_settings(settings: Settings) -> None:
    path = settings_file()
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("Could not save settings to %s: %s", path, exc)
