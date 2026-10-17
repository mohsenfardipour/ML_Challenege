"""Speech-to-text model catalog and where the model files are on disk.

Models are faster-whisper (CTranslate2 int8) folders in D:\\VoiceApp\\models\\<key>.
They are never bundled into the exe. They are published as zip files on this
project's GitHub Release (converted from Hugging Face by
.github/workflows/model-pipeline.yml), because Hugging Face is often blocked.

A model folder is valid when it contains model.bin, config.json,
tokenizer.json and a vocabulary file. It may also contain voiceapp_model.json
with model-specific settings written by the conversion pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.paths import models_dir

RELEASE_TAG = "voiceapp-2.0"
RELEASE_BASE_URL = f"https://github.com/mohsenfardipour/ML_Challenege/releases/download/{RELEASE_TAG}"


@dataclass(frozen=True)
class SttModel:
    key: str            # also the folder name inside models/
    label: str          # shown in the UI
    hf_repo: str        # original model on Hugging Face
    size_gb: float      # approximate download size
    speed_factor: float  # expected processing time / audio length on a low-power laptop CPU
    fallback_only: bool = False

    @property
    def zip_name(self) -> str:
        return f"{self.key}-ct2-int8.zip"

    @property
    def download_url(self) -> str:
        return f"{RELEASE_BASE_URL}/{self.zip_name}"


STT_MODELS = (
    SttModel("whisper-large-fa-v1", "Persian – fast (recommended)",
             "vhdm/whisper-large-fa-v1", 0.85, 0.6),
    # Requested as the "high accuracy" option, but in our FLEURS test it was less accurate
    # than the fast model (WER 44% / 32% vs 14.5% / 24%) and ~2.7x slower, so it is labelled honestly.
    SttModel("buzzasr-persian", "Persian – alternative (large-v3 based, slower)",
             "BuzzASR/persian", 1.6, 1.6),
    SttModel("large-v3-turbo", "Base large-v3-turbo (fallback, not Persian-tuned)",
             "openai/whisper-large-v3-turbo", 0.85, 0.6, fallback_only=True),
)
DEFAULT_MODEL_KEY = "whisper-large-fa-v1"
FALLBACK_MODEL_KEY = "large-v3-turbo"

REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json")


def get_model(key: str) -> SttModel:
    for model in STT_MODELS:
        if model.key == key:
            return model
    raise ValueError(f"Unknown model: {key}")


def default_model_path(key: str) -> Path:
    return models_dir() / key


def is_valid_model_dir(folder: str | os.PathLike | None) -> bool:
    if not folder:
        return False
    folder = Path(folder)
    if not folder.is_dir():
        return False
    if not all((folder / name).is_file() for name in REQUIRED_FILES):
        return False
    return any(folder.glob("vocabulary.*"))


def missing_files(folder: str | os.PathLike) -> list[str]:
    folder = Path(folder)
    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if not any(folder.glob("vocabulary.*")):
        missing.append("vocabulary.json (or .txt)")
    return missing


def find_model_dir(folder: str | os.PathLike) -> Path | None:
    """The user may pick the unzipped folder or its parent: look one level down too."""
    folder = Path(folder)
    if is_valid_model_dir(folder):
        return folder
    if folder.is_dir():
        for child in sorted(folder.iterdir()):
            if is_valid_model_dir(child):
                return child
    return None


def resolve_model_dir(key: str, custom_dirs: dict[str, str] | None = None) -> Path | None:
    """Folder chosen by the user for this model, else the standard models/<key> folder."""
    custom = (custom_dirs or {}).get(key)
    for candidate in ([Path(custom)] if custom else []) + [default_model_path(key)]:
        found = find_model_dir(candidate)
        if found:
            return found
    return None


def model_options(folder: str | os.PathLike) -> dict:
    """Extra settings written by the conversion pipeline (e.g. timestamps unsupported)."""
    try:
        return json.loads((Path(folder) / "voiceapp_model.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
