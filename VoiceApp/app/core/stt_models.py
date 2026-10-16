"""Speech-to-text model catalog and where the model files are on disk.

Models are faster-whisper (CTranslate2) folders stored in D:\\VoiceApp\\models\\<key>.
They are never bundled into the exe. A model folder is valid when it contains
model.bin, config.json, tokenizer.json and a vocabulary file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.paths import models_dir


@dataclass(frozen=True)
class SttModel:
    key: str            # also the folder name inside models/
    label: str          # shown in the UI
    hf_repo: str        # Hugging Face repository with the CTranslate2 files
    size_gb: float      # approximate download size


STT_MODELS = (
    SttModel("large-v3-turbo", "large-v3-turbo (recommended)",
             "mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1.6),
    SttModel("large-v3", "large-v3 (high accuracy, slow)",
             "Systran/faster-whisper-large-v3", 3.1),
)
DEFAULT_MODEL_KEY = "large-v3-turbo"

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
