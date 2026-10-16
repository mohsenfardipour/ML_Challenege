"""Output file naming. Existing files are never overwritten."""

from __future__ import annotations

import os
from pathlib import Path


def candidate_paths(folder: Path, stem: str, extension: str):
    """Yield name.ext, name_1.ext, name_2.ext, ..."""
    yield folder / f"{stem}{extension}"
    n = 1
    while True:
        yield folder / f"{stem}_{n}{extension}"
        n += 1


def unique_output_path(folder: Path, stem: str, extension: str) -> Path:
    """First candidate name that does not exist yet."""
    for path in candidate_paths(folder, stem, extension):
        if not path.exists():
            return path
    raise AssertionError("unreachable")


def move_to_unique_name(temp_file: Path, folder: Path, stem: str, extension: str) -> Path:
    """Rename the finished temp file to a free name and return that name.

    On Windows os.rename refuses to replace an existing file, so even if another
    program creates the same name at the same moment we simply try the next one.
    """
    for path in candidate_paths(folder, stem, extension):
        if path.exists():
            continue
        try:
            os.rename(temp_file, path)
            return path
        except FileExistsError:
            continue
    raise AssertionError("unreachable")
