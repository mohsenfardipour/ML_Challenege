import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools.make_test_samples import main as make_samples  # noqa: E402


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    folder = tmp_path_factory.mktemp("samples")
    make_samples(str(folder))
    return folder


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    import app.paths
    monkeypatch.setattr(app.paths, "_data_dir", tmp_path / "data")
    (tmp_path / "data").mkdir()
