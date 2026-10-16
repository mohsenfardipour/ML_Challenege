"""Drives the real converter tab headlessly (QT_QPA_PLATFORM=offscreen)."""

import time

import pytest
from PySide6.QtWidgets import QApplication

from app.settings import Settings
from app.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, condition, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if condition():
            return
        time.sleep(0.02)
    raise TimeoutError


def test_batch_conversion_through_gui(qapp, samples, tmp_path):
    settings = Settings(output_format="m4a", use_custom_output_dir=True, custom_output_dir=str(tmp_path))
    window = MainWindow(settings)
    tab = window.tabs.widget(0)
    tab.add_paths([str(samples), str(samples / "صدا.oga"), str(tmp_path / "x.txt")])
    count = len(list(samples.iterdir()))
    assert tab.table.rowCount() == count          # folder expanded, duplicate ignored
    assert tab.table.item(0, 0).text()             # names shown

    tab.start_conversion()
    assert tab.is_busy() and not tab.convert_btn.isEnabled()
    wait_until(qapp, lambda: not tab.is_busy())
    statuses = [tab.table.item(r, 1).text() for r in range(count)]
    assert statuses == ["Done"] * count, statuses
    assert tab.progress.value() == tab.progress.maximum()
    assert len(list(tmp_path.glob("*.m4a"))) == count
    window.close()


def test_cancel_through_gui(qapp, samples, tmp_path):
    settings = Settings(use_custom_output_dir=True, custom_output_dir=str(tmp_path))
    window = MainWindow(settings)
    tab = window.tabs.widget(0)
    tab.add_paths([str(samples)])
    tab.start_conversion()
    tab.cancel_conversion()
    wait_until(qapp, lambda: not tab.is_busy())
    statuses = [tab.table.item(r, 1).text() for r in range(tab.table.rowCount())]
    assert "Cancelled" in statuses
    assert not list(tmp_path.glob("*.part"))
    window.close()
