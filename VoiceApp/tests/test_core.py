import threading

import pytest

from app.core import ffmpeg
from app.core.converter import convert_file, friendly_error
from app.core.formats import OUTPUT_FORMATS, get_format
from app.core.naming import move_to_unique_name, unique_output_path
from app.settings import Settings, load_settings, save_settings


def test_unique_name_never_overwrites(tmp_path):
    (tmp_path / "a.mp3").write_text("x")
    (tmp_path / "a_1.mp3").write_text("x")
    assert unique_output_path(tmp_path, "a", ".mp3").name == "a_2.mp3"
    tmp = tmp_path / "t.part"
    tmp.write_text("new")
    final = move_to_unique_name(tmp, tmp_path, "a", ".mp3")
    assert final.name == "a_2.mp3"
    assert (tmp_path / "a.mp3").read_text() == "x"


@pytest.mark.parametrize("fmt", OUTPUT_FORMATS, ids=lambda f: f.key)
def test_convert_persian_ogg_all_formats(samples, tmp_path, fmt):
    src = samples / "پیام صوتی ۱.ogg"
    before = src.read_bytes()
    progress = []
    result = convert_file(src, fmt, 128, tmp_path, on_progress=progress.append)
    assert result.ok, result.message
    assert result.output_path.name == "پیام صوتی ۱" + fmt.extension
    assert not result.warning
    info = ffmpeg.probe(str(result.output_path))
    assert info.has_audio and abs(info.duration - 3.0) < 0.2
    assert progress[-1] == 1.0
    assert src.read_bytes() == before
    assert not list(tmp_path.glob("*.part"))


def test_same_folder_output_does_not_touch_source(samples, tmp_path):
    src = tmp_path / "آهنگ.mp3"
    src.write_bytes((samples / "آهنگ.mp3").read_bytes())
    result = convert_file(src, get_format("mp3"), 128, tmp_path)
    assert result.ok
    assert result.output_path.name == "آهنگ_1.mp3"
    assert src.read_bytes() == (samples / "آهنگ.mp3").read_bytes()


def test_bitrate_is_applied(samples, tmp_path):
    result = convert_file(samples / "ضبط.wav", get_format("mp3"), 64, tmp_path)
    assert result.ok
    size_kbps = result.output_path.stat().st_size * 8 / 1000 / 7.0
    assert 55 < size_kbps < 75


def test_corrupt_and_empty_files_give_readable_errors(tmp_path):
    bad = tmp_path / "bad.ogg"
    bad.write_bytes(b"this is not audio" * 100)
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")
    r1 = convert_file(bad, get_format("mp3"), 128, tmp_path / "out")
    r2 = convert_file(empty, get_format("mp3"), 128, tmp_path / "out")
    r3 = convert_file(tmp_path / "missing.mp3", get_format("mp3"), 128, tmp_path / "out")
    assert not r1.ok and "damaged" in r1.message
    assert not r2.ok and "empty" in r2.message
    assert not r3.ok and "could not be found" in r3.message
    assert not list((tmp_path / "out").glob("*"))


def test_cancel_removes_temp_file(samples, tmp_path):
    cancel = threading.Event()
    cancel.set()
    out = tmp_path / "out"
    result = convert_file(samples / "sample.flac", get_format("m4a"), 128, out, cancel_event=cancel)
    assert result.cancelled and not result.ok
    assert not list(out.iterdir())


def test_friendly_error_fallback():
    assert friendly_error("Permission denied") .startswith("Permission denied")
    assert friendly_error("weird thing\nlast line").endswith("last line")


def test_settings_roundtrip():
    s = Settings(output_format="m4a", bitrate_kbps=192, use_custom_output_dir=True,
                 custom_output_dir="D:/خروجی")
    save_settings(s)
    assert load_settings() == s


def test_bad_settings_fall_back_to_defaults():
    from app.settings import settings_file
    settings_file().write_text('{"output_format": "xyz", "bitrate_kbps": 7, "unknown": 1}')
    s = load_settings()
    assert s.output_format == "mp3" and s.bitrate_kbps == 128
    settings_file().write_text("not json")
    assert load_settings() == Settings()
