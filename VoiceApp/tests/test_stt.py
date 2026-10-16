"""Speech-to-text features, tested with a fake model (the real models are 1-2 GB).

A real end-to-end run with the downloaded models is done by
tools/model_pipeline/evaluate.py (CI) and tests/test_stt_real.py (if a model is installed).
"""

import time
import zipfile

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.core import ffmpeg
from app.core.exporters import export_docx, export_srt, export_txt
from app.core.persian_text import clean_transcript, normalize_for_scoring
from app.core.personal import PersonalStore, Personalization, Rule, apply_rules, build_hotwords, suggest_rules, suggest_vocabulary
from app.core.stt_models import find_model_dir, get_model, resolve_model_dir
from app.core.text_align import dropped_sentences, score
from app.core.transcriber import Segment, TranscriptResult

REF = "مردم از روزگار باستان عناصر شیمیایی بنیادی مثل طلا، نقره، و مس را می‌شناختند."
HYP = "مردم از روزگار باستان اناثر شیمیایی بنیادی مثل تلا، نقره و مصر را می شناختند"


# ---------------------------------------------------------------- text
def test_normalization():
    assert clean_transcript("كتاب ها را مي خوانم") == "کتاب‌ها را می‌خوانم"
    assert clean_transcript("نمی دانم") == "نمی‌دانم"
    assert clean_transcript("میز") == "میز"                       # not a می- prefix
    assert normalize_for_scoring("سَلام، دنيا!") == "سلام دنیا"


def test_score_and_dropped():
    s = score(REF, HYP)
    assert abs(s.wer - 3 / 15) < 1e-9 and s.cer < s.wer          # اناثر، تلا، مصر wrong
    assert score(REF, REF).wer == 0
    assert dropped_sentences(["جمله اول اینجاست", "جمله دوم کاملا جا افتاده"], "جمله اول اینجاست") == [1]


def test_rules_whole_words_only():
    rules = [Rule("تلا", "طلا", True), Rule("بینال ملالی", "بین‌المللی", True), Rule("x", "y", False)]
    out = apply_rules("تلا و تلاش و بینال ملالی و بینال‌ملالی x", rules)
    assert out == "طلا و تلاش و بین‌المللی و بین‌المللی x"


def test_rule_suggestions_and_safety():
    rules = suggest_rules([(REF, HYP), ("مصر کشور بزرگی است", "مصر کشور بزرگی است")])
    by_wrong = {r.wrong: r for r in rules}
    assert by_wrong["اناثر"].right == "عناصر" and by_wrong["اناثر"].safe
    assert by_wrong["تلا"].right == "طلا"
    assert not by_wrong["مصر"].safe                  # مصر (Egypt) is used correctly in another sample
    assert not any(r.approved for r in rules)        # the user approves, never automatic
    for r in rules:
        r.approved = r.safe
    fixed = Personalization(None, rules).apply(HYP)
    assert score(REF, fixed).wer < score(REF, HYP).wer


def test_vocabulary_and_hotword_budget():
    words = suggest_vocabulary([(REF, HYP)])
    assert words[0] in ("عناصر", "طلا")
    many = [f"کلمه{i}" for i in range(200)]
    hot = build_hotwords(many, lambda s: len(s.split()) * 2, budget=80)
    assert len(hot.split()) == 40


def test_personal_store(tmp_path, samples):
    store = PersonalStore(tmp_path / "personal")
    src = samples / "پیام صوتی ۱.ogg"
    before = src.read_bytes()
    s = store.add_sample(src, "  سلام دنیا  ")
    assert s.audio.read_bytes() == before and s.audio != src
    assert store.list_samples()[0].transcript.strip() == "سلام دنیا"
    store.update_transcript(s, "سلام دنیای خوب")
    s.save_model_output("m", "سلام دنیا")
    assert store.list_samples()[0].model_output("m") == "سلام دنیا"
    store.save_vocabulary(["طلا", "عناصر"])
    store.save_rules([Rule("تلا", "طلا", True)])
    p = store.personalization()
    assert p.hotwords == "طلا عناصر" and len(p.rules) == 1
    store.delete_sample(s)
    assert store.list_samples() == [] and src.read_bytes() == before


# ---------------------------------------------------------------- files
def test_exporters(tmp_path):
    export_txt(tmp_path / "a.txt", "سلام")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8-sig") == "سلام\n"
    export_srt(tmp_path / "a.srt", [Segment(0, 1.5, "سلام"), Segment(61.2, 65, "دنیا")])
    srt = (tmp_path / "a.srt").read_text(encoding="utf-8-sig")
    assert "00:01:01,200 --> 00:01:05,000" in srt and "دنیا" in srt
    export_docx(tmp_path / "a.docx", "سلام دنیا\nخط دوم", title="عنوان")
    xml = zipfile.ZipFile(tmp_path / "a.docx").read("word/document.xml").decode()
    assert "<w:bidi/>" in xml and "<w:rtl/>" in xml and 'w:cs="Tahoma"' in xml and "خط دوم" in xml


def test_decode_pcm(samples):
    pcm = ffmpeg.decode_to_pcm16k(str(samples / "پیام صوتی ۱.ogg"))
    assert abs(len(pcm) / 16000 - 3.0) < 0.1 and pcm.dtype.name == "float32"


def fake_model_folder(folder):
    folder.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
        (folder / name).write_text("{}")
    return folder


def test_model_folder_detection_and_zip_install(tmp_path, monkeypatch):
    import app.core.model_download as md
    import app.core.stt_models as sm
    monkeypatch.setattr(sm, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(md, "models_dir", lambda: tmp_path / "models")
    key = "whisper-large-fa-v1"
    src = fake_model_folder(tmp_path / "build" / key)
    assert find_model_dir(tmp_path / "build") == src             # parent folder chosen: found inside
    assert resolve_model_dir(key) is None
    assert resolve_model_dir(key, {key: str(tmp_path / "build")}) == src
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in src.iterdir():
            zf.write(f, f"{key}/{f.name}")
    installed = md.install_zip(z)
    assert installed == tmp_path / "models" / key and resolve_model_dir(key) == installed
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(md.DownloadError):
        md.install_zip(bad)


def test_download_resume_and_errors(tmp_path, monkeypatch):
    """Serve a model zip over local HTTP; interrupt once, then resume."""
    import http.server
    import threading

    import app.core.model_download as md
    import app.core.stt_models as sm
    monkeypatch.setattr(sm, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(md, "models_dir", lambda: tmp_path / "models")
    key = "whisper-large-fa-v1"
    src = fake_model_folder(tmp_path / "srv" / key)
    (src / "model.bin").write_bytes(b"x" * 300_000)
    z = tmp_path / "srv" / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in src.iterdir():
            zf.write(f, f"{key}/{f.name}")
    data = z.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/missing.zip":
                self.send_error(404)
                return
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].split("-")[0]) if rng else 0
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    model = get_model(key)
    part = tmp_path / "models" / ".download" / (model.zip_name + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:1000])                     # an earlier, interrupted download
    progress = []
    folder = md.download_model(model, lambda d, t, p: progress.append((d, t, p)), url=f"{base}/m.zip")
    assert (folder / "model.bin").stat().st_size == 300_000
    assert progress[0][0] > 1000 and progress[0][1] == len(data)
    with pytest.raises(md.DownloadError, match="404"):
        md.download_model(model, url=f"{base}/missing.zip")
    server.shutdown()


# ---------------------------------------------------------------- GUI with a fake model
class FakeTranscriber:
    """Stands in for faster-whisper: returns a fixed text after a short delay."""
    load_seconds = 0.1

    def __init__(self, model_dir, threads):
        self.model_dir = model_dir

    def count_tokens(self, text):
        return len(text.split()) * 2

    def transcribe(self, audio, personalization=None, on_progress=None, cancel_event=None):
        for i in range(3):
            time.sleep(0.05)
            if on_progress:
                on_progress((i + 1) / 3, i + 1.0)
        raw = clean_transcript(HYP)
        text = personalization.apply(raw) if personalization else raw
        return TranscriptResult([Segment(0, 3, text)], raw, text, 3.0, 0.15, personalization is not None)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(qapp, cond, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return
        time.sleep(0.02)
    raise TimeoutError


@pytest.fixture
def gui_env(tmp_path, monkeypatch, samples):
    import app.paths
    import app.ui.stt_workers as workers
    monkeypatch.setenv("VOICEAPP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("VOICEAPP_PERSONAL_DIR", str(tmp_path / "personal"))
    fake_model_folder(tmp_path / "models" / "whisper-large-fa-v1")
    monkeypatch.setattr(workers, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    assert app.paths.personal_dir() == tmp_path / "personal"
    return tmp_path


def test_transcribe_tab_end_to_end(qapp, gui_env, samples):
    from app.core.personal import PersonalStore
    from app.settings import Settings
    from app.ui.main_window import MainWindow
    from app.ui.tabs.transcribe_tab import TranscribeTab

    window = MainWindow(Settings())
    tab = next(window.tabs.widget(i) for i in range(window.tabs.count())
               if isinstance(window.tabs.widget(i), TranscribeTab))
    tab.add_paths([str(samples / "پیام صوتی ۱.ogg"), str(samples / "صدا.oga")])
    assert len(tab.rows) == 2 and tab.rows[0].duration > 2
    assert "estimated time" in tab.time_label.text()
    assert tab.transcribe_btn.isEnabled()
    tab.start()
    wait_until(qapp, lambda: not tab.is_busy())
    assert all(r.status.startswith("Done") for r in tab.rows)
    tab.table.selectRow(0)
    assert "اناثر" in tab.editor.toPlainText()
    tab.editor.setPlainText(REF)                      # the user corrects the text
    assert tab.rows[0].text == REF
    out = gui_env / "out"
    out.mkdir()
    from PySide6.QtWidgets import QFileDialog
    import unittest.mock as um
    with um.patch.object(QFileDialog, "getExistingDirectory", return_value=str(out)):
        tab.srt_check.setChecked(True)
        tab._save_all()
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(["پیام صوتی ۱.txt", "پیام صوتی ۱.docx", "پیام صوتی ۱.srt", "صدا.txt", "صدا.docx", "صدا.srt"])
    tab._save_as_sample()
    store = PersonalStore(gui_env / "personal")
    assert len(store.list_samples()) == 1 and store.list_samples()[0].transcript.strip() == REF
    window.close()


def test_personalize_tab_flow(qapp, gui_env, samples):
    from app.core.personal import PersonalStore
    from app.settings import Settings
    from app.ui.tabs.personalize_tab import PersonalizeTab

    store = PersonalStore(gui_env / "personal")
    for name in ("پیام صوتی ۱.ogg", "صدا.oga", "ضبط.wav"):
        store.add_sample(samples / name, REF)
    settings = Settings()
    tab = PersonalizeTab(settings)
    assert tab.samples_table.rowCount() == 3
    tab._analyze()
    wait_until(qapp, lambda: not tab.is_busy())
    wrongs = {tab.rules_table.item(r, 1).text() for r in range(tab.rules_table.rowCount())}
    assert {"اناثر", "تلا"} <= wrongs
    tab._approve_safe()
    tab._save_rules()
    assert any(r.approved and r.wrong == "اناثر" for r in store.load_rules())
    tab._suggest_vocab()
    tab._save_vocab()
    assert "عناصر" in store.load_vocabulary()
    tab._run_test()
    wait_until(qapp, lambda: not tab.is_busy())
    assert tab.test_table.rowCount() == 1
    row = tab._eval_rows[0]
    assert row.pers_wer < row.plain_wer and row.loo_wer is not None
    tab._use_best()
    assert settings.stt_model == "whisper-large-fa-v1"
