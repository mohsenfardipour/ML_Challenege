"""The "Personalize" tab: teach VoiceApp your words without training the model.

Sub-tabs:
  Samples          your voice messages + the exact correct text
  Vocabulary       special words/names given to the model as "hotwords"
  Correction rules wrong -> right replacements applied after transcription (you approve each)
  Test             compare models with/without personalization on your samples (WER/CER)
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QTextOption
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from app.core.formats import file_dialog_filter, is_supported_input
from app.core.personal import HOTWORDS_TOKEN_BUDGET, PersonalStore, Rule, build_hotwords, suggest_rules, suggest_vocabulary
from app.core.stt_models import get_model
from app.paths import personal_dir
from app.settings import Settings
from app.ui.stt_common import JobGuard, fallback_dir, format_duration, model_dir_for, selectable_models
from app.ui.stt_workers import EvaluationWorker, SampleAnalysisWorker
from app.ui.tabs.transcribe_tab import persian_font

log = logging.getLogger(__name__)


def rtl_editor(placeholder: str = "") -> QPlainTextEdit:
    editor = QPlainTextEdit()
    editor.setFont(persian_font(12))
    editor.setLayoutDirection(Qt.RightToLeft)
    option = QTextOption()
    option.setTextDirection(Qt.RightToLeft)
    option.setAlignment(Qt.AlignRight)
    option.setWrapMode(QTextOption.WordWrap)
    editor.document().setDefaultTextOption(option)
    editor.setPlaceholderText(placeholder)
    return editor


class SampleDialog(QDialog):
    """Add a sample (voice file + correct text) or edit the text of an existing one."""

    def __init__(self, parent, start_dir: str = "", audio: str = "", text: str = "", audio_fixed=False):
        super().__init__(parent)
        self.setWindowTitle("Personal sample")
        self.resize(640, 420)
        self.audio_edit = QLineEdit(audio)
        self.audio_edit.setReadOnly(True)
        browse = QPushButton("Choose voice file…")
        browse.setEnabled(not audio_fixed)
        browse.clicked.connect(lambda: self._browse(start_dir))
        self.text = rtl_editor("متن دقیق این پیام صوتی را اینجا بنویسید یا بچسبانید.")
        self.text.setPlainText(text)
        load_txt = QPushButton("Load text from TXT…")
        load_txt.clicked.connect(self._load_txt)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(self.audio_edit, 1)
        row.addWidget(browse)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Voice file (a copy is stored; your original is not changed):"))
        layout.addLayout(row)
        layout.addWidget(QLabel("The exact correct Persian text of this voice message:"))
        layout.addWidget(self.text, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(load_txt)
        bottom.addStretch()
        bottom.addWidget(buttons)
        layout.addLayout(bottom)

    def _browse(self, start_dir):
        path, _ = QFileDialog.getOpenFileName(self, "Choose voice file", start_dir, file_dialog_filter())
        if path:
            self.audio_edit.setText(path)
            txt = Path(path).with_suffix(".txt")
            if not self.text.toPlainText().strip() and txt.is_file():
                self.text.setPlainText(txt.read_text(encoding="utf-8-sig", errors="replace"))

    def _load_txt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose text file", "", "Text (*.txt)")
        if path:
            self.text.setPlainText(Path(path).read_text(encoding="utf-8-sig", errors="replace"))

    def _accept(self):
        if not self.audio_edit.text() or not Path(self.audio_edit.text()).is_file():
            QMessageBox.warning(self, "Voice file missing", "Choose the voice file for this sample.")
            return
        if not is_supported_input(self.audio_edit.text()):
            QMessageBox.warning(self, "Unsupported file", "This is not a supported audio file.")
            return
        if len(self.text.toPlainText().split()) < 2:
            QMessageBox.warning(self, "Text missing", "Type or paste the correct text of the voice message.")
            return
        self.accept()


RULE_COLS = ("Use", "Model wrote (wrong)", "Correct", "Seen", "Note")


class PersonalizeTab(QWidget):
    TAB_TITLE = "Personalize"

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.store = PersonalStore(personal_dir())
        self._thread: QThread | None = None
        self._worker = None
        self._eval_rows = []
        self._rules: list[Rule] = []
        self._build_ui()
        self.reload()

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        self.sub = QTabWidget()
        self.sub.addTab(self._samples_page(), "1. Samples")
        self.sub.addTab(self._vocab_page(), "2. Vocabulary")
        self.sub.addTab(self._rules_page(), "3. Correction rules")
        self.sub.addTab(self._test_page(), "4. Test")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.progress)
        bottom.addWidget(self.cancel_btn)
        intro = QLabel("VoiceApp cannot retrain the model on this laptop, but it can learn from your samples: "
                       "your special words are given to the model as hints, and mistakes it repeats are "
                       f"corrected automatically after you approve the rule. Data folder: {self.store.root}")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.sub, 1)
        layout.addLayout(bottom)

    def _samples_page(self):
        self.samples_table = QTableWidget(0, 4)
        self.samples_table.setHorizontalHeaderLabels(["Voice file", "Added", "Words", "Correct text"])
        h = self.samples_table.horizontalHeader()
        for c in range(3):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        self.samples_table.verticalHeader().setVisible(False)
        self.samples_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.samples_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.samples_table.doubleClicked.connect(self._edit_sample)
        add = QPushButton("Add sample…")
        edit = QPushButton("Edit text…")
        delete = QPushButton("Delete")
        play = QPushButton("Play voice")
        add.clicked.connect(self._add_sample)
        edit.clicked.connect(self._edit_sample)
        delete.clicked.connect(self._delete_sample)
        play.clicked.connect(self._play_sample)
        self.sample_buttons = [add, edit, delete]
        row = QHBoxLayout()
        for b in (add, edit, delete, play):
            row.addWidget(b)
        row.addStretch()
        hint = QLabel("Add 5–20 of your typical voice messages with their exact text. Tip: in the Transcribe tab, "
                      "correct a transcript and click “Save as personal sample”.")
        hint.setWordWrap(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addLayout(row)
        layout.addWidget(self.samples_table, 1)
        layout.addWidget(hint)
        return page

    def _vocab_page(self):
        self.vocab_edit = rtl_editor("هر کلمه یا نام در یک خط. مهم‌ترین‌ها اول.")
        self.vocab_edit.textChanged.connect(self._update_vocab_info)
        suggest = QPushButton("Suggest from my samples")
        save = QPushButton("Save vocabulary")
        suggest.clicked.connect(self._suggest_vocab)
        save.clicked.connect(self._save_vocab)
        self.vocab_info = QLabel()
        self.vocab_info.setWordWrap(True)
        self.hints_check = QCheckBox("Give these words to the model as hints while transcribing (experimental)")
        self.hints_check.setChecked(self.settings.use_vocabulary_hints)
        self.hints_check.setToolTip("Off by default: in our tests hints made the Persian model skip or garble "
                                    "sentences. Turn on only if “4. Test” shows they help with your messages.")
        self.hints_check.toggled.connect(lambda v: setattr(self.settings, "use_vocabulary_hints", v))
        row = QHBoxLayout()
        row.addWidget(suggest)
        row.addWidget(save)
        row.addStretch()
        row.addWidget(self.hints_check)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Names, places and technical terms you use. They are shown to the model before "
                                "each 30-second part as hints (faster-whisper “hotwords”). Only the first words "
                                f"that fit into about {HOTWORDS_TOKEN_BUDGET} tokens are used, so put the most "
                                "important ones first."))
        layout.addWidget(self.vocab_edit, 1)
        layout.addWidget(self.vocab_info)
        layout.addLayout(row)
        return page

    def _rules_page(self):
        self.rules_table = QTableWidget(0, len(RULE_COLS))
        self.rules_table.setHorizontalHeaderLabels(RULE_COLS)
        h = self.rules_table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.Stretch)
        self.rules_table.verticalHeader().setVisible(False)
        self.rules_table.setFont(persian_font(11))
        analyze = QPushButton("Analyze samples (suggest rules)")
        analyze.setToolTip("Transcribes your samples with the selected model and compares with your correct text.")
        add = QPushButton("Add rule")
        delete = QPushButton("Delete rule")
        approve = QPushButton("Approve all safe")
        save = QPushButton("Save rules")
        analyze.clicked.connect(self._analyze)
        add.clicked.connect(self._add_rule)
        delete.clicked.connect(self._delete_rule)
        approve.clicked.connect(self._approve_safe)
        save.clicked.connect(self._save_rules)
        self.analyze_btn = analyze
        row = QHBoxLayout()
        for b in (analyze, add, delete, approve, save):
            row.addWidget(b)
        row.addStretch()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("After every transcription, approved rules replace the wrong word/phrase with the "
                                "correct one – whole words only, never inside another word. Persian ي/ك and "
                                "half-spaces (می‌، ‌ها) are always normalized. Check each suggestion before "
                                "ticking it: a rule is applied everywhere."))
        layout.addWidget(self.rules_table, 1)
        layout.addLayout(row)
        return page

    def _test_page(self):
        self.test_table = QTableWidget(0, 7)
        self.test_table.setHorizontalHeaderLabels(
            ["Model", "WER plain", "WER personalized", "WER on new messages (est.)", "CER plain",
             "CER personalized", "Speed"])
        self.test_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.test_table.horizontalHeader().setStretchLastSection(True)
        self.test_table.verticalHeader().setVisible(False)
        self.test_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.test_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.test_btn = QPushButton("Test on my samples")
        self.test_btn.clicked.connect(self._run_test)
        self.best_btn = QPushButton("Use best model as default")
        self.best_btn.setEnabled(False)
        self.best_btn.clicked.connect(self._use_best)
        row = QHBoxLayout()
        row.addWidget(self.test_btn)
        row.addWidget(self.best_btn)
        row.addStretch()
        note = QLabel("Runs every installed Persian model on all your samples, without and with personalization. "
                      "WER = share of wrong words, CER = share of wrong letters (lower is better). Because the "
                      "vocabulary and rules were learned from these same samples, “personalized” is optimistic; "
                      "the “new messages” column (3+ samples) uses only rules learned from the other samples.")
        note.setWordWrap(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(note)
        layout.addLayout(row)
        layout.addWidget(self.test_table, 1)
        return page

    # ---------------------------------------------------------------- data
    def reload(self):
        samples = self.store.list_samples()
        self.samples_table.setRowCount(0)
        for s in samples:
            r = self.samples_table.rowCount()
            self.samples_table.insertRow(r)
            name = QTableWidgetItem(s.source_name)
            name.setToolTip(str(s.audio))
            self.samples_table.setItem(r, 0, name)
            self.samples_table.setItem(r, 1, QTableWidgetItem(s.added))
            self.samples_table.setItem(r, 2, QTableWidgetItem(str(len(s.transcript.split()))))
            text = QTableWidgetItem(s.transcript.replace("\n", " ")[:200])
            text.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.samples_table.setItem(r, 3, text)
        self._samples = samples
        self.vocab_edit.blockSignals(True)
        self.vocab_edit.setPlainText("\n".join(self.store.load_vocabulary()))
        self.vocab_edit.blockSignals(False)
        self._update_vocab_info()
        self._rules = self.store.load_rules()
        self._fill_rules()
        self.status.setText(f"{len(samples)} sample(s), {len(self.store.load_vocabulary())} vocabulary words, "
                            f"{sum(r.approved for r in self._rules)} approved rules.")

    def showEvent(self, event):
        if not self.is_busy():
            self.reload()
        super().showEvent(event)

    def _selected_sample(self):
        rows = self.samples_table.selectionModel().selectedRows()
        return self._samples[rows[0].row()] if rows else None

    # ---------------------------------------------------------------- samples
    def _add_sample(self):
        dlg = SampleDialog(self, self.settings.stt_last_dir)
        if dlg.exec() == QDialog.Accepted:
            try:
                self.store.add_sample(Path(dlg.audio_edit.text()), dlg.text.toPlainText())
            except OSError as exc:
                QMessageBox.warning(self, "Could not add sample", str(exc))
            self.reload()

    def _edit_sample(self):
        sample = self._selected_sample()
        if not sample or self.is_busy():
            return
        dlg = SampleDialog(self, audio=str(sample.audio), text=sample.transcript, audio_fixed=True)
        if dlg.exec() == QDialog.Accepted:
            self.store.update_transcript(sample, dlg.text.toPlainText())
            self.reload()

    def _delete_sample(self):
        sample = self._selected_sample()
        if not sample or self.is_busy():
            return
        if QMessageBox.question(self, "Delete sample", f"Delete the sample “{sample.source_name}”?\n"
                                "(Only the copy in the personal folder is deleted.)") == QMessageBox.Yes:
            self.store.delete_sample(sample)
            self.reload()

    def _play_sample(self):
        sample = self._selected_sample()
        if sample:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(sample.audio)))

    # ---------------------------------------------------------------- vocabulary
    def _vocab_words(self) -> list[str]:
        return [w.strip() for w in self.vocab_edit.toPlainText().splitlines() if w.strip() and not w.startswith("#")]

    def _token_counter(self):
        folder = model_dir_for(self.settings, self.settings.stt_model)
        if folder and (folder / "tokenizer.json").is_file():
            try:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
                return lambda s: len(tok.encode(" " + s, add_special_tokens=False).ids)
            except Exception:  # noqa: BLE001
                pass
        return None

    def _update_vocab_info(self):
        words = self._vocab_words()
        used = build_hotwords(words, self._token_counter())
        n_used = len(used.split()) if used else 0
        self.vocab_info.setText(f"{len(words)} words; the first {n_used} fit into the model hint. "
                                "Click “Save vocabulary” to use changes.")

    def _suggest_vocab(self):
        key = self.settings.stt_model
        pairs = [(s.transcript, s.model_output(key)) for s in self._samples]
        if not pairs:
            QMessageBox.information(self, "No samples", "Add some samples first (sub-tab 1).")
            return
        suggested = suggest_vocabulary(pairs)
        current = self._vocab_words()
        new = [w for w in suggested if w not in current]
        if not new:
            self.status.setText("No new words to suggest.")
            return
        self.vocab_edit.setPlainText("\n".join(current + new))
        analyzed = sum(1 for _, h in pairs if h)
        self.status.setText(f"Added {len(new)} suggested word(s) at the end – review, reorder, then Save. "
                            + ("" if analyzed else "Tip: run “Analyze samples” first so words the model gets "
                                                   "wrong are ranked first."))

    def _save_vocab(self):
        self.store.save_vocabulary(self._vocab_words())
        self.status.setText("Vocabulary saved.")

    # ---------------------------------------------------------------- rules
    def _fill_rules(self):
        t = self.rules_table
        t.blockSignals(True)
        t.setRowCount(0)
        for rule in self._rules:
            r = t.rowCount()
            t.insertRow(r)
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            use.setCheckState(Qt.Checked if rule.approved else Qt.Unchecked)
            t.setItem(r, 0, use)
            for c, value in ((1, rule.wrong), (2, rule.right)):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, item)
            seen = QTableWidgetItem(str(rule.count))
            seen.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            t.setItem(r, 3, seen)
            note = "manual" if rule.source == "manual" else (
                "" if rule.safe else "⚠ common word or also used correctly in your texts – risky")
            n = QTableWidgetItem(note)
            n.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            t.setItem(r, 4, n)
        t.blockSignals(False)

    def _rules_from_table(self) -> list[Rule]:
        rules = []
        for r in range(self.rules_table.rowCount()):
            wrong = self.rules_table.item(r, 1).text().strip()
            right = self.rules_table.item(r, 2).text().strip()
            if not wrong or wrong == right:
                continue
            old = self._rules[r] if r < len(self._rules) else Rule(wrong, right, source="manual")
            rules.append(Rule(wrong, right, self.rules_table.item(r, 0).checkState() == Qt.Checked,
                              old.count, old.safe, old.source if (old.wrong, old.right) == (wrong, right)
                              else "manual"))
        return rules

    def _add_rule(self):
        self._rules = self._rules_from_table() + [Rule("", "", True, 0, True, "manual")]
        self._fill_rules()
        self.rules_table.scrollToBottom()
        self.rules_table.editItem(self.rules_table.item(self.rules_table.rowCount() - 1, 1))

    def _delete_rule(self):
        rows = sorted({i.row() for i in self.rules_table.selectedIndexes()}, reverse=True)
        current = self._rules_from_table_keep_empty()
        for r in rows:
            if r < len(current):
                del current[r]
        self._rules = current
        self._fill_rules()

    def _rules_from_table_keep_empty(self) -> list[Rule]:
        keep = []
        for r in range(self.rules_table.rowCount()):
            old = self._rules[r] if r < len(self._rules) else Rule("", "")
            keep.append(Rule(self.rules_table.item(r, 1).text(), self.rules_table.item(r, 2).text(),
                             self.rules_table.item(r, 0).checkState() == Qt.Checked, old.count, old.safe, old.source))
        return keep

    def _approve_safe(self):
        for r, rule in enumerate(self._rules):
            if rule.safe and r < self.rules_table.rowCount():
                self.rules_table.item(r, 0).setCheckState(Qt.Checked)

    def _save_rules(self):
        self._rules = self._rules_from_table()
        self.store.save_rules(self._rules)
        self._fill_rules()
        self.status.setText(f"Saved {len(self._rules)} rules ({sum(r.approved for r in self._rules)} in use).")

    def _analyze(self):
        if not self._samples:
            QMessageBox.information(self, "No samples", "Add some samples first (sub-tab 1).")
            return
        key = self.settings.stt_model
        folder = model_dir_for(self.settings, key)
        if not folder:
            QMessageBox.information(self, "Model needed", "Install the selected model in the Transcribe tab first.")
            return
        audio = sum(_duration(s.audio) for s in self._samples if s.model_output(key) is None)
        self._start(SampleAnalysisWorker(self.store, key, folder, self.settings.cpu_threads,
                                         fallback_dir(self.settings, key)),
                    f"Analyzing samples with {get_model(key).label} "
                    f"(about {format_duration(audio * get_model(key).speed_factor + 20)})…",
                    self._analysis_done)

    def _analysis_done(self, ok: bool, message: str):
        self.status.setText(message)
        if not ok:
            return
        key = self.settings.stt_model
        pairs = [(s.transcript, s.model_output(key)) for s in self.store.list_samples() if s.model_output(key)]
        self._rules = suggest_rules(pairs, self._rules_from_table())
        self._fill_rules()
        self.status.setText(f"{message} {len(self._rules)} rule(s) in the list – tick the correct ones and "
                            "click “Save rules”. Then use “Suggest from my samples” in Vocabulary.")

    # ---------------------------------------------------------------- test
    def _run_test(self):
        if not self._samples:
            QMessageBox.information(self, "No samples", "Add some samples first (sub-tab 1).")
            return
        models = [(m.key, model_dir_for(self.settings, m.key)) for m in selectable_models() if not m.fallback_only]
        models = [(k, d) for k, d in models if d]
        if not models:
            QMessageBox.information(self, "Model needed", "Install at least one Persian model first.")
            return
        audio = sum(_duration(s.audio) for s in self._samples)
        estimate = sum(audio * 2 * get_model(k).speed_factor + 20 for k, _ in models)
        if QMessageBox.question(self, "Test on my samples",
                                f"This transcribes {len(self._samples)} sample(s) "
                                f"({format_duration(audio)} of audio) twice with {len(models)} model(s).\n"
                                f"Estimated time: about {format_duration(estimate)}. Start?") != QMessageBox.Yes:
            return
        self._start(EvaluationWorker(self.store, models, self.settings.cpu_threads,
                                     self.settings.use_vocabulary_hints),
                    "Testing…", self._test_done)

    def _test_done(self, ok: bool, message: str, rows):
        self.status.setText(message)
        self._eval_rows = rows
        t = self.test_table
        t.setRowCount(0)
        for row in rows:
            r = t.rowCount()
            t.insertRow(r)
            values = [get_model(row.model_key).label, f"{row.plain_wer:.1%}", f"{row.pers_wer:.1%}",
                      f"{row.loo_wer:.1%}" if row.loo_wer is not None else "needs 3+ samples",
                      f"{row.plain_cer:.1%}", f"{row.pers_cer:.1%}", f"{row.speed_ratio:.2f}× audio length"]
            for c, v in enumerate(values):
                t.setItem(r, c, QTableWidgetItem(v))
        self.best_btn.setEnabled(bool(rows))
        if rows:
            best = min(rows, key=lambda x: x.pers_wer)
            self.status.setText(f"{message} Lowest error: {get_model(best.model_key).label} "
                                f"({best.pers_wer:.1%} WER with personalization).")

    def _use_best(self):
        if not self._eval_rows:
            return
        best = min(self._eval_rows, key=lambda x: x.pers_wer)
        self.settings.stt_model = best.model_key
        self.status.setText(f"Default model is now: {get_model(best.model_key).label}")

    # ---------------------------------------------------------------- workers
    def is_busy(self) -> bool:
        return self._thread is not None

    def _start(self, worker, text: str, on_done):
        if self.is_busy() or not JobGuard.acquire("personalization", self):
            return
        self._thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.status.connect(self.status.setText)
        worker.progress.connect(lambda d, t: self.progress.setValue(int(d / max(t, 1) * 1000)))
        worker.finished.connect(on_done)
        worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker_done)
        self.status.setText(text)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.cancel_btn.setEnabled(True)
        for w in (self.analyze_btn, self.test_btn, *self.sample_buttons):
            w.setEnabled(False)
        self._thread.start()

    def _worker_done(self):
        self._thread.deleteLater()
        self._worker.deleteLater()
        self._thread = self._worker = None
        JobGuard.release()
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        for w in (self.analyze_btn, self.test_btn, *self.sample_buttons):
            w.setEnabled(True)

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status.setText("Cancelling…")

    def shutdown(self):
        if self._worker:
            self._worker.cancel()
        if self._thread:
            self._thread.quit()
            self._thread.wait(30_000)


def _duration(path: Path) -> float:
    from app.core import ffmpeg
    try:
        return ffmpeg.probe(str(path)).duration or 0.0
    except Exception:  # noqa: BLE001
        return 0.0
