"""Self-test without the GUI: converts sample files to every output format and
checks that each result is a valid audio file with the same length as the source.

Used to verify the packaged exe (which has no console), so the report is written
to a text file and the exit code is 0 (all passed) or 1 (something failed).
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core import ffmpeg
from app.core.converter import convert_file
from app.core.formats import DEFAULT_BITRATE_KBPS, OUTPUT_FORMATS, is_supported_input

log = logging.getLogger(__name__)


def run_selftest(input_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "selftest_report.txt"
    sources = sorted(p for p in input_dir.iterdir() if p.is_file() and is_supported_input(p.name))
    lines = [f"FFmpeg: {ffmpeg.ffmpeg_path()}", f"Input: {input_dir}", ""]
    failures = 0
    if not sources:
        lines.append("FAIL: no supported audio files found in the input folder")
        failures += 1
    for src in sources:
        before = (src.stat().st_size, src.stat().st_mtime)
        src_dur = ffmpeg.probe(str(src)).duration
        for fmt in OUTPUT_FORMATS:
            result = convert_file(src, fmt, DEFAULT_BITRATE_KBPS, output_dir)
            if not result.ok:
                failures += 1
                lines.append(f"FAIL {src.name} -> {fmt.key}: {result.message}")
                continue
            info = ffmpeg.probe(str(result.output_path))
            ok = info.has_audio and info.duration is not None and src_dur is not None \
                and abs(info.duration - src_dur) <= max(1.0, src_dur * 0.02) and not result.warning
            failures += 0 if ok else 1
            lines.append(f"{'PASS' if ok else 'FAIL'} {src.name} -> {result.output_path.name}: "
                         f"source {src_dur}s, output {info.duration}s {result.warning}")
        if (src.stat().st_size, src.stat().st_mtime) != before:
            failures += 1
            lines.append(f"FAIL {src.name}: the source file was modified!")
    leftovers = list(output_dir.glob("*.voiceapp.part"))
    if leftovers:
        failures += 1
        lines.append(f"FAIL temporary files left behind: {leftovers}")
    lines += ["", "RESULT: " + ("ALL PASSED" if failures == 0 else f"{failures} FAILURE(S)")]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Self-test finished, report: %s\n%s", report, "\n".join(lines))
    return 0 if failures == 0 else 1


def run_stt_selftest(model_dir: Path, audio: Path, output_dir: Path, reference: Path | None = None) -> int:
    """Transcribe one file with a real model, export TXT/DOCX/SRT, and score it against a reference.

    Used to check the packaged exe on Windows: the model loads, Persian text comes out,
    and personalization (correction rules) and the exporters work.
    """
    from app.core.exporters import export_docx, export_srt, export_txt
    from app.core.personal import PersonalStore, suggest_rules
    from app.core.text_align import score
    from app.core.transcriber import Transcriber

    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "stt_selftest_report.txt"
    lines = [f"Model: {model_dir}", f"Audio: {audio}"]
    failures = 0
    log.info("STT self-test: loading %s", model_dir)
    t = Transcriber(model_dir, 4)
    log.info("STT self-test: model loaded, transcribing %s", audio)
    lines.append(f"Model loaded in {t.load_seconds:.1f}s")
    result = t.transcribe(audio)
    speed = result.processing_seconds / max(result.audio_seconds, 0.1)
    lines += [f"Audio {result.audio_seconds:.1f}s, processing {result.processing_seconds:.1f}s, "
              f"speed ratio {speed:.2f}", "Transcript:", result.text]
    persian_letters = sum("\u0600" <= c <= "\u06ff" for c in result.text)
    if persian_letters < 10:
        failures += 1
        lines.append("FAIL: no Persian text recognized")
    for kind, func in (("txt", lambda p: export_txt(p, result.text)),
                       ("docx", lambda p: export_docx(p, result.text, title=audio.stem)),
                       ("srt", lambda p: export_srt(p, result.segments))):
        path = output_dir / f"transcript.{kind}"
        func(path)
        if not path.is_file() or path.stat().st_size == 0:
            failures += 1
            lines.append(f"FAIL: {kind} export")
    if reference and reference.is_file():
        ref = reference.read_text(encoding="utf-8-sig")
        plain = score(ref, result.text)
        lines.append(f"WER {plain.wer:.1%}  CER {plain.cer:.1%}")
        # personalization: learn rules from this sample, apply them
        store = PersonalStore(output_dir / "personal")
        sample = store.add_sample(audio, ref)
        rules = suggest_rules([(sample.transcript, result.raw_text)])
        for r in rules:
            r.approved = r.safe
        store.save_rules(rules)
        fixed = store.personalization().apply(result.raw_text)
        pers = score(ref, fixed)
        lines.append(f"With {sum(r.approved for r in rules)} learned correction rules: WER {pers.wer:.1%}  "
                     f"CER {pers.cer:.1%}")
        if plain.cer > 0.6:
            failures += 1
            lines.append("FAIL: transcript is very different from the reference")
    lines.append("RESULT: " + ("ALL PASSED" if failures == 0 else f"{failures} FAILURE(S)"))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("STT self-test finished:\n%s", "\n".join(lines))
    return 0 if failures == 0 else 1
