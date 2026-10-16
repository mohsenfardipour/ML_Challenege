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
