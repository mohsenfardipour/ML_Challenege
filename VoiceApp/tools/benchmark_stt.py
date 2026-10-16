"""Speech-to-text feasibility benchmark (Phase 2, step 1).

Transcribes audio files with faster-whisper on the CPU (int8, language="fa",
VAD on) for every combination of model and cpu_threads, and writes a report
with speed, RAM use and the Persian transcripts.

    python tools/benchmark_stt.py                      (first 2 files in the samples folder)
    python tools/benchmark_stt.py D:\\VoiceApp\\samples\\voice.ogg
    python tools/benchmark_stt.py FILE [FILE ...] --models large-v3-turbo large-v3 --threads 4 6 8
    python tools/benchmark_stt.py FILE --reference reference.txt     (also computes WER/CER)

Models are downloaded on first use into the models folder (D:\\VoiceApp\\models).
Each run happens in a fresh Python process so RAM figures don't add up and every
run starts from the same state. Close other programs while it runs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.stt_models import STT_MODELS, default_model_path, get_model, is_valid_model_dir  # noqa: E402
from app.paths import models_dir  # noqa: E402

# Keep Hugging Face's own cache next to the models, never on drive C:.
os.environ.setdefault("HF_HOME", str(models_dir() / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ---------------------------------------------------------------- child process
def run_child(model_dir: str, threads: int, audio: str) -> None:
    """Load the model, transcribe one file, print one JSON line with the results."""
    import psutil
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    proc = psutil.Process()
    peak = {"rss": proc.memory_info().rss}
    stop = threading.Event()

    def sample_ram():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            time.sleep(0.1)

    sampler = threading.Thread(target=sample_ram, daemon=True)
    sampler.start()

    samples = decode_audio(audio, sampling_rate=16000)
    duration = len(samples) / 16000

    t0 = time.perf_counter()
    model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=threads)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        samples, language="fa", task="transcribe", vad_filter=True, beam_size=5)
    seg_list = [(s.start, s.end, s.text.strip()) for s in segments]   # work happens here
    transcribe_s = time.perf_counter() - t0

    stop.set()
    sampler.join()
    mem = proc.memory_info()
    peak_rss = max(peak["rss"], getattr(mem, "peak_wset", 0))   # peak_wset: Windows only
    print("RESULT_JSON " + json.dumps({
        "audio_seconds": duration,
        "speech_seconds_after_vad": getattr(info, "duration_after_vad", None),
        "load_seconds": load_s,
        "transcribe_seconds": transcribe_s,
        "speed_ratio": transcribe_s / duration if duration else None,
        "peak_ram_mb": peak_rss / 1_048_576,
        "segments": seg_list,
        "text": " ".join(t for _, _, t in seg_list),
    }, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- WER / CER
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "\u200c": " "})
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_PUNCT = re.compile(r"[^\w\s]|_")


def normalize_fa(text: str) -> str:
    text = _DIACRITICS.sub("", text.translate(_ARABIC_TO_PERSIAN))
    return " ".join(_PUNCT.sub(" ", text).split())


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    ref, hyp = normalize_fa(reference), normalize_fa(hypothesis)
    wer = edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))
    ref_c, hyp_c = ref.replace(" ", ""), hyp.replace(" ", "")
    cer = edit_distance(list(ref_c), list(hyp_c)) / max(1, len(ref_c))
    return wer, cer


# ---------------------------------------------------------------- parent
def ensure_model(key: str) -> Path:
    folder = default_model_path(key)
    if is_valid_model_dir(folder):
        return folder
    from faster_whisper.utils import download_model
    model = get_model(key)
    print(f"Downloading {key} (~{model.size_gb} GB) to {folder} ...", flush=True)
    download_model(model.hf_repo, output_dir=str(folder))
    if not is_valid_model_dir(folder):
        raise SystemExit(f"Download of {key} looks incomplete: {folder}")
    return folder


def expand_audio_args(items: list[str], max_files: int) -> list[str]:
    from app.core.formats import is_supported_input
    files = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.is_file() and is_supported_input(p.name))
            files += [str(p) for p in found[:max_files]]
        elif path.is_file():
            files.append(str(path))
        else:
            print(f"Skipping missing file: {item}")
    return files


def cpu_description() -> str:
    name = platform.processor() or platform.machine()
    if sys.platform.startswith("win"):
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20).stdout.strip()
            name = out or name
        except Exception:  # noqa: BLE001
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    import psutil
    return (f"{name} - {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads, "
            f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="*")
    parser.add_argument("--models", nargs="+", default=["large-v3-turbo"],
                        choices=[m.key for m in STT_MODELS])
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--reference", help="text file with the correct transcript (for WER/CER)")
    parser.add_argument("--max-files", type=int, default=2,
                        help="when a folder is given, use only this many files from it (default 2)")
    parser.add_argument("--report", default=str(ROOT / "benchmark_report.md"))
    parser.add_argument("--child", nargs=3, metavar=("MODEL_DIR", "THREADS", "AUDIO"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_child(args.child[0], int(args.child[1]), args.child[2])
        return 0
    audio_files = expand_audio_args(args.audio or [str(ROOT / "samples")], args.max_files)
    if not audio_files:
        parser.error("no audio files found (put voice messages in the samples folder)")

    reference = Path(args.reference).read_text(encoding="utf-8") if args.reference else None
    import faster_whisper
    lines = [
        "# VoiceApp speech-to-text benchmark", "",
        f"- CPU: {cpu_description()}",
        f"- OS: {platform.platform()}, Python {platform.python_version()}, "
        f"faster-whisper {faster_whisper.__version__}",
        "- Settings: device=cpu, compute_type=int8, language=fa, vad_filter=True, beam_size=5",
        "- speed ratio = transcription time / audio length (below 1.0 = faster than real time)",
        "", "| model | threads | file | audio (s) | load (s) | transcribe (s) | speed ratio | peak RAM (MB) |"
        + (" WER | CER |" if reference else ""),
        "|---|---|---|---|---|---|---|---|" + ("---|---|" if reference else ""),
    ]
    transcripts = []
    results = []
    for key in args.models:
        model_dir = ensure_model(key)
        for threads in args.threads:
            for audio in audio_files:
                print(f"Running {key}, cpu_threads={threads}, {Path(audio).name} ...", flush=True)
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--child", str(model_dir), str(threads), audio],
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
                found = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON ")]
                if proc.returncode != 0 or not found:
                    print(proc.stdout[-2000:], proc.stderr[-4000:])
                    lines.append(f"| {key} | {threads} | {Path(audio).name} | FAILED (exit {proc.returncode}) |||||")
                    continue
                r = json.loads(found[0][len("RESULT_JSON "):])
                r.update(model=key, threads=threads, file=Path(audio).name)
                results.append(r)
                row = (f"| {key} | {threads} | {r['file']} | {r['audio_seconds']:.1f} | {r['load_seconds']:.1f} | "
                       f"{r['transcribe_seconds']:.1f} | **{r['speed_ratio']:.2f}** | {r['peak_ram_mb']:.0f} |")
                if reference:
                    w, c = wer_cer(reference, r["text"])
                    r.update(wer=w, cer=c)
                    row += f" {w:.1%} | {c:.1%} |"
                lines.append(row)
                print("  " + row, flush=True)
                if threads == args.threads[0]:
                    transcripts.append((key, r))

    if results:
        lines += ["", "## Fastest cpu_threads per model", ""]
        for key in args.models:
            rs = [r for r in results if r["model"] == key]
            if rs:
                by_threads = {}
                for r in rs:
                    by_threads.setdefault(r["threads"], []).append(r["transcribe_seconds"])
                best = min(by_threads, key=lambda t: sum(by_threads[t]))
                lines.append(f"- {key}: cpu_threads={best}")
    lines += ["", "## Transcripts (first thread setting)", ""]
    if reference:
        lines += ["### Reference (correct text)", "", '<div dir="rtl">', "", reference.strip(), "", "</div>", ""]
    for key, r in transcripts:
        lines += [f"### {key} - {r['file']}", "", '<div dir="rtl">', ""]
        lines += [f"[{s:6.1f}s - {e:6.1f}s] {t}  " for s, e, t in r["segments"]]
        lines += ["", "</div>", ""]

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport written to {args.report}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
