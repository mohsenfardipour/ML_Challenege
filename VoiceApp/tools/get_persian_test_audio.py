"""Download real Persian test speech with a known correct transcript.

    python tools/get_persian_test_audio.py OUTPUT_DIR [--count 8]

Uses the Persian (fa_ir) test set of Google's FLEURS dataset (CC-BY 4.0,
read speech). Several sentences are joined into one ~1 minute file and saved
as OGG/Opus like a Telegram voice message, plus reference.txt with the
correct text. Used by CI for the speech-to-text benchmark; your own voice
messages in samples\\ are the better test.
"""

import argparse
import csv
import io
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import find_tool  # noqa: E402

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/fa_ir"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out_dir)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(f"{BASE}/test.tsv", timeout=60) as resp:
        rows = list(csv.reader(io.TextIOWrapper(resp, encoding="utf-8"), delimiter="\t",
                               quoting=csv.QUOTE_NONE))
    # columns: id, file name, raw transcription, normalized transcription, ...
    seen_ids, wanted = set(), {}
    for row in rows:
        if row[0] not in seen_ids:          # several speakers may read the same sentence
            seen_ids.add(row[0])
            wanted[row[1]] = row[2]
        if len(wanted) >= args.count:
            break

    found = {}
    with urllib.request.urlopen(f"{BASE}/audio/test.tar.gz", timeout=120) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if name in wanted and member.isfile():
                    (clips / name).write_bytes(tar.extractfile(member).read())
                    found[name] = wanted[name]
                    print("got", name)
                    if len(found) == len(wanted):
                        break

    order = [n for n in wanted if n in found]
    list_file = clips / "list.txt"
    list_file.write_text("".join(f"file '{n}'\n" for n in order), encoding="utf-8")
    ffmpeg = find_tool("ffmpeg")
    target = out / "fleurs_fa_sample.ogg"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(list_file), "-ac", "1", "-c:a", "libopus", "-b:a", "32k", str(target)],
                   check=True)
    (out / "reference.txt").write_text("\n".join(found[n] for n in order) + "\n", encoding="utf-8")
    print(f"Wrote {target} ({len(order)} sentences) and reference.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
