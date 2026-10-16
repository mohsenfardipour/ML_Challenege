"""Create small test audio files (one per supported input format, Persian names).

    python tools/make_test_samples.py OUTPUT_DIR

Needs ffmpeg (bundled copy in ./ffmpeg or on PATH). Used by the tests and by CI.
Your real voice messages in D:\\VoiceApp\\samples are better - use them too.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import find_tool  # noqa: E402

# (file name, ffmpeg encoder arguments)
SAMPLES = [
    ("پیام صوتی ۱.ogg", ["-c:a", "libopus", "-b:a", "32k", "-ac", "1", "-ar", "48000"]),
    ("voice message 2.opus", ["-c:a", "libopus", "-b:a", "24k", "-ac", "1"]),
    ("صدا.oga", ["-c:a", "libopus", "-ac", "1"]),
    ("آهنگ.mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
    ("ضبط.wav", ["-c:a", "pcm_s16le"]),
    ("گفتگو.m4a", ["-c:a", "aac", "-b:a", "96k"]),
    ("sample.aac", ["-c:a", "aac", "-f", "adts"]),
    ("sample.flac", ["-c:a", "flac"]),
    ("sample.webm", ["-c:a", "libopus"]),
]


def main(out_dir: str) -> None:
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, enc) in enumerate(SAMPLES):
        seconds = 3 + i  # different lengths, so duration checks are meaningful
        source = f"sine=frequency={300 + 50 * i}:duration={seconds}"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", source, *enc, str(out / name)], check=True)
        print("created", out / name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_samples")
