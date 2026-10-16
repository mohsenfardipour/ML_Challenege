"""Download the FFmpeg Windows build (once) into the ./ffmpeg folder.

    python tools/get_ffmpeg.py

A fixed version is used and its SHA-256 checksum is verified, so every build
of VoiceApp contains exactly the same, untampered FFmpeg.
Only ffmpeg.exe is shipped inside the app; ffprobe.exe is kept for testing.
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "7.1.1"
URL = (f"https://github.com/GyanD/codexffmpeg/releases/download/{VERSION}/"
       f"ffmpeg-{VERSION}-essentials_build.zip")
SHA256 = "04861d3339c5ebe38b56c19a15cf2c0cc97f5de4fa8910e4d47e5e6404e4a2d4"
WANTED = ("bin/ffmpeg.exe", "bin/ffprobe.exe", "LICENSE")

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "ffmpeg"
CACHE = ROOT / ".cache"


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    tmp = dest.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1024 * 256):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total:3d}%  ({done // 1_048_576} MB)", end="", flush=True)
    print()
    tmp.replace(dest)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if (TARGET / "ffmpeg.exe").is_file():
        print(f"FFmpeg already present in {TARGET}")
        return 0
    CACHE.mkdir(exist_ok=True)
    archive = CACHE / f"ffmpeg-{VERSION}-essentials_build.zip"
    if not archive.is_file() or sha256_of(archive) != SHA256:
        download(URL, archive)
    actual = sha256_of(archive)
    if actual != SHA256:
        archive.unlink()
        print(f"ERROR: checksum mismatch (got {actual}). The download was damaged - run again.")
        return 1
    TARGET.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            for wanted in WANTED:
                if info.filename.endswith("/" + wanted):
                    dest = TARGET / Path(wanted).name
                    with z.open(info) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    print(f"  extracted {dest.name}")
    if not (TARGET / "ffmpeg.exe").is_file():
        print("ERROR: ffmpeg.exe not found in the archive")
        return 1
    print(f"FFmpeg {VERSION} is ready in {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
