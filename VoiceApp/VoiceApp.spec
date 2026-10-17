# PyInstaller build recipe. Run build.bat instead of calling this directly.
# onedir mode: dist\VoiceApp\VoiceApp.exe + an _internal folder (starts faster
# and is more reliable than a single self-extracting exe).

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ffmpeg_exe = os.path.join("ffmpeg", "ffmpeg.exe")
if sys.platform.startswith("win") and not os.path.isfile(ffmpeg_exe):
    raise SystemExit("ffmpeg\\ffmpeg.exe is missing - run setup.bat first.")

datas = []
if os.path.isfile(ffmpeg_exe):
    datas += [(ffmpeg_exe, "ffmpeg"), (os.path.join("ffmpeg", "LICENSE"), "ffmpeg")]

# Speech-to-text: CTranslate2 native libraries and faster-whisper's voice-activity model.
# The Whisper models themselves are NOT bundled (they live in D:\\VoiceApp\\models).
binaries = collect_dynamic_libs("ctranslate2")
datas += collect_data_files("faster_whisper") + collect_data_files("docx")

a = Analysis(
    ["VoiceApp.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    # numpy loads some of its own modules from C code, which PyInstaller cannot see.
    hiddenimports=collect_submodules("numpy._core"),
    excludes=["tkinter", "unittest", "pydoc", "pytest", "torch", "transformers", "tensorflow",
              "matplotlib", "scipy", "pandas", "IPython", "pkg_resources", "setuptools"],
    noarchive=False,
)

# Drop Qt parts this app never uses (keeps the folder ~40 MB smaller).
UNUSED = (
    "opengl32sw", "Qt6Pdf", "Qt6Quick", "Qt6Qml", "Qt6VirtualKeyboard", "Qt6Network",
    "Qt6OpenGL", "Qt6Svg", "qtvirtualkeyboard", "qpdf", "qsvg", "networkinformation",
    "/tls/", "\\tls\\", "translations",
)


def keep(entry):
    name = entry[0].replace("\\", "/")
    return not any(u.replace("\\", "/") in name for u in UNUSED)


a.binaries = [b for b in a.binaries if keep(b)]
a.datas = [d for d in a.datas if keep(d)]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceApp",
    console=False,          # no black console window
    upx=False,              # UPX often triggers antivirus false positives
)
coll = COLLECT(exe, a.binaries, a.datas, name="VoiceApp", upx=False)
