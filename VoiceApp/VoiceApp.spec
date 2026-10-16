# PyInstaller build recipe. Run build.bat instead of calling this directly.
# onedir mode: dist\VoiceApp\VoiceApp.exe + an _internal folder (starts faster
# and is more reliable than a single self-extracting exe).

import os
import sys

ffmpeg_exe = os.path.join("ffmpeg", "ffmpeg.exe")
if sys.platform.startswith("win") and not os.path.isfile(ffmpeg_exe):
    raise SystemExit("ffmpeg\\ffmpeg.exe is missing - run setup.bat first.")

datas = []
if os.path.isfile(ffmpeg_exe):
    datas += [(ffmpeg_exe, "ffmpeg"), (os.path.join("ffmpeg", "LICENSE"), "ffmpeg")]

a = Analysis(
    ["VoiceApp.py"],
    pathex=["."],
    datas=datas,
    excludes=["tkinter", "unittest", "pydoc", "pytest", "email.mime", "xmlrpc"],
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
