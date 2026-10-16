"""Launcher used by PyInstaller (and handy for running: python VoiceApp.py)."""

import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
