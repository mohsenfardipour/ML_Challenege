"""Launcher used by PyInstaller (and handy for running: python VoiceApp.py)."""

import os
import sys
import types


def _install_pkg_resources_shim() -> None:
    """CTranslate2 on Windows imports pkg_resources only for resource_filename().
    The packaged app leaves setuptools out, so provide just that function."""
    try:
        import pkg_resources  # noqa: F401
        return
    except ImportError:
        pass
    shim = types.ModuleType("pkg_resources")

    def resource_filename(package: str, resource: str = "") -> str:
        module = __import__(package, fromlist=["__file__"])
        return os.path.join(os.path.dirname(module.__file__), resource)

    shim.resource_filename = resource_filename
    sys.modules["pkg_resources"] = shim


if getattr(sys, "frozen", False):
    _install_pkg_resources_shim()

from app.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
