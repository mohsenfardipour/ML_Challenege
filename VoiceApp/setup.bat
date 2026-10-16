@echo off
rem ==================================================================
rem  VoiceApp - one-time setup (run again any time; it is safe).
rem  Creates .venv, installs packages and downloads FFmpeg.
rem  Everything stays inside this folder, nothing is written to C:.
rem ==================================================================
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0.cache\pip"
set "PYTHONUTF8=1"

echo.
echo === Checking your system ===

set "PY="
py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
if not defined PY (
    python -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" >nul 2>nul && set "PY=python"
)
if not defined PY goto :nopython

%PY% -c "import sys, platform; print('Python', platform.python_version(), platform.architecture()[0], '-', sys.executable)"
%PY% -c "import shutil; [print(f'Free space on {d}: {shutil.disk_usage(d).free // 2**30} GB') for d in ('C:\\', 'D:\\') if __import__('os').path.exists(d)]"
echo Project folder: %~dp0

echo.
echo === Creating virtual environment (.venv) ===
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv || goto :fail
) else (
    echo .venv already exists - reusing it.
)

echo.
echo === Installing Python packages (about 250 MB, takes a few minutes) ===
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :fail

echo.
echo === Downloading FFmpeg (about 90 MB, only the first time) ===
".venv\Scripts\python.exe" tools\get_ffmpeg.py || goto :fail

echo.
echo ==================================================================
echo  Setup finished.  Start the app with run_dev.bat
echo  or build the standalone exe with build.bat
echo ==================================================================
if not defined VOICEAPP_NOPAUSE pause
exit /b 0

:nopython
echo.
echo  Python 3.11 was not found.
echo.
echo  1. Open https://www.python.org/downloads/release/python-3119/
echo  2. Download "Windows installer (64-bit)"
echo  3. Run it. On the first screen tick "Add python.exe to PATH",
echo     then click "Install Now". (The "py" launcher is included.)
echo  4. Close this window and run setup.bat again.
echo.
if not defined VOICEAPP_NOPAUSE pause
exit /b 1

:fail
echo.
echo  *** Setup FAILED - see the messages above. ***
echo  Check your internet connection and run setup.bat again.
if not defined VOICEAPP_NOPAUSE pause
exit /b 1
