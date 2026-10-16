@echo off
rem ==================================================================
rem  Speech-to-text speed test on THIS computer.
rem    benchmark.bat                       Persian fast model, first 2 files in samples\
rem    benchmark.bat --models whisper-large-fa-v1 buzzasr-persian
rem    benchmark.bat D:\VoiceApp\samples\voice1.ogg --threads 4 6 8
rem  First run downloads the model(s) into models\ (0.7 GB / 1.3 GB).
rem  Plug in the charger and close other programs while it runs.
rem  The result is written to benchmark_report.md
rem ==================================================================
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
set "PYTHONUTF8=1"
set "HF_HOME=%~dp0models\.hf-cache"
".venv\Scripts\python.exe" tools\benchmark_stt.py %*
echo.
echo Report: %~dp0benchmark_report.md
if not defined VOICEAPP_NOPAUSE pause
