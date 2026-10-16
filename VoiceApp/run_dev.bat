@echo off
rem Run VoiceApp from the source code (for development).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" VoiceApp.py %*
if errorlevel 1 (
    echo.
    echo VoiceApp exited with an error. See data\logs\voiceapp.log
    pause
)
