@echo off
rem ==================================================================
rem  Build the standalone app:  dist\VoiceApp\VoiceApp.exe
rem  1) runs the tests  2) builds with PyInstaller  3) self-tests the exe
rem  Speech-to-text models are NOT included (they are downloaded in the app).
rem ==================================================================
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    goto :fail
)
if not exist "ffmpeg\ffmpeg.exe" (
    echo ffmpeg\ffmpeg.exe is missing - please run setup.bat first.
    goto :fail
)

rem Keep build caches and temp files on this drive instead of C:
set "PYINSTALLER_CONFIG_DIR=%~dp0.cache\pyinstaller"
set "TEMP=%~dp0.cache\tmp"
set "TMP=%~dp0.cache\tmp"
set "PYTHONUTF8=1"
if not exist "%TEMP%" mkdir "%TEMP%"

echo === 1/3 Running tests ===
".venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests || goto :fail

echo.
echo === 2/3 Building with PyInstaller ===
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean VoiceApp.spec || goto :fail

echo.
echo === 3/3 Testing the built exe ===
set "CHECK=%~dp0.cache\buildcheck"
if exist "%CHECK%" rmdir /s /q "%CHECK%"
".venv\Scripts\python.exe" tools\make_test_samples.py "%CHECK%\in" >nul || goto :fail
set "VOICEAPP_DATA_DIR=%CHECK%\data"
start "" /wait "dist\VoiceApp\VoiceApp.exe" --selftest "%CHECK%\in" "%CHECK%\out"
if errorlevel 1 (
    type "%CHECK%\out\selftest_report.txt"
    echo The built exe FAILED its self-test.
    goto :fail
)
type "%CHECK%\out\selftest_report.txt"
set "VOICEAPP_DATA_DIR="

echo.
echo === Creating zip for copying to other PCs ===
copy /y README.md dist\VoiceApp\ >nul
copy /y START_HERE.md dist\VoiceApp\ >nul
if exist "dist\VoiceApp-win64.zip" del "dist\VoiceApp-win64.zip"
tar -a -c -f "dist\VoiceApp-win64.zip" -C dist VoiceApp || goto :fail

echo.
echo ==================================================================
echo  Build OK:  dist\VoiceApp\VoiceApp.exe
echo  Zip:       dist\VoiceApp-win64.zip  (unzip anywhere, no Python needed)
echo ==================================================================
if not defined VOICEAPP_NOPAUSE pause
exit /b 0

:fail
echo.
echo  *** BUILD FAILED - see the messages above. ***
if not defined VOICEAPP_NOPAUSE pause
exit /b 1
