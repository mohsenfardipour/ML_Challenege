@echo off
chcp 65001 >nul
REM ساخت فایل exe تکی (این فایل را روی ویندوز اجرا کنید)
cd /d "%~dp0"

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install -r requirements-dev.txt

REM --onefile  : یک فایل exe تکی
REM --windowed : بدون پنجره‌ی سیاه (کنسول)
pyinstaller --noconfirm --clean --onefile --windowed --name DownloadsOrganizer main.py

echo.
echo تمام شد. فایل برنامه: dist\DownloadsOrganizer.exe
pause
