# VoiceApp — offline audio converter (Phase 1)

A small Windows desktop app that converts voice messages and other audio files
to **MP3**, **MP4 (audio only, AAC)** or **M4A**. It works completely offline.

* Input: OGG/Opus (Telegram, WhatsApp), OPUS, OGA, MP3, WAV, M4A, AAC, FLAC, WEBM
* Batch conversion, drag-and-drop of files **and folders**
* Bitrate 64–320 kbps (default 128 kbps)
* Output next to each source file, or in one folder you choose
* Originals are **never** changed; existing files are **never** overwritten
  (`name.mp3` → `name_1.mp3` → `name_2.mp3` …)
* Per-file status, overall progress bar, Cancel, “Open output folder”
* Persian file names are shown correctly
* Remembers your last settings; writes a log file for troubleshooting

---

## 1. Folder layout

Copy the `VoiceApp` folder to **`D:\VoiceApp`**. All downloads, caches,
settings and logs stay inside this folder (nothing large goes to drive C:).

```
D:\VoiceApp\
├─ setup.bat            one-time setup (Python packages + FFmpeg)
├─ run_dev.bat          run the app from source
├─ build.bat            build the standalone exe (dist\VoiceApp\VoiceApp.exe)
├─ VoiceApp.py          entry point
├─ VoiceApp.spec        PyInstaller build recipe
├─ requirements.txt     pinned Python packages
├─ app\
│  ├─ main.py           starts the GUI (also --selftest / --smoke-test)
│  ├─ paths.py          where the data folder and FFmpeg are
│  ├─ settings.py       remembers last-used settings (data\settings.json)
│  ├─ logging_setup.py  log file (data\logs\voiceapp.log)
│  ├─ selftest.py       command-line self-test used after building
│  ├─ core\             conversion logic, NO GUI code (easy to test)
│  │  ├─ formats.py     supported inputs, output formats, bitrates
│  │  ├─ ffmpeg.py      runs ffmpeg, reads duration, reports progress
│  │  ├─ converter.py   converts one file safely, friendly error messages
│  │  └─ naming.py      never-overwrite naming (name_1.mp3 …)
│  └─ ui\
│     ├─ main_window.py window with one tab per feature
│     ├─ workers.py     background thread so the window never freezes
│     └─ tabs\
│        ├─ __init__.py list of tabs  ← Phase 2 adds its tab here
│        └─ converter_tab.py
├─ tests\               automated tests (pytest)
├─ tools\
│  ├─ get_ffmpeg.py     downloads FFmpeg 7.1.1 and checks its SHA-256
│  └─ make_test_samples.py  creates test audio files with Persian names
├─ ffmpeg\              (created by setup.bat) ffmpeg.exe, ffprobe.exe
├─ data\                (created at runtime) settings.json, logs\
├─ samples\             your own test files (not used by the code)
├─ .venv\  .cache\      (created by setup/build) Python env and caches
└─ dist\                (created by build.bat) the finished app
```

## 2. Setup (once)

1. **Install Python 3.11 (64-bit)** if you don't have it:
   <https://www.python.org/downloads/release/python-3119/> → “Windows installer (64-bit)”.
   On the first screen tick **“Add python.exe to PATH”**, then “Install Now”.
   Check it in a new Command Prompt: `py -3.11 --version` → `Python 3.11.x`.
2. Double-click **`setup.bat`**. It:
   * checks Python and shows free disk space on C: and D:
   * creates `.venv` (a private Python environment for this project)
   * installs PySide6, PyInstaller and pytest (pip cache kept in `.cache`)
   * downloads FFmpeg (~90 MB) into `ffmpeg\` and verifies its checksum

   Needs internet once. Total size after setup: about 700 MB, all on D:.

## 3. Run from source

Double-click **`run_dev.bat`**.

## 4. Build the standalone app

Double-click **`build.bat`**. It runs the tests, builds the app with
PyInstaller, then runs the built exe in self-test mode (converts test files
to all three formats and checks them). If everything passes you get:

* `dist\VoiceApp\VoiceApp.exe` — the app (keep the whole `VoiceApp` folder together)
* `dist\VoiceApp-win64.zip` — the same folder zipped, to copy to another PC.
  Unzip it anywhere and run `VoiceApp.exe`. **No Python needed.**

Build takes 1–3 minutes. The app folder is about 150 MB (FFmpeg alone is 87 MB); the zip is about 57 MB.

Each push to GitHub also builds the exe on a clean Windows machine
(`.github/workflows/voiceapp-windows.yml`). Download it from the run’s
**Artifacts** section in the repository’s *Actions* tab.

## 5. Using the app

1. **Add files…** / **Add folder…**, or drag files/folders onto the list.
   Folders are searched including sub-folders; unsupported files are skipped.
2. Choose **Output format** and **Bitrate**.
   For voice messages 64–96 kbps is plenty; 128 kbps is the safe default.
3. Choose **Save to**: next to each source file, or one folder.
4. Click **Convert**. Each row shows its status; hover over an error to read it.
   **Cancel** stops after cleaning up the file being written.
5. Click **Open output folder** — Explorer opens with the last converted file selected.

Clicking **Convert** again converts only the files that are not *Done* yet
(failed or cancelled ones). If everything is done, it converts all files again,
for example in another format. The new files get new names, nothing is replaced.

### Self-test from the command line

```bat
dist\VoiceApp\VoiceApp.exe --selftest D:\VoiceApp\samples D:\VoiceApp\selftest_out
type D:\VoiceApp\selftest_out\selftest_report.txt
```

This converts every sample to MP3, MP4 and M4A and compares each result's length
with the source (`PASS`/`FAIL` per file). Use `run_dev.bat --selftest …` to do
the same from source.

## 6. Troubleshooting

| Problem | What to do |
|---|---|
| `Python 3.11 was not found` | Install it as in step 2.1, open a **new** window, run `setup.bat` again. |
| Setup fails while downloading | Check internet/proxy and run `setup.bat` again (it continues where it stopped). |
| “FFmpeg missing” message | `ffmpeg\ffmpeg.exe` is missing. Run `setup.bat`. In the built app, keep the `_internal` folder next to `VoiceApp.exe`. |
| A file shows *Error: The file is damaged…* | The file is incomplete or not really audio. Try playing it; re-download it. |
| *Permission denied* | Output folder is read-only (e.g. `C:\Program Files`) or the file is open in a player. Choose another folder. |
| *Done (warning)* | Converted, but the length differs from the source by more than 1 s / 2 %. Usually the source file is damaged. |
| Windows SmartScreen “protected your PC” | The exe isn't code-signed. Click **More info → Run anyway**. |
| Antivirus deletes the exe | A known false positive with PyInstaller apps. Add an exception for `D:\VoiceApp`. |
| Anything else | Send `data\logs\voiceapp.log` (shown at the bottom of the window). |

Run the tests yourself: `.venv\Scripts\python.exe -m pytest tests`

## 7. Design decisions (short)

* **PySide6-Essentials** instead of full PySide6: same Qt widgets, about half the download.
* **Only `ffmpeg.exe` is bundled.** Duration is read from ffmpeg's own output,
  so `ffprobe.exe` (another 87 MB) is only used for development/testing.
* **FFmpeg version is pinned** (7.1.1, “essentials” build by Gyan Doshi) and
  its SHA-256 is checked, so every build contains exactly the same, untampered FFmpeg.
* **Files are converted one at a time.** A voice message takes well under a second,
  and running one ffmpeg at a time keeps a low-power laptop responsive.
* **Safe writing.** ffmpeg writes to a temporary `.voiceapp.part` file; only a
  complete result is renamed to its final, unused name. Cancelled or failed files leave nothing behind.
* **onedir build, no UPX.** Starts faster than a one-file exe and causes fewer antivirus false positives.
* **Settings and logs in `data\` next to the app** (falls back to `%LOCALAPPDATA%\VoiceApp`
  only if that folder is read-only).
* **Tabs are a plug-in list** (`app/ui/tabs/__init__.py`). Phase 2 adds a
  `TranscribeTab` there; the converter code does not need to change.

## 8. Known limitations

* Converts only the **first audio track** of a file; video is ignored.
* Browser-recorded WEBM files often have no stored duration; they convert fine,
  but the progress bar for that file jumps from 0 to 100 %.
* Mono voice messages stay mono, and the sample rate is kept
  (for Opus that is 48 kHz). This is fine for all players.
* Tags (title/artist) are copied when the source has them; cover art is dropped.
* The exe is not code-signed (see SmartScreen above).
* English user interface only (Persian file names and text display correctly).

## 9. Adding a feature later (Phase 2)

1. Create `app/ui/tabs/transcribe_tab.py` with `class TranscribeTab(QWidget)`,
   a `TAB_TITLE`, and `__init__(self, settings, parent=None)`.
   Optionally add `is_busy()` and `shutdown()` (see `converter_tab.py`).
2. Put the non-GUI logic in `app/core/` and run long work in a worker like `app/ui/workers.py`.
3. Add the class to `TABS` in `app/ui/tabs/__init__.py`.
4. Add new settings fields to `Settings` in `app/settings.py` (old settings files keep working).
5. Add new packages to `requirements.txt`, and their data files to `VoiceApp.spec` if needed.

---

## خلاصه فارسی

**VoiceApp** برنامه‌ای کوچک و کاملاً آفلاین برای ویندوز است که پیام‌های صوتی
(تلگرام، واتس‌اپ) و فایل‌های صوتی دیگر را به **MP3**، **MP4 (فقط صدا)** یا **M4A** تبدیل می‌کند.

**نصب (یک بار):**
۱. پوشه `VoiceApp` را در `D:\VoiceApp` کپی کنید.
۲. پایتون 3.11 نسخه ۶۴ بیتی را از python.org نصب کنید (گزینه *Add python.exe to PATH* را تیک بزنید).
۳. روی `setup.bat` دوبار کلیک کنید (به اینترنت نیاز دارد؛ همه چیز روی درایو D ذخیره می‌شود).

**اجرا:** `run_dev.bat`
**ساخت فایل exe:** `build.bat` ← خروجی در `dist\VoiceApp\VoiceApp.exe` و فایل زیپ
`dist\VoiceApp-win64.zip` که روی هر کامپیوتر ویندوزی بدون پایتون اجرا می‌شود.

**استفاده:** فایل‌ها یا پوشه‌ها را به لیست بکشید، فرمت و بیت‌ریت را انتخاب کنید و روی
**Convert** کلیک کنید. فایل‌های اصلی هرگز تغییر نمی‌کنند و هیچ فایلی بازنویسی نمی‌شود
(مثلاً `name_1.mp3` ساخته می‌شود).

**عیب‌یابی:** پیام خطای هر فایل در ستون Status نمایش داده می‌شود. فایل گزارش در
`data\logs\voiceapp.log` است. اگر ویندوز هشدار SmartScreen داد، روی
*More info → Run anyway* کلیک کنید.
