# VoiceApp 2.0 — offline audio converter + Persian speech-to-text

A small Windows desktop app with three tabs, working **completely offline**:

1. **Convert audio** – voice messages and other audio to **MP3**, **MP4 (audio only, AAC)** or **M4A**.
2. **Transcribe (Persian)** – Persian speech to editable text with local Whisper models
   (faster-whisper, CPU only). Export TXT, DOCX (right-to-left) and SRT subtitles.
3. **Personalize** – teach the app your names and words, and fix its repeated mistakes,
   from your own samples (no model training needed).

The only network access is the optional, one-time model download. No cloud APIs, no telemetry.

> **Quick start in Persian: [START_HERE.md](START_HERE.md)**

---

## 1. Folder layout

Put everything under **`D:\VoiceApp`**. Models, personal data, settings, logs and
build caches all stay there (nothing large goes to drive C:).

```
D:\VoiceApp\
├─ VoiceApp\VoiceApp.exe   the app, if you use the ready-made zip from the release
├─ models\                 speech-to-text models (one folder per model)
│  └─ whisper-large-fa-v1\ model.bin, config.json, tokenizer.json, vocabulary.json …
├─ personal\               your samples, vocabulary.txt, rules.json
│
│  (the rest is only needed to run/build from source)
├─ setup.bat  run_dev.bat  build.bat  benchmark.bat
├─ VoiceApp.py  VoiceApp.spec  requirements.txt
├─ app\
│  ├─ main.py  paths.py  settings.py  logging_setup.py  selftest.py
│  ├─ core\                logic without GUI (easy to test)
│  │  ├─ formats.py  ffmpeg.py  converter.py  naming.py      audio conversion
│  │  ├─ stt_models.py     model list, release URLs, model folders
│  │  ├─ model_download.py download (with resume) + unzip from the GitHub release
│  │  ├─ transcriber.py    faster-whisper wrapper (CPU, int8, Persian)
│  │  ├─ persian_text.py   ي/ك → ی/ک, half-spaces, scoring normalization
│  │  ├─ text_align.py     word alignment, WER/CER, dropped sentences
│  │  ├─ personal.py       samples, vocabulary hints, correction rules
│  │  └─ exporters.py      TXT / DOCX (RTL) / SRT
│  └─ ui\
│     ├─ main_window.py    one tab per feature
│     ├─ workers.py  stt_workers.py   background threads (window never freezes)
│     ├─ stt_common.py     model status, download dialog
│     └─ tabs\ converter_tab.py  transcribe_tab.py  personalize_tab.py  (+ __init__.py: tab list)
├─ tests\                  pytest (fake model for speed, real model in CI)
└─ tools\
   ├─ get_ffmpeg.py  make_test_samples.py  get_persian_test_audio.py  benchmark_stt.py
   └─ model_pipeline\      converts the Hugging Face models (runs on GitHub Actions)
```

## 2. Install

### A. Ready-made app (recommended, no Python needed)

1. Download **`VoiceApp-2.0-win64.zip`** from the release page:
   <https://github.com/mohsenfardipour/ML_Challenege/releases/tag/voiceapp-2.0>
2. Unzip it into `D:\VoiceApp` → you get `D:\VoiceApp\VoiceApp\VoiceApp.exe`.
3. Run `VoiceApp.exe`. Then install a model (section 4).

The zip is about 150 MB; unpacked the app folder is about 370 MB (FFmpeg, Qt, CTranslate2,
ONNX Runtime). Models are separate downloads.

### B. From source

1. Install **Python 3.11 (64-bit)**: <https://www.python.org/downloads/release/python-3119/>
   (tick **“Add python.exe to PATH”**). Check: `py -3.11 --version`.
2. Double-click **`setup.bat`**: creates `.venv`, installs the pinned packages
   (PySide6, faster-whisper, CTranslate2, python-docx … about 450 MB, pip cache in `.cache`)
   and downloads FFmpeg (~90 MB, checksum verified).
3. **`run_dev.bat`** starts the app. **`build.bat`** builds `dist\VoiceApp\VoiceApp.exe`
   (runs the tests, builds with PyInstaller, self-tests the exe, makes `dist\VoiceApp-win64.zip`).
   Models are never put into the exe.

Every push also builds and tests the exe on a clean Windows machine on GitHub
(`.github/workflows/voiceapp-windows.yml`), including a real Persian transcription
with the packaged exe, and attaches the zip to the release.

## 3. Convert audio (tab 1)

Input: OGG/Opus (Telegram, WhatsApp), OPUS, OGA, MP3, WAV, M4A, AAC, FLAC, WEBM.
Add files/folders (or drag and drop), choose format and bitrate (64–320 kbps, default 128),
choose where to save, click **Convert**. Originals are never changed and existing files
are never overwritten (`name.mp3` → `name_1.mp3`). Per-file status, overall progress,
Cancel, and **Open output folder**.

## 4. Speech-to-text models

| Model in the app | Zip on the release | Size | Based on | Expected time on an i3-1215U* |
|---|---|---|---|---|
| **Persian – fast (recommended)** | `whisper-large-fa-v1-ct2-int8.zip` | 0.7 GB | [vhdm/whisper-large-fa-v1](https://huggingface.co/vhdm/whisper-large-fa-v1) (large-v3-turbo fine-tuned on Persian, MIT) | ≈ 0.6 × audio length |
| Persian – alternative (slower) | `buzzasr-persian-ct2-int8.zip` | 1.3 GB | [BuzzASR/persian](https://huggingface.co/BuzzASR/persian) (large-v3 fine-tuned, MIT) | ≈ 1.6 × audio length |
| Base large-v3-turbo (fallback) | `large-v3-turbo-ct2-int8.zip` | 0.7 GB | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | ≈ 0.6 × audio length |

\* Estimates for a low-power laptop CPU with 4 threads, on the charger. So a 1-minute voice
message takes about 35–45 s with the fast model. RAM use: about 1.5 GB (fast) / 2.5 GB (alternative).
The base model is only used automatically if a Persian model fails to load.

The models were converted from Hugging Face to CTranslate2 int8 by
`.github/workflows/model-pipeline.yml` (because Hugging Face is often blocked in Iran),
checked against the original models, and published as release assets. The release
notes contain the full comparison (see section 7).

### Automatic download

In the **Transcribe** tab choose a model and click **Download model**. A progress bar is shown;
if the connection drops, click again and it continues where it stopped. Files go to `D:\VoiceApp\models\`.

### Manual install (if the download does not work)

1. Download the zip from <https://github.com/mohsenfardipour/ML_Challenege/releases/tag/voiceapp-2.0>
   with a browser or download manager (the repository is public, no account needed).
2. Unzip it into **`D:\VoiceApp\models\`** so that e.g. `D:\VoiceApp\models\whisper-large-fa-v1\model.bin` exists.
3. In the app click **Select local model folder…** → *Unzipped folder…* → choose
   `D:\VoiceApp\models\whisper-large-fa-v1` (or choose *Zip file…* and let the app unzip it).

A model folder anywhere else works too; the app remembers the folder you select.

## 5. Transcribe (tab 2)

1. **Add files…** (or drag files/folders onto the list). The estimated time is shown before you start.
2. Choose the model and **CPU threads** (default **4**: on the i3-1215U the 2 fast cores
   + hyper-threading do most of the work; more threads add heat, not much speed).
3. **Transcribe**. Progress, elapsed and remaining time are shown; the estimate is refined
   from the real speed. **Cancel** stops after the current part (a few seconds).
   The model is loaded once and reused for all files of the batch.
4. Click a finished file to see its text in the right-to-left editor. Edit it, then
   **Copy**, **Save TXT…** (UTF-8), **Save DOCX…** (RTL paragraphs, Tahoma), **Save SRT…**,
   or **Save all…** (TXT + DOCX, optional SRT, never overwrites).
5. **Save as personal sample**: after correcting the text, this stores the voice file (a copy)
   and the corrected text as a sample for the Personalize tab.

Decoding settings (fixed): `language="fa"` (no auto-detect), `task="transcribe"`, VAD filter on,
`condition_on_previous_text=False` (fewer repeated/skipped sentences on long audio), `beam_size=5`,
temperature fallback 0.0–1.0, `compute_type="int8"`, CPU only.
Every transcript is normalized: Arabic ي/ك → Persian ی/ک, half-space after می/نمی and before ها/های.

**Speed tips:** plug in the charger; Windows *Settings → System → Power → Power mode → Best performance*;
close heavy programs (browsers with many tabs); keep CPU threads at 4.

## 6. Personalize (tab 3) – learning from your samples without training

Real fine-tuning is not possible on a laptop CPU, so VoiceApp uses three lightweight methods:

1. **Samples** – pairs of (voice file + exact correct text). Add them here or with
   *Save as personal sample*. Stored in `D:\VoiceApp\personal\samples\`. List, edit, delete.
2. **Vocabulary** – names and special words, one per line, most important first. *Suggest from
   my samples* ranks words the model got wrong first. They are given to the model as
   **hotwords** before every 30-second part, limited to ~40 tokens. **Off by default**
   (*Give these words to the model as hints*): in our tests hints made the Persian model skip or garble
   sentences (section 7). The list is still used to rank suggestions; turn hints on only if *Test* shows a gain.
3. **Correction rules** – *Analyze samples* transcribes your samples, aligns the result with your
   correct text and suggests word-level rules such as `اناثر → عناصر`. You tick the ones to use.
   Approved rules are applied after every transcription, **whole words only** (never inside another
   word). Rules whose "wrong" word is a common word or appears correctly in your own texts are
   marked ⚠ and not approved by *Approve all safe*.
4. **Test** – *Test on my samples* runs every installed Persian model on your samples with and
   without personalization and shows WER/CER (lower is better), plus an honest estimate for new
   messages (rules learned only from the *other* samples, needs 3+ samples). *Use best model as
   default* switches the default model.
5. In the Transcribe tab, **Use my personalization** switches all of this on/off.

## 7. Test results

Persian FLEURS test set (8 read sentences, 145 s, converted to OGG/Opus like a voice message),
GitHub runner CPU, int8, 4 threads. WER = wrong words, CER = wrong letters (lower is better).
"Speaker 2" reads the same sentences: it shows how much personalization learned from speaker 1
helps on a *new* recording.

| Model | WER speaker 1 | WER speaker 2 | CER speaker 1 | Dropped sentences | Speed (× audio length) |
|---|---|---|---|---|---|
| **Persian – fast (whisper-large-fa-v1)** | **14.5 %** | **23.9 %** | **5.2 %** | **0 / 8** | **0.33** |
| Persian – alternative (BuzzASR, large-v3) | 44.0 % | 32.1 % | 34.1 % | 2 / 8 | 0.91 |
| Base large-v3-turbo (not fine-tuned) | 29.1 % | 56.8 % | 14.1 % | 1 / 8 | 0.26 |

* The fast Persian model is the most accurate **and** among the fastest → it is the default.
* BuzzASR was requested as the "high accuracy" option. After fixing its tokenizer it reproduces the
  original model (1.4 % CER difference vs. transformers), but on this test it was *less* accurate and
  ~2.7× slower, so the app labels it "alternative". Try it on your own messages with *Personalize → Test*.
* Conversion check against the original models (transformers, float32): fast model 3.7 % CER difference,
  BuzzASR 1.4 % – normal int8 rounding differences.
* Speed on the GitHub CPU (4 threads): 0.33× for the fast model. The packaged Windows exe
  transcribed 60 s of Persian speech in 45 s (0.75×, incl. long pauses) on a 4-thread Windows runner.
  Expect roughly 0.6× audio length on an i3-1215U on the charger.

**Personalization** (WER; learned from speaker 1's recording and its correct text):

| Model | Test recording | Plain | + correction rules | + vocabulary hints | + both |
|---|---|---|---|---|---|
| Persian fast | speaker 1 (same recording) | 14.5 % | **6.4 %** | 48.3 % | 43.6 % |
| Persian fast | speaker 2 (new recording) | 23.9 % | **23.5 %** | 25.2 % | 24.4 % |
| BuzzASR | speaker 1 | 44.0 % | 38.9 % | 34.6 % | 32.1 % |
| BuzzASR | speaker 2 | 32.1 % | 32.1 % | 38.0 % | 37.6 % |
| Base turbo | speaker 1 | 29.1 % | 17.9 % | 36.3 % | 30.3 % |
| Base turbo | speaker 2 | 56.8 % | 55.1 % | 19.7 % | 18.4 % |

* **Correction rules** never made things worse. They remove the model's repeated mistakes
  (14.5 → 6.4 % on the same recording) but only help on new messages when the same words come back
  (23.9 → 23.5 % here, where the sentences share few rare words). With your own messages, which repeat
  names and terms, expect a gain between these two numbers.
* **Vocabulary hints** were unstable: they made the fast model skip/garble sentences, so they are
  **off by default** (Personalize → Vocabulary → checkbox). The speaker-1 numbers are optimistic because
  the rules were learned from that same recording.

## 8. Command-line self-tests

```bat
VoiceApp.exe --selftest D:\VoiceApp\samples D:\VoiceApp\selftest_out
VoiceApp.exe --stt-selftest D:\VoiceApp\models\whisper-large-fa-v1 D:\VoiceApp\samples\voice.ogg D:\VoiceApp\stt_out [correct_text.txt]
benchmark.bat                       (speed test of the models on your PC, writes benchmark_report.md)
```

The first converts every sample to all formats and checks the lengths; the second transcribes one
file with a real model, exports TXT/DOCX/SRT and (with a correct text) reports WER/CER.

## 9. Troubleshooting

| Problem | What to do |
|---|---|
| “Model needed” / Transcribe is greyed out | Install a model (section 4). |
| Download fails or is very slow | Click Download again (it resumes), or install manually (section 4). |
| “Not a model folder” | Select the folder that directly contains `model.bin` and `tokenizer.json`. |
| Transcription is slow | Charger + Best performance mode, 4 threads, close other programs, use the fast model. |
| A sentence is missing in the text | Turn vocabulary hints off (Personalize → Vocabulary), or try the other model. |
| Strange repeated words | Usually music/noise. Cut the noisy part or try the other model. |
| The packaged exe shows an error at start | Keep the whole `VoiceApp` folder (with `_internal`) together; re-unzip it. |
| `Python 3.11 was not found` (source) | Install Python 3.11 as in 2B and run `setup.bat` again. |
| *Error: The file is damaged…* | The audio file is incomplete or not audio. |
| Windows SmartScreen | The exe is not code-signed: **More info → Run anyway**. |
| Antivirus removes the exe | False positive with PyInstaller apps; add an exception for `D:\VoiceApp`. |
| Anything else | Send `data\logs\voiceapp.log` (path shown at the bottom of the window). |

## 10. Design decisions (short)

* **faster-whisper + CTranslate2 int8 on CPU**: the fastest practical way to run Whisper on a laptop
  without a GPU; no PyTorch in the app (keeps it ~400 MB smaller).
* **Models outside the exe**, downloaded from this project's GitHub Release (Hugging Face is often
  blocked); resumable download; manual install always possible.
* **BuzzASR/persian** ships an inconsistent tokenizer: leftover entries at ids 0–1608 carry special
  names (`<|endoftext|>` = 0), while the model was trained with the standard Whisper layout shifted by
  1608 (`<|endoftext|>` = 51865, `<|fa|>` = 51908), as `added_tokens.json` and `generation_config.json`
  say. A diagnostic run proved which layout the model really uses; the pipeline rebuilds `tokenizer.json`
  accordingly (weights unchanged) and checks the converted model against the original in `transformers`.
* **Persian fast model as default**: in our tests it had the lowest error rate *and* is ~2.5× faster
  than the large-v3 based model on CPU.
* **Hotwords instead of initial_prompt**: with `condition_on_previous_text=False`, faster-whisper only
  uses `initial_prompt` for the first 30 seconds, while hotwords are given for every part.
* **Correction rules need your approval** and match whole words only, because an automatic rule
  (e.g. مصر → مس) can also damage correct text.
* Phase 1 decisions still apply: PySide6-Essentials, pinned & checksum-verified FFmpeg, safe
  never-overwrite writing, onedir build without UPX, settings/logs in `data\` next to the app.

## 11. Known limitations

* Accuracy depends a lot on audio quality: noisy, very fast, dialect or mixed-language speech
  produces more errors. Always read the transcript before using it.
* Persian homophones (e.g. عناصر/اناثر, طلا/تلا) are a typical error type; correction rules help
  only for words that repeat in your messages.
* Personalization is not training: it cannot teach the model new pronunciations; the measured gain
  is small unless your messages share names/terms with your samples.
* Vocabulary hints can occasionally cause a skipped sentence (see section 7); switch them off if you notice it.
* Cancel waits for the current part of the audio (up to ~20 s with the alternative model).
* SRT timestamps come from the model's segments (sentence level, not word level). Edits in the text
  box are used for TXT/DOCX; SRT always uses the original segments.
* The alternative (BuzzASR) model needs ~2.5 GB RAM, is ~1.6× slower than real time on an i3 and was
  less accurate than the default model in our test.
* English interface (Persian text displays correctly). The exe is not code-signed.
* Converter limitations from Phase 1: first audio track only; WEBM without stored duration shows
  no per-file progress.

## 12. Adding another feature

Create a new tab class in `app/ui/tabs/`, put logic in `app/core/`, long work in a worker thread,
and add the class to `TABS` in `app/ui/tabs/__init__.py`. New settings go into `Settings`
(`app/settings.py`); old settings files keep working.

---

## خلاصه فارسی

**VoiceApp 2.0** یک برنامه کاملاً آفلاین برای ویندوز است:
۱) تبدیل فایل‌های صوتی به MP3/MP4/M4A، ۲) **تبدیل گفتار فارسی به متن** با مدل‌های Whisper فارسی
روی همین لپ‌تاپ (بدون اینترنت و بدون کارت گرافیک)، و ۳) **شخصی‌سازی**: برنامه از نمونه‌های صوتی
شما کلمات خاص و اشتباه‌های تکراری را یاد می‌گیرد.

راهنمای قدم‌به‌قدم فارسی: **[START_HERE.md](START_HERE.md)**

- مدل پیشنهادی: «Persian – fast»، حدود ۰٫۷ گیگابایت؛ هر دقیقه صدا حدود ۳۵ تا ۴۵ ثانیه روی i3.
- دانلود مدل از داخل برنامه (دکمه Download model) یا دستی از صفحه Release و باز کردن در `D:\VoiceApp\models\`.
- خروجی: TXT، Word (راست‌به‌چپ) و زیرنویس SRT.
- برای سرعت بیشتر: شارژر وصل باشد و حالت Best performance ویندوز روشن باشد.
