"""منطق اصلی مرتب‌سازی (بدون هیچ رابط گرافیکی).

قانون‌های ایمنی این فایل:
1. هیچ فایلی حذف نمی‌شود؛ فقط جابه‌جا می‌شود.
2. اگر در مقصد فایلی با همان اسم باشد، رویش نوشته نمی‌شود؛ شماره اضافه می‌شود.
3. فایل‌های نیمه‌دانلود (.crdownload, .part, .tmp) دست نمی‌خورند.
"""

import shutil
import time
from pathlib import Path

# هر دسته و پسوندهایش. (پسوندها با حروف کوچک و با نقطه)
CATEGORIES = {
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".ico", ".tiff", ".heic"},
    "Videos": {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"},
    "Music": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"},
    "Documents": {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".csv", ".rtf", ".odt"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
    "Installers": {".exe", ".msi"},
}
OTHERS = "Others"

# فایل‌های نیمه‌دانلود که نباید جابه‌جا شوند
PARTIAL_EXTENSIONS = {".crdownload", ".part", ".tmp", ".download", ".partial"}

# فایل‌های سیستمی که به آن‌ها کاری نداریم
IGNORED_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}

ALL_FOLDERS = set(CATEGORIES) | {OTHERS}


def get_downloads_folder() -> Path:
    """پوشه‌ی Downloads کاربر (معمولاً C:\\Users\\<name>\\Downloads)."""
    return Path.home() / "Downloads"


def is_partial(path: Path) -> bool:
    return path.suffix.lower() in PARTIAL_EXTENSIONS


def category_for(path: Path) -> str:
    ext = path.suffix.lower()
    for name, extensions in CATEGORIES.items():
        if ext in extensions:
            return name
    return OTHERS


def should_skip(path: Path) -> bool:
    """آیا این مورد نباید جابه‌جا شود؟"""
    if not path.is_file():
        return True  # پوشه‌ها را دست نمی‌زنیم
    name = path.name.lower()
    return is_partial(path) or name in IGNORED_NAMES


def unique_path(dest: Path) -> Path:
    """اگر dest وجود داشت، به اسمش شماره اضافه می‌کند: a.txt -> a (1).txt"""
    if not dest.exists():
        return dest
    counter = 1
    while True:
        candidate = dest.with_name(f"{dest.stem} ({counter}){dest.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_file(file: Path, root: Path):
    """یک فایل را به زیرپوشه‌ی مناسبش در root می‌برد.

    خروجی: (مبدأ، مقصد) یا None اگر فایل نباید جابه‌جا شود.
    """
    if should_skip(file):
        return None
    target_dir = root / category_for(file)
    target_dir.mkdir(exist_ok=True)
    dest = unique_path(target_dir / file.name)
    shutil.move(str(file), str(dest))
    return file, dest


def organize_folder(root: Path, log=print):
    """همه‌ی فایل‌های مستقیمِ root را مرتب می‌کند و لیست جابه‌جایی‌ها را برمی‌گرداند.

    فقط فایل‌های سطح اول بررسی می‌شوند، پس فایل‌های داخل زیرپوشه‌ها دست نمی‌خورند.
    """
    root = Path(root)
    moves = []
    if not root.is_dir():
        log(f"پوشه پیدا نشد: {root}")
        return moves

    for item in sorted(root.iterdir()):
        try:
            if should_skip(item):
                if item.is_file():
                    log(f"رد شد (نیمه‌دانلود/سیستمی): {item.name}")
                continue
            result = move_file(item, root)
        except OSError as error:  # مثلاً فایل در حال استفاده است
            log(f"خطا در جابه‌جایی {item.name}: {error}")
            continue
        if result:
            src, dst = result
            moves.append(result)
            log(f"{src.name}  ←  {dst.parent.name}/{dst.name}")

    log(f"تمام شد: {len(moves)} فایل جابه‌جا شد.")
    return moves


def undo_moves(moves, log=print):
    """جابه‌جایی‌ها را برعکس می‌کند (فایل‌ها به جای اولشان برمی‌گردند)."""
    restored = 0
    for src, dst in reversed(moves):
        try:
            if not dst.exists():
                log(f"فایل دیگر وجود ندارد، رد شد: {dst.name}")
                continue
            back = unique_path(src)  # اگر جای قبلی پُر شده بود، روی آن نمی‌نویسیم
            shutil.move(str(dst), str(back))
            restored += 1
            log(f"برگشت: {dst.name}  →  {back.parent.name}/{back.name}")
        except OSError as error:
            log(f"خطا در برگرداندن {dst.name}: {error}")
    log(f"برگرداندن تمام شد: {restored} فایل.")
    return restored


def wait_until_stable(path: Path, checks: int = 3, interval: float = 0.5) -> bool:
    """صبر می‌کند تا اندازه‌ی فایل ثابت شود (یعنی کپی/دانلود تمام شده)."""
    last_size = -1
    stable = 0
    for _ in range(60):  # حداکثر حدود ۳۰ ثانیه
        try:
            size = path.stat().st_size
        except OSError:
            return False  # فایل ناپدید شد
        if size == last_size:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
            last_size = size
        time.sleep(interval)
    return False
