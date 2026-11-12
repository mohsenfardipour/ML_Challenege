"""Persian text, number and unit normalisation helpers.

Everything here is pure (no I/O) so it can be unit-tested offline.
"""
from __future__ import annotations

import re
from typing import Optional

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(_PERSIAN_DIGITS)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_ARABIC_DIGITS)})
_CHAR_MAP = {
    ord("ي"): "ی", ord("ى"): "ی", ord("ك"): "ک", ord("ة"): "ه",
    ord("ۀ"): "ه", ord("أ"): "ا", ord("إ"): "ا", ord("ؤ"): "و",
    ord("‌"): " ", ord("‏"): "", ord("‎"): "", ord("\xa0"): " ",
    ord("ـ"): "",
}


def to_latin_digits(s: str) -> str:
    return s.translate(_DIGIT_MAP)


def normalize_text(s) -> str:
    """Canonical Persian text: Arabic letters -> Persian, ZWNJ -> space,
    Persian digits -> Latin, collapsed whitespace."""
    if s is None:
        return ""
    s = to_latin_digits(str(s)).translate(_CHAR_MAP)
    return re.sub(r"\s+", " ", s).strip()


def label_key(s) -> str:
    """Aggressive key used to match row labels across report versions:
    drops '(زیان)', punctuation and all whitespace."""
    s = normalize_text(s)
    s = re.sub(r"\(\s*زیان\s*\)", "", s)
    s = s.replace("(زیان)", "")
    s = re.sub(r"[\s\-–_:،,.()\[\]«»\"'/]", "", s)
    return s


def normalize_symbol(s) -> str:
    return re.sub(r"\s+", "", normalize_text(s))


_NUM_CLEAN = re.compile(r"[,٬،٬\s]")


def parse_number(raw) -> Optional[float]:
    """Parse a CODAL numeric cell. Handles Persian digits, thousand
    separators, parentheses / trailing minus for negatives, percent sign.
    Returns None for blanks, dashes or non-numeric text (never guesses)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = normalize_text(raw)
    if s in {"", "-", "--", "—", "–", "ـ", "*", "N/A", "n/a"}:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    s = s.replace("%", "").replace("٪", "")
    s = _NUM_CLEAN.sub("", s)
    s = s.replace("/", ".") if re.fullmatch(r"-?\d+/\d+", s) else s  # Persian decimal mark
    if s.endswith("-") and s[:-1].replace(".", "", 1).isdigit():
        neg, s = True, s[:-1]
    if s.startswith("-"):
        neg, s = (not neg), s[1:]
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    v = float(s)
    return -v if neg else v


# Unit factor relative to "میلیون ریال" (million IRR), the CODAL default.
UNIT_FACTORS = [
    ("میلیارد ریال", 1_000.0),
    ("میلیون ریال", 1.0),
    ("هزار ریال", 0.001),
    ("ریال", 0.000001),
]


def detect_unit(text: str) -> Optional[str]:
    """Find the 'مبالغ به ... ریال' unit statement in a page."""
    t = normalize_text(text)
    m = re.search(r"(?:مبالغ|ارقام)\s*(?:به|بر\s*حسب)\s*(میلیارد ریال|میلیون ریال|هزار ریال|ریال)", t)
    if m:
        return m.group(1)
    for name, _ in UNIT_FACTORS:
        if f"({name})" in t:
            return name
    return None


def unit_factor(unit: Optional[str]) -> Optional[float]:
    if not unit:
        return None
    for name, f in UNIT_FACTORS:
        if normalize_text(unit) == name:
            return f
    return None
