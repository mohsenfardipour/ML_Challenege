"""Persian text clean-up used after every transcription, and for scoring (WER/CER)."""

from __future__ import annotations

import re

ZWNJ = "\u200c"   # half-space (نیم‌فاصله)

_ARABIC_TO_PERSIAN = str.maketrans({"\u064a": "\u06cc", "\u0649": "\u06cc", "\u0643": "\u06a9", "\u0640": ""})  # ي ى ك -> ی ی ک, tatweel removed
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

# می / نمی + verb  ->  می‌رود ، نمی‌رود
_MI = re.compile(r"(?<![\w\u200c])(ن?می) +(?=[\u0600-\u06FF])")
# word + ها / های / هایی / هایم ...  ->  کتاب‌ها
_HA = re.compile(r"(?<=[\u0600-\u06FF]) +(ها|های|هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان)(?![\w\u200c])")
_SPACES = re.compile(r"[ \t]+")


def normalize_chars(text: str) -> str:
    """Arabic ي/ك -> Persian ی/ک, remove tatweel."""
    return text.translate(_ARABIC_TO_PERSIAN)


def fix_half_spaces(text: str) -> str:
    """Put a half-space after می/نمی and before ها/های where a normal space was written."""
    text = _MI.sub(lambda m: m.group(1) + ZWNJ, text)
    return _HA.sub(lambda m: ZWNJ + m.group(1), text)


def clean_transcript(text: str) -> str:
    """Normalization applied to every transcript shown to the user."""
    text = fix_half_spaces(normalize_chars(text))
    return "\n".join(_SPACES.sub(" ", line).strip() for line in text.splitlines())


# ---------------------------------------------------------------- scoring
_PUNCT = re.compile(r"[^\w\s]|_")


def normalize_for_scoring(text: str) -> str:
    """Lenient form for WER/CER: no punctuation, no diacritics, half-space = space."""
    text = _DIACRITICS.sub("", normalize_chars(text)).replace(ZWNJ, " ")
    return " ".join(_PUNCT.sub(" ", text).split())
