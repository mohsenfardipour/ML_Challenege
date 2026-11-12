"""Jalali (Solar Hijri) <-> Gregorian conversion (jalaali algorithm,
no third-party dependency)."""
from __future__ import annotations

import datetime as _dt
import re
from typing import Optional, Tuple

from .textutil import to_latin_digits

_BREAKS = [-61, 9, 38, 199, 426, 686, 756, 818, 1111, 1181, 1210, 1635,
           2060, 2097, 2192, 2262, 2324, 2394, 2456, 3178]


def _div(a, b):
    return int(a / b)


def _jal_cal(jy):
    bl = len(_BREAKS)
    gy = jy + 621
    leap_j = -14
    jp = _BREAKS[0]
    if jy < jp or jy >= _BREAKS[-1]:
        raise ValueError(f"Invalid Jalali year {jy}")
    jump = 0
    for i in range(1, bl):
        jm = _BREAKS[i]
        jump = jm - jp
        if jy < jm:
            break
        leap_j += _div(jump, 33) * 8 + _div(jump % 33, 4)
        jp = jm
    n = jy - jp
    leap_j += _div(n, 33) * 8 + _div(n % 33 + 3, 4)
    if jump % 33 == 4 and jump - n == 4:
        leap_j += 1
    leap_g = _div(gy, 4) - _div((_div(gy, 100) + 1) * 3, 4) - 150
    march = 20 + leap_j - leap_g
    if jump - n < 6:
        n = n - jump + _div(jump + 4, 33) * 33
    leap = (((n + 1) % 33) - 1) % 4
    if leap == -1:
        leap = 4
    return leap, gy, march


def _g2d(gy, gm, gd):
    d = _div((gy + _div(gm - 8, 6) + 100100) * 1461, 4) + _div(153 * ((gm + 9) % 12) + 2, 5) + gd - 34840408
    return d - _div(_div(gy + 100100 + _div(gm - 8, 6), 100) * 3, 4) + 752


def _d2g(jdn):
    j = 4 * jdn + 139361631
    j = j + _div(_div(4 * jdn + 183187720, 146097) * 3, 4) * 4 - 3908
    i = _div(j % 1461, 4) * 5 + 308
    gd = _div(i % 153, 5) + 1
    gm = (_div(i, 153) % 12) + 1
    gy = _div(j, 1461) - 100100 + _div(8 - gm, 6)
    return gy, gm, gd


def _j2d(jy, jm, jd):
    _, gy, march = _jal_cal(jy)
    return _g2d(gy, 3, march) + (jm - 1) * 31 - _div(jm, 7) * (jm - 7) + jd - 1


def _d2j(jdn):
    gy = _d2g(jdn)[0]
    jy = gy - 621
    leap, _, march = _jal_cal(jy)
    jdn1f = _g2d(gy, 3, march)
    k = jdn - jdn1f
    if k >= 0:
        if k <= 185:
            return jy, 1 + _div(k, 31), (k % 31) + 1
        k -= 186
    else:
        jy -= 1
        k += 179
        if leap == 1:
            k += 1
    return jy, 7 + _div(k, 30), (k % 30) + 1


def jalali_to_gregorian(jy, jm, jd) -> _dt.date:
    return _dt.date(*_d2g(_j2d(jy, jm, jd)))


def gregorian_to_jalali(d: _dt.date) -> Tuple[int, int, int]:
    return _d2j(_g2d(d.year, d.month, d.day))


_DATE_RE = re.compile(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})")


def parse_jalali(s) -> Optional[Tuple[int, int, int]]:
    """'۱۴۰۲/۰۶/۳۱ ۱۲:۰۰:۰۰' -> (1402, 6, 31). None if not parseable."""
    if not s:
        return None
    m = _DATE_RE.search(to_latin_digits(str(s)))
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return y, mo, d


def fmt_jalali(t: Optional[Tuple[int, int, int]]) -> str:
    return f"{t[0]:04d}/{t[1]:02d}/{t[2]:02d}" if t else ""


def to_jalali_str(user_date: str) -> str:
    """Accept a user date as Jalali (yyyy/mm/dd, year < 1700) or Gregorian
    ISO (yyyy-mm-dd, year >= 1700) and return Jalali 'yyyy/mm/dd' as CODAL
    expects."""
    t = parse_jalali(user_date)
    if not t:
        raise ValueError(f"Unrecognised date: {user_date!r}")
    if t[0] >= 1700:
        t = gregorian_to_jalali(_dt.date(*t))
    else:
        jalali_to_gregorian(*t)  # validates
    return fmt_jalali(t)


def jalali_str_to_gregorian(s) -> Optional[_dt.date]:
    t = parse_jalali(s)
    if not t:
        return None
    try:
        return jalali_to_gregorian(*t)
    except ValueError:
        return None
