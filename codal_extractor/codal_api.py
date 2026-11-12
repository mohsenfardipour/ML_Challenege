"""CODAL access: the public search API used by codal.ir's own search page
(search.codal.ir/api/search/v2/q, 20 letters per page) and the letter
pages (www.codal.ir/Reports/Decision.aspx?LetterSerial=...)."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from .http import Fetcher, FetchError
from .jalali import fmt_jalali, parse_jalali
from .textutil import normalize_symbol, normalize_text, to_latin_digits

log = logging.getLogger(__name__)

PAGE_SIZE = 20

# Defaults mirror the codal.ir search form with "all" selected.
BASE_SEARCH_PARAMS = {
    "Audited": "true", "AuditorRef": "-1", "Category": "-1", "Childs": "false",
    "CompanyState": "-1", "CompanyType": "-1", "Consolidatable": "true",
    "IsNotAudited": "false", "Length": "-1", "LetterType": "-1", "Mains": "true",
    "NotAudited": "true", "NotConsolidatable": "true", "Publisher": "false",
    "TracingNo": "-1", "search": "true",
}


def build_search_url(search_url: str, symbol: str, start: str, end: str, page: int,
                     childs: bool = False, extra: Optional[Dict[str, str]] = None) -> str:
    params = dict(BASE_SEARCH_PARAMS)
    params.update({"Symbol": symbol, "FromDate": start, "ToDate": end,
                   "PageNumber": str(page), "Childs": "true" if childs else "false"})
    if extra:
        params.update({k: str(v) for k, v in extra.items()})
    return f"{search_url}?{urlencode(params)}"


def search_letters(fetcher: Fetcher, search_url: str, symbol: str, start: str, end: str,
                   childs: bool = False, extra: Optional[Dict[str, str]] = None,
                   max_pages: int = 500, use_cache: bool = True
                   ) -> Iterator[Tuple[int, Optional[int], Optional[int], List[dict]]]:
    """Yield (page_number, reported_total, reported_pages, letters) for every
    result page. Raises FetchError for a page that cannot be retrieved."""
    page, pages = 1, None
    while True:
        url = build_search_url(search_url, symbol, start, end, page, childs, extra)
        body = fetcher.get(url, use_cache=use_cache, expect_json=True)
        try:
            data = json.loads(body)
        except ValueError as e:
            raise FetchError(url, f"invalid JSON: {e}")
        total = _int(data.get("Total"))
        pages = _int(data.get("Page")) if data.get("Page") is not None else (
            -(-total // PAGE_SIZE) if total is not None else None)
        letters = data.get("Letters") or []
        yield page, total, pages, letters
        if not letters or (pages is not None and page >= pages) or page >= max_pages:
            break
        page += 1


def _int(v) -> Optional[int]:
    try:
        return int(to_latin_digits(str(v)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- classify
_RULES = [  # (category, subtype, title keywords) — first match wins
    ("financial_statements", "financial_statements",
     ["صورتهای مالی", "صورت های مالی", "صورت‌های مالی", "اطلاعات و صورت"]),
    ("monthly_reports", "monthly_activity", ["گزارش فعالیت ماهانه", "فعالیت ماهانه", "گزارش عملکرد ماهانه"]),
    ("announcements", "capital_increase", ["افزایش سرمایه"]),
    ("announcements", "general_assembly", ["مجمع"]),
    ("announcements", "board_decision", ["هیئت مدیره", "هیات مدیره", "هیأت مدیره"]),
    ("announcements", "material_contract", ["قرارداد", "انعقاد"]),
    ("announcements", "production_sales", ["تولید و فروش", "تولید", "فروش"]),
    ("announcements", "important_disclosure", ["اطلاعات بااهمیت", "اطلاعات با اهمیت", "افشای"]),
]
_RULES_N = [(c, s, [normalize_text(k) for k in kws]) for c, s, kws in _RULES]


def classify(title: str, letter_code: str = "") -> Tuple[str, str]:
    t = normalize_text(title)
    code = normalize_text(letter_code).replace(" ", "")
    # Letter-code hints (CODAL form codes): ن-۱۰ interim statements, ن-۳۰ monthly activity.
    if code in ("ن-10", "ن-11"):
        return "financial_statements", "financial_statements"
    if code == "ن-30":
        return "monthly_reports", "monthly_activity"
    for cat, sub, kws in _RULES_N:
        if any(k in t for k in kws):
            # "صورتجلسه مجمع ... صورتهای مالی" is an assembly, not a statement
            if cat == "financial_statements" and "مجمع" in t:
                return "announcements", "general_assembly"
            return cat, sub
    return "announcements", "other"


ANNOUNCEMENT_SUBTYPE_MAP = {
    "important_disclosure": "important_disclosure", "material_contract": "material_contract",
    "capital_increase": "capital_increase", "board_decision": "board_decision",
    "general_assembly": "general_assembly", "production_sales": "production_sales",
    "other": "other",
}


# ---------------------------------------------------------------- letter meta
def letter_record(raw: dict, query_symbol: str, base_url: str) -> dict:
    title = raw.get("Title") or ""
    code = raw.get("LetterCode") or ""
    cat, sub = classify(title, code)
    url = raw.get("Url") or ""
    url = urljoin(base_url + "/", url.lstrip("/")) if url else ""
    tracing = str(raw.get("TracingNo") or "").strip()
    if not tracing:  # fall back to LetterSerial so the record is still unique & traceable
        q = dict(parse_qsl(urlparse(url).query))
        tracing = "LS:" + q.get("LetterSerial", url)
    return {
        "tracing_no": to_latin_digits(tracing), "query_symbol": query_symbol,
        "symbol": normalize_text(raw.get("Symbol") or ""), "company": normalize_text(raw.get("CompanyName") or ""),
        "title": normalize_text(title), "letter_code": normalize_text(code),
        "sent": fmt_jalali(parse_jalali(raw.get("SentDateTime"))) + _time(raw.get("SentDateTime")),
        "published": fmt_jalali(parse_jalali(raw.get("PublishDateTime"))) + _time(raw.get("PublishDateTime")),
        "url": url, "category": cat, "subtype": sub, "raw": raw,
    }


def _time(s) -> str:
    m = re.search(r"\d{1,2}:\d{2}(:\d{2})?", to_latin_digits(str(s or "")))
    return (" " + m.group(0)) if m else ""


def symbol_matches(letter_symbol: str, query_symbol: str) -> bool:
    return normalize_symbol(letter_symbol) == normalize_symbol(query_symbol)


_PERIOD_RE = re.compile(r"(\d+)\s*ماهه\s*منتهی\s*به\s*(\d{4}/\d{1,2}/\d{1,2})")
_YEAR_RE = re.compile(r"سال\s*مالی\s*منتهی\s*به\s*(\d{4}/\d{1,2}/\d{1,2})")


def title_meta(title: str) -> dict:
    """Reporting period, audit and consolidation status from the letter
    title — CODAL titles state these explicitly."""
    t = normalize_text(title)
    months, period_end, fy_end = None, "", ""
    m = _PERIOD_RE.search(t)
    if m:
        months, period_end = int(m.group(1)), fmt_jalali(parse_jalali(m.group(2)))
    y = _YEAR_RE.search(t)
    if y:
        fy_end = fmt_jalali(parse_jalali(y.group(1)))
        if months is None:
            months, period_end = 12, fy_end
    if "حسابرسی نشده" in t:
        audited = False
    elif "حسابرسی شده" in t:
        audited = True
    else:
        audited = None
    # CODAL convention: consolidated letters carry "تلفیقی" in the title;
    # otherwise the letter holds the standalone (شرکت اصلی) statements.
    consolidated = "تلفیقی" in t
    return {
        "period_months": months, "period_end": period_end, "fiscal_year_end": fy_end,
        "audited": audited, "consolidated": consolidated,
        "is_amendment": "اصلاحیه" in t,
        "report_type": ("annual" if months == 12 else "interim" if months else ""),
    }


def with_sheet(url: str, sheet_id) -> str:
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query) if k.lower() != "sheetid"]
    q.append(("sheetId", str(sheet_id)))
    return urlunparse(p._replace(query=urlencode(q)))
