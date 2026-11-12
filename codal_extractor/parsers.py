"""Parsers for CODAL letter pages.

Modern CODAL statement pages (since ~1398) embed the whole report as a JSON
object assigned to a JS variable (``var datasource = {...}``) holding
sheets -> tables -> cells (rowSequence / columnSequence / value). Older or
free-text letters are plain HTML tables. Both are reduced to the same
``Grid`` structure so metric extraction is format-independent.

Every extracted value keeps its original string, the cell address/position,
the column header and the row label, so it can be checked against the source.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from .textutil import detect_unit, label_key, normalize_text, parse_number, unit_factor


@dataclass
class Cell:
    row: int
    col: int
    value: str
    address: str = ""
    header: bool = False


@dataclass
class Grid:
    sheet: str
    table: str
    rows: List[List[Cell]] = field(default_factory=list)

    # ---- derived helpers
    def header_rows(self) -> List[List[Cell]]:
        hdr = [r for r in self.rows if r and all(c.header for c in r)]
        if hdr:
            return hdr
        # no explicit header flag: leading rows with no numeric cell
        out = []
        for r in self.rows:
            if any(parse_number(c.value) is not None for c in r):
                break
            out.append(r)
        return out[:3]

    def data_rows(self) -> List[List[Cell]]:
        hdr = {id(r) for r in self.header_rows()}
        return [r for r in self.rows if id(r) not in hdr]

    def column_headers(self) -> Dict[int, str]:
        heads: Dict[int, List[str]] = {}
        for r in self.header_rows():
            for c in r:
                v = normalize_text(c.value)
                if v:
                    heads.setdefault(c.col, []).append(v)
        return {k: " | ".join(dict.fromkeys(v)) for k, v in heads.items()}

    def label_col(self) -> Optional[int]:
        score: Dict[int, int] = {}
        for r in self.data_rows():
            for c in r:
                v = normalize_text(c.value)
                if v and parse_number(v) is None and re.search(r"[آ-ی]", v):
                    score[c.col] = score.get(c.col, 0) + 1
        return max(score, key=lambda k: (score[k], -k)) if score else None


# ---------------------------------------------------------------- datasource
_DS_RE = re.compile(r"var\s+datasource\s*=\s*", re.I)


def extract_datasource(html: str) -> Optional[dict]:
    m = _DS_RE.search(html or "")
    if not m:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(html, m.end())
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


def _get(d: dict, *names, default=None):
    low = {k.lower(): v for k, v in d.items()} if isinstance(d, dict) else {}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return default


def find_key(obj, name: str):
    """Depth-first search for the first value under key `name` (case-insensitive)."""
    stack = [obj]
    name = name.lower()
    while stack:
        o = stack.pop(0)
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lower() == name and v not in (None, ""):
                    return v
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return None


def grids_from_datasource(ds: dict) -> List[Grid]:
    grids = []
    sheets = _get(ds, "sheets", default=[]) or []
    for si, sh in enumerate(sheets):
        sname = normalize_text(_get(sh, "title_Fa", "title", "titleFa", default="") or f"sheet{si}")
        for ti, tb in enumerate(_get(sh, "tables", default=[]) or []):
            tname = normalize_text(_get(tb, "title_Fa", "title", default="") or f"table{ti}")
            cells = _get(tb, "cells", default=[]) or []
            by_row: Dict[int, List[Cell]] = {}
            for c in cells:
                r = _get(c, "rowSequence", "rowCode")
                k = _get(c, "columnSequence", "columnCode")
                try:
                    r, k = int(r), int(k)
                except (TypeError, ValueError):
                    continue
                val = _get(c, "value", default="")
                by_row.setdefault(r, []).append(Cell(
                    row=r, col=k, value="" if val is None else str(val),
                    address=str(_get(c, "address", default="") or ""),
                    header=str(_get(c, "cellGroupName", default="")).lower() == "header"))
            g = Grid(sname, tname, [sorted(by_row[r], key=lambda c: c.col) for r in sorted(by_row)])
            if g.rows:
                grids.append(g)
    return grids


def datasource_meta(ds: dict) -> dict:
    meta = {}
    for key, out in (("periodEndToDate", "period_end"), ("yearEndToDate", "fiscal_year_end"),
                     ("period", "period_months"), ("isAudited", "audited"),
                     ("isConsolidated", "consolidated"), ("title_Fa", "title")):
        v = find_key(ds, key)
        if v is not None:
            meta[out] = v
    return meta


# ---------------------------------------------------------------- HTML fallback
def grids_from_html(html: str, sheet_name: str = "") -> List[Grid]:
    soup = BeautifulSoup(html or "", "lxml")
    grids = []
    for ti, table in enumerate(soup.find_all("table")):
        if table.find("table"):  # layout wrapper; inner tables are handled separately
            continue
        rows = []
        for ri, tr in enumerate(table.find_all("tr")):
            row, col = [], 0
            for td in tr.find_all(["td", "th"]):
                span = int(td.get("colspan", 1) or 1) if str(td.get("colspan", 1)).isdigit() else 1
                row.append(Cell(ri, col, td.get_text(" ", strip=True), f"T{ti}R{ri}C{col}", td.name == "th"))
                col += span
            if row:
                rows.append(row)
        if len(rows) >= 2:
            grids.append(Grid(sheet_name or "html", f"table{ti}", rows))
    return grids


def selected_sheet(html: str) -> Optional[str]:
    soup = BeautifulSoup(html or "", "lxml")
    sel = soup.find("select", id=re.compile("ddlTable", re.I))
    opt = sel.find("option", selected=True) if sel else None
    return opt.get("value") if opt else None


def sheet_options(html: str) -> List[Tuple[str, str]]:
    """(sheetId, name) pairs from the statement selector on a letter page."""
    soup = BeautifulSoup(html or "", "lxml")
    sel = soup.find("select", id=re.compile("ddlTable", re.I)) or soup.find("select", attrs={"name": re.compile("ddlTable", re.I)})
    if not sel:
        return []
    return [(o.get("value", ""), normalize_text(o.get_text())) for o in sel.find_all("option") if o.get("value") not in (None, "")]


def page_text(html: str) -> str:
    return BeautifulSoup(html or "", "lxml").get_text(" ", strip=True)


def parse_page(html: str, sheet_name: str = "") -> dict:
    """Grids + unit + embedded metadata for one fetched page."""
    ds = extract_datasource(html)
    grids = grids_from_datasource(ds) if ds else []
    fmt = "datasource" if grids else "html"
    if not grids:
        grids = grids_from_html(html, sheet_name)
    unit = None
    if ds:
        unit = detect_unit(json.dumps(ds, ensure_ascii=False))
    if not unit:
        unit = detect_unit(page_text(html))
    return {"grids": grids, "unit": unit, "format": fmt,
            "meta": datasource_meta(ds) if ds else {}}


# ---------------------------------------------------------------- metrics
# metric -> (persian name, ordered exact aliases, is_monetary)
METRICS = {
    "revenue": ("درآمد عملیاتی", ["درآمدهای عملیاتی", "درآمد عملیاتی", "جمع درآمدهای عملیاتی", "فروش خالص",
                                  "درآمدهای عملیاتی خالص", "فروش", "درآمد حاصل از فروش", "فروش خالص و درآمد ارائه خدمات"], True),
    "cost_of_revenue": ("بهای تمام‌شده", ["بهای تمام شده درآمدهای عملیاتی", "بهای تمام شده کالای فروش رفته",
                                         "بهای تمام شده فروش", "بهای تمام شده"], True),
    "gross_profit": ("سود ناخالص", ["سود ناخالص"], True),
    "operating_profit": ("سود عملیاتی", ["سود عملیاتی"], True),
    "net_profit": ("سود خالص", ["سود خالص", "سود خالص دوره", "سود خالص سال", "سود خالص پس از کسر مالیات"], True),
    "eps": ("سود هر سهم (EPS)", ["سود پایه هر سهم", "سود خالص هر سهم", "سود پایه هر سهم ریال",
                                 "سود هر سهم پس از کسر مالیات", "سود خالص هر سهم ریال"], False),
    "dps": ("سود نقدی هر سهم (DPS)", ["سود نقدی هر سهم", "سود سهام مصوب هر سهم", "سود نقدی هر سهم ریال"], False),
    "total_assets": ("جمع دارایی‌ها", ["جمع داراییها", "جمع دارایی ها"], True),
    "total_liabilities": ("جمع بدهی‌ها", ["جمع بدهیها", "جمع بدهی ها"], True),
    "equity": ("حقوق مالکانه", ["جمع حقوق مالکانه", "جمع حقوق صاحبان سهام", "جمع حقوق مالکانه قابل انتساب به مالکان شرکت اصلی",
                                "حقوق مالکانه"], True),
}
_ALIAS_KEYS = {m: [label_key(a) for a in al] for m, (_, al, _) in METRICS.items()}
METRIC_FA = {m: v[0] for m, v in METRICS.items()}
MONETARY = {m for m, v in METRICS.items() if v[2]}


def _pick_column(grid: Grid, period_end: str, label_col: int) -> Tuple[Optional[int], str]:
    heads = grid.column_headers()
    cands = sorted(c for c in {c.col for r in grid.data_rows() for c in r} if c != label_col)
    bad = ("درصد", "تغییر", "%", "یادداشت")
    usable = [c for c in cands if not any(b in heads.get(c, "") for b in bad)]
    if period_end:
        for c in usable:
            if period_end in heads.get(c, ""):
                return c, heads.get(c, "")
    for c in usable:  # first column that actually holds numbers
        if any(parse_number(x.value) is not None for r in grid.data_rows() for x in r if x.col == c):
            return c, heads.get(c, "")
    return None, ""


def extract_metrics(grids: List[Grid], unit: Optional[str], period_end: str = "") -> Dict[str, dict]:
    """Return metric -> {value, value_original, unit_original, label, column,
    sheet, address}. Metrics not found are simply absent (never guessed)."""
    found: Dict[str, Tuple[int, dict]] = {}
    factor = unit_factor(unit)
    for g in grids:
        lc = g.label_col()
        if lc is None:
            continue
        col, head = _pick_column(g, period_end, lc)
        if col is None:
            continue
        for r in g.data_rows():
            lab = next((c for c in r if c.col == lc), None)
            if not lab or not lab.value:
                continue
            k = label_key(lab.value)
            for m, aliases in _ALIAS_KEYS.items():
                if k not in aliases:
                    continue
                rank = aliases.index(k)
                if m in found and found[m][0] <= rank:
                    continue
                cell = next((c for c in r if c.col == col), None)
                raw = cell.value if cell else ""
                v = parse_number(raw)
                std = (v * factor if (v is not None and factor is not None and m in MONETARY)
                       else (v if m not in MONETARY else None))
                found[m] = (rank, {
                    "value": std, "value_parsed": v, "value_original": raw,
                    "unit_original": unit or "", "label": normalize_text(lab.value),
                    "column": head, "sheet": g.sheet, "table": g.table,
                    "address": cell.address if cell else "",
                    "unit_missing": m in MONETARY and v is not None and factor is None,
                })
    return {m: d for m, (_, d) in found.items()}


def grid_long_rows(grids: List[Grid]) -> List[dict]:
    """All non-empty data cells in long format (label, column header, value)."""
    out = []
    for g in grids:
        lc = g.label_col()
        heads = g.column_headers()
        for r in g.data_rows():
            lab = next((c.value for c in r if c.col == lc), "") if lc is not None else ""
            for c in r:
                if c.col == lc or not normalize_text(c.value):
                    continue
                out.append({"sheet": g.sheet, "table": g.table, "row": c.row, "col": c.col,
                            "address": c.address, "label": normalize_text(lab),
                            "column": heads.get(c.col, ""), "value_original": c.value,
                            "value": parse_number(c.value)})
    return out


def announcement_text(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    main = soup.find(id=re.compile("(ctl00_)?(Content|Main|Body)", re.I)) or soup.body or soup
    return normalize_text(main.get_text(" ", strip=True))[:max_chars]
