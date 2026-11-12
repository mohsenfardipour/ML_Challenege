"""Validation rules. Records are never dropped here — problems are reported
as issues and flagged on the record itself."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .jalali import parse_jalali

REL_TOL = 0.01      # 1 % tolerance for accounting identities
ABS_TOL = 2.0       # rounding tolerance in million IRR


def _close(a, b) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def financial_checks(row: dict) -> List[dict]:
    """Return issues for one financial-statement row (metric values in a
    common unit)."""
    out = []
    g = lambda k: row.get(k)
    rev, cost, gp = g("revenue"), g("cost_of_revenue"), g("gross_profit")
    if None not in (rev, cost, gp):
        expected = rev + cost if cost < 0 else rev - cost
        if not _close(expected, gp):
            out.append(("gross_profit", f"سود ناخالص ({gp:,.0f}) با درآمد±بهای تمام‌شده ({expected:,.0f}) همخوانی ندارد"))
    ta, tl, eq = g("total_assets"), g("total_liabilities"), g("equity")
    if None not in (ta, tl, eq) and not _close(ta, tl + eq):
        # equity may exclude non-controlling interests in consolidated statements
        out.append(("total_assets", f"جمع دارایی‌ها ({ta:,.0f}) ≠ بدهی‌ها + حقوق مالکانه ({tl + eq:,.0f})"))
    np_, eps = g("net_profit"), g("eps")
    if None not in (np_, eps) and np_ != 0 and eps != 0 and (np_ > 0) != (eps > 0):
        out.append(("eps", "علامت EPS با علامت سود خالص متفاوت است"))
    if None not in (gp, g("operating_profit"), rev) and rev > 0 and abs(g("operating_profit")) > 5 * rev:
        out.append(("operating_profit", "سود عملیاتی بیش از ۵ برابر درآمد — احتمال خطای واحد"))
    if rev is not None and rev < 0:
        out.append(("revenue", "درآمد عملیاتی منفی"))
    if ta is not None and ta <= 0:
        out.append(("total_assets", "جمع دارایی‌ها صفر یا منفی"))
    pe, pub = parse_jalali(row.get("period_end")), parse_jalali(row.get("published"))
    if pe and pub and pe > pub:
        out.append(("period_end", "پایان دوره گزارش بعد از تاریخ انتشار است"))
    if not row.get("period_end"):
        out.append(("period_end", "دوره گزارشگری مشخص نشد"))
    return [{"kind": "validation", "severity": "warning", "field": f, "message": m} for f, m in out]


def revision_groups(fin_rows: List[dict]) -> Dict[tuple, List[dict]]:
    """Several letters for the same symbol/period/scope/audit (e.g. original
    + اصلاحیه). All are kept; caller flags them."""
    groups = defaultdict(list)
    for r in fin_rows:
        if r.get("period_end"):
            groups[(r["symbol_query"], r["period_end"], r.get("period_months"),
                    r.get("consolidated"), r.get("audited"))].append(r)
    return {k: v for k, v in groups.items() if len(v) > 1}
