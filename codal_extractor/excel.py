"""Workbook writer: six sheets, Persian headers, Excel tables (filters),
frozen headers, fitted widths, number formats, clickable source URLs."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

INT = "#,##0;[Red](#,##0)"
DEC = "#,##0.00;[Red](#,##0.00)"
TXT = "@"
ColSpec = Tuple[str, str, str]   # (record key, Persian header, number format or "")

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=13)


def _width(v) -> int:
    if v is None:
        return 0
    if isinstance(v, float):
        return len(f"{v:,.2f}")
    return len(str(v))


def write_table(ws, cols: Sequence[ColSpec], rows: List[dict], name: str,
                start_row: int = 1, title: str = "", freeze: bool = True) -> int:
    """Write rows as an Excel table. Returns the next free row."""
    r0 = start_row
    if title:
        ws.cell(r0, 1, title).font = TITLE_FONT
        r0 += 1
    for j, (_, head, _) in enumerate(cols, 1):
        c = ws.cell(r0, j, head)
        c.fill, c.font = HEADER_FILL, HEADER_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    widths = [max(10, len(h) + 2) for _, h, _ in cols]
    for i, rec in enumerate(rows, 1):
        for j, (key, _, fmt) in enumerate(cols, 1):
            v = rec.get(key)
            if isinstance(v, bool):
                v = "بله" if v else "خیر"
            elif isinstance(v, (list, dict)):
                v = str(v)
            c = ws.cell(r0 + i, j, v)
            if fmt and isinstance(v, (int, float)):
                c.number_format = fmt
            if key.endswith("url") and isinstance(v, str) and v.startswith("http"):
                c.hyperlink = v
                c.font = Font(color="0563C1", underline="single")
            widths[j - 1] = max(widths[j - 1], min(60, _width(v) + 2))
    for j, w in enumerate(widths, 1):
        col = get_column_letter(j)
        ws.column_dimensions[col].width = max(ws.column_dimensions[col].width or 0, w)
    last = r0 + max(len(rows), 1)
    if not rows:  # an Excel table needs at least one body row
        ws.cell(r0 + 1, 1, "— بدون رکورد —")
    ref = f"A{r0}:{get_column_letter(len(cols))}{last}"
    t = Table(displayName=name, ref=ref)
    t.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    ws.add_table(t)
    if freeze and start_row == 1 and not title:
        ws.freeze_panes = ws.cell(r0 + 1, 1)
    return last + 2


# ---------------------------------------------------------------- column specs
FIN_COLS: List[ColSpec] = [
    ("symbol_query", "نماد", TXT), ("symbol", "نماد در کدال", TXT), ("company", "نام شرکت", TXT),
    ("tracing_no", "شماره پیگیری", TXT), ("letter_code", "کد اطلاعیه", TXT),
    ("report_type", "نوع گزارش (سالانه/میاندوره‌ای)", TXT), ("period_months", "طول دوره (ماه)", INT),
    ("period_end", "پایان دوره", TXT), ("fiscal_year_end", "پایان سال مالی", TXT),
    ("scope_fa", "تلفیقی/شرکت اصلی", TXT), ("audit_fa", "وضعیت حسابرسی", TXT),
    ("is_amendment", "اصلاحیه", TXT), ("published", "تاریخ انتشار", TXT),
    ("statements_found", "صورت‌های یافت‌شده", TXT),
    ("revenue", "درآمد عملیاتی (م.ریال)", INT), ("cost_of_revenue", "بهای تمام‌شده (م.ریال)", INT),
    ("gross_profit", "سود ناخالص (م.ریال)", INT), ("operating_profit", "سود عملیاتی (م.ریال)", INT),
    ("net_profit", "سود خالص (م.ریال)", INT), ("eps", "EPS (ریال)", INT), ("dps", "DPS (ریال)", INT),
    ("total_assets", "جمع دارایی‌ها (م.ریال)", INT), ("total_liabilities", "جمع بدهی‌ها (م.ریال)", INT),
    ("equity", "حقوق مالکانه (م.ریال)", INT),
    ("unit_original", "واحد اصلی گزارش", TXT), ("column_used", "ستون منبع ارقام", TXT),
    ("flags", "هشدارها", TXT), ("title", "عنوان اطلاعیه", TXT), ("url", "لینک منبع (url)", TXT),
]
MONTHLY_COLS: List[ColSpec] = [
    ("symbol_query", "نماد", TXT), ("company", "نام شرکت", TXT), ("tracing_no", "شماره پیگیری", TXT),
    ("period_months", "طول دوره (ماه)", INT), ("period_end", "پایان دوره", TXT),
    ("published", "تاریخ انتشار", TXT), ("is_amendment", "اصلاحیه", TXT),
    ("sheet", "برگه", TXT), ("table", "جدول", TXT), ("label", "شرح / محصول", TXT),
    ("column", "ستون", TXT), ("value", "مقدار", DEC), ("value_original", "مقدار اصلی", TXT),
    ("unit_original", "واحد مبالغ گزارش", TXT), ("url", "لینک منبع (url)", TXT),
]
ANN_COLS: List[ColSpec] = [
    ("symbol_query", "نماد", TXT), ("symbol", "نماد در کدال", TXT), ("company", "نام شرکت", TXT),
    ("tracing_no", "شماره پیگیری", TXT), ("letter_code", "کد اطلاعیه", TXT),
    ("subtype_fa", "نوع اطلاعیه", TXT), ("title", "عنوان", TXT), ("sent", "تاریخ ارسال", TXT),
    ("published", "تاریخ انتشار", TXT), ("has_attachment", "پیوست دارد", TXT),
    ("body", "متن (خلاصه)", TXT), ("attachment_url", "لینک پیوست (url)", TXT),
    ("url", "لینک منبع (url)", TXT),
]
RAW_COLS: List[ColSpec] = [
    ("symbol_query", "نماد", TXT), ("tracing_no", "شماره پیگیری", TXT), ("category_fa", "دسته", TXT),
    ("source", "منبع/فرمت", TXT), ("sheet", "برگه", TXT), ("table", "جدول", TXT),
    ("row", "ردیف", INT), ("col", "ستون#", INT), ("address", "آدرس سلول", TXT),
    ("label", "شرح ردیف", TXT), ("column", "عنوان ستون", TXT),
    ("value_original", "مقدار اصلی", TXT), ("value", "مقدار عددی", DEC), ("url", "لینک منبع (url)", TXT),
]
LETTER_COLS: List[ColSpec] = [
    ("symbol_query", "نماد", TXT), ("tracing_no", "شماره پیگیری", TXT), ("category_fa", "دسته", TXT),
    ("title", "عنوان", TXT), ("letter_code", "کد", TXT), ("sent", "ارسال", TXT),
    ("published", "انتشار", TXT), ("status_fa", "وضعیت پردازش", TXT), ("url", "لینک منبع (url)", TXT),
]
DQ_COLS: List[ColSpec] = [
    ("kind_fa", "نوع مشکل", TXT), ("severity_fa", "شدت", TXT), ("symbol", "نماد", TXT),
    ("tracing_no", "شماره پیگیری", TXT), ("field", "فیلد", TXT), ("message", "شرح", TXT),
    ("url", "لینک منبع (url)", TXT),
]
SUMMARY_COLS: List[ColSpec] = [
    ("symbol", "نماد", TXT), ("company", "نام شرکت", TXT), ("reported_total", "تعداد گزارش‌شده توسط کدال", INT),
    ("letters", "اطلاعیه‌های دریافت‌شده", INT), ("financial", "صورت‌های مالی", INT),
    ("monthly", "گزارش‌های ماهانه", INT), ("announcements", "سایر اطلاعیه‌ها", INT),
    ("failed", "ناموفق", INT), ("first_date", "اولین انتشار", TXT), ("last_date", "آخرین انتشار", TXT),
    ("search_complete", "جستجو کامل", TXT),
]
KV_COLS: List[ColSpec] = [("k", "شاخص", TXT), ("v", "مقدار", TXT)]


def build_workbook(path: str, summary_kv: List[dict], coverage: List[dict], fin: List[dict],
                   monthly: List[dict], ann: List[dict], raw: List[dict], letters: List[dict],
                   dq: List[dict]):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    nxt = write_table(ws, KV_COLS, summary_kv, "tblRun", title="خلاصه اجرا (Run summary)")
    write_table(ws, SUMMARY_COLS, coverage, "tblCoverage", start_row=nxt, title="پوشش بر اساس نماد (Coverage)")
    ws.freeze_panes = "A3"
    sheets: Dict[str, tuple] = {
        "Financial Statements": (FIN_COLS, fin, "tblFinancial"),
        "Monthly Reports": (MONTHLY_COLS, monthly, "tblMonthly"),
        "Announcements": (ANN_COLS, ann, "tblAnnouncements"),
    }
    for title, (cols, rows, tname) in sheets.items():
        write_table(wb.create_sheet(title), cols, rows, tname)
    wr = wb.create_sheet("Raw Data")
    nxt = write_table(wr, LETTER_COLS, letters, "tblLetters", title="فهرست کامل اطلاعیه‌ها (All letters, search metadata)")
    write_table(wr, RAW_COLS, raw, "tblRawCells", start_row=nxt, title="مقادیر اصلی استخراج‌شده (Original cell values)")
    wr.freeze_panes = "A3"
    write_table(wb.create_sheet("Data Quality"), DQ_COLS, dq, "tblQuality")
    for s in wb.worksheets:
        s.sheet_view.rightToLeft = True
    wb.save(path)
