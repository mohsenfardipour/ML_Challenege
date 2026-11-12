import json

from openpyxl import load_workbook

from codal_extractor.config import Config
from codal_extractor.http import Fetcher, FetchError
from codal_extractor.pipeline import Pipeline
from tests import fixtures as fx


class FakeFetcher(Fetcher):
    """Serves synthetic responses; counts calls so caching can be asserted."""

    def __init__(self, tmp, routes):
        super().__init__(tmp / "cache", min_interval=0)
        self.routes, self.calls = routes, []

    def get(self, url, use_cache=True, expect_json=False):
        self.calls.append(url)
        for pred, body in self.routes:
            if pred(url):
                if isinstance(body, Exception):
                    raise body
                return body
        raise FetchError(url, "no route", 404)


FIN_T = "اطلاعات و صورت‌های مالی میاندوره‌ای دوره ۶ ماهه منتهی به ۱۴۰۲/۰۶/۳۱ (حسابرسی نشده)"


def _routes():
    p1 = {"Total": 4, "Page": 2, "Letters": [fx.letter("1001", FIN_T, serial="S1"),
                                            fx.letter("1002", "گزارش فعالیت ماهانه دوره ۱ ماهه منتهی به ۱۴۰۲/۰۱/۳۱", "ن-۳۰"),
                                            fx.letter("1003", "افشای اطلاعات بااهمیت - انعقاد قرارداد", "ن-۴۱")]}
    p2 = {"Total": 4, "Page": 2, "Letters": [fx.letter("1003", "افشای اطلاعات بااهمیت - انعقاد قرارداد", "ن-۴۱"),
                                            fx.letter("1004", FIN_T.replace("۶ ماهه", "۳ ماهه").replace("۰۶/۳۱", "۰۳/۳۱"),
                                                      serial="BROKEN")]}
    return [
        (lambda u: "search.codal.ir" in u and "PageNumber=1" in u, json.dumps(p1, ensure_ascii=False)),
        (lambda u: "search.codal.ir" in u and "PageNumber=2" in u, json.dumps(p2, ensure_ascii=False)),
        (lambda u: "LetterSerial=S1" in u and "sheetId=0" in u, fx.statement_html("صورت وضعیت مالی", fx.BALANCE)),
        (lambda u: "LetterSerial=S1" in u and "sheetId=9" in u, fx.statement_html("صورت جریان‌های نقدی", [])),
        (lambda u: "LetterSerial=S1" in u, fx.statement_html("صورت سود و زیان", fx.INCOME)),
        (lambda u: "LetterSerial=1002" in u, fx.monthly_html()),
        (lambda u: "LetterSerial=BROKEN" in u, FetchError("x", "blocked", 403)),
    ]


def test_end_to_end(tmp_path):
    cfg = Config(symbols=[fx.SYM], start_date="1402/01/01", end_date="1402/12/29",
                 work_dir=str(tmp_path / "w"), output=str(tmp_path / "out.xlsx"), verify_sample=1)
    f = FakeFetcher(tmp_path, _routes())
    rep = Pipeline(cfg, fetcher=f).run()
    assert rep["letters_found"] == 4            # duplicate 1003 stored once
    assert rep["failed"] == 1 and rep["ok"] == 3  # failed record kept & reported
    wb = load_workbook(tmp_path / "out.xlsx")
    assert wb.sheetnames == ["Summary", "Financial Statements", "Monthly Reports", "Announcements", "Raw Data", "Data Quality"]
    ws = wb["Financial Statements"]
    hdr = [c.value for c in ws[1]]
    rows = [dict(zip(hdr, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
    ok = next(r for r in rows if r["شماره پیگیری"] == "1001")
    assert ok["درآمد عملیاتی (م.ریال)"] == 100000 and ok["جمع دارایی‌ها (م.ریال)"] == 500000
    assert ok["DPS (ریال)"] is None                       # not in document -> empty, not invented
    assert ok["لینک منبع (url)"].startswith("https://www.codal.ir/Reports/Decision.aspx")
    bad = next(r for r in rows if r["شماره پیگیری"] == "1004")
    assert bad["درآمد عملیاتی (م.ریال)"] is None and "ناموفق" in bad["هشدارها"]
    assert ws.freeze_panes == "A2" and ws.tables["tblFinancial"].ref.startswith("A1:")
    assert ws.sheet_view.rightToLeft
    dq = [[c.value for c in r] for r in wb["Data Quality"].iter_rows(min_row=2)]
    kinds = {r[0] for r in dq}
    assert {"تکراری", "خطای دریافت", "اعتبارسنجی", "مقایسه با منبع"} <= kinds
    assert any("بعد از تاریخ انتشار" in (r[5] or "") for r in dq)  # fixture publishes before period end
    mon = list(wb["Monthly Reports"].iter_rows(min_row=2, values_only=True))
    assert any("محصول الف" in (r[9] or "") for r in mon)
    assert len(list(wb["Announcements"].iter_rows(min_row=2))) == 1

    # rerun: successful letters are not re-fetched, failed one is retried
    f2 = FakeFetcher(tmp_path, _routes())
    Pipeline(cfg, fetcher=f2).run()
    letter_calls = [u for u in f2.calls if "Decision.aspx" in u]
    assert all("BROKEN" in u or "S1" in u for u in letter_calls)  # only retry + verification sample
    assert not any("search.codal.ir" in u for u in f2.calls)       # complete search reused


def test_filters(tmp_path):
    cfg = Config(symbols=[fx.SYM], start_date="2023-03-21", end_date="1402/12/29", categories=["financial_statements"],
                 period_months=[6], work_dir=str(tmp_path / "w"), output=str(tmp_path / "o.xlsx"), verify_sample=0)
    rep = Pipeline(cfg, fetcher=FakeFetcher(tmp_path, _routes())).run()
    assert rep["selected"] == 1 and rep["ok"] == 1 and rep["financial_rows"] == 1


def test_missing_values_reported_not_invented(tmp_path):
    t = "صورت‌های مالی سال مالی منتهی به ۱۴۰۲/۱۲/۲۹ (حسابرسی شده)"
    page = {"Total": 1, "Page": 1, "Letters": [fx.letter("2001", t, pub="۱۴۰۳/۰۳/۰۱ ۱۰:۰۰:۰۰")]}
    html = fx.statement_html("صورت سود و زیان", fx.INCOME[:2], period_end="۱۴۰۲/۱۲/۲۹", selector=False)
    routes = [(lambda u: "search.codal.ir" in u, json.dumps(page, ensure_ascii=False)),
              (lambda u: "LetterSerial=2001" in u, html)]
    cfg = Config(symbols=[fx.SYM], start_date="1402/01/01", end_date="1403/12/30", work_dir=str(tmp_path / "w"),
                 output=str(tmp_path / "o.xlsx"), verify_sample=0)
    rep = Pipeline(cfg, fetcher=FakeFetcher(tmp_path, routes)).run()
    assert rep["ok"] == 1
    wb = load_workbook(tmp_path / "o.xlsx")
    dq = [r for r in wb["Data Quality"].iter_rows(min_row=2, values_only=True)]
    missing = {r[4] for r in dq if r[0] == "مقدار ناموجود"}
    assert {"سود خالص", "جمع دارایی‌ها", "سود هر سهم (EPS)"} <= missing
    row = list(wb["Financial Statements"].iter_rows(min_row=2, values_only=True))[0]
    hdr = [c.value for c in wb["Financial Statements"][1]]
    rec = dict(zip(hdr, row))
    assert rec["سود خالص (م.ریال)"] is None and rec["درآمد عملیاتی (م.ریال)"] == 100000
    assert rec["وضعیت حسابرسی"] == "حسابرسی‌شده" and rec["نوع گزارش (سالانه/میاندوره‌ای)"] == "annual"
