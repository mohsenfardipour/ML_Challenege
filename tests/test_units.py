import datetime as dt

from codal_extractor import codal_api as api
from codal_extractor.jalali import gregorian_to_jalali, jalali_to_gregorian, to_jalali_str
from codal_extractor.parsers import extract_metrics, parse_page, sheet_options
from codal_extractor.textutil import detect_unit, label_key, parse_number
from tests.fixtures import BALANCE, INCOME, statement_html


def test_jalali_roundtrip():
    assert jalali_to_gregorian(1402, 1, 1) == dt.date(2023, 3, 21)
    assert jalali_to_gregorian(1403, 12, 30) == dt.date(2025, 3, 20)  # 1403 is leap
    assert gregorian_to_jalali(dt.date(2024, 3, 20)) == (1403, 1, 1)
    assert to_jalali_str("2023-03-21") == "1402/01/01"
    assert to_jalali_str("۱۴۰۲/۶/۳۱") == "1402/06/31"
    d = dt.date(2020, 1, 1)
    for i in range(0, 3000, 7):
        x = d + dt.timedelta(days=i)
        assert jalali_to_gregorian(*gregorian_to_jalali(x)) == x


def test_parse_number():
    assert parse_number("(1,234)") == -1234
    assert parse_number("۱۲٬۳۴۵") == 12345
    assert parse_number("-1,000") == -1000
    assert parse_number("12.5%") == 12.5
    for x in ("", "-", "abc", None, "—"):
        assert parse_number(x) is None


def test_labels_and_units():
    assert label_key("سود (زیان) خالص") == label_key("سود خالص")
    assert label_key("جمع دارایی‌ها") == label_key("جمع داراییها")
    assert detect_unit("(مبالغ به میلیون ریال)") == "میلیون ریال"
    assert detect_unit("ارقام به هزار ريال") == "هزار ریال"


def test_classify_and_title_meta():
    t = "اطلاعات و صورت‌های مالی میاندوره‌ای دوره ۶ ماهه منتهی به ۱۴۰۲/۰۶/۳۱ (حسابرسی نشده)"
    assert api.classify(t)[0] == "financial_statements"
    m = api.title_meta(t)
    assert (m["period_months"], m["period_end"], m["audited"], m["consolidated"]) == (6, "1402/06/31", False, False)
    assert api.title_meta("صورتهای مالی تلفیقی سال مالی منتهی به ۱۴۰۱/۱۲/۲۹ (حسابرسی شده)")["consolidated"] is True
    assert api.classify("گزارش فعالیت ماهانه دوره ۱ ماهه منتهی به ۱۴۰۲/۰۱/۳۱")[0] == "monthly_reports"
    assert api.classify("آگهی دعوت به مجمع عمومی عادی سالیانه") == ("announcements", "general_assembly")
    assert api.classify("افشای اطلاعات بااهمیت - انعقاد قرارداد")[1] == "material_contract"
    assert api.classify("تصمیمات هیئت مدیره در خصوص افزایش سرمایه")[1] == "capital_increase"


def test_statement_parsing_picks_current_period_column():
    p = parse_page(statement_html("صورت سود و زیان", INCOME))
    assert p["format"] == "datasource" and p["unit"] == "میلیون ریال"
    m = extract_metrics(p["grids"], p["unit"], "1402/06/31")
    assert m["revenue"]["value"] == 100000 and m["revenue"]["value_original"] == "۱۰۰٬۰۰۰"
    assert m["cost_of_revenue"]["value"] == -60000
    assert m["net_profit"]["value"] == 25000 and m["eps"]["value"] == 125
    assert "1402/06/31" in m["revenue"]["column"]
    assert "dps" not in m  # absent stays absent


def test_unit_scaling_and_exact_label_match():
    p = parse_page(statement_html("صورت وضعیت مالی", BALANCE, unit="هزار ریال"))
    m = extract_metrics(p["grids"], p["unit"], "1402/06/31")
    assert m["total_assets"]["value"] == 500.0          # 500,000 thousand IRR -> 500 million
    assert m["total_assets"]["value_parsed"] == 500000  # original kept
    assert m["total_assets"]["label"] == "جمع دارایی ها"  # not "جمع دارایی‌های جاری"


def test_sheet_selector():
    assert [s for s, _ in sheet_options(statement_html("x", INCOME))] == ["0", "1", "9"]


def test_with_sheet():
    u = api.with_sheet("https://www.codal.ir/Reports/Decision.aspx?LetterSerial=A&sheetId=0", 1)
    assert u.endswith("sheetId=1") and u.count("sheetId") == 1
