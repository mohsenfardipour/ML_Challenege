"""SYNTHETIC test fixtures shaped like CODAL responses. These numbers are
invented for unit tests only and never reach a real workbook."""
import json

SYM = "تست"


def letter(tn, title, code="ن-۱۰", serial=None, pub="۱۴۰۲/۰۵/۰۱ ۱۰:۰۰:۰۰", symbol=SYM):
    return {"TracingNo": tn, "Symbol": symbol, "CompanyName": "شركت تست", "Title": title,
            "LetterCode": code, "SentDateTime": pub, "PublishDateTime": pub, "HasHtml": True,
            "HasAttachment": False, "AttachmentUrl": "",
            "Url": f"/Reports/Decision.aspx?LetterSerial={serial or tn}&rt=0&let=6&ct=0&ft=-1"}


def cell(r, c, v, header=False):
    return {"rowSequence": r, "columnSequence": c, "value": v, "address": f"R{r}C{c}",
            "cellGroupName": "Header" if header else "Body"}


def statement_html(sheet_title, rows, period_end="۱۴۰۲/۰۶/۳۱", unit="میلیون ریال", selector=True):
    head = [cell(1, 1, "شرح", True), cell(1, 2, "یادداشت", True),
            cell(1, 3, f"دوره ۶ ماهه منتهی به {period_end}", True),
            cell(1, 4, "دوره ۶ ماهه منتهی به ۱۴۰۱/۰۶/۳۱", True), cell(1, 5, "درصد تغییر", True)]
    body = []
    for i, (lab, cur, prev) in enumerate(rows, 2):
        body += [cell(i, 1, lab), cell(i, 2, "5"), cell(i, 3, cur), cell(i, 4, prev), cell(i, 5, "10")]
    ds = {"sheets": [{"title_Fa": sheet_title, "tables": [{"title_Fa": sheet_title, "cells": head + body}]}],
          "periodEndToDate": period_end, "yearEndToDate": "۱۴۰۲/۱۲/۲۹", "isAudited": False}
    sel = ("<select id='ctl00_cphBody_ddlTable'><option value='0'>صورت وضعیت مالی</option>"
           "<option value='1'>صورت سود و زیان</option><option value='9'>صورت جریان‌های نقدی</option></select>"
           if selector else "")
    return (f"<html><body>{sel}<span>مبالغ به {unit}</span>"
            f"<script>var datasource = {json.dumps(ds, ensure_ascii=False)};</script></body></html>")


INCOME = [("درآمدهای عملیاتی", "۱۰۰٬۰۰۰", "80,000"), ("بهاى تمام شده درآمدهای عملیاتی", "(60,000)", "(50,000)"),
          ("سود (زیان) ناخالص", "40,000", "30,000"), ("سود (زیان) عملیاتی", "30,000", "20,000"),
          ("سود (زیان) خالص", "25,000", "15,000"), ("سود (زیان) پایه هر سهم", "125", "75")]
BALANCE = [("جمع دارایی‌های جاری", "300,000", "1"), ("جمع دارایی‌ها", "500,000", "400,000"),
           ("جمع بدهی‌ها", "200,000", "150,000"), ("جمع حقوق مالکانه", "300,000", "250,000")]
MONTHLY = [cell(1, 1, "نام محصول", True), cell(1, 2, "واحد", True), cell(1, 3, "تعداد تولید", True),
           cell(1, 4, "مبلغ فروش", True), cell(2, 1, "محصول الف"), cell(2, 2, "تن"), cell(2, 3, "1,200"),
           cell(2, 4, "45,000"), cell(3, 1, "جمع"), cell(3, 3, "1,200"), cell(3, 4, "45,000")]


def monthly_html():
    ds = {"sheets": [{"title_Fa": "تولید و فروش", "tables": [{"title_Fa": "تولید و فروش", "cells": MONTHLY}]}]}
    return f"<html><body>مبالغ به میلیون ریال<script>var datasource = {json.dumps(ds, ensure_ascii=False)};</script></body></html>"
