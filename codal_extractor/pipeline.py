"""End-to-end orchestration:  search -> filter -> fetch/parse -> validate ->
verify -> workbook.  Every stage persists to the SQLite store, so a rerun
resumes and only re-tries what failed."""
from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from . import codal_api as api
from .config import Config
from .excel import build_workbook
from .http import Fetcher, FetchError
from .jalali import parse_jalali
from .parsers import (METRIC_FA, METRICS, announcement_text, extract_metrics, grid_long_rows, parse_page,
                      selected_sheet, sheet_options)
from .store import Store
from .textutil import normalize_text
from .validate import financial_checks, revision_groups

log = logging.getLogger(__name__)

PARSER_VERSION = 1   # bump to force re-parsing of cached letters

CAT_FA = {"financial_statements": "صورت‌های مالی", "monthly_reports": "گزارش ماهانه", "announcements": "اطلاعیه"}
SUB_FA = {
    "financial_statements": "صورت‌های مالی", "monthly_activity": "گزارش فعالیت ماهانه",
    "important_disclosure": "افشای اطلاعات بااهمیت", "material_contract": "قرارداد بااهمیت",
    "capital_increase": "افزایش سرمایه", "board_decision": "تصمیمات هیئت مدیره",
    "general_assembly": "مجمع عمومی", "production_sales": "تولید و فروش", "other": "سایر",
}
KIND_FA = {"fetch_error": "خطای دریافت", "parse_error": "خطای استخراج", "missing_value": "مقدار ناموجود",
           "duplicate": "تکراری", "validation": "اعتبارسنجی", "count_mismatch": "مغایرت تعداد",
           "symbol_mismatch": "مغایرت نماد", "verification": "مقایسه با منبع", "metadata": "فراداده",
           "search_error": "خطای جستجو", "truncation": "برش داده"}
SEV_FA = {"error": "خطا", "warning": "هشدار", "info": "اطلاع"}

# Statement sheets to fetch, matched on the CODAL statement selector names.
STATEMENT_PATTERNS = {
    "balance_sheet": ["صورت وضعیت مالی", "ترازنامه"],
    "income_statement": ["سود و زیان", "صورت سود"],
    "cash_flow": ["جریان های نقدی", "جریانهای نقدی", "جریان وجوه نقد", "جریان نقدی"],
}
STATEMENT_FA = {"balance_sheet": "وضعیت مالی", "income_statement": "سود و زیان", "cash_flow": "جریان‌های نقدی"}


class Pipeline:
    def __init__(self, cfg: Config, fetcher: Optional[Fetcher] = None):
        self.cfg = cfg
        wp = cfg.work_path
        wp.mkdir(parents=True, exist_ok=True)
        self.store = Store(wp / "state.sqlite")
        self.fetcher = fetcher or Fetcher(wp / "cache", cfg.min_interval_sec, cfg.timeout_sec,
                                          cfg.max_retries, cfg.backoff_base_sec, cfg.use_browser)
        self.started = time.time()

    # ------------------------------------------------------------ stage 1
    def search(self):
        c = self.cfg
        for sym in c.symbols:
            prev = self.store.search_run(sym, c.start_date, c.end_date)
            if prev and prev["complete"] and not c.refresh_search:
                log.info("[%s] search already complete (%d letters) — reusing", sym, prev["fetched"])
                continue
            self.store.clear_sightings(sym)
            self.store.clear_issues("search_error", sym)
            total = pages = None
            fetched, new, complete = 0, 0, False
            try:
                for page, total, pages, letters in api.search_letters(
                        self.fetcher, c.search_url, sym, c.start_date, c.end_date, c.include_children,
                        c.extra_search_params, c.max_pages_per_symbol, use_cache=not c.refresh_search):
                    for raw in letters:
                        rec = api.letter_record(raw, sym, c.base_url)
                        self.store.add_sighting(rec["tracing_no"], sym, page)
                        new += self.store.upsert_letter(rec)
                        fetched += 1
                    self.store.commit()
                    log.info("[%s] page %d/%s: %d letters", sym, page, pages or "?", len(letters))
                complete = pages is None or pages <= c.max_pages_per_symbol
            except FetchError as e:
                self.store.add_issue("search_error", "error", f"جستجو متوقف شد: {e.message}", sym,
                                     url=e.url)
                log.error("[%s] search failed: %s", sym, e.message)
            self.store.save_search_run(sym, c.start_date, c.end_date, total, pages, fetched, complete)
            log.info("[%s] search: reported=%s fetched=%d new=%d complete=%s", sym, total, fetched, new, complete)

    # ------------------------------------------------------------ stage 2
    def selected(self, rec: dict) -> bool:
        c = self.cfg
        if rec["category"] not in c.categories:
            return False
        if rec["category"] == "announcements" and rec["subtype"] not in c.announcement_types:
            return False
        if rec["category"] == "financial_statements":
            m = api.title_meta(rec["title"])
            if c.period_months and m["period_months"] is not None and m["period_months"] not in c.period_months:
                return False
            if c.fiscal_year_ends and m["fiscal_year_end"] and m["fiscal_year_end"] not in c.fiscal_year_ends:
                return False
            if c.audit_status != "all" and m["audited"] is not None and m["audited"] != (c.audit_status == "audited"):
                return False
            if c.statement_scope != "all" and m["consolidated"] != (c.statement_scope == "consolidated"):
                return False
        return True

    # ------------------------------------------------------------ stage 3
    def process(self):
        letters = [r for r in self.store.letters(self.cfg.symbols) if self.selected(r)]
        todo = []
        for r in letters:
            prev = self.store.result(r["tracing_no"])
            if prev and prev["status"] == "ok" and prev.get("parser_version") == PARSER_VERSION:
                continue
            todo.append(r)
        log.info("process: %d selected, %d to fetch/parse (%d reused)", len(letters), len(todo), len(letters) - len(todo))
        for i, r in enumerate(todo, 1):
            try:
                res = self._process_letter(r)
                res.update(status="ok", parser_version=PARSER_VERSION)
            except FetchError as e:
                res = {"status": "failed", "error": e.message, "error_url": e.url, "parser_version": PARSER_VERSION}
            except Exception as e:  # parser bug: keep the record, report it
                log.exception("parse failure %s", r["tracing_no"])
                res = {"status": "failed", "error": f"parse error: {type(e).__name__}: {e}",
                       "error_url": r["url"], "parser_version": PARSER_VERSION}
            self.store.save_result(r["tracing_no"], res["status"], {k: v for k, v in res.items() if k != "status"})
            if i % 10 == 0 or i == len(todo):
                log.info("process: %d/%d (cache=%d live=%d failed=%d)", i, len(todo),
                         self.fetcher.stats["cache"], self.fetcher.stats["live"], self.fetcher.stats["failed"])
        return letters

    def _process_letter(self, r: dict) -> dict:
        if not r["url"]:
            raise FetchError("", "letter has no URL in search results")
        if r["category"] == "financial_statements":
            return self._financial(r)
        if r["category"] == "monthly_reports":
            html = self.fetcher.get(r["url"])
            p = parse_page(html)
            meta = api.title_meta(r["title"])
            return {"kind": "monthly", "meta": meta, "unit": p["unit"], "format": p["format"],
                    "rows": grid_long_rows(p["grids"]), "issues": [] if p["grids"] else [
                        {"kind": "parse_error", "severity": "error", "field": "",
                         "message": "هیچ جدولی در گزارش ماهانه یافت نشد"}]}
        body = ""
        issues = []
        if self.cfg.fetch_announcement_body:
            body = announcement_text(self.fetcher.get(r["url"]), self.cfg.body_max_chars)
            if not body:
                issues.append({"kind": "missing_value", "severity": "info", "field": "body", "message": "متن اطلاعیه خالی است"})
        return {"kind": "announcement", "body": body, "issues": issues}

    def _financial(self, r: dict) -> dict:
        meta = api.title_meta(r["title"])
        issues: List[dict] = []
        main_html = self.fetcher.get(r["url"])
        opts = sheet_options(main_html)
        pages = []  # (statement_key or "", sheet name, html)
        if opts:
            wanted = {}
            for sid, name in opts:
                n = normalize_text(name)
                for key, pats in STATEMENT_PATTERNS.items():
                    if key not in wanted and any(normalize_text(p) in n for p in pats):
                        wanted[key] = (sid, name)
            current = selected_sheet(main_html)
            for key, (sid, name) in wanted.items():
                try:
                    html = main_html if sid == current else self.fetcher.get(api.with_sheet(r["url"], sid))
                    pages.append((key, name, html))
                except FetchError as e:
                    issues.append({"kind": "fetch_error", "severity": "error", "field": STATEMENT_FA[key],
                                   "message": f"دریافت صورت {name} ناموفق: {e.message}", "url": e.url})
            for key in STATEMENT_PATTERNS:
                if key not in wanted:
                    issues.append({"kind": "missing_value", "severity": "warning", "field": STATEMENT_FA[key],
                                   "message": f"صورت {STATEMENT_FA[key]} در فهرست صورت‌های این اطلاعیه نبود"})
        if not pages:
            pages = [("", "", main_html)]
        grids, units, fmts, ds_meta, parsed = [], set(), set(), {}, []
        for key, name, html in pages:
            p = parse_page(html, name)
            parsed.append(p)
            grids += p["grids"]
            if p["unit"]:
                units.add(p["unit"])
            fmts.add(p["format"])
            ds_meta = ds_meta or p["meta"]
        unit = next(iter(units)) if len(units) == 1 else None
        if len(units) > 1:  # each statement is scaled with its own unit below
            issues.append({"kind": "validation", "severity": "info", "field": "unit",
                           "message": f"واحدهای متفاوت در صورت‌ها (هر صورت با واحد خودش تبدیل شد): {sorted(units)}"})
        # reconcile title metadata with the document's own metadata
        ds_pe = api.fmt_jalali(parse_jalali(ds_meta.get("period_end"))) if ds_meta.get("period_end") else ""
        if ds_pe:
            if meta["period_end"] and ds_pe != meta["period_end"]:
                issues.append({"kind": "metadata", "severity": "warning", "field": "period_end",
                               "message": f"پایان دوره در عنوان ({meta['period_end']}) با سند ({ds_pe}) متفاوت است"})
            meta["period_end"] = meta["period_end"] or ds_pe
        if not meta["fiscal_year_end"] and ds_meta.get("fiscal_year_end"):
            meta["fiscal_year_end"] = api.fmt_jalali(parse_jalali(ds_meta["fiscal_year_end"]))
        if meta["audited"] is None and isinstance(ds_meta.get("audited"), bool):
            meta["audited"] = ds_meta["audited"]
        sheet_names = " ".join(g.sheet for g in grids)
        if "تلفیقی" in normalize_text(sheet_names):
            meta["consolidated"] = True
        metrics: Dict[str, dict] = {}
        for p in parsed:  # first statement that reports a metric wins; unit applied per page
            for m, d in extract_metrics(p["grids"], p["unit"], meta["period_end"]).items():
                metrics.setdefault(m, d)
        if not grids:
            issues.append({"kind": "parse_error", "severity": "error", "field": "",
                           "message": "هیچ جدولی در سند یافت نشد (ممکن است PDF/پیوست باشد)"})
        if not units and grids:
            issues.append({"kind": "validation", "severity": "warning", "field": "unit",
                           "message": "واحد مبالغ تشخیص داده نشد؛ مقادیر استاندارد نشدند (مقدار اصلی موجود است)"})
        for m, d in metrics.items():
            if d["value_original"] and d["value_parsed"] is None:
                issues.append({"kind": "parse_error", "severity": "warning", "field": METRIC_FA[m],
                               "message": f"مقدار «{d['value_original']}» عددی نیست"})
        return {"kind": "financial", "meta": meta, "unit": unit or "، ".join(sorted(units)), "format": "+".join(sorted(fmts)),
                "statements": [STATEMENT_FA[k] for k, _, _ in pages if k], "metrics": metrics,
                "rows": grid_long_rows(grids), "issues": issues}

    # ------------------------------------------------------------ stage 4
    def verify(self, letters: List[dict]) -> List[dict]:
        """Re-download a sample of processed financial letters (cache bypass)
        and confirm each extracted original value is unchanged."""
        out = []
        fin = [r for r in letters if r["category"] == "financial_statements"
               and (self.store.result(r["tracing_no"]) or {}).get("status") == "ok"]
        rnd = random.Random(42)
        for r in rnd.sample(fin, min(self.cfg.verify_sample, len(fin))):
            res = self.store.result(r["tracing_no"])
            try:
                fresh = self._financial_nocache(r)
            except FetchError as e:
                out.append({"kind": "verification", "severity": "warning", "symbol": r["query_symbol"],
                            "tracing_no": r["tracing_no"], "field": "", "url": r["url"],
                            "message": f"دریافت مجدد برای مقایسه ناموفق: {e.message}"})
                continue
            mism = [m for m in set(res["metrics"]) | set(fresh["metrics"])
                    if (res["metrics"].get(m) or {}).get("value_original") != (fresh["metrics"].get(m) or {}).get("value_original")]
            out.append({"kind": "verification", "severity": "error" if mism else "info",
                        "symbol": r["query_symbol"], "tracing_no": r["tracing_no"], "url": r["url"],
                        "field": ", ".join(METRIC_FA[m] for m in mism),
                        "message": ("مغایرت با سند اصلی در: " + ", ".join(METRIC_FA[m] for m in mism)) if mism
                        else f"{len(res['metrics'])} مقدار با سند اصلی (دریافت مجدد) مطابقت دارد"})
        return out

    def _financial_nocache(self, r):
        orig_get = self.fetcher.get
        self.fetcher.get = lambda url, use_cache=True, expect_json=False: orig_get(url, False, expect_json)
        try:
            return self._financial(r)
        finally:
            self.fetcher.get = orig_get

    # ------------------------------------------------------------ stage 5
    def run(self) -> dict:
        self.search()
        letters = self.process()
        verification = self.verify(letters) if self.cfg.verify_sample else []
        report = self.build_output(letters, verification)
        self.fetcher.close()
        return report

    def build_output(self, letters: List[dict], verification: List[dict]) -> dict:
        c = self.cfg
        fin, monthly, ann, raw, dq, letter_rows = [], [], [], [], [], []
        status_count = Counter()
        per_sym = defaultdict(Counter)

        def issue(kind, sev, msg, sym="", tn="", fld="", url=""):
            dq.append({"kind": kind, "kind_fa": KIND_FA.get(kind, kind), "severity_fa": SEV_FA.get(sev, sev),
                       "severity": sev, "symbol": sym, "tracing_no": tn, "field": fld, "message": msg, "url": url})

        for s in self.store.issues():
            issue(s["kind"], s["severity"], s["message"], s["symbol"], s["tracing_no"], s["field"], s["url"])

        all_letters = self.store.letters(c.symbols)
        for r in all_letters:
            res = self.store.result(r["tracing_no"]) if self.selected(r) else None
            st = res["status"] if res else ("skipped" if not self.selected(r) else "pending")
            letter_rows.append({**r, "category_fa": CAT_FA[r["category"]] + " / " + SUB_FA.get(r["subtype"], r["subtype"]),
                                "status_fa": {"ok": "موفق", "failed": "ناموفق", "skipped": "خارج از فیلتر",
                                              "pending": "در انتظار"}[st]})
            if st == "skipped":
                continue
            status_count[st] += 1
            sym = r["query_symbol"]
            if st == "failed":
                per_sym[sym]["failed"] += 1
                issue("fetch_error" if "parse error" not in res["error"] else "parse_error", "error",
                      f"{CAT_FA[r['category']]}: {res['error']} — رکورد حفظ شد ولی داده ندارد",
                      sym, r["tracing_no"], "", res.get("error_url") or r["url"])
            if not api.symbol_matches(r["symbol"], sym):
                issue("symbol_mismatch", "warning", f"نماد اطلاعیه «{r['symbol']}» با نماد درخواستی «{sym}» یکسان نیست",
                      sym, r["tracing_no"], "symbol", r["url"])
            base = {"symbol_query": sym, "symbol": r["symbol"], "company": r["company"], "tracing_no": r["tracing_no"],
                    "letter_code": r["letter_code"], "title": r["title"], "published": r["published"],
                    "sent": r["sent"], "url": r["url"]}
            for x in (res or {}).get("issues", []):
                issue(x["kind"], x["severity"], x["message"], sym, r["tracing_no"], x.get("field", ""), x.get("url") or r["url"])

            if r["category"] == "financial_statements":
                per_sym[sym]["financial"] += 1
                meta = (res or {}).get("meta") or api.title_meta(r["title"])
                row = {**base, **{k: meta.get(k) for k in ("period_months", "period_end", "fiscal_year_end", "report_type")},
                       "consolidated": meta.get("consolidated"), "audited": meta.get("audited"),
                       "scope_fa": "تلفیقی" if meta.get("consolidated") else "شرکت اصلی",
                       "audit_fa": {True: "حسابرسی‌شده", False: "حسابرسی‌نشده"}.get(meta.get("audited"), "نامشخص"),
                       "is_amendment": meta.get("is_amendment"), "unit_original": (res or {}).get("unit") or "",
                       "statements_found": "، ".join((res or {}).get("statements", []))}
                metrics = (res or {}).get("metrics", {})
                cols = set()
                for m in METRICS:
                    d = metrics.get(m)
                    row[m] = d["value"] if d else None
                    if d:
                        cols.add(d["column"])
                    if st == "ok" and (not d or d["value"] is None) and m != "dps":
                        issue("missing_value", "info" if m in ("cost_of_revenue", "gross_profit") else "warning",
                              f"{METRIC_FA[m]} در سند یافت نشد" if not d else f"{METRIC_FA[m]}: مقدار خالی/غیرعددی",
                              sym, r["tracing_no"], METRIC_FA[m], r["url"])
                row["column_used"] = " ; ".join(sorted(x for x in cols if x))
                if meta.get("audited") is None:
                    issue("metadata", "info", "وضعیت حسابرسی در عنوان/سند مشخص نیست", sym, r["tracing_no"], "audited", r["url"])
                flags = []
                if st == "ok":
                    for x in financial_checks(row):
                        flags.append(x["message"])
                        issue(x["kind"], x["severity"], x["message"], sym, r["tracing_no"], x["field"], r["url"])
                if st == "failed":
                    flags.append("دریافت/استخراج ناموفق")
                row["flags"] = " | ".join(flags)
                fin.append(row)
            elif r["category"] == "monthly_reports":
                per_sym[sym]["monthly"] += 1
                meta = (res or {}).get("meta") or api.title_meta(r["title"])
                for x in (res or {}).get("rows", []):
                    monthly.append({**base, **x, "period_months": meta.get("period_months"),
                                    "period_end": meta.get("period_end"), "is_amendment": meta.get("is_amendment"),
                                    "unit_original": (res or {}).get("unit") or ""})
            else:
                per_sym[sym]["announcements"] += 1
                rawj = r["raw"]
                att = rawj.get("AttachmentUrl") or ""
                ann.append({**base, "subtype_fa": SUB_FA.get(r["subtype"], r["subtype"]),
                            "has_attachment": bool(rawj.get("HasAttachment")),
                            "attachment_url": (c.base_url + "/" + att.lstrip("/")) if att and not att.startswith("http") else att,
                            "body": (res or {}).get("body", "")})
            for x in (res or {}).get("rows", []):
                raw.append({"symbol_query": sym, "tracing_no": r["tracing_no"], "category_fa": CAT_FA[r["category"]],
                            "source": (res or {}).get("format", ""), "url": r["url"], **x})

        # duplicates: same tracing number seen twice in one symbol's search
        for d in self.store.duplicate_sightings():
            issue("duplicate", "info", f"اطلاعیه {d['n']} بار در نتایج جستجو تکرار شد (صفحات {d['pages']}); یک بار ذخیره شد",
                  d["query_symbol"], d["tracing_no"])
        # same tracing number returned for two requested symbols (stored once, under the first)
        for d in self.store.cross_symbol_sightings():
            issue("duplicate", "info", f"اطلاعیه در نتایج چند نماد آمد ({d['syms']}); یک بار ذخیره شد",
                  d["syms"].split(",")[0], d["tracing_no"])
        # revisions of the same report: kept, flagged
        for key, rows in revision_groups([f for f in fin if f.get("period_end")]).items():
            tns = [x["tracing_no"] for x in rows]
            for x in rows:
                x["flags"] = (x["flags"] + " | " if x["flags"] else "") + f"چند نسخه برای این دوره ({len(rows)})"
            issue("duplicate", "warning", f"{len(rows)} اطلاعیه برای یک دوره/نوع (احتمالاً اصلاحیه): {', '.join(tns)}",
                  key[0], tns[0], "period_end")
        # count verification
        for sym in c.symbols:
            run = self.store.search_run(sym, c.start_date, c.end_date) or {}
            got = self.store.sighted_count(sym)
            rep = run.get("reported_total")
            if rep is None:
                issue("count_mismatch", "error", "تعداد کل از کدال دریافت نشد (جستجو ناموفق)", sym)
            elif rep != got:
                issue("count_mismatch", "warning", f"کدال {rep} اطلاعیه گزارش کرد، {got} اطلاعیه یکتا دریافت شد", sym)
            if run and not run.get("complete"):
                issue("count_mismatch", "error", "جستجو کامل نشد؛ ممکن است اطلاعیه‌هایی از قلم افتاده باشد", sym)
        for v in verification:
            issue(v["kind"], v["severity"], v["message"], v["symbol"], v["tracing_no"], v["field"], v["url"])
        if len(raw) > c.raw_max_rows:
            issue("truncation", "warning", f"برگه Raw Data به {c.raw_max_rows:,} ردیف محدود شد ({len(raw):,} ردیف). داده کامل در state.sqlite است.")
            raw = raw[: c.raw_max_rows]

        coverage = []
        for sym in c.symbols:
            rows = [r for r in all_letters if r["query_symbol"] == sym]
            dates = sorted(r["published"] for r in rows if r["published"])
            run = self.store.search_run(sym, c.start_date, c.end_date) or {}
            coverage.append({"symbol": sym, "company": next((r["company"] for r in rows if r["company"]), ""),
                             "reported_total": run.get("reported_total"), "letters": len(rows),
                             "financial": per_sym[sym]["financial"], "monthly": per_sym[sym]["monthly"],
                             "announcements": per_sym[sym]["announcements"], "failed": per_sym[sym]["failed"],
                             "first_date": dates[0][:10] if dates else "", "last_date": dates[-1][:10] if dates else "",
                             "search_complete": bool(run.get("complete"))})
        dates = sorted(r["published"][:10] for r in all_letters if r["published"])
        sev = Counter(d["severity"] for d in dq)
        summary_kv = [
            {"k": "نمادهای درخواستی", "v": "، ".join(c.symbols)},
            {"k": "بازه درخواستی (شمسی)", "v": f"{c.start_date} تا {c.end_date}"},
            {"k": "پوشش واقعی تاریخ انتشار", "v": f"{dates[0]} تا {dates[-1]}" if dates else "— هیچ داده‌ای دریافت نشد —"},
            {"k": "دسته‌های انتخابی", "v": "، ".join(CAT_FA[x] for x in c.categories)},
            {"k": "فیلتر دوره مالی", "v": json.dumps({"period_months": c.period_months, "fiscal_year_ends": c.fiscal_year_ends,
                                                       "audit": c.audit_status, "scope": c.statement_scope}, ensure_ascii=False)},
            {"k": "کل اطلاعیه‌های یافت‌شده", "v": len(all_letters)},
            {"k": "اطلاعیه‌های مشمول فیلتر", "v": sum(status_count.values())},
            {"k": "پردازش موفق", "v": status_count["ok"]},
            {"k": "پردازش ناموفق", "v": status_count["failed"]},
            {"k": "ردیف‌های صورت مالی", "v": len(fin)},
            {"k": "ردیف‌های گزارش ماهانه", "v": len(monthly)},
            {"k": "اطلاعیه‌ها", "v": len(ann)},
            {"k": "مشکلات کیفیت داده (خطا/هشدار/اطلاع)", "v": f"{sev['error']} / {sev['warning']} / {sev['info']}"},
            {"k": "درخواست‌ها (زنده/کش/تلاش مجدد/مرورگر)", "v": "{live}/{cache}/{retries}/{browser}".format(**self.fetcher.stats)},
            {"k": "منبع", "v": "سامانه کدال — https://www.codal.ir (همه ارقام از اسناد اصلی؛ هیچ مقداری تخمین زده نشده)"},
            {"k": "واحد استاندارد", "v": "مبالغ: میلیون ریال؛ EPS و DPS: ریال؛ مقدار اصلی در برگه Raw Data"},
            {"k": "زمان اجرا", "v": time.strftime("%Y-%m-%d %H:%M:%S")},
        ]
        out = Path(c.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fin.sort(key=lambda x: (x["symbol_query"], x.get("period_end") or "", x["published"]))
        monthly.sort(key=lambda x: (x["symbol_query"], x.get("period_end") or "", x["tracing_no"], x["row"], x["col"]))
        ann.sort(key=lambda x: (x["symbol_query"], x["published"]))
        build_workbook(str(out), summary_kv, coverage, fin, monthly, ann, raw, letter_rows, dq)
        report = {
            "symbols": len(c.symbols), "letters_found": len(all_letters), "selected": sum(status_count.values()),
            "ok": status_count["ok"], "failed": status_count["failed"], "financial_rows": len(fin),
            "monthly_rows": len(monthly), "announcements": len(ann), "raw_rows": len(raw),
            "date_coverage": f"{dates[0]} .. {dates[-1]}" if dates else None,
            "issues": dict(sev), "issue_kinds": dict(Counter(d["kind"] for d in dq)),
            "requests": dict(self.fetcher.stats), "output": str(out), "elapsed_sec": round(time.time() - self.started, 1),
        }
        (c.work_path / "execution_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
