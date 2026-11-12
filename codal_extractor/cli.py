"""Command line interface.

    python -m codal_extractor run   --config config.json
    python -m codal_extractor run   --symbols فولاد خودرو --start 1402/01/01 --end 1402/12/29
    python -m codal_extractor probe --symbol فولاد --start 1402/01/01 --end 1402/12/29
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import codal_api as api
from .config import ANNOUNCEMENT_TYPES, CATEGORIES, Config
from .http import Fetcher, FetchError
from .parsers import METRIC_FA, extract_datasource, extract_metrics, parse_page, sheet_options


def _setup_logging(work_dir: str, verbose: bool):
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(Path(work_dir) / "extract.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers[:] = [fh, sh]
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _common(p):
    p.add_argument("--config", help="JSON or YAML config file")
    p.add_argument("--start", dest="start_date", help="start date, Jalali 1402/01/01 or ISO 2023-03-21")
    p.add_argument("--end", dest="end_date", help="end date")
    p.add_argument("--work-dir", dest="work_dir")
    p.add_argument("--use-browser", dest="use_browser", choices=["never", "auto", "always"])
    p.add_argument("-v", "--verbose", action="store_true")


def build_config(a) -> Config:
    over = {k: getattr(a, k, None) for k in (
        "symbols", "start_date", "end_date", "categories", "announcement_types", "period_months",
        "fiscal_year_ends", "audit_status", "statement_scope", "output", "work_dir", "use_browser")}
    if getattr(a, "refresh_search", False):
        over["refresh_search"] = True
    if getattr(a, "fetch_body", False):
        over["fetch_announcement_body"] = True
    if getattr(a, "children", False):
        over["include_children"] = True
    if a.config:
        return Config.from_file(a.config, **over)
    return Config(**{k: v for k, v in over.items() if v is not None})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="codal_extractor", description="CODAL disclosures -> Excel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="full extraction")
    _common(r)
    r.add_argument("--symbols", nargs="+")
    r.add_argument("--categories", nargs="+", choices=CATEGORIES)
    r.add_argument("--announcement-types", dest="announcement_types", nargs="+", choices=ANNOUNCEMENT_TYPES)
    r.add_argument("--period-months", dest="period_months", nargs="+", type=int)
    r.add_argument("--fiscal-year-ends", dest="fiscal_year_ends", nargs="+")
    r.add_argument("--audit", dest="audit_status", choices=["all", "audited", "unaudited"])
    r.add_argument("--scope", dest="statement_scope", choices=["all", "consolidated", "standalone"])
    r.add_argument("--output", "-o")
    r.add_argument("--refresh-search", action="store_true", help="re-query CODAL search (ignore cached pages)")
    r.add_argument("--fetch-body", action="store_true", help="also download announcement text")
    r.add_argument("--children", action="store_true", help="include subsidiaries' letters (CODAL Childs flag)")
    p = sub.add_parser("probe", help="proof of concept on one symbol: inspect search + one statement")
    _common(p)
    p.add_argument("--symbol", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "probe":
        a.symbols = [a.symbol]
        a.output = None
        cfg = build_config(a)
        _setup_logging(cfg.work_dir, a.verbose)
        return probe(cfg)
    cfg = build_config(a)
    _setup_logging(cfg.work_dir, a.verbose)
    from .pipeline import Pipeline
    rep = Pipeline(cfg).run()
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["failed"] == 0 and rep["letters_found"] > 0 else 2


def probe(cfg: Config) -> int:
    """Step 2/3 of the workflow: small POC on one symbol, printing only a
    compact structural summary so it can be compared with the website."""
    f = Fetcher(cfg.work_path / "cache", cfg.min_interval_sec, cfg.timeout_sec, cfg.max_retries,
                cfg.backoff_base_sec, cfg.use_browser)
    sym = cfg.symbols[0]
    try:
        page, total, pages, letters = next(api.search_letters(f, cfg.search_url, sym, cfg.start_date, cfg.end_date,
                                                             cfg.include_children, cfg.extra_search_params, 1))
    except FetchError as e:
        print(f"SEARCH FAILED: {e}")
        return 2
    print(f"search: total={total} pages={pages} first_page={len(letters)}")
    if letters:
        print("letter keys:", sorted(letters[0].keys()))
    recs = [api.letter_record(x, sym, cfg.base_url) for x in letters]
    for rec in recs[:10]:
        print(f"  {rec['tracing_no']:>8} {rec['published'][:10]} {rec['category'][:10]:<10} {rec['title'][:70]}")
    fin = next((x for x in recs if x["category"] == "financial_statements"), None)
    if not fin:
        print("no financial statement on first page; widen the date range to inspect one")
        return 0
    print("\nstatement:", fin["url"])
    print("title meta:", api.title_meta(fin["title"]))
    try:
        html = f.get(fin["url"])
    except FetchError as e:
        print(f"LETTER FAILED: {e}")
        return 2
    opts = sheet_options(html)
    print("sheet selector:", opts[:12])
    ds = extract_datasource(html)
    print("datasource:", "found, keys=" + str(sorted(ds.keys())[:15]) if ds else "not found (HTML fallback)")
    p = parse_page(html)
    print(f"format={p['format']} unit={p['unit']} grids={len(p['grids'])} meta={p['meta']}")
    for key, pats in (("income", ["سود و زیان"]), ("balance", ["وضعیت مالی", "ترازنامه"])):
        sid = next((s for s, n in opts if any(x in n for x in pats)), None)
        if sid is not None:
            try:
                p2 = parse_page(f.get(api.with_sheet(fin["url"], sid)))
                p["grids"] += p2["grids"]
            except FetchError as e:
                print(f"{key} sheet failed: {e}")
    m = extract_metrics(p["grids"], p["unit"], api.title_meta(fin["title"])["period_end"])
    print("\nmetrics (verify these against the page in a browser):")
    for k, d in m.items():
        print(f"  {METRIC_FA[k]:<22} {d['value_original']!s:>18} -> {d['value']}  [{d['label']} | {d['column'][:40]}]")
    print("requests:", f.stats)
    f.close()
    return 0
