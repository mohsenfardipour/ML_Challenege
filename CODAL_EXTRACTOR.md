# CODAL extractor (سامانه کدال → Excel)

Downloads disclosures for a list of TSE/IFB symbols from **codal.ir** and writes one structured,
traceable Excel workbook. Nothing is hardcoded: you pass the symbols, dates, categories and period filters.

## Install & run

```bash
pip install -r requirements.txt
```

```bash
python -m codal_extractor probe --symbol فولاد --start 1402/01/01 --end 1402/12/29
```

```bash
python -m codal_extractor run --config config.example.json
```

```bash
python -m codal_extractor run --symbols فولاد خودرو --start 2023-03-21 --end 1402/12/29 \
    --categories financial_statements monthly_reports --period-months 6 12 --audit audited -o output/out.xlsx
```

Dates may be Jalali (`1402/01/01`) or Gregorian ISO (`2023-03-21`); CODAL is queried in Jalali.
`probe` is the single-symbol proof of concept. It prints the search totals, the letter fields, the statement
selector and the extracted metrics next to their original strings, so you can compare them with the page in a browser.

## How it works

| Stage | What happens |
|---|---|
| Search | `search.codal.ir/api/search/v2/q` (the API behind codal.ir's own search page), 20 letters per page, every page followed. Results are stored once per `TracingNo`; repeated sightings are logged as duplicates. |
| Filter | The category comes from the letter code (ن-۱۰ statements, ن-۳۰ monthly), then from title keywords. Period, audit and scope filters are read from the title before any document is downloaded, so no requests are wasted. |
| Fetch/parse | Letter pages (`Reports/Decision.aspx`). Modern reports embed `var datasource = {...}`, which holds sheets, tables and cells. Older reports fall back to HTML tables. For statements, the balance sheet, income statement and cash-flow sheets are found through the page's `ddlTable` selector. |
| Metrics | Rows are matched on exact normalized Persian labels (so «جمع دارایی‌ها» never matches «جمع دارایی‌های جاری»). The current-period column is chosen by matching the period-end date in the header. A metric that isn't found stays **empty** and is reported. |
| Units | Each statement's «مبالغ به … ریال» is detected and the amounts are converted to **million IRR**. EPS and DPS stay in IRR. The original string, unit, cell address and column header are kept. |
| Validate | Checks that gross profit = revenue ± cost of revenue, and that assets = liabilities + equity. Also checks EPS sign against net profit, negative revenue, period end after publish date, a symbol different from the one requested, the CODAL `Total` against unique letters received, incomplete searches, and revisions (اصلاحیه) of the same period. |
| Verify | Re-downloads `verify_sample` statements without the cache and compares every extracted original value. |
| Output | Six sheets: Summary, Financial Statements, Monthly Reports, Announcements, Raw Data, Data Quality. They are right-to-left, with Persian headers, Excel tables (filters), frozen headers, fitted widths, number formats and clickable source URLs. |

Robustness:
* Requests are rate limited (`min_interval_sec`) and retried with exponential backoff (honouring `Retry-After`).
* With `use_browser: auto` it falls back to Playwright when blocked (`pip install playwright && playwright install chromium`).
* Every response is cached on disk under `work_dir/cache`.
* State is kept in `work_dir/state.sqlite`, so a rerun reuses completed searches and processed letters and only retries failures. Use `--refresh-search` when the end date is still open.
* Failed letters stay in the workbook with empty values, a flag and a Data Quality row. Nothing is silently dropped or estimated.
* The log is in `work_dir/extract.log`. A JSON execution report is written to `work_dir/execution_report.json`.

## Tests

```bash
python -m pytest -q tests
```

The tests use **synthetic fixtures** (`tests/fixtures.py`) shaped like CODAL responses. They cover Jalali conversion,
number parsing, label matching, column choice, unit scaling, pagination and duplicates, failure retention,
resume without re-fetching, filters, and the workbook layout.

## Known limitations

* The parsers follow CODAL's public page structure as documented here. They were tested on synthetic fixtures, not live
  pages, because the development sandbox could not reach codal.ir. Run `probe` first and check its output against the site.
  If CODAL has renamed a field, adjust `METRICS` aliases or `STATEMENT_PATTERNS`.
* Letters whose content is only a PDF or scanned attachment produce a "no table found" error row. There is no OCR.
* DPS is filled only when a financial-statement document itself contains a «سود نقدی هر سهم» row. DPS from
  general-assembly decisions is not parsed yet (those letters appear in Announcements with their URL), so the column is often empty.
