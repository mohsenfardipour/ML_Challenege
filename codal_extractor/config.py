"""Run configuration. Nothing is hardcoded: symbols, dates, categories and
filters come from a JSON/YAML file and/or CLI flags."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .jalali import to_jalali_str

CATEGORIES = ("financial_statements", "monthly_reports", "announcements")

# Sub-types of announcements that can be selected individually.
ANNOUNCEMENT_TYPES = (
    "important_disclosure", "material_contract", "capital_increase",
    "board_decision", "general_assembly", "production_sales", "other",
)


@dataclass
class Config:
    symbols: List[str]
    start_date: str                     # Jalali yyyy/mm/dd (Gregorian accepted, converted)
    end_date: str
    categories: List[str] = field(default_factory=lambda: list(CATEGORIES))
    announcement_types: List[str] = field(default_factory=lambda: list(ANNOUNCEMENT_TYPES))
    # Optional financial-period filters
    period_months: List[int] = field(default_factory=list)       # e.g. [3, 6, 9, 12]; empty = all
    fiscal_year_ends: List[str] = field(default_factory=list)    # e.g. ["1402/12/29"]; empty = all
    audit_status: str = "all"           # all | audited | unaudited
    statement_scope: str = "all"        # all | consolidated | standalone
    include_children: bool = False      # CODAL 'Childs' flag (subsidiary letters)
    # Network behaviour
    base_url: str = "https://www.codal.ir"
    search_url: str = "https://search.codal.ir/api/search/v2/q"
    min_interval_sec: float = 1.5       # politeness delay between live requests
    timeout_sec: float = 30.0
    max_retries: int = 5
    backoff_base_sec: float = 2.0
    use_browser: str = "auto"           # never | auto | always  (Playwright fallback)
    extra_search_params: Dict[str, str] = field(default_factory=dict)
    # Processing
    fetch_announcement_body: bool = False   # extra request per announcement for its text
    body_max_chars: int = 2000
    max_pages_per_symbol: int = 500     # safety cap (20 letters/page)
    verify_sample: int = 3              # letters re-downloaded (cache bypass) and re-compared
    raw_max_rows: int = 900_000         # Excel row limit guard for Raw Data sheet
    # Paths
    work_dir: str = "codal_work"
    output: str = "codal_output.xlsx"
    refresh_search: bool = False        # re-query search API even if cached

    def __post_init__(self):
        if not self.symbols:
            raise ValueError("At least one symbol is required")
        self.symbols = [s.strip() for s in self.symbols if s and s.strip()]
        self.start_date = to_jalali_str(self.start_date)
        self.end_date = to_jalali_str(self.end_date)
        if self.start_date > self.end_date:
            raise ValueError("start_date is after end_date")
        bad = set(self.categories) - set(CATEGORIES)
        if bad:
            raise ValueError(f"Unknown categories {bad}; valid: {CATEGORIES}")
        bad = set(self.announcement_types) - set(ANNOUNCEMENT_TYPES)
        if bad:
            raise ValueError(f"Unknown announcement types {bad}; valid: {ANNOUNCEMENT_TYPES}")
        if self.audit_status not in ("all", "audited", "unaudited"):
            raise ValueError("audit_status must be all|audited|unaudited")
        if self.statement_scope not in ("all", "consolidated", "standalone"):
            raise ValueError("statement_scope must be all|consolidated|standalone")
        if self.use_browser not in ("never", "auto", "always"):
            raise ValueError("use_browser must be never|auto|always")
        self.fiscal_year_ends = [to_jalali_str(d) for d in self.fiscal_year_ends]

    @property
    def work_path(self) -> Path:
        return Path(self.work_dir)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str, **overrides) -> "Config":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yml", ".yaml"):
            import yaml  # optional dependency
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)
