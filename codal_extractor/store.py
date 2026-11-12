"""SQLite state store: intermediate results survive crashes and let a rerun
skip letters that were already processed successfully."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS letters (
    tracing_no TEXT PRIMARY KEY,
    query_symbol TEXT, symbol TEXT, company TEXT, title TEXT, letter_code TEXT,
    sent TEXT, published TEXT, url TEXT, category TEXT, subtype TEXT,
    raw TEXT, first_seen REAL
);
CREATE TABLE IF NOT EXISTS sightings (       -- every time a tracing_no appears in search results
    tracing_no TEXT, query_symbol TEXT, page INTEGER, ts REAL
);
CREATE TABLE IF NOT EXISTS search_runs (
    query_symbol TEXT, start_date TEXT, end_date TEXT, reported_total INTEGER,
    pages INTEGER, fetched INTEGER, complete INTEGER, ts REAL,
    PRIMARY KEY (query_symbol, start_date, end_date)
);
CREATE TABLE IF NOT EXISTS processed (
    tracing_no TEXT PRIMARY KEY, status TEXT, result TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS issues (
    kind TEXT, severity TEXT, symbol TEXT, tracing_no TEXT, field TEXT,
    message TEXT, url TEXT, ts REAL
);
"""


class Store:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # letters
    def upsert_letter(self, rec: dict) -> bool:
        """Insert letter; returns True if it was new."""
        cur = self.db.execute("SELECT 1 FROM letters WHERE tracing_no=?", (rec["tracing_no"],))
        new = cur.fetchone() is None
        if new:
            self.db.execute(
                "INSERT INTO letters VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["tracing_no"], rec["query_symbol"], rec["symbol"], rec["company"], rec["title"],
                 rec["letter_code"], rec["sent"], rec["published"], rec["url"], rec["category"],
                 rec["subtype"], json.dumps(rec["raw"], ensure_ascii=False), time.time()))
        return new

    def add_sighting(self, tracing_no: str, query_symbol: str, page: int):
        self.db.execute("INSERT INTO sightings VALUES (?,?,?,?)",
                        (tracing_no, query_symbol, page, time.time()))

    def clear_sightings(self, query_symbol: str):
        self.db.execute("DELETE FROM sightings WHERE query_symbol=?", (query_symbol,))

    def sighted_count(self, query_symbol: str) -> int:
        return self.db.execute("SELECT COUNT(DISTINCT tracing_no) FROM sightings WHERE query_symbol=?",
                               (query_symbol,)).fetchone()[0]

    def cross_symbol_sightings(self) -> List[dict]:
        rows = self.db.execute(
            "SELECT tracing_no, GROUP_CONCAT(DISTINCT query_symbol) syms, COUNT(DISTINCT query_symbol) n "
            "FROM sightings GROUP BY tracing_no HAVING n > 1").fetchall()
        return [dict(r) for r in rows]

    def letters(self, symbols: Optional[Iterable[str]] = None) -> List[dict]:
        rows = self.db.execute("SELECT * FROM letters ORDER BY published").fetchall()
        out = [dict(r) for r in rows]
        if symbols is not None:
            s = set(symbols)
            out = [r for r in out if r["query_symbol"] in s]
        for r in out:
            r["raw"] = json.loads(r["raw"]) if r["raw"] else {}
        return out

    def duplicate_sightings(self) -> List[dict]:
        rows = self.db.execute(
            "SELECT tracing_no, query_symbol, COUNT(*) n, GROUP_CONCAT(page) pages FROM sightings "
            "GROUP BY tracing_no, query_symbol HAVING n > 1").fetchall()
        return [dict(r) for r in rows]

    # search runs
    def save_search_run(self, sym, start, end, total, pages, fetched, complete):
        self.db.execute("INSERT OR REPLACE INTO search_runs VALUES (?,?,?,?,?,?,?,?)",
                        (sym, start, end, total, pages, fetched, int(complete), time.time()))
        self.db.commit()

    def search_run(self, sym, start, end) -> Optional[dict]:
        r = self.db.execute("SELECT * FROM search_runs WHERE query_symbol=? AND start_date=? AND end_date=?",
                            (sym, start, end)).fetchone()
        return dict(r) if r else None

    # processing results
    def save_result(self, tracing_no: str, status: str, result: dict):
        self.db.execute("INSERT OR REPLACE INTO processed VALUES (?,?,?,?)",
                        (tracing_no, status, json.dumps(result, ensure_ascii=False), time.time()))
        self.db.commit()

    def result(self, tracing_no: str) -> Optional[dict]:
        r = self.db.execute("SELECT * FROM processed WHERE tracing_no=?", (tracing_no,)).fetchone()
        if not r:
            return None
        return {"status": r["status"], **json.loads(r["result"])}

    # issues
    def add_issue(self, kind, severity, message, symbol="", tracing_no="", field="", url=""):
        self.db.execute("INSERT INTO issues VALUES (?,?,?,?,?,?,?,?)",
                        (kind, severity, symbol, tracing_no, field, message, url, time.time()))
        self.db.commit()

    def clear_issues(self, kind: str, symbol: str):
        self.db.execute("DELETE FROM issues WHERE kind=? AND symbol=?", (kind, symbol))
        self.db.commit()

    def issues(self) -> List[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM issues ORDER BY ts").fetchall()]

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()
