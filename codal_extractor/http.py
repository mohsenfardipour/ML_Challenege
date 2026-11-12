"""HTTP layer: polite rate limiting, retries with exponential backoff,
on-disk response cache, and an optional Playwright fallback for pages that
need a real browser (JS challenge / WAF)."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524}
BLOCK_STATUS = {401, 403}


def _short_exc(e: Exception) -> str:
    msg = str(e)
    if "Tunnel connection failed" in msg or "ProxyError" in type(e).__name__:
        return "network error: proxy/firewall refused connection"
    if "timed out" in msg.lower():
        return "network error: timeout"
    return f"network error: {type(e).__name__}"


class FetchError(Exception):
    def __init__(self, url: str, message: str, status: Optional[int] = None):
        short = url if len(url) <= 90 else url[:60] + "…" + url[-25:]
        super().__init__(f"{message}" + (f" [HTTP {status}]" if status else "") + f" {short}")
        self.url, self.status, self.message = url, status, message


class Fetcher:
    def __init__(self, cache_dir: Path, min_interval: float = 1.5, timeout: float = 30,
                 max_retries: int = 5, backoff_base: float = 2.0, use_browser: str = "auto",
                 session: Optional[requests.Session] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval, self.timeout = min_interval, timeout
        self.max_retries, self.backoff_base = max_retries, backoff_base
        self.use_browser = use_browser
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._last = 0.0
        self._browser = None
        self.stats = {"live": 0, "cache": 0, "retries": 0, "browser": 0, "failed": 0}

    # -- cache -----------------------------------------------------------
    def _cache_file(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.cache"

    def cached(self, url: str) -> Optional[str]:
        f = self._cache_file(url)
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))["body"]
        return None

    def _store(self, url: str, body: str):
        f = self._cache_file(url)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"url": url, "ts": time.time(), "body": body},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)

    # -- fetching --------------------------------------------------------
    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, url: str, use_cache: bool = True, expect_json: bool = False) -> str:
        if use_cache:
            body = self.cached(url)
            if body is not None:
                self.stats["cache"] += 1
                return body
        body = None
        if self.use_browser != "always":
            try:
                body = self._get_requests(url, expect_json)
            except FetchError as e:
                if self.use_browser == "auto" and (e.status in BLOCK_STATUS or e.status is None
                                                   or "challenge" in e.message):
                    log.warning("requests failed (%s); trying browser fallback", e.message)
                else:
                    self.stats["failed"] += 1
                    raise
        if body is None:
            if self.use_browser == "never":
                self.stats["failed"] += 1
                raise FetchError(url, "requests failed and browser fallback disabled")
            body = self._get_browser(url, expect_json)
        self._store(url, body)
        return body

    def _get_requests(self, url: str, expect_json: bool) -> str:
        last_err = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.session.get(url, timeout=self.timeout)
                self.stats["live"] += 1
                if r.status_code == 200:
                    r.encoding = r.encoding if r.encoding and "utf" in r.encoding.lower() else "utf-8"
                    text = r.text
                    if expect_json:
                        try:
                            json.loads(text)
                        except ValueError:
                            raise FetchError(url, "expected JSON, got non-JSON (challenge page?)", 200)
                    return text
                if r.status_code in BLOCK_STATUS:
                    raise FetchError(url, "blocked", r.status_code)
                if r.status_code not in RETRY_STATUS:
                    raise FetchError(url, "HTTP error", r.status_code)
                last_err = FetchError(url, "retryable HTTP error", r.status_code)
                retry_after = r.headers.get("Retry-After")
            except requests.RequestException as e:
                last_err, retry_after = FetchError(url, _short_exc(e)), None
            if attempt < self.max_retries:
                self.stats["retries"] += 1
                delay = float(retry_after) if retry_after and retry_after.isdigit() else \
                    self.backoff_base * (2 ** attempt) + random.uniform(0, 1)
                log.info("retry %d/%d in %.1fs: %s", attempt + 1, self.max_retries, delay, last_err.message)
                time.sleep(min(delay, 120))
        raise last_err

    def _get_browser(self, url: str, expect_json: bool) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.stats["failed"] += 1
            raise FetchError(url, "browser fallback needed but Playwright is not installed")
        try:
            if self._browser is None:
                self._pw = sync_playwright().start()
                self._browser = self._pw.chromium.launch()
                self._ctx = self._browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"],
                                                      locale="fa-IR")
                self._page = self._ctx.new_page()
            self._throttle()
            self.stats["browser"] += 1
            if expect_json:
                resp = self._ctx.request.get(url, timeout=self.timeout * 1000)
                if not resp.ok:
                    raise FetchError(url, "browser HTTP error", resp.status)
                return resp.text()
            resp = self._page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
            if resp is not None and resp.status >= 400:
                raise FetchError(url, "browser HTTP error", resp.status)
            return self._page.content()
        except FetchError:
            self.stats["failed"] += 1
            raise
        except Exception as e:  # playwright errors
            self.stats["failed"] += 1
            raise FetchError(url, f"browser error: {type(e).__name__}: {e}")

    def close(self):
        if self._browser is not None:
            try:
                self._browser.close()
                self._pw.stop()
            except Exception:
                pass
            self._browser = None
