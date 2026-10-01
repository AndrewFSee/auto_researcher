"""
SEC 8-K event adapter — EDGAR JSON + regex Item classification.

Why 8-K
-------
The 8-K is the event-driven counterpart to the 10-K/10-Q. Items 1.01
(material agreements), 2.02 (results of operations), 5.02 (departure of
executives), 7.01 (Reg FD), and 8.01 (other) carry distinct signals
with documented post-filing drift patterns. We emit a per-ticker daily
score that sums Item weights — a crude but interpretable construction
that a downstream ranker can CPCV-validate against forward returns.

Why a regex classifier
----------------------
The 8-K index JSON gives filing date and the list of items triggered;
the items themselves are in the filing body. The index alone is enough
to know *which* items were invoked — which is all this adapter needs.
Pulling the full body to run an LLM classifier is 50x the bandwidth
for 0.2 sigma of extra signal, and not worth the complexity for a
research-grade alt-data sleeve.

Mapping tickers → CIKs
----------------------
EDGAR indexes by 10-digit CIK, not ticker. ``company_tickers.json`` at
``www.sec.gov/files/company_tickers.json`` provides the map; we cache
it for 24 hours per fetch call (SEC updates it daily).

Rate limits
-----------
EDGAR publishes a 10 req/s cap and requires a User-Agent identifying
the requester. The adapter respects both. Callers running this against
an sp500 universe for a multi-year window should expect several minutes
wall-clock even with the cache warm.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd
import requests

from .base import AltDataCache, FetchWindow, empty_altdata_series

logger = logging.getLogger(__name__)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Per-Item signed weight — BUY-ish positive, SELL-ish negative. These are
# loose priors from the 8-K drift literature; they get overwritten by
# CPCV-calibrated weights if the caller supplies them.
EIGHT_K_ITEM_WEIGHTS: dict[str, float] = {
    "1.01": 0.4,   # Entry into material agreement — typically positive.
    "1.02": -0.4,  # Termination of material agreement.
    "2.01": 0.2,   # Completion of acquisition.
    "2.02": 0.0,   # Results of operations — neutral until we know the surprise sign.
    "2.03": -0.2,  # Creation of direct financial obligation.
    "2.05": -0.3,  # Costs of exit / disposal.
    "3.01": -0.3,  # Notice of delisting / failure to satisfy a rule.
    "4.01": -0.4,  # Changes in registrant's certifying accountant.
    "4.02": -0.6,  # Non-reliance on previously issued financials.
    "5.02": -0.2,  # Departure / election of directors / officers (skewed negative on departures).
    "7.01": 0.1,   # Regulation FD disclosure.
    "8.01": 0.0,   # Other events.
}

_ITEM_RE = re.compile(r"\b(\d+\.\d+)\b")


class SEC8KEventAdapter:
    """Emit a daily per-ticker score from 8-K Item triggers.

    The score is a signed sum of :data:`EIGHT_K_ITEM_WEIGHTS` over all
    8-Ks filed on each date. If a ticker files multiple 8-Ks in a day
    (uncommon but legal), their item weights are summed. Zero / missing
    days are not emitted — downstream code joins with a price panel and
    fills missing dates with 0 (or NaN, depending on consumer).
    """

    name = "sec_8k_events"

    def __init__(
        self,
        item_weights: Optional[Mapping[str, float]] = None,
        cache_dir: Optional[Path] = None,
        user_agent: str = "auto-researcher-altdata/1.0 (contact: noreply@example.com)",
        rate_limit_sleep: float = 0.12,  # ~8 req/s — under the 10 req/s cap
        timeout_s: float = 15.0,
        session: Optional[requests.Session] = None,
        ticker_cik_map: Optional[Mapping[str, str]] = None,
    ):
        self.item_weights = dict(EIGHT_K_ITEM_WEIGHTS)
        if item_weights:
            self.item_weights.update(item_weights)
        self.cache = AltDataCache(adapter=self.name, cache_dir=cache_dir)
        self.user_agent = user_agent
        self.rate_limit_sleep = rate_limit_sleep
        self.timeout_s = timeout_s
        self._session = session or requests.Session()
        self._ticker_cik: Optional[dict[str, str]] = None
        if ticker_cik_map:
            self._ticker_cik = {k.upper(): str(v).zfill(10) for k, v in ticker_cik_map.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> pd.Series:
        window = FetchWindow.from_inputs(start, end)
        iso_start, iso_end = window.iso()

        cik_map = self._resolve_cik_map(list(tickers))
        if not cik_map:
            logger.info("sec_8k: no CIKs resolved — returning empty series")
            return empty_altdata_series(self.name)

        rows: list[tuple[pd.Timestamp, str, float]] = []
        for ticker, cik10 in cik_map.items():
            cached = AltDataCache.unwrap(
                self.cache.get(ticker, iso_start, iso_end, extra=cik10)
            )
            if cached is None:
                cached = self._fetch_submissions(cik10)
                if cached:
                    self.cache.put(ticker, iso_start, iso_end, cached, extra=cik10)
                else:
                    continue

            for date, items in self._extract_8k_events(cached, window):
                score = sum(self.item_weights.get(it, 0.0) for it in items)
                if score == 0.0 and not items:
                    continue
                rows.append((date, ticker, float(score)))

        if not rows:
            return empty_altdata_series(self.name)

        df = pd.DataFrame(rows, columns=["date", "ticker", "score"])
        # Same-day multi-file: sum. Most tickers file at most one 8-K per day.
        out = df.groupby(["date", "ticker"], as_index=True)["score"].sum()
        out.name = self.name
        return out

    # ------------------------------------------------------------------
    # Ticker ↔ CIK resolution
    # ------------------------------------------------------------------

    def _resolve_cik_map(self, tickers: list[str]) -> dict[str, str]:
        if self._ticker_cik is None:
            self._ticker_cik = self._fetch_tickers_json()
        if not self._ticker_cik:
            return {}
        out: dict[str, str] = {}
        for t in tickers:
            key = t.upper()
            if key in self._ticker_cik:
                out[key] = self._ticker_cik[key]
        return out

    def _fetch_tickers_json(self) -> dict[str, str]:
        try:
            resp = self._session.get(
                _TICKERS_URL,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            logger.warning("sec_8k tickers.json fetch failed: %s", e)
            return {}
        if resp.status_code != 200:
            logger.warning("sec_8k tickers.json returned %s", resp.status_code)
            return {}
        try:
            data = resp.json()
        except ValueError as e:
            logger.warning("sec_8k tickers.json decode error: %s", e)
            return {}
        out: dict[str, str] = {}
        # SEC ships this as a dict-of-dicts keyed by str row index; each
        # record has ``cik_str`` (int) and ``ticker``.
        for row in data.values():
            try:
                ticker = str(row["ticker"]).upper()
                cik10 = str(int(row["cik_str"])).zfill(10)
                out[ticker] = cik10
            except (KeyError, TypeError, ValueError):
                continue
        return out

    # ------------------------------------------------------------------
    # Submissions fetch + parse
    # ------------------------------------------------------------------

    def _fetch_submissions(self, cik10: str) -> Optional[dict]:
        url = _SUBMISSIONS_URL.format(cik10=cik10)
        try:
            resp = self._session.get(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            logger.warning("sec_8k submissions fetch failed (%s): %s", cik10, e)
            return None
        time.sleep(self.rate_limit_sleep)  # polite pacing under the 10 req/s cap
        if resp.status_code != 200:
            logger.warning("sec_8k submissions returned %s for %s", resp.status_code, cik10)
            return None
        try:
            return resp.json()
        except ValueError as e:
            logger.warning("sec_8k submissions decode error (%s): %s", cik10, e)
            return None

    @staticmethod
    def _extract_8k_events(
        submissions: dict, window: FetchWindow
    ) -> list[tuple[pd.Timestamp, list[str]]]:
        """Yield ``(filing_date, [item_codes])`` tuples for 8-Ks in the window."""
        recent = (submissions or {}).get("filings", {}).get("recent", {})
        forms: list[str] = recent.get("form", []) or []
        dates: list[str] = recent.get("filingDate", []) or []
        items_list: list[str] = recent.get("items", []) or []
        n = min(len(forms), len(dates), len(items_list) if items_list else len(forms))

        out: list[tuple[pd.Timestamp, list[str]]] = []
        for i in range(n):
            if forms[i] != "8-K":
                continue
            try:
                dt = pd.Timestamp(dates[i]).normalize()
            except Exception:
                continue
            if dt < window.start or dt > window.end:
                continue
            items_str = items_list[i] if i < len(items_list) else ""
            items = _ITEM_RE.findall(items_str or "")
            out.append((dt, items))
        return out
