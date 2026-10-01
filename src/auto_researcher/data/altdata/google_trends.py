"""
Google Trends adapter — search-interest z-score per ticker/brand.

Why
---
Da/Engelberg/Gao (2011, "In Search of Attention") showed that spikes in
retail search interest for a ticker's brand name predict short-horizon
excess return — positive in the short window (1-2 weeks) then reversing.
Google Trends exposes this via a non-public but scraped API, which
``pytrends`` wraps.

Stability warning
-----------------
pytrends is a reverse-engineered client for a non-public API. Google
routinely IP-bans the endpoint when it detects scraping, so:

* Treat this adapter as **best-effort** — the happy path is a signal
  boost; the degraded path is an empty series and a warning.
* Always hit the cache first. CPCV reruns against cached payloads are
  fine even if the live endpoint is banned.

Import is guarded so the rest of the altdata package is usable without
pytrends installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from .base import AltDataCache, FetchWindow, empty_altdata_series

logger = logging.getLogger(__name__)

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError as _e:
    HAS_PYTRENDS = False
    TrendReq = None
    _IMPORT_ERR = _e
else:
    _IMPORT_ERR = None


class GoogleTrendsAdapter:
    """Daily search-interest index per ticker.

    Emits the raw 0-100 index from Google Trends (as a float). Callers
    should :func:`~auto_researcher.data.altdata.base.zscore_panel`
    cross-sectionally before mixing with other alt-data — the raw index
    is relative to the per-query max and doesn't compare across tickers.
    """

    name = "google_trends"

    def __init__(
        self,
        ticker_keyword: Optional[Mapping[str, str]] = None,
        cache_dir: Optional[Path] = None,
        hl: str = "en-US",
        tz_offset: int = 360,
        geo: str = "US",
        timeout_s: tuple[int, int] = (10, 25),
        retries: int = 2,
        backoff_factor: float = 1.5,
        sleep_between_tickers: float = 1.5,
        trend_req: Optional[object] = None,
    ):
        if not HAS_PYTRENDS:
            raise ImportError(
                "pytrends is required for GoogleTrendsAdapter. "
                f"Install with `pip install pytrends`. (import error: {_IMPORT_ERR!r})"
            )
        self.ticker_keyword = dict(ticker_keyword) if ticker_keyword else {}
        self.cache = AltDataCache(adapter=self.name, cache_dir=cache_dir)
        self.hl = hl
        self.tz_offset = tz_offset
        self.geo = geo
        self.sleep_between_tickers = sleep_between_tickers
        # Allow tests to inject a pre-built/stubbed ``TrendReq``-like object.
        self._trend_req = trend_req or TrendReq(
            hl=hl, tz=tz_offset, timeout=timeout_s,
            retries=retries, backoff_factor=backoff_factor,
        )

    def fetch(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> pd.Series:
        window = FetchWindow.from_inputs(start, end)
        iso_start, iso_end = window.iso()
        timeframe = f"{iso_start} {iso_end}"

        rows: list[tuple[pd.Timestamp, str, float]] = []
        for ticker in tickers:
            keyword = self._keyword_for(ticker)
            if not keyword:
                logger.debug("google_trends: no keyword mapped for %s — skipping", ticker)
                continue

            cached = AltDataCache.unwrap(
                self.cache.get(ticker, iso_start, iso_end, extra=keyword)
            )
            if cached is None:
                cached = self._fetch_keyword(keyword, timeframe)
                if cached is not None:
                    self.cache.put(ticker, iso_start, iso_end, cached, extra=keyword)
                else:
                    continue
                time.sleep(self.sleep_between_tickers)

            for entry in cached:
                try:
                    dt = pd.Timestamp(entry["date"]).normalize()
                    val = float(entry["value"])
                except (KeyError, ValueError, TypeError):
                    continue
                rows.append((dt, ticker, val))

        if not rows:
            return empty_altdata_series(self.name)

        df = pd.DataFrame(rows, columns=["date", "ticker", "value"])
        out = df.groupby(["date", "ticker"], as_index=True)["value"].mean()
        out.name = self.name
        return out

    def _keyword_for(self, ticker: str) -> Optional[str]:
        """Prefer caller-supplied brand names; fall back to the bare ticker."""
        kw = self.ticker_keyword.get(ticker.upper())
        if kw:
            return kw
        return ticker.upper()

    def _fetch_keyword(self, keyword: str, timeframe: str) -> Optional[list[dict]]:
        try:
            self._trend_req.build_payload(
                kw_list=[keyword], timeframe=timeframe, geo=self.geo, cat=0,
            )
            df = self._trend_req.interest_over_time()
        except Exception as e:
            logger.warning("google_trends fetch failed for %s: %s", keyword, e)
            return None
        if df is None or df.empty:
            return None
        if keyword not in df.columns:
            logger.warning("google_trends: expected column %s, got %s", keyword, list(df.columns))
            return None
        df = df.reset_index().rename(columns={keyword: "value"})
        return [
            {"date": row["date"].strftime("%Y-%m-%d"), "value": float(row["value"])}
            for _, row in df.iterrows()
        ]
