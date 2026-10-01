"""
Wikipedia pageviews adapter.

Why Wikipedia pageviews
-----------------------
Retail attention is a documented source of short-horizon excess return
(Da/Engelberg/Gao 2011, "In Search of Attention"). Google Trends is the
classic proxy but has gotten IP-hostile; Wikipedia pageviews is:

* free, public, and rate-limited at 100 req/s shared,
* surprisingly high-fidelity for brand-name stocks (AAPL, NVDA, TSLA),
* well-specified: daily counts per article, queryable by UTC date.

Endpoint: ``https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/...``.
We emit the log of (1 + pageviews) per ticker per day and the module
caller normalizes via :func:`~auto_researcher.data.altdata.base.zscore_panel`.

Ticker → article mapping
------------------------
The adapter ships with a small built-in map for the S&P 100 / FAANG
tickers. Anything not in the map falls back to a generated guess
(``"{Ticker}_Inc."``), which is brittle — callers should pass an
explicit ``ticker_article`` map for anything beyond that.

Failure mode
------------
Any HTTP error, timeout, or empty response returns the canonical empty
series. The adapter never raises — it logs and degrades gracefully so a
single adapter failure can't poison a composite-feature build.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import requests

from .base import AltDataCache, FetchWindow, empty_altdata_series

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/{access}/{agent}/{article}/daily/{start}/{end}"
)

# Starter map — extend as needed. Article strings are case-sensitive and
# must use underscores for spaces (Wikipedia's URL convention).
DEFAULT_TICKER_ARTICLE: dict[str, str] = {
    "AAPL": "Apple_Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet_Inc.",
    "GOOG": "Alphabet_Inc.",
    "AMZN": "Amazon_(company)",
    "NVDA": "Nvidia",
    "META": "Meta_Platforms",
    "TSLA": "Tesla,_Inc.",
    "BRK-B": "Berkshire_Hathaway",
    "JPM": "JPMorgan_Chase",
    "JNJ": "Johnson_%26_Johnson",
    "V": "Visa_Inc.",
    "WMT": "Walmart",
    "XOM": "ExxonMobil",
    "UNH": "UnitedHealth_Group",
    "HD": "Home_Depot",
    "PG": "Procter_%26_Gamble",
    "MA": "Mastercard",
    "BAC": "Bank_of_America",
    "DIS": "The_Walt_Disney_Company",
    "KO": "The_Coca-Cola_Company",
    "PEP": "PepsiCo",
    "NFLX": "Netflix",
    "ADBE": "Adobe_Inc.",
    "CRM": "Salesforce",
    "CSCO": "Cisco",
    "INTC": "Intel",
    "AMD": "AMD",
    "QCOM": "Qualcomm",
    "ORCL": "Oracle_Corporation",
    "IBM": "IBM",
    "AVGO": "Broadcom",
    "TXN": "Texas_Instruments",
    "ABNB": "Airbnb",
    "UBER": "Uber",
    "PYPL": "PayPal",
    "SHOP": "Shopify",
    "SPOT": "Spotify",
    "SBUX": "Starbucks",
    "MCD": "McDonald%27s",
    "NKE": "Nike,_Inc.",
    "COST": "Costco",
    "TGT": "Target_Corporation",
    "F": "Ford_Motor_Company",
    "GM": "General_Motors",
    "BA": "Boeing",
    "GE": "General_Electric",
    "CAT": "Caterpillar_Inc.",
    "CVX": "Chevron_Corporation",
}


class WikipediaPageviewsAdapter:
    """Fetch daily pageviews per ticker, emit ``log(1+views)``.

    The output series is NOT cross-sectionally normalized — do that
    explicitly via :func:`~auto_researcher.data.altdata.base.zscore_panel`
    at call time. This keeps the raw signal cacheable and lets different
    consumers pick their normalization.
    """

    name = "wikipedia_pageviews"

    def __init__(
        self,
        ticker_article: Optional[Mapping[str, str]] = None,
        cache_dir: Optional[Path] = None,
        user_agent: str = "auto-researcher-altdata/1.0 (contact: noreply@example.com)",
        access: str = "all-access",
        agent_type: str = "user",
        timeout_s: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        self.ticker_article = dict(DEFAULT_TICKER_ARTICLE)
        if ticker_article:
            self.ticker_article.update(ticker_article)
        self.cache = AltDataCache(adapter=self.name, cache_dir=cache_dir)
        self.user_agent = user_agent
        self.access = access
        self.agent_type = agent_type
        self.timeout_s = timeout_s
        # Accept an injected session so tests can stub ``requests.get`` cleanly.
        self._session = session or requests.Session()

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
        start_str = window.start.strftime("%Y%m%d")
        end_str = window.end.strftime("%Y%m%d")
        iso_start, iso_end = window.iso()

        rows: list[tuple[pd.Timestamp, str, float]] = []
        for ticker in tickers:
            article = self._article_for(ticker)
            if article is None:
                logger.debug("wikipedia: no article mapped for %s — skipping", ticker)
                continue

            cached = AltDataCache.unwrap(self.cache.get(ticker, iso_start, iso_end))
            if cached is None:
                cached = self._fetch_article(article, start_str, end_str)
                if cached:
                    self.cache.put(ticker, iso_start, iso_end, cached)

            if not cached:
                continue

            for item in cached.get("items", []):
                ts_raw = item.get("timestamp", "")
                views = item.get("views", 0)
                try:
                    dt = pd.Timestamp(ts_raw[:8])
                except Exception:
                    continue
                rows.append((dt, ticker, float(np.log1p(max(views, 0)))))

        if not rows:
            return empty_altdata_series(self.name)

        df = pd.DataFrame(rows, columns=["date", "ticker", "views_log1p"])
        # Collapse duplicate (date, ticker) pairs — API occasionally returns them.
        df = df.groupby(["date", "ticker"], as_index=True)["views_log1p"].mean()
        df.name = self.name
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _article_for(self, ticker: str) -> Optional[str]:
        t = ticker.upper().strip()
        if t in self.ticker_article:
            return self.ticker_article[t]
        return None

    def _fetch_article(
        self, article: str, start_str: str, end_str: str
    ) -> Optional[dict]:
        url = _BASE_URL.format(
            access=self.access,
            agent=self.agent_type,
            article=article,
            start=start_str,
            end=end_str,
        )
        try:
            resp = self._session.get(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            logger.warning("wikipedia fetch failed for %s: %s", article, e)
            return None
        if resp.status_code != 200:
            # 404 is common (article title mismatch) — log at info, not warning.
            level = logger.info if resp.status_code == 404 else logger.warning
            level(
                "wikipedia %s returned %s for %s",
                resp.url, resp.status_code, article,
            )
            return None
        try:
            return resp.json()
        except ValueError as e:
            logger.warning("wikipedia JSON decode error for %s: %s", article, e)
            return None
