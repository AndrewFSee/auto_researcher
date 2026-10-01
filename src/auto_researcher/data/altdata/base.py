"""
Shared plumbing for alt-data adapters.

* :class:`AltDataAdapter` — Protocol every adapter implements.
* :class:`AltDataCache`  — JSON-on-disk cache keyed by ``(adapter, ticker, window)``.
* :func:`zscore_panel`   — cross-sectional z-score normalizer, used by all adapters
  so their outputs are comparable across sources.

Design notes
------------
Alt-data fetches are slow and rate-limited — Wikipedia pageviews caps at
100 req/s shared, EDGAR is throttled, and Google Trends is notoriously
IP-banned. Every successful fetch is cached on disk so a CPCV rerun
never pays the network cost twice, and failed fetches short-circuit
with an empty series so one flaky source can't poison the composite.

Normalization lives here (not in each adapter) because every downstream
consumer wants the same thing: a cross-sectional z-score that can be
dropped into the composite without rescaling. Raw page-view counts and
raw 8-K counts should not be comparable by magnitude — but their
cross-sectional ranks on a given date should.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NormalizeMode = Literal["zscore_xsec", "zscore_time", "raw"]


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

class AltDataAdapter(Protocol):
    """Contract every alt-data adapter fulfills.

    Implementations may accept extra kwargs in ``__init__`` (API creds,
    cache dir, sampling options). But ``fetch`` must always return a
    ``pd.Series`` indexed by ``(date, ticker)`` — even if empty — so
    downstream code never has to branch on adapter identity.
    """

    name: str

    def fetch(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> pd.Series: ...


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class AltDataCache:
    """On-disk JSON cache for raw fetch responses, keyed by a hash of inputs.

    One file per ``(adapter, ticker, start, end, extra_key)`` tuple. The
    value is whatever the adapter wants to persist — usually the raw
    JSON response from the upstream API. Adapters are responsible for
    their own parsing; the cache is format-agnostic.
    """

    def __init__(self, adapter: str, cache_dir: Optional[Path] = None):
        self.adapter = adapter
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(
        adapter: str, ticker: str, start: str, end: str, extra: str = ""
    ) -> str:
        h = hashlib.sha256()
        for part in (adapter, ticker, start, end, extra):
            h.update(part.encode())
            h.update(b"|")
        return h.hexdigest()[:32]

    def get(
        self, ticker: str, start: str, end: str, extra: str = ""
    ) -> Optional[dict]:
        if self.cache_dir is None:
            return None
        key = self._key(self.adapter, ticker, start, end, extra)
        f = self.cache_dir / f"{key}.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("altdata cache read error (%s, %s): %s", self.adapter, ticker, e)
            return None

    def put(
        self, ticker: str, start: str, end: str, payload: dict, extra: str = ""
    ) -> None:
        if self.cache_dir is None:
            return
        key = self._key(self.adapter, ticker, start, end, extra)
        f = self.cache_dir / f"{key}.json"
        try:
            f.write_text(
                json.dumps({"_ts": time.time(), "payload": payload}, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("altdata cache write error (%s, %s): %s", self.adapter, ticker, e)

    @staticmethod
    def unwrap(cached: Optional[dict]) -> Optional[dict]:
        """Return the ``payload`` field from a cached record, if present."""
        if cached is None:
            return None
        return cached.get("payload", cached)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def zscore_panel(series: pd.Series, mode: NormalizeMode = "zscore_xsec") -> pd.Series:
    """Normalize a ``(date, ticker)`` series.

    ``zscore_xsec`` is the default — rescale *within each date* so the
    score has mean 0 and std 1 across tickers. This is what you almost
    always want for a cross-sectional ranker: a hot WSB post count on
    2022-01-28 should be judged against *that day's* universe, not the
    all-time baseline. ``zscore_time`` does the transpose (per-ticker
    history), and ``raw`` is a no-op for debugging.
    """
    if series.empty:
        return series
    if mode == "raw":
        return series.copy()

    if mode == "zscore_xsec":
        group = series.groupby(level=0)
    elif mode == "zscore_time":
        group = series.groupby(level=1)
    else:
        raise ValueError(f"unknown normalize mode: {mode}")

    def _z(s: pd.Series) -> pd.Series:
        std = s.std()
        if not np.isfinite(std) or std < 1e-12:
            return s * 0.0
        return (s - s.mean()) / std

    return group.transform(_z)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FetchWindow:
    """Resolved start/end timestamps for consistent cache keying."""

    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_inputs(
        cls, start: pd.Timestamp | str, end: pd.Timestamp | str
    ) -> "FetchWindow":
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize()
        if e < s:
            raise ValueError(f"end {e} before start {s}")
        return cls(s, e)

    def iso(self) -> tuple[str, str]:
        return self.start.strftime("%Y-%m-%d"), self.end.strftime("%Y-%m-%d")


def empty_altdata_series(name: str) -> pd.Series:
    """Return the canonical empty Series for an adapter that produced nothing."""
    return pd.Series(
        [],
        index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
        name=name,
        dtype=float,
    )


def daily_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Business-day range — mirrors the main feature pipeline's calendar."""
    return pd.date_range(start, end, freq="B")
