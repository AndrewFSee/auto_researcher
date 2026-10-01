"""
Earnings history with analyst consensus from Financial Modeling Prep.

``https://financialmodelingprep.com/stable/earnings?symbol=...`` returns every
reported quarter with its date, actual EPS/revenue and the consensus estimate,
back to the 1990s for large caps. The free plan allows about 250 requests per
day, so downloads are cached one file per symbol and resume across days.

Data-quality caveat: some older rows have an "estimate" equal to the actual,
down to the dollar of revenue, which real consensus essentially never is.
Those rows look backfilled with actuals and are flagged ``suspect_backfill``;
surprise events exclude them.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FMP_EARNINGS_URL = "https://financialmodelingprep.com/stable/earnings"
COLUMNS = ["symbol", "date", "eps_actual", "eps_estimated", "revenue_actual",
           "revenue_estimated", "last_updated", "suspect_backfill"]


class FMPQuotaExceeded(RuntimeError):
    """The daily request limit was reached; rerun later to resume."""


def parse_fmp_earnings(payload: list[dict], symbol: str) -> pd.DataFrame:
    """Normalize an FMP ``stable/earnings`` response into ``COLUMNS``."""
    if not payload:
        return pd.DataFrame(columns=COLUMNS)
    raw = pd.DataFrame(payload)
    df = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(raw.get("date"), errors="coerce"),
        "eps_actual": pd.to_numeric(raw.get("epsActual"), errors="coerce"),
        "eps_estimated": pd.to_numeric(raw.get("epsEstimated"), errors="coerce"),
        "revenue_actual": pd.to_numeric(raw.get("revenueActual"), errors="coerce"),
        "revenue_estimated": pd.to_numeric(raw.get("revenueEstimated"), errors="coerce"),
        "last_updated": pd.to_datetime(raw.get("lastUpdated"), errors="coerce"),
    })
    df["suspect_backfill"] = (
        df["revenue_actual"].notna()
        & (df["revenue_actual"] != 0)
        & (df["revenue_actual"] == df["revenue_estimated"])
    )
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)[COLUMNS]


@dataclass
class DownloadSummary:
    downloaded: list[str] = field(default_factory=list)
    already_cached: int = 0
    unavailable: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    quota_hit: bool = False


def download_fmp_earnings(
    symbols: list[str],
    api_key: str,
    cache_dir: Path,
    max_requests: int = 240,
    pause: float = 0.3,
    session: Any = None,
    retry_unavailable: bool = False,
) -> DownloadSummary:
    """
    Download earnings history for ``symbols`` not yet cached.

    Each symbol is saved to ``cache_dir/<SYMBOL>.parquet``; symbols the plan
    cannot serve (the free plan covers only a fixed list of popular stocks) are
    recorded in ``cache_dir/_unavailable.json`` and skipped unless
    ``retry_unavailable`` is set, e.g. after upgrading the plan. Stops after ``max_requests`` new requests or when FMP reports the
    daily limit; rerunning resumes where it stopped.
    """
    import requests

    session = session or requests.Session()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "_unavailable.json"
    unavailable = set()
    if marker.exists() and not retry_unavailable:
        unavailable = set(json.loads(marker.read_text()))

    summary = DownloadSummary()
    todo = []
    for sym in dict.fromkeys(symbols):
        if (cache_dir / f"{sym}.parquet").exists() or sym in unavailable:
            summary.already_cached += 1
        else:
            todo.append(sym)

    requests_made = 0
    for i, sym in enumerate(todo):
        if requests_made >= max_requests:
            summary.remaining = todo[i:]
            break
        resp = session.get(FMP_EARNINGS_URL, params={"symbol": sym, "apikey": api_key}, timeout=30)
        requests_made += 1
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            message = str(body.get("Error Message", ""))
        elif resp.status_code != 200:
            message = str(getattr(resp, "text", "") or body or "")
        else:
            message = ""
        if resp.status_code == 429 or "limit" in message.lower():
            summary.quota_hit = True
            summary.remaining = todo[i:]
            logger.warning("FMP daily limit reached after %d requests; rerun later", requests_made - 1)
            break
        if resp.status_code in (401, 402, 403) or message:
            unavailable.add(sym)
            summary.unavailable.append(sym)
            logger.info("FMP has no data for %s on this plan: %s", sym, message[:80])
        elif resp.status_code == 200 and isinstance(body, list):
            parse_fmp_earnings(body, sym).to_parquet(cache_dir / f"{sym}.parquet")
            summary.downloaded.append(sym)
        else:
            logger.warning("Unexpected FMP response for %s: HTTP %s", sym, resp.status_code)
            summary.remaining.append(sym)
        time.sleep(pause)

    marker.write_text(json.dumps(sorted(unavailable)))
    return summary


def load_fmp_earnings(cache_dir: Path) -> pd.DataFrame:
    """All cached symbols concatenated (empty frame if nothing is cached)."""
    files = sorted(Path(cache_dir).glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def consensus_surprises(df: pd.DataFrame, min_abs_estimate: float = 0.01) -> pd.DataFrame:
    """
    Reported quarters as surprise events: ``symbol, announce_date, sue``.

    ``sue = (actual - estimate) / max(|estimate|, min_abs_estimate)``. Rows
    without an actual (future quarters), without an estimate, or flagged as
    backfilled are dropped.
    """
    ok = df["eps_actual"].notna() & df["eps_estimated"].notna() & ~df["suspect_backfill"]
    d = df.loc[ok]
    denom = np.maximum(d["eps_estimated"].abs(), min_abs_estimate)
    return pd.DataFrame({
        "symbol": d["symbol"].to_numpy(),
        "announce_date": pd.to_datetime(d["date"]).to_numpy(),
        "sue": ((d["eps_actual"] - d["eps_estimated"]) / denom).to_numpy(),
    })
