"""
Earnings history with analyst consensus from Alpha Vantage.

``function=EARNINGS`` returns every reported quarter with its report date,
reported EPS, consensus estimate and (recently) whether the report came before
the open or after the close, back to about 1996 for long-listed companies. The
free key allows 25 requests a day and about one per second, so downloads are
cached one file per symbol and resume across days.

Rows are normalized to the same columns as ``data.fmp_earnings`` so both feed
``fmp_earnings.consensus_surprises`` unchanged.

Data-quality caveat: before 2010 an unusually large share of quarters report an
estimate exactly equal to the actual, which looks like actuals copied into the
estimate field. Alpha Vantage has no revenue figures to confirm it, so
pre-2010 exact matches are flagged ``suspect_backfill``. Such rows would rank
as zero surprises, so dropping them loses events rather than adding bias.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from auto_researcher.data.fmp_earnings import COLUMNS, DownloadSummary

logger = logging.getLogger(__name__)

AV_URL = "https://www.alphavantage.co/query"
BACKFILL_CUTOFF = pd.Timestamp("2010-01-01")
AV_COLUMNS = [*COLUMNS, "report_time"]


def parse_av_earnings(payload: dict, symbol: str) -> pd.DataFrame:
    """Normalize an Alpha Vantage ``EARNINGS`` response."""
    rows = payload.get("quarterlyEarnings") or []
    if not rows:
        return pd.DataFrame(columns=AV_COLUMNS)
    raw = pd.DataFrame(rows)

    def num(col: str) -> pd.Series:
        return pd.to_numeric(raw.get(col, pd.Series(index=raw.index, dtype=object)), errors="coerce")

    df = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(raw.get("reportedDate"), errors="coerce"),
        "eps_actual": num("reportedEPS"),
        "eps_estimated": num("estimatedEPS"),
        "revenue_actual": float("nan"),
        "revenue_estimated": float("nan"),
        "last_updated": pd.Timestamp.now().normalize(),  # download date
        "report_time": raw.get("reportTime", pd.Series(index=raw.index, dtype=object)),
    })
    df["suspect_backfill"] = (
        (df["date"] < BACKFILL_CUTOFF)
        & df["eps_actual"].notna()
        & (df["eps_actual"] == df["eps_estimated"])
    )
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)[AV_COLUMNS]


def _throttle_note(body: object) -> str:
    """Alpha Vantage signals rate limits with HTTP 200 and an Information/Note field."""
    if isinstance(body, dict):
        return str(body.get("Information") or body.get("Note") or "")
    return ""


def download_av_earnings(
    symbols: list[str],
    api_key: str,
    cache_dir: Path,
    max_requests: int = 25,
    pause: float = 1.5,
    session: Any = None,
    retry_unavailable: bool = False,
    refresh_after_days: float | None = None,
) -> DownloadSummary:
    """
    Download earnings history for ``symbols`` not yet cached.

    Each symbol is saved to ``cache_dir/<SYMBOL>.parquet``; symbols Alpha
    Vantage does not know are recorded in ``cache_dir/_unavailable.json``. A
    throttling reply is retried once after a pause; a second one means the daily
    quota is used up, and the run stops so the next one resumes.

    With ``refresh_after_days``, budget left after the uncached symbols is
    spent re-downloading cached files older than that, oldest first, so a
    daily run keeps the newest quarters current once the universe is complete.
    """
    import requests

    session = session or requests.Session()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "_unavailable.json"
    unavailable: set[str] = set()
    if marker.exists() and not retry_unavailable:
        unavailable = set(json.loads(marker.read_text()))

    summary = DownloadSummary()
    todo, stale = [], []
    now = time.time()
    for sym in dict.fromkeys(symbols):
        path = cache_dir / f"{sym}.parquet"
        if sym in unavailable:
            summary.already_cached += 1
        elif not path.exists():
            todo.append(sym)
        else:
            summary.already_cached += 1
            age_days = (now - path.stat().st_mtime) / 86400
            if refresh_after_days is not None and age_days > refresh_after_days:
                stale.append((path.stat().st_mtime, sym))
    n_new = len(todo)
    todo += [sym for _, sym in sorted(stale)]

    requests_made = 0
    for i, sym in enumerate(todo):
        if requests_made >= max_requests:
            summary.remaining = todo[i:n_new]
            break
        body = None
        for attempt in range(2):
            resp = session.get(AV_URL, params={"function": "EARNINGS", "symbol": sym,
                                               "apikey": api_key}, timeout=30)
            requests_made += 1
            try:
                body = resp.json()
            except ValueError:
                body = None
            if not _throttle_note(body):
                break
            if attempt == 0:
                time.sleep(max(pause, 2.0))
        if _throttle_note(body):
            summary.quota_hit = True
            summary.remaining = todo[i:n_new]
            logger.warning("Alpha Vantage quota reached after %d requests; rerun tomorrow",
                           requests_made)
            break
        if isinstance(body, dict) and body.get("quarterlyEarnings"):
            parse_av_earnings(body, sym).to_parquet(cache_dir / f"{sym}.parquet")
            summary.downloaded.append(sym)
        elif i >= n_new:
            logger.warning("Refresh of %s returned no earnings; keeping the cached file", sym)
        else:
            unavailable.add(sym)
            summary.unavailable.append(sym)
            logger.info("Alpha Vantage has no earnings for %s: %s", sym, str(body)[:80])
        time.sleep(pause)

    marker.write_text(json.dumps(sorted(unavailable)))
    return summary
