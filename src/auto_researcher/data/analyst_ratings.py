"""
Analyst rating actions (upgrades, downgrades, initiations) and price targets.

Two free sources, normalized to the same columns (``COLUMNS``):

* **Yahoo** (``yfinance.Ticker(sym).upgrades_downgrades``): every action with the
  broker, old and new rating, and old and new price target, time-stamped to the
  second (Yahoo's epoch time, i.e. UTC). Dense from 2019, patchy 2012-2018.
* **FMP** (``stable/grades``): actions back to 2012 with a date but no time; the
  free plan serves only a fixed list of popular stocks (others answer HTTP 402),
  so it is used to cross-check Yahoo.

``timestamp_et`` (Yahoo only) is New York time, which decides whether an action
came before the open, during the session or after the close. ``date`` is the
New York calendar date for Yahoo and FMP's own date for FMP.

Downloads are cached one file per symbol and resume across runs, like
``data.fmp_earnings``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auto_researcher.data.fmp_earnings import DownloadSummary

logger = logging.getLogger(__name__)

FMP_GRADES_URL = "https://financialmodelingprep.com/stable/grades"
COLUMNS = ["symbol", "source", "date", "timestamp_et", "firm", "action", "from_grade",
           "to_grade", "price_target", "prior_price_target"]
ACTIONS = ("up", "down", "init", "main", "reit")
OPEN, CLOSE = 9.5, 16.0  # New York market hours, in hours
_FMP_ACTIONS = {"upgrade": "up", "downgrade": "down", "maintain": "main", "initialize": "init",
                "initiate": "init", "init": "init", "reiterate": "reit", "reiterated": "reit"}


def normalize_firm(name: object) -> str:
    """A broker name reduced for matching across sources ("Morgan Stanley & Co." -> "morgan stanley")."""
    text = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower())
    text = re.sub(r"\b(inc|llc|ltd|co|corp|corporation|group|securities|capital markets|capital|"
                  r"partners|research|financial|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_yahoo_ratings(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize ``yfinance``'s ``upgrades_downgrades`` frame (index = UTC time of the action)."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.reset_index()
    stamp = pd.to_datetime(df[df.columns[0]], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    out = pd.DataFrame({
        "symbol": symbol,
        "source": "yahoo",
        "date": stamp.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"),
        "timestamp_et": stamp.dt.tz_localize(None).astype("datetime64[ns]"),
        "firm": df.get("Firm"),
        "action": df.get("Action").astype(str).str.lower(),
        "from_grade": df.get("FromGrade"),
        "to_grade": df.get("ToGrade"),
        "price_target": pd.to_numeric(df.get("currentPriceTarget"), errors="coerce"),
        "prior_price_target": pd.to_numeric(df.get("priorPriceTarget"), errors="coerce"),
    })
    # Yahoo uses 0 for "no target"
    for col in ("price_target", "prior_price_target"):
        out.loc[out[col] <= 0, col] = float("nan")
    return out.dropna(subset=["date"]).sort_values("timestamp_et").reset_index(drop=True)[COLUMNS]


def parse_fmp_grades(payload: list[dict], symbol: str) -> pd.DataFrame:
    """Normalize an FMP ``stable/grades`` response."""
    if not payload:
        return pd.DataFrame(columns=COLUMNS)
    raw = pd.DataFrame(payload)
    out = pd.DataFrame({
        "symbol": symbol,
        "source": "fmp",
        "date": pd.to_datetime(raw.get("date"), errors="coerce").astype("datetime64[ns]"),
        "timestamp_et": pd.NaT,
        "firm": raw.get("gradingCompany"),
        "action": raw.get("action").astype(str).str.lower().map(lambda a: _FMP_ACTIONS.get(a, a)),
        "from_grade": raw.get("previousGrade"),
        "to_grade": raw.get("newGrade"),
        "price_target": float("nan"),
        "prior_price_target": float("nan"),
    })
    out["timestamp_et"] = out["timestamp_et"].astype("datetime64[ns]")
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)[COLUMNS]


def session_bucket(ts: pd.Series) -> pd.Series:
    """'pre-market', 'session' or 'after close' from a New York timestamp; NaT -> 'date only'."""
    hour = ts.dt.hour + ts.dt.minute / 60
    out = pd.Series("date only", index=ts.index)
    out[ts.notna() & (hour < OPEN)] = "pre-market"
    out[ts.notna() & (hour >= OPEN) & (hour < CLOSE)] = "session"
    out[ts.notna() & (hour >= CLOSE)] = "after close"
    return out


def first_reacting_session(df: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    """Index into ``cal`` of the first session whose close comes after the action."""
    dates = df["date"].to_numpy(dtype="datetime64[ns]")
    after_close = (session_bucket(df["timestamp_et"]) == "after close").to_numpy()
    pos = cal.searchsorted(dates, side="left")      # first session on or after the date
    nxt = cal.searchsorted(dates, side="right")     # first session after the date
    return pd.Series(np.where(after_close, nxt, pos), index=df.index)


def _yahoo_frame(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(symbol).upgrades_downgrades


def download_yahoo_ratings(
    symbols: list[str],
    cache_dir: Path,
    pause: float = 1.0,
    fetch: Callable[[str], pd.DataFrame] = _yahoo_frame,
    retry_empty: bool = False,
) -> DownloadSummary:
    """
    Download Yahoo rating histories for ``symbols`` not yet cached.

    An empty reply is ambiguous (no coverage, or a throttled request that
    yfinance turned into an empty frame), so empty symbols are listed in
    ``_empty.json`` and retried with ``retry_empty``. A rate-limit error stops
    the run; rerunning resumes.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "_empty.json"
    empty: set[str] = set() if retry_empty or not marker.exists() else set(json.loads(marker.read_text()))
    summary = DownloadSummary()
    todo = []
    for sym in dict.fromkeys(symbols):
        if (cache_dir / f"{sym}.parquet").exists() or sym in empty:
            summary.already_cached += 1
        else:
            todo.append(sym)
    for i, sym in enumerate(todo):
        try:
            raw = fetch(sym)
        except Exception as exc:  # yfinance raises its own types for throttling and gaps
            if "rate" in type(exc).__name__.lower() or "too many" in str(exc).lower():
                summary.quota_hit = True
                summary.remaining = todo[i:]
                logger.warning("Yahoo rate limit after %d symbols; rerun later", i)
                break
            raw = None
            logger.info("No Yahoo rating history for %s (%s)", sym, str(exc)[:80])
        df = parse_yahoo_ratings(raw, sym)
        if df.empty:
            empty.add(sym)
            summary.unavailable.append(sym)
        else:
            df.to_parquet(cache_dir / f"{sym}.parquet")
            summary.downloaded.append(sym)
            empty.discard(sym)
        time.sleep(pause)
    marker.write_text(json.dumps(sorted(empty)))
    return summary


def download_fmp_grades(
    symbols: list[str],
    api_key: str,
    cache_dir: Path,
    max_requests: int = 240,
    pause: float = 0.3,
    session: Any = None,
) -> DownloadSummary:
    """FMP grades for ``symbols``; symbols the plan refuses (HTTP 402) go to ``_unavailable.json``."""
    import requests

    session = session or requests.Session()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "_unavailable.json"
    unavailable = set(json.loads(marker.read_text())) if marker.exists() else set()
    summary = DownloadSummary()
    todo = []
    for sym in dict.fromkeys(symbols):
        if (cache_dir / f"{sym}.parquet").exists() or sym in unavailable:
            summary.already_cached += 1
        else:
            todo.append(sym)
    for i, sym in enumerate(todo):
        if i >= max_requests:
            summary.remaining = todo[i:]
            break
        try:
            resp = session.get(FMP_GRADES_URL, params={"symbol": sym, "apikey": api_key}, timeout=30)
        except requests.RequestException as exc:
            summary.remaining = todo[i:]
            logger.warning("Network error for %s (%s); the next run resumes", sym, exc)
            break
        text = getattr(resp, "text", "") or ""
        if resp.status_code == 429 or "limit reach" in text.lower():
            summary.quota_hit = True
            summary.remaining = todo[i:]
            logger.warning("FMP daily limit reached after %d requests", i)
            break
        if resp.status_code in (401, 402, 403):
            unavailable.add(sym)
            summary.unavailable.append(sym)
        elif resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                body = None
            if isinstance(body, list):
                parse_fmp_grades(body, sym).to_parquet(cache_dir / f"{sym}.parquet")
                summary.downloaded.append(sym)
            else:
                summary.remaining.append(sym)
        else:
            logger.warning("Unexpected FMP reply for %s: HTTP %s", sym, resp.status_code)
            summary.remaining.append(sym)
        time.sleep(pause)
    marker.write_text(json.dumps(sorted(unavailable)))
    return summary


def load_ratings(cache_dir: Path) -> pd.DataFrame:
    """All cached rating actions of one source."""
    files = sorted(Path(cache_dir).glob("*.parquet"))
    frames = [f for f in (pd.read_parquet(p) for p in files) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for col in ("date", "timestamp_et"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df[COLUMNS]
