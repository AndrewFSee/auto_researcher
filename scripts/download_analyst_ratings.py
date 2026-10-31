"""
Download analyst rating actions (Yahoo, FMP) for the S&P 500 and validate them.

Downloads (``auto_researcher.data.analyst_ratings``):

* Yahoo for every stock in the universe: actions with broker, old and new rating,
  price targets and the time of day.
* FMP ``stable/grades`` for the stocks the free plan serves (by default the ones
  it served for ``download_fmp_earnings.py``), to cross-check Yahoo.

The report (``docs/results/analyst_ratings_validation.md``) checks the data
before any signal is tested: coverage by year, time of day, agreement between
the sources, and whether the timestamps are right. For that last check it looks
only at returns from two days before to one day after each action (the price
should move on the first session that could react to it, not the day before);
returns after that, where a tradable signal would be, are not looked at.

Example::

    python scripts/download_analyst_ratings.py              # download, then validate
    python scripts/download_analyst_ratings.py --validate-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_fmp_earnings import universe  # noqa: E402
from feature_research import BENCHMARK, load_ohlcv  # noqa: E402

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.data.analyst_ratings import (  # noqa: E402
    download_fmp_grades,
    download_yahoo_ratings,
    first_reacting_session,
    load_ratings,
    normalize_firm,
    session_bucket,
)

logger = logging.getLogger("download_analyst_ratings")
CACHE = ROOT / "data" / "research_cache"
YAHOO_DENSE_FROM = "2019-01-01"
WINDOW = (-2, -1, 0, 1)  # trading days around the first session that could react


def event_returns(df: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Excess return vs SPY on each day of ``WINDOW`` for each action."""
    rets = close.pct_change(fill_method=None)
    excess = rets.sub(rets[BENCHMARK], axis=0)
    cal = close.index
    day0 = first_reacting_session(df, cal)
    col = {t: j for j, t in enumerate(close.columns)}
    arr = excess.to_numpy()
    out = {}
    for k in WINDOW:
        idx = day0.to_numpy() + k
        cols = df["symbol"].map(col)
        ok = cols.notna().to_numpy() & (idx >= 1) & (idx < len(cal))
        vals = np.full(len(df), np.nan)
        vals[ok] = arr[idx[ok], cols[ok].astype(int).to_numpy()]
        out[k] = vals
    return pd.DataFrame(out, index=df.index)


def match_sources(yahoo: pd.DataFrame, fmp: pd.DataFrame, days: int = 2) -> pd.DataFrame:
    """For each FMP up/down/init action, the nearest Yahoo action by the same broker on the stock."""
    keep = ["up", "down", "init"]
    fmp = fmp[fmp["date"] >= YAHOO_DENSE_FROM]  # Yahoo is patchy before 2019
    y = yahoo[yahoo["action"].isin(keep) & yahoo["symbol"].isin(fmp["symbol"].unique())].copy()
    f = fmp[fmp["action"].isin(keep)].copy()
    for d in (y, f):
        d["firm_key"] = d["firm"].map(normalize_firm)
    merged = f.merge(y, on=["symbol", "firm_key"], suffixes=("_fmp", "_yahoo"))
    merged["lag_days"] = (merged["date_yahoo"] - merged["date_fmp"]).dt.days
    merged = merged[merged["lag_days"].abs() <= days]
    # the nearest Yahoo action for each FMP action
    return merged.sort_values("lag_days", key=abs).drop_duplicates(
        ["symbol", "firm_key", "date_fmp", "action_fmp"])


def write_report(yahoo: pd.DataFrame, fmp: pd.DataFrame, close: pd.DataFrame, n_universe: int,
                 out: Path) -> None:
    pct = "{:.0%}".format
    lines = [
        "# Analyst rating actions: data validation", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/download_analyst_ratings.py`. Yahoo: "
        f"{yahoo['symbol'].nunique()} of {n_universe} stocks, {len(yahoo):,} actions. FMP (free plan, "
        f"cross-check only): {fmp['symbol'].nunique()} stocks, {len(fmp):,} actions.", "",
        "## Coverage by year (Yahoo)", "",
        "| Year | Actions | Stocks | Upgrades | Downgrades | Initiations | Maintained / reiterated | "
        "With price target |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    y = yahoo.assign(year=yahoo["date"].dt.year)
    for year, g in y.groupby("year"):
        a = g["action"].value_counts()
        lines.append(f"| {year} | {len(g):,} | {g['symbol'].nunique()} | {a.get('up', 0):,} | "
                     f"{a.get('down', 0):,} | {a.get('init', 0):,} | "
                     f"{a.get('main', 0) + a.get('reit', 0):,} | {pct(g['price_target'].notna().mean())} |")
    other = sorted(set(yahoo["action"]) - {"up", "down", "init", "main", "reit"})
    if other:
        lines += ["", f"Other Yahoo action codes: {', '.join(other)}."]

    buckets = session_bucket(yahoo["timestamp_et"]).value_counts(normalize=True)
    midnight = (yahoo["timestamp_et"].dt.hour.eq(0) & yahoo["timestamp_et"].dt.minute.eq(0)).mean()
    lines += ["", "## Time of day (Yahoo, New York time)", "",
              "| Before the open | During the session | After the close |", "| ---: | ---: | ---: |",
              f"| {pct(buckets.get('pre-market', 0))} | {pct(buckets.get('session', 0))} | "
              f"{pct(buckets.get('after close', 0))} |", "",
              f"{pct(midnight)} of timestamps are exactly midnight (probably a date without a time)."]

    if not fmp.empty:
        matched = match_sources(yahoo, fmp)
        f = fmp[fmp["action"].isin(["up", "down", "init"]) & (fmp["date"] >= YAHOO_DENSE_FROM)]
        same_grade = (matched["to_grade_fmp"].str.lower().str.strip()
                      == matched["to_grade_yahoo"].str.lower().str.strip())
        lag = matched["lag_days"].value_counts(normalize=True)
        lines += ["", "## Yahoo vs FMP (upgrades, downgrades, initiations)", "",
                  f"On the {fmp['symbol'].nunique()} stocks FMP serves, from "
                  f"{YAHOO_DENSE_FROM}: {len(f):,} FMP actions, "
                  f"{pct(len(matched) / max(len(f), 1))} found in Yahoo (same stock and broker, within "
                  f"2 days); of those, same day {pct(lag.get(0, 0))}, FMP a day earlier "
                  f"{pct(lag.get(1, 0))}, a day later {pct(lag.get(-1, 0))}; same action "
                  f"{pct((matched['action_fmp'] == matched['action_yahoo']).mean())}, same new rating "
                  f"{pct(same_grade.mean())}.", ""]

    lines += ["", "## Timing check: returns around each action", "",
              "Day 0 is the first session that could react: the action's day if it came before "
              "the close, the next session if after (FMP: its date). Mean return in excess of "
              "SPY; a right timestamp puts the move on day 0, not day -1. Later days are not "
              "looked at here (that is where a signal would be).", "",
              "| Source | Action | When | N | Day -2 | Day -1 | Day 0 | Day +1 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, src in (("Yahoo", yahoo), ("FMP", fmp)):
        if src.empty:
            continue
        for action in ("up", "down"):
            ev = src[src["action"] == action]
            if ev.empty:
                continue
            ret = event_returns(ev, close)
            when = session_bucket(ev["timestamp_et"])
            groups = [("all", ret)] + [(b, ret[when == b]) for b in
                                       ("pre-market", "session", "after close") if (when == b).any()]
            for label, r in groups:
                r = r.dropna()
                if len(r) < 30:
                    continue
                m = r.mean()
                lines.append(f"| {name} | {'upgrade' if action == 'up' else 'downgrade'} | {label} | "
                             f"{len(r):,} | " + " | ".join(f"{m[k]:+.2%}" for k in WINDOW) + " |")
    lines += ["", "Universe: current S&P 500 members (survivorship: companies that left the index "
              "are missing)."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out)


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yahoo-dir", default=str(CACHE / "analyst_ratings_yahoo"))
    ap.add_argument("--fmp-dir", default=str(CACHE / "analyst_ratings_fmp"))
    ap.add_argument("--sources", nargs="+", choices=["yahoo", "fmp"], default=["yahoo", "fmp"])
    ap.add_argument("--fmp-all", action="store_true",
                    help="ask FMP for every stock (most answer 402 on the free plan)")
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--retry-empty", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "analyst_ratings_validation.md"))
    args = ap.parse_args(argv)

    symbols = universe()
    if not args.validate_only:
        if "yahoo" in args.sources:
            s = download_yahoo_ratings(symbols, Path(args.yahoo_dir), pause=args.pause,
                                       retry_empty=args.retry_empty)
            logger.info("Yahoo: downloaded %d, cached %d, empty %d, remaining %d", len(s.downloaded),
                        s.already_cached, len(s.unavailable), len(s.remaining))
        if "fmp" in args.sources:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
            fmp_symbols = symbols if args.fmp_all else sorted(
                p.stem for p in (CACHE / "fmp_earnings").glob("*.parquet"))
            s = download_fmp_grades(fmp_symbols, os.environ["FMP_API_KEY"], Path(args.fmp_dir))
            logger.info("FMP: downloaded %d, cached %d, refused %d, remaining %d", len(s.downloaded),
                        s.already_cached, len(s.unavailable), len(s.remaining))

    yahoo, fmp = load_ratings(Path(args.yahoo_dir)), load_ratings(Path(args.fmp_dir))
    names = sorted(set(yahoo["symbol"]) | set(fmp["symbol"]))
    close, _ = load_ohlcv(names + [BENCHMARK], CACHE, start="2012-01-01")
    write_report(yahoo, fmp, close, len(symbols), Path(args.out))


if __name__ == "__main__":
    main()
