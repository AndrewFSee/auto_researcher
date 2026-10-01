"""
Event-driven post-earnings-announcement-drift strategy.

Rules (fixed before running; see ``backtest.event_strategy``):

* Surprise sources: ``ts_sue`` (year-over-year EPS change scaled by its
  volatility, 2014 onward, from DefeatBeta EPS history) and ``analyst``
  ((actual - consensus) / |consensus| from ``data/pead_backtest_results.parquet``,
  2023 onward). Announcement dates are inferred from SEC filings and usable
  two trading days after filing.
* Each surprise is ranked against all announcements of the previous year;
  long the top 20%, short the bottom 20%.
* Enter at the close one trading day after the surprise is usable; hold 40
  trading days (20 and 60 shown as sensitivity). Equal weight within each
  leg, 10 bps per unit traded.
* Universe: current S&P 500 members, each from its index add date.

Example::

    python scripts/earnings_event_strategy.py --cache-dir data/research_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from earnings_feature_research import build_events, defeatbeta_table  # noqa: E402
from feature_research import BENCHMARK, load_ohlcv, sp500_constituents  # noqa: E402

from auto_researcher.backtest.event_strategy import (  # noqa: E402
    EventStrategyConfig,
    classify_events,
    return_stats,
    simulate_event_strategy,
)
from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.data.fmp_earnings import consensus_surprises, load_fmp_earnings  # noqa: E402
from auto_researcher.features.earnings_events import announcement_dates, attach_surprises  # noqa: E402

logger = logging.getLogger("earnings_event_strategy")
HOLDS = (20, 40, 60)
PRIMARY_HOLD = 40
MIN_VENDOR_SYMBOLS = 100
VENDOR_SOURCES = {"fmp_consensus": "fmp_earnings", "av_consensus": "av_earnings"}


def usable_dates(announce: pd.Series, cal: pd.DatetimeIndex, delay: int = 2) -> pd.Series:
    """Close at which a filing on day D becomes usable (D + ``delay`` trading days)."""
    ann = pd.to_datetime(announce).to_numpy()
    d = cal.searchsorted(ann)
    on_day = (d < len(cal)) & (cal[np.minimum(d, len(cal) - 1)] == ann)
    pos = np.where(on_day, d + delay, d + delay - 1)
    out = pd.Series(pd.NaT, index=announce.index, dtype="datetime64[ns]")
    ok = pos < len(cal)
    out[ok] = cal[pos[ok]]
    return out


def analyst_events(cache: Path) -> pd.DataFrame:
    """Consensus surprises from the local PEAD file, re-keyed to SEC announcement dates."""
    pead = pd.read_parquet(ROOT / "data" / "pead_backtest_results.parquet")
    surprises = pd.DataFrame({
        "symbol": pead["ticker"],
        "period_end": pd.to_datetime(pead["quarter_date"]),
        "sue": pd.to_numeric(pead["sue"], errors="coerce"),
    }).dropna()
    surprises["beat"] = (surprises["sue"] > 0).astype(float)
    filings = defeatbeta_table("stock_sec_filing", cache,
                               columns=["symbol", "form_type", "filing_date", "report_date"],
                               filters=[("form_type", "in", ["8-K", "10-Q", "10-K"])])
    return attach_surprises(announcement_dates(filings), surprises)


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "research_cache"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "earnings_event_strategy"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("auto_researcher", "numexpr", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    cache = Path(args.cache_dir)

    members = sp500_constituents()
    added = members.set_index("ticker")["date_added"]
    names = sorted(members["ticker"])
    close, _ = load_ohlcv(sorted(set(names) | {BENCHMARK}), cache)
    names = [t for t in names if t in close.columns]
    cal = close.index

    # Benchmarks: SPY and the equal-weight portfolio of index members.
    rets = close[names].pct_change(fill_method=None)
    member_mask = pd.DataFrame({t: cal >= (added.get(t) if pd.notna(added.get(t)) else cal[0])
                                for t in names}, index=cal)
    ew = rets.where(member_mask).mean(axis=1)
    spy = close[BENCHMARK].pct_change(fill_method=None)

    # Each source: events plus the price panel / equal-weight benchmark it trades in.
    sources, panels, overrides = {}, {}, {}
    for name, ev in (("ts_sue", build_events(cache)), ("analyst", analyst_events(cache))):
        ev = ev[ev["symbol"].isin(names)].dropna(subset=["sue"]).copy()
        ev["usable_date"] = usable_dates(ev["announce_date"], cal)
        ev = ev.dropna(subset=["usable_date"])
        start = ev["symbol"].map(added)
        ev = ev[start.isna() | (ev["usable_date"] >= start)]
        ev = ev[ev["usable_date"] >= "2014-06-01"]
        sources[name] = ev.rename(columns={"symbol": "ticker", "sue": "value"})[
            ["ticker", "usable_date", "value"]]
        panels[name] = (close, ew)
        logger.info("%s: %d events, %s to %s", name, len(ev), ev["usable_date"].min().date(),
                    ev["usable_date"].max().date())

    # Vendor consensus histories (scripts/download_fmp_earnings.py,
    # scripts/download_av_earnings.py): decades of history for the stocks
    # downloaded. Each gets its own price panel from 1994, its own equal-weight
    # benchmark, and a lower minimum history because the universe is smaller.
    # Below ~100 stocks each leg holds one or two names and the result is noise.
    for src, folder in VENDOR_SOURCES.items():
        vendor = load_fmp_earnings(cache / folder) if (cache / folder).exists() else None
        if vendor is None or vendor.empty:
            continue
        n_vendor = vendor["symbol"].nunique()
        if n_vendor < MIN_VENDOR_SYMBOLS:
            logger.info("%s skipped: %d stocks downloaded, need %d", src, n_vendor, MIN_VENDOR_SYMBOLS)
            continue
        v_names = sorted(set(vendor["symbol"]) & set(members["ticker"]))
        v_close, _ = load_ohlcv(v_names + [BENCHMARK], cache, start="1994-01-01")
        v_names = [t for t in v_names if t in v_close.columns]
        v_mask = pd.DataFrame({t: v_close.index >= (added.get(t) if pd.notna(added.get(t)) else v_close.index[0])
                               for t in v_names}, index=v_close.index)
        v_ew = v_close[v_names].pct_change(fill_method=None).where(v_mask).mean(axis=1)
        ev = consensus_surprises(vendor)
        ev = ev[ev["symbol"].isin(v_names)].copy()
        ev["usable_date"] = usable_dates(ev["announce_date"], v_close.index)
        ev = ev.dropna(subset=["usable_date"])
        start = ev["symbol"].map(added)
        ev = ev[start.isna() | (ev["usable_date"] >= start)]
        sources[src] = ev.rename(columns={"symbol": "ticker", "sue": "value"})[
            ["ticker", "usable_date", "value"]]
        panels[src] = (v_close, v_ew)
        overrides[src] = {"min_history": 60}
        logger.info("%s: %d events for %d stocks, %s to %s", src, len(ev), len(v_names),
                    ev["usable_date"].min().date(), ev["usable_date"].max().date())

    n_trials = len(sources) * len(HOLDS)
    results, yearly = {}, {}
    for src, ev in sources.items():
        src_close, src_ew = panels[src]
        for hold in HOLDS:
            cfg = EventStrategyConfig(hold_days=hold, **overrides.get(src, {}))
            classified = classify_events(ev, cfg)
            daily = simulate_event_strategy(classified, src_close, cfg)
            first = classified.loc[classified["side"] != 0, "usable_date"].min()
            daily = daily.loc[first:]
            key = f"{src}_{hold}d"
            ls = daily["long_short_net"]
            long_vs_ew = (daily["long_net"] - src_ew.reindex(daily.index)).where(daily["n_long"] > 0)
            short_vs_ew = (daily["short"] - src_ew.reindex(daily.index)).where(daily["n_short"] > 0)
            long_vs_spy = (daily["long_net"] - spy.reindex(daily.index)).where(daily["n_long"] > 0)
            results[key] = {
                "long_short_gross": return_stats(daily["long"] - daily["short"], n_trials),
                "long_short_net": return_stats(ls, n_trials),
                "long_vs_spy": return_stats(long_vs_spy, n_trials),
                "long_vs_equal_weight": return_stats(long_vs_ew, n_trials),
                "short_leg_vs_equal_weight": return_stats(short_vs_ew, n_trials),
                "avg_long_positions": float(daily["n_long"].mean()),
                "avg_short_positions": float(daily["n_short"].mean()),
                "start": str(daily.index.min().date()),
            }
            if hold == PRIMARY_HOLD:
                yearly[src] = ls.groupby(ls.index.year).sum()
            s = results[key]["long_short_net"]
            logger.info("%-14s L/S net %+.1f%%/yr  Sharpe %.2f  t %+.2f  maxDD %.1f%%", key,
                        100 * s["ann_return"], s["sharpe"], s["t_stat"], 100 * s["max_drawdown"])

    # Same-window comparison: ts_sue restricted to the analyst sample period.
    a_start = pd.Timestamp(results[f"analyst_{PRIMARY_HOLD}d"]["start"])
    cfg = EventStrategyConfig(hold_days=PRIMARY_HOLD)
    ts_recent = simulate_event_strategy(classify_events(sources["ts_sue"], cfg), close, cfg).loc[a_start:]
    results[f"ts_sue_{PRIMARY_HOLD}d_same_window"] = {
        "long_short_net": return_stats(ts_recent["long_short_net"], n_trials),
        "start": str(a_start.date()),
    }

    def row(label, s):
        if not np.isfinite(s.get("sharpe", np.nan)):
            return f"| {label} | n/a | n/a | n/a | n/a | n/a |"
        return (f"| {label} | {s['ann_return']:+.1%} | {s['ann_vol']:.1%} | {s['sharpe']:.2f} | "
                f"{s['t_stat']:+.2f} | {s['max_drawdown']:.1%} |")

    md = [
        "# Event-driven earnings drift strategy", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/earnings_event_strategy.py`.", "",
        "Rules fixed before running: rank each surprise against the previous year's "
        "announcements, long the top 20% and short the bottom 20%, enter one day after the "
        "surprise is usable (two trading days after the SEC filing), hold 40 trading days "
        "(20/60 as sensitivity), equal weight per leg, 10 bps per unit traded. Universe: "
        "current S&P 500 members from their index add dates. Deflated Sharpe probabilities "
        f"assume {n_trials} variants.", "",
        *([
            "`fmp_consensus` and `av_consensus` use vendor earnings histories (actual vs. "
            "consensus EPS from FMP and Alpha Vantage; validated in "
            "[fmp_earnings_validation.md](fmp_earnings_validation.md) and "
            "[av_earnings_validation.md](av_earnings_validation.md)), each with its own price "
            "panel from 1994 and equal-weight benchmark; a surprise needs 60 (not 200) earlier "
            "announcements in the past year to be ranked. Universe: current S&P 500 members, so "
            "early years carry survivorship bias.", "",
        ] if set(VENDOR_SOURCES) & set(sources) else []),
        "Costs are charged on daily changes in equal-weight targets, which re-weights every "
        "open position when one enters or exits; a buy-and-hold-per-position implementation "
        "would pay roughly half. Gross results are shown for that reason.", "",
        "## Long-short, gross of costs", "",
        "| Strategy | Return / yr | Vol | Sharpe | t | Max drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[row(f"`{k}` (from {r['start']})", r["long_short_gross"])
          for k, r in results.items() if "long_short_gross" in r],
        "",
        "## Long-short, net of costs", "",
        "| Strategy | Return / yr | Vol | Sharpe | t | Max drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, r in results.items():
        md.append(row(f"`{key}` (from {r['start']})", r["long_short_net"]))
    md += ["", f"## Legs, {PRIMARY_HOLD}-day hold (excess returns)", "",
           "| Strategy | Leg | Return / yr | Vol | Sharpe | t | Max drawdown |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for src in sources:
        r = results[f"{src}_{PRIMARY_HOLD}d"]
        for leg, label in (("long_vs_spy", "long vs SPY (net)"),
                           ("long_vs_equal_weight", "long vs equal-weight (net)"),
                           ("short_leg_vs_equal_weight", "shorted names vs equal-weight")):
            md.append(row(f"`{src}` | {label}", r[leg]).replace("| `", "| `", 1))
        md.append(f"| `{src}` | avg positions | {r['avg_long_positions']:.0f} long, "
                  f"{r['avg_short_positions']:.0f} short | | | | |")
    md += ["", f"## Long-short net return by year, {PRIMARY_HOLD}-day hold", "",
           "| Year | " + " | ".join(f"`{s}`" for s in yearly) + " |",
           "| --- | " + " | ".join("---:" for _ in yearly) + " |"]
    years = sorted(set().union(*[set(v.index) for v in yearly.values()]))
    for y in years:
        cells = [f"{yearly[s].get(y):+.1%}" if y in yearly[s].index else "" for s in yearly]
        md.append(f"| {y} | " + " | ".join(cells) + " |")
    md += ["", "Survivorship caveat: companies that left the S&P 500 are missing, which flatters "
           "the short leg (failed companies with bad surprises are absent)."]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        return None if isinstance(o, float) and not np.isfinite(o) else o

    out.with_suffix(".json").write_text(json.dumps(clean({
        "generated": datetime.now().isoformat(timespec="seconds"), "n_trials": n_trials,
        "results": results, "yearly": {k: v.to_dict() for k, v in yearly.items()}}), indent=2),
        encoding="utf-8")
    logger.info("Wrote %s.md/.json", out)


if __name__ == "__main__":
    main()
