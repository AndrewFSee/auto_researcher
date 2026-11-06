"""
Analyst rating changes and price-target revisions as event signals.

Rules pre-registered in ``docs/results/analyst_ratings_protocol.md`` (approved
2026-10-31, before any post-event return was computed):

* Events: Yahoo rating actions on current S&P 500 members (each from its index
  add date), day 0 from 2013-01-01. Day 0 = the first session whose close comes
  after the action; enter at the close of day +1 (``entry_lag = 1``).
* ``rating``: long upgrades, short downgrades. ``target``: revision = new / old
  price target - 1; long raises in the top 20% of the previous year's raises,
  short cuts in the top 20% of the previous year's cuts (trailing percentiles).
* One position per stock, day 0 and side; a stock with both sides on one day 0
  is dropped. Hold 20 trading days (5 and 60 as sensitivity), equal weight per
  leg, 10 bps per unit traded.
* ``ex_earnings``: without actions whose day 0 is within 3 trading days of an
  Alpha Vantage earnings report.
* Development: day 0 to 2025-10-31; holdout: day 0 from 2025-11-01.
* Supported if the 20-day, unfiltered long-short net return has t >= 2.0 in
  development and a positive mean in the holdout.

Example::

    python scripts/analyst_ratings_strategy.py
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

from feature_research import BENCHMARK, load_ohlcv, sp500_constituents  # noqa: E402

from auto_researcher.backtest.event_strategy import (  # noqa: E402
    EventStrategyConfig,
    classify_events,
    return_stats,
    simulate_event_strategy,
)
from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.data.analyst_ratings import first_reacting_session, load_ratings  # noqa: E402
from auto_researcher.data.fmp_earnings import load_fmp_earnings  # noqa: E402

logger = logging.getLogger("analyst_ratings_strategy")
CACHE = ROOT / "data" / "research_cache"
START = pd.Timestamp("2013-01-01")
HOLDOUT = pd.Timestamp("2025-11-01")
HOLDS = (5, 20, 60)
PRIMARY_HOLD = 20
EARNINGS_WINDOW = 3
T_MIN = 2.0
FINDINGS = [
    "", "## Findings (first run, 2026-10-31; no deviations from the protocol)", "",
    "* Neither signal is supported. Rating changes: development t = −3.75 net, holdout −3.4% a "
    "year. Price targets: t = −0.80, holdout +1.6% a year.",
    "* Gross of costs, rating changes earn nothing after day +1 (−0.1% a year over 20 days, "
    "t = −0.04); the net loss is trading costs. The stock moves on day 0 (+1.6% for upgrades, "
    "−1.9% for downgrades vs SPY in the validation), and by the next close the news is priced. "
    "Excluding actions near earnings changes nothing (+0.8% gross, t = +0.37).",
    "* Price-target revisions: +3.6% a year gross over 20 days (t = +0.70), with 18% volatility "
    "and a −54% drawdown; noise, not a signal.",
    "* The long legs trail SPY (rating upgrades −2.7% a year net), and the shorted names do not "
    "trail it: there is no drift on either side. Rating changes lose money in 11 of 13 "
    "development years.",
    "* Costs: the engine charges for re-weighting every open position whenever one enters or "
    "exits, so net figures overstate costs (roughly double a buy-and-hold-per-position "
    "implementation); the conclusion rests on the gross figures, which are zero.",
]


def one_per_stock_day(ev: pd.DataFrame) -> pd.DataFrame:
    """One event per (ticker, day 0, side); drop stock-days with both sides."""
    ev = ev[ev["side"] != 0].drop_duplicates(["ticker", "usable_date", "side"])
    sides = ev.groupby(["ticker", "usable_date"])["side"].transform("nunique")
    return ev[sides == 1].sort_values("usable_date").reset_index(drop=True)


def rating_events(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df["action"].isin(["up", "down"])]
    out = pd.DataFrame({"ticker": ev["symbol"], "usable_date": ev["day0"],
                        "side": np.where(ev["action"] == "up", 1, -1)})
    return one_per_stock_day(out)


def target_events(df: pd.DataFrame, cfg: EventStrategyConfig) -> pd.DataFrame:
    """Top-20% raises (long) and top-20% cuts by size (short), each ranked among earlier ones."""
    ok = (df["price_target"] > 0) & (df["prior_price_target"] > 0)
    rev = df.loc[ok, "price_target"] / df.loc[ok, "prior_price_target"] - 1
    base = pd.DataFrame({"ticker": df.loc[ok, "symbol"], "usable_date": df.loc[ok, "day0"], "value": rev})
    legs = []
    for side, part in ((1, base[base["value"] > 0]), (-1, base[base["value"] < 0].assign(value=lambda d: -d["value"]))):
        c = classify_events(part, cfg)
        legs.append(c.assign(side=np.where(c["pct"] >= cfg.long_pct, side, 0))[["ticker", "usable_date", "side"]])
    return one_per_stock_day(pd.concat(legs, ignore_index=True))


def near_earnings(ev: pd.DataFrame, earnings: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    """True where day 0 is within EARNINGS_WINDOW sessions of an earnings report; NaN if no history."""
    e = earnings.dropna(subset=["date"])
    pos = cal.searchsorted(e["date"].to_numpy(dtype="datetime64[ns]"), side="left")
    nxt = cal.searchsorted(e["date"].to_numpy(dtype="datetime64[ns]"), side="right")
    sess = pd.Series(np.where(e["report_time"].eq("post-market").to_numpy(), nxt, pos), index=e.index)
    by_sym = {s: np.sort(sess[g.index].to_numpy()) for s, g in e.groupby("symbol")}
    d0 = cal.searchsorted(ev["usable_date"].to_numpy(dtype="datetime64[ns]"))
    out = np.full(len(ev), np.nan)
    for i, (sym, d) in enumerate(zip(ev["ticker"], d0)):
        arr = by_sym.get(sym)
        if arr is None or not len(arr):
            continue
        j = np.searchsorted(arr, d)
        gaps = [abs(arr[k] - d) for k in (j - 1, j) if 0 <= k < len(arr)]
        out[i] = float(min(gaps) <= EARNINGS_WINDOW)
    return pd.Series(out, index=ev.index)


def period_stats(ev: pd.DataFrame, close: pd.DataFrame, spy: pd.Series, cfg: EventStrategyConfig,
                 n_trials: int) -> tuple[dict, pd.Series]:
    daily = simulate_event_strategy(ev, close, cfg)
    active = daily.index[(daily["n_long"] > 0) | (daily["n_short"] > 0)]
    if not len(active):
        return {}, pd.Series(dtype=float)
    daily = daily.loc[active.min():active.max()]
    sp = spy.reindex(daily.index)
    ls = daily["long_short_net"]
    return {
        "long_short_net": return_stats(ls, n_trials),
        "long_short_gross": return_stats(daily["long"] - daily["short"], n_trials),
        "long_vs_spy": return_stats((daily["long_net"] - sp).where(daily["n_long"] > 0), n_trials),
        "short_leg_vs_spy": return_stats((daily["short"] - sp).where(daily["n_short"] > 0), n_trials),
        "events_long": int((ev["side"] == 1).sum()), "events_short": int((ev["side"] == -1).sum()),
        "avg_long_positions": float(daily["n_long"].mean()),
        "avg_short_positions": float(daily["n_short"].mean()),
        "start": str(daily.index.min().date()), "end": str(daily.index.max().date()),
    }, ls


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "analyst_ratings_strategy"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("auto_researcher", "numexpr", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    members = sp500_constituents()
    added = members.set_index("ticker")["date_added"]
    yahoo = load_ratings(CACHE / "analyst_ratings_yahoo")
    fmp = load_ratings(CACHE / "analyst_ratings_fmp")
    # Same ticker list as the validation step, so its cached price panel is reused
    close, _ = load_ohlcv(sorted(set(yahoo["symbol"]) | set(fmp["symbol"])) + [BENCHMARK], CACHE,
                          start="2012-01-01")
    cal = close.index
    spy = close[BENCHMARK].pct_change(fill_method=None)

    df = yahoo[yahoo["symbol"].isin(set(members["ticker"]) & set(close.columns))].copy()
    idx = first_reacting_session(df, cal)
    df = df[idx < len(cal)]
    df["day0"] = cal[idx[idx < len(cal)].to_numpy()]
    start = df["symbol"].map(added)
    df = df[(df["day0"] >= START) & (start.isna() | (df["day0"] >= start))]
    logger.info("%d actions on %d stocks, day 0 %s to %s", len(df), df["symbol"].nunique(),
                df["day0"].min().date(), df["day0"].max().date())

    earnings = load_fmp_earnings(CACHE / "av_earnings")
    signals = {"rating": rating_events(df), "target": target_events(df, EventStrategyConfig())}
    variants: dict[str, pd.DataFrame] = {}
    coverage = {}
    for name, ev in signals.items():
        near = near_earnings(ev, earnings, cal)
        coverage[name] = {"events": len(ev), "near_earnings": int((near == 1).sum()),
                          "no_earnings_history": int(near.isna().sum())}
        variants[name] = ev
        variants[f"{name}_ex_earnings"] = ev[near != 1].reset_index(drop=True)
    n_trials = len(variants) * len(HOLDS)

    results: dict[str, dict] = {}
    yearly: dict[str, pd.Series] = {}
    for key, ev in variants.items():
        for hold in HOLDS:
            cfg = EventStrategyConfig(hold_days=hold)
            for period, part in (("development", ev[ev["usable_date"] < HOLDOUT]),
                                 ("holdout", ev[ev["usable_date"] >= HOLDOUT])):
                stats, ls = period_stats(part, close, spy, cfg, n_trials)
                results[f"{key}_{hold}d_{period}"] = stats
                if hold == PRIMARY_HOLD and period == "development":
                    yearly[key] = ls.groupby(ls.index.year).sum()
            s = results[f"{key}_{hold}d_development"]["long_short_net"]
            logger.info("%-22s %2dd dev L/S net %+.1f%%/yr Sharpe %.2f t %+.2f", key, hold,
                        100 * s["ann_return"], s["sharpe"], s["t_stat"])

    verdict = {}
    for name in signals:
        dev = results[f"{name}_{PRIMARY_HOLD}d_development"]["long_short_net"]
        hold_ = results[f"{name}_{PRIMARY_HOLD}d_holdout"]["long_short_net"]
        verdict[name] = {"dev_t": dev["t_stat"], "holdout_ann_return": hold_["ann_return"],
                         "supported": bool(dev["t_stat"] >= T_MIN and hold_["ann_return"] > 0)}

    def row(label, s):
        if not s or not np.isfinite(s.get("sharpe", np.nan)):
            return f"| {label} | n/a | n/a | n/a | n/a | n/a |"
        return (f"| {label} | {s['ann_return']:+.1%} | {s['ann_vol']:.1%} | {s['sharpe']:.2f} | "
                f"{s['t_stat']:+.2f} | {s['max_drawdown']:.1%} |")

    md = [
        "# Analyst rating changes and price-target revisions", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/analyst_ratings_strategy.py`. Rules "
        "pre-registered in [analyst_ratings_protocol.md](analyst_ratings_protocol.md); data checks in "
        "[analyst_ratings_validation.md](analyst_ratings_validation.md). Deflated Sharpe "
        f"probabilities assume {n_trials} variants.", "",
        "## Verdict (20-day hold, no earnings filter)", "",
        "| Signal | Development t | Holdout return / yr | Supported (t ≥ 2 and holdout > 0) |",
        "| --- | ---: | ---: | --- |",
        *[f"| `{k}` | {v['dev_t']:+.2f} | {v['holdout_ann_return']:+.1%} | {'yes' if v['supported'] else 'no'} |"
          for k, v in verdict.items()],
        "",
        "## Events", "",
        "| Signal | Events (long / short positions opened) | Near earnings (±3 days) | No earnings history |",
        "| --- | ---: | ---: | ---: |",
        *[f"| `{k}` | {c['events']:,} | {c['near_earnings']:,} ({c['near_earnings'] / max(c['events'], 1):.0%}) "
          f"| {c['no_earnings_history']:,} |" for k, c in coverage.items()],
    ]
    for period in ("development", "holdout"):
        for measure, title in (("long_short_net", "long-short, net of costs"),
                               ("long_short_gross", "long-short, gross of costs")):
            md += ["", f"## {period.capitalize()}: {title}", "",
                   "| Variant | Hold | Return / yr | Vol | Sharpe | t | Max drawdown |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
            for key in variants:
                for hold in HOLDS:
                    r = results[f"{key}_{hold}d_{period}"]
                    md.append(row(f"`{key}` | {hold}d", r.get(measure, {})))
    md += ["", f"## Legs, {PRIMARY_HOLD}-day hold, development (excess over SPY)", "",
           "| Variant | Leg | Return / yr | Vol | Sharpe | t | Max drawdown |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in variants:
        r = results[f"{key}_{PRIMARY_HOLD}d_development"]
        md.append(row(f"`{key}` | long (net) − SPY", r["long_vs_spy"]))
        md.append(row(f"`{key}` | shorted names − SPY", r["short_leg_vs_spy"]))
        md.append(f"| `{key}` | avg positions | {r['avg_long_positions']:.0f} long, "
                  f"{r['avg_short_positions']:.0f} short | | | | |")
    md += ["", f"## Long-short net return by year, {PRIMARY_HOLD}-day hold, development", "",
           "| Year | " + " | ".join(f"`{s}`" for s in yearly) + " |",
           "| --- | " + " | ".join("---:" for _ in yearly) + " |"]
    for y in sorted(set().union(*[set(v.index) for v in yearly.values()])):
        md.append(f"| {y} | " + " | ".join(f"{yearly[s].get(y):+.1%}" if y in yearly[s].index else ""
                                           for s in yearly) + " |")
    md += FINDINGS + ["", "Survivorship caveat: companies that left the S&P 500 are missing, which flatters "
                      "the short leg (failed companies that were downgraded are absent)."]

    out = Path(args.out)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        return None if isinstance(o, float) and not np.isfinite(o) else o

    out.with_suffix(".json").write_text(json.dumps(clean({
        "generated": datetime.now().isoformat(timespec="seconds"), "n_trials": n_trials,
        "verdict": verdict, "coverage": coverage, "results": results,
        "yearly": {k: v.to_dict() for k, v in yearly.items()}}), indent=2), encoding="utf-8")
    logger.info("Wrote %s.md/.json", out)


if __name__ == "__main__":
    main()
