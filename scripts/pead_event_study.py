"""
PEAD event study keyed on *announcement* dates.

Why this exists: ``data/pead_backtest_results.parquet`` (built by the old
``scripts/backtest_pead.py``) dates every earnings event at the *fiscal
quarter end*. Companies report ~30 calendar days later (median; 10th-90th
percentile 19-41 days), so a "40-day" forward return measured from the
quarter end contains the announcement-day price reaction. Correlating the
surprise with that return measures how the market reacts to news it has not
seen yet, not post-announcement drift. That is why the file's IC rises from
+0.01 at 1-5 days to +0.17 at 40 days, and why the README once reported a
PEAD IC of +0.22.

This script re-keys the same surprises to the actual report dates (from any
CSV with ``symbol``/``ticker`` and ``report_date`` columns, e.g.
``data/sentiment_500.csv``) and splits the forward return into:

* **reaction**: close before the announcement to the close of the next
  trading day (the day the news is priced; not tradeable on the surprise);
* **drift**: from that close forward h trading days (tradeable after the fact).

Example::

    python scripts/pead_event_study.py --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.backtest.metrics import compute_ic_stats  # noqa: E402

DRIFT_HORIZONS = (5, 20, 40, 60)


def load_events(surprises: Path, report_dates: Path, max_gap_days: int = 100) -> pd.DataFrame:
    """Attach the first report date after each fiscal quarter end."""
    ev = pd.read_parquet(surprises)
    ev["quarter_date"] = pd.to_datetime(ev["quarter_date"]).astype("datetime64[ns]")
    rep = pd.read_csv(report_dates)
    rep = rep.rename(columns={"symbol": "ticker"})[["ticker", "report_date"]]
    rep["report_date"] = pd.to_datetime(rep["report_date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        ev.sort_values("quarter_date"),
        rep.sort_values("report_date"),
        left_on="quarter_date",
        right_on="report_date",
        by="ticker",
        direction="forward",
        tolerance=pd.Timedelta(days=max_gap_days),
    )
    return merged.dropna(subset=["report_date", "sue"]).reset_index(drop=True)


def _excess_return(p: np.ndarray, b: np.ndarray, i: int, j: int) -> float:
    """Stock return minus benchmark return from row i to row j (NaN if unavailable)."""
    if j >= len(p) or not (np.isfinite(p[i]) and np.isfinite(p[j])):
        return np.nan
    return (p[j] / p[i] - 1) - (b[j] / b[i] - 1)


def event_returns(events: pd.DataFrame, prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """Market-adjusted reaction and drift returns for each event (NaN when prices are missing)."""
    cal = prices.index
    rows = []
    for e in events.itertuples(index=False):
        if e.ticker not in prices.columns:
            continue
        # First trading day on/after the report date; the reaction window runs
        # from the prior close through the next close so both pre-open and
        # after-close announcements are captured.
        t0 = cal.searchsorted(e.report_date)
        q0 = cal.searchsorted(e.quarter_date)
        if t0 < 1 or t0 + 1 >= len(cal):
            continue
        p, b = prices[e.ticker].to_numpy(), prices[benchmark].to_numpy()

        def excess(i: int, j: int, p: np.ndarray = p, b: np.ndarray = b) -> float:
            return _excess_return(p, b, i, j)

        row = {
            "ticker": e.ticker,
            "report_date": e.report_date,
            "quarter_date": e.quarter_date,
            "sue": e.sue,
            "gap_trading_days": int(t0 - q0),
            "reaction": excess(t0 - 1, t0 + 1),
            # The legacy measurement: 40 trading days from the fiscal quarter end.
            "legacy_ret40_from_quarter_end": excess(q0, q0 + 40),
        }
        for h in DRIFT_HORIZONS:
            row[f"drift_{h}d"] = excess(t0 + 1, t0 + 1 + h)
        rows.append(row)
    return pd.DataFrame(rows)


def ic_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled Spearman IC and the mean of per-quarter cross-sectional ICs, per return column."""
    df = df.copy()
    df["quarter"] = df["report_date"].dt.to_period("Q")
    out = {}
    cols = ["legacy_ret40_from_quarter_end", "reaction"] + [f"drift_{h}d" for h in DRIFT_HORIZONS]
    for col in cols:
        valid = df[["sue", col, "quarter"]].dropna()
        pooled = stats.spearmanr(valid["sue"], valid[col]).statistic if len(valid) > 10 else np.nan
        per_q = valid.groupby("quarter").apply(
            lambda g, col=col: stats.spearmanr(g["sue"], g[col]).statistic if len(g) >= 10 else np.nan,
            include_groups=False,
        ).dropna()
        s = compute_ic_stats(per_q.to_numpy(), horizon_days=1)
        top, bottom = valid[col][valid["sue"] >= 0.20], valid[col][valid["sue"] <= -0.20]
        beat, miss = valid[col][valid["sue"] >= 0.05], valid[col][valid["sue"] <= -0.05]
        out[col] = {
            "n_events": float(len(valid)),
            "pooled_ic": float(pooled),
            "quarterly_mean_ic": s["mean"],
            "quarterly_ic_t": s["t_stat_iid"],
            "n_quarters": s["n"],
            "big_beat_minus_big_miss": float(top.mean() - bottom.mean()) if len(top) and len(bottom) else np.nan,
            "beat_minus_miss_5pct": float(beat.mean() - miss.mean()) if len(beat) and len(miss) else np.nan,
        }
    return pd.DataFrame(out).T


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surprises", default=str(ROOT / "data" / "pead_backtest_results.parquet"))
    ap.add_argument("--report-dates", default=str(ROOT / "data" / "sentiment_500.csv"))
    ap.add_argument("--offline", action="store_true", help="Use the local price cache (required for now)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "pead_event_study"))
    args = ap.parse_args(argv)

    if not args.offline:
        raise SystemExit("Only --offline is implemented: prices come from data/price_cache.")
    from auto_researcher.data.price_loader import load_cached_price_panel

    events = load_events(Path(args.surprises), Path(args.report_dates))
    tickers = sorted(set(events["ticker"])) + ["SPY"]
    prices = load_cached_price_panel(tickers=tickers)
    rets = event_returns(events, prices)
    table = ic_table(rets)

    gap = rets["gap_trading_days"]
    print(f"{len(rets):,} events, {rets['ticker'].nunique()} tickers, "
          f"{rets['report_date'].min().date()} to {rets['report_date'].max().date()}")
    print(f"Trading days from fiscal quarter end to report: median {gap.median():.0f}, "
          f"10th-90th pct {gap.quantile(0.1):.0f}-{gap.quantile(0.9):.0f}")
    print(table.round(4).to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = {"n_events": "{:,.0f}", "pooled_ic": "{:+.3f}", "quarterly_mean_ic": "{:+.3f}",
           "quarterly_ic_t": "{:+.2f}", "n_quarters": "{:.0f}", "big_beat_minus_big_miss": "{:+.2%}",
           "beat_minus_miss_5pct": "{:+.2%}"}
    labels = {
        "legacy_ret40_from_quarter_end": "Legacy: 40d from fiscal quarter end (contains announcement)",
        "reaction": "Announcement reaction (close before -> next close)",
        **{f"drift_{h}d": f"Post-announcement drift, {h} trading days" for h in DRIFT_HORIZONS},
    }
    lines = [
        "# PEAD event study (announcement-dated)",
        "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/pead_event_study.py --offline`.",
        "",
        f"* **Events.** {len(rets):,} earnings events for {rets['ticker'].nunique()} tickers with cached "
        f"prices, {rets['report_date'].min().date()} to {rets['report_date'].max().date()}. Surprise = "
        "(actual - estimate) / |estimate| from `data/pead_backtest_results.parquet`; report dates from "
        f"`{Path(args.report_dates).name}`.",
        f"* **Timing.** Reports arrive a median {gap.median():.0f} trading days after the fiscal quarter "
        f"end (10th-90th percentile {gap.quantile(0.1):.0f}-{gap.quantile(0.9):.0f}). All returns are in "
        "excess of SPY.",
        "* **IC.** Pooled Spearman over all events, and the mean of per-quarter cross-sectional ICs "
        "(t-stat across quarters). The last two columns compare mean returns for surprises beyond "
        "+/-20% and beyond +/-5%.",
        "",
        "| Return window | Events | Pooled IC | Mean quarterly IC | t | Quarters | Beat - miss (20%) | Beat - miss (5%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in table.iterrows():
        cells = [fmt[c].format(row[c]) if np.isfinite(row[c]) else "n/a" for c in fmt]
        lines.append(f"| {labels[key]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Survivorship caveat: tickers are today's large caps, so firms that collapsed after bad "
        "surprises are missing, which biases the short side toward smaller losses.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                    "n_events": len(rets), "results": table.to_dict(orient="index")}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out}.md/.json")


if __name__ == "__main__":
    main()
