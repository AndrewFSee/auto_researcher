"""
Audit of the news-sentiment agent's evidence.

The agent's composite weight rested on an IC of +0.020 from
``scripts/backtest_news_combined.py``. That backtest (a) bucketed articles by
calendar date and started forward returns at that day's close, so news
published after the close was credited to a return window containing the
market's reaction; (b) chose its signal weights on the full sample, including
the test period; (c) split train/test without a purge gap.

This script re-measures the signal the agent actually computes (mean FinBERT
score over recent articles) point-in-time (``features.news_signals``): an
article dated day d is used from the close of the next trading day, and
positions are entered one day later. Four pre-specified signals are tested at
5-day (weekly rebalance) and 21-day (monthly) horizons on the S&P 500 names
covered by ``data/news.db`` with the purged walk-forward harness, followed by
an ablation that restores the legacy same-day alignment.

The LLM component of the agent cannot be backtested honestly (the model has
read about these periods) and is out of scope.

Example::

    python scripts/sentiment_audit.py --cache-dir data/research_cache
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

from auto_researcher.backtest.baselines import FeatureScoreModel  # noqa: E402
from auto_researcher.backtest.walk_forward import WalkForwardConfig, run_walk_forward  # noqa: E402
from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.features.news_signals import (  # noqa: E402
    NEWS_SIGNALS,
    load_articles,
    news_signal_panel,
)

logger = logging.getLogger("sentiment_audit")

HORIZONS = {"weekly (5d)": (5, 5), "monthly (21d)": (21, 21)}
ROWS = [
    ("n_periods", "Periods", "{:.0f}"),
    ("ic_mean", "Mean IC", "{:+.4f}"),
    ("ic_t_nw", "IC t (NW)", "{:+.2f}"),
    ("ic_hit_rate", "IC > 0", "{:.0%}"),
    ("spread_mean", "Top-bottom quintile / period", "{:+.2%}"),
    ("net_ir_vs_equal_weight", "Top-50 IR vs EW (net)", "{:+.2f}"),
    ("net_ir_vs_equal_weight_deflated_prob", "P(IR > 0), deflated", "{:.2f}"),
    ("avg_turnover", "Turnover / rebalance", "{:.0%}"),
]


def evaluate(panel, prices, signal, horizon, step, n_trials, lag=1, name=None):
    cfg = WalkForwardConfig(horizon=horizon, rebalance_every=step, execution_lag=lag,
                            train_window=None, min_train_dates=21, top_k=50, cost_bps=10.0,
                            n_trials=n_trials, n_random_paths=0, min_names=30)
    feats = panel[[signal]].dropna()
    return run_walk_forward(feats, prices, lambda: FeatureScoreModel(signal), cfg,
                            benchmark=BENCHMARK, name=name or signal).summary()


def table(results: dict[str, dict]) -> str:
    names = list(results)
    lines = ["| Metric | " + " | ".join(f"`{n}`" for n in names) + " |",
             "| --- | " + " | ".join("---:" for _ in names) + " |"]
    for key, label, fmt in ROWS:
        cells = []
        for n in names:
            v = results[n].get(key)
            cells.append("n/a" if v is None or not np.isfinite(v) else fmt.format(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "research_cache"))
    ap.add_argument("--db", default=str(ROOT / "data" / "news.db"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "sentiment_audit"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("auto_researcher", "numexpr", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    members = sp500_constituents()
    articles = load_articles(args.db, start="2023-10-01")
    tickers = sorted(set(members["ticker"]) & set(articles["ticker"]))
    close, _ = load_ohlcv(sorted(set(members["ticker"]) | {BENCHMARK}), Path(args.cache_dir))
    tickers = [t for t in tickers if t in close.columns]
    prices = close.loc["2023-10-01":, tickers + [BENCHMARK]]
    articles = articles[articles["ticker"].isin(tickers)]
    logger.info("%d scored articles for %d S&P 500 tickers, %s to %s", len(articles), len(tickers),
                articles["published"].min().date(), articles["published"].max().date())

    panel = news_signal_panel(articles, prices.index)
    panel = panel[panel.index.get_level_values("date") >= "2024-01-15"]
    coverage = panel.groupby(level="date")["sent_30d"].count()
    logger.info("Median names with 30-day sentiment per day: %.0f", coverage.median())

    n_trials = len(NEWS_SIGNALS) * len(HORIZONS)
    results: dict[str, dict[str, dict]] = {}
    for label, (h, step) in HORIZONS.items():
        results[label] = {}
        for sig in NEWS_SIGNALS:
            s = evaluate(panel, prices, sig, h, step, n_trials)
            results[label][sig] = s
            logger.info("%-14s %-12s IC %+.4f (t %+.2f, n=%d)", label, sig, s["ic_mean"],
                        s["ic_t_nw"], s["n_periods"])

    # Ablation: the legacy same-day alignment, no execution lag.
    legacy_panel = news_signal_panel(articles, prices.index, usable_same_day=True)
    legacy_panel = legacy_panel[legacy_panel.index.get_level_values("date") >= "2024-01-15"]
    ablation = {}
    for label, (h, step) in HORIZONS.items():
        ablation[f"legacy {label}"] = evaluate(legacy_panel, prices, "sent_7d", h, step, n_trials,
                                               lag=0, name="legacy")
        ablation[f"point-in-time {label}"] = results[label]["sent_7d"]
        logger.info("legacy %-14s sent_7d IC %+.4f (t %+.2f)", label,
                    ablation[f"legacy {label}"]["ic_mean"], ablation[f"legacy {label}"]["ic_t_nw"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Sentiment agent audit", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/sentiment_audit.py`.", "",
        f"* **Data.** {len(articles):,} FinBERT-scored articles for {len(tickers)} S&P 500 tickers "
        f"from `data/news.db`; coverage is dense only from 2024 (median {coverage.median():.0f} names "
        "per day with a 30-day score). Test dates run from February 2024.",
        "* **Timing.** An article dated day d is used from the close of the next trading day; "
        "positions are entered one trading day later. Long-only top-50 portfolios, 10 bps costs.",
        f"* **Multiple testing.** {n_trials} pre-specified signal/horizon pairs; deflated "
        "probabilities account for all of them.", "",
        "## Previous evidence", "",
        "The composite weight used IC +0.020 over 90 periods from `scripts/backtest_news_combined.py`, "
        "which aligned articles by calendar date with returns starting at that day's close "
        "(after-close news leaks into the window), chose its signal weights on the full sample "
        "including the test period, and used an 80/20 split without a purge gap.", "",
    ]
    for label in HORIZONS:
        md += [f"## Point-in-time results, {label}", "", table(results[label]), ""]
    md += [
        "## Timing ablation (`sent_7d`)", "",
        "Legacy alignment: articles usable at the close of their own calendar day and traded at "
        "that close.", "", table(ablation), "",
    ]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        return None if isinstance(o, float) and not np.isfinite(o) else o

    out.with_suffix(".json").write_text(json.dumps(clean({
        "generated": datetime.now().isoformat(timespec="seconds"), "n_trials": n_trials,
        "results": results, "ablation": ablation}), indent=2), encoding="utf-8")
    logger.info("Wrote %s.md/.json", out)


if __name__ == "__main__":
    main()
