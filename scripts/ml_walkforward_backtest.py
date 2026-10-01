"""
Purged walk-forward evaluation of the ML ranking models and simple baselines.

Replaces the original script of the same name, which trained on every date
before each test date. With 21-day forward-return labels that leaked ~20 days
of the test label into training and produced the IC of +0.145 (t = 7.6) once
quoted in the README. All evaluation now goes through
``auto_researcher.backtest.walk_forward``.

Examples::

    # Offline, on the local price cache (103 large caps, 2015-2026)
    python scripts/ml_walkforward_backtest.py --offline

    # Measure how much each timing fix changes the ML model's results
    python scripts/ml_walkforward_backtest.py --offline --leak-ablation

    # Does the training-window choice survive purging?
    python scripts/ml_walkforward_backtest.py --offline --window-sweep 126,252,504

    # Live data (needs network): current S&P 100, 10 years
    python scripts/ml_walkforward_backtest.py --universe sp100 --lookback-years 10

Outputs ``<out>.json`` and ``<out>.md`` (default ``docs/results/walkforward``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.backtest.baselines import FeatureScoreModel, LinearICModel  # noqa: E402
from auto_researcher.backtest.walk_forward import (  # noqa: E402
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)
from auto_researcher.models.xgb_ranking_model import (  # noqa: E402
    XGBRegressionConfig,
    XGBRegressionModel,
)
from auto_researcher.screening import (  # noqa: E402
    UNIVERSES,
    ScreeningModel,
    build_feature_panel,
    fetch_prices,
)

logger = logging.getLogger("walkforward")
BENCHMARK = "SPY"


# =============================================================================
# Models
# =============================================================================


def _xgb() -> XGBRegressionModel:
    return XGBRegressionModel(
        XGBRegressionConfig(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            early_stopping_rounds=None,
        )
    )


@dataclass(frozen=True)
class ModelSpec:
    factory: Callable[[], object]
    target: str
    description: str
    train_window: int | None = None  # overrides --train-window when set


MODELS = {
    "xgb": ModelSpec(_xgb, "rank", "XGBoost regression on the per-date rank of the forward return"),
    "screening": ModelSpec(
        ScreeningModel, "vol_norm",
        "Live screening recipe (auto_researcher.screening): feature pruning, recency "
        "weights, pseudo-Huber XGBoost, 126-day window", train_window=126,
    ),
    "linear_ic": ModelSpec(LinearICModel, "rank", "Features weighted by their training-window IC (no tuning)"),
    "momentum": ModelSpec(lambda: FeatureScoreModel("tech_resid_mom_252"), "rank",
                          "12-month residual momentum, no model"),
    "reversal": ModelSpec(lambda: FeatureScoreModel("tech_mom_5d", sign=-1), "rank",
                          "5-day reversal, no model"),
    "low_vol": ModelSpec(lambda: FeatureScoreModel("tech_vol_idio_63", sign=-1), "rank",
                         "Low idiosyncratic volatility, no model"),
}


# =============================================================================
# Data
# =============================================================================


def load_prices(args: argparse.Namespace) -> pd.DataFrame:
    """Wide price panel with the benchmark, restricted to complete histories."""
    if args.offline:
        from auto_researcher.data.price_loader import load_cached_price_panel

        tickers = None
        if args.universe != "cache":
            tickers = [*UNIVERSES[args.universe](), BENCHMARK]
        prices = load_cached_price_panel(tickers=tickers)
        prices = prices.loc[args.start : args.end]
        complete = prices.notna().mean() >= 0.99
        prices = prices.loc[:, complete]
        prices = prices.loc[prices.notna().all(axis=1)]
    else:
        tickers = UNIVERSES[args.universe]()
        prices = fetch_prices(tickers, lookback_days=int(args.lookback_years * 365.25), benchmark=BENCHMARK)
    if BENCHMARK not in prices.columns:
        raise SystemExit(f"{BENCHMARK} missing from price data")
    return prices


# =============================================================================
# Runs
# =============================================================================


def base_config(args: argparse.Namespace, **overrides) -> WalkForwardConfig:
    params = dict(
        horizon=args.horizon,
        rebalance_every=args.rebalance,
        execution_lag=args.lag,
        train_window=args.train_window,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        n_random_paths=args.random_paths,
        n_trials=args.n_trials,
    )
    params.update(overrides)
    return WalkForwardConfig(**params)


def run_one(name, model_key, features, prices, cfg, respect_model_window=True) -> WalkForwardResult:
    spec = MODELS[model_key]
    cfg.target = spec.target
    if respect_model_window and spec.train_window is not None:
        cfg.train_window = spec.train_window
    start = time.time()
    result = run_walk_forward(features, prices, spec.factory, cfg, benchmark=BENCHMARK, name=name)
    s = result.summary()
    logger.info(
        "%-28s IC %+.4f (NW t %+.2f)  net Sharpe %.2f  EW Sharpe %.2f  pctl vs random %.0f%%  [%.0fs]",
        name, s["ic_mean"], s["ic_t_nw"], s.get("net_sharpe", np.nan),
        s.get("equal_weight_sharpe", np.nan),
        100 * s.get("gross_sharpe_percentile_vs_random", np.nan), time.time() - start,
    )
    return result


# =============================================================================
# Reporting
# =============================================================================

ROWS = [
    ("n_periods", "Rebalance periods", "{:.0f}"),
    ("ic_mean", "Mean IC (Spearman)", "{:+.4f}"),
    ("ic_t_nw", "IC t-stat (Newey-West)", "{:+.2f}"),
    ("ic_hit_rate", "Periods with IC > 0", "{:.0%}"),
    ("spread_mean", "Top-bottom quintile fwd return / period", "{:+.2%}"),
    ("gross_cagr", "Top-k CAGR (gross)", "{:+.1%}"),
    ("net_cagr", "Top-k CAGR (net of costs)", "{:+.1%}"),
    ("net_sharpe", "Top-k Sharpe (net)", "{:.2f}"),
    ("net_max_drawdown", "Top-k max drawdown (net)", "{:.1%}"),
    ("avg_turnover", "Avg one-way turnover / rebalance", "{:.0%}"),
    ("equal_weight_sharpe", "Equal-weight universe Sharpe", "{:.2f}"),
    ("benchmark_sharpe", "SPY Sharpe", "{:.2f}"),
    ("net_active_vs_equal_weight_ann", "Net active return vs equal-weight / yr", "{:+.2%}"),
    ("net_ir_vs_equal_weight", "Info ratio vs equal-weight", "{:+.2f}"),
    ("net_ir_vs_equal_weight_deflated_prob", "P(true IR vs EW > 0), deflated", "{:.2f}"),
    ("gross_sharpe_percentile_vs_random", "Sharpe percentile vs random top-k", "{:.0%}"),
]


def _fmt(value, pattern: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return pattern.format(value)


def summary_table(results: list[WalkForwardResult]) -> str:
    header = "| Metric | " + " | ".join(f"`{r.name}`" for r in results) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in results) + " |"
    lines = [header, sep]
    summaries = [r.summary() for r in results]
    for key, label, pattern in ROWS:
        lines.append(f"| {label} | " + " | ".join(_fmt(s.get(key), pattern) for s in summaries) + " |")
    return "\n".join(lines)


def write_report(path: Path, args, prices, sections: dict[str, list[WalkForwardResult]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = base_config(args)
    tickers = [c for c in prices.columns if c != BENCHMARK]
    md = [
        "# Purged walk-forward evaluation",
        "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/ml_walkforward_backtest.py "
        + " ".join(sys.argv[1:]) + "`.",
        "",
        f"* **Universe.** {len(tickers)} stocks ({'local price cache' if args.offline else args.universe}), "
        f"{prices.index.min().date()} to {prices.index.max().date()}. The list is today's "
        "constituents, so every backtest here is survivorship-biased: judge models "
        "against the equal-weight portfolio of the same names, not only against SPY.",
        f"* **Protocol.** Rebalance every {cfg.rebalance_every} trading days; signal at the close "
        f"of t, trade at the close of t+{cfg.execution_lag}; {cfg.horizon}-day labels; training "
        f"rows purged so no label overlaps the test date; rolling {cfg.train_window}-day training "
        f"window. Long-only top-{cfg.top_k} equal weight, {cfg.cost_bps:g} bps per unit traded.",
        f"* **Statistics.** IC t-stats use Newey-West with lag "
        f"ceil(horizon / rebalance) - 1. The deflated probability assumes {cfg.n_trials} "
        "configuration(s) were tried. The random-selection percentile compares the gross "
        f"Sharpe with {cfg.n_random_paths} random top-{cfg.top_k} portfolios drawn from the same "
        "cross-sections.",
        "",
    ]
    for title, results in sections.items():
        md += [f"## {title}", "", summary_table(results), ""]
        if title == "Models and baselines":
            md += ["Model descriptions:", ""]
            md += [f"* `{r.name}`: {MODELS[r.name].description}" for r in results if r.name in MODELS]
            md += [""]

    main = sections.get("Models and baselines", [])
    if main:
        md += ["## Mean IC by year", "", "| Year | " + " | ".join(f"`{r.name}`" for r in main) + " |",
               "| --- | " + " | ".join("---:" for _ in main) + " |"]
        by_year = {r.name: r.ic_by_year()["mean_ic"] for r in main}
        years = sorted(set().union(*[set(s.index) for s in by_year.values()]))
        for y in years:
            md.append(f"| {y} | " + " | ".join(_fmt(by_year[r.name].get(y), "{:+.3f}") for r in main) + " |")
        md.append("")

    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "universe_size": len(tickers),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "sections": {
            title: {r.name: {k: (None if not np.isfinite(v) else v) for k, v in r.summary().items()}
                    for r in results}
            for title, results in sections.items()
        },
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s(.md/.json)", path)


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offline", action="store_true", help="Use the local price cache instead of yfinance")
    p.add_argument("--universe", default="cache", help="'cache' (offline only) or a key of screening.UNIVERSES")
    p.add_argument("--start", default="2015-01-01", help="First price date (offline)")
    p.add_argument("--end", default="2026-01-15", help="Last price date (offline)")
    p.add_argument("--lookback-years", type=float, default=10.0, help="History to download (online)")
    p.add_argument("--models", default="xgb,screening,linear_ic,momentum,reversal,low_vol")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--rebalance", type=int, default=21)
    p.add_argument("--lag", type=int, default=1, help="Execution lag in trading days")
    p.add_argument("--train-window", type=int, default=504)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--random-paths", type=int, default=500)
    p.add_argument("--n-trials", type=int, default=None,
                   help="Configurations tried, for the deflated Sharpe (default: number of models run)")
    p.add_argument("--leak-ablation", action="store_true",
                   help="Also run the first model under the legacy leaky split")
    p.add_argument("--window-sweep", default="",
                   help="Comma-separated training windows to compare, leaky vs purged")
    p.add_argument("--out", default=str(ROOT / "docs" / "results" / "walkforward"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("auto_researcher", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not args.offline and args.universe == "cache":
        args.universe = "sp100"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {list(MODELS)}")
    if args.n_trials is None:
        args.n_trials = len(models)

    prices = load_prices(args)
    logger.info("Prices: %d tickers x %d days (%s to %s)", prices.shape[1] - 1, len(prices),
                prices.index.min().date(), prices.index.max().date())
    t = time.time()
    features = build_feature_panel(prices, benchmark=BENCHMARK)
    logger.info("Features: %s rows x %d columns [%.0fs]", f"{len(features):,}", features.shape[1], time.time() - t)

    sections: dict[str, list[WalkForwardResult]] = {}
    sections["Models and baselines"] = [
        run_one(m, m, features, prices, base_config(args)) for m in models
    ]

    if args.leak_ablation:
        m = models[0]
        sections[f"Leak ablation ({m})"] = [
            run_one("legacy_leaky", m, features, prices, base_config(args, purge=False, execution_lag=0)),
            run_one("purged_same_close", m, features, prices, base_config(args, execution_lag=0)),
            run_one("purged_next_close", m, features, prices, base_config(args)),
        ]

    if args.window_sweep:
        m = models[0]
        runs = []
        for w in [int(x) for x in args.window_sweep.split(",")]:
            runs.append(run_one(f"w{w}_leaky", m, features, prices,
                                base_config(args, train_window=w, purge=False, execution_lag=0),
                                respect_model_window=False))
            runs.append(run_one(f"w{w}_purged", m, features, prices, base_config(args, train_window=w),
                                respect_model_window=False))
        sections[f"Training-window sweep ({m})"] = runs

    for title, results in sections.items():
        print(f"\n## {title}\n")
        print(summary_table(results))
    write_report(Path(args.out), args, prices, sections)


if __name__ == "__main__":
    main()
