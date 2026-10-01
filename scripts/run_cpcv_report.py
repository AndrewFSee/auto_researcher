"""
Combinatorial Purged CV (CPCV) report.

Runs C(N, K) = 15 purged-and-embargoed train/test splits over the feature
matrix, fits the XGBoost ranker on each train fold, scores the test fold,
and aggregates per-split Information Coefficients into a distribution.

Why this exists
---------------
A single walk-forward path gives you one realization — a headline IC of +0.15
could be a lucky sequence. CPCV emits many test paths, which lets us compute:

* Mean IC and a Newey-West t-stat robust to overlapping-horizon autocorr.
* A *deflated* Sharpe that accounts for the fact we tried hundreds of Optuna
  configurations on this same data.
* An IC violin plot — the shape tells us whether the edge is broad or
  concentrated in a few lucky folds.

Usage::

    python scripts/run_cpcv_report.py --universe sp100 --n-splits 6 \
        --n-test 2 --horizon 21 --embargo-pct 0.01

Outputs land in ``results/cpcv/`` (JSON summary + violin plot).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import yfinance as yf

from auto_researcher.features.enhanced import (
    EnhancedFeatureConfig,
    compute_all_enhanced_features,
)
from auto_researcher.features.targets import TargetConfig, build_targets
from auto_researcher.models.xgb_ranking_model import (
    XGBRegressionConfig,
    XGBRegressionModel,
)
from auto_researcher.backtest.metrics import compute_ic_stats
from auto_researcher.validation.cpcv import (
    combinatorial_purged_splits,
    n_cpcv_paths,
)
from auto_researcher.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
)
from auto_researcher.screening import UNIVERSES


def _newey_west_mean_tstat(values: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean / NW-stderr with a user-supplied lag (Bartlett kernel)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mean = float(x.mean())
    centered = x - mean
    gamma0 = float((centered**2).sum() / n)
    var = gamma0
    lag = max(0, int(lag))
    for k in range(1, min(lag, n - 1) + 1):
        gamma_k = float((centered[k:] * centered[:-k]).sum() / n)
        weight = 1.0 - k / (lag + 1.0)
        var += 2.0 * weight * gamma_k
    var = max(var, 1e-12)
    stderr = (var / n) ** 0.5
    return mean, mean / stderr


def _rank_ic(preds: np.ndarray, actual: np.ndarray) -> float:
    """Per-date Spearman rank correlation."""
    p = pd.Series(preds).rank()
    a = pd.Series(actual).rank()
    if p.std() == 0 or a.std() == 0:
        return float("nan")
    return float(p.corr(a))


def run_cpcv_report(
    universe: str = "sp100",
    lookback_years: int = 5,
    horizon_days: int = 21,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
    n_trials_assumed: int = 50,
    out_dir: Path = REPO_ROOT / "results" / "cpcv",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = UNIVERSES[universe]()
    end = datetime.now()
    start = end - timedelta(days=lookback_years * 365)
    print(f"[CPCV] Fetching {len(tickers)} tickers + SPY from {start.date()} to {end.date()}")

    prices = yf.download(
        tickers + ["SPY"], start=start, end=end, progress=False
    )["Close"]
    prices = prices.dropna(axis=1, how="all")
    print(f"[CPCV] {len(prices)} days, {len(prices.columns)} stocks")

    feature_config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,
        use_cross_sec_norm=True,
    )
    print("[CPCV] Building features")
    features = compute_all_enhanced_features(prices, benchmark="SPY", config=feature_config)
    features_long = features.stack(level=0, future_stack=True).dropna()
    features_long.index.names = ["date", "ticker"]
    for col in features_long.columns:
        features_long[col] = features_long[col].clip(-3, 3)

    print("[CPCV] Building targets")
    target_config = TargetConfig(mode="vol_norm", horizon_days=horizon_days)
    targets = build_targets(prices, target_config, benchmark="SPY")
    targets_long = targets.stack()
    targets_long.index.names = ["date", "ticker"]

    aligned = features_long.join(targets_long.rename("target"), how="inner").dropna()
    print(f"[CPCV] Aligned: {len(aligned)} samples")

    dates = aligned.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    if len(unique_dates) < n_splits * 5:
        raise ValueError(
            f"Too few unique dates ({len(unique_dates)}) for n_splits={n_splits}"
        )

    # Indices on the sorted unique date axis — CPCV groups are contiguous
    # chunks of unique dates, and each row's group membership is determined
    # by its date's position.
    date_to_pos = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    row_date_pos = date_to_pos.reindex(dates).to_numpy()

    n_paths_expected = n_cpcv_paths(n_splits, n_test_splits)
    print(
        f"[CPCV] n_splits={n_splits}, n_test_splits={n_test_splits} "
        f"-> {n_paths_expected} distinct paths / "
        f"C({n_splits},{n_test_splits}) = {math.comb(n_splits, n_test_splits)} train/test combos"
    )

    results: list[dict] = []
    for split in combinatorial_purged_splits(
        dates=unique_dates,
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        horizon_days=horizon_days,
        embargo_pct=embargo_pct,
    ):
        train_date_set = set(split.train_idx.tolist())
        test_date_set = set(split.test_idx.tolist())

        train_mask = np.isin(row_date_pos, list(train_date_set))
        test_mask = np.isin(row_date_pos, list(test_date_set))

        if train_mask.sum() < 1000 or test_mask.sum() < 50:
            continue

        train = aligned[train_mask]
        test = aligned[test_mask]

        X_train = train.drop(columns=["target"])
        y_train = train["target"]
        X_test = test.drop(columns=["target"])
        y_test = test["target"]

        model = XGBRegressionModel(
            XGBRegressionConfig(
                objective="reg:pseudohubererror",
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
            )
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        pred_series = pd.Series(preds, index=X_test.index)

        # Per-date IC across the test fold, then average — gives one IC per
        # date so the split's mean reflects cross-sectional skill.
        per_date_ic = []
        for d, grp in pred_series.groupby(level="date"):
            actuals = y_test.loc[grp.index]
            ic = _rank_ic(grp.to_numpy(), actuals.to_numpy())
            if np.isfinite(ic):
                per_date_ic.append(ic)

        if not per_date_ic:
            continue

        split_ic_mean = float(np.mean(per_date_ic))
        split_ic_ts, split_ic_tstat = _newey_west_mean_tstat(
            np.asarray(per_date_ic), lag=max(horizon_days - 1, 1)
        )

        results.append(
            {
                "test_groups": list(split.test_group_ids),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_test_dates": len(per_date_ic),
                "mean_ic": split_ic_mean,
                "ic_tstat_newey_west": split_ic_tstat,
            }
        )
        print(
            f"  combo {split.test_group_ids}: IC={split_ic_mean:+.4f} "
            f"t={split_ic_tstat:+.2f} (n_dates={len(per_date_ic)})"
        )

    if not results:
        raise RuntimeError("No CPCV splits produced usable results")

    ic_values = np.array([r["mean_ic"] for r in results])
    mean_ic = float(ic_values.mean())
    std_ic = float(ic_values.std(ddof=1)) if len(ic_values) > 1 else float("nan")

    # Cross-split IC statistics with both IID and Newey-West adjustment. Splits
    # share training rows under CPCV, so IC draws across splits are positively
    # correlated — the NW lag on the horizon-in-split-steps is a small
    # correction that at least reports the right sign/magnitude.
    ic_stats_across_splits = compute_ic_stats(ic_values, horizon_days=1)

    # Approximate Sharpe from the IC distribution: per-split IC is the mean
    # of many date-level IC draws; scale by sqrt(avg n_test_dates) to get
    # roughly a per-period signal-to-noise ratio. This is heuristic — DSR
    # should be interpreted qualitatively (pass/fail selection-bias check)
    # rather than as a precise Sharpe estimate.
    avg_n_test_dates = float(np.mean([r["n_test_dates"] for r in results]))
    sharpe_proxy = mean_ic / (std_ic / np.sqrt(len(results))) if std_ic > 0 else float("nan")

    dsr = deflated_sharpe_ratio(
        sharpe=mean_ic,
        n_obs=int(avg_n_test_dates * len(results)),
        n_trials=n_trials_assumed,
        sharpe_std=std_ic if std_ic and std_ic > 0 else 1.0,
    )
    expected_max = expected_max_sharpe_under_null(
        n_trials_assumed,
        sharpe_std=std_ic if std_ic and std_ic > 0 else 1.0,
    )

    summary = {
        "universe": universe,
        "lookback_years": lookback_years,
        "horizon_days": horizon_days,
        "n_splits": n_splits,
        "n_test_splits": n_test_splits,
        "embargo_pct": embargo_pct,
        "n_trials_assumed": n_trials_assumed,
        "n_combos": len(results),
        "n_paths": n_paths_expected,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "min_ic": float(ic_values.min()),
        "max_ic": float(ic_values.max()),
        "pct_ic_positive": float((ic_values > 0).mean()),
        "ic_tstat_across_splits": sharpe_proxy,
        "ic_stats_across_splits": ic_stats_across_splits,
        "expected_max_ic_under_null": expected_max,
        "deflated_sharpe_prob_positive": dsr,
        "per_split": results,
    }

    summary_path = out_dir / f"cpcv_{universe}_{n_splits}_{n_test_splits}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[CPCV] Summary written to {summary_path}")

    plot_path: Path | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.violinplot(ic_values, showmeans=True, showmedians=True)
        ax.set_ylabel("Per-split mean IC")
        ax.set_title(
            f"CPCV IC distribution — {universe}, "
            f"{len(results)} combos, mean={mean_ic:+.4f}, DSR P(>0)={dsr:.2f}"
        )
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axhline(expected_max, color="r", linewidth=0.5, linestyle="--",
                   label=f"E[max IC | null, {n_trials_assumed} trials]")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        plot_path = out_dir / f"cpcv_{universe}_{n_splits}_{n_test_splits}.png"
        fig.savefig(plot_path, dpi=120)
        # Also write a canonical-name copy so downstream docs don't bake in
        # the universe/N/K permutation.
        canonical_plot = REPO_ROOT / "results" / "cpcv_distributions.png"
        canonical_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(canonical_plot, dpi=120)
        plt.close(fig)
        print(f"[CPCV] Violin plot written to {plot_path} (+ {canonical_plot})")
    except ImportError:
        print("[CPCV] matplotlib not available — skipped violin plot")

    md_path = REPO_ROOT / "results" / "cpcv_report.md"
    _write_markdown_report(
        md_path, summary, ic_stats_across_splits, plot_path
    )
    print(f"[CPCV] Markdown report written to {md_path}")

    print("\n=== CPCV SUMMARY ===")
    print(f"  Mean IC:        {mean_ic:+.4f}")
    print(f"  Std IC:         {std_ic:.4f}")
    print(f"  % splits IC>0:  {summary['pct_ic_positive']:.1%}")
    print(f"  E[max | null]:  {expected_max:+.4f}  (n_trials={n_trials_assumed})")
    print(f"  Deflated P(>0): {dsr:.3f}")

    return summary


def _write_markdown_report(
    path: Path,
    summary: dict,
    ic_stats: dict,
    plot_path: Path | None,
) -> None:
    """Render a compact Markdown rollup of a CPCV run."""
    path.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now().date().isoformat()

    def _fmt(x, d: int = 4, signed: bool = True) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        if not isinstance(x, float):
            return str(x)
        return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"

    per_split = summary.get("per_split", [])
    rows = "\n".join(
        f"| {tuple(r['test_groups'])} | {r['n_train']:>7d} | "
        f"{r['n_test']:>6d} | {r['n_test_dates']:>4d} | "
        f"{_fmt(r['mean_ic'])} | {_fmt(r['ic_tstat_newey_west'], d=2)} |"
        for r in per_split
    )

    rel_plot = ""
    if plot_path is not None:
        try:
            rel = plot_path.resolve().relative_to(
                Path(REPO_ROOT).resolve()
            ).as_posix()
            rel_plot = f"![CPCV IC distribution]({rel})\n\n"
        except ValueError:
            rel_plot = f"![CPCV IC distribution]({plot_path.as_posix()})\n\n"

    dsr = summary["deflated_sharpe_prob_positive"]
    dsr_verdict = (
        "strategy survives the multiple-testing penalty"
        if dsr >= 0.95
        else "likely fluke — tighten search space or extend sample"
        if dsr < 0.5
        else "ambiguous — consider longer sample or fewer trials"
    )

    body = f"""# Combinatorial Purged CV — {today}

**Universe.** {summary['universe']} · **Horizon.** {summary['horizon_days']} trading days ·
**Partition.** N={summary['n_splits']}, K={summary['n_test_splits']} →
{summary['n_combos']} usable of C(N,K)={math.comb(summary['n_splits'], summary['n_test_splits'])}
combos ({summary['n_paths']} distinct stitched paths possible).
**Embargo.** {summary['embargo_pct']:.2%} of calendar span on top of horizon purge.

{rel_plot}## Headline numbers

| Statistic | Value |
| --- | ---: |
| Mean IC across combos | **{_fmt(summary['mean_ic'])}** |
| Std IC across combos | {_fmt(summary['std_ic'], signed=False)} |
| IC range | [{_fmt(summary['min_ic'])}, {_fmt(summary['max_ic'])}] |
| % combos with IC > 0 | {summary['pct_ic_positive']:.1%} |
| IID t-stat across combos | {_fmt(ic_stats['t_stat_iid'], d=2)} |
| IID p-value | {_fmt(ic_stats['p_value_iid'], d=4, signed=False)} |
| Newey-West t-stat | {_fmt(ic_stats['t_stat_nw'], d=2)} |
| Newey-West p-value | {_fmt(ic_stats['p_value_nw'], d=4, signed=False)} |
| E[max IC \\| null, N={summary['n_trials_assumed']} trials] | {_fmt(summary['expected_max_ic_under_null'])} |
| **Deflated Sharpe P(SR* > 0)** | **{dsr:.3f}** — {dsr_verdict} |

## Per-combo detail

| Test groups | n_train | n_test | n_dates | mean IC | NW t |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

## How to read this

* **Deflated Sharpe** is the Bailey / López de Prado penalty that converts the
  headline Sharpe into P(true SR > 0) after accounting for (a) the N_trials
  the pipeline evaluated on this data and (b) the cross-trial Sharpe stdev.
  Values above 0.95 are meaningful; below 0.5 means the headline is likely a
  selection-bias artifact.
* **CPCV combos share training rows** — the reported IC std across combos is
  a lower bound on the true OOS uncertainty, not an unbiased estimate.
* **Purge = horizon** drops training rows whose forward-return window overlaps
  the test group. **Embargo** drops a further ~{summary['embargo_pct']:.1%} of
  the calendar span after each test group to defend against residual serial
  correlation.

## Re-run

```
python scripts/run_cpcv_report.py --universe {summary['universe']} \\
    --n-splits {summary['n_splits']} --n-test {summary['n_test_splits']} \\
    --horizon {summary['horizon_days']} --embargo-pct {summary['embargo_pct']} \\
    --n-trials-assumed {summary['n_trials_assumed']}
```
"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="sp100")
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--n-splits", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=2)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--n-trials-assumed", type=int, default=50,
                        help="How many Optuna configs the pipeline nominally searched; "
                             "used to deflate Sharpe for selection bias.")
    args = parser.parse_args()

    run_cpcv_report(
        universe=args.universe,
        lookback_years=args.lookback_years,
        horizon_days=args.horizon,
        n_splits=args.n_splits,
        n_test_splits=args.n_test,
        embargo_pct=args.embargo_pct,
        n_trials_assumed=args.n_trials_assumed,
    )


if __name__ == "__main__":
    main()
