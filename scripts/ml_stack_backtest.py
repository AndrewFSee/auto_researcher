"""
ML-stack CPCV backtest: XGBoost + Transformer + GNN + IC-weighted ensemble.

For each combinatorial-purged train/test split we:

1. Carve a small **calibration** slice off the tail of the train fold
   (last ``cal_frac`` by date). That slice never touches model fitting --
   it's held out purely to measure per-model IC for the ensemble
   blender.
2. Fit XGBoost, TransformerRanker, and GNNRanker on the train fold
   *minus* the calibration slice.
3. Compute each model's mean cross-sectional IC on the calibration
   slice -> feed to ``ICWeightedEnsemble`` -> get blend weights.
4. Score the test fold with each individual model and with the
   ensemble; report per-model and ensemble mean-IC distributions.

Why this sequencing
-------------------
The ensemble weights are a learned parameter -- leaking the test fold
into the calibration measurement would be a garden-variety future-leak.
Pulling the calibration slice from the train-fold tail keeps everything
causal without eating into the test-fold statistical power.

Usage::

    python scripts/ml_stack_backtest.py --universe sp100 --n-splits 6 \\
        --n-test 2 --horizon 21 --epochs 10

Outputs land in ``results/ml_stack/``.
"""

from __future__ import annotations

import argparse
import json
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
from auto_researcher.models.ensemble_ranker import (
    ICWeightedEnsemble,
    _mean_cross_sectional_ic,
)
from auto_researcher.models.gnn_ranker import GNNRanker, GNNRankerConfig
from auto_researcher.models.transformer_ranker import (
    TransformerRanker,
    TransformerRankerConfig,
)
from auto_researcher.models.xgb_ranking_model import (
    XGBRegressionConfig,
    XGBRegressionModel,
)
from auto_researcher.validation.cpcv import combinatorial_purged_splits
from auto_researcher.screening import UNIVERSES


def _ic_by_date(preds: pd.Series, y: pd.Series) -> float:
    return _mean_cross_sectional_ic(preds, y)


def _carve_calibration(
    train: pd.DataFrame, cal_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train by date: last ``cal_frac`` of unique dates -> calibration."""
    dates = train.index.get_level_values("date").unique().sort_values()
    if len(dates) < 10:
        # Too small to safely carve -- fall back to using all of train for fit
        # and a copy of tail-5 dates for calibration (cheap heuristic).
        split_idx = max(1, len(dates) - 2)
    else:
        split_idx = int(len(dates) * (1.0 - cal_frac))
    fit_dates = set(dates[:split_idx])
    cal_dates = set(dates[split_idx:])
    fit = train[train.index.get_level_values("date").isin(fit_dates)]
    cal = train[train.index.get_level_values("date").isin(cal_dates)]
    return fit, cal


def _per_split_returns_df(
    prices: pd.DataFrame, tickers: list[str], through: pd.Timestamp
) -> pd.DataFrame:
    """Daily returns up to ``through`` -- supplies the GNN's causal adjacency."""
    sub = prices.loc[prices.index <= through, tickers].dropna(axis=1, how="all")
    return sub.pct_change().dropna(how="all")


def run_ml_stack_backtest(
    universe: str = "sp100",
    lookback_years: int = 5,
    horizon_days: int = 21,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
    epochs: int = 10,
    cal_frac: float = 0.15,
    out_dir: Path = REPO_ROOT / "results" / "ml_stack",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = UNIVERSES[universe]()
    end = datetime.now()
    start = end - timedelta(days=lookback_years * 365)
    print(f"[ML-STACK] Fetching {len(tickers)} tickers + SPY from {start.date()} to {end.date()}")

    prices = yf.download(tickers + ["SPY"], start=start, end=end, progress=False)["Close"]
    prices = prices.dropna(axis=1, how="all")
    print(f"[ML-STACK] {len(prices)} days, {len(prices.columns)} stocks")

    feature_config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,
        use_cross_sec_norm=True,
    )
    print("[ML-STACK] Building features")
    features = compute_all_enhanced_features(prices, benchmark="SPY", config=feature_config)
    features_long = features.stack(level=0, future_stack=True).dropna()
    features_long.index.names = ["date", "ticker"]
    for col in features_long.columns:
        features_long[col] = features_long[col].clip(-3, 3)

    print("[ML-STACK] Building targets")
    target_config = TargetConfig(mode="vol_norm", horizon_days=horizon_days)
    targets = build_targets(prices, target_config, benchmark="SPY")
    targets_long = targets.stack()
    targets_long.index.names = ["date", "ticker"]

    aligned = features_long.join(targets_long.rename("target"), how="inner").dropna()
    print(f"[ML-STACK] Aligned: {len(aligned)} samples")

    dates = aligned.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()

    date_to_pos = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    row_date_pos = date_to_pos.reindex(dates).to_numpy()

    # One daily-returns frame, reused per split (sliced by through-date).
    price_tickers = [t for t in tickers if t in prices.columns]
    returns_full = prices[price_tickers].pct_change().dropna(how="all")

    per_split: list[dict] = []

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

        fit, cal = _carve_calibration(train, cal_frac)
        if len(fit) < 500 or len(cal) < 50:
            continue

        X_fit, y_fit = fit.drop(columns=["target"]), fit["target"]
        X_cal, y_cal = cal.drop(columns=["target"]), cal["target"]
        X_te, y_te = test.drop(columns=["target"]), test["target"]

        # GNN adjacency must be causal w.r.t. the latest train date -- slice
        # the returns frame to only dates seen during fit.
        fit_end = X_fit.index.get_level_values("date").max()
        returns_for_gnn = returns_full.loc[returns_full.index <= fit_end]

        # 1. XGB.
        xgb = XGBRegressionModel(
            XGBRegressionConfig(
                objective="reg:pseudohubererror",
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
            )
        )
        xgb.fit(X_fit, y_fit)

        # XGB doesn't expose predict_with_index, so wrap it.
        class _XGBWrap:
            def __init__(self, m): self._m = m
            def predict_with_index(self, X):
                return pd.Series(self._m.predict(X), index=X.index)
        xgb_wrapped = _XGBWrap(xgb)

        # 2. Transformer.
        tx = TransformerRanker(
            TransformerRankerConfig(
                d_model=32, n_heads=4, n_layers=2,
                n_epochs=epochs, lr=1e-3, batch_dates=4,
                random_state=0,
            )
        )
        tx.fit(X_fit, y_fit)

        # 3. GNN.
        gnn = GNNRanker(
            GNNRankerConfig(
                d_model=32, n_layers=2, n_epochs=epochs, lr=1e-3,
                corr_window=60, corr_threshold=0.5,
                use_sector_edges=False, random_state=0,
            ),
            returns_df=returns_for_gnn,
        )
        gnn.fit(X_fit, y_fit)

        models = {"xgb": xgb_wrapped, "tx": tx, "gnn": gnn}

        # Calibrate ensemble on held-out cal slice.
        ens = ICWeightedEnsemble(models)
        weights_info = ens.fit_weights(X_cal, y_cal)

        # Per-model and ensemble IC on the test fold.
        model_ics = {}
        for name, m in models.items():
            preds = m.predict_with_index(X_te)
            model_ics[name] = _ic_by_date(preds, y_te)
        ens_preds = ens.predict_with_index(X_te)
        ens_ic = _ic_by_date(ens_preds, y_te)

        record = {
            "test_groups": list(split.test_group_ids),
            "n_fit": int(len(X_fit)),
            "n_cal": int(len(X_cal)),
            "n_test": int(len(X_te)),
            "per_model_cal_ic": weights_info.per_model_ic,
            "ensemble_weights": weights_info.weights,
            "dropped_cal": weights_info.dropped,
            "per_model_test_ic": model_ics,
            "ensemble_test_ic": ens_ic,
        }
        per_split.append(record)

        ic_line = ", ".join(f"{n}={v:+.3f}" for n, v in model_ics.items())
        w_line = ", ".join(f"{n}={v:.2f}" for n, v in weights_info.weights.items())
        print(
            f"  combo {split.test_group_ids}: "
            f"per-model IC [{ic_line}] | ens IC {ens_ic:+.3f} | weights {{{w_line}}}"
        )

    if not per_split:
        raise RuntimeError("No splits produced usable results")

    def _agg(key: str) -> dict:
        vals = np.array([r[key] for r in per_split if np.isfinite(r.get(key, np.nan))])
        if not len(vals):
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "pct_positive": float((vals > 0).mean()),
            "n": int(len(vals)),
        }

    def _agg_per_model(name: str) -> dict:
        vals = np.array(
            [r["per_model_test_ic"].get(name, np.nan) for r in per_split],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "pct_positive": float((vals > 0).mean()),
            "n": int(len(vals)),
        }

    summary = {
        "universe": universe,
        "lookback_years": lookback_years,
        "horizon_days": horizon_days,
        "n_splits": n_splits,
        "n_test_splits": n_test_splits,
        "embargo_pct": embargo_pct,
        "epochs": epochs,
        "cal_frac": cal_frac,
        "n_splits_used": len(per_split),
        "ensemble_test_ic": _agg("ensemble_test_ic"),
        "per_model_test_ic": {
            name: _agg_per_model(name) for name in ("xgb", "tx", "gnn")
        },
        "per_split": per_split,
    }

    summary_path = out_dir / f"ml_stack_{universe}_{n_splits}_{n_test_splits}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[ML-STACK] Summary written to {summary_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        series = {
            "xgb": [r["per_model_test_ic"].get("xgb", np.nan) for r in per_split],
            "tx": [r["per_model_test_ic"].get("tx", np.nan) for r in per_split],
            "gnn": [r["per_model_test_ic"].get("gnn", np.nan) for r in per_split],
            "ensemble": [r["ensemble_test_ic"] for r in per_split],
        }
        positions = list(range(1, len(series) + 1))
        data = [np.array(v, dtype=float) for v in series.values()]
        data = [v[np.isfinite(v)] for v in data]
        ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(list(series.keys()))
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel("Per-split mean IC")
        ax.set_title(
            f"ML-stack CPCV IC -- {universe}, {len(per_split)} splits, "
            f"ens mean={summary['ensemble_test_ic']['mean']:+.4f}"
        )
        fig.tight_layout()
        plot_path = out_dir / f"ml_stack_{universe}_{n_splits}_{n_test_splits}.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[ML-STACK] Violin plot written to {plot_path}")
    except ImportError:
        print("[ML-STACK] matplotlib not available -- skipped violin plot")

    print("\n=== ML-STACK SUMMARY ===")
    for name in ("xgb", "tx", "gnn"):
        s = summary["per_model_test_ic"][name]
        print(
            f"  {name:>9}: mean={s['mean']:+.4f}  "
            f"std={s['std']:.4f}  %IC>0={s['pct_positive']:.0%}  n={s['n']}"
        )
    s = summary["ensemble_test_ic"]
    print(
        f"  {'ensemble':>9}: mean={s['mean']:+.4f}  "
        f"std={s['std']:.4f}  %IC>0={s['pct_positive']:.0%}  n={s['n']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="sp100")
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--n-splits", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=2)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs for the transformer and GNN (XGB uses its own schedule).",
    )
    parser.add_argument(
        "--cal-frac", type=float, default=0.15,
        help="Tail fraction of the train fold carved off for ensemble calibration.",
    )
    args = parser.parse_args()

    run_ml_stack_backtest(
        universe=args.universe,
        lookback_years=args.lookback_years,
        horizon_days=args.horizon,
        n_splits=args.n_splits,
        n_test_splits=args.n_test,
        embargo_pct=args.embargo_pct,
        epochs=args.epochs,
        cal_frac=args.cal_frac,
    )


if __name__ == "__main__":
    main()
