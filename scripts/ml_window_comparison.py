"""
Compare different training window lengths for ML model.

Tests 6, 12, 24, and 36-month rolling windows across multiple years
to see which generalizes best across different market regimes.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from recommend import UNIVERSES
from auto_researcher.features.enhanced import EnhancedFeatureConfig, compute_all_enhanced_features
from auto_researcher.features.targets import TargetConfig, build_targets
from auto_researcher.models.xgb_ranking_model import XGBRegressionModel, XGBRegressionConfig


def run_window_comparison():
    print("Extended Walk-Forward: Comparing Training Windows")
    print("=" * 70)

    tickers = UNIVERSES["sp100"]()
    end = datetime.now()
    start = end - timedelta(days=2520)  # 10 years

    print("Fetching 10 years of data...")
    prices = yf.download(tickers + ["SPY"], start=start, end=end, progress=False)["Close"]
    prices = prices.dropna(axis=1, how="all")
    print(f"Got {len(prices)} days, {len(prices.columns)} stocks")

    feature_config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,
        use_cross_sec_norm=True,
        cross_sec_norm_by_sector=True,
    )

    print("Computing features...")
    features = compute_all_enhanced_features(prices, benchmark="SPY", config=feature_config)
    features_long = features.stack(level=0, future_stack=True).dropna()
    features_long.index.names = ["date", "ticker"]

    for col in features_long.columns:
        features_long[col] = features_long[col].clip(-3, 3)

    print("Computing targets...")
    target_config = TargetConfig(mode="vol_norm", horizon_days=21)
    targets = build_targets(prices, target_config, benchmark="SPY")
    targets_long = targets.stack()
    targets_long.index.names = ["date", "ticker"]

    aligned = features_long.join(targets_long.rename("target"), how="inner").dropna()
    dates = aligned.index.get_level_values("date").unique().sort_values()
    print(f"Aligned data: {len(aligned)} samples, {len(dates)} dates")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

    config = XGBRegressionConfig(objective="reg:pseudohubererror", n_estimators=100, max_depth=3)

    # Test multiple training windows
    windows = [6, 12, 24, 36]
    results = {w: [] for w in windows}

    # Monthly rebalancing - need 36 months of data first
    min_start = dates[0] + pd.Timedelta(days=36 * 21)
    rebal_dates = [d for d in dates[::21] if d >= min_start]
    print(f"Testing on {len(rebal_dates)} rebalance dates from {rebal_dates[0].date()}")
    print()

    for i, test_date in enumerate(rebal_dates):
        test_mask = aligned.index.get_level_values("date") == test_date
        test_data = aligned[test_mask]

        if len(test_data) < 20:
            continue

        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]

        for lookback_months in windows:
            lookback_days = lookback_months * 21
            train_start = test_date - pd.Timedelta(days=lookback_days)

            train_mask = (aligned.index.get_level_values("date") >= train_start) & \
                         (aligned.index.get_level_values("date") < test_date)
            train_data = aligned[train_mask]

            if len(train_data) < 500:
                continue

            X_train = train_data.drop(columns=["target"])
            y_train = train_data["target"]

            model = XGBRegressionModel(config)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            ic = pd.Series(preds).corr(y_test.reset_index(drop=True))

            # Quintile spread
            pred_series = pd.Series(preds, index=X_test.index.get_level_values("ticker"))
            actual_series = y_test.droplevel(0)
            n = len(pred_series)
            q = max(n // 5, 1)
            top = pred_series.nlargest(q).index
            bot = pred_series.nsmallest(q).index
            spread = actual_series[top].mean() - actual_series[bot].mean()

            results[lookback_months].append({
                "date": test_date,
                "ic": ic,
                "spread": spread,
                "year": test_date.year
            })

        if i % 20 == 0:
            print(f"  Processed {i+1}/{len(rebal_dates)} dates...")

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print()
    print("Training Window Comparison (Full Period):")
    print("-" * 50)
    
    for w in windows:
        df = pd.DataFrame(results[w])
        if len(df) > 0:
            mean_ic = df["ic"].mean()
            ic_std = df["ic"].std()
            ic_tstat = mean_ic / (ic_std / np.sqrt(len(df)))
            mean_spread = df["spread"].mean()
            pct_pos = (df["ic"] > 0).mean() * 100
            print(f"{w:2d}-month window:")
            print(f"   Mean IC: {mean_ic:+.4f} (t={ic_tstat:.2f})")
            print(f"   IC > 0:  {pct_pos:.1f}%")
            print(f"   L/S Spread: {mean_spread:+.3f}")
            print()

    print()
    print("Performance BY YEAR:")
    print("-" * 70)
    all_years = sorted(set(r["year"] for w in windows for r in results[w]))

    # Print header
    header = "Year   "
    for w in windows:
        header += f"  {w:2d}m IC  "
    print(header)
    print("-" * len(header))

    for year in all_years:
        row = f"{year}   "
        for w in windows:
            df = pd.DataFrame(results[w])
            year_df = df[df["year"] == year]
            if len(year_df) > 0:
                row += f"  {year_df['ic'].mean():+.3f}  "
            else:
                row += "   N/A   "
        print(row)
    
    return results


if __name__ == "__main__":
    results = run_window_comparison()
