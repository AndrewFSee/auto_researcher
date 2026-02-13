"""
Walk-Forward Backtest of ML Ranking Model

This script tests the ML model's out-of-sample predictive power using
a proper walk-forward methodology.
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


def run_walkforward_backtest(
    universe: str = "sp100",
    lookback_years: int = 5,
    min_train_days: int = 252,
    rebal_freq: int = 21,
    horizon: int = 21,
):
    """Run walk-forward backtest of ML model."""
    
    print("Walk-Forward Backtest of ML Model")
    print("=" * 60)
    
    # Get tickers
    tickers = UNIVERSES[universe]()
    
    # Fetch data
    end = datetime.now()
    start = end - timedelta(days=lookback_years * 365)
    print(f"Fetching data from {start.date()} to {end.date()}...")
    
    prices = yf.download(tickers + ["SPY"], start=start, end=end, progress=False)["Close"]
    prices = prices.dropna(axis=1, how="all")
    print(f"Got {len(prices)} days, {len(prices.columns)} stocks")
    
    # Feature config
    feature_config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,
        use_cross_sec_norm=True,
    )
    
    # Compute all features upfront
    print("Computing features...")
    features = compute_all_enhanced_features(prices, benchmark="SPY", config=feature_config)
    features_long = features.stack(level=0, future_stack=True).dropna()
    features_long.index.names = ["date", "ticker"]
    
    # Cap at +/- 3 z-scores
    for col in features_long.columns:
        features_long[col] = features_long[col].clip(-3, 3)
    
    # Compute targets
    print("Computing targets...")
    target_config = TargetConfig(mode="vol_norm", horizon_days=horizon)
    targets = build_targets(prices, target_config, benchmark="SPY")
    targets_long = targets.stack()
    targets_long.index.names = ["date", "ticker"]
    
    # Align
    aligned = features_long.join(targets_long.rename("target"), how="inner").dropna()
    print(f"Aligned data: {len(aligned)} samples")
    
    # Get unique dates
    dates = aligned.index.get_level_values("date").unique().sort_values()
    print(f"Date range: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")
    
    # Walk-forward loop
    results = []
    rebal_dates = dates[min_train_days::rebal_freq]
    print(f"\nRunning walk-forward with {len(rebal_dates)} rebalance dates...")
    print("-" * 60)
    
    for i, rebal_date in enumerate(rebal_dates):
        if i >= len(rebal_dates) - 1:
            break
        
        # Training data: all data before rebal_date
        train_mask = aligned.index.get_level_values("date") < rebal_date
        train_data = aligned[train_mask]
        
        if len(train_data) < 1000:
            continue
        
        X_train = train_data.drop(columns=["target"])
        y_train = train_data["target"]
        
        # Fit model
        config = XGBRegressionConfig(
            objective="reg:pseudohubererror",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
        )
        model = XGBRegressionModel(config)
        model.fit(X_train, y_train)
        
        # Predict on rebal_date
        test_mask = aligned.index.get_level_values("date") == rebal_date
        test_data = aligned[test_mask]
        
        if len(test_data) < 10:
            continue
        
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]
        
        preds = model.predict(X_test)
        
        # Compute quintile returns
        pred_series = pd.Series(preds, index=X_test.index.get_level_values("ticker"))
        actual_series = y_test.droplevel(0)
        
        # Top and bottom quintile
        n = len(pred_series)
        q = max(n // 5, 1)
        top_tickers = pred_series.nlargest(q).index
        bot_tickers = pred_series.nsmallest(q).index
        
        top_ret = actual_series[top_tickers].mean()
        bot_ret = actual_series[bot_tickers].mean()
        spread = top_ret - bot_ret
        
        ic = pred_series.corr(actual_series)
        
        results.append({
            "date": rebal_date,
            "ic": ic,
            "top_ret": top_ret,
            "bot_ret": bot_ret,
            "spread": spread,
            "n_stocks": n,
        })
        
        print(f"{rebal_date.date()}: IC={ic:+.3f}, Top={top_ret:+.3f}, Bot={bot_ret:+.3f}, L/S={spread:+.3f}")
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print()
    print("=" * 60)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 60)
    print(f"Periods: {len(results_df)}")
    print(f"Date range: {results_df['date'].min().date()} to {results_df['date'].max().date()}")
    print()
    print("Information Coefficient (IC):")
    print(f"  Mean IC: {results_df['ic'].mean():.4f}")
    print(f"  IC Std:  {results_df['ic'].std():.4f}")
    print(f"  IC t-stat: {results_df['ic'].mean() / (results_df['ic'].std() / np.sqrt(len(results_df))):.2f}")
    print(f"  IC > 0:  {(results_df['ic'] > 0).mean():.1%}")
    print()
    print("Long/Short Spread (Top vs Bottom Quintile):")
    print(f"  Mean L/S Spread: {results_df['spread'].mean():.4f}")
    print(f"  L/S Std:         {results_df['spread'].std():.4f}")
    print(f"  L/S t-stat:      {results_df['spread'].mean() / (results_df['spread'].std() / np.sqrt(len(results_df))):.2f}")
    print(f"  L/S > 0:         {(results_df['spread'] > 0).mean():.1%}")
    print()
    print("Quintile Returns:")
    print(f"  Mean Top Quintile Return:    {results_df['top_ret'].mean():.4f}")
    print(f"  Mean Bottom Quintile Return: {results_df['bot_ret'].mean():.4f}")
    
    # By year analysis
    results_df["year"] = results_df["date"].dt.year
    print()
    print("Performance by Year:")
    print("-" * 40)
    for year in sorted(results_df["year"].unique()):
        year_data = results_df[results_df["year"] == year]
        print(f"  {year}: IC={year_data['ic'].mean():+.3f}, L/S={year_data['spread'].mean():+.3f} ({len(year_data)} periods)")
    
    return results_df


if __name__ == "__main__":
    results = run_walkforward_backtest(
        universe="sp100",
        lookback_years=5,
        min_train_days=252,
        rebal_freq=21,
    )
