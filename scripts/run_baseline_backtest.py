#!/usr/bin/env python3
"""
Baseline Evaluation Script for Auto-Researcher.

This script performs a complete sanity check of the research pipeline:
- Data loading from yfinance
- Technical feature engineering
- GBDT model training
- Walk-forward backtesting
- Metrics computation and validation

Run with:
    python scripts/run_baseline_backtest.py

Optional: Set ENABLE_PLOTS = True to generate matplotlib visualizations.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
import pandas as pd
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ENABLE_PLOTS = False  # Set to True to generate plots

# Universe and date range
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"

# Backtest parameters
HORIZON_DAYS = 63  # ~3 months forward returns
REBALANCE_FREQ = "M"  # Monthly rebalancing
TOP_K = 3
BENCHMARK = "SPY"
MIN_TRAIN_PERIODS = 252  # 1 year minimum training data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# ==============================================================================
# STEP A: SETUP & IMPORTS
# ==============================================================================

print_header("STEP A: Setup & Imports")

try:
    from auto_researcher.config import (
        PipelineConfig,
        ResearchConfig,
        FeatureConfig,
        ModelConfig,
    )
    from auto_researcher.data.universe import get_universe
    from auto_researcher.data.price_loader import (
        download_price_history,
        get_adjusted_close,
        validate_price_data,
    )
    from auto_researcher.features.feature_pipeline import (
        build_feature_matrix,
        build_training_data,
        prepare_training_data,
    )
    from auto_researcher.features.technical import compute_returns
    from auto_researcher.models.gbdt_model import GBDTModel
    from auto_researcher.backtest.runner import run_backtest, BacktestResult
    from auto_researcher.backtest.metrics import (
        compute_cumulative_returns,
        compute_annualized_return,
        compute_annualized_volatility,
        compute_sharpe_ratio,
        compute_hit_rate,
        compute_ic,
        compute_all_metrics,
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Make sure the package is installed: pip install -e .")
    sys.exit(1)

# Create configuration
config = PipelineConfig(
    research=ResearchConfig(
        universe=tuple(UNIVERSE),
        start_date=START_DATE,
        end_date=END_DATE,
        horizon_days=HORIZON_DAYS,
        top_k=TOP_K,
        rebalance_frequency=REBALANCE_FREQ,
        benchmark=BENCHMARK,
        lookback_days=MIN_TRAIN_PERIODS,
    ),
    features=FeatureConfig(
        momentum_windows=(21, 63, 126, 252),
        volatility_windows=(21, 63),
        include_fundamentals=False,
        include_sentiment=False,
    ),
    model=ModelConfig(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        early_stopping_rounds=None,
    ),
)

print(f"✓ Configuration created")
print(f"  Universe: {UNIVERSE}")
print(f"  Date range: {START_DATE} to {END_DATE}")
print(f"  Horizon: {HORIZON_DAYS} days")
print(f"  Rebalance: {REBALANCE_FREQ}")
print(f"  Top-K: {TOP_K}")
print(f"  Benchmark: {BENCHMARK}")


# ==============================================================================
# STEP B: LOAD DATA
# ==============================================================================

print_header("STEP B: Load Price Data")

try:
    print(f"Downloading price data for {len(UNIVERSE)} tickers...")
    raw_prices = download_price_history(
        tickers=UNIVERSE,
        start=START_DATE,
        end=END_DATE,
    )
    prices = get_adjusted_close(raw_prices)
    print(f"✓ Downloaded {len(prices)} trading days")
except Exception as e:
    print(f"✗ Data download failed: {e}")
    sys.exit(1)

# Validate data
print_subheader("Data Validation")

# Check non-empty
if prices.empty:
    print("✗ FAIL: Price DataFrame is empty")
    sys.exit(1)
print(f"✓ DataFrame non-empty: {prices.shape}")

# Check all tickers present
missing_tickers = set(UNIVERSE) - set(prices.columns)
if missing_tickers:
    print(f"✗ FAIL: Missing tickers: {missing_tickers}")
    sys.exit(1)
print(f"✓ All tickers present: {list(prices.columns)}")

# Check for all-NaN columns
all_nan_cols = prices.columns[prices.isna().all()].tolist()
if all_nan_cols:
    print(f"✗ FAIL: All-NaN columns: {all_nan_cols}")
    sys.exit(1)
print(f"✓ No all-NaN columns")

# Print data summary
print_subheader("Data Summary")
print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
print(f"  Shape: {prices.shape}")
print(f"  Missing values per ticker:")
for col in prices.columns:
    nan_count = prices[col].isna().sum()
    nan_pct = nan_count / len(prices) * 100
    print(f"    {col}: {nan_count} ({nan_pct:.1f}%)")


# ==============================================================================
# STEP C: BUILD FEATURES + LABELS
# ==============================================================================

print_header("STEP C: Build Features & Labels")

print("Computing returns...")
returns = compute_returns(prices)
print(f"✓ Returns shape: {returns.shape}")

print("Building feature matrix...")
feature_matrix = build_feature_matrix(prices, config.features)
print(f"✓ Feature matrix shape: {feature_matrix.shape}")

# Feature summary
print_subheader("Feature Summary")
tickers_in_features = feature_matrix.columns.get_level_values("ticker").unique()
features_names = feature_matrix.columns.get_level_values("feature").unique()
print(f"  Tickers: {list(tickers_in_features)}")
print(f"  Features per ticker: {len(features_names)}")
print(f"  Feature names: {list(features_names)}")

# Check for NaN in features after warmup
warmup_days = 252  # Conservative warmup
features_after_warmup = feature_matrix.iloc[warmup_days:]
nan_pct = features_after_warmup.isna().mean().mean() * 100
print(f"  NaN % after warmup: {nan_pct:.2f}%")

print("\nBuilding training data (features + forward return labels)...")
try:
    X, y = prepare_training_data(
        prices=prices,
        horizon_days=HORIZON_DAYS,
        benchmark=BENCHMARK,
        config=config.features,
    )
    print(f"✓ Training data prepared")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
except Exception as e:
    print(f"✗ Failed to prepare training data: {e}")
    sys.exit(1)

# Check for NaN/inf in training data
print_subheader("Training Data Validation")
x_nan = X.isna().any().any()
y_nan = y.isna().any()
x_inf = np.isinf(X.values).any()
y_inf = np.isinf(y.values).any()

print(f"  X contains NaN: {x_nan}")
print(f"  y contains NaN: {y_nan}")
print(f"  X contains Inf: {x_inf}")
print(f"  y contains Inf: {y_inf}")

if x_nan or y_nan or x_inf or y_inf:
    print("✗ FAIL: Training data contains invalid values")
    sys.exit(1)
print("✓ Training data is clean (no NaN/Inf)")

# Label statistics
print_subheader("Label (Forward Return) Statistics")
print(f"  Mean: {y.mean():.4f}")
print(f"  Std:  {y.std():.4f}")
print(f"  Min:  {y.min():.4f}")
print(f"  Max:  {y.max():.4f}")


# ==============================================================================
# STEP D: TRAIN + WALK-FORWARD BACKTEST
# ==============================================================================

print_header("STEP D: Train Model & Run Walk-Forward Backtest")

print("Initializing GBDT model...")
model = GBDTModel(config.model)
print(f"✓ Model initialized with config:")
print(f"    n_estimators: {config.model.n_estimators}")
print(f"    learning_rate: {config.model.learning_rate}")
print(f"    max_depth: {config.model.max_depth}")

print("\nRunning walk-forward backtest...")
print("(This may take a minute...)")

try:
    result: BacktestResult = run_backtest(
        prices=prices,
        horizon_days=HORIZON_DAYS,
        top_k=TOP_K,
        model=model,
        benchmark=BENCHMARK,
        rebalance_frequency=REBALANCE_FREQ,
        min_train_periods=MIN_TRAIN_PERIODS,
        config=config,
    )
    print(f"✓ Backtest completed successfully")
    print(f"  Number of rebalances: {len(result.weights_history)}")
    print(f"  Trading periods: {len(result.portfolio_returns)}")
except Exception as e:
    print(f"✗ Backtest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Validate predictions
print_subheader("Prediction Validation")
port_returns = result.portfolio_returns
if port_returns.isna().any():
    print("⚠ WARNING: Portfolio returns contain NaN")
if np.isinf(port_returns.values).any():
    print("✗ FAIL: Portfolio returns contain Inf")
    sys.exit(1)
print(f"✓ Portfolio returns are finite: {len(port_returns)} periods")


# ==============================================================================
# STEP E: PRINT KEY METRICS
# ==============================================================================

print_header("STEP E: Performance Metrics")

metrics = result.metrics

# Portfolio vs Benchmark cumulative returns
port_cumret = compute_cumulative_returns(result.portfolio_returns)
bench_cumret = compute_cumulative_returns(result.benchmark_returns)

print_subheader("Cumulative Returns")
print(f"  Portfolio final:  {(port_cumret.iloc[-1] - 1) * 100:.2f}%")
print(f"  Benchmark final:  {(bench_cumret.iloc[-1] - 1) * 100:.2f}%")
print(f"  Excess return:    {((port_cumret.iloc[-1] / bench_cumret.iloc[-1]) - 1) * 100:.2f}%")

print_subheader("Risk-Adjusted Metrics")
print(f"  Annualized Return:     {metrics.get('annualized_return', 0) * 100:>8.2f}%")
print(f"  Annualized Volatility: {metrics.get('annualized_volatility', 0) * 100:>8.2f}%")
print(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>8.3f}")
print(f"  Sortino Ratio:         {metrics.get('sortino_ratio', 0):>8.3f}")
print(f"  Max Drawdown:          {metrics.get('max_drawdown', 0) * 100:>8.2f}%")

print_subheader("Ranking Metrics")
hit_rate = metrics.get('hit_rate', 0) * 100
print(f"  Hit Rate (vs SPY):     {hit_rate:>8.2f}%")

# IC statistics
if result.ic_history:
    ic_values = [ic for _, ic in result.ic_history]
    ic_mean = np.mean(ic_values)
    ic_std = np.std(ic_values)
    ic_median = np.median(ic_values)
    print(f"  Average IC:            {ic_mean:>8.4f}")
    print(f"  IC Std Dev:            {ic_std:>8.4f}")
    print(f"  Median IC:             {ic_median:>8.4f}")
else:
    ic_mean = 0
    ic_std = 0
    print("  IC: No data available")

print_subheader("Benchmark Comparison")
print(f"  Benchmark Ann. Return: {metrics.get('benchmark_return', 0) * 100:>8.2f}%")
print(f"  Excess Ann. Return:    {metrics.get('excess_return', 0) * 100:>8.2f}%")
print(f"  Information Ratio:     {metrics.get('information_ratio', 0):>8.3f}")


# ==============================================================================
# METRIC SANITY CHECK EXPECTATIONS
# ==============================================================================

print_header("METRIC SANITY CHECK EXPECTATIONS")

print("""
These are typical realistic ranges for a technical-only factor model
on US large-cap equities. Values outside these ranges may indicate
data issues, lookahead bias, or implementation bugs.

┌────────────────────┬────────────────────────────────────────────────────┐
│ Metric             │ Realistic Range / Notes                            │
├────────────────────┼────────────────────────────────────────────────────┤
│ IC (average)       │ 0.00 to 0.05 for technical factors                │
│                    │ > 0.10 consistently is suspicious (possible leak)  │
├────────────────────┼────────────────────────────────────────────────────┤
│ Hit Rate           │ Noise baseline: ~50%                               │
│                    │ Realistic weak edge: 52–57%                        │
│                    │ > 60% long-term is suspect for simple signals      │
├────────────────────┼────────────────────────────────────────────────────┤
│ Annualized Return  │ Should not be absurd                               │
│                    │ > 40% on simple signals is unlikely                │
├────────────────────┼────────────────────────────────────────────────────┤
│ Volatility         │ Concentrated US equity: 10–30%                     │
│                    │ < 5% or > 50% is suspicious                        │
├────────────────────┼────────────────────────────────────────────────────┤
│ Sharpe Ratio       │ Plausible pre-cost: 0.3 to 1.0                     │
│                    │ > 2.0 long-term is usually suspicious              │
└────────────────────┴────────────────────────────────────────────────────┘
""")

# Automated sanity checks
print_subheader("Automated Sanity Checks")

warnings = []
passes = []

# IC check
if abs(ic_mean) > 0.15:
    warnings.append(f"IC mean ({ic_mean:.4f}) is suspiciously high - check for lookahead bias")
else:
    passes.append(f"IC mean ({ic_mean:.4f}) is within expected range")

# Hit rate check
if hit_rate > 65:
    warnings.append(f"Hit rate ({hit_rate:.1f}%) is suspiciously high")
elif hit_rate < 40:
    warnings.append(f"Hit rate ({hit_rate:.1f}%) is suspiciously low")
else:
    passes.append(f"Hit rate ({hit_rate:.1f}%) is within expected range")

# Return check
ann_ret = metrics.get('annualized_return', 0) * 100
if abs(ann_ret) > 50:
    warnings.append(f"Annualized return ({ann_ret:.1f}%) seems unrealistic")
else:
    passes.append(f"Annualized return ({ann_ret:.1f}%) is within expected range")

# Volatility check
ann_vol = metrics.get('annualized_volatility', 0) * 100
if ann_vol < 5:
    warnings.append(f"Volatility ({ann_vol:.1f}%) is suspiciously low")
elif ann_vol > 50:
    warnings.append(f"Volatility ({ann_vol:.1f}%) is suspiciously high")
else:
    passes.append(f"Volatility ({ann_vol:.1f}%) is within expected range")

# Sharpe check
sharpe = metrics.get('sharpe_ratio', 0)
if sharpe > 2.5:
    warnings.append(f"Sharpe ratio ({sharpe:.2f}) is suspiciously high")
else:
    passes.append(f"Sharpe ratio ({sharpe:.2f}) is within expected range")

for msg in passes:
    print(f"  ✓ {msg}")
for msg in warnings:
    print(f"  ⚠ {msg}")


# ==============================================================================
# STEP F: OPTIONAL PLOTS
# ==============================================================================

if ENABLE_PLOTS:
    print_header("STEP F: Generating Plots")
    
    try:
        import matplotlib.pyplot as plt
        
        # Create plots directory
        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Cumulative Returns
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(port_cumret.index, port_cumret.values, label="Portfolio", linewidth=2)
        ax.plot(bench_cumret.index, bench_cumret.values, label="SPY Benchmark", linewidth=2, alpha=0.7)
        ax.set_title("Cumulative Returns: Portfolio vs SPY", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = plots_dir / "cumulative_returns.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {plot_path}")
        plt.close(fig)
        
        # Plot 2: IC Timeseries
        if result.ic_history:
            fig, ax = plt.subplots(figsize=(12, 4))
            ic_dates = [d for d, _ in result.ic_history]
            ic_vals = [ic for _, ic in result.ic_history]
            
            ax.bar(ic_dates, ic_vals, width=20, alpha=0.7)
            ax.axhline(y=ic_mean, color='red', linestyle='--', label=f'Mean IC: {ic_mean:.4f}')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title("Information Coefficient (IC) Over Time", fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Spearman IC")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = plots_dir / "ic_timeseries.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {plot_path}")
            plt.close(fig)
        
        print("✓ All plots generated successfully")
        
    except ImportError:
        print("⚠ matplotlib not installed - skipping plots")
    except Exception as e:
        print(f"⚠ Error generating plots: {e}")
else:
    print_header("STEP F: Plots Disabled")
    print("Set ENABLE_PLOTS = True at the top of this script to generate plots.")


# ==============================================================================
# SUMMARY
# ==============================================================================

print_header("SUMMARY")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  BASELINE TEST COMPLETED SUCCESSFULLY                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  ✓ Data loaded and validated                                         ║
║  ✓ Features built correctly                                          ║
║  ✓ Model trained and predicted                                       ║
║  ✓ Backtest ran end-to-end                                           ║
║  ✓ Metrics are finite and computed                                   ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  METRICS SUMMARY                                                      ║
╟──────────────────────────────────────────────────────────────────────╢""")

print(f"║  Annualized Return:     {metrics.get('annualized_return', 0) * 100:>7.2f}%                              ║")
print(f"║  Annualized Volatility: {metrics.get('annualized_volatility', 0) * 100:>7.2f}%                              ║")
print(f"║  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>7.3f}                               ║")
print(f"║  Hit Rate:              {hit_rate:>7.2f}%                              ║")
print(f"║  Average IC:            {ic_mean:>7.4f}                               ║")

assessment = "REASONABLE" if len(warnings) == 0 else "NEEDS REVIEW"
print(f"""╟──────────────────────────────────────────────────────────────────────╢
║  Overall Assessment: {assessment:<15}                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  NEXT STEPS                                                          ║
╟──────────────────────────────────────────────────────────────────────╢
║  1. Add fundamental factors (value, quality, growth)                 ║
║  2. Add sentiment features (FinBERT on news/earnings)                ║
║  3. Implement Bayesian risk model for position sizing                ║
║  4. Add transaction cost modeling                                    ║
║  5. Expand universe to full S&P 500                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

if warnings:
    print("⚠ WARNINGS:")
    for w in warnings:
        print(f"   - {w}")
    print()

print("Baseline test completed successfully.")
sys.exit(0)
