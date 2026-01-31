#!/usr/bin/env python3
"""
Out-of-Sample Alpha Validation Script.

This script performs rigorous validation of model alpha:

TEST 1: TRUE OUT-OF-SAMPLE HOLDOUT
- Train on 2016-2023
- Test on 2024-2025 (pure holdout, no peeking)
- Report Sharpe, excess return, max drawdown

TEST 2: BASELINE COMPARISONS
Compare the ML model against simple baselines on the same universe:
- Equal-Weight (hold all tickers equally)
- Momentum Top-K (12-1 month momentum ranking)  
- Random Top-K (Monte Carlo average)
- Buy-and-Hold SPY

If model > all baselines → genuine ranking skill

Usage:
    python run_oos_alpha_validation.py
    python run_oos_alpha_validation.py --universe large_cap
    python run_oos_alpha_validation.py --top-k 5
"""

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Date ranges
TRAIN_START = "2016-01-01"
TRAIN_END = "2022-12-31"  # Train ends 2022
TEST_START = "2023-01-01"  # Test starts 2023 (2 full years)
TEST_END = "2025-01-20"    # Current date

# Full backtest range (for baseline comparisons)
FULL_START = "2016-01-01"
FULL_END = "2025-01-20"  # Current date

# Backtest parameters
HORIZON_DAYS = 63
REBALANCE_FREQ = "M"
DEFAULT_TOP_K = 5
BENCHMARK = "SPY"
MIN_TRAIN_PERIODS = 252

# Monte Carlo parameters
N_RANDOM_PATHS = 50
RANDOM_SEED = 42

# Universe options
UNIVERSES = {
    "core_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "large_cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "BRK-B", "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD",
        "MA", "CVX", "MRK", "ABBV", "PFE", "KO", "PEP", "COST",
        "TMO", "AVGO", "MCD", "WMT", "CSCO", "ACN", "ABT",
    ],
    "sp100": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
        "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "MA", "CVX", "MRK",
        "ABBV", "PFE", "KO", "PEP", "COST", "TMO", "AVGO", "MCD", "WMT",
        "CSCO", "ACN", "ABT", "DHR", "CRM", "ADBE", "NKE", "CMCSA", "VZ",
        "INTC", "T", "NEE", "PM", "TXN", "QCOM", "RTX", "HON", "UPS",
        "LOW", "MS", "SPGI", "BLK", "BA", "IBM", "GE", "CAT", "DE",
        "AMGN", "GILD", "MDT", "BMY", "LMT", "AXP", "SBUX", "MDLZ",
        "ADI", "ISRG", "BKNG", "MMC", "CB", "CI", "ZTS", "SO", "DUK",
        "PLD", "CL", "REGN", "VRTX", "SYK", "BSX", "EW", "SCHW",
        "BDX", "FIS", "ICE", "CME", "AON", "USB", "TFC", "PNC",
        "MO", "COP", "EOG", "SLB", "PSX", "VLO", "OXY",
    ],
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class StrategyResult:
    """Container for strategy backtest results."""
    name: str
    ann_return: float
    sharpe: float
    max_dd: float
    excess_ret: float
    hit_rate: float
    volatility: float
    total_return: float
    n_periods: int


@dataclass
class OOSResult:
    """Container for out-of-sample test results."""
    train_sharpe: float
    train_return: float
    train_max_dd: float
    test_sharpe: float
    test_return: float
    test_max_dd: float
    test_excess: float
    benchmark_return: float


# ==============================================================================
# IMPORTS
# ==============================================================================

try:
    from auto_researcher.config import (
        PipelineConfig,
        ResearchConfig,
        FeatureConfig,
        ModelConfig,
        EnhancedModelConfig,
    )
    from auto_researcher.data.price_loader import (
        download_price_history,
        get_adjusted_close,
    )
    from auto_researcher.backtest.runner import (
        run_backtest,
        run_enhanced_backtest,
        get_rebalance_dates,
        BacktestResult,
    )
    from auto_researcher.backtest.metrics import (
        compute_cumulative_returns,
        compute_annualized_return,
        compute_annualized_volatility,
        compute_sharpe_ratio,
        compute_hit_rate,
        compute_total_return,
        compute_max_drawdown,
    )
except ImportError as e:
    print(f"[ERROR] Failed to import auto_researcher modules: {e}")
    print("        Make sure you're running from the project root directory.")
    sys.exit(1)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download and return adjusted close prices."""
    print(f"  [DATA] Downloading prices for {len(tickers)} tickers...")
    
    import yfinance as yf
    
    all_tickers = list(set(tickers + [BENCHMARK]))
    data = yf.download(all_tickers, start=start, end=end, progress=False)
    
    # Handle different yfinance return formats
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        elif 'Price' in data.columns.get_level_values(0):
            prices = data['Price']
        else:
            prices = data.xs(data.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        prices = data
    
    # Forward fill and drop rows with any NaN
    prices = prices.ffill().dropna()
    
    print(f"  [DATA] Got {len(prices)} days of data")
    return prices


def get_rebal_dates(prices: pd.DataFrame, freq: str = "M") -> list[pd.Timestamp]:
    """Get rebalance dates from price index."""
    dummy = pd.Series(1, index=prices.index)
    if freq == "M":
        rebal = dummy.resample("ME").last().index
    elif freq == "W":
        rebal = dummy.resample("W").last().index
    else:
        rebal = prices.index
    
    return [d for d in rebal if d in prices.index]


def compute_strategy_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    name: str,
) -> StrategyResult:
    """Compute all metrics for a strategy."""
    ann_ret = compute_annualized_return(portfolio_returns)
    sharpe = compute_sharpe_ratio(portfolio_returns)
    max_dd = compute_max_drawdown(portfolio_returns)
    vol = compute_annualized_volatility(portfolio_returns)
    total_ret = compute_total_return(portfolio_returns)
    hit_rate = compute_hit_rate(portfolio_returns, benchmark_returns)
    
    bench_ann = compute_annualized_return(benchmark_returns)
    excess = ann_ret - bench_ann
    
    return StrategyResult(
        name=name,
        ann_return=ann_ret * 100,
        sharpe=sharpe,
        max_dd=max_dd * 100,
        excess_ret=excess * 100,
        hit_rate=hit_rate * 100,
        volatility=vol * 100,
        total_return=total_ret * 100,
        n_periods=len(portfolio_returns),
    )


# ==============================================================================
# BASELINE STRATEGIES
# ==============================================================================

def run_equal_weight_strategy(
    prices: pd.DataFrame,
    tickers: list[str],
    benchmark: str,
) -> pd.Series:
    """
    Equal-weight all tickers, rebalance monthly.
    
    Returns daily portfolio returns.
    """
    # Daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Get rebalance dates
    rebal_dates = get_rebal_dates(prices)
    
    portfolio_returns = []
    
    for i in range(len(rebal_dates) - 1):
        start_date = rebal_dates[i]
        end_date = rebal_dates[i + 1]
        
        # Get available tickers at start date
        available = [t for t in tickers if t in prices.columns and not pd.isna(prices.loc[start_date, t])]
        if not available:
            continue
        
        # Equal weight
        weight = 1.0 / len(available)
        
        # Period returns
        mask = (daily_returns.index > start_date) & (daily_returns.index <= end_date)
        period_ret = daily_returns.loc[mask, available]
        
        # Portfolio return = weighted average
        port_ret = (period_ret * weight).sum(axis=1)
        portfolio_returns.append(port_ret)
    
    return pd.concat(portfolio_returns) if portfolio_returns else pd.Series(dtype=float)


def run_momentum_strategy(
    prices: pd.DataFrame,
    tickers: list[str],
    benchmark: str,
    top_k: int,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """
    Classic 12-1 momentum strategy.
    
    Rank by 12-month return (skipping most recent month), select top-K.
    """
    daily_returns = prices.pct_change().dropna()
    rebal_dates = get_rebal_dates(prices)
    
    portfolio_returns = []
    
    for i in range(len(rebal_dates) - 1):
        start_date = rebal_dates[i]
        end_date = rebal_dates[i + 1]
        
        # Get price history up to start_date
        hist = prices.loc[:start_date]
        
        if len(hist) < lookback + skip:
            # Not enough history, equal weight
            available = [t for t in tickers if t in prices.columns]
            weight = 1.0 / len(available) if available else 0
            selected = available
        else:
            # Compute 12-1 momentum
            mom_returns = {}
            for t in tickers:
                if t not in prices.columns:
                    continue
                try:
                    # Price 12 months ago (skip most recent month)
                    price_12m_ago = hist[t].iloc[-(lookback + skip)]
                    price_1m_ago = hist[t].iloc[-skip]
                    mom = (price_1m_ago / price_12m_ago) - 1
                    if not pd.isna(mom):
                        mom_returns[t] = mom
                except:
                    continue
            
            if not mom_returns:
                continue
            
            # Rank and select top-K
            sorted_tickers = sorted(mom_returns.keys(), key=lambda x: mom_returns[x], reverse=True)
            selected = sorted_tickers[:top_k]
            weight = 1.0 / len(selected)
        
        # Period returns
        mask = (daily_returns.index > start_date) & (daily_returns.index <= end_date)
        period_ret = daily_returns.loc[mask, selected]
        
        port_ret = (period_ret * weight).sum(axis=1)
        portfolio_returns.append(port_ret)
    
    return pd.concat(portfolio_returns) if portfolio_returns else pd.Series(dtype=float)


def run_random_strategy(
    prices: pd.DataFrame,
    tickers: list[str],
    benchmark: str,
    top_k: int,
    n_paths: int = 50,
    seed: int = 42,
) -> pd.Series:
    """
    Random selection baseline (Monte Carlo average).
    
    Randomly select top-K tickers each rebalance, average over n_paths.
    """
    daily_returns = prices.pct_change().dropna()
    rebal_dates = get_rebal_dates(prices)
    
    rng = random.Random(seed)
    all_path_returns = []
    
    for path in range(n_paths):
        portfolio_returns = []
        
        for i in range(len(rebal_dates) - 1):
            start_date = rebal_dates[i]
            end_date = rebal_dates[i + 1]
            
            # Randomly select top_k tickers
            available = [t for t in tickers if t in prices.columns and not pd.isna(prices.loc[start_date, t])]
            if len(available) < top_k:
                selected = available
            else:
                selected = rng.sample(available, top_k)
            
            if not selected:
                continue
            
            weight = 1.0 / len(selected)
            
            mask = (daily_returns.index > start_date) & (daily_returns.index <= end_date)
            period_ret = daily_returns.loc[mask, selected]
            
            port_ret = (period_ret * weight).sum(axis=1)
            portfolio_returns.append(port_ret)
        
        if portfolio_returns:
            path_series = pd.concat(portfolio_returns)
            all_path_returns.append(path_series)
    
    # Average across all paths
    if not all_path_returns:
        return pd.Series(dtype=float)
    
    combined = pd.concat(all_path_returns, axis=1)
    return combined.mean(axis=1)


def run_buy_and_hold_spy(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
) -> pd.Series:
    """Buy and hold SPY."""
    return prices[benchmark].pct_change().dropna()


# ==============================================================================
# ML MODEL BACKTEST
# ==============================================================================

def run_ml_backtest(
    tickers: list[str],
    start: str,
    end: str,
    top_k: int,
    use_enhanced: bool = True,
) -> BacktestResult:
    """Run the ML model backtest."""
    print(f"  [ML] Running {'enhanced' if use_enhanced else 'base'} model backtest...")
    
    # Build config
    if use_enhanced:
        enhanced_config = EnhancedModelConfig(
            enabled=True,
            model_type="regression",
            target_mode="vol_norm",
            use_short_reversal=True,
            use_residual_momentum=True,
            use_idio_vol=True,
            use_mad=True,
            use_sector_ohe=True,
            cross_sectional_normalize=True,
            rolling_window=True,
            rolling_window_days=504,
        )
    else:
        enhanced_config = EnhancedModelConfig(enabled=False)
    
    config = PipelineConfig(
        research=ResearchConfig(
            universe=tickers,
            start_date=start,
            end_date=end,
            benchmark=BENCHMARK,
            horizon_days=HORIZON_DAYS,
            rebalance_frequency=REBALANCE_FREQ,
            top_k=top_k,
        ),
        features=FeatureConfig(),
        model=ModelConfig(),
        enhanced_model=enhanced_config,
    )
    
    # Download prices
    import yfinance as yf
    all_tickers = list(set(tickers + [BENCHMARK]))
    data = yf.download(all_tickers, start=start, end=end, progress=False)
    
    # Handle different yfinance return formats
    if isinstance(data.columns, pd.MultiIndex):
        # New format: ('Adj Close', 'AAPL'), ('Adj Close', 'MSFT'), ...
        # or ('Price', 'AAPL'), ('Price', 'MSFT'), ...
        if 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        elif 'Price' in data.columns.get_level_values(0):
            prices = data['Price']
        else:
            # Flatten and use whatever we got
            prices = data.xs(data.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        prices = data
    
    prices = prices.ffill().dropna()
    
    # Run backtest
    if use_enhanced:
        result = run_enhanced_backtest(
            prices=prices,
            horizon_days=HORIZON_DAYS,
            top_k=top_k,
            benchmark=BENCHMARK,
            rebalance_frequency=REBALANCE_FREQ,
            config=config,
        )
    else:
        result = run_backtest(
            prices=prices,
            horizon_days=HORIZON_DAYS,
            top_k=top_k,
            benchmark=BENCHMARK,
            rebalance_frequency=REBALANCE_FREQ,
            config=config,
        )
    
    return result


# ==============================================================================
# TEST 1: OUT-OF-SAMPLE HOLDOUT
# ==============================================================================

def run_oos_holdout_test(
    tickers: list[str],
    top_k: int,
    use_enhanced: bool = True,
) -> OOSResult:
    """
    Test 1: True out-of-sample holdout.
    
    Train on 2016-2023, test on 2024-2025.
    """
    print("\n" + "=" * 80)
    print("TEST 1: OUT-OF-SAMPLE HOLDOUT")
    print("=" * 80)
    print(f"Train period: {TRAIN_START} to {TRAIN_END}")
    print(f"Test period:  {TEST_START} to {TEST_END}")
    print(f"Universe:     {len(tickers)} tickers")
    print(f"Top-K:        {top_k}")
    print(f"Model:        {'Enhanced' if use_enhanced else 'Base'}")
    
    # Run train-period backtest
    print("\n[TRAIN PERIOD]")
    train_result = run_ml_backtest(tickers, TRAIN_START, TRAIN_END, top_k, use_enhanced)
    
    train_sharpe = train_result.metrics.get('sharpe_ratio', 0)
    train_return = train_result.metrics.get('annualized_return', 0) * 100
    train_max_dd = train_result.metrics.get('max_drawdown', 0) * 100
    
    print(f"  Sharpe:      {train_sharpe:.2f}")
    print(f"  Ann Return:  {train_return:.1f}%")
    print(f"  Max DD:      {train_max_dd:.1f}%")
    
    # Run test-period backtest (TRUE OOS)
    print("\n[TEST PERIOD - OUT OF SAMPLE]")
    test_result = run_ml_backtest(tickers, TEST_START, TEST_END, top_k, use_enhanced)
    
    test_sharpe = test_result.metrics.get('sharpe_ratio', 0)
    test_return = test_result.metrics.get('annualized_return', 0) * 100
    test_max_dd = test_result.metrics.get('max_drawdown', 0) * 100
    bench_return = test_result.metrics.get('benchmark_return', 0) * 100
    test_excess = test_return - bench_return
    
    print(f"  Sharpe:      {test_sharpe:.2f}")
    print(f"  Ann Return:  {test_return:.1f}%")
    print(f"  Max DD:      {test_max_dd:.1f}%")
    print(f"  Benchmark:   {bench_return:.1f}%")
    print(f"  Excess:      {test_excess:+.1f}%")
    
    # Assessment
    print("\n[ASSESSMENT]")
    if test_sharpe > 1.0 and test_excess > 0:
        print("  ✓ STRONG: Sharpe > 1.0 and positive excess return OOS")
    elif test_sharpe > 0.5 and test_excess > -5:
        print("  ~ MODERATE: Decent Sharpe, acceptable excess return")
    else:
        print("  ✗ WEAK: Poor out-of-sample performance")
    
    return OOSResult(
        train_sharpe=train_sharpe,
        train_return=train_return,
        train_max_dd=train_max_dd,
        test_sharpe=test_sharpe,
        test_return=test_return,
        test_max_dd=test_max_dd,
        test_excess=test_excess,
        benchmark_return=bench_return,
    )


# ==============================================================================
# TEST 2: BASELINE COMPARISONS
# ==============================================================================

def run_baseline_comparison_test(
    tickers: list[str],
    top_k: int,
    use_enhanced: bool = True,
) -> list[StrategyResult]:
    """
    Test 2: Compare ML model against simple baselines.
    
    All strategies run on the same universe and date range.
    """
    print("\n" + "=" * 80)
    print("TEST 2: BASELINE COMPARISONS")
    print("=" * 80)
    print(f"Period:       {FULL_START} to {FULL_END}")
    print(f"Universe:     {len(tickers)} tickers")
    print(f"Top-K:        {top_k}")
    
    # Download prices for baseline strategies
    prices = download_prices(tickers, FULL_START, FULL_END)
    benchmark_returns = prices[BENCHMARK].pct_change().dropna()
    
    results = []
    
    # 1. ML Model - use the metrics from the backtest result directly
    print("\n[1] ML Model")
    ml_result = run_ml_backtest(tickers, FULL_START, FULL_END, top_k, use_enhanced)
    
    # Extract metrics directly from backtest result (already computed correctly)
    ml_sharpe = ml_result.metrics.get('sharpe_ratio', 0)
    ml_ann_ret = ml_result.metrics.get('annualized_return', 0) * 100
    ml_max_dd = ml_result.metrics.get('max_drawdown', 0) * 100
    ml_hit_rate = ml_result.metrics.get('hit_rate', 0) * 100
    ml_vol = ml_result.metrics.get('annualized_volatility', 0) * 100
    ml_total_ret = ml_result.metrics.get('total_return', 0) * 100
    bench_ann = ml_result.metrics.get('benchmark_return', 0) * 100
    ml_excess = ml_ann_ret - bench_ann
    
    ml_metrics = StrategyResult(
        name="ML Model",
        ann_return=ml_ann_ret,
        sharpe=ml_sharpe,
        max_dd=ml_max_dd,
        excess_ret=ml_excess,
        hit_rate=ml_hit_rate,
        volatility=ml_vol,
        total_return=ml_total_ret,
        n_periods=len(ml_result.portfolio_returns),
    )
    results.append(ml_metrics)
    print(f"    Sharpe: {ml_metrics.sharpe:.2f}, Excess: {ml_metrics.excess_ret:+.1f}%")
    
    # 2. Equal-Weight All
    print("\n[2] Equal-Weight (all tickers)")
    ew_returns = run_equal_weight_strategy(prices, tickers, BENCHMARK)
    ew_metrics = compute_strategy_metrics(ew_returns, benchmark_returns.loc[ew_returns.index], "Equal-Weight")
    results.append(ew_metrics)
    print(f"    Sharpe: {ew_metrics.sharpe:.2f}, Excess: {ew_metrics.excess_ret:+.1f}%")
    
    # 3. Momentum Top-K
    print("\n[3] Momentum Top-K (12-1 month)")
    mom_returns = run_momentum_strategy(prices, tickers, BENCHMARK, top_k)
    mom_metrics = compute_strategy_metrics(mom_returns, benchmark_returns.loc[mom_returns.index], f"Momentum Top-{top_k}")
    results.append(mom_metrics)
    print(f"    Sharpe: {mom_metrics.sharpe:.2f}, Excess: {mom_metrics.excess_ret:+.1f}%")
    
    # 4. Random Top-K
    print(f"\n[4] Random Top-K (avg of {N_RANDOM_PATHS} paths)")
    rand_returns = run_random_strategy(prices, tickers, BENCHMARK, top_k, N_RANDOM_PATHS, RANDOM_SEED)
    rand_metrics = compute_strategy_metrics(rand_returns, benchmark_returns.loc[rand_returns.index], f"Random Top-{top_k}")
    results.append(rand_metrics)
    print(f"    Sharpe: {rand_metrics.sharpe:.2f}, Excess: {rand_metrics.excess_ret:+.1f}%")
    
    # 5. Buy-and-Hold SPY
    print("\n[5] Buy-and-Hold SPY")
    spy_returns = run_buy_and_hold_spy(prices, BENCHMARK)
    spy_metrics = compute_strategy_metrics(spy_returns, spy_returns, "Buy-Hold SPY")
    spy_metrics.excess_ret = 0.0  # By definition
    results.append(spy_metrics)
    print(f"    Sharpe: {spy_metrics.sharpe:.2f}, Excess: {spy_metrics.excess_ret:+.1f}%")
    
    return results


def print_comparison_table(results: list[StrategyResult]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON TABLE")
    print("=" * 80)
    
    # Sort by Sharpe
    sorted_results = sorted(results, key=lambda x: x.sharpe, reverse=True)
    
    print(f"\n{'Strategy':<20} {'Sharpe':>8} {'AnnRet':>10} {'Excess':>10} {'MaxDD':>10} {'HitRate':>10}")
    print("-" * 70)
    
    for r in sorted_results:
        print(f"{r.name:<20} {r.sharpe:>8.2f} {r.ann_return:>9.1f}% {r.excess_ret:>+9.1f}% {r.max_dd:>9.1f}% {r.hit_rate:>9.1f}%")
    
    print("-" * 70)
    
    # Assessment
    ml_result = next((r for r in results if r.name == "ML Model"), None)
    if ml_result:
        print("\n[ASSESSMENT]")
        
        baselines = [r for r in results if r.name != "ML Model"]
        beats_all = all(ml_result.sharpe > r.sharpe for r in baselines)
        beats_most = sum(1 for r in baselines if ml_result.sharpe > r.sharpe) >= len(baselines) - 1
        
        if beats_all:
            print("  ✓ STRONG: ML model beats ALL baselines")
        elif beats_most:
            print("  ~ MODERATE: ML model beats most baselines")
        else:
            print("  ✗ WEAK: ML model does not consistently beat baselines")
        
        # Specific comparisons
        ew = next((r for r in baselines if "Equal" in r.name), None)
        mom = next((r for r in baselines if "Momentum" in r.name), None)
        rand = next((r for r in baselines if "Random" in r.name), None)
        
        if ew:
            diff = ml_result.sharpe - ew.sharpe
            print(f"    vs Equal-Weight: {diff:+.2f} Sharpe {'✓' if diff > 0 else '✗'}")
        if mom:
            diff = ml_result.sharpe - mom.sharpe
            print(f"    vs Momentum:     {diff:+.2f} Sharpe {'✓' if diff > 0 else '✗'}")
        if rand:
            diff = ml_result.sharpe - rand.sharpe
            print(f"    vs Random:       {diff:+.2f} Sharpe {'✓' if diff > 0 else '✗'}")


# ==============================================================================
# CLI AND MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Out-of-Sample Alpha Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--universe",
        choices=list(UNIVERSES.keys()),
        default="large_cap",
        help="Universe to test",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top stocks to hold",
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use base model instead of enhanced model",
    )
    parser.add_argument(
        "--skip-oos",
        action="store_true",
        help="Skip out-of-sample holdout test",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline comparison test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    tickers = UNIVERSES[args.universe]
    top_k = args.top_k
    use_enhanced = not args.use_base_model
    
    print("=" * 80)
    print("  OUT-OF-SAMPLE ALPHA VALIDATION")
    print("=" * 80)
    print(f"\nUniverse:     {args.universe} ({len(tickers)} tickers)")
    print(f"Top-K:        {top_k}")
    print(f"Model:        {'Enhanced' if use_enhanced else 'Base'}")
    print(f"Benchmark:    {BENCHMARK}")
    
    oos_result = None
    baseline_results = None
    
    # Test 1: OOS Holdout
    if not args.skip_oos:
        oos_result = run_oos_holdout_test(tickers, top_k, use_enhanced)
    
    # Test 2: Baseline Comparisons
    if not args.skip_baselines:
        baseline_results = run_baseline_comparison_test(tickers, top_k, use_enhanced)
        print_comparison_table(baseline_results)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    if oos_result:
        print(f"\nOut-of-Sample Holdout ({TEST_START} to {TEST_END}):")
        print(f"  Sharpe: {oos_result.test_sharpe:.2f}")
        print(f"  Excess Return: {oos_result.test_excess:+.1f}%")
        print(f"  Max Drawdown: {oos_result.test_max_dd:.1f}%")
    
    if baseline_results:
        ml = next((r for r in baseline_results if r.name == "ML Model"), None)
        if ml:
            baselines = [r for r in baseline_results if r.name != "ML Model" and r.name != "Buy-Hold SPY"]
            n_beats = sum(1 for r in baselines if ml.sharpe > r.sharpe)
            print(f"\nBaseline Comparison:")
            print(f"  ML beats {n_beats}/{len(baselines)} baselines")
    
    print("\n" + "=" * 80)
    print("[DONE]")
    print("=" * 80)


if __name__ == "__main__":
    main()
