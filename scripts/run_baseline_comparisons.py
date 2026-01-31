#!/usr/bin/env python3
"""
Baseline Strategy Comparisons for Auto-Researcher.

This script compares the ML-based strategy against simple baseline strategies:
1. ML Strategy (GBDT with technical features)
2. Equal-Weight Big Tech (AAPL, MSFT, GOOGL, AMZN)
3. Simple Momentum Top-3 (12-month return ranking)
4. Random Top-3 (Monte Carlo simulation)

Run with:
    python scripts/run_baseline_comparisons.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
import pandas as pd
from typing import Callable

# ==============================================================================
# CONFIGURATION (Same as run_baseline_backtest.py)
# ==============================================================================

# Universe and date range
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
TRADEABLE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # Exclude benchmark
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"

# Backtest parameters
HORIZON_DAYS = 63  # ~3 months forward returns
REBALANCE_FREQ = "M"  # Monthly rebalancing
TOP_K = 3
BENCHMARK = "SPY"
MIN_TRAIN_PERIODS = 252  # 1 year minimum training data

# Monte Carlo parameters for random strategy
N_RANDOM_PATHS = 100
RANDOM_SEED = 42

# Configure logging (reduce noise)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# IMPORTS
# ==============================================================================

from auto_researcher.config import (
    PipelineConfig,
    ResearchConfig,
    FeatureConfig,
    ModelConfig,
)
from auto_researcher.data.price_loader import (
    download_price_history,
    get_adjusted_close,
)
from auto_researcher.backtest.runner import (
    run_backtest,
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
from auto_researcher.models.gbdt_model import GBDTModel


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class StrategyResult:
    """Container for strategy backtest results."""
    name: str
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    hit_rate: float
    max_drawdown: float
    ic: float | None = None  # Only for ML strategy


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_period_returns(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    tickers: list[str],
) -> pd.DataFrame:
    """
    Compute period returns between consecutive rebalance dates for each ticker.
    
    Returns DataFrame with rebalance dates as index, tickers as columns.
    """
    period_returns = []
    
    for i in range(len(rebal_dates) - 1):
        start_date = rebal_dates[i]
        end_date = rebal_dates[i + 1]
        
        start_prices = prices.loc[start_date, tickers]
        end_prices = prices.loc[end_date, tickers]
        period_ret = (end_prices / start_prices) - 1
        period_ret.name = end_date
        period_returns.append(period_ret)
    
    return pd.DataFrame(period_returns)


def run_simple_strategy(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    benchmark: str,
    get_weights: Callable[[pd.Timestamp, pd.DataFrame], dict[str, float]],
    strategy_name: str,
) -> StrategyResult:
    """
    Run a simple strategy with custom weight selection function.
    
    Args:
        prices: Price DataFrame with all tickers.
        rebal_dates: List of rebalance dates.
        benchmark: Benchmark ticker.
        get_weights: Function(date, prices_up_to_date) -> {ticker: weight}
        strategy_name: Name for the strategy.
    
    Returns:
        StrategyResult with computed metrics.
    """
    tradeable = [t for t in prices.columns if t != benchmark]
    
    portfolio_returns = []
    benchmark_returns = []
    
    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal = rebal_dates[i + 1]
        
        # Get historical prices up to rebal_date (for momentum calc)
        prices_hist = prices.loc[:rebal_date]
        
        # Get weights from strategy
        weights = get_weights(rebal_date, prices_hist)
        
        # Compute period returns
        start_prices = prices.loc[rebal_date]
        end_prices = prices.loc[next_rebal]
        
        # Portfolio return
        port_ret = sum(
            weights.get(t, 0) * ((end_prices[t] / start_prices[t]) - 1)
            for t in tradeable
        )
        portfolio_returns.append((next_rebal, port_ret))
        
        # Benchmark return
        bench_ret = (end_prices[benchmark] / start_prices[benchmark]) - 1
        benchmark_returns.append((next_rebal, bench_ret))
    
    # Convert to Series
    port_dates, port_rets = zip(*portfolio_returns)
    port_series = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates))
    
    bench_dates, bench_rets = zip(*benchmark_returns)
    bench_series = pd.Series(bench_rets, index=pd.DatetimeIndex(bench_dates))
    
    # Compute metrics (monthly data, so periods_per_year=12)
    periods_per_year = 12
    
    return StrategyResult(
        name=strategy_name,
        portfolio_returns=port_series,
        benchmark_returns=bench_series,
        cumulative_return=compute_total_return(port_series),
        annualized_return=compute_annualized_return(port_series, periods_per_year),
        annualized_volatility=compute_annualized_volatility(port_series, periods_per_year),
        sharpe_ratio=compute_sharpe_ratio(port_series, 0.0, periods_per_year),
        hit_rate=compute_hit_rate(port_series, bench_series),
        max_drawdown=compute_max_drawdown(port_series),
        ic=None,
    )


# ==============================================================================
# STRATEGY IMPLEMENTATIONS
# ==============================================================================

def run_ml_strategy(
    prices: pd.DataFrame,
    config: PipelineConfig,
) -> StrategyResult:
    """Run the ML (GBDT) strategy using existing backtest infrastructure."""
    print("  Running ML Strategy (GBDT)...")
    
    model = GBDTModel(config.model)
    
    result = run_backtest(
        prices=prices,
        horizon_days=config.research.horizon_days,
        top_k=config.research.top_k,
        model=model,
        benchmark=config.research.benchmark,
        rebalance_frequency=config.research.rebalance_frequency,
        min_train_periods=config.research.lookback_days,
        config=config,
    )
    
    # Extract IC
    if result.ic_history:
        avg_ic = np.mean([ic for _, ic in result.ic_history])
    else:
        avg_ic = None
    
    return StrategyResult(
        name="ML (GBDT)",
        portfolio_returns=result.portfolio_returns,
        benchmark_returns=result.benchmark_returns,
        cumulative_return=result.metrics.get("total_return", 0),
        annualized_return=result.metrics.get("annualized_return", 0),
        annualized_volatility=result.metrics.get("annualized_volatility", 0),
        sharpe_ratio=result.metrics.get("sharpe_ratio", 0),
        hit_rate=result.metrics.get("hit_rate", 0),
        max_drawdown=result.metrics.get("max_drawdown", 0),
        ic=avg_ic,
    )


def run_equal_weight_strategy(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    benchmark: str,
) -> StrategyResult:
    """Equal-weight all Big Tech stocks (AAPL, MSFT, GOOGL, AMZN)."""
    print("  Running Equal-Weight Big Tech...")
    
    tradeable = [t for t in prices.columns if t != benchmark]
    weight_per_stock = 1.0 / len(tradeable)
    
    def get_weights(date: pd.Timestamp, prices_hist: pd.DataFrame) -> dict[str, float]:
        return {t: weight_per_stock for t in tradeable}
    
    return run_simple_strategy(
        prices=prices,
        rebal_dates=rebal_dates,
        benchmark=benchmark,
        get_weights=get_weights,
        strategy_name="Equal-Weight Big Tech",
    )


def run_momentum_strategy(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    benchmark: str,
    lookback_days: int = 252,
    top_k: int = 3,
) -> StrategyResult:
    """Select top-k stocks by 12-month momentum, equal-weighted."""
    print("  Running Momentum Top-3...")
    
    tradeable = [t for t in prices.columns if t != benchmark]
    
    def get_weights(date: pd.Timestamp, prices_hist: pd.DataFrame) -> dict[str, float]:
        # Compute 12-month (252 trading days) returns
        if len(prices_hist) < lookback_days:
            # Not enough history, equal weight all
            return {t: 1.0 / len(tradeable) for t in tradeable}
        
        # Get prices lookback_days ago and today
        current_prices = prices_hist.iloc[-1][tradeable]
        past_prices = prices_hist.iloc[-lookback_days][tradeable]
        
        # Compute momentum (total return over lookback period)
        momentum = (current_prices / past_prices) - 1
        
        # Rank and pick top-k
        ranked = momentum.sort_values(ascending=False)
        top_tickers = ranked.head(top_k).index.tolist()
        
        # Equal weight top-k
        weight = 1.0 / top_k
        return {t: weight if t in top_tickers else 0.0 for t in tradeable}
    
    return run_simple_strategy(
        prices=prices,
        rebal_dates=rebal_dates,
        benchmark=benchmark,
        get_weights=get_weights,
        strategy_name="Momentum Top-3",
    )


def run_random_strategy(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    benchmark: str,
    top_k: int = 3,
    n_paths: int = 100,
    seed: int = 42,
) -> StrategyResult:
    """
    Randomly select top-k stocks each rebalance, run Monte Carlo simulation.
    
    Returns average metrics across all paths.
    """
    print(f"  Running Random Top-3 ({n_paths} Monte Carlo paths)...")
    
    tradeable = [t for t in prices.columns if t != benchmark]
    rng = np.random.default_rng(seed)
    
    all_results: list[StrategyResult] = []
    
    for path in range(n_paths):
        # Generate random selections for each rebalance date
        random_selections = {}
        for date in rebal_dates[:-1]:
            selected = rng.choice(tradeable, size=top_k, replace=False)
            random_selections[date] = set(selected)
        
        def get_weights(date: pd.Timestamp, prices_hist: pd.DataFrame) -> dict[str, float]:
            selected = random_selections.get(date, set())
            weight = 1.0 / top_k if len(selected) > 0 else 0.0
            return {t: weight if t in selected else 0.0 for t in tradeable}
        
        result = run_simple_strategy(
            prices=prices,
            rebal_dates=rebal_dates,
            benchmark=benchmark,
            get_weights=get_weights,
            strategy_name=f"Random Path {path}",
        )
        all_results.append(result)
    
    # Average metrics across all paths
    avg_cum_ret = np.mean([r.cumulative_return for r in all_results])
    avg_ann_ret = np.mean([r.annualized_return for r in all_results])
    avg_vol = np.mean([r.annualized_volatility for r in all_results])
    avg_sharpe = np.mean([r.sharpe_ratio for r in all_results])
    avg_hit_rate = np.mean([r.hit_rate for r in all_results])
    avg_max_dd = np.mean([r.max_drawdown for r in all_results])
    
    # Use the first path's returns for reference (we report averages anyway)
    return StrategyResult(
        name=f"Random Top-3 (n={n_paths})",
        portfolio_returns=all_results[0].portfolio_returns,
        benchmark_returns=all_results[0].benchmark_returns,
        cumulative_return=avg_cum_ret,
        annualized_return=avg_ann_ret,
        annualized_volatility=avg_vol,
        sharpe_ratio=avg_sharpe,
        hit_rate=avg_hit_rate,
        max_drawdown=avg_max_dd,
        ic=None,
    )


# ==============================================================================
# MAIN
# ==============================================================================

def print_comparison_table(results: list[StrategyResult], benchmark_return: float) -> None:
    """Print a formatted comparison table of all strategies."""
    
    print("\n" + "=" * 100)
    print("  STRATEGY COMPARISON")
    print("=" * 100)
    
    # Header
    header = (
        f"{'Strategy':<25} | {'Cum.Ret':>10} | {'Ann.Ret':>10} | "
        f"{'Vol':>10} | {'Sharpe':>8} | {'Hit Rate':>10} | {'Max DD':>10} | {'IC':>8}"
    )
    print(header)
    print("-" * 100)
    
    # Benchmark row
    print(
        f"{'SPY (Benchmark)':<25} | "
        f"{'-':>10} | "
        f"{benchmark_return * 100:>9.2f}% | "
        f"{'-':>10} | "
        f"{'-':>8} | "
        f"{'-':>10} | "
        f"{'-':>10} | "
        f"{'-':>8}"
    )
    print("-" * 100)
    
    # Strategy rows
    for r in results:
        ic_str = f"{r.ic:.4f}" if r.ic is not None else "-"
        print(
            f"{r.name:<25} | "
            f"{r.cumulative_return * 100:>9.2f}% | "
            f"{r.annualized_return * 100:>9.2f}% | "
            f"{r.annualized_volatility * 100:>9.2f}% | "
            f"{r.sharpe_ratio:>8.3f} | "
            f"{r.hit_rate * 100:>9.2f}% | "
            f"{r.max_drawdown * 100:>9.2f}% | "
            f"{ic_str:>8}"
        )
    
    print("=" * 100)


def main():
    print("\n" + "=" * 70)
    print("  BASELINE STRATEGY COMPARISON")
    print("  Auto-Researcher for Asset Selection")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading price data...")
    
    raw_prices = download_price_history(
        tickers=UNIVERSE,
        start=START_DATE,
        end=END_DATE,
    )
    prices = get_adjusted_close(raw_prices)
    print(f"      Loaded {len(prices)} trading days for {len(UNIVERSE)} tickers")
    
    # -------------------------------------------------------------------------
    # Setup config
    # -------------------------------------------------------------------------
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
        ),
    )
    
    # Get rebalance dates
    all_dates = prices.index
    rebal_dates = get_rebalance_dates(all_dates, REBALANCE_FREQ)
    
    # Filter to dates with enough history
    min_start_idx = MIN_TRAIN_PERIODS + HORIZON_DAYS
    min_start_date = all_dates[min_start_idx]
    rebal_dates = [d for d in rebal_dates if d >= min_start_date]
    
    print(f"      {len(rebal_dates)} rebalance dates from {rebal_dates[0].date()} to {rebal_dates[-1].date()}")
    
    # -------------------------------------------------------------------------
    # Run strategies
    # -------------------------------------------------------------------------
    print("\n[2/5] Running strategies...")
    
    results: list[StrategyResult] = []
    
    # 1. ML Strategy
    ml_result = run_ml_strategy(prices, config)
    results.append(ml_result)
    
    # 2. Equal-Weight Big Tech
    ew_result = run_equal_weight_strategy(prices, rebal_dates, BENCHMARK)
    results.append(ew_result)
    
    # 3. Momentum Top-3
    mom_result = run_momentum_strategy(
        prices, rebal_dates, BENCHMARK, 
        lookback_days=252, top_k=TOP_K
    )
    results.append(mom_result)
    
    # 4. Random Top-3
    rand_result = run_random_strategy(
        prices, rebal_dates, BENCHMARK,
        top_k=TOP_K, n_paths=N_RANDOM_PATHS, seed=RANDOM_SEED
    )
    results.append(rand_result)
    
    # -------------------------------------------------------------------------
    # Compute benchmark return for reference
    # -------------------------------------------------------------------------
    print("\n[3/5] Computing benchmark metrics...")
    
    # Benchmark period returns (same as portfolio periods)
    bench_period_returns = []
    for i in range(len(rebal_dates) - 1):
        start_price = prices.loc[rebal_dates[i], BENCHMARK]
        end_price = prices.loc[rebal_dates[i + 1], BENCHMARK]
        bench_period_returns.append((end_price / start_price) - 1)
    
    bench_series = pd.Series(bench_period_returns, index=pd.DatetimeIndex(rebal_dates[1:]))
    benchmark_ann_return = compute_annualized_return(bench_series, periods_per_year=12)
    
    print(f"      SPY Annualized Return: {benchmark_ann_return * 100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Print comparison table
    # -------------------------------------------------------------------------
    print("\n[4/5] Results...")
    print_comparison_table(results, benchmark_ann_return)
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    print("\n[5/5] Analysis")
    print("-" * 70)
    
    # Find best strategy by Sharpe
    best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
    print(f"  Best Sharpe Ratio: {best_sharpe.name} ({best_sharpe.sharpe_ratio:.3f})")
    
    # Compare ML vs Random (statistical significance)
    ml_sharpe = results[0].sharpe_ratio  # ML is first
    rand_sharpe = rand_result.sharpe_ratio
    sharpe_diff = ml_sharpe - rand_sharpe
    print(f"  ML vs Random Sharpe Delta: {sharpe_diff:+.3f}")
    
    # Hit rate interpretation
    print("\n  Hit Rate Interpretation:")
    for r in results:
        if r.hit_rate > 0.55:
            interpretation = "Above chance (possible edge)"
        elif r.hit_rate < 0.45:
            interpretation = "Below chance (underperforming)"
        else:
            interpretation = "Near chance (~50%)"
        print(f"    {r.name}: {r.hit_rate * 100:.1f}% - {interpretation}")
    
    # IC interpretation (ML only)
    if ml_result.ic is not None:
        print(f"\n  ML Strategy IC: {ml_result.ic:.4f}")
        if ml_result.ic > 0.05:
            print("    Note: IC > 0.05 is good for cross-sectional ranking")
        elif ml_result.ic > 0.02:
            print("    Note: IC 0.02-0.05 is typical for technical factors")
        else:
            print("    Note: IC < 0.02 suggests weak predictive signal")
    
    print("\n" + "=" * 70)
    print("  Comparison complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
