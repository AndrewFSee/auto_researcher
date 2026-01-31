#!/usr/bin/env python3
"""
Large-Cap Universe Backtest for Auto-Researcher.

This script tests whether the GBDT model generalizes beyond the tiny
4-stock tech universe to a broader ~50 stock large-cap universe.

It compares the ML strategy against simple baselines:
1. ML Strategy (GBDT with technical features)
2. Equal-Weight Universe (all 50 stocks)
3. Simple 12M Momentum Top-K
4. Random Top-K (Monte Carlo)

Run with:
    # Technical-only ML (default)
    python scripts/run_large_cap_backtest.py

    # Tech + Fundamentals comparison mode
    python scripts/run_large_cap_backtest.py --use-fundamentals --fundamentals-csv data/fundamentals.csv
"""

import sys
import argparse
import itertools
from pathlib import Path
from dataclasses import dataclass, replace, field
from typing import Callable, NamedTuple, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Date range - longer history for larger universe
START_DATE = "2012-01-01"  # 2012 to avoid some early data gaps
END_DATE = "2024-01-01"

# Backtest parameters
HORIZON_DAYS = 63  # ~3 months forward returns
REBALANCE_FREQ = "M"  # Monthly rebalancing
TOP_K = 15  # Top 15 out of ~50 stocks
BENCHMARK = "SPY"
MIN_TRAIN_PERIODS = 252  # 1 year minimum training data

# Monte Carlo parameters for random strategy
N_RANDOM_PATHS = 50
RANDOM_SEED = 42

# ==============================================================================
# TRANSACTION COST CONFIGURATION
# ==============================================================================

# Cost model: 10 bps per side (buy or sell), applied to all strategies
# Total round-trip cost = turnover * 10 bps * 2 = 20 bps per 100% turnover
# Note: This is imported after metrics import to avoid circular dependency
COST_MODEL = None  # Initialized after imports

# ==============================================================================
# SUBPERIOD CONFIGURATION
# ==============================================================================

class Subperiod(NamedTuple):
    """Define a subperiod for performance analysis."""
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


# Default subperiods aligned with market regimes
SUBPERIODS = [
    Subperiod(name="2013-2016", start=pd.Timestamp("2013-01-01"), end=pd.Timestamp("2016-12-31")),
    Subperiod(name="2017-2019", start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2019-12-31")),
    Subperiod(name="2020-2023", start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2023-12-31")),
]

# ==============================================================================
# OUT-OF-SAMPLE (OOS) CONFIGURATION
# ==============================================================================

# OOS train/test split for robustness analysis
OOS_TRAIN_START = pd.Timestamp("2013-01-01")
OOS_TRAIN_END = pd.Timestamp("2018-12-31")
OOS_TEST_START = pd.Timestamp("2019-01-01")
OOS_TEST_END = pd.Timestamp("2023-12-31")

# Subperiods within the training window for feature selection
OOS_TRAIN_SUBPERIODS = [
    Subperiod(name="2013-2014", start=pd.Timestamp("2013-01-01"), end=pd.Timestamp("2014-12-31")),
    Subperiod(name="2015-2016", start=pd.Timestamp("2015-01-01"), end=pd.Timestamp("2016-12-31")),
    Subperiod(name="2017-2018", start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2018-12-31")),
]


# ==============================================================================
# MULTI-SPLIT OOS ROBUSTNESS SUITE
# ==============================================================================

@dataclass
class OOSRobustnessSplit:
    """Definition of a single train/test split for OOS robustness testing."""
    
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    
    @property
    def train_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_start)
    
    @property
    def train_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_end)
    
    @property
    def test_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_start)
    
    @property
    def test_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_end)


@dataclass
class CoreOOSRobustnessConfig:
    """Configuration for the multi-split OOS robustness suite."""
    
    enable: bool = False
    splits: list[OOSRobustnessSplit] = field(default_factory=list)


# Default OOS splits for Core Factors [Simple] robustness testing
DEFAULT_CORE_OOS_SPLITS = [
    # 1) Train early, test mid
    OOSRobustnessSplit(
        name="train_2013_2016_test_2017_2019",
        train_start="2013-01-01",
        train_end="2016-12-31",
        test_start="2017-01-01",
        test_end="2019-12-31",
    ),
    # 2) Train mid, test late (stress on COVID regime)
    OOSRobustnessSplit(
        name="train_2017_2019_test_2020_2023",
        train_start="2017-01-01",
        train_end="2019-12-31",
        test_start="2020-01-01",
        test_end="2023-12-31",
    ),
    # 3) Train early+mid, test late (main split)
    OOSRobustnessSplit(
        name="train_2013_2019_test_2020_2023",
        train_start="2013-01-01",
        train_end="2019-12-31",
        test_start="2020-01-01",
        test_end="2023-12-31",
    ),
]

# Best regularized model params (from regularization sweep)
BEST_CORE_SIMPLE_PARAMS = {
    "max_depth": 3,
    "n_estimators": 75,
    "reg_lambda": 0.0,
    "subsample": 0.60,
    "colsample_bytree": 0.80,
}


# Configure logging (reduce noise during data download)
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
    SIMPLE_MODEL_CONFIG,
    RegularizationSweepConfig,
    DEFAULT_REG_SWEEP_CONFIG,
    build_model_config_from_params,
)
from auto_researcher.data.universe import get_large_cap_universe
from auto_researcher.data.price_loader import (
    download_price_history,
    get_adjusted_close,
)
from auto_researcher.backtest.runner import (
    run_backtest,
    get_rebalance_dates,
    BacktestResult,
    run_regime_switch_strategy,
    run_ic_weighted_ensemble,
)
from auto_researcher.backtest.portfolio import (
    build_equal_weight_portfolio,
    compute_portfolio_return,
)
from auto_researcher.backtest.metrics import (
    compute_cumulative_returns,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_hit_rate,
    compute_total_return,
    compute_max_drawdown,
    compute_ir,
    compute_ic,
    ICWeightConfig,
    CostModelConfig,
    compute_turnover_series,
    compute_net_returns,
    compute_net_metrics,
)
from auto_researcher.backtest.regime_signal import (
    compute_regime_series,
    compute_regime_summary,
    MarketRegime,
    RegimeConfig,
)
from auto_researcher.models.gbdt_model import GBDTModel
from auto_researcher.models.regimes import (
    RegimeMode,
    assign_regime,
    add_regime_feature,
    split_data_by_regime,
    select_model_for_regime,
    get_regime_aware_features,
    get_regime_strategy_suffix,
    REGIME_LABELS,
)
from auto_researcher.features.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_matrix,
    prepare_training_data,
    get_features_for_date,
)
from auto_researcher.features.fundamentals import FundamentalFactorConfig
from auto_researcher.features.coverage import (
    compute_fundamentals_coverage,
    select_good_fundamentals_tickers,
    identify_fundamental_columns,
    get_tech_only_tickers,
    mask_fundamentals_for_tech_only_tickers,
    get_excluded_tickers,
    format_coverage_summary,
    FUNDAMENTAL_PREFIXES,
)


# Initialize COST_MODEL after imports
COST_MODEL = CostModelConfig(
    bps_per_side=10.0,
    apply_to_all_strategies=True,
)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class FundamentalsCoverageState:
    """
    Tracks the state of fundamentals coverage filtering.
    
    Used to pass coverage information between functions and for final reporting.
    """
    mode: str = "full"  # "full", "good_only", or "hybrid"
    min_coverage: float = 0.0
    min_test_year: int | None = None
    coverage_matrix: pd.DataFrame | None = None
    good_tickers: list[str] = field(default_factory=list)
    excluded_tickers: list[str] = field(default_factory=list)
    tech_only_tickers: list[str] = field(default_factory=list)
    n_fundamentals: int = 0
    n_tech_only: int = 0


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
    information_ratio: float
    ic: float | None = None  # Only for ML strategy
    ic_std: float | None = None
    ic_median: float | None = None
    ic_series: pd.Series | None = None  # Full IC history for subperiod analysis
    # Transaction cost fields
    avg_turnover: float | None = None
    p95_turnover: float | None = None
    net_annualized_return: float | None = None
    net_sharpe_ratio: float | None = None
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]] | None = None


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def run_simple_strategy(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    benchmark: str,
    get_weights: Callable[[pd.Timestamp, pd.DataFrame], dict[str, float]],
    strategy_name: str,
    cost_config: CostModelConfig | None = None,
) -> StrategyResult:
    """
    Run a simple strategy with custom weight selection function.
    """
    tradeable = [t for t in prices.columns if t != benchmark]
    
    portfolio_returns = []
    benchmark_returns = []
    weights_history = []
    
    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal = rebal_dates[i + 1]
        
        # Get historical prices up to rebal_date
        prices_hist = prices.loc[:rebal_date]
        
        # Get weights from strategy
        weights = get_weights(rebal_date, prices_hist)
        weights_history.append((rebal_date, weights))
        
        # Compute period returns
        start_prices = prices.loc[rebal_date]
        end_prices = prices.loc[next_rebal]
        
        # Portfolio return
        port_ret = sum(
            weights.get(t, 0) * ((end_prices[t] / start_prices[t]) - 1)
            for t in tradeable
            if t in weights and not pd.isna(end_prices.get(t)) and not pd.isna(start_prices.get(t))
        )
        portfolio_returns.append((next_rebal, port_ret))
        
        # Benchmark return
        bench_ret = (end_prices[benchmark] / start_prices[benchmark]) - 1
        benchmark_returns.append((next_rebal, bench_ret))
    
    # Convert to Series
    if not portfolio_returns:
        return None
    
    port_dates, port_rets = zip(*portfolio_returns)
    port_series = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates))
    
    bench_dates, bench_rets = zip(*benchmark_returns)
    bench_series = pd.Series(bench_rets, index=pd.DatetimeIndex(bench_dates))
    
    # Compute metrics (monthly data)
    periods_per_year = 12
    
    # Compute turnover and net metrics
    avg_turnover = None
    p95_turnover = None
    net_ann_ret = None
    net_sharpe = None
    
    if cost_config is not None and len(weights_history) > 0:
        turnover_raw = compute_turnover_series(weights_history, include_initial_build=False)
        
        if len(turnover_raw) > 0 and len(port_series) > 0:
            # Align turnover to returns
            turnover_aligned = pd.Series(
                turnover_raw.iloc[1:].values,
                index=port_series.index[:len(turnover_raw) - 1],
                name="turnover"
            )
            
            if len(turnover_aligned) < len(port_series):
                missing_dates = port_series.index[len(turnover_aligned):]
                extra = pd.Series(np.nan, index=missing_dates, name="turnover")
                turnover_aligned = pd.concat([turnover_aligned, extra])
            
            valid_turnover = turnover_aligned.dropna()
            if len(valid_turnover) > 0:
                avg_turnover = float(valid_turnover.mean())
                p95_turnover = float(np.percentile(valid_turnover, 95))
            
            net_returns = compute_net_returns(port_series, turnover_aligned, cost_config)
            net_metrics = compute_net_metrics(net_returns, bench_series, periods_per_year)
            net_ann_ret = net_metrics.get("net_ann_ret")
            net_sharpe = net_metrics.get("net_sharpe")
    
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
        information_ratio=compute_ir(port_series, bench_series, periods_per_year),
        ic=None,
        avg_turnover=avg_turnover,
        p95_turnover=p95_turnover,
        net_annualized_return=net_ann_ret,
        net_sharpe_ratio=net_sharpe,
        weights_history=weights_history,
    )


# ==============================================================================
# FUNDAMENTALS COVERAGE FILTERING
# ==============================================================================

def compute_coverage_and_filter(
    prices: pd.DataFrame,
    tradeable_tickers: list[str],
    args,
    benchmark: str = "SPY",
) -> FundamentalsCoverageState:
    """
    Compute fundamentals coverage and filter tickers based on coverage mode.
    
    This function builds a temporary feature matrix with fundamentals enabled
    to compute coverage, then determines which tickers to use based on the
    --fundamentals-universe-mode setting.
    
    Args:
        prices: Price DataFrame (adjusted close).
        tradeable_tickers: List of tradeable tickers (excluding benchmark).
        args: Parsed command-line arguments.
        benchmark: Benchmark ticker symbol.
    
    Returns:
        FundamentalsCoverageState with coverage info and filtered ticker lists.
    """
    state = FundamentalsCoverageState(
        mode=args.fundamentals_universe_mode,
        min_coverage=args.min_fundamentals_coverage,
        min_test_year=args.min_fundamentals_test_year,
    )
    
    # If mode is "full" and no coverage threshold, skip computation
    if state.mode == "full" and state.min_coverage == 0.0:
        state.good_tickers = tradeable_tickers.copy()
        state.n_fundamentals = len(tradeable_tickers)
        state.n_tech_only = 0
        return state
    
    print("\n  Computing fundamentals coverage...")
    
    # Determine available factor families based on source
    if args.fundamentals_source == "finagg":
        available_factor_families = {
            'value': False,
            'quality': True,
            'growth': True,
            'profitability': True,
            'size': False,
        }
    else:
        available_factor_families = {
            'value': False,
            'quality': False,
            'growth': False,
            'profitability': True,
            'size': False,
        }
    
    # Build temporary feature config with fundamentals
    temp_feature_config = FeatureConfig(
        momentum_windows=(21, 63, 126, 252),
        volatility_windows=(21, 63),
        include_technical=False,  # Only need fundamentals for coverage
        include_fundamentals=True,
        include_sentiment=False,
        fundamentals_csv_path=args.fundamentals_csv,
        fundamental_factor_families=available_factor_families,
        fundamentals_source=args.fundamentals_source,
        finagg_mode=args.finagg_mode,
    )
    
    try:
        # Build feature matrix with fundamentals only
        feature_df = build_feature_matrix(prices, temp_feature_config)
        
        # Identify fundamental columns
        fundamental_cols = identify_fundamental_columns(feature_df)
        
        if not fundamental_cols:
            logger.warning("No fundamental columns found. Using full universe.")
            state.good_tickers = tradeable_tickers.copy()
            state.n_fundamentals = len(tradeable_tickers)
            return state
        
        # Compute coverage matrix
        coverage = compute_fundamentals_coverage(
            feature_df,
            fundamental_cols=fundamental_cols,
            coverage_freq="Y",
            benchmark=benchmark,
        )
        state.coverage_matrix = coverage
        
        print(f"    Found {len(fundamental_cols)} fundamental columns")
        print(f"    Coverage computed for {len(coverage)} tickers over {len(coverage.columns)} years")
        
    except Exception as e:
        logger.warning(f"Failed to compute fundamentals coverage: {e}")
        state.good_tickers = tradeable_tickers.copy()
        state.n_fundamentals = len(tradeable_tickers)
        return state
    
    # Apply filtering based on mode
    if state.mode == "full":
        # Full mode: use all tickers, but compute stats for reporting
        state.good_tickers = tradeable_tickers.copy()
        state.tech_only_tickers = []
        state.n_fundamentals = len(tradeable_tickers)
        state.n_tech_only = 0
        
    elif state.mode == "good_only":
        # Good-only mode: restrict to tickers passing coverage filter
        try:
            state.good_tickers = select_good_fundamentals_tickers(
                coverage,
                min_avg_coverage=state.min_coverage,
                min_test_year=state.min_test_year,
            )
            state.excluded_tickers = get_excluded_tickers(tradeable_tickers, state.good_tickers)
            state.n_fundamentals = len(state.good_tickers)
            state.n_tech_only = 0
            
            print(f"    Tickers passing coverage filter: {len(state.good_tickers)}/{len(tradeable_tickers)}")
            
        except ValueError as e:
            # Zero tickers pass filter - fail fast with descriptive error
            print(f"\n  [Error] {e}")
            raise
            
    elif state.mode == "hybrid":
        # Hybrid mode: keep all tickers, but mark low-coverage as tech-only
        state.good_tickers = tradeable_tickers.copy()
        state.tech_only_tickers = get_tech_only_tickers(coverage, state.min_coverage)
        state.n_tech_only = len(state.tech_only_tickers)
        state.n_fundamentals = len(tradeable_tickers) - state.n_tech_only
        
        print(f"    Tickers with fundamentals: {state.n_fundamentals}")
        print(f"    Tickers tech-only: {state.n_tech_only}")
    
    return state


def print_coverage_report(state: FundamentalsCoverageState) -> None:
    """Print the fundamentals coverage summary report."""
    report = format_coverage_summary(
        mode=state.mode,
        min_coverage=state.min_coverage,
        n_fundamentals=state.n_fundamentals,
        n_tech_only=state.n_tech_only,
        good_tickers=state.good_tickers if state.mode == "good_only" else None,
        excluded_tickers=state.excluded_tickers if state.mode == "good_only" else None,
    )
    print(report)


# ==============================================================================
# STRATEGY IMPLEMENTATIONS
# ==============================================================================
def run_ml_strategy(
    prices: pd.DataFrame,
    config: PipelineConfig,
    tradeable_tickers: list[str],
    pipeline_config: FeaturePipelineConfig | None = None,
    label: str = "ML (GBDT)",
    regime_mode: str | RegimeMode = "none",
    cost_config: CostModelConfig | None = None,
) -> tuple[StrategyResult, GBDTModel | None, BacktestResult | None]:
    """
    Run the ML (GBDT) strategy using existing backtest infrastructure.
    
    Args:
        prices: Price data.
        config: Pipeline configuration (includes FeatureConfig with fundamentals settings).
        tradeable_tickers: List of tradeable tickers.
        pipeline_config: Deprecated - use config.features instead.
        label: Label for the strategy result.
        regime_mode: Regime-aware ML mode: "none", "feature", or "split".
        cost_config: Transaction cost configuration. If None, no costs applied.
    
    Returns:
        Tuple of (StrategyResult, trained_model, BacktestResult).
        - BacktestResult contains scores_history for regime-switch ensemble.
    """
    # Normalize and append suffix to label if regime-aware
    if isinstance(regime_mode, str):
        regime_mode_enum = RegimeMode(regime_mode)
    else:
        regime_mode_enum = regime_mode
    
    suffix = get_regime_strategy_suffix(regime_mode_enum)
    strategy_label = label + suffix
    
    print(f"  Running {strategy_label}...")
    
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
        regime_mode=regime_mode,
        cost_config=cost_config,
    )
    
    # Extract IC statistics and series
    ic_series = None
    if result.ic_history:
        ic_dates, ic_values = zip(*result.ic_history)
        ic_series = pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates))
        avg_ic = ic_series.mean()
        ic_std = ic_series.std()
        ic_median = ic_series.median()
    else:
        avg_ic = ic_std = ic_median = None
    
    # Extract net metrics
    net_ann_ret = result.net_metrics.get("net_ann_ret") if result.net_metrics else None
    net_sharpe = result.net_metrics.get("net_sharpe") if result.net_metrics else None
    
    return StrategyResult(
        name=strategy_label,
        portfolio_returns=result.portfolio_returns,
        benchmark_returns=result.benchmark_returns,
        cumulative_return=result.metrics.get("total_return", 0),
        annualized_return=result.metrics.get("annualized_return", 0),
        annualized_volatility=result.metrics.get("annualized_volatility", 0),
        sharpe_ratio=result.metrics.get("sharpe_ratio", 0),
        hit_rate=result.metrics.get("hit_rate", 0),
        max_drawdown=result.metrics.get("max_drawdown", 0),
        information_ratio=result.metrics.get("information_ratio", 0),
        ic=avg_ic,
        ic_std=ic_std,
        ic_median=ic_median,
        ic_series=ic_series,
        avg_turnover=result.avg_turnover,
        p95_turnover=result.p95_turnover,
        net_annualized_return=net_ann_ret,
        net_sharpe_ratio=net_sharpe,
        weights_history=result.weights_history,
    ), model, result


def run_equal_weight_strategy(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    benchmark: str,
    tradeable_tickers: list[str],
) -> StrategyResult:
    """Equal-weight all stocks in the universe."""
    print("  Running Equal-Weight Universe...")
    
    n_stocks = len(tradeable_tickers)
    weight_per_stock = 1.0 / n_stocks
    
    def get_weights(date: pd.Timestamp, prices_hist: pd.DataFrame) -> dict[str, float]:
        # Only include stocks with valid price on this date
        valid_tickers = [
            t for t in tradeable_tickers 
            if t in prices_hist.columns and not pd.isna(prices_hist[t].iloc[-1])
        ]
        if not valid_tickers:
            return {}
        weight = 1.0 / len(valid_tickers)
        return {t: weight for t in valid_tickers}
    
    return run_simple_strategy(
        prices=prices,
        rebal_dates=rebal_dates,
        benchmark=benchmark,
        get_weights=get_weights,
        strategy_name="Equal-Weight Universe",
        cost_config=COST_MODEL,
    )


def run_momentum_strategy(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    benchmark: str,
    tradeable_tickers: list[str],
    lookback_days: int = 252,
    top_k: int = 15,
) -> StrategyResult:
    """Select top-k stocks by 12-month momentum, equal-weighted."""
    print("  Running 12M Momentum Top-K...")
    
    def get_weights(date: pd.Timestamp, prices_hist: pd.DataFrame) -> dict[str, float]:
        # Compute 12-month returns
        if len(prices_hist) < lookback_days:
            # Not enough history, equal weight all
            valid_tickers = [t for t in tradeable_tickers if t in prices_hist.columns]
            return {t: 1.0 / len(valid_tickers) for t in valid_tickers}
        
        # Get valid tickers with price data
        valid_tickers = [
            t for t in tradeable_tickers 
            if t in prices_hist.columns 
            and not pd.isna(prices_hist[t].iloc[-1])
            and not pd.isna(prices_hist[t].iloc[-lookback_days])
        ]
        
        if len(valid_tickers) == 0:
            return {}
        
        # Compute momentum
        current_prices = prices_hist[valid_tickers].iloc[-1]
        past_prices = prices_hist[valid_tickers].iloc[-lookback_days]
        momentum = (current_prices / past_prices) - 1
        
        # Rank and pick top-k
        ranked = momentum.sort_values(ascending=False)
        selected = ranked.head(min(top_k, len(ranked))).index.tolist()
        
        # Equal weight top-k
        weight = 1.0 / len(selected)
        return {t: weight if t in selected else 0.0 for t in valid_tickers}
    
    return run_simple_strategy(
        prices=prices,
        rebal_dates=rebal_dates,
        benchmark=benchmark,
        get_weights=get_weights,
        strategy_name=f"12M Momentum Top-{top_k}",
        cost_config=COST_MODEL,
    )


def run_random_strategy(
    prices: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    benchmark: str,
    tradeable_tickers: list[str],
    top_k: int = 15,
    n_paths: int = 50,
    seed: int = 42,
) -> StrategyResult:
    """
    Randomly select top-k stocks each rebalance, run Monte Carlo simulation.
    """
    print(f"  Running Random Top-K ({n_paths} Monte Carlo paths)...")
    
    rng = np.random.default_rng(seed)
    
    all_results: list[StrategyResult] = []
    
    for path in range(n_paths):
        # Generate random selections for each rebalance date
        random_selections = {}
        for date in rebal_dates[:-1]:
            # Get valid tickers for this date
            if date in prices.index:
                valid = [t for t in tradeable_tickers if t in prices.columns and not pd.isna(prices.loc[date, t])]
                if len(valid) >= top_k:
                    selected = rng.choice(valid, size=top_k, replace=False)
                else:
                    selected = valid
                random_selections[date] = set(selected)
            else:
                random_selections[date] = set()
        
        def get_weights(date: pd.Timestamp, prices_hist: pd.DataFrame) -> dict[str, float]:
            selected = random_selections.get(date, set())
            if not selected:
                return {}
            weight = 1.0 / len(selected)
            return {t: weight if t in selected else 0.0 for t in tradeable_tickers}
        
        result = run_simple_strategy(
            prices=prices,
            rebal_dates=rebal_dates,
            benchmark=benchmark,
            get_weights=get_weights,
            strategy_name=f"Random Path {path}",
            cost_config=COST_MODEL,
        )
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        return None
    
    # Average metrics across all paths
    avg_cum_ret = np.mean([r.cumulative_return for r in all_results])
    avg_ann_ret = np.mean([r.annualized_return for r in all_results])
    avg_vol = np.mean([r.annualized_volatility for r in all_results])
    avg_sharpe = np.mean([r.sharpe_ratio for r in all_results])
    avg_hit_rate = np.mean([r.hit_rate for r in all_results])
    avg_max_dd = np.mean([r.max_drawdown for r in all_results])
    avg_ir = np.mean([r.information_ratio for r in all_results])
    
    # Average turnover/net metrics across paths
    avg_turnover = np.mean([r.avg_turnover for r in all_results if r.avg_turnover is not None]) if any(r.avg_turnover is not None for r in all_results) else None
    avg_p95_turnover = np.mean([r.p95_turnover for r in all_results if r.p95_turnover is not None]) if any(r.p95_turnover is not None for r in all_results) else None
    avg_net_ann_ret = np.mean([r.net_annualized_return for r in all_results if r.net_annualized_return is not None]) if any(r.net_annualized_return is not None for r in all_results) else None
    avg_net_sharpe = np.mean([r.net_sharpe_ratio for r in all_results if r.net_sharpe_ratio is not None]) if any(r.net_sharpe_ratio is not None for r in all_results) else None
    
    return StrategyResult(
        name=f"Random Top-{top_k} (n={n_paths})",
        portfolio_returns=all_results[0].portfolio_returns,
        benchmark_returns=all_results[0].benchmark_returns,
        cumulative_return=avg_cum_ret,
        annualized_return=avg_ann_ret,
        annualized_volatility=avg_vol,
        sharpe_ratio=avg_sharpe,
        hit_rate=avg_hit_rate,
        max_drawdown=avg_max_dd,
        information_ratio=avg_ir,
        ic=None,
        avg_turnover=avg_turnover,
        p95_turnover=avg_p95_turnover,
        net_annualized_return=avg_net_ann_ret,
        net_sharpe_ratio=avg_net_sharpe,
    )


# ==============================================================================
# OUTPUT FORMATTING
# ==============================================================================

def print_comparison_table(results: list[StrategyResult], benchmark_return: float) -> None:
    """Print a formatted comparison table of all strategies with gross and net metrics."""
    
    table_width = 160
    print("\n" + "=" * table_width)
    print("  STRATEGY COMPARISON - LARGE CAP UNIVERSE (with Transaction Costs)")
    print("=" * table_width)
    
    # Header - now includes gross and net metrics, plus turnover
    header = (
        f"{'Strategy':<28} | "
        f"{'Ann.Ret(G)':>10} | {'Shrp(G)':>7} | "
        f"{'Ann.Ret(N)':>10} | {'Shrp(N)':>7} | "
        f"{'AvgTO':>6} | {'Hit%':>6} | {'MaxDD':>7} | "
        f"{'IC':>7} | {'IC_std':>7}"
    )
    print(header)
    print("-" * table_width)
    
    # Benchmark row
    print(
        f"{'SPY (Benchmark)':<28} | "
        f"{benchmark_return * 100:>9.2f}% | "
        f"{'-':>7} | "
        f"{'-':>10} | "
        f"{'-':>7} | "
        f"{'-':>6} | "
        f"{'-':>6} | "
        f"{'-':>7} | "
        f"{'-':>7} | "
        f"{'-':>7}"
    )
    print("-" * table_width)
    
    # Strategy rows
    for r in results:
        ic_str = f"{r.ic:.4f}" if r.ic is not None else "-"
        ic_std_str = f"{r.ic_std:.4f}" if r.ic_std is not None else "-"
        net_ret_str = f"{r.net_annualized_return * 100:>9.2f}%" if r.net_annualized_return is not None else f"{'-':>10}"
        net_sharpe_str = f"{r.net_sharpe_ratio:>7.3f}" if r.net_sharpe_ratio is not None else f"{'-':>7}"
        turnover_str = f"{r.avg_turnover:.2f}" if r.avg_turnover is not None else "-"
        
        print(
            f"{r.name:<28} | "
            f"{r.annualized_return * 100:>9.2f}% | "
            f"{r.sharpe_ratio:>7.3f} | "
            f"{net_ret_str} | "
            f"{net_sharpe_str} | "
            f"{turnover_str:>6} | "
            f"{r.hit_rate * 100:>5.1f}% | "
            f"{r.max_drawdown * 100:>6.1f}% | "
            f"{ic_str:>7} | "
            f"{ic_std_str:>7}"
        )
    
    print("=" * table_width)
    print("  Note: (G)=Gross, (N)=Net after 10 bps/side costs, AvgTO=Avg Turnover per rebalance")
    print("=" * table_width)


def print_interpretation(results: list[StrategyResult]) -> None:
    """Print interpretation of results."""
    
    ml_result = next((r for r in results if "ML" in r.name), None)
    momentum_result = next((r for r in results if "Momentum" in r.name), None)
    equal_weight_result = next((r for r in results if "Equal-Weight" in r.name), None)
    random_result = next((r for r in results if "Random" in r.name), None)
    
    print("\n" + "=" * 80)
    print("  INTERPRETATION GUIDE")
    print("=" * 80)
    
    print("""
    How to interpret these results:

    1. ML vs Momentum:
       - If ML Sharpe ~ Momentum Sharpe: ML is likely just capturing momentum
       - If ML clearly beats Momentum: ML is adding structure beyond simple momentum
       - If ML worse than Momentum: Possible overfitting or model too complex

    2. ML vs Equal-Weight:
       - If ML ~ Equal-Weight: Stock selection not adding value
       - If ML >> Equal-Weight: Model is successfully identifying outperformers
       - If ML << Equal-Weight: Model is counterproductive

    3. ML vs Random:
       - ML should beat Random baseline by a clear margin
       - If ML ~ Random: No predictive signal detected
       - The margin indicates the value of the ranking model

    4. IC Interpretation:
       - IC < 0.02: Weak/no signal
       - IC 0.02-0.05: Typical for technical factors
       - IC > 0.05: Good signal for cross-sectional ranking
       - IC > 0.10: Suspiciously high (check for leakage)
    """)
    
    if ml_result and momentum_result:
        sharpe_diff = ml_result.sharpe_ratio - momentum_result.sharpe_ratio
        print(f"  ML vs Momentum Sharpe Delta: {sharpe_diff:+.3f}")
        if abs(sharpe_diff) < 0.1:
            print("  -> ML is performing similarly to simple momentum")
        elif sharpe_diff > 0.1:
            print("  -> ML is outperforming simple momentum - adding value")
        else:
            print("  -> ML is underperforming simple momentum - possible overfitting")
    
    if ml_result and random_result:
        sharpe_diff = ml_result.sharpe_ratio - random_result.sharpe_ratio
        print(f"\n  ML vs Random Sharpe Delta: {sharpe_diff:+.3f}")
        if sharpe_diff > 0.2:
            print("  -> ML shows meaningful improvement over random selection")
        else:
            print("  -> ML shows marginal/no improvement over random selection")
    
    if ml_result and ml_result.ic is not None:
        print(f"\n  ML IC Analysis:")
        print(f"    Mean IC: {ml_result.ic:.4f}")
        if ml_result.ic_std:
            print(f"    IC Std:  {ml_result.ic_std:.4f}")
            # IC Information Ratio
            ic_ir = ml_result.ic / ml_result.ic_std if ml_result.ic_std > 0 else 0
            print(f"    IC IR:   {ic_ir:.2f} (IC / IC_std)")
    
    print("\n" + "=" * 80)


def print_ml_comparison(tech_result: StrategyResult, fund_result: StrategyResult) -> None:
    """Print a comparison table between Tech-only and Tech+Fundamentals ML strategies."""
    
    print("\n" + "=" * 80)
    print("  ML STRATEGY COMPARISON: TECH-ONLY vs TECH+FUNDAMENTALS")
    print("=" * 80)
    
    header = f"{'Strategy':<30} | {'Ann.Ret':>10} | {'Sharpe':>8} | {'Hit%':>8} | {'AvgIC':>10}"
    print(header)
    print("-" * 80)
    
    # Tech-only row
    ic_str = f"{tech_result.ic:.4f}" if tech_result.ic is not None else "-"
    print(
        f"{'ML Tech Only':<30} | "
        f"{tech_result.annualized_return * 100:>9.2f}% | "
        f"{tech_result.sharpe_ratio:>8.3f} | "
        f"{tech_result.hit_rate * 100:>7.1f}% | "
        f"{ic_str:>10}"
    )
    
    # Tech+Fundamentals row
    ic_str = f"{fund_result.ic:.4f}" if fund_result.ic is not None else "-"
    print(
        f"{'ML Tech + Fundamentals':<30} | "
        f"{fund_result.annualized_return * 100:>9.2f}% | "
        f"{fund_result.sharpe_ratio:>8.3f} | "
        f"{fund_result.hit_rate * 100:>7.1f}% | "
        f"{ic_str:>10}"
    )
    
    print("-" * 80)
    
    # Delta row
    delta_ret = fund_result.annualized_return - tech_result.annualized_return
    delta_sharpe = fund_result.sharpe_ratio - tech_result.sharpe_ratio
    delta_hit = fund_result.hit_rate - tech_result.hit_rate
    
    delta_ic = None
    if fund_result.ic is not None and tech_result.ic is not None:
        delta_ic = fund_result.ic - tech_result.ic
    
    delta_ic_str = f"{delta_ic:+.4f}" if delta_ic is not None else "-"
    
    print(
        f"{'Delta (Fund - Tech)':<30} | "
        f"{delta_ret * 100:>+9.2f}% | "
        f"{delta_sharpe:>+8.3f} | "
        f"{delta_hit * 100:>+7.1f}% | "
        f"{delta_ic_str:>10}"
    )
    
    print("=" * 80)
    
    # Interpretation
    print("\n  Interpretation:")
    if delta_sharpe > 0.05:
        print("  -> Adding fundamentals IMPROVED Sharpe ratio")
    elif delta_sharpe < -0.05:
        print("  -> Adding fundamentals DEGRADED Sharpe ratio")
    else:
        print("  -> Adding fundamentals had MINIMAL impact on Sharpe")
    
    if delta_ic is not None:
        if delta_ic > 0.005:
            print("  -> Fundamentals IMPROVED cross-sectional signal (higher IC)")
        elif delta_ic < -0.005:
            print("  -> Fundamentals DEGRADED cross-sectional signal (lower IC)")
        else:
            print("  -> Fundamentals had MINIMAL impact on IC")


def print_factor_family_importance(model: GBDTModel) -> None:
    """Print feature importance breakdown by factor family."""
    
    try:
        family_df = model.get_feature_importance_by_family()
    except ValueError:
        print("\n  [Warning] Model not fitted, cannot show feature importance.")
        return
    
    if family_df.empty:
        print("\n  [Warning] No feature importance available.")
        return
    
    print("\n" + "=" * 50)
    print("  FACTOR FAMILY IMPORTANCE")
    print("=" * 50)
    
    header = f"{'FACTOR FAMILY':<20} | {'IMPORTANCE':>12}"
    print(header)
    print("-" * 50)
    
    for _, row in family_df.iterrows():
        print(f"{row['family']:<20} | {row['pct']:>12.2f}")
    
    print("=" * 50)
    
    # Show top individual features
    print("\n  Top 10 Individual Features:")
    top_features = model.get_feature_importance().head(10)
    total_imp = model.get_feature_importance().sum()
    for feat, imp in top_features.items():
        pct = imp / total_imp if total_imp > 0 else 0
        print(f"    {feat:<30} {pct:.3f}")


# ==============================================================================
# CORE FACTOR EXTRACTION
# ==============================================================================

@dataclass
class CoreFactorConfig:
    """Configuration for core factor extraction."""
    
    min_subperiod_support: int = 2  # Must appear in at least N subperiods
    min_importance_threshold: float = 0.01  # Min importance to be considered
    min_mean_importance: float = 0.03  # Min average importance across subperiods


def compute_feature_importance_by_subperiod(
    prices: pd.DataFrame,
    config: PipelineConfig,
    subperiods: list[Subperiod],
    benchmark: str = BENCHMARK,
    horizon_days: int = HORIZON_DAYS,
) -> dict[str, pd.Series]:
    """
    Train a model on each subperiod and extract feature importances.
    
    For each subperiod, trains a GBDT model on data within that period
    and extracts the feature importances. This helps identify which
    features are stable predictors across different market regimes.
    
    Args:
        prices: Price DataFrame with DatetimeIndex.
        config: Pipeline configuration with feature and model settings.
        subperiods: List of Subperiod namedtuples defining time ranges.
        benchmark: Benchmark ticker to exclude from training.
        horizon_days: Forward return horizon for labels.
    
    Returns:
        Dictionary mapping subperiod name to feature importance Series.
        Each Series is sorted by importance descending.
    """
    print("\n" + "=" * 60)
    print("  EXTRACTING FEATURE IMPORTANCE BY SUBPERIOD")
    print("=" * 60)
    
    importance_by_subperiod: dict[str, pd.Series] = {}
    
    for subperiod in subperiods:
        # Filter prices to subperiod
        mask = (prices.index >= subperiod.start) & (prices.index <= subperiod.end)
        subperiod_prices = prices.loc[mask]
        
        if len(subperiod_prices) < 252:  # Need at least 1 year
            print(f"  [Skip] {subperiod.name}: insufficient data ({len(subperiod_prices)} days)")
            continue
        
        # Prepare training data for the subperiod
        try:
            X_train, y_train = prepare_training_data(
                subperiod_prices,
                horizon_days=horizon_days,
                benchmark=benchmark,
                config=config.features,
            )
        except Exception as e:
            print(f"  [Error] {subperiod.name}: {e}")
            continue
        
        if len(X_train) < 100:
            print(f"  [Skip] {subperiod.name}: insufficient training samples ({len(X_train)})")
            continue
        
        # Train model
        model = GBDTModel(config.model)
        model.fit(X_train, y_train)
        
        # Extract importance
        importance = model.get_feature_importance()
        importance_by_subperiod[subperiod.name] = importance
        
        n_features = len(importance)
        print(f"  {subperiod.name}: trained on {len(X_train):,} samples, {n_features} features")
        print(f"    Top 5: {list(importance.head(5).index)}")
    
    return importance_by_subperiod


def select_core_features(
    importance_by_subperiod: dict[str, pd.Series],
    core_config: CoreFactorConfig | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Identify core features that are stable across subperiods.
    
    A feature is considered "core" if:
    1. It appears in at least min_subperiod_support subperiods with 
       importance >= min_importance_threshold
    2. Its mean importance across subperiods >= min_mean_importance
    
    Args:
        importance_by_subperiod: Dict mapping subperiod name to importance Series.
        core_config: Configuration for selection thresholds.
    
    Returns:
        Tuple of:
            - List of core feature names, sorted by mean importance descending.
            - DataFrame with detailed stats per feature (importance by subperiod,
              support count, mean importance).
    """
    if core_config is None:
        core_config = CoreFactorConfig()
    
    # Build a DataFrame: rows = features, columns = subperiods
    # Fill missing importance with 0
    all_features: set[str] = set()
    for imp_series in importance_by_subperiod.values():
        all_features.update(imp_series.index)
    
    importance_df = pd.DataFrame(index=sorted(all_features))
    for subperiod_name, imp_series in importance_by_subperiod.items():
        importance_df[subperiod_name] = imp_series
    
    importance_df = importance_df.fillna(0)
    
    # Compute support count (# subperiods where importance >= threshold)
    support_mask = importance_df >= core_config.min_importance_threshold
    importance_df["support_count"] = support_mask.sum(axis=1)
    
    # Compute mean importance across subperiods (excluding support_count column)
    subperiod_cols = [c for c in importance_df.columns if c != "support_count"]
    importance_df["mean_importance"] = importance_df[subperiod_cols].mean(axis=1)
    
    # Filter to core features
    core_mask = (
        (importance_df["support_count"] >= core_config.min_subperiod_support) &
        (importance_df["mean_importance"] >= core_config.min_mean_importance)
    )
    
    core_df = importance_df[core_mask].sort_values("mean_importance", ascending=False)
    core_features = list(core_df.index)
    
    return core_features, importance_df


def print_core_factor_summary(
    core_features: list[str],
    importance_df: pd.DataFrame,
    subperiods: list[Subperiod],
) -> None:
    """Print a summary of the core factor extraction results."""
    
    print("\n" + "=" * 70)
    print("  CORE FACTOR ANALYSIS RESULTS")
    print("=" * 70)
    
    subperiod_cols = [sp.name for sp in subperiods if sp.name in importance_df.columns]
    
    print(f"\n  Total features analyzed: {len(importance_df)}")
    print(f"  Core features selected:  {len(core_features)}")
    
    if not core_features:
        print("\n  [Warning] No core features met the selection criteria.")
        return
    
    # Show core features table
    print("\n  CORE FEATURES (sorted by mean importance):")
    print("-" * 70)
    
    # Header
    header = f"  {'Feature':<30}"
    for sp in subperiod_cols:
        header += f" | {sp[:7]:>7}"
    header += f" | {'Support':>7} | {'Mean':>6}"
    print(header)
    print("-" * 70)
    
    # Rows (top 15 core features)
    core_df = importance_df.loc[core_features].head(15)
    for feature in core_df.index:
        row = importance_df.loc[feature]
        line = f"  {feature:<30}"
        for sp in subperiod_cols:
            val = row.get(sp, 0)
            line += f" | {val:>7.3f}"
        line += f" | {int(row['support_count']):>7} | {row['mean_importance']:>6.3f}"
        print(line)
    
    if len(core_features) > 15:
        print(f"  ... and {len(core_features) - 15} more core features")
    
    print("-" * 70)
    
    # Show family breakdown of core features
    print("\n  CORE FEATURES BY FAMILY:")
    family_prefixes = {
        "tech_": "Technical",
        "value_": "Value",
        "quality_": "Quality", 
        "growth_": "Growth",
        "prof_": "Profitability",
        "size_": "Size",
    }
    
    family_counts: dict[str, int] = {}
    for feature in core_features:
        matched = False
        for prefix, family in family_prefixes.items():
            if feature.startswith(prefix):
                family_counts[family] = family_counts.get(family, 0) + 1
                matched = True
                break
        if not matched:
            family_counts["Other"] = family_counts.get("Other", 0) + 1
    
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"    {family:<20}: {count:>3} features")


# ==============================================================================
# OUT-OF-SAMPLE ROBUSTNESS EXPERIMENT
# ==============================================================================

@dataclass
class OOSMetrics:
    """Container for OOS experiment metrics."""
    
    period_name: str
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    avg_ic: float | None
    ic_std: float | None
    avg_turnover: float | None
    cumulative_return: float
    annualized_volatility: float


def run_oos_backtest_fixed_model(
    prices: pd.DataFrame,
    trained_model: GBDTModel,
    core_features: list[str],
    config: PipelineConfig,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    benchmark: str = BENCHMARK,
    top_k: int = TOP_K,
    horizon_days: int = HORIZON_DAYS,
    rebalance_freq: str = REBALANCE_FREQ,
    cost_config: CostModelConfig | None = None,
) -> tuple[StrategyResult, BacktestResult]:
    """
    Run backtest with a pre-trained, frozen model on a specific date range.
    
    Unlike run_backtest which retrains at each rebalance, this function
    uses the same pre-trained model throughout, simulating true OOS evaluation.
    
    Args:
        prices: Full price DataFrame (will be filtered to date range).
        trained_model: Pre-trained GBDT model (frozen, no retraining).
        core_features: List of feature names to use for prediction.
        config: Pipeline configuration.
        start_date: Start of backtest period.
        end_date: End of backtest period.
        benchmark: Benchmark ticker.
        top_k: Number of stocks to hold.
        horizon_days: Forward return horizon for IC computation.
        rebalance_freq: Rebalancing frequency.
        cost_config: Transaction cost configuration.
    
    Returns:
        Tuple of (StrategyResult, BacktestResult).
    """
    # Filter prices to the evaluation period
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    period_prices = prices.loc[mask]
    
    if len(period_prices) < 60:
        raise ValueError(f"Insufficient price data for period {start_date} to {end_date}")
    
    # Build feature matrix for the entire period
    feature_matrix = build_feature_matrix(period_prices, config.features)
    
    # Get tradeable tickers
    tradeable_tickers = [t for t in period_prices.columns if t != benchmark]
    
    # Get rebalance dates
    rebal_dates = get_rebalance_dates(period_prices.index, rebalance_freq)
    
    # Need some warmup for feature computation
    min_warmup = max(config.features.momentum_windows) if config.features.momentum_windows else 252
    min_start_date = period_prices.index[min(min_warmup, len(period_prices) - 1)]
    rebal_dates = [d for d in rebal_dates if d >= min_start_date]
    
    if len(rebal_dates) < 2:
        raise ValueError("Not enough rebalance dates in period")
    
    # Storage
    portfolio_returns = []
    weights_history = []
    ic_history = []
    scores_history = []
    prev_weights: dict[str, float] = {}
    
    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]
        
        # Get features for current date
        try:
            current_features = get_features_for_date(feature_matrix, rebal_date)
        except (KeyError, Exception):
            continue
        
        # Filter to tradeable tickers
        current_features = current_features.loc[
            current_features.index.isin(tradeable_tickers)
        ]
        
        if len(current_features) == 0:
            continue
        
        # Filter to core features only
        available_features = [f for f in core_features if f in current_features.columns]
        if len(available_features) < len(core_features) * 0.5:
            # Less than 50% of core features available
            continue
        
        current_features = current_features[available_features]
        
        # Fill NaN with 0 for missing fundamental data
        current_features = current_features.fillna(0)
        
        # Drop any remaining NaN rows
        current_features = current_features.dropna()
        
        if len(current_features) < top_k:
            continue
        
        # Use the frozen model to score stocks
        scores = trained_model.rank_stocks(current_features)
        scores_history.append((rebal_date, scores.copy()))
        
        # Build portfolio
        weights = build_equal_weight_portfolio(scores, top_k)
        weights_history.append((rebal_date, weights))
        
        # Compute holding period returns
        if next_rebal in period_prices.index and rebal_date in period_prices.index:
            start_prices = period_prices.loc[rebal_date, tradeable_tickers]
            end_prices = period_prices.loc[next_rebal, tradeable_tickers]
            period_ret = (end_prices / start_prices) - 1
            
            port_return = compute_portfolio_return(weights, period_ret)
            portfolio_returns.append((next_rebal, port_return))
            
            # Compute IC
            realized = period_ret.loc[scores.index]
            ic = compute_ic(scores, realized)
            ic_history.append((next_rebal, ic))
        
        prev_weights = weights
    
    # Convert to Series
    if portfolio_returns:
        dates, returns = zip(*portfolio_returns)
        port_ret_series = pd.Series(returns, index=pd.DatetimeIndex(dates), name="portfolio")
    else:
        raise ValueError("No portfolio returns computed")
    
    # Compute benchmark returns
    benchmark_returns = []
    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]
        if next_rebal in period_prices.index and rebal_date in period_prices.index:
            start_price = period_prices.loc[rebal_date, benchmark]
            end_price = period_prices.loc[next_rebal, benchmark]
            bench_ret = (end_price / start_price) - 1
            benchmark_returns.append((next_rebal, bench_ret))
    
    if benchmark_returns:
        bench_dates, bench_rets = zip(*benchmark_returns)
        bench_ret_series = pd.Series(bench_rets, index=pd.DatetimeIndex(bench_dates), name="benchmark")
    else:
        bench_ret_series = pd.Series(dtype=float, name="benchmark")
    
    # Compute metrics
    periods_per_year = 12  # Monthly rebalancing
    
    total_return = compute_total_return(port_ret_series)
    ann_return = compute_annualized_return(port_ret_series, periods_per_year)
    ann_vol = compute_annualized_volatility(port_ret_series, periods_per_year)
    sharpe = compute_sharpe_ratio(port_ret_series, 0.0, periods_per_year)
    
    max_dd = compute_max_drawdown(port_ret_series)
    hit_rate = compute_hit_rate(port_ret_series, bench_ret_series)
    ir = compute_ir(port_ret_series, bench_ret_series, periods_per_year)
    
    metrics = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "information_ratio": ir,
    }
    
    # Compute turnover and net metrics
    turnover_series = None
    avg_turnover = None
    p95_turnover = None
    net_returns = None
    net_metrics = None
    
    if cost_config is not None and len(weights_history) >= 2:
        turnover_series = compute_turnover_series(weights_history)
        avg_turnover = turnover_series.mean()
        p95_turnover = turnover_series.quantile(0.95)
        net_returns = compute_net_returns(port_ret_series, turnover_series, cost_config)
        net_metrics = compute_net_metrics(net_returns, bench_ret_series, periods_per_year)
    
    # Create BacktestResult
    backtest_result = BacktestResult(
        portfolio_returns=port_ret_series,
        benchmark_returns=bench_ret_series,
        weights_history=weights_history,
        ic_history=ic_history,
        metrics=metrics,
        config=config,
        scores_history=scores_history,
        turnover_series=turnover_series,
        avg_turnover=avg_turnover,
        p95_turnover=p95_turnover,
        net_returns=net_returns,
        net_metrics=net_metrics,
    )
    
    # Create StrategyResult
    ic_series = None
    avg_ic = None
    ic_std = None
    ic_median = None
    
    if ic_history:
        ic_dates, ic_values = zip(*ic_history)
        ic_series = pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates))
        avg_ic = ic_series.mean()
        ic_std = ic_series.std()
        ic_median = ic_series.median()
    
    net_ann_ret = net_metrics.get("net_ann_ret") if net_metrics else None
    net_sharpe = net_metrics.get("net_sharpe") if net_metrics else None
    
    strategy_result = StrategyResult(
        name="Core Factors OOS",
        portfolio_returns=port_ret_series,
        benchmark_returns=bench_ret_series,
        cumulative_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        hit_rate=hit_rate,
        max_drawdown=max_dd,
        information_ratio=ir,
        ic=avg_ic,
        ic_std=ic_std,
        ic_median=ic_median,
        ic_series=ic_series,
        avg_turnover=avg_turnover,
        p95_turnover=p95_turnover,
        net_annualized_return=net_ann_ret,
        net_sharpe_ratio=net_sharpe,
        weights_history=weights_history,
    )
    
    return strategy_result, backtest_result


def compute_ew_metrics_for_period(
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    benchmark: str = BENCHMARK,
    rebalance_freq: str = REBALANCE_FREQ,
) -> OOSMetrics:
    """Compute Equal-Weight strategy metrics for a specific period."""
    
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    period_prices = prices.loc[mask]
    
    tradeable_tickers = [t for t in period_prices.columns if t != benchmark]
    rebal_dates = get_rebalance_dates(period_prices.index, rebalance_freq)
    
    # Warmup
    if len(rebal_dates) < 2:
        raise ValueError("Not enough rebalance dates")
    
    portfolio_returns = []
    benchmark_returns = []
    
    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal = rebal_dates[i + 1]
        
        # Equal weight all available stocks
        valid_tickers = [
            t for t in tradeable_tickers 
            if t in period_prices.columns and not pd.isna(period_prices.loc[rebal_date, t])
        ]
        if not valid_tickers:
            continue
        
        weight = 1.0 / len(valid_tickers)
        
        # Portfolio return
        start_prices = period_prices.loc[rebal_date]
        end_prices = period_prices.loc[next_rebal]
        
        port_ret = sum(
            weight * ((end_prices[t] / start_prices[t]) - 1)
            for t in valid_tickers
            if not pd.isna(end_prices.get(t)) and not pd.isna(start_prices.get(t))
        )
        portfolio_returns.append((next_rebal, port_ret))
        
        # Benchmark return
        bench_ret = (end_prices[benchmark] / start_prices[benchmark]) - 1
        benchmark_returns.append((next_rebal, bench_ret))
    
    if not portfolio_returns:
        raise ValueError("No returns computed")
    
    port_dates, port_rets = zip(*portfolio_returns)
    port_series = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates))
    
    bench_dates, bench_rets = zip(*benchmark_returns)
    bench_series = pd.Series(bench_rets, index=pd.DatetimeIndex(bench_dates))
    
    periods_per_year = 12
    
    return OOSMetrics(
        period_name=f"EW {start_date.year}-{end_date.year % 100:02d}",
        annualized_return=compute_annualized_return(port_series, periods_per_year),
        sharpe_ratio=compute_sharpe_ratio(port_series, 0.0, periods_per_year),
        max_drawdown=compute_max_drawdown(port_series),
        hit_rate=compute_hit_rate(port_series, bench_series),
        avg_ic=None,  # EW has no IC
        ic_std=None,
        avg_turnover=0.0,  # EW has minimal turnover
        cumulative_return=compute_total_return(port_series),
        annualized_volatility=compute_annualized_volatility(port_series, periods_per_year),
    )


def print_oos_summary_table(
    train_metrics: OOSMetrics,
    test_metrics: OOSMetrics,
    ew_test_metrics: OOSMetrics | None = None,
) -> None:
    """Print formatted OOS experiment summary table."""
    
    print("\n" + "=" * 76)
    print("  CORE FACTOR OOS ROBUSTNESS (Train 2013-2018, Test 2019-2023)")
    print("=" * 76)
    
    header = f"  {'Dataset':<18} | {'Ann.Ret':>8} | {'Sharpe':>7} | {'MaxDD':>8} | {'Hit%':>6} | {'AvgIC':>7} | {'TO%':>5}"
    print(header)
    print("-" * 76)
    
    def format_row(m: OOSMetrics) -> str:
        ann_ret_str = f"{m.annualized_return * 100:>7.2f}%"
        sharpe_str = f"{m.sharpe_ratio:>7.3f}"
        maxdd_str = f"{m.max_drawdown * 100:>7.1f}%"
        hit_str = f"{m.hit_rate * 100:>5.1f}%"
        ic_str = f"{m.avg_ic:>7.4f}" if m.avg_ic is not None else "    N/A"
        to_str = f"{m.avg_turnover * 100:>5.1f}" if m.avg_turnover is not None else "  0.0"
        return f"  {m.period_name:<18} | {ann_ret_str} | {sharpe_str} | {maxdd_str} | {hit_str} | {ic_str} | {to_str}"
    
    print(format_row(train_metrics))
    print(format_row(test_metrics))
    
    if ew_test_metrics is not None:
        print("-" * 76)
        print(format_row(ew_test_metrics))
    
    print("=" * 76)
    
    # Interpretation
    print("\n  Interpretation:")
    
    # Compare Sharpe ratios
    sharpe_decay = (train_metrics.sharpe_ratio - test_metrics.sharpe_ratio) / train_metrics.sharpe_ratio if train_metrics.sharpe_ratio != 0 else 0
    
    if sharpe_decay > 0.5:
        print(f"  [!] SIGNIFICANT DECAY: Test Sharpe is {sharpe_decay*100:.1f}% lower than Train.")
        print("      Model may be overfit to in-sample period.")
    elif sharpe_decay > 0.25:
        print(f"  [!] MODERATE DECAY: Test Sharpe is {sharpe_decay*100:.1f}% lower than Train.")
        print("      Some overfitting present, but model retains predictive power.")
    elif sharpe_decay > 0:
        print(f"  [+] MILD DECAY: Test Sharpe is only {sharpe_decay*100:.1f}% lower than Train.")
        print("      Good evidence of robustness.")
    else:
        print(f"  [+] NO DECAY: Test Sharpe is equal or higher than Train.")
        print("      Excellent out-of-sample performance!")
    
    # Compare IC
    if train_metrics.avg_ic is not None and test_metrics.avg_ic is not None:
        if test_metrics.avg_ic < 0.01:
            print(f"  [!] IC COLLAPSE: Test IC ({test_metrics.avg_ic:.4f}) near zero.")
        elif test_metrics.avg_ic < train_metrics.avg_ic * 0.5:
            print(f"  [!] IC degraded significantly: Train={train_metrics.avg_ic:.4f}, Test={test_metrics.avg_ic:.4f}")
        else:
            print(f"  [+] IC stable: Train={train_metrics.avg_ic:.4f}, Test={test_metrics.avg_ic:.4f}")
    
    # Compare to EW benchmark
    if ew_test_metrics is not None:
        if test_metrics.sharpe_ratio > ew_test_metrics.sharpe_ratio:
            alpha = test_metrics.sharpe_ratio - ew_test_metrics.sharpe_ratio
            print(f"  [+] Core Factors beat EW by {alpha:.3f} Sharpe in test period.")
        else:
            print(f"  [!] Core Factors underperformed EW in test period.")


# ==============================================================================
# REGULARIZATION SWEEP FOR CORE FACTORS [SIMPLE]
# ==============================================================================

@dataclass
class SweepResult:
    """Result from a single configuration in the regularization sweep."""
    
    max_depth: int
    n_estimators: int
    reg_lambda: float
    subsample: float
    colsample: float
    train_sharpe: float
    train_ic: float | None
    train_ann_ret: float
    train_maxdd: float
    test_sharpe: float
    test_ic: float | None
    test_ann_ret: float
    test_maxdd: float
    test_turnover: float


def print_sweep_results_table(results: list[SweepResult]) -> None:
    """
    Print a sorted table of regularization sweep results.
    
    Sorts by test Sharpe (descending), then by test IC (descending).
    """
    if not results:
        print("  No results to display.")
        return
    
    # Sort by test Sharpe descending, then test IC descending
    sorted_results = sorted(
        results,
        key=lambda r: (r.test_sharpe, r.test_ic if r.test_ic else 0),
        reverse=True,
    )
    
    print("\n" + "=" * 100)
    print("  REGULARIZATION SWEEP RESULTS (Core Factors [Simple], OOS 2019-2023)")
    print("=" * 100)
    
    # Header
    header = (
        f"{'max_depth':>9} | {'n_estim':>7} | {'reg_lam':>7} | {'subsamp':>7} | "
        f"{'colsamp':>7} | {'Sharpe(Tr)':>10} | {'Sharpe(Te)':>10} | {'IC(Te)':>8} | {'TO%':>5}"
    )
    print(header)
    print("-" * 100)
    
    # Rows
    for r in sorted_results:
        ic_str = f"{r.test_ic:.4f}" if r.test_ic is not None else "N/A"
        row = (
            f"{r.max_depth:>9} | {r.n_estimators:>7} | {r.reg_lambda:>7.1f} | "
            f"{r.subsample:>7.2f} | {r.colsample:>7.2f} | "
            f"{r.train_sharpe:>10.3f} | {r.test_sharpe:>10.3f} | "
            f"{ic_str:>8} | {r.test_turnover * 100:>5.1f}"
        )
        print(row)
    
    print("=" * 100)
    
    # Top 3 summary
    print("\n  Top 3 Configs by Test Sharpe:")
    print("  " + "-" * 70)
    
    for i, r in enumerate(sorted_results[:3], 1):
        ic_str = f"{r.test_ic:.4f}" if r.test_ic is not None else "N/A"
        print(
            f"  {i}) max_depth={r.max_depth}, n_estim={r.n_estimators}, "
            f"reg_lambda={r.reg_lambda:.1f}, subsample={r.subsample:.2f}, colsample={r.colsample:.2f}"
        )
        print(
            f"     Test Sharpe={r.test_sharpe:.3f}, Test IC={ic_str}, "
            f"TO={r.test_turnover * 100:.1f}%, Train Sharpe={r.train_sharpe:.3f}"
        )
    
    # Additional analysis: compute decay stats
    print("\n  Sharpe Decay Analysis (Train -> Test):")
    print("  " + "-" * 70)
    
    for r in sorted_results[:3]:
        if r.train_sharpe > 0:
            decay_pct = (r.train_sharpe - r.test_sharpe) / r.train_sharpe * 100
            print(
                f"  max_depth={r.max_depth}, reg_lambda={r.reg_lambda:.1f}: "
                f"decay = {decay_pct:.1f}% (Train {r.train_sharpe:.3f} -> Test {r.test_sharpe:.3f})"
            )


def run_core_simple_reg_sweep(
    prices: pd.DataFrame,
    config: PipelineConfig,
    core_factor_config: CoreFactorConfig,
    available_factor_families: dict,
    sweep_config: RegularizationSweepConfig,
    args,
) -> list[SweepResult]:
    """
    Run a regularization sweep for the Core Factors [Simple] model.
    
    Iterates over a grid of hyperparameter combinations, trains each model
    on the training period (2013-2018), and evaluates on the test period
    (2019-2023). Returns sorted results for analysis.
    
    Args:
        prices: Full price DataFrame.
        config: Base pipeline configuration.
        core_factor_config: Configuration for core factor selection.
        available_factor_families: Dict of which fundamental families are available.
        sweep_config: Configuration specifying the hyperparameter grid.
        args: Command-line arguments.
    
    Returns:
        List of SweepResult objects, one per hyperparameter combination.
    """
    print("\n" + "=" * 76)
    print("  REGULARIZATION SWEEP – CORE FACTORS [Simple]")
    print("  Train 2013-2018 vs Test 2019-2023")
    print("=" * 76)
    
    total_configs = sweep_config.grid_size()
    print(f"\n  Grid size: {total_configs} configurations")
    print(f"    max_depth: {sweep_config.max_depth_grid}")
    print(f"    n_estimators: {sweep_config.n_estimators_grid}")
    print(f"    reg_lambda: {sweep_config.reg_lambda_grid}")
    print(f"    subsample: {sweep_config.subsample_grid}")
    print(f"    colsample: {sweep_config.colsample_grid}")
    
    # -------------------------------------------------------------------------
    # Step 1: Build config for fundamentals (same as OOS experiment)
    # -------------------------------------------------------------------------
    oos_config = PipelineConfig(
        research=config.research,
        features=FeatureConfig(
            momentum_windows=(21, 63, 126, 252),
            volatility_windows=(21, 63),
            include_fundamentals=True,
            include_sentiment=False,
            fundamentals_csv_path=args.fundamentals_csv if hasattr(args, 'fundamentals_csv') else None,
            fundamental_factor_families=available_factor_families,
            fundamentals_source=args.fundamentals_source if hasattr(args, 'fundamentals_source') else 'finagg',
            finagg_mode=args.finagg_mode if hasattr(args, 'finagg_mode') else 'quarterly_refined',
        ),
        model=SIMPLE_MODEL_CONFIG,  # Will be overridden per sweep config
    )
    
    # -------------------------------------------------------------------------
    # Step 2: Feature selection using TRAINING WINDOW ONLY
    # -------------------------------------------------------------------------
    print("\n  [1/3] Extracting core features from training period (2013-2018)...")
    
    train_mask = (prices.index >= OOS_TRAIN_START) & (prices.index <= OOS_TRAIN_END)
    train_prices = prices.loc[train_mask]
    
    # Compute importance by subperiod within training window
    importance_by_subperiod = compute_feature_importance_by_subperiod(
        prices=train_prices,
        config=oos_config,
        subperiods=OOS_TRAIN_SUBPERIODS,
        benchmark=BENCHMARK,
        horizon_days=HORIZON_DAYS,
    )
    
    if len(importance_by_subperiod) < 1:
        print("  [Error] Could not compute importance for any subperiods.")
        return []
    
    if len(importance_by_subperiod) < 2:
        print(f"  [Warning] Only {len(importance_by_subperiod)} subperiod(s) available. Relaxing min_subperiod_support to 1.")

    # Select core features using training-only importances
    # Dynamically adjust min_subperiod_support based on available subperiods
    adjusted_core_config = CoreFactorConfig(
        min_subperiod_support=min(core_factor_config.min_subperiod_support, len(importance_by_subperiod)),
        min_importance_threshold=core_factor_config.min_importance_threshold,
        min_mean_importance=core_factor_config.min_mean_importance,
    )
    core_features, _ = select_core_features(
        importance_by_subperiod,
        core_config=adjusted_core_config,
    )
    
    print(f"        Selected {len(core_features)} core features")
    
    if not core_features:
        print("  [Error] No core features selected.")
        return []
    
    # -------------------------------------------------------------------------
    # Step 3: Prepare training data ONCE (reused for all configs)
    # -------------------------------------------------------------------------
    print("\n  [2/3] Preparing training data...")
    
    try:
        X_train, y_train = prepare_training_data(
            train_prices,
            horizon_days=HORIZON_DAYS,
            benchmark=BENCHMARK,
            config=oos_config.features,
        )
    except Exception as e:
        print(f"  [Error] Failed to prepare training data: {e}")
        return []
    
    # Filter to core features
    available_core = [f for f in core_features if f in X_train.columns]
    X_train_core = X_train[available_core].fillna(0)
    
    print(f"        Training samples: {len(X_train_core):,}")
    print(f"        Features used: {len(available_core)}")
    
    # -------------------------------------------------------------------------
    # Step 4: Run sweep over hyperparameter grid
    # -------------------------------------------------------------------------
    print(f"\n  [3/3] Running sweep over {total_configs} configurations...")
    
    results: list[SweepResult] = []
    
    # Generate all combinations
    param_grid = list(itertools.product(
        sweep_config.max_depth_grid,
        sweep_config.n_estimators_grid,
        sweep_config.reg_lambda_grid,
        sweep_config.subsample_grid,
        sweep_config.colsample_grid,
    ))
    
    for i, (max_depth, n_estimators, reg_lambda, subsample, colsample) in enumerate(param_grid, 1):
        # Progress indicator
        if i % 10 == 0 or i == 1:
            print(f"        Config {i}/{total_configs}...")
        
        # Build model config for this sweep point
        model_config = build_model_config_from_params(
            base_config=SIMPLE_MODEL_CONFIG,
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample,
        )
        
        # Train model
        try:
            model = GBDTModel(model_config)
            model.fit(X_train_core, y_train)
        except Exception as e:
            print(f"  [Warning] Failed to train config {i}: {e}")
            continue
        
        # Update config with this model config for backtest
        sweep_oos_config = replace(oos_config, model=model_config)
        
        # Evaluate on training period (in-sample)
        try:
            train_result, _ = run_oos_backtest_fixed_model(
                prices=prices,
                trained_model=model,
                core_features=available_core,
                config=sweep_oos_config,
                start_date=OOS_TRAIN_START,
                end_date=OOS_TRAIN_END,
                benchmark=BENCHMARK,
                top_k=TOP_K,
                horizon_days=HORIZON_DAYS,
                rebalance_freq=REBALANCE_FREQ,
                cost_config=COST_MODEL,
            )
        except Exception as e:
            print(f"  [Warning] Failed to evaluate train period for config {i}: {e}")
            continue
        
        # Evaluate on test period (out-of-sample)
        try:
            test_result, _ = run_oos_backtest_fixed_model(
                prices=prices,
                trained_model=model,
                core_features=available_core,
                config=sweep_oos_config,
                start_date=OOS_TEST_START,
                end_date=OOS_TEST_END,
                benchmark=BENCHMARK,
                top_k=TOP_K,
                horizon_days=HORIZON_DAYS,
                rebalance_freq=REBALANCE_FREQ,
                cost_config=COST_MODEL,
            )
        except Exception as e:
            print(f"  [Warning] Failed to evaluate test period for config {i}: {e}")
            continue
        
        # Store results
        results.append(SweepResult(
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample=colsample,
            train_sharpe=train_result.sharpe_ratio,
            train_ic=train_result.ic,
            train_ann_ret=train_result.annualized_return,
            train_maxdd=train_result.max_drawdown,
            test_sharpe=test_result.sharpe_ratio,
            test_ic=test_result.ic,
            test_ann_ret=test_result.annualized_return,
            test_maxdd=test_result.max_drawdown,
            test_turnover=test_result.avg_turnover if test_result.avg_turnover else 0.0,
        ))
    
    print(f"\n        Completed {len(results)}/{total_configs} configurations successfully.")
    
    return results


# ==============================================================================
# MULTI-SPLIT OOS ROBUSTNESS SUITE - CORE FUNCTIONS
# ==============================================================================

@dataclass
class OOSSplitResult:
    """Result from a single OOS split in the robustness suite."""
    
    split_name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # Core model metrics
    core_sharpe_train: float
    core_sharpe_test: float
    core_ic_test: float | None
    core_ann_ret_test: float
    core_maxdd_test: float
    core_hit_test: float
    core_turnover_test: float
    # EW baseline metrics
    ew_sharpe_test: float
    ew_ann_ret_test: float


@dataclass
class HybridOOSSplitResult:
    """Result from a single OOS split comparing Tech-only vs Hybrid strategies."""
    
    split_name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    
    # Coverage stats for this split
    n_tickers_with_fundamentals: int
    n_tickers_tech_only: int
    
    # Tech-only model metrics
    tech_sharpe_train: float
    tech_sharpe_test: float
    tech_ic_test: float | None
    tech_ann_ret_test: float
    tech_maxdd_test: float
    tech_hit_test: float
    tech_turnover_test: float
    
    # Hybrid model metrics (tech + fundamentals where available)
    hybrid_sharpe_train: float
    hybrid_sharpe_test: float
    hybrid_ic_test: float | None
    hybrid_ann_ret_test: float
    hybrid_maxdd_test: float
    hybrid_hit_test: float
    hybrid_turnover_test: float
    
    # EW baseline metrics
    ew_sharpe_test: float
    ew_ann_ret_test: float
    
    @property
    def delta_sharpe_hybrid_vs_tech(self) -> float:
        """Difference in test Sharpe: Hybrid - Tech."""
        return self.hybrid_sharpe_test - self.tech_sharpe_test
    
    @property
    def delta_sharpe_hybrid_vs_ew(self) -> float:
        """Difference in test Sharpe: Hybrid - EW."""
        return self.hybrid_sharpe_test - self.ew_sharpe_test


def run_core_simple_oos_once(
    prices: pd.DataFrame,
    config: PipelineConfig,
    available_factor_families: dict,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    model_config: ModelConfig | None = None,
    args=None,
    verbose: bool = True,
) -> tuple[OOSMetrics, OOSMetrics]:
    """
    Train Core Factors [Simple] on [train_start, train_end],
    evaluate on [test_start, test_end].
    
    Args:
        prices: Full price DataFrame.
        config: Base pipeline configuration.
        available_factor_families: Dict of which fundamental families are available.
        train_start: Start of training period.
        train_end: End of training period.
        test_start: Start of test period.
        test_end: End of test period.
        model_config: Optional model config override (e.g., from reg sweep).
        args: Command-line arguments.
        verbose: Whether to print progress.
    
    Returns:
        Tuple of (train_metrics, test_metrics).
    """
    if model_config is None:
        model_config = SIMPLE_MODEL_CONFIG
    
    # Build OOS config with fundamentals
    oos_config = PipelineConfig(
        research=config.research,
        features=FeatureConfig(
            momentum_windows=(21, 63, 126, 252),
            volatility_windows=(21, 63),
            include_fundamentals=True,
            include_sentiment=False,
            fundamentals_csv_path=getattr(args, 'fundamentals_csv', None) if args else None,
            fundamental_factor_families=available_factor_families,
            fundamentals_source=getattr(args, 'fundamentals_source', 'finagg') if args else 'finagg',
            finagg_mode=getattr(args, 'finagg_mode', 'quarterly_refined') if args else 'quarterly_refined',
        ),
        model=model_config,
    )
    
    # Build subperiods dynamically based on training window
    train_years = list(range(train_start.year, train_end.year + 1))
    
    # Create 2-year subperiods for feature importance
    train_subperiods = []
    for i in range(0, len(train_years), 2):
        if i + 1 < len(train_years):
            start_year = train_years[i]
            end_year = train_years[i + 1]
        else:
            start_year = train_years[i]
            end_year = train_years[i]
        
        train_subperiods.append(Subperiod(
            name=f"{start_year}-{end_year}",
            start=pd.Timestamp(f"{start_year}-01-01"),
            end=pd.Timestamp(f"{end_year}-12-31"),
        ))
    
    # Make sure we have at least 2 subperiods for feature selection
    if len(train_subperiods) < 2:
        # Fallback: split training window in half
        mid_year = train_start.year + (train_end.year - train_start.year) // 2
        train_subperiods = [
            Subperiod(
                name=f"{train_start.year}-{mid_year}",
                start=train_start,
                end=pd.Timestamp(f"{mid_year}-12-31"),
            ),
            Subperiod(
                name=f"{mid_year + 1}-{train_end.year}",
                start=pd.Timestamp(f"{mid_year + 1}-01-01"),
                end=train_end,
            ),
        ]
    
    # Filter prices to training window
    train_mask = (prices.index >= train_start) & (prices.index <= train_end)
    train_prices = prices.loc[train_mask]
    
    if verbose:
        print(f"      Training: {train_start.date()} to {train_end.date()} ({len(train_prices)} days)")
    
    # Compute feature importance by subperiod
    importance_by_subperiod = compute_feature_importance_by_subperiod(
        prices=train_prices,
        config=oos_config,
        subperiods=train_subperiods,
        benchmark=BENCHMARK,
        horizon_days=HORIZON_DAYS,
    )
    
    if len(importance_by_subperiod) < 1:
        raise ValueError("Could not compute importance for any subperiods")
    
    if len(importance_by_subperiod) < 2:
        print(f"      [Warning] Only {len(importance_by_subperiod)} subperiod(s) available. Relaxing min_subperiod_support to 1.")
    
    # Select core features
    # Dynamically adjust min_subperiod_support based on available subperiods
    core_config = CoreFactorConfig(
        min_subperiod_support=min(2, len(importance_by_subperiod)),
        min_importance_threshold=0.01,
        min_mean_importance=0.03,
    )
    core_features, _ = select_core_features(importance_by_subperiod, core_config=core_config)
    
    if not core_features:
        raise ValueError("No core features selected")
    
    if verbose:
        print(f"      Core features: {len(core_features)}")
    
    # Prepare training data
    X_train, y_train = prepare_training_data(
        train_prices,
        horizon_days=HORIZON_DAYS,
        benchmark=BENCHMARK,
        config=oos_config.features,
    )
    
    # Filter to available core features
    available_core = [f for f in core_features if f in X_train.columns]
    X_train_core = X_train[available_core].fillna(0)
    
    # Train the model
    model = GBDTModel(model_config)
    model.fit(X_train_core, y_train)
    
    # Evaluate on training period
    train_result, _ = run_oos_backtest_fixed_model(
        prices=prices,
        trained_model=model,
        core_features=available_core,
        config=oos_config,
        start_date=train_start,
        end_date=train_end,
        benchmark=BENCHMARK,
        top_k=TOP_K,
        horizon_days=HORIZON_DAYS,
        rebalance_freq=REBALANCE_FREQ,
        cost_config=COST_MODEL,
    )
    
    train_metrics = OOSMetrics(
        period_name=f"Train {train_start.year}-{train_end.year}",
        annualized_return=train_result.annualized_return,
        sharpe_ratio=train_result.sharpe_ratio,
        max_drawdown=train_result.max_drawdown,
        hit_rate=train_result.hit_rate,
        avg_ic=train_result.ic,
        ic_std=train_result.ic_std,
        avg_turnover=train_result.avg_turnover,
        cumulative_return=train_result.cumulative_return,
        annualized_volatility=train_result.annualized_volatility,
    )
    
    # Evaluate on test period
    test_result, _ = run_oos_backtest_fixed_model(
        prices=prices,
        trained_model=model,
        core_features=available_core,
        config=oos_config,
        start_date=test_start,
        end_date=test_end,
        benchmark=BENCHMARK,
        top_k=TOP_K,
        horizon_days=HORIZON_DAYS,
        rebalance_freq=REBALANCE_FREQ,
        cost_config=COST_MODEL,
    )
    
    test_metrics = OOSMetrics(
        period_name=f"Test {test_start.year}-{test_end.year}",
        annualized_return=test_result.annualized_return,
        sharpe_ratio=test_result.sharpe_ratio,
        max_drawdown=test_result.max_drawdown,
        hit_rate=test_result.hit_rate,
        avg_ic=test_result.ic,
        ic_std=test_result.ic_std,
        avg_turnover=test_result.avg_turnover,
        cumulative_return=test_result.cumulative_return,
        annualized_volatility=test_result.annualized_volatility,
    )
    
    return train_metrics, test_metrics


def run_hybrid_oos_once(
    prices: pd.DataFrame,
    config: PipelineConfig,
    available_factor_families: dict,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    min_coverage: float,
    model_config: ModelConfig | None = None,
    args=None,
    verbose: bool = True,
) -> HybridOOSSplitResult:
    """
    Run both Tech-only and Hybrid (Tech + Fundamentals) models on a single OOS split.
    
    For hybrid mode:
    - Computes coverage to identify tickers_with_fundamentals vs tickers_tech_only
    - Masks fundamentals to NaN for tech-only tickers
    - Trains and evaluates both models on the same split
    
    Args:
        prices: Full price DataFrame.
        config: Base pipeline configuration.
        available_factor_families: Dict of which fundamental families are available.
        train_start: Start of training period.
        train_end: End of training period.
        test_start: Start of test period.
        test_end: End of test period.
        min_coverage: Minimum coverage threshold for fundamentals.
        model_config: Optional model config override.
        args: Command-line arguments.
        verbose: Whether to print progress.
    
    Returns:
        HybridOOSSplitResult with metrics for both Tech and Hybrid models.
    """
    if model_config is None:
        model_config = SIMPLE_MODEL_CONFIG
    
    # Get tradeable tickers
    tradeable_tickers = [t for t in prices.columns if t != BENCHMARK]
    
    # -------------------------------------------------------------------------
    # Step 1: Compute coverage and identify tech-only tickers
    # -------------------------------------------------------------------------
    # Build temporary config with fundamentals for coverage computation
    temp_feature_config = FeatureConfig(
        momentum_windows=(21, 63, 126, 252),
        volatility_windows=(21, 63),
        include_technical=False,  # Only need fundamentals for coverage
        include_fundamentals=True,
        include_sentiment=False,
        fundamentals_csv_path=getattr(args, 'fundamentals_csv', None) if args else None,
        fundamental_factor_families=available_factor_families,
        fundamentals_source=getattr(args, 'fundamentals_source', 'finagg') if args else 'finagg',
        finagg_mode=getattr(args, 'finagg_mode', 'quarterly_refined') if args else 'quarterly_refined',
    )
    
    # Build feature matrix for coverage computation
    try:
        feature_df = build_feature_matrix(prices, temp_feature_config)
        fundamental_cols = identify_fundamental_columns(feature_df)
        
        if fundamental_cols:
            coverage = compute_fundamentals_coverage(
                feature_df, fundamental_cols, 
                coverage_freq='YE', benchmark=BENCHMARK
            )
            tech_only_tickers = get_tech_only_tickers(coverage, min_coverage)
            tickers_with_fundamentals = [t for t in tradeable_tickers if t not in tech_only_tickers]
        else:
            tech_only_tickers = []
            tickers_with_fundamentals = tradeable_tickers.copy()
    except Exception as e:
        logger.warning(f"Coverage computation failed: {e}. Using all tickers with fundamentals.")
        tech_only_tickers = []
        tickers_with_fundamentals = tradeable_tickers.copy()
    
    n_with_fund = len(tickers_with_fundamentals)
    n_tech_only = len(tech_only_tickers)
    
    if verbose:
        print(f"      Coverage: {n_with_fund} tickers with fundamentals, {n_tech_only} tech-only")
    
    # -------------------------------------------------------------------------
    # Step 2: Build configs for both strategies
    # -------------------------------------------------------------------------
    # Tech-only config (no fundamentals)
    tech_config = PipelineConfig(
        research=config.research,
        features=FeatureConfig(
            momentum_windows=(21, 63, 126, 252),
            volatility_windows=(21, 63),
            include_fundamentals=False,
            include_sentiment=False,
        ),
        model=model_config,
    )
    
    # Hybrid config (fundamentals + tech, with masking for low-coverage tickers)
    hybrid_config = PipelineConfig(
        research=config.research,
        features=FeatureConfig(
            momentum_windows=(21, 63, 126, 252),
            volatility_windows=(21, 63),
            include_fundamentals=True,
            include_sentiment=False,
            fundamentals_csv_path=getattr(args, 'fundamentals_csv', None) if args else None,
            fundamental_factor_families=available_factor_families,
            fundamentals_source=getattr(args, 'fundamentals_source', 'finagg') if args else 'finagg',
            finagg_mode=getattr(args, 'finagg_mode', 'quarterly_refined') if args else 'quarterly_refined',
            tech_only_tickers=tuple(tech_only_tickers),  # Mask fundamentals for these
        ),
        model=model_config,
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Build subperiods for feature selection (same for both)
    # -------------------------------------------------------------------------
    train_years = list(range(train_start.year, train_end.year + 1))
    train_subperiods = []
    for i in range(0, len(train_years), 2):
        if i + 1 < len(train_years):
            start_year = train_years[i]
            end_year = train_years[i + 1]
        else:
            start_year = train_years[i]
            end_year = train_years[i]
        train_subperiods.append(Subperiod(
            name=f"{start_year}-{end_year}",
            start=pd.Timestamp(f"{start_year}-01-01"),
            end=pd.Timestamp(f"{end_year}-12-31"),
        ))
    
    if len(train_subperiods) < 2:
        mid_year = train_start.year + (train_end.year - train_start.year) // 2
        train_subperiods = [
            Subperiod(
                name=f"{train_start.year}-{mid_year}",
                start=train_start,
                end=pd.Timestamp(f"{mid_year}-12-31"),
            ),
            Subperiod(
                name=f"{mid_year + 1}-{train_end.year}",
                start=pd.Timestamp(f"{mid_year + 1}-01-01"),
                end=train_end,
            ),
        ]
    
    # Filter prices to training window
    train_mask = (prices.index >= train_start) & (prices.index <= train_end)
    train_prices = prices.loc[train_mask]
    
    if verbose:
        print(f"      Training: {train_start.date()} to {train_end.date()}")
    
    # -------------------------------------------------------------------------
    # Step 4: Run Tech-only model
    # -------------------------------------------------------------------------
    if verbose:
        print("      Running Tech-only model...")
    
    # Compute tech feature importance
    tech_importance = compute_feature_importance_by_subperiod(
        prices=train_prices,
        config=tech_config,
        subperiods=train_subperiods,
        benchmark=BENCHMARK,
        horizon_days=HORIZON_DAYS,
    )
    
    core_config_tech = CoreFactorConfig(
        min_subperiod_support=min(2, len(tech_importance)),
        min_importance_threshold=0.01,
        min_mean_importance=0.03,
    )
    tech_core_features, _ = select_core_features(tech_importance, core_config=core_config_tech)
    
    if not tech_core_features:
        raise ValueError("No core features selected for Tech-only model")
    
    # Prepare training data for tech
    X_train_tech, y_train_tech = prepare_training_data(
        train_prices,
        horizon_days=HORIZON_DAYS,
        benchmark=BENCHMARK,
        config=tech_config.features,
    )
    available_tech_core = [f for f in tech_core_features if f in X_train_tech.columns]
    X_train_tech_core = X_train_tech[available_tech_core].fillna(0)
    
    # Train tech model
    tech_model = GBDTModel(model_config)
    tech_model.fit(X_train_tech_core, y_train_tech)
    
    # Evaluate tech on train
    tech_train_result, _ = run_oos_backtest_fixed_model(
        prices=prices, trained_model=tech_model, core_features=available_tech_core,
        config=tech_config, start_date=train_start, end_date=train_end,
        benchmark=BENCHMARK, top_k=TOP_K, horizon_days=HORIZON_DAYS,
        rebalance_freq=REBALANCE_FREQ, cost_config=COST_MODEL,
    )
    
    # Evaluate tech on test
    tech_test_result, _ = run_oos_backtest_fixed_model(
        prices=prices, trained_model=tech_model, core_features=available_tech_core,
        config=tech_config, start_date=test_start, end_date=test_end,
        benchmark=BENCHMARK, top_k=TOP_K, horizon_days=HORIZON_DAYS,
        rebalance_freq=REBALANCE_FREQ, cost_config=COST_MODEL,
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Run Hybrid model
    # -------------------------------------------------------------------------
    if verbose:
        print("      Running Hybrid model...")
    
    # Compute hybrid feature importance
    hybrid_importance = compute_feature_importance_by_subperiod(
        prices=train_prices,
        config=hybrid_config,
        subperiods=train_subperiods,
        benchmark=BENCHMARK,
        horizon_days=HORIZON_DAYS,
    )
    
    core_config_hybrid = CoreFactorConfig(
        min_subperiod_support=min(2, len(hybrid_importance)),
        min_importance_threshold=0.01,
        min_mean_importance=0.03,
    )
    hybrid_core_features, _ = select_core_features(hybrid_importance, core_config=core_config_hybrid)
    
    if not hybrid_core_features:
        raise ValueError("No core features selected for Hybrid model")
    
    # Prepare training data for hybrid
    X_train_hybrid, y_train_hybrid = prepare_training_data(
        train_prices,
        horizon_days=HORIZON_DAYS,
        benchmark=BENCHMARK,
        config=hybrid_config.features,
    )
    available_hybrid_core = [f for f in hybrid_core_features if f in X_train_hybrid.columns]
    X_train_hybrid_core = X_train_hybrid[available_hybrid_core].fillna(0)
    
    # Train hybrid model
    hybrid_model = GBDTModel(model_config)
    hybrid_model.fit(X_train_hybrid_core, y_train_hybrid)
    
    # Evaluate hybrid on train
    hybrid_train_result, _ = run_oos_backtest_fixed_model(
        prices=prices, trained_model=hybrid_model, core_features=available_hybrid_core,
        config=hybrid_config, start_date=train_start, end_date=train_end,
        benchmark=BENCHMARK, top_k=TOP_K, horizon_days=HORIZON_DAYS,
        rebalance_freq=REBALANCE_FREQ, cost_config=COST_MODEL,
    )
    
    # Evaluate hybrid on test
    hybrid_test_result, _ = run_oos_backtest_fixed_model(
        prices=prices, trained_model=hybrid_model, core_features=available_hybrid_core,
        config=hybrid_config, start_date=test_start, end_date=test_end,
        benchmark=BENCHMARK, top_k=TOP_K, horizon_days=HORIZON_DAYS,
        rebalance_freq=REBALANCE_FREQ, cost_config=COST_MODEL,
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Compute EW baseline
    # -------------------------------------------------------------------------
    ew_metrics = compute_ew_metrics_for_period(
        prices=prices,
        start_date=test_start,
        end_date=test_end,
        benchmark=BENCHMARK,
        rebalance_freq=REBALANCE_FREQ,
    )
    
    # -------------------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------------------
    return HybridOOSSplitResult(
        split_name=f"train_{train_start.year}_{train_end.year}_test_{test_start.year}_{test_end.year}",
        train_start=str(train_start.date()),
        train_end=str(train_end.date()),
        test_start=str(test_start.date()),
        test_end=str(test_end.date()),
        n_tickers_with_fundamentals=n_with_fund,
        n_tickers_tech_only=n_tech_only,
        # Tech metrics
        tech_sharpe_train=tech_train_result.sharpe_ratio,
        tech_sharpe_test=tech_test_result.sharpe_ratio,
        tech_ic_test=tech_test_result.ic,
        tech_ann_ret_test=tech_test_result.annualized_return,
        tech_maxdd_test=tech_test_result.max_drawdown,
        tech_hit_test=tech_test_result.hit_rate,
        tech_turnover_test=tech_test_result.avg_turnover or 0.0,
        # Hybrid metrics
        hybrid_sharpe_train=hybrid_train_result.sharpe_ratio,
        hybrid_sharpe_test=hybrid_test_result.sharpe_ratio,
        hybrid_ic_test=hybrid_test_result.ic,
        hybrid_ann_ret_test=hybrid_test_result.annualized_return,
        hybrid_maxdd_test=hybrid_test_result.max_drawdown,
        hybrid_hit_test=hybrid_test_result.hit_rate,
        hybrid_turnover_test=hybrid_test_result.avg_turnover or 0.0,
        # EW baseline
        ew_sharpe_test=ew_metrics.sharpe_ratio,
        ew_ann_ret_test=ew_metrics.annualized_return,
    )


def run_hybrid_oos_robustness_suite(
    prices: pd.DataFrame,
    config: PipelineConfig,
    available_factor_families: dict,
    suite_cfg: CoreOOSRobustnessConfig,
    min_coverage: float,
    model_config: ModelConfig | None = None,
    args=None,
) -> list[HybridOOSSplitResult]:
    """
    Run Hybrid OOS robustness suite: Tech-only vs Hybrid across all splits.
    
    Args:
        prices: Full price DataFrame.
        config: Base pipeline configuration.
        available_factor_families: Dict of which fundamental families are available.
        suite_cfg: Configuration with list of OOS splits.
        min_coverage: Minimum fundamentals coverage threshold.
        model_config: Optional model config override.
        args: Command-line arguments.
    
    Returns:
        List of HybridOOSSplitResult objects, one per split.
    """
    print("\n" + "=" * 100)
    print("  CORE FACTOR HYBRID OOS ROBUSTNESS SUITE")
    print("  Comparing Tech-only vs Hybrid (Tech + Fundamentals) across OOS splits")
    print("=" * 100)
    
    print(f"\n  Min fundamentals coverage: {min_coverage:.0%}")
    print(f"  Running {len(suite_cfg.splits)} OOS splits...")
    
    if model_config:
        print(f"  Model config: max_depth={model_config.max_depth}, "
              f"n_estimators={model_config.n_estimators}, "
              f"reg_lambda={model_config.reg_lambda}")
    
    results: list[HybridOOSSplitResult] = []
    
    for i, split in enumerate(suite_cfg.splits, 1):
        print(f"\n  [{i}/{len(suite_cfg.splits)}] {split.name}")
        
        try:
            result = run_hybrid_oos_once(
                prices=prices,
                config=config,
                available_factor_families=available_factor_families,
                train_start=split.train_start_ts,
                train_end=split.train_end_ts,
                test_start=split.test_start_ts,
                test_end=split.test_end_ts,
                min_coverage=min_coverage,
                model_config=model_config,
                args=args,
                verbose=True,
            )
            results.append(result)
            
            delta = result.delta_sharpe_hybrid_vs_tech
            print(f"      Tech Test Sharpe: {result.tech_sharpe_test:.3f}, "
                  f"Hybrid Test Sharpe: {result.hybrid_sharpe_test:.3f}, "
                  f"Delta: {delta:+.3f}")
            
        except Exception as e:
            print(f"      [Error] Failed to run split: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n  Completed {len(results)}/{len(suite_cfg.splits)} splits successfully.")
    
    return results


def print_hybrid_oos_robustness_summary(results: list[HybridOOSSplitResult]) -> None:
    """
    Print a comparison summary table for Hybrid OOS robustness suite.
    """
    if not results:
        print("\n  No results to display.")
        return
    
    print("\n" + "=" * 130)
    print("  CORE FACTOR HYBRID OOS COMPARISON")
    print("=" * 130)
    
    # Header
    header = (
        f"{'Split':<35} | {'Tech Shp(Tr)':>11} | {'Tech Shp(Te)':>11} | "
        f"{'Hyb Shp(Te)':>11} | {'EW Shp':>8} | {'dShp(Hyb-Tech)':>14} | "
        f"{'Hyb IC(Te)':>10}"
    )
    print(header)
    print("-" * 130)
    
    # Rows
    for r in results:
        delta = r.delta_sharpe_hybrid_vs_tech
        delta_str = f"{delta:+.3f}"
        ic_str = f"{r.hybrid_ic_test:.4f}" if r.hybrid_ic_test is not None else "N/A"
        
        row = (
            f"{r.split_name:<35} | {r.tech_sharpe_train:>11.3f} | "
            f"{r.tech_sharpe_test:>11.3f} | {r.hybrid_sharpe_test:>11.3f} | "
            f"{r.ew_sharpe_test:>8.3f} | {delta_str:>14} | {ic_str:>10}"
        )
        print(row)
    
    print("-" * 130)
    
    # Summary statistics
    avg_tech_test = sum(r.tech_sharpe_test for r in results) / len(results)
    avg_hybrid_test = sum(r.hybrid_sharpe_test for r in results) / len(results)
    avg_delta = sum(r.delta_sharpe_hybrid_vs_tech for r in results) / len(results)
    avg_ew = sum(r.ew_sharpe_test for r in results) / len(results)
    
    print(f"\n  Avg Tech Test Sharpe:   {avg_tech_test:.3f}")
    print(f"  Avg Hybrid Test Sharpe: {avg_hybrid_test:.3f}")
    print(f"  Avg Delta (Hybrid - Tech): {avg_delta:+.3f}")
    print(f"  Avg EW Test Sharpe:     {avg_ew:.3f}")
    
    # Count how many splits Hybrid beats Tech
    hybrid_beats_tech = sum(1 for r in results if r.hybrid_sharpe_test > r.tech_sharpe_test)
    hybrid_beats_ew = sum(1 for r in results if r.hybrid_sharpe_test > r.ew_sharpe_test)
    tech_beats_ew = sum(1 for r in results if r.tech_sharpe_test > r.ew_sharpe_test)
    
    print(f"\n  Splits where Hybrid > Tech: {hybrid_beats_tech}/{len(results)}")
    print(f"  Splits where Hybrid > EW:   {hybrid_beats_ew}/{len(results)}")
    print(f"  Splits where Tech > EW:     {tech_beats_ew}/{len(results)}")
    
    # Coverage summary
    avg_n_fund = sum(r.n_tickers_with_fundamentals for r in results) / len(results)
    avg_n_tech = sum(r.n_tickers_tech_only for r in results) / len(results)
    print(f"\n  Avg tickers with fundamentals: {avg_n_fund:.0f}")
    print(f"  Avg tickers tech-only:         {avg_n_tech:.0f}")
    
    # Detailed comparison table
    print("\n  Per-Split Details:")
    print("  " + "-" * 100)
    print(f"  {'Split':<35} | {'Tech Ann.Ret':>11} | {'Hyb Ann.Ret':>11} | "
          f"{'Tech MaxDD':>10} | {'Hyb MaxDD':>10}")
    print("  " + "-" * 100)
    
    for r in results:
        print(f"  {r.split_name:<35} | {r.tech_ann_ret_test * 100:>10.1f}% | "
              f"{r.hybrid_ann_ret_test * 100:>10.1f}% | "
              f"{r.tech_maxdd_test * 100:>9.1f}% | {r.hybrid_maxdd_test * 100:>9.1f}%")
    
    # Interpretation
    print("\n  Interpretation:")
    
    if avg_delta > 0.1 and hybrid_beats_tech >= len(results) * 0.66:
        print("    [+] HYBRID ADDS VALUE: Fundamentals provide meaningful alpha boost.")
    elif avg_delta > 0 and hybrid_beats_tech >= len(results) * 0.5:
        print("    [~] MODEST IMPROVEMENT: Hybrid marginally better, but not consistent.")
    elif avg_delta < -0.1:
        print("    [-] TECH BETTER: Fundamentals may be introducing noise or overfitting.")
    else:
        print("    [~] INCONCLUSIVE: Results are mixed; consider per-regime analysis.")
    
    # Stability check
    if hybrid_beats_ew >= len(results) - 1 and tech_beats_ew >= len(results) - 1:
        print("    [+] STABLE: Both Tech and Hybrid consistently beat Equal Weight.")
    elif hybrid_beats_ew < len(results) * 0.5:
        print("    [!] CAUTION: Hybrid underperforms Equal Weight in many splits.")


def run_core_oos_robustness_suite(
    prices: pd.DataFrame,
    config: PipelineConfig,
    available_factor_families: dict,
    suite_cfg: CoreOOSRobustnessConfig,
    model_config: ModelConfig | None = None,
    args=None,
) -> list[OOSSplitResult]:
    """
    Run Core Factors [Simple] across all OOS splits in suite_cfg.
    
    Args:
        prices: Full price DataFrame.
        config: Base pipeline configuration.
        available_factor_families: Dict of which fundamental families are available.
        suite_cfg: Configuration with list of OOS splits.
        model_config: Optional model config override.
        args: Command-line arguments.
    
    Returns:
        List of OOSSplitResult objects, one per split.
    """
    print("\n" + "=" * 100)
    print("  CORE FACTOR OOS ROBUSTNESS SUITE (Core Factors [Simple])")
    print("=" * 100)
    
    print(f"\n  Running {len(suite_cfg.splits)} OOS splits...")
    
    if model_config:
        print(f"  Model config: max_depth={model_config.max_depth}, "
              f"n_estimators={model_config.n_estimators}, "
              f"reg_lambda={model_config.reg_lambda}")
    
    results: list[OOSSplitResult] = []
    
    for i, split in enumerate(suite_cfg.splits, 1):
        print(f"\n  [{i}/{len(suite_cfg.splits)}] {split.name}")
        
        try:
            # Run model on this split
            train_metrics, test_metrics = run_core_simple_oos_once(
                prices=prices,
                config=config,
                available_factor_families=available_factor_families,
                train_start=split.train_start_ts,
                train_end=split.train_end_ts,
                test_start=split.test_start_ts,
                test_end=split.test_end_ts,
                model_config=model_config,
                args=args,
                verbose=True,
            )
            
            # Compute EW baseline for test period
            ew_metrics = compute_ew_metrics_for_period(
                prices=prices,
                start_date=split.test_start_ts,
                end_date=split.test_end_ts,
                benchmark=BENCHMARK,
                rebalance_freq=REBALANCE_FREQ,
            )
            
            results.append(OOSSplitResult(
                split_name=split.name,
                train_start=split.train_start,
                train_end=split.train_end,
                test_start=split.test_start,
                test_end=split.test_end,
                core_sharpe_train=train_metrics.sharpe_ratio,
                core_sharpe_test=test_metrics.sharpe_ratio,
                core_ic_test=test_metrics.avg_ic,
                core_ann_ret_test=test_metrics.annualized_return,
                core_maxdd_test=test_metrics.max_drawdown,
                core_hit_test=test_metrics.hit_rate,
                core_turnover_test=test_metrics.avg_turnover if test_metrics.avg_turnover else 0.0,
                ew_sharpe_test=ew_metrics.sharpe_ratio,
                ew_ann_ret_test=ew_metrics.annualized_return,
            ))
            
            print(f"      Train Sharpe: {train_metrics.sharpe_ratio:.3f}, "
                  f"Test Sharpe: {test_metrics.sharpe_ratio:.3f}, "
                  f"EW Sharpe: {ew_metrics.sharpe_ratio:.3f}")
            
        except Exception as e:
            print(f"      [Error] Failed to run split: {e}")
            continue
    
    print(f"\n  Completed {len(results)}/{len(suite_cfg.splits)} splits successfully.")
    
    return results


def print_core_oos_robustness_summary(results: list[OOSSplitResult]) -> None:
    """
    Print a summary table and decay analysis for the OOS robustness suite.
    """
    if not results:
        print("\n  No results to display.")
        return
    
    print("\n" + "=" * 120)
    print("  CORE FACTOR OOS ROBUSTNESS SUITE RESULTS")
    print("=" * 120)
    
    # Header
    header = (
        f"{'Split':<35} | {'Train Shp':>9} | {'Test Shp':>8} | "
        f"{'Test IC':>8} | {'EW Shp':>8} | {'dShp(EW)':>8} | "
        f"{'Ann.Ret':>8} | {'MaxDD':>8} | {'TO%':>6}"
    )
    print(header)
    print("-" * 120)
    
    # Rows
    for r in results:
        ic_str = f"{r.core_ic_test:.4f}" if r.core_ic_test is not None else "N/A"
        delta_sharpe = r.core_sharpe_test - r.ew_sharpe_test
        delta_str = f"{delta_sharpe:+.3f}"
        
        row = (
            f"{r.split_name:<35} | {r.core_sharpe_train:>9.3f} | "
            f"{r.core_sharpe_test:>8.3f} | {ic_str:>8} | "
            f"{r.ew_sharpe_test:>8.3f} | {delta_str:>8} | "
            f"{r.core_ann_ret_test * 100:>7.1f}% | {r.core_maxdd_test * 100:>7.1f}% | "
            f"{r.core_turnover_test * 100:>5.1f}"
        )
        print(row)
    
    print("=" * 120)
    
    # Summary statistics
    avg_test_sharpe = sum(r.core_sharpe_test for r in results) / len(results)
    avg_train_sharpe = sum(r.core_sharpe_train for r in results) / len(results)
    avg_delta = sum(r.core_sharpe_test - r.ew_sharpe_test for r in results) / len(results)
    
    print(f"\n  Average Test Sharpe: {avg_test_sharpe:.3f}")
    print(f"  Average Train Sharpe: {avg_train_sharpe:.3f}")
    print(f"  Average Delta vs EW: {avg_delta:+.3f}")
    
    # Count how many splits beat EW
    beats_ew = sum(1 for r in results if r.core_sharpe_test > r.ew_sharpe_test)
    print(f"  Splits beating EW: {beats_ew}/{len(results)}")
    
    # Sharpe decay analysis
    print("\n  Sharpe Decay (Train -> Test):")
    print("  " + "-" * 80)
    
    for r in results:
        if r.core_sharpe_train > 0:
            decay_pct = (r.core_sharpe_train - r.core_sharpe_test) / r.core_sharpe_train * 100
            print(f"    {r.split_name}: {decay_pct:.1f}% decay "
                  f"(Train {r.core_sharpe_train:.3f} -> Test {r.core_sharpe_test:.3f})")
    
    # Interpretation
    print("\n  Interpretation:")
    
    all_beat_ew = beats_ew == len(results)
    avg_decay = sum(
        (r.core_sharpe_train - r.core_sharpe_test) / r.core_sharpe_train * 100
        for r in results if r.core_sharpe_train > 0
    ) / len(results)
    
    if all_beat_ew and avg_decay < 50:
        print("    [+] ROBUST: Model beats EW in all splits with moderate Sharpe decay.")
    elif beats_ew >= len(results) * 0.66 and avg_decay < 60:
        print("    [+] MOSTLY ROBUST: Model beats EW in most splits, acceptable decay.")
    elif avg_decay > 70:
        print("    [!] HIGH DECAY: Train->Test Sharpe drops significantly, possible overfitting.")
    else:
        print("    [~] MIXED: Some splits perform well, others struggle. Use with caution.")


def run_core_factor_oos_experiment(
    prices: pd.DataFrame,
    config: PipelineConfig,
    core_factor_config: CoreFactorConfig,
    available_factor_families: dict,
    args,
) -> None:
    """
    Run out-of-sample robustness experiment for Core Factors [Simple] model.
    
    This function:
    1. Uses only the training window (2013-2018) for feature selection
    2. Trains model on training window only
    3. Evaluates on held-out test window (2019-2023)
    4. Compares train vs test performance
    
    Args:
        prices: Full price DataFrame.
        config: Base pipeline configuration.
        core_factor_config: Configuration for core factor selection.
        available_factor_families: Dict of which fundamental families are available.
        args: Command-line arguments.
    """
    print("\n" + "=" * 76)
    print("  RUNNING CORE FACTOR OOS EXPERIMENT (Train 2013-2018, Test 2019-2023)")
    print("=" * 76)
    
    # -------------------------------------------------------------------------
    # Step 1: Build config for fundamentals
    # -------------------------------------------------------------------------
    oos_config = PipelineConfig(
        research=config.research,
        features=FeatureConfig(
            momentum_windows=(21, 63, 126, 252),
            volatility_windows=(21, 63),
            include_fundamentals=True,
            include_sentiment=False,
            fundamentals_csv_path=args.fundamentals_csv if hasattr(args, 'fundamentals_csv') else None,
            fundamental_factor_families=available_factor_families,
            fundamentals_source=args.fundamentals_source if hasattr(args, 'fundamentals_source') else 'finagg',
            finagg_mode=args.finagg_mode if hasattr(args, 'finagg_mode') else 'quarterly_refined',
        ),
        model=SIMPLE_MODEL_CONFIG,  # Use simplified model
    )
    
    # -------------------------------------------------------------------------
    # Step 2: Feature selection using TRAINING WINDOW ONLY
    # -------------------------------------------------------------------------
    print("\n  [1/4] Extracting feature importance from TRAINING period only...")
    
    # Filter prices to training window
    train_mask = (prices.index >= OOS_TRAIN_START) & (prices.index <= OOS_TRAIN_END)
    train_prices = prices.loc[train_mask]
    
    print(f"        Training period: {OOS_TRAIN_START.date()} to {OOS_TRAIN_END.date()}")
    print(f"        Training days: {len(train_prices)}")
    
    # Compute importance by subperiod within training window
    importance_by_subperiod = compute_feature_importance_by_subperiod(
        prices=train_prices,
        config=oos_config,
        subperiods=OOS_TRAIN_SUBPERIODS,
        benchmark=BENCHMARK,
        horizon_days=HORIZON_DAYS,
    )
    
    if len(importance_by_subperiod) < 1:
        print("  [Error] Could not compute importance for any subperiods. Aborting OOS experiment.")
        return
    
    if len(importance_by_subperiod) < 2:
        print(f"  [Warning] Only {len(importance_by_subperiod)} subperiod(s) available. Relaxing min_subperiod_support to 1.")

    # Select core features using training-only importances
    # Dynamically adjust min_subperiod_support based on available subperiods
    adjusted_core_config = CoreFactorConfig(
        min_subperiod_support=min(core_factor_config.min_subperiod_support, len(importance_by_subperiod)),
        min_importance_threshold=core_factor_config.min_importance_threshold,
        min_mean_importance=core_factor_config.min_mean_importance,
    )
    core_features, importance_df = select_core_features(
        importance_by_subperiod,
        core_config=adjusted_core_config,
    )
    
    print(f"\n  [2/4] Selected {len(core_features)} core features from training data:")
    for i, feat in enumerate(core_features[:10]):
        print(f"        {i+1:2d}. {feat}")
    if len(core_features) > 10:
        print(f"        ... and {len(core_features) - 10} more")
    
    if not core_features:
        print("  [Error] No core features selected. Aborting OOS experiment.")
        return
    
    # -------------------------------------------------------------------------
    # Step 3: Train model on TRAINING WINDOW ONLY
    # -------------------------------------------------------------------------
    print("\n  [3/4] Training Core Factors [Simple] model on training period...")
    
    # Prepare training data from training period only
    try:
        X_train, y_train = prepare_training_data(
            train_prices,
            horizon_days=HORIZON_DAYS,
            benchmark=BENCHMARK,
            config=oos_config.features,
        )
    except Exception as e:
        print(f"  [Error] Failed to prepare training data: {e}")
        return
    
    # Filter to core features
    available_core = [f for f in core_features if f in X_train.columns]
    if len(available_core) < len(core_features) * 0.5:
        print(f"  [Warning] Only {len(available_core)}/{len(core_features)} core features available in training data.")
    
    X_train_core = X_train[available_core].fillna(0)
    
    print(f"        Training samples: {len(X_train_core):,}")
    print(f"        Features used: {len(available_core)}")
    
    # Train the model
    model = GBDTModel(SIMPLE_MODEL_CONFIG)
    model.fit(X_train_core, y_train)
    
    print("        Model trained successfully.")
    
    # -------------------------------------------------------------------------
    # Step 4: Evaluate on both TRAIN and TEST periods
    # -------------------------------------------------------------------------
    print("\n  [4/4] Evaluating model on train and test periods...")
    
    # Evaluate on training period (in-sample)
    try:
        train_result, _ = run_oos_backtest_fixed_model(
            prices=prices,
            trained_model=model,
            core_features=available_core,
            config=oos_config,
            start_date=OOS_TRAIN_START,
            end_date=OOS_TRAIN_END,
            benchmark=BENCHMARK,
            top_k=TOP_K,
            horizon_days=HORIZON_DAYS,
            rebalance_freq=REBALANCE_FREQ,
            cost_config=COST_MODEL,
        )
        
        train_metrics = OOSMetrics(
            period_name="Train 2013-2018",
            annualized_return=train_result.annualized_return,
            sharpe_ratio=train_result.sharpe_ratio,
            max_drawdown=train_result.max_drawdown,
            hit_rate=train_result.hit_rate,
            avg_ic=train_result.ic,
            ic_std=train_result.ic_std,
            avg_turnover=train_result.avg_turnover,
            cumulative_return=train_result.cumulative_return,
            annualized_volatility=train_result.annualized_volatility,
        )
        train_ic_str = f"{train_metrics.avg_ic:.4f}" if train_metrics.avg_ic is not None else "N/A"
        print(f"        Train period: Sharpe={train_metrics.sharpe_ratio:.3f}, IC={train_ic_str}")
        
    except Exception as e:
        print(f"  [Error] Failed to evaluate on training period: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on test period (out-of-sample)
    try:
        test_result, _ = run_oos_backtest_fixed_model(
            prices=prices,
            trained_model=model,
            core_features=available_core,
            config=oos_config,
            start_date=OOS_TEST_START,
            end_date=OOS_TEST_END,
            benchmark=BENCHMARK,
            top_k=TOP_K,
            horizon_days=HORIZON_DAYS,
            rebalance_freq=REBALANCE_FREQ,
            cost_config=COST_MODEL,
        )
        
        test_metrics = OOSMetrics(
            period_name="Test 2019-2023",
            annualized_return=test_result.annualized_return,
            sharpe_ratio=test_result.sharpe_ratio,
            max_drawdown=test_result.max_drawdown,
            hit_rate=test_result.hit_rate,
            avg_ic=test_result.ic,
            ic_std=test_result.ic_std,
            avg_turnover=test_result.avg_turnover,
            cumulative_return=test_result.cumulative_return,
            annualized_volatility=test_result.annualized_volatility,
        )
        test_ic_str = f"{test_metrics.avg_ic:.4f}" if test_metrics.avg_ic is not None else "N/A"
        print(f"        Test period:  Sharpe={test_metrics.sharpe_ratio:.3f}, IC={test_ic_str}")
        
    except Exception as e:
        print(f"  [Error] Failed to evaluate on test period: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute EW baseline for test period
    try:
        ew_test_metrics = compute_ew_metrics_for_period(
            prices=prices,
            start_date=OOS_TEST_START,
            end_date=OOS_TEST_END,
            benchmark=BENCHMARK,
            rebalance_freq=REBALANCE_FREQ,
        )
        print(f"        EW Test:      Sharpe={ew_test_metrics.sharpe_ratio:.3f}")
    except Exception as e:
        print(f"  [Warning] Could not compute EW baseline: {e}")
        ew_test_metrics = None
    
    # -------------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------------
    print_oos_summary_table(train_metrics, test_metrics, ew_test_metrics)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Large-Cap Universe Backtest for Auto-Researcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Technical-only ML (default)
  python scripts/run_large_cap_backtest.py

  # Tech + Fundamentals with CSV data
  python scripts/run_large_cap_backtest.py --use-fundamentals --fundamentals-csv data/fundamentals.csv

  # Tech + Fundamentals with SEC EDGAR via finagg
  python scripts/run_large_cap_backtest.py --use-fundamentals --fundamentals-source finagg
        """,
    )
    
    parser.add_argument(
        "--use-fundamentals",
        action="store_true",
        default=False,
        help="Enable fundamental factors in the ML strategy. When combined with --fundamentals-csv, "
             "runs a comparison between Tech-only and Tech+Fundamentals.",
    )
    
    parser.add_argument(
        "--fundamentals-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to CSV file containing fundamental data. "
             "Expected columns: date, ticker, market_cap, pe_ratio, pb_ratio, etc.",
    )
    
    parser.add_argument(
        "--fundamentals-source",
        type=str,
        choices=["csv", "finagg", "yfinance"],
        default="csv",
        metavar="SOURCE",
        help="Data source for fundamentals: 'csv' (default), 'finagg' (SEC EDGAR), "
             "or 'yfinance'. For finagg, requires: pip install finagg; "
             "export SEC_API_USER_AGENT='Your Name email@example.com'",
    )
    
    parser.add_argument(
        "--finagg-mode",
        type=str,
        choices=["quarterly_refined", "quarterly_api", "annual_refined"],
        default="quarterly_refined",
        metavar="MODE",
        help="Finagg data mode: 'quarterly_refined' (local DB, fast), "
             "'quarterly_api' (remote API, slow), or 'annual_refined'.",
    )
    
    parser.add_argument(
        "--enable-subperiods",
        action="store_true",
        dest="enable_subperiods",
        default=True,
        help="Enable subperiod performance analysis (default: enabled).",
    )
    parser.add_argument(
        "--no-subperiods",
        action="store_false",
        dest="enable_subperiods",
        help="Disable subperiod performance analysis.",
    )
    
    parser.add_argument(
        "--regime-mode",
        type=str,
        choices=["none", "feature", "split"],
        default="none",
        metavar="MODE",
        help="Regime-aware ML mode: 'none' (default, current behavior), "
             "'feature' (single model with regime_id feature), "
             "'split' (separate model per regime). "
             "Regimes: 2013-2016, 2017-2019, 2020-2023.",
    )
    
    parser.add_argument(
        "--use-core-factors",
        action="store_true",
        default=False,
        help="Extract stable 'core factors' across subperiods and run a strategy "
             "using only those features. Requires --use-fundamentals.",
    )
    
    parser.add_argument(
        "--simple-model",
        action="store_true",
        default=False,
        help="Use simplified model config (shallower trees, fewer estimators) "
             "when running core factor strategy. Reduces overfitting risk.",
    )
    
    parser.add_argument(
        "--oos-mode",
        type=str,
        choices=["none", "train_2013_2018_test_2019_2023"],
        default="none",
        metavar="MODE",
        help="Out-of-sample robustness mode for Core Factors [Simple] model: "
             "'none' (default, no OOS analysis), "
             "'train_2013_2018_test_2019_2023' (train on 2013-2018, test on 2019-2023).",
    )
    
    parser.add_argument(
        "--reg-sweep-core-simple",
        action="store_true",
        default=False,
        help="Run a regularization hyperparameter sweep for the Core Factors [Simple] model. "
             "Requires --use-fundamentals, --use-core-factors, --simple-model. "
             "When enabled, skips the normal single-model backtest and runs a grid search "
             "over regularization settings (max_depth, n_estimators, reg_lambda, subsample, colsample).",
    )
    
    parser.add_argument(
        "--core-oos-suite",
        action="store_true",
        default=False,
        help="Run multi-split OOS robustness suite for Core Factors [Simple] model. "
             "Requires --use-fundamentals, --use-core-factors, --simple-model. "
             "Tests the model across multiple train/test splits to verify robustness. "
             "Skips the normal multi-strategy backtest when enabled.",
    )
    
    # -------------------------------------------------------------------------
    # Fundamentals Coverage Filtering
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--min-fundamentals-coverage",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Minimum average fundamental data coverage (0.0-1.0) for a ticker "
             "to be included with fundamentals. Default 0.0 (no filtering). "
             "Example: 0.4 requires 40%% non-NaN fundamentals on average.",
    )
    
    parser.add_argument(
        "--min-fundamentals-test-year",
        type=int,
        default=None,
        metavar="YEAR",
        help="If set, also require minimum coverage for years >= this value. "
             "Useful to ensure good recent data quality. Example: 2017.",
    )
    
    parser.add_argument(
        "--fundamentals-universe-mode",
        type=str,
        choices=["full", "good_only", "hybrid"],
        default="full",
        metavar="MODE",
        help="How to handle tickers with poor fundamental coverage: "
             "'full' (default) = use all tickers regardless of coverage; "
             "'good_only' = restrict ML universe to tickers passing coverage filter; "
             "'hybrid' = keep full universe but set fundamentals to NaN for poor-coverage tickers.",
    )
    
    return parser.parse_args()


# ==============================================================================
# SUBPERIOD ANALYSIS
# ==============================================================================

# Default reference strategy for relative metrics
REFERENCE_STRATEGY = "Equal-Weight Universe"


def compute_subperiod_metrics(
    strategy_returns: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    ic_series_by_strategy: dict[str, pd.Series | None] | None,
    subperiods: list[Subperiod],
    periods_per_year: int = 12,
    reference_strategy: str = REFERENCE_STRATEGY,
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    """
    Compute performance metrics for each strategy across subperiods.
    
    Args:
        strategy_returns: Dict mapping strategy name -> periodic returns Series.
        benchmark_returns: Benchmark (SPY) periodic returns Series.
        ic_series_by_strategy: Dict mapping strategy name -> IC Series (or None).
        subperiods: List of Subperiod namedtuples defining analysis windows.
        periods_per_year: Number of return periods per year (12 for monthly).
        reference_strategy: Strategy name to use as reference for relative metrics.
    
    Returns:
        Tuple of:
        - Dict mapping strategy_name -> list of metric dicts, one per subperiod.
          Each metric dict contains: subperiod, start, end, ann_ret, vol, sharpe,
          hit_rate, max_dd, avg_ic, excess_ret, active_sharpe.
        - Boolean indicating if relative metrics are available (reference found).
    """
    results: dict[str, list[dict[str, Any]]] = {}
    has_reference = reference_strategy in strategy_returns
    
    # First pass: compute absolute metrics for all strategies
    for strategy_name, returns in strategy_returns.items():
        strategy_metrics: list[dict[str, Any]] = []
        
        for sub in subperiods:
            # Slice returns to subperiod
            mask = (returns.index >= sub.start) & (returns.index <= sub.end)
            sub_ret = returns[mask]
            
            # Slice benchmark
            bench_mask = (benchmark_returns.index >= sub.start) & (benchmark_returns.index <= sub.end)
            sub_bm = benchmark_returns[bench_mask]
            
            # Check if we have enough data
            if len(sub_ret) < 6:  # Minimum 6 periods (6 months)
                logger.warning(
                    f"Insufficient data for {strategy_name} in {sub.name}: "
                    f"{len(sub_ret)} periods"
                )
                strategy_metrics.append({
                    "subperiod": sub.name,
                    "start": sub.start,
                    "end": sub.end,
                    "ann_ret": np.nan,
                    "vol": np.nan,
                    "sharpe": np.nan,
                    "hit_rate": np.nan,
                    "max_dd": np.nan,
                    "avg_ic": np.nan,
                    "excess_ret": np.nan,
                    "active_sharpe": np.nan,
                })
                continue
            
            # Compute metrics using existing helpers
            ann_ret = compute_annualized_return(sub_ret, periods_per_year)
            vol = compute_annualized_volatility(sub_ret, periods_per_year)
            sharpe = compute_sharpe_ratio(sub_ret, 0.0, periods_per_year)
            hit_rate = compute_hit_rate(sub_ret, sub_bm) if len(sub_bm) > 0 else np.nan
            max_dd = compute_max_drawdown(sub_ret)
            
            # Slice IC series if available
            avg_ic = None
            if ic_series_by_strategy and ic_series_by_strategy.get(strategy_name) is not None:
                ic_series = ic_series_by_strategy[strategy_name]
                ic_mask = (ic_series.index >= sub.start) & (ic_series.index <= sub.end)
                sub_ic = ic_series[ic_mask]
                if len(sub_ic) > 0:
                    avg_ic = sub_ic.mean()
            
            strategy_metrics.append({
                "subperiod": sub.name,
                "start": sub.start,
                "end": sub.end,
                "ann_ret": ann_ret,
                "vol": vol,
                "sharpe": sharpe,
                "hit_rate": hit_rate,
                "max_dd": max_dd,
                "avg_ic": avg_ic,
                "excess_ret": np.nan,  # Placeholder, filled in second pass
                "active_sharpe": np.nan,
            })
        
        results[strategy_name] = strategy_metrics
    
    # Second pass: compute relative metrics vs reference strategy
    if has_reference:
        ref_metrics = results[reference_strategy]
        for strategy_name, strategy_metrics in results.items():
            for i, metrics in enumerate(strategy_metrics):
                ref_ann_ret = ref_metrics[i]["ann_ret"]
                ref_sharpe = ref_metrics[i]["sharpe"]
                
                # Compute excess return (strategy - reference)
                if not np.isnan(metrics["ann_ret"]) and not np.isnan(ref_ann_ret):
                    metrics["excess_ret"] = metrics["ann_ret"] - ref_ann_ret
                else:
                    metrics["excess_ret"] = np.nan
                
                # Compute active Sharpe (strategy Sharpe - reference Sharpe)
                if not np.isnan(metrics["sharpe"]) and not np.isnan(ref_sharpe):
                    metrics["active_sharpe"] = metrics["sharpe"] - ref_sharpe
                else:
                    metrics["active_sharpe"] = np.nan
    else:
        logger.warning(
            f"Reference strategy '{reference_strategy}' not found. "
            "Skipping relative metrics."
        )
    
    return results, has_reference


def _format_signed_pct(value: float, width: int = 7) -> str:
    """Format a percentage value with +/- sign, or N/A if NaN."""
    if np.isnan(value):
        return "N/A".rjust(width)
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}%".rjust(width)


def _format_signed_float(value: float, width: int = 7, decimals: int = 3) -> str:
    """Format a float value with +/- sign, or N/A if NaN."""
    if np.isnan(value):
        return "N/A".rjust(width)
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}".rjust(width)


def print_subperiod_table(
    subperiod_metrics: dict[str, list[dict[str, Any]]],
    subperiods: list[Subperiod],
    has_relative_metrics: bool = True,
) -> None:
    """
    Print a formatted table of subperiod metrics.
    
    Args:
        subperiod_metrics: Output from compute_subperiod_metrics().
        subperiods: List of Subperiod namedtuples for headers.
        has_relative_metrics: If True, include ExRet and ActSharpe columns.
    """
    # Determine table width based on whether relative metrics are shown
    table_width = 120 if has_relative_metrics else 100
    
    print("\n" + "=" * table_width)
    print("  SUBPERIOD PERFORMANCE ANALYSIS")
    print("=" * table_width)
    
    strategy_names = list(subperiod_metrics.keys())
    
    for i, sub in enumerate(subperiods):
        print(f"\n  Subperiod: {sub.name} ({sub.start.strftime('%Y-%m-%d')} to {sub.end.strftime('%Y-%m-%d')})")
        print("-" * table_width)
        
        # Header - with or without relative metrics columns
        if has_relative_metrics:
            header = (
                f"{'Strategy':<30} | {'Ann.Ret':>9} | {'Sharpe':>7} | "
                f"{'ExRet':>8} | {'ActShp':>7} | "
                f"{'Hit%':>7} | {'MaxDD':>8} | {'AvgIC':>8}"
            )
        else:
            header = f"{'Strategy':<30} | {'Ann.Ret':>9} | {'Sharpe':>7} | {'Hit%':>7} | {'MaxDD':>8} | {'AvgIC':>8}"
        print(header)
        print("-" * table_width)
        
        # Rows for each strategy
        for name in strategy_names:
            metrics = subperiod_metrics[name][i]
            
            # Format absolute values, handling NaN gracefully
            ann_ret_str = f"{metrics['ann_ret'] * 100:>8.2f}%" if not np.isnan(metrics['ann_ret']) else "     N/A"
            sharpe_str = f"{metrics['sharpe']:>7.3f}" if not np.isnan(metrics['sharpe']) else "    N/A"
            hit_str = f"{metrics['hit_rate'] * 100:>6.1f}%" if not np.isnan(metrics['hit_rate']) else "    N/A"
            max_dd_str = f"{metrics['max_dd'] * 100:>7.1f}%" if not np.isnan(metrics['max_dd']) else "    N/A"
            
            if metrics['avg_ic'] is not None and not np.isnan(metrics['avg_ic']):
                ic_str = f"{metrics['avg_ic']:>8.4f}"
            else:
                ic_str = "     N/A"
            
            if has_relative_metrics:
                # Format relative metrics with +/- sign
                ex_ret_str = _format_signed_pct(metrics.get('excess_ret', np.nan), width=8)
                act_sharpe_str = _format_signed_float(metrics.get('active_sharpe', np.nan), width=7, decimals=3)
                
                print(
                    f"{name:<30} | {ann_ret_str} | {sharpe_str} | "
                    f"{ex_ret_str} | {act_sharpe_str} | "
                    f"{hit_str} | {max_dd_str} | {ic_str}"
                )
            else:
                print(f"{name:<30} | {ann_ret_str} | {sharpe_str} | {hit_str} | {max_dd_str} | {ic_str}")
    
    print("\n" + "=" * table_width)
    if has_relative_metrics:
        print(f"  Note: ExRet and ActShp are relative to '{REFERENCE_STRATEGY}'.")
    print("  Note: Subperiod metrics use the same rebalance schedule as the full sample, sliced by date.")
    print("=" * table_width)


def compute_subperiod_consistency(
    subperiod_metrics: dict[str, list[dict[str, Any]]],
    reference_strategy: str = REFERENCE_STRATEGY,
) -> dict[str, dict[str, tuple[int, int]]]:
    """
    Compute consistency counts: how many subperiods each strategy beat the reference.
    
    Args:
        subperiod_metrics: Output from compute_subperiod_metrics().
        reference_strategy: The strategy to exclude from the summary.
    
    Returns:
        Dict mapping strategy_name -> {
            'excess_ret': (wins, valid_count),
            'active_sharpe': (wins, valid_count)
        }
        where wins = number of subperiods with metric > 0,
        and valid_count = number of subperiods where metric is not NaN.
    """
    results: dict[str, dict[str, tuple[int, int]]] = {}
    
    for strategy_name, metrics_list in subperiod_metrics.items():
        # Skip the reference strategy itself
        if strategy_name == reference_strategy:
            continue
        
        ex_ret_wins = 0
        ex_ret_valid = 0
        act_sharpe_wins = 0
        act_sharpe_valid = 0
        
        for metrics in metrics_list:
            # Excess return
            ex_ret = metrics.get('excess_ret', np.nan)
            if not np.isnan(ex_ret):
                ex_ret_valid += 1
                if ex_ret > 0:
                    ex_ret_wins += 1
            
            # Active Sharpe
            act_sharpe = metrics.get('active_sharpe', np.nan)
            if not np.isnan(act_sharpe):
                act_sharpe_valid += 1
                if act_sharpe > 0:
                    act_sharpe_wins += 1
        
        results[strategy_name] = {
            'excess_ret': (ex_ret_wins, ex_ret_valid),
            'active_sharpe': (act_sharpe_wins, act_sharpe_valid),
        }
    
    return results


def print_subperiod_consistency_summary(
    consistency: dict[str, dict[str, tuple[int, int]]],
    reference_strategy: str = REFERENCE_STRATEGY,
) -> None:
    """
    Print a summary table showing how consistently each strategy beat the reference.
    
    Args:
        consistency: Output from compute_subperiod_consistency().
        reference_strategy: Name of the reference strategy (for display only).
    """
    if not consistency:
        return
    
    table_width = 80
    
    print("\n" + "=" * table_width)
    print(f"  SUBPERIOD CONSISTENCY SUMMARY (vs {reference_strategy})")
    print("=" * table_width)
    
    # Header
    header = f"{'Strategy':<30} | {'Beat EW on Ann.Ret':^20} | {'Beat EW on Sharpe':^20}"
    print(header)
    print("-" * table_width)
    
    # Rows for each strategy (sorted for consistent output)
    for name in sorted(consistency.keys()):
        counts = consistency[name]
        
        ex_wins, ex_total = counts['excess_ret']
        sharpe_wins, sharpe_total = counts['active_sharpe']
        
        # Format as "X / Y" or "N/A" if no valid subperiods
        if ex_total > 0:
            ex_ret_str = f"{ex_wins} / {ex_total}"
        else:
            ex_ret_str = "N/A"
        
        if sharpe_total > 0:
            sharpe_str = f"{sharpe_wins} / {sharpe_total}"
        else:
            sharpe_str = "N/A"
        
        print(f"{name:<30} | {ex_ret_str:^20} | {sharpe_str:^20}")
    
    print("=" * table_width)


# Strategy name variants for interpretation matching
# Include regime-aware variants with [RegFeature] and [RegSplit] suffixes
ML_TECH_KEYS = (
    "ML Tech Only",
    "ML Tech Only [RegFeature]",
    "ML Tech Only [RegSplit]",
    "ML Tech",
    "ML (GBDT)",
    "ML (GBDT) [RegFeature]",
    "ML (GBDT) [RegSplit]",
)
ML_FUND_KEYS = (
    "ML Tech + Fundamentals",
    "ML Tech + Fundamentals [RegFeature]",
    "ML Tech + Fundamentals [RegSplit]",
    "ML+Fundamentals",
    "ML (GBDT) w/ Fundamentals",
    "ML (GBDT) w/ Fundamentals [RegFeature]",
    "ML (GBDT) w/ Fundamentals [RegSplit]",
)
ML_QUALITY_KEYS = (
    "ML Quality Only",
    "ML Quality Only [RegFeature]",
    "ML Quality Only [RegSplit]",
)
ML_REGIME_SWITCH_KEYS = (
    "ML Regime Switch",
    "ML Regime Switch [RegSwitch]",
)
ML_IC_BLEND_KEYS = (
    "ML IC-Blend",
    "ML IC-Blend (Mom+Qual)",
    "ML IC-Blend (Mom+Qual) [ICBlend]",
    "ML IC-Weighted Ensemble",
)


def _find_strategy_key(
    subperiod_metrics: dict[str, list[dict[str, Any]]],
    candidates: tuple[str, ...],
) -> str | None:
    """Find the first matching strategy key from a list of candidates."""
    for key in candidates:
        if key in subperiod_metrics:
            return key
    return None


def _classify_fundamentals_impact(delta_sharpe: float, delta_exret: float) -> str:
    """
    Classify the impact of fundamentals vs tech-only based on delta metrics.
    
    Rules (thresholds chosen for interpretability):
    - delta_sharpe > 0.5 and delta_exret > 0: "strongly additive"
    - 0.2 < delta_sharpe <= 0.5 and delta_exret > 0: "clearly additive"
    - -0.2 <= delta_sharpe <= 0.2: "roughly neutral"
    - delta_sharpe < -0.2: "negative (tech-only preferred)"
    """
    if delta_sharpe > 0.5 and delta_exret > 0:
        return "Fundamentals strongly additive"
    elif delta_sharpe > 0.2 and delta_exret > 0:
        return "Fundamentals clearly additive"
    elif -0.2 <= delta_sharpe <= 0.2:
        return "Fundamentals roughly neutral"
    else:
        return "Fundamentals negative (tech-only preferred)"


def print_subperiod_interpretation(
    subperiod_metrics: dict[str, list[dict[str, Any]]],
    subperiods: list[Subperiod],
) -> None:
    """
    Print a rule-based interpretation comparing ML+Fundamentals vs ML Tech-Only.
    
    Args:
        subperiod_metrics: Output from compute_subperiod_metrics().
        subperiods: List of Subperiod namedtuples for labels.
    """
    # Find strategy keys
    ml_tech_key = _find_strategy_key(subperiod_metrics, ML_TECH_KEYS)
    ml_fund_key = _find_strategy_key(subperiod_metrics, ML_FUND_KEYS)
    
    # Skip if either strategy is missing
    if ml_tech_key is None or ml_fund_key is None:
        return
    
    tech_metrics = subperiod_metrics[ml_tech_key]
    fund_metrics = subperiod_metrics[ml_fund_key]
    
    # Compute per-subperiod deltas
    interpretations: list[tuple[str, str, float, float]] = []  # (label, phrase, delta_sharpe, delta_exret)
    additive_count = 0
    neutral_or_negative_count = 0
    
    for i, sub in enumerate(subperiods):
        tech_m = tech_metrics[i]
        fund_m = fund_metrics[i]
        
        # Skip if metrics are NaN
        if np.isnan(tech_m['sharpe']) or np.isnan(fund_m['sharpe']):
            continue
        
        delta_sharpe = fund_m['sharpe'] - tech_m['sharpe']
        
        # For excess return delta: compare fund vs tech (not vs EW)
        # Use ann_ret difference since both are relative to same baseline
        if not np.isnan(tech_m['ann_ret']) and not np.isnan(fund_m['ann_ret']):
            delta_exret = fund_m['ann_ret'] - tech_m['ann_ret']
        else:
            delta_exret = 0.0
        
        phrase = _classify_fundamentals_impact(delta_sharpe, delta_exret)
        interpretations.append((sub.name, phrase, delta_sharpe, delta_exret))
        
        # Count for overall summary
        if "additive" in phrase:
            additive_count += 1
        else:
            neutral_or_negative_count += 1
    
    if not interpretations:
        return
    
    # Print interpretation block
    table_width = 100
    print("\n" + "=" * table_width)
    print("  SUBPERIOD INTERPRETATION (Fundamentals vs Tech-Only)")
    print("=" * table_width)
    
    for label, phrase, delta_sharpe, delta_exret in interpretations:
        # Format delta values with +/- sign
        sharpe_sign = "+" if delta_sharpe >= 0 else ""
        exret_sign = "+" if delta_exret >= 0 else ""
        
        line = (
            f"  {label}: {phrase} "
            f"(dSharpe {sharpe_sign}{delta_sharpe:.2f}, "
            f"dRet {exret_sign}{delta_exret * 100:.1f}% vs {ml_tech_key})"
        )
        print(line)
    
    # Overall summary
    print("")
    total = additive_count + neutral_or_negative_count
    
    if additive_count == total:
        overall = "Fundamentals are consistently additive across all subperiods."
    elif neutral_or_negative_count == total:
        overall = "Fundamentals provide no consistent benefit; tech-only may be preferred."
    elif additive_count > neutral_or_negative_count:
        # Identify which periods are additive vs not
        additive_periods = [label for label, phrase, _, _ in interpretations if "additive" in phrase]
        other_periods = [label for label, phrase, _, _ in interpretations if "additive" not in phrase]
        overall = (
            f"Overall: Fundamentals are regime-dependent; "
            f"additive in {', '.join(additive_periods)}, "
            f"neutral/negative in {', '.join(other_periods)}."
        )
    else:
        additive_periods = [label for label, phrase, _, _ in interpretations if "additive" in phrase]
        other_periods = [label for label, phrase, _, _ in interpretations if "additive" not in phrase]
        overall = (
            f"Overall: Fundamentals show limited benefit; "
            f"additive only in {', '.join(additive_periods) if additive_periods else 'no periods'}, "
            f"neutral/negative in {', '.join(other_periods)}."
        )
    
    print(f"  {overall}")
    print("=" * table_width)


def print_regime_switch_interpretation(results: list[StrategyResult]) -> None:
    """
    Print interpretation of the regime-switch strategy.
    
    Args:
        results: List of all strategy results.
    """
    # Find regime-switch strategy
    regime_switch = None
    ml_tech = None
    ml_quality = None
    equal_weight = None
    
    for r in results:
        if "Regime Switch" in r.name:
            regime_switch = r
        elif any(key in r.name for key in ML_TECH_KEYS):
            ml_tech = r
        elif any(key in r.name for key in ML_QUALITY_KEYS):
            ml_quality = r
        elif "Equal-Weight" in r.name:
            equal_weight = r
    
    if regime_switch is None:
        return
    
    table_width = 100
    print("\n" + "=" * table_width)
    print("  REGIME-SWITCH STRATEGY INTERPRETATION")
    print("=" * table_width)
    
    print("""
  ML Regime Switch (Momentum vs Quality) [RegSwitch]:
    - Uses Tech-only (Momentum) model in low-vol, up-trend regimes
    - Uses Quality-only (Fundamentals) model in high-vol or down-trend regimes
    - Designed to stabilize performance across different market environments
""")
    
    # Compare to component strategies
    if ml_tech is not None and ml_quality is not None:
        print("  Component Model Comparison:")
        print(f"    ML Tech (Momentum):   Sharpe={ml_tech.sharpe_ratio:.3f}")
        print(f"    ML Quality:           Sharpe={ml_quality.sharpe_ratio:.3f}")
        print(f"    ML Regime Switch:     Sharpe={regime_switch.sharpe_ratio:.3f}")
        
        # Analyze if switching adds value
        better_component = max(ml_tech.sharpe_ratio, ml_quality.sharpe_ratio)
        worse_component = min(ml_tech.sharpe_ratio, ml_quality.sharpe_ratio)
        
        if regime_switch.sharpe_ratio > better_component:
            print("    -> Regime switching BEATS both component models!")
        elif regime_switch.sharpe_ratio > worse_component:
            print("    -> Regime switching is between component models (partial benefit)")
        else:
            print("    -> Regime switching underperforms both components (timing issues)")
    
    # Compare to equal-weight
    if equal_weight is not None:
        delta_vs_ew = regime_switch.sharpe_ratio - equal_weight.sharpe_ratio
        print(f"\n  vs Equal-Weight:")
        print(f"    Sharpe Delta: {delta_vs_ew:+.3f}")
        if delta_vs_ew > 0.1:
            print("    -> Regime switching adds significant value over passive")
        elif delta_vs_ew > 0:
            print("    -> Regime switching adds modest value over passive")
        else:
            print("    -> Regime switching underperforms passive (keep it simple)")
    
    print("=" * table_width)


def print_ic_blend_interpretation(results: list[StrategyResult]) -> None:
    """
    Print interpretation of the IC-weighted ensemble strategy.
    
    Args:
        results: List of all strategy results.
    """
    # Find IC-blend strategy
    ic_blend = None
    ml_tech = None
    ml_quality = None
    equal_weight = None
    
    for r in results:
        if any(key in r.name for key in ML_IC_BLEND_KEYS):
            ic_blend = r
        elif any(key in r.name for key in ML_TECH_KEYS):
            ml_tech = r
        elif any(key in r.name for key in ML_QUALITY_KEYS):
            ml_quality = r
        elif "Equal-Weight" in r.name:
            equal_weight = r
    
    if ic_blend is None:
        return
    
    table_width = 100
    print("\n" + "=" * table_width)
    print("  IC-WEIGHTED ENSEMBLE STRATEGY INTERPRETATION")
    print("=" * table_width)
    
    print("""
  ML IC-Weighted Ensemble (Mom+Quality) [ICBlend]:
    - Blends Tech-only (Momentum) and Quality-only (Fundamentals) model scores
    - Weights determined by rolling IC: higher recent IC = higher weight
    - Rolling windows: Momentum=6 months, Quality=12 months
    - Negative ICs clipped to 0; fallback to 50/50 if both non-positive
""")
    
    # Compare to component strategies
    if ml_tech is not None and ml_quality is not None:
        print("  Component Model Comparison:")
        print(f"    ML Tech (Momentum):   Sharpe={ml_tech.sharpe_ratio:.3f}")
        print(f"    ML Quality:           Sharpe={ml_quality.sharpe_ratio:.3f}")
        print(f"    ML IC-Blend:          Sharpe={ic_blend.sharpe_ratio:.3f}")
        
        # Analyze if blending adds value
        better_component = max(ml_tech.sharpe_ratio, ml_quality.sharpe_ratio)
        worse_component = min(ml_tech.sharpe_ratio, ml_quality.sharpe_ratio)
        
        if ic_blend.sharpe_ratio > better_component:
            print("    -> IC-weighted blending BEATS both component models!")
        elif ic_blend.sharpe_ratio > worse_component:
            print("    -> IC-weighted blending is between component models")
        else:
            print("    -> IC-weighted blending underperforms both components")
    
    # Compare to equal-weight
    if equal_weight is not None:
        delta_vs_ew = ic_blend.sharpe_ratio - equal_weight.sharpe_ratio
        print(f"\n  vs Equal-Weight:")
        print(f"    Sharpe Delta: {delta_vs_ew:+.3f}")
        if delta_vs_ew > 0.1:
            print("    -> IC-blending adds significant value over passive")
        elif delta_vs_ew > 0:
            print("    -> IC-blending adds modest value over passive")
        else:
            print("    -> IC-blending underperforms passive")
    
    print("=" * table_width)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Parse arguments
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("  LARGE-CAP UNIVERSE BACKTEST")
    print("  Auto-Researcher for Asset Selection")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Configuration summary
    # -------------------------------------------------------------------------
    print("\n  Configuration:")
    print(f"    Use Fundamentals:    {args.use_fundamentals}")
    print(f"    Fundamentals Source: {args.fundamentals_source}")
    print(f"    Regime Mode:         {args.regime_mode}")
    if args.fundamentals_csv:
        print(f"    Fundamentals CSV:    {args.fundamentals_csv}")
    if args.fundamentals_source == "finagg":
        print(f"    Finagg Mode:         {args.finagg_mode}")
        print("\n    [finagg] Prerequisites:")
        print("      - pip install finagg")
        print("      - export SEC_API_USER_AGENT='Your Name email@example.com'")
        print("      - OR run: finagg sec install")
    
    if args.use_fundamentals:
        if args.fundamentals_source == "finagg":
            # finagg provides more comprehensive fundamentals
            print(f"    Factor Families (SEC EDGAR via finagg):")
            print(f"      - Value:         True   (computed from financials)")
            print(f"      - Quality:       True   (ROE, ROA from filings)")
            print(f"      - Growth:        True   (computed YoY)")
            print(f"      - Profitability: True   (gross, operating margin)")
            print(f"      - Size:          False  (requires price data)")
        else:
            # Alpha Vantage/CSV limitations
            print(f"    Factor Families (Alpha Vantage limited):")
            print(f"      - Value:         False  (no historical P/E, P/B, P/S)")
            print(f"      - Quality:       False  (ROE/ROA require balance sheet alignment)")
            print(f"      - Growth:        False  (no historical growth rates)")
            print(f"      - Profitability: True   (gross, operating margin)")
            print(f"      - Size:          False  (no historical market cap)")
        
        # Print coverage filter settings if applicable
        if args.min_fundamentals_coverage > 0 or args.fundamentals_universe_mode != "full":
            print(f"\n    Coverage Filtering:")
            print(f"      - Universe Mode:     {args.fundamentals_universe_mode}")
            print(f"      - Min Avg Coverage:  {args.min_fundamentals_coverage:.0%}")
            if args.min_fundamentals_test_year:
                print(f"      - Min Test Year:     {args.min_fundamentals_test_year}")
    
    # -------------------------------------------------------------------------
    # Get universe
    # -------------------------------------------------------------------------
    print("\n[1/6] Setting up universe...")
    
    universe = get_large_cap_universe()
    # Add benchmark to universe if not present
    if BENCHMARK not in universe:
        universe = universe + [BENCHMARK]
    
    tradeable_tickers = [t for t in universe if t != BENCHMARK]
    
    print(f"      Universe size: {len(tradeable_tickers)} stocks + SPY benchmark")
    print(f"      Top-K selection: {TOP_K} stocks per rebalance")
    print(f"      Date range: {START_DATE} to {END_DATE}")
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading price data (this may take a minute)...")
    
    raw_prices = download_price_history(
        tickers=universe,
        start=START_DATE,
        end=END_DATE,
    )
    prices = get_adjusted_close(raw_prices)
    
    # Check which tickers have sufficient data
    valid_tickers = []
    for t in tradeable_tickers:
        if t in prices.columns:
            nan_pct = prices[t].isna().sum() / len(prices)
            if nan_pct < 0.1:  # Less than 10% missing
                valid_tickers.append(t)
            else:
                print(f"      Warning: {t} has {nan_pct*100:.1f}% missing data, excluding")
        else:
            print(f"      Warning: {t} not found in downloaded data, excluding")
    
    tradeable_tickers = valid_tickers
    prices = prices[[BENCHMARK] + tradeable_tickers].copy()
    
    # Forward fill small gaps
    prices = prices.ffill(limit=5)
    
    print(f"      Loaded {len(prices)} trading days")
    print(f"      Valid tickers: {len(tradeable_tickers)}")
    
    # -------------------------------------------------------------------------
    # Setup config
    # -------------------------------------------------------------------------
    config = PipelineConfig(
        research=ResearchConfig(
            universe=tuple(tradeable_tickers + [BENCHMARK]),
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
    if min_start_idx >= len(all_dates):
        print("ERROR: Not enough data for backtest")
        return
    
    min_start_date = all_dates[min_start_idx]
    rebal_dates = [d for d in rebal_dates if d >= min_start_date]
    
    print(f"      Rebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")
    
    # -------------------------------------------------------------------------
    # Fundamentals Coverage Filtering (if --use-fundamentals enabled)
    # -------------------------------------------------------------------------
    coverage_state = None
    
    if args.use_fundamentals and (
        args.fundamentals_universe_mode != "full" or args.min_fundamentals_coverage > 0
    ):
        print("\n[2.5/6] Filtering tickers by fundamentals coverage...")
        
        try:
            coverage_state = compute_coverage_and_filter(
                prices=prices,
                tradeable_tickers=tradeable_tickers,
                args=args,
                benchmark=BENCHMARK,
            )
            
            # Apply filtering based on mode
            if coverage_state.mode == "good_only":
                # Restrict universe to good-coverage tickers only
                original_count = len(tradeable_tickers)
                tradeable_tickers = coverage_state.good_tickers
                
                # Update prices to only include filtered tickers + benchmark
                prices = prices[[BENCHMARK] + tradeable_tickers].copy()
                
                # Update config with new universe
                config = replace(
                    config,
                    research=replace(
                        config.research,
                        universe=tuple(tradeable_tickers + [BENCHMARK]),
                    ),
                )
                
                print(f"      Filtered to {len(tradeable_tickers)}/{original_count} tickers with good coverage")
                
        except ValueError as e:
            # Zero tickers pass filter - abort
            print(f"\n  [FATAL] {e}")
            print("  Aborting backtest due to insufficient fundamentals coverage.")
            return
    
    # Initialize coverage state for reporting if not set
    if coverage_state is None:
        coverage_state = FundamentalsCoverageState(
            mode="full",
            min_coverage=0.0,
            good_tickers=tradeable_tickers.copy(),
            n_fundamentals=len(tradeable_tickers) if args.use_fundamentals else 0,
            n_tech_only=0 if args.use_fundamentals else len(tradeable_tickers),
        )

    # Get tech-only tickers for hybrid mode feature masking
    tech_only_tickers: tuple[str, ...] = ()
    if coverage_state.mode == "hybrid" and coverage_state.tech_only_tickers:
        tech_only_tickers = tuple(coverage_state.tech_only_tickers)
        print(f"      Hybrid mode: {len(tech_only_tickers)} tickers will use tech-only features")

    # -------------------------------------------------------------------------
    # Run strategies
    # -------------------------------------------------------------------------
    print("\n[3/6] Running strategies...")
    
    results: list[StrategyResult] = []
    ml_tech_result = None
    ml_fund_result = None
    ml_quality_result = None
    last_ml_model = None
    
    # Store BacktestResults for regime-switch ensemble
    ml_tech_backtest_result = None
    ml_quality_backtest_result = None
    
    # -------------------------------------------------------------------------
    # ML Strategy - with potential A/B comparison
    # -------------------------------------------------------------------------
    
    # Determine if we should run comparison mode
    # Comparison mode: finagg or CSV provided with --use-fundamentals
    run_comparison = args.use_fundamentals and (
        args.fundamentals_csv is not None or args.fundamentals_source == "finagg"
    )
    
    # Determine which factor families to enable based on source
    if args.fundamentals_source == "finagg":
        # finagg provides comprehensive SEC data
        available_factor_families = {
            'value': False,       # Still need price data for P/E, P/B, P/S
            'quality': True,      # ROE, ROA available from SEC filings
            'growth': True,       # Can compute YoY from historical data
            'profitability': True,  # Gross margin, operating margin
            'size': False,        # Need price * shares for market cap
        }
    else:
        # Alpha Vantage / CSV limitations
        available_factor_families = {
            'value': False,       # No historical P/E, P/B, P/S
            'quality': False,     # ROE/ROA are NaN (balance sheet alignment issue)
            'growth': False,      # No historical rev/eps growth
            'profitability': True,  # Gross margin, operating margin AVAILABLE
            'size': False,        # No historical market cap
        }
    
    # -------------------------------------------------------------------------
    # Regularization Sweep Mode (early exit if --reg-sweep-core-simple)
    # -------------------------------------------------------------------------
    if getattr(args, 'reg_sweep_core_simple', False):
        # Validate required flags
        if not args.use_fundamentals:
            print("\n  [Error] --reg-sweep-core-simple requires --use-fundamentals.")
            return
        if not args.use_core_factors:
            print("\n  [Error] --reg-sweep-core-simple requires --use-core-factors.")
            return
        if not args.simple_model:
            print("\n  [Error] --reg-sweep-core-simple requires --simple-model.")
            return
        
        # Run the regularization sweep
        core_factor_sweep_config = CoreFactorConfig(
            min_subperiod_support=2,
            min_importance_threshold=0.01,
            min_mean_importance=0.03,
        )
        
        sweep_results = run_core_simple_reg_sweep(
            prices=prices,
            config=config,
            core_factor_config=core_factor_sweep_config,
            available_factor_families=available_factor_families,
            sweep_config=DEFAULT_REG_SWEEP_CONFIG,
            args=args,
        )
        
        # Print results table
        if sweep_results:
            print_sweep_results_table(sweep_results)
        else:
            print("\n  [Error] No sweep results generated.")
        
        print("\n" + "=" * 80)
        print("  Regularization sweep complete.")
        print("=" * 80 + "\n")
        return  # Early exit - skip normal backtest
    
    # -------------------------------------------------------------------------
    # OOS Robustness Suite Mode (--core-oos-suite)
    # -------------------------------------------------------------------------
    if getattr(args, 'core_oos_suite', False):
        # Validate required flags
        if not args.use_fundamentals:
            print("\n  [Error] --core-oos-suite requires --use-fundamentals.")
            return
        if not args.use_core_factors:
            print("\n  [Error] --core-oos-suite requires --use-core-factors.")
            return
        if not args.simple_model:
            print("\n  [Error] --core-oos-suite requires --simple-model.")
            return
        
        # Build model config from best regularization params
        best_model_config = build_model_config_from_params(**BEST_CORE_SIMPLE_PARAMS)
        
        # Check if hybrid mode is enabled
        is_hybrid_mode = getattr(args, 'fundamentals_universe_mode', 'full') == 'hybrid'
        min_coverage = getattr(args, 'min_fundamentals_coverage', 0.0)
        
        if is_hybrid_mode:
            # Run Hybrid OOS suite: Tech-only vs Hybrid comparison
            print("\n" + "=" * 80)
            print("  Running Hybrid OOS Robustness Suite")
            print("  Comparing Tech-only vs Hybrid (Tech + Fundamentals) across splits...")
            print(f"  Min fundamentals coverage: {min_coverage:.0%}")
            print("=" * 80 + "\n")
            
            hybrid_results = run_hybrid_oos_robustness_suite(
                prices=prices,
                config=config,
                available_factor_families=available_factor_families,
                suite_cfg=CoreOOSRobustnessConfig(enable=True, splits=DEFAULT_CORE_OOS_SPLITS),
                min_coverage=min_coverage,
                model_config=best_model_config,
                args=args,
            )
            
            # Print hybrid comparison summary
            print_hybrid_oos_robustness_summary(hybrid_results)
            
            print("\n" + "=" * 80)
            print("  Hybrid OOS robustness suite complete.")
            print("=" * 80 + "\n")
        else:
            # Run standard OOS suite (existing behavior)
            print("\n" + "=" * 80)
            print("  Running Core Factors [Simple] OOS Robustness Suite")
            print("  Testing model across multiple train/test splits...")
            print("=" * 80 + "\n")
            
            suite_results = run_core_oos_robustness_suite(
                prices=prices,
                config=config,
                available_factor_families=available_factor_families,
                suite_cfg=CoreOOSRobustnessConfig(enable=True, splits=DEFAULT_CORE_OOS_SPLITS),
                model_config=best_model_config,
                args=args,
            )
            
            # Print summary table
            print_core_oos_robustness_summary(suite_results)
            
            print("\n" + "=" * 80)
            print("  OOS robustness suite complete.")
            print("=" * 80 + "\n")
        
        return  # Early exit - skip normal backtest
    
    if run_comparison:
        # Run BOTH Tech-only and Tech+Fundamentals for comparison
        
        # (A) Tech-only - use a config without fundamentals
        config_tech_only = PipelineConfig(
            research=config.research,
            features=FeatureConfig(
                momentum_windows=(21, 63, 126, 252),
                volatility_windows=(21, 63),
                include_fundamentals=False,
                include_sentiment=False,
                fundamentals_csv_path=None,
            ),
            model=config.model,
        )
        ml_tech_result, _, ml_tech_backtest_result = run_ml_strategy(
            prices, config_tech_only, tradeable_tickers,
            pipeline_config=None,
            label="ML Tech Only",
            regime_mode=args.regime_mode,
            cost_config=COST_MODEL,
        )
        results.append(ml_tech_result)
        
        # (B) Tech + Fundamentals
        config_with_fund = PipelineConfig(
            research=config.research,
            features=FeatureConfig(
                momentum_windows=(21, 63, 126, 252),
                volatility_windows=(21, 63),
                include_fundamentals=True,
                include_sentiment=False,
                fundamentals_csv_path=args.fundamentals_csv,
                fundamental_factor_families=available_factor_families,
                fundamentals_source=args.fundamentals_source,
                finagg_mode=args.finagg_mode,
                tech_only_tickers=tech_only_tickers,
            ),
            model=config.model,
        )
        ml_fund_result, last_ml_model, _ = run_ml_strategy(
            prices, config_with_fund, tradeable_tickers,
            pipeline_config=None,
            label="ML Tech + Fundamentals",
            regime_mode=args.regime_mode,
            cost_config=COST_MODEL,
        )
        results.append(ml_fund_result)
        
        # (C) Quality-only (Fundamentals without technicals)
        config_quality_only = PipelineConfig(
            research=config.research,
            features=FeatureConfig(
                momentum_windows=(21, 63, 126, 252),
                volatility_windows=(21, 63),
                include_technical=False,  # No technicals - Quality only
                include_fundamentals=True,
                include_sentiment=False,
                fundamentals_csv_path=args.fundamentals_csv,
                fundamental_factor_families=available_factor_families,
                fundamentals_source=args.fundamentals_source,
                finagg_mode=args.finagg_mode,
                tech_only_tickers=tech_only_tickers,
            ),
            model=config.model,
        )
        ml_quality_result, _, ml_quality_backtest_result = run_ml_strategy(
            prices, config_quality_only, tradeable_tickers,
            pipeline_config=None,
            label="ML Quality Only",
            regime_mode=args.regime_mode,
            cost_config=COST_MODEL,
        )
        results.append(ml_quality_result)
        
        # (D) Regime-Switch Strategy (Momentum vs Quality)
        if ml_tech_backtest_result is not None and ml_quality_backtest_result is not None:
            print("  Running ML Regime Switch (Mom vs Quality)...")
            try:
                regime_switch_backtest = run_regime_switch_strategy(
                    prices=prices,
                    spy_prices=prices[BENCHMARK],
                    momentum_result=ml_tech_backtest_result,
                    quality_result=ml_quality_backtest_result,
                    top_k=TOP_K,
                    benchmark=BENCHMARK,
                    cost_config=COST_MODEL,
                )
                
                # Extract IC statistics
                ic_series = None
                if regime_switch_backtest.ic_history:
                    ic_dates, ic_values = zip(*regime_switch_backtest.ic_history)
                    ic_series = pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates))
                    avg_ic = ic_series.mean()
                    ic_std = ic_series.std()
                    ic_median = ic_series.median()
                else:
                    avg_ic = ic_std = ic_median = None
                
                # Create StrategyResult
                net_ann_ret = regime_switch_backtest.net_metrics.get("net_ann_ret") if regime_switch_backtest.net_metrics else None
                net_sharpe = regime_switch_backtest.net_metrics.get("net_sharpe") if regime_switch_backtest.net_metrics else None
                
                regime_switch_result = StrategyResult(
                    name="ML Regime Switch [RegSwitch]",
                    portfolio_returns=regime_switch_backtest.portfolio_returns,
                    benchmark_returns=regime_switch_backtest.benchmark_returns,
                    cumulative_return=regime_switch_backtest.metrics.get("total_return", 0),
                    annualized_return=regime_switch_backtest.metrics.get("annualized_return", 0),
                    annualized_volatility=regime_switch_backtest.metrics.get("annualized_volatility", 0),
                    sharpe_ratio=regime_switch_backtest.metrics.get("sharpe_ratio", 0),
                    hit_rate=regime_switch_backtest.metrics.get("hit_rate", 0),
                    max_drawdown=regime_switch_backtest.metrics.get("max_drawdown", 0),
                    information_ratio=regime_switch_backtest.metrics.get("information_ratio", 0),
                    ic=avg_ic,
                    ic_std=ic_std,
                    ic_median=ic_median,
                    ic_series=ic_series,
                    avg_turnover=regime_switch_backtest.avg_turnover,
                    p95_turnover=regime_switch_backtest.p95_turnover,
                    net_annualized_return=net_ann_ret,
                    net_sharpe_ratio=net_sharpe,
                    weights_history=regime_switch_backtest.weights_history,
                )
                results.append(regime_switch_result)
                
                # Print regime summary
                mom_count = regime_switch_backtest.metrics.get("regime_momentum_count", 0)
                qual_count = regime_switch_backtest.metrics.get("regime_quality_count", 0)
                print(f"    Regime distribution: MOMENTUM={mom_count}, QUALITY={qual_count}")
                
            except Exception as e:
                logger.warning(f"Failed to run regime-switch strategy: {e}")
                import traceback
                traceback.print_exc()
        
        # (E) IC-Weighted Ensemble Strategy (Momentum + Quality)
        if ml_tech_backtest_result is not None and ml_quality_backtest_result is not None:
            print("  Running ML IC-Weighted Ensemble (Mom+Quality)...")
            try:
                ic_weight_config = ICWeightConfig(
                    window_mom=6,   # 6 months rolling IC for momentum
                    window_qual=12, # 12 months rolling IC for quality
                    min_weight=0.0,
                    fallback_weight_mom=0.5,
                )
                
                ic_blend_backtest = run_ic_weighted_ensemble(
                    prices=prices,
                    momentum_result=ml_tech_backtest_result,
                    quality_result=ml_quality_backtest_result,
                    top_k=TOP_K,
                    benchmark=BENCHMARK,
                    ic_weight_config=ic_weight_config,
                    cost_config=COST_MODEL,
                )
                
                # Extract IC statistics
                ic_series = None
                if ic_blend_backtest.ic_history:
                    ic_dates, ic_values = zip(*ic_blend_backtest.ic_history)
                    ic_series = pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates))
                    avg_ic = ic_series.mean()
                    ic_std = ic_series.std()
                    ic_median = ic_series.median()
                else:
                    avg_ic = ic_std = ic_median = None
                
                # Create StrategyResult
                net_ann_ret = ic_blend_backtest.net_metrics.get("net_ann_ret") if ic_blend_backtest.net_metrics else None
                net_sharpe = ic_blend_backtest.net_metrics.get("net_sharpe") if ic_blend_backtest.net_metrics else None
                
                ic_blend_result = StrategyResult(
                    name="ML IC-Blend (Mom+Qual) [ICBlend]",
                    portfolio_returns=ic_blend_backtest.portfolio_returns,
                    benchmark_returns=ic_blend_backtest.benchmark_returns,
                    cumulative_return=ic_blend_backtest.metrics.get("total_return", 0),
                    annualized_return=ic_blend_backtest.metrics.get("annualized_return", 0),
                    annualized_volatility=ic_blend_backtest.metrics.get("annualized_volatility", 0),
                    sharpe_ratio=ic_blend_backtest.metrics.get("sharpe_ratio", 0),
                    hit_rate=ic_blend_backtest.metrics.get("hit_rate", 0),
                    max_drawdown=ic_blend_backtest.metrics.get("max_drawdown", 0),
                    information_ratio=ic_blend_backtest.metrics.get("information_ratio", 0),
                    ic=avg_ic,
                    ic_std=ic_std,
                    ic_median=ic_median,
                    ic_series=ic_series,
                    avg_turnover=ic_blend_backtest.avg_turnover,
                    p95_turnover=ic_blend_backtest.p95_turnover,
                    net_annualized_return=net_ann_ret,
                    net_sharpe_ratio=net_sharpe,
                    weights_history=ic_blend_backtest.weights_history,
                )
                results.append(ic_blend_result)
                
                # Print weight summary
                avg_w_mom = ic_blend_backtest.metrics.get("avg_weight_momentum", 0.5)
                avg_w_qual = ic_blend_backtest.metrics.get("avg_weight_quality", 0.5)
                print(f"    Avg IC weights: Momentum={avg_w_mom:.1%}, Quality={avg_w_qual:.1%}")
                
            except Exception as e:
                logger.warning(f"Failed to run IC-weighted ensemble strategy: {e}")
                import traceback
                traceback.print_exc()
        
    elif args.use_fundamentals:
        # Fundamentals enabled but no CSV and not finagg - run with yfinance fallback
        config_with_fund = PipelineConfig(
            research=config.research,
            features=FeatureConfig(
                momentum_windows=(21, 63, 126, 252),
                volatility_windows=(21, 63),
                include_fundamentals=True,
                include_sentiment=False,
                fundamentals_csv_path=args.fundamentals_csv,
                fundamental_factor_families=available_factor_families,
                fundamentals_source=args.fundamentals_source,
                finagg_mode=args.finagg_mode,
                tech_only_tickers=tech_only_tickers,
            ),
            model=config.model,
        )
        ml_result, last_ml_model, _ = run_ml_strategy(
            prices, config_with_fund, tradeable_tickers,
            pipeline_config=None,
            label="ML (GBDT) w/ Fundamentals",
            regime_mode=args.regime_mode,
            cost_config=COST_MODEL,
        )
        results.append(ml_result)
        
    else:
        # Default: Tech-only
        ml_result, last_ml_model, _ = run_ml_strategy(
            prices, config, tradeable_tickers,
            pipeline_config=None,
            label="ML (GBDT)",
            regime_mode=args.regime_mode,
            cost_config=COST_MODEL,
        )
        results.append(ml_result)
    
    # 2. Equal-Weight Universe
    ew_result = run_equal_weight_strategy(prices, rebal_dates, BENCHMARK, tradeable_tickers)
    if ew_result:
        results.append(ew_result)
    
    # 3. 12M Momentum Top-K
    mom_result = run_momentum_strategy(
        prices, rebal_dates, BENCHMARK, tradeable_tickers,
        lookback_days=252, top_k=TOP_K
    )
    if mom_result:
        results.append(mom_result)
    
    # 4. Random Top-K
    rand_result = run_random_strategy(
        prices, rebal_dates, BENCHMARK, tradeable_tickers,
        top_k=TOP_K, n_paths=N_RANDOM_PATHS, seed=RANDOM_SEED
    )
    if rand_result:
        results.append(rand_result)
    
    # -------------------------------------------------------------------------
    # Compute benchmark metrics
    # -------------------------------------------------------------------------
    print("\n[4/6] Computing benchmark metrics...")
    
    bench_period_returns = []
    for i in range(len(rebal_dates) - 1):
        start_price = prices.loc[rebal_dates[i], BENCHMARK]
        end_price = prices.loc[rebal_dates[i + 1], BENCHMARK]
        bench_period_returns.append((end_price / start_price) - 1)
    
    bench_series = pd.Series(bench_period_returns, index=pd.DatetimeIndex(rebal_dates[1:]))
    benchmark_ann_return = compute_annualized_return(bench_series, periods_per_year=12)
    
    print(f"      SPY Annualized Return: {benchmark_ann_return * 100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Print detailed ML metrics
    # -------------------------------------------------------------------------
    print("\n[5/6] ML Strategy Details...")
    
    # Get the main ML result for detailed printing
    if run_comparison:
        # Use the Tech+Fundamentals result for details
        ml_result = ml_fund_result
    else:
        ml_result = results[0] if results else None
    
    if ml_result:
        print(f"      Strategy:           {ml_result.name}")
        print(f"      Cumulative Return:  {ml_result.cumulative_return * 100:.1f}%")
        print(f"      Annualized Return:  {ml_result.annualized_return * 100:.2f}%")
        print(f"      Annualized Vol:     {ml_result.annualized_volatility * 100:.1f}%")
        print(f"      Sharpe Ratio:       {ml_result.sharpe_ratio:.3f}")
        print(f"      Hit Rate vs SPY:    {ml_result.hit_rate * 100:.1f}%")
        print(f"      Information Ratio:  {ml_result.information_ratio:.3f}")
        print(f"      Max Drawdown:       {ml_result.max_drawdown * 100:.1f}%")
        if ml_result.ic is not None:
            print(f"      Average IC:         {ml_result.ic:.4f}")
            print(f"      IC Std Dev:         {ml_result.ic_std:.4f}")
            print(f"      Median IC:          {ml_result.ic_median:.4f}")
    
    # -------------------------------------------------------------------------
    # Print comparison table
    # -------------------------------------------------------------------------
    print("\n[6/6] Results Comparison...")
    print_comparison_table(results, benchmark_ann_return)
    
    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    print_interpretation(results)
    
    # -------------------------------------------------------------------------
    # ML Comparison (if running both Tech and Tech+Fundamentals)
    # -------------------------------------------------------------------------
    if run_comparison and ml_tech_result and ml_fund_result:
        print_ml_comparison(ml_tech_result, ml_fund_result)
    
    # -------------------------------------------------------------------------
    # Factor Family Importance (if fundamentals enabled)
    # -------------------------------------------------------------------------
    if args.use_fundamentals and last_ml_model is not None:
        print_factor_family_importance(last_ml_model)
    
    # -------------------------------------------------------------------------
    # Core Factor Extraction (if --use-core-factors flag)
    # -------------------------------------------------------------------------
    if args.use_core_factors:
        if not args.use_fundamentals:
            print("\n  [Warning] --use-core-factors requires --use-fundamentals. Skipping.")
        else:
            print("\n" + "=" * 80)
            print("  CORE FACTOR ANALYSIS")
            print("=" * 80)
            
            # Use the config that includes fundamentals
            core_config_base = PipelineConfig(
                research=config.research,
                features=FeatureConfig(
                    momentum_windows=(21, 63, 126, 252),
                    volatility_windows=(21, 63),
                    include_fundamentals=True,
                    include_sentiment=False,
                    fundamentals_csv_path=args.fundamentals_csv,
                    fundamental_factor_families=available_factor_families,
                    fundamentals_source=args.fundamentals_source,
                    finagg_mode=args.finagg_mode,
                    tech_only_tickers=tech_only_tickers,
                ),
                model=config.model,
            )
            
            # Step 1: Extract feature importances by subperiod
            importance_by_subperiod = compute_feature_importance_by_subperiod(
                prices=prices,
                config=core_config_base,
                subperiods=SUBPERIODS,
                benchmark=BENCHMARK,
                horizon_days=HORIZON_DAYS,
            )
            
            if len(importance_by_subperiod) >= 2:
                # Step 2: Select core features
                core_factor_config = CoreFactorConfig(
                    min_subperiod_support=2,
                    min_importance_threshold=0.01,
                    min_mean_importance=0.03,
                )
                core_features, importance_df = select_core_features(
                    importance_by_subperiod,
                    core_config=core_factor_config,
                )
                
                # Step 3: Print summary
                print_core_factor_summary(core_features, importance_df, SUBPERIODS)
                
                if core_features:
                    # Step 4: Run strategy with core features only
                    print("\n  Running ML strategy with CORE FACTORS ONLY...")
                    
                    # Choose model config
                    model_config_to_use = SIMPLE_MODEL_CONFIG if args.simple_model else config.model
                    
                    if args.simple_model:
                        print(f"    Using SIMPLE_MODEL_CONFIG: max_depth={SIMPLE_MODEL_CONFIG.max_depth}, "
                              f"n_estimators={SIMPLE_MODEL_CONFIG.n_estimators}")
                    
                    # Create config with core features
                    core_features_config = PipelineConfig(
                        research=config.research,
                        features=FeatureConfig(
                            momentum_windows=(21, 63, 126, 252),
                            volatility_windows=(21, 63),
                            include_fundamentals=True,
                            include_sentiment=False,
                            fundamentals_csv_path=args.fundamentals_csv,
                            fundamental_factor_families=available_factor_families,
                            fundamentals_source=args.fundamentals_source,
                            finagg_mode=args.finagg_mode,
                            use_core_features_only=True,
                            core_features=tuple(core_features),
                            tech_only_tickers=tech_only_tickers,
                        ),
                        model=model_config_to_use,
                    )
                    
                    # Run the core factor strategy
                    label_suffix = " [Simple]" if args.simple_model else ""
                    core_factor_result, _, _ = run_ml_strategy(
                        prices, core_features_config, tradeable_tickers,
                        pipeline_config=None,
                        label=f"ML Core Factors{label_suffix}",
                        regime_mode=args.regime_mode,
                        cost_config=COST_MODEL,
                    )
                    results.append(core_factor_result)
                    
                    # Step 5: Print comparison
                    print("\n" + "=" * 70)
                    print("  CORE FACTOR STRATEGY COMPARISON")
                    print("=" * 70)
                    
                    # Find the Tech+Fund result for comparison
                    baseline_result = ml_fund_result
                    if baseline_result is not None:
                        print(f"\n  {'Metric':<25} | {'Tech+Fund':>12} | {'Core Factors':>12} | {'Delta':>10}")
                        print("-" * 70)
                        
                        # Gross metrics
                        baseline_sharpe = baseline_result.sharpe_ratio
                        core_sharpe = core_factor_result.sharpe_ratio
                        delta_sharpe = core_sharpe - baseline_sharpe
                        print(f"  {'Gross Sharpe Ratio':<25} | {baseline_sharpe:>12.3f} | {core_sharpe:>12.3f} | {delta_sharpe:>+10.3f}")
                        
                        baseline_ret = baseline_result.annualized_return * 100
                        core_ret = core_factor_result.annualized_return * 100
                        delta_ret = core_ret - baseline_ret
                        print(f"  {'Gross Ann. Return (%)':<25} | {baseline_ret:>12.2f} | {core_ret:>12.2f} | {delta_ret:>+10.2f}")
                        
                        # Net metrics
                        if baseline_result.net_sharpe_ratio is not None and core_factor_result.net_sharpe_ratio is not None:
                            baseline_net_sharpe = baseline_result.net_sharpe_ratio
                            core_net_sharpe = core_factor_result.net_sharpe_ratio
                            delta_net_sharpe = core_net_sharpe - baseline_net_sharpe
                            print(f"  {'Net Sharpe Ratio':<25} | {baseline_net_sharpe:>12.3f} | {core_net_sharpe:>12.3f} | {delta_net_sharpe:>+10.3f}")
                        
                        if baseline_result.net_annualized_return is not None and core_factor_result.net_annualized_return is not None:
                            baseline_net_ret = baseline_result.net_annualized_return * 100
                            core_net_ret = core_factor_result.net_annualized_return * 100
                            delta_net_ret = core_net_ret - baseline_net_ret
                            print(f"  {'Net Ann. Return (%)':<25} | {baseline_net_ret:>12.2f} | {core_net_ret:>12.2f} | {delta_net_ret:>+10.2f}")
                        
                        # Turnover
                        if baseline_result.avg_turnover is not None and core_factor_result.avg_turnover is not None:
                            baseline_turnover = baseline_result.avg_turnover * 100
                            core_turnover = core_factor_result.avg_turnover * 100
                            delta_turnover = core_turnover - baseline_turnover
                            print(f"  {'Avg Turnover (%)':<25} | {baseline_turnover:>12.1f} | {core_turnover:>12.1f} | {delta_turnover:>+10.1f}")
                        
                        # IC
                        if baseline_result.ic is not None and core_factor_result.ic is not None:
                            baseline_ic = baseline_result.ic
                            core_ic = core_factor_result.ic
                            delta_ic = core_ic - baseline_ic
                            print(f"  {'Average IC':<25} | {baseline_ic:>12.4f} | {core_ic:>12.4f} | {delta_ic:>+10.4f}")
                        
                        print("-" * 70)
                        
                        # Interpretation
                        print("\n  Interpretation:")
                        n_core = len(core_features)
                        n_total = len(importance_df)
                        print(f"    - Core factors: {n_core} features (from {n_total} total)")
                        
                        if delta_net_sharpe > 0.05:
                            print(f"    - Core factors IMPROVED net Sharpe by {delta_net_sharpe:.3f}")
                            print("    - Simpler model generalizes better out-of-sample")
                        elif delta_net_sharpe < -0.05:
                            print(f"    - Core factors REDUCED net Sharpe by {abs(delta_net_sharpe):.3f}")
                            print("    - Full feature set may capture more alpha despite complexity")
                        else:
                            print(f"    - Core factors had SIMILAR performance (delta={delta_net_sharpe:.3f})")
                            print("    - Parsimony achieved without sacrificing returns")
                        
                        if core_turnover < baseline_turnover:
                            print(f"    - Core factors REDUCED turnover by {abs(delta_turnover):.1f}%")
                        else:
                            print(f"    - Core factors had similar turnover")
            else:
                print("\n  [Warning] Need at least 2 subperiods for core factor analysis. "
                      f"Got {len(importance_by_subperiod)}.")
    
    # -------------------------------------------------------------------------
    # Subperiod Performance Analysis
    # -------------------------------------------------------------------------
    if args.enable_subperiods:
        try:
            # Build strategy returns dictionary
            strategy_returns: dict[str, pd.Series] = {}
            ic_series_by_strategy: dict[str, pd.Series | None] = {}
            
            for r in results:
                strategy_returns[r.name] = r.portfolio_returns
                ic_series_by_strategy[r.name] = r.ic_series  # Will be None for non-ML
            
            # Compute subperiod metrics (returns tuple: metrics dict, has_relative flag)
            subperiod_metrics, has_relative = compute_subperiod_metrics(
                strategy_returns=strategy_returns,
                benchmark_returns=bench_series,
                ic_series_by_strategy=ic_series_by_strategy,
                subperiods=SUBPERIODS,
                periods_per_year=12,  # Monthly rebalancing
            )
            
            # Print the table (with or without relative metrics columns)
            print_subperiod_table(subperiod_metrics, SUBPERIODS, has_relative_metrics=has_relative)
            
            # Print consistency summary (only if relative metrics are available)
            if has_relative:
                consistency = compute_subperiod_consistency(subperiod_metrics)
                print_subperiod_consistency_summary(consistency)
            
            # Print interpretation (Fundamentals vs Tech-Only comparison)
            print_subperiod_interpretation(subperiod_metrics, SUBPERIODS)
            
        except Exception as e:
            logger.warning(f"Failed to compute subperiod analysis: {e}")
    
    # -------------------------------------------------------------------------
    # Regime-Switch Strategy Interpretation
    # -------------------------------------------------------------------------
    print_regime_switch_interpretation(results)
    
    # -------------------------------------------------------------------------
    # IC-Weighted Ensemble Interpretation
    # -------------------------------------------------------------------------
    print_ic_blend_interpretation(results)
    
    # -------------------------------------------------------------------------
    # Out-of-Sample Robustness Experiment (if --oos-mode specified)
    # -------------------------------------------------------------------------
    if args.oos_mode == "train_2013_2018_test_2019_2023":
        if not args.use_fundamentals:
            print("\n  [Warning] --oos-mode requires --use-fundamentals. Skipping OOS experiment.")
        elif not args.use_core_factors:
            print("\n  [Warning] --oos-mode requires --use-core-factors. Skipping OOS experiment.")
        else:
            # Run the OOS experiment for Core Factors [Simple]
            core_factor_oos_config = CoreFactorConfig(
                min_subperiod_support=2,
                min_importance_threshold=0.01,
                min_mean_importance=0.03,
            )
            
            run_core_factor_oos_experiment(
                prices=prices,
                config=config,
                core_factor_config=core_factor_oos_config,
                available_factor_families=available_factor_families,
                args=args,
            )
    
    # -------------------------------------------------------------------------
    # Fundamentals Coverage Summary (if coverage filtering was applied)
    # -------------------------------------------------------------------------
    print_coverage_report(coverage_state)
    
    print("\n" + "=" * 80)
    print("  Large-cap backtest complete.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
