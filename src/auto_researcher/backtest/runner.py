"""
Backtest runner.

This module provides the main backtest loop for walk-forward validation
of the stock ranking and portfolio construction strategy.
"""

import logging
from typing import Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

from auto_researcher.models.gbdt_model import GBDTModel
from auto_researcher.models.regimes import (
    RegimeMode,
    assign_regime,
    add_regime_feature,
    split_data_by_regime,
    select_model_for_regime,
    get_regime_aware_features,
    REGIME_LABELS,
)
from auto_researcher.backtest.regime_signal import (
    compute_regime_series,
    get_regime_for_date,
    compute_regime_summary,
    MarketRegime,
    RegimeConfig,
)
from auto_researcher.features.feature_pipeline import (
    build_feature_matrix,
    prepare_training_data,
    get_features_for_date,
)
from auto_researcher.backtest.portfolio import (
    build_equal_weight_portfolio,
    compute_portfolio_return,
    get_turnover,
)
from auto_researcher.backtest.metrics import (
    compute_all_metrics,
    compute_ic,
    compute_ic_weights,
    ICWeightConfig,
    CostModelConfig,
    compute_turnover_series,
    compute_net_returns,
    compute_net_metrics,
)
from auto_researcher.config import (
    PipelineConfig,
    get_default_config,
    FeatureConfig,
    ModelConfig,
    EnhancedModelConfig,
    EnhancedPortfolioConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """
    Container for backtest results.

    Attributes:
        portfolio_returns: Series of portfolio returns indexed by date.
        benchmark_returns: Series of benchmark returns.
        weights_history: List of (date, weights_dict) tuples.
        ic_history: List of (date, IC) tuples.
        metrics: Dictionary of performance metrics.
        config: Configuration used for the backtest.
        scores_history: List of (date, scores_series) tuples with per-ticker scores.
            Used for regime-switch ensemble strategies.
        turnover_series: Series of turnover values indexed by rebalance date.
        avg_turnover: Average turnover across all rebalances.
        p95_turnover: 95th percentile of turnover values.
        net_returns: Series of net returns after transaction costs.
        net_metrics: Dictionary of net-of-cost performance metrics.
    """

    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]]
    ic_history: list[tuple[pd.Timestamp, float]]
    metrics: dict[str, float]
    config: PipelineConfig
    scores_history: list[tuple[pd.Timestamp, pd.Series]] | None = None
    # Transaction cost fields (optional for backward compatibility)
    turnover_series: pd.Series | None = None
    avg_turnover: float | None = None
    p95_turnover: float | None = None
    net_returns: pd.Series | None = None
    net_metrics: dict[str, float] | None = None


def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "M",
) -> list[pd.Timestamp]:
    """
    Get rebalance dates based on frequency.

    Args:
        dates: Full date range.
        frequency: Rebalance frequency ('M' for monthly, 'W' for weekly).

    Returns:
        List of rebalance dates.
    """
    # Resample to get period ends
    dummy = pd.Series(1, index=dates)
    
    if frequency == "M":
        rebal = dummy.resample("ME").last().index
    elif frequency == "W":
        rebal = dummy.resample("W").last().index
    elif frequency == "D":
        rebal = dates
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    # Ensure dates are in the original index
    rebal_dates = [d for d in rebal if d in dates]
    
    return rebal_dates


def run_backtest(
    prices: pd.DataFrame,
    horizon_days: int,
    top_k: int,
    model: GBDTModel | None = None,
    benchmark: str = "SPY",
    rebalance_frequency: str = "M",
    min_train_periods: int = 252,
    config: PipelineConfig | None = None,
    regime_mode: str | RegimeMode = "none",
    cost_config: CostModelConfig | None = None,
) -> BacktestResult:
    """
    Run a walk-forward backtest.

    At each rebalance date:
    1. Train model on all available history up to that point
    2. Predict scores for all stocks
    3. Select top-k stocks with equal weight
    4. Hold until next rebalance date

    Args:
        prices: Adjusted close prices with tickers as columns.
        horizon_days: Forward return horizon in trading days.
        top_k: Number of stocks to select.
        model: GBDT model instance. If None, creates default.
        benchmark: Benchmark ticker for comparison.
        rebalance_frequency: How often to rebalance ('M', 'W', 'D').
        min_train_periods: Minimum training periods before first trade.
        config: Pipeline configuration.
        regime_mode: Regime-aware ML mode:
            - "none": Current behavior (default)
            - "feature": Single model with regime_id as feature
            - "split": Separate model per regime
        cost_config: Transaction cost configuration. If None, no costs applied.

    Returns:
        BacktestResult with portfolio returns, metrics, etc.

    Raises:
        ValueError: If benchmark not in prices or insufficient data.
    """
    # Normalize regime_mode to enum
    if isinstance(regime_mode, str):
        regime_mode = RegimeMode(regime_mode)
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} not found in prices")
    
    if config is None:
        config = get_default_config()
    
    if model is None:
        model = GBDTModel(config.model)
    
    # Get rebalance dates
    all_dates = prices.index
    rebal_dates = get_rebalance_dates(all_dates, rebalance_frequency)
    
    # Filter to dates with enough history
    min_start_idx = min_train_periods + horizon_days
    if min_start_idx >= len(all_dates):
        raise ValueError("Insufficient data for backtest")
    
    min_start_date = all_dates[min_start_idx]
    rebal_dates = [d for d in rebal_dates if d >= min_start_date]
    
    if len(rebal_dates) < 2:
        raise ValueError("Not enough rebalance dates after warmup")
    
    logger.info(f"Running backtest with {len(rebal_dates)} rebalance dates")
    
    # Build feature matrix once (will slice as needed)
    feature_matrix = build_feature_matrix(prices, config.features)
    
    # Get tickers excluding benchmark
    tradeable_tickers = [t for t in prices.columns if t != benchmark]
    
    # Storage
    portfolio_returns = []
    weights_history = []
    ic_history = []
    scores_history = []  # Store per-date scores for regime-switch ensemble
    prev_weights: dict[str, float] = {}
    
    # Storage for regime-split models (only used when regime_mode == SPLIT)
    regime_models: dict[str, GBDTModel] = {}
    
    # Track regime transitions (to log once per transition, not every date)
    prev_regime: str | None = None
    regimes_using_fallback: set[str] = set()  # Track which regimes needed fallback
    
    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]
        
        logger.debug(f"Rebalancing on {rebal_date}")
        
        # Get training data up to rebal_date (inclusive)
        # LOOKAHEAD SAFETY CHECK:
        # - train_prices ends at rebal_date (the current decision point)
        # - prepare_training_data computes forward returns using shift(-horizon)
        # - For the last `horizon_days` rows, forward returns will be NaN
        #   because prices.shift(-horizon) extends beyond our cutoff
        # - These NaN rows are dropped, so we only train on labels that are
        #   fully realized BEFORE the rebal_date
        # 
        # OVERLAPPING HORIZONS NOTE:
        # - With monthly rebalancing (~21 days) and 63-day horizon, consecutive
        #   training labels overlap (share ~42 days of returns)
        # - This is standard practice but may inflate apparent Sharpe/IC
        # - For production, consider purging/embargo or non-overlapping periods
        train_end_idx = all_dates.get_loc(rebal_date)
        train_prices = prices.iloc[:train_end_idx + 1]
        
        # Prepare training data
        try:
            X_train, y_train = prepare_training_data(
                train_prices,
                horizon_days=horizon_days,
                benchmark=benchmark,
                config=config.features,
            )
        except Exception as e:
            logger.warning(f"Failed to prepare training data for {rebal_date}: {e}")
            continue
        
        if len(X_train) < 50:
            logger.warning(f"Insufficient training data for {rebal_date}")
            continue
        
        # =====================================================================
        # REGIME-AWARE TRAINING
        # =====================================================================
        if regime_mode == RegimeMode.NONE:
            # Current behavior: train single model on all data
            model.fit(X_train, y_train)
            active_model = model
            
        elif regime_mode == RegimeMode.FEATURE:
            # Add regime_id feature and train single model
            X_train_regime = add_regime_feature(X_train)
            model.fit(X_train_regime, y_train)
            active_model = model
            
        elif regime_mode == RegimeMode.SPLIT:
            # First, train the global fallback model on all data
            # This ensures we always have a model even for unseen regimes
            model.fit(X_train, y_train)
            
            # Then train separate models per regime
            # Note: We retrain all regime models at each rebal date
            # (could be optimized to only train when new data available)
            regime_data = split_data_by_regime(X_train, y_train)
            
            for regime_label, (X_regime, y_regime) in regime_data.items():
                if len(X_regime) < 50:
                    logger.debug(
                        f"Insufficient data for regime '{regime_label}' on {rebal_date}: "
                        f"{len(X_regime)} samples (using fallback)"
                    )
                    continue
                
                # Create fresh model for this regime
                regime_model = GBDTModel(config.model)
                regime_model.fit(X_regime, y_regime)
                regime_models[regime_label] = regime_model
            
            # Detect current regime for tracking transitions
            current_regime = assign_regime(rebal_date)
            
            # Log regime transition (once per regime)
            if current_regime != prev_regime:
                if current_regime not in regime_models:
                    if current_regime not in regimes_using_fallback:
                        logger.info(
                            f"Entering regime '{current_regime}' at {rebal_date.date()} - "
                            f"no trained model yet, using global fallback until data accumulates"
                        )
                        regimes_using_fallback.add(current_regime)
                else:
                    logger.info(
                        f"Entering regime '{current_regime}' at {rebal_date.date()} - "
                        f"using dedicated regime model"
                    )
                prev_regime = current_regime
            
            # Select model for current regime (with global model as fallback)
            active_model = select_model_for_regime(
                regime_models, rebal_date, fallback_model=model
            )
            
            if active_model is None:
                # Should never happen since we have fallback, but just in case
                logger.warning(f"No model available for {rebal_date}, using global model")
                active_model = model
        else:
            # Fallback to default
            model.fit(X_train, y_train)
            active_model = model
        
        # Get features for current date
        try:
            current_features = get_features_for_date(feature_matrix, rebal_date)
        except KeyError:
            logger.warning(f"No features for {rebal_date}")
            continue
        
        # Filter to tradeable tickers
        current_features = current_features.loc[
            current_features.index.isin(tradeable_tickers)
        ]
        
        # Identify technical vs fundamental columns
        tech_cols = [c for c in current_features.columns if c.startswith('tech_')]
        fund_cols = [c for c in current_features.columns if not c.startswith('tech_')]
        
        # Fill NaN fundamentals with 0 (neutral value for missing data)
        if fund_cols:
            current_features[fund_cols] = current_features[fund_cols].fillna(0.0)
        
        # Only drop rows where technical features are NaN
        if tech_cols:
            current_features = current_features.dropna(subset=tech_cols)
        
        if len(current_features) == 0:
            logger.warning(f"No valid features for {rebal_date}")
            continue
        
        # =====================================================================
        # REGIME-AWARE INFERENCE
        # =====================================================================
        # Add regime feature if using FEATURE mode
        if regime_mode == RegimeMode.FEATURE:
            current_features = get_regime_aware_features(
                current_features, rebal_date, regime_mode
            )
        
        # Rank stocks using the active model (selected during training phase)
        scores = active_model.rank_stocks(current_features)
        
        # Store scores for regime-switch ensemble strategies
        scores_history.append((rebal_date, scores.copy()))
        
        # Build portfolio
        weights = build_equal_weight_portfolio(scores, top_k)
        weights_history.append((rebal_date, weights))
        
        # Compute holding period returns
        if next_rebal in prices.index and rebal_date in prices.index:
            start_prices = prices.loc[rebal_date, tradeable_tickers]
            end_prices = prices.loc[next_rebal, tradeable_tickers]
            period_returns = (end_prices / start_prices) - 1
            
            port_return = compute_portfolio_return(weights, period_returns)
            portfolio_returns.append((next_rebal, port_return))
            
            # Compute IC
            realized = period_returns.loc[scores.index]
            ic = compute_ic(scores, realized)
            ic_history.append((next_rebal, ic))
        
        prev_weights = weights
    
    # Convert to Series
    if portfolio_returns:
        dates, returns = zip(*portfolio_returns)
        port_ret_series = pd.Series(returns, index=pd.DatetimeIndex(dates), name="portfolio")
    else:
        port_ret_series = pd.Series(dtype=float, name="portfolio")
    
    # Compute benchmark returns for the same holding periods as portfolio
    # BUG FIX: Previously used daily pct_change reindexed to monthly dates,
    # which only captured a single day's return instead of the full period.
    # Now we compute period returns between consecutive rebalance dates.
    benchmark_returns = []
    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]
        if next_rebal in prices.index and rebal_date in prices.index:
            start_price = prices.loc[rebal_date, benchmark]
            end_price = prices.loc[next_rebal, benchmark]
            bench_ret = (end_price / start_price) - 1
            benchmark_returns.append((next_rebal, bench_ret))
    
    if benchmark_returns:
        bench_dates, bench_rets = zip(*benchmark_returns)
        bench_ret_series = pd.Series(
            bench_rets, 
            index=pd.DatetimeIndex(bench_dates), 
            name="benchmark"
        )
    else:
        bench_ret_series = pd.Series(dtype=float, name="benchmark")
    
    # Align benchmark to portfolio dates (should already match, but ensure consistency)
    if len(port_ret_series) > 0 and len(bench_ret_series) > 0:
        common_dates = port_ret_series.index.intersection(bench_ret_series.index)
        port_ret_series = port_ret_series.loc[common_dates]
        bench_ret_series = bench_ret_series.loc[common_dates]
    
    # Compute metrics
    periods_per_year = 12 if rebalance_frequency == "M" else 52
    metrics = compute_all_metrics(
        port_ret_series,
        bench_ret_series,
        periods_per_year=periods_per_year,
    )
    
    # Add average IC
    if ic_history:
        metrics["average_ic"] = np.mean([ic for _, ic in ic_history])
    
    # =========================================================================
    # TRANSACTION COST MODELING
    # =========================================================================
    turnover_series_out = None
    avg_turnover = None
    p95_turnover = None
    net_returns_out = None
    net_metrics_out = None
    
    if cost_config is not None and len(weights_history) > 0:
        # Compute turnover series from weights history
        # Note: weights_history dates are rebalance dates; returns are indexed by next_rebal
        # We need to shift turnover dates to match return dates
        turnover_raw = compute_turnover_series(weights_history, include_initial_build=False)
        
        # Shift turnover to next period (turnover at date t affects returns for period t to t+1)
        # Since returns are indexed by next_rebal, we need to align turnover with that
        if len(turnover_raw) > 0 and len(port_ret_series) > 0:
            # Create mapping: turnover at weights_history[i] applies to return at next_rebal[i]
            # weights_history dates are rebal dates, returns are indexed by next_rebal
            # So we need to shift turnover forward by one period
            turnover_dates = []
            turnover_values = []
            for i, (rebal_date, _) in enumerate(weights_history):
                if i < len(weights_history) - 1:
                    # Next rebal date is when the return is realized
                    next_rebal = weights_history[i + 1][0] if i + 1 < len(weights_history) else None
                    # But returns use the actual next rebalance, which may be different
                    # Match by index position
                    pass
            
            # Simpler approach: match by position
            # port_ret_series has len(weights_history) - 1 entries (one per holding period)
            # turnover_raw has len(weights_history) entries
            # turnover[i] (for i > 0) is the turnover from period i-1 to i
            # This turnover applies to the return for period i
            
            # Skip first turnover (NaN), align rest with returns
            turnover_aligned = pd.Series(
                turnover_raw.iloc[1:].values,  # Skip first (NaN)
                index=port_ret_series.index[:len(turnover_raw) - 1],
                name="turnover"
            )
            
            # Extend to full return series length if needed
            if len(turnover_aligned) < len(port_ret_series):
                missing_dates = port_ret_series.index[len(turnover_aligned):]
                extra = pd.Series(np.nan, index=missing_dates, name="turnover")
                turnover_aligned = pd.concat([turnover_aligned, extra])
            
            turnover_series_out = turnover_aligned
            
            # Compute turnover statistics (excluding NaN)
            valid_turnover = turnover_aligned.dropna()
            if len(valid_turnover) > 0:
                avg_turnover = float(valid_turnover.mean())
                p95_turnover = float(np.percentile(valid_turnover, 95))
            
            # Compute net returns
            net_returns_out = compute_net_returns(port_ret_series, turnover_aligned, cost_config)
            
            # Compute net metrics
            net_metrics_out = compute_net_metrics(
                net_returns_out,
                bench_ret_series,
                periods_per_year=periods_per_year,
            )
    
    logger.info(f"Backtest complete. Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    
    return BacktestResult(
        portfolio_returns=port_ret_series,
        benchmark_returns=bench_ret_series,
        weights_history=weights_history,
        ic_history=ic_history,
        metrics=metrics,
        config=config,
        scores_history=scores_history if scores_history else None,
        turnover_series=turnover_series_out,
        avg_turnover=avg_turnover,
        p95_turnover=p95_turnover,
        net_returns=net_returns_out,
        net_metrics=net_metrics_out,
    )


def run_simple_backtest(
    prices: pd.DataFrame,
    config: PipelineConfig | None = None,
) -> BacktestResult:
    """
    Run a backtest with default settings from config.

    Convenience wrapper around run_backtest.

    Args:
        prices: Adjusted close prices.
        config: Pipeline configuration.

    Returns:
        BacktestResult.
    """
    if config is None:
        config = get_default_config()
    
    return run_backtest(
        prices=prices,
        horizon_days=config.research.horizon_days,
        top_k=config.research.top_k,
        benchmark=config.research.benchmark,
        rebalance_frequency=config.research.rebalance_frequency,
        min_train_periods=config.research.lookback_days,
        config=config,
    )


def run_regime_switch_strategy(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    momentum_result: BacktestResult,
    quality_result: BacktestResult,
    top_k: int,
    benchmark: str = "SPY",
    regime_config: RegimeConfig | None = None,
    cost_config: CostModelConfig | None = None,
) -> BacktestResult:
    """
    Run a regime-switching strategy that switches between Momentum and Quality models.
    
    At each rebalance date:
    - Compute the market regime (MOMENTUM or QUALITY) based on vol + trend
    - Use the corresponding model's scores to select top-K stocks
    - Compute portfolio returns
    
    Args:
        prices: Adjusted close prices with tickers as columns.
        spy_prices: SPY price series for regime computation.
        momentum_result: BacktestResult from Momentum (Tech-only) model.
        quality_result: BacktestResult from Quality (Fundamentals-only) model.
        top_k: Number of stocks to select.
        benchmark: Benchmark ticker.
        regime_config: Configuration for regime signal. Uses defaults if None.
        cost_config: Transaction cost configuration. If None, no costs applied.
    
    Returns:
        BacktestResult for the regime-switch strategy.
    
    Raises:
        ValueError: If either result lacks scores_history.
    """
    if momentum_result.scores_history is None:
        raise ValueError("momentum_result must have scores_history")
    if quality_result.scores_history is None:
        raise ValueError("quality_result must have scores_history")
    
    # Build lookup dicts for scores by date
    momentum_scores_by_date = {d: s for d, s in momentum_result.scores_history}
    quality_scores_by_date = {d: s for d, s in quality_result.scores_history}
    
    # Compute regime series
    regime_series = compute_regime_series(spy_prices, regime_config)
    
    # Get common rebalance dates
    momentum_dates = set(momentum_scores_by_date.keys())
    quality_dates = set(quality_scores_by_date.keys())
    common_dates = sorted(momentum_dates & quality_dates)
    
    if len(common_dates) < 2:
        raise ValueError("Not enough common rebalance dates")
    
    logger.info(f"Running regime-switch strategy over {len(common_dates)} dates")
    
    # Log regime summary
    regime_summary = compute_regime_summary(regime_series, common_dates)
    logger.info(
        f"Regime distribution: MOMENTUM={regime_summary['MOMENTUM']}, "
        f"QUALITY={regime_summary['QUALITY']}"
    )
    
    # Storage
    portfolio_returns = []
    weights_history = []
    ic_history = []
    scores_history = []
    regime_history = []  # Track which regime was used
    
    tradeable_tickers = [t for t in prices.columns if t != benchmark]
    
    for i in range(len(common_dates) - 1):
        rebal_date = common_dates[i]
        next_rebal = common_dates[i + 1]
        
        # Get regime for this date
        regime = get_regime_for_date(regime_series, rebal_date)
        regime_history.append((rebal_date, regime.value))
        
        # Select scores based on regime
        if regime == MarketRegime.MOMENTUM:
            scores = momentum_scores_by_date[rebal_date]
        else:
            scores = quality_scores_by_date[rebal_date]
        
        scores_history.append((rebal_date, scores.copy()))
        
        # Build portfolio (top-K equal weight)
        weights = build_equal_weight_portfolio(scores, top_k)
        weights_history.append((rebal_date, weights))
        
        # Compute holding period returns
        if next_rebal in prices.index and rebal_date in prices.index:
            start_prices = prices.loc[rebal_date, tradeable_tickers]
            end_prices = prices.loc[next_rebal, tradeable_tickers]
            period_returns = (end_prices / start_prices) - 1
            
            port_return = compute_portfolio_return(weights, period_returns)
            portfolio_returns.append((next_rebal, port_return))
            
            # Compute IC
            realized = period_returns.loc[scores.index]
            ic = compute_ic(scores, realized)
            ic_history.append((next_rebal, ic))
    
    # Convert to Series
    if portfolio_returns:
        dates, returns = zip(*portfolio_returns)
        port_ret_series = pd.Series(returns, index=pd.DatetimeIndex(dates), name="portfolio")
    else:
        port_ret_series = pd.Series(dtype=float, name="portfolio")
    
    # Compute benchmark returns for the same holding periods
    benchmark_returns = []
    for i in range(len(common_dates) - 1):
        rebal_date = common_dates[i]
        next_rebal = common_dates[i + 1]
        if next_rebal in prices.index and rebal_date in prices.index:
            start_price = prices.loc[rebal_date, benchmark]
            end_price = prices.loc[next_rebal, benchmark]
            bench_ret = (end_price / start_price) - 1
            benchmark_returns.append((next_rebal, bench_ret))
    
    if benchmark_returns:
        bench_dates, bench_rets = zip(*benchmark_returns)
        bench_ret_series = pd.Series(
            bench_rets,
            index=pd.DatetimeIndex(bench_dates),
            name="benchmark"
        )
    else:
        bench_ret_series = pd.Series(dtype=float, name="benchmark")
    
    # Align series
    if len(port_ret_series) > 0 and len(bench_ret_series) > 0:
        common_idx = port_ret_series.index.intersection(bench_ret_series.index)
        port_ret_series = port_ret_series.loc[common_idx]
        bench_ret_series = bench_ret_series.loc[common_idx]
    
    # Compute metrics
    metrics = compute_all_metrics(
        port_ret_series,
        bench_ret_series,
        periods_per_year=12,  # Assumes monthly rebalancing
    )
    
    # Add average IC
    if ic_history:
        metrics["average_ic"] = np.mean([ic for _, ic in ic_history])
    
    # Add regime-specific info to metrics
    metrics["regime_momentum_count"] = regime_summary[MarketRegime.MOMENTUM.value]
    metrics["regime_quality_count"] = regime_summary[MarketRegime.QUALITY.value]
    
    # =========================================================================
    # TRANSACTION COST MODELING
    # =========================================================================
    turnover_series_out = None
    avg_turnover = None
    p95_turnover = None
    net_returns_out = None
    net_metrics_out = None
    
    if cost_config is not None and len(weights_history) > 0:
        turnover_raw = compute_turnover_series(weights_history, include_initial_build=False)
        
        if len(turnover_raw) > 0 and len(port_ret_series) > 0:
            turnover_aligned = pd.Series(
                turnover_raw.iloc[1:].values,
                index=port_ret_series.index[:len(turnover_raw) - 1],
                name="turnover"
            )
            
            if len(turnover_aligned) < len(port_ret_series):
                missing_dates = port_ret_series.index[len(turnover_aligned):]
                extra = pd.Series(np.nan, index=missing_dates, name="turnover")
                turnover_aligned = pd.concat([turnover_aligned, extra])
            
            turnover_series_out = turnover_aligned
            
            valid_turnover = turnover_aligned.dropna()
            if len(valid_turnover) > 0:
                avg_turnover = float(valid_turnover.mean())
                p95_turnover = float(np.percentile(valid_turnover, 95))
            
            net_returns_out = compute_net_returns(port_ret_series, turnover_aligned, cost_config)
            net_metrics_out = compute_net_metrics(net_returns_out, bench_ret_series, periods_per_year=12)
    
    logger.info(f"Regime-switch backtest complete. Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    
    # Use momentum config as the base (regime-switch doesn't have its own config)
    return BacktestResult(
        portfolio_returns=port_ret_series,
        benchmark_returns=bench_ret_series,
        weights_history=weights_history,
        ic_history=ic_history,
        metrics=metrics,
        config=momentum_result.config,
        scores_history=scores_history if scores_history else None,
        turnover_series=turnover_series_out,
        avg_turnover=avg_turnover,
        p95_turnover=p95_turnover,
        net_returns=net_returns_out,
        net_metrics=net_metrics_out,
    )


def run_ic_weighted_ensemble(
    prices: pd.DataFrame,
    momentum_result: BacktestResult,
    quality_result: BacktestResult,
    top_k: int,
    benchmark: str = "SPY",
    ic_weight_config: ICWeightConfig | None = None,
    cost_config: CostModelConfig | None = None,
) -> BacktestResult:
    """
    Run an IC-weighted ensemble that blends Momentum and Quality model scores.
    
    At each rebalance date:
    - Compute rolling mean IC for each model
    - Set weights proportional to max(rolling_IC, 0)
    - Blend scores: score_ensemble = w_mom * score_mom + w_qual * score_qual
    - Select top-K stocks by ensemble score
    - Compute portfolio returns
    
    Args:
        prices: Adjusted close prices with tickers as columns.
        momentum_result: BacktestResult from Momentum (Tech-only) model.
        quality_result: BacktestResult from Quality (Fundamentals-only) model.
        top_k: Number of stocks to select.
        benchmark: Benchmark ticker.
        ic_weight_config: Configuration for IC-based weighting. Uses defaults if None.
        cost_config: Transaction cost configuration. If None, no costs applied.
    
    Returns:
        BacktestResult for the IC-weighted ensemble strategy.
    
    Raises:
        ValueError: If either result lacks scores_history or ic_history.
    """
    if momentum_result.scores_history is None:
        raise ValueError("momentum_result must have scores_history")
    if quality_result.scores_history is None:
        raise ValueError("quality_result must have scores_history")
    if not momentum_result.ic_history:
        raise ValueError("momentum_result must have ic_history")
    if not quality_result.ic_history:
        raise ValueError("quality_result must have ic_history")
    
    if ic_weight_config is None:
        ic_weight_config = ICWeightConfig()
    
    # Build lookup dicts for scores by date
    momentum_scores_by_date = {d: s for d, s in momentum_result.scores_history}
    quality_scores_by_date = {d: s for d, s in quality_result.scores_history}
    
    # Build IC series
    ic_mom_dates, ic_mom_values = zip(*momentum_result.ic_history)
    ic_mom = pd.Series(ic_mom_values, index=pd.DatetimeIndex(ic_mom_dates))
    
    ic_qual_dates, ic_qual_values = zip(*quality_result.ic_history)
    ic_qual = pd.Series(ic_qual_values, index=pd.DatetimeIndex(ic_qual_dates))
    
    # Compute IC-based weights
    ic_weights_df = compute_ic_weights(ic_mom, ic_qual, ic_weight_config)
    
    # Get common rebalance dates (where we have scores from both models)
    momentum_dates = set(momentum_scores_by_date.keys())
    quality_dates = set(quality_scores_by_date.keys())
    common_dates = sorted(momentum_dates & quality_dates)
    
    if len(common_dates) < 2:
        raise ValueError("Not enough common rebalance dates")
    
    logger.info(f"Running IC-weighted ensemble over {len(common_dates)} dates")
    
    # Storage
    portfolio_returns = []
    weights_history = []
    ic_history = []
    scores_history = []
    weight_history = []  # Track w_mom, w_qual over time
    
    tradeable_tickers = [t for t in prices.columns if t != benchmark]
    
    for i in range(len(common_dates) - 1):
        rebal_date = common_dates[i]
        next_rebal = common_dates[i + 1]
        
        # Get scores from both models
        scores_mom = momentum_scores_by_date[rebal_date]
        scores_qual = quality_scores_by_date[rebal_date]
        
        # Align tickers (use intersection)
        common_tickers = scores_mom.index.intersection(scores_qual.index)
        scores_mom = scores_mom.loc[common_tickers]
        scores_qual = scores_qual.loc[common_tickers]
        
        # Get IC weights for this date
        # Use the most recent available weight (IC is computed from previous period)
        available_weight_dates = ic_weights_df.index[ic_weights_df.index <= rebal_date]
        
        if len(available_weight_dates) == 0:
            # No IC history yet - use equal weights
            w_mom = 0.5
            w_qual = 0.5
        else:
            latest_weight_date = available_weight_dates[-1]
            w_mom = ic_weights_df.loc[latest_weight_date, 'w_mom']
            w_qual = ic_weights_df.loc[latest_weight_date, 'w_qual']
        
        weight_history.append((rebal_date, w_mom, w_qual))
        
        # Blend scores
        scores_ensemble = w_mom * scores_mom + w_qual * scores_qual
        scores_history.append((rebal_date, scores_ensemble.copy()))
        
        # Build portfolio (top-K equal weight)
        weights = build_equal_weight_portfolio(scores_ensemble, top_k)
        weights_history.append((rebal_date, weights))
        
        # Compute holding period returns
        if next_rebal in prices.index and rebal_date in prices.index:
            start_prices = prices.loc[rebal_date, tradeable_tickers]
            end_prices = prices.loc[next_rebal, tradeable_tickers]
            period_returns = (end_prices / start_prices) - 1
            
            port_return = compute_portfolio_return(weights, period_returns)
            portfolio_returns.append((next_rebal, port_return))
            
            # Compute IC for ensemble scores
            realized = period_returns.loc[scores_ensemble.index]
            ic = compute_ic(scores_ensemble, realized)
            ic_history.append((next_rebal, ic))
    
    # Log weight summary
    if weight_history:
        avg_w_mom = np.mean([w[1] for w in weight_history])
        avg_w_qual = np.mean([w[2] for w in weight_history])
        logger.info(
            f"Average IC weights: Momentum={avg_w_mom:.2f}, Quality={avg_w_qual:.2f}"
        )
    
    # Convert to Series
    if portfolio_returns:
        dates, returns = zip(*portfolio_returns)
        port_ret_series = pd.Series(returns, index=pd.DatetimeIndex(dates), name="portfolio")
    else:
        port_ret_series = pd.Series(dtype=float, name="portfolio")
    
    # Compute benchmark returns for the same holding periods
    benchmark_returns = []
    for i in range(len(common_dates) - 1):
        rebal_date = common_dates[i]
        next_rebal = common_dates[i + 1]
        if next_rebal in prices.index and rebal_date in prices.index:
            start_price = prices.loc[rebal_date, benchmark]
            end_price = prices.loc[next_rebal, benchmark]
            bench_ret = (end_price / start_price) - 1
            benchmark_returns.append((next_rebal, bench_ret))
    
    if benchmark_returns:
        bench_dates, bench_rets = zip(*benchmark_returns)
        bench_ret_series = pd.Series(
            bench_rets,
            index=pd.DatetimeIndex(bench_dates),
            name="benchmark"
        )
    else:
        bench_ret_series = pd.Series(dtype=float, name="benchmark")
    
    # Align series
    if len(port_ret_series) > 0 and len(bench_ret_series) > 0:
        common_idx = port_ret_series.index.intersection(bench_ret_series.index)
        port_ret_series = port_ret_series.loc[common_idx]
        bench_ret_series = bench_ret_series.loc[common_idx]
    
    # Compute metrics
    metrics = compute_all_metrics(
        port_ret_series,
        bench_ret_series,
        periods_per_year=12,  # Assumes monthly rebalancing
    )
    
    # Add average IC
    if ic_history:
        metrics["average_ic"] = np.mean([ic for _, ic in ic_history])
    
    # Add weight summary to metrics
    if weight_history:
        metrics["avg_weight_momentum"] = np.mean([w[1] for w in weight_history])
        metrics["avg_weight_quality"] = np.mean([w[2] for w in weight_history])
    
    # =========================================================================
    # TRANSACTION COST MODELING
    # =========================================================================
    turnover_series_out = None
    avg_turnover = None
    p95_turnover = None
    net_returns_out = None
    net_metrics_out = None
    
    if cost_config is not None and len(weights_history) > 0:
        turnover_raw = compute_turnover_series(weights_history, include_initial_build=False)
        
        if len(turnover_raw) > 0 and len(port_ret_series) > 0:
            turnover_aligned = pd.Series(
                turnover_raw.iloc[1:].values,
                index=port_ret_series.index[:len(turnover_raw) - 1],
                name="turnover"
            )
            
            if len(turnover_aligned) < len(port_ret_series):
                missing_dates = port_ret_series.index[len(turnover_aligned):]
                extra = pd.Series(np.nan, index=missing_dates, name="turnover")
                turnover_aligned = pd.concat([turnover_aligned, extra])
            
            turnover_series_out = turnover_aligned
            
            valid_turnover = turnover_aligned.dropna()
            if len(valid_turnover) > 0:
                avg_turnover = float(valid_turnover.mean())
                p95_turnover = float(np.percentile(valid_turnover, 95))
            
            net_returns_out = compute_net_returns(port_ret_series, turnover_aligned, cost_config)
            net_metrics_out = compute_net_metrics(net_returns_out, bench_ret_series, periods_per_year=12)
    
    logger.info(f"IC-weighted ensemble complete. Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    
    # Use momentum config as the base
    return BacktestResult(
        portfolio_returns=port_ret_series,
        benchmark_returns=bench_ret_series,
        weights_history=weights_history,
        ic_history=ic_history,
        metrics=metrics,
        config=momentum_result.config,
        scores_history=scores_history if scores_history else None,
        turnover_series=turnover_series_out,
        avg_turnover=avg_turnover,
        p95_turnover=p95_turnover,
        net_returns=net_returns_out,
        net_metrics=net_metrics_out,
    )


def run_enhanced_backtest(
    prices: pd.DataFrame,
    horizon_days: int,
    top_k: int,
    benchmark: str = "SPY",
    rebalance_frequency: str = "M",
    min_train_periods: int = 252,
    config: PipelineConfig | None = None,
) -> BacktestResult:
    """
    Run a walk-forward backtest with enhanced model.

    This function uses:
    - XGBoost regression (default) or ranking model
    - Volatility-normalized forward returns as target (default)
    - Enhanced features (short reversal, residual momentum, idio vol, MAD, sector OHE)
    - Rank-weighted portfolio construction

    Args:
        prices: Adjusted close prices with tickers as columns.
        horizon_days: Forward return horizon in trading days.
        top_k: Number of stocks to select.
        benchmark: Benchmark ticker for comparison.
        rebalance_frequency: How often to rebalance ('M', 'W', 'D').
        min_train_periods: Minimum training periods before first trade.
        config: Pipeline configuration with enhanced_model and enhanced_portfolio.

    Returns:
        BacktestResult with portfolio returns, metrics, etc.
    """
    from auto_researcher.models.xgb_ranking_model import (
        XGBRankingModel, XGBRankingConfig,
        XGBRegressionModel, XGBRegressionConfig,
        create_model,
    )
    from auto_researcher.features.enhanced import (
        EnhancedFeatureConfig,
        compute_all_enhanced_features,
    )
    from auto_researcher.features.targets import (
        TargetConfig,
        build_targets,
        compute_forward_returns,
    )
    from auto_researcher.backtest.enhanced_portfolio import (
        EnhancedPortfolioConfig as PortfolioConfig,
        build_rank_weighted_portfolio,
        build_sector_neutral_portfolio,
        get_ticker_sectors,
    )
    
    # Feature selection helper using XGBoost's native importance (more reliable than SHAP)
    def get_top_features_by_importance(model, X_train: pd.DataFrame, top_k: int) -> list[str]:
        """Select top features using XGBoost's native feature importance."""
        try:
            # Get gain-based importance from XGBoost
            importance_dict = model.model.get_booster().get_score(importance_type='gain')
            # Map feature indices (f0, f1, etc.) to column names
            feature_names = X_train.columns.tolist()
            importance_series = pd.Series(0.0, index=feature_names)
            for key, value in importance_dict.items():
                if key.startswith('f'):
                    idx = int(key[1:])
                    if idx < len(feature_names):
                        importance_series.iloc[idx] = value
            top_features = importance_series.nlargest(min(top_k, len(importance_series))).index.tolist()
            
            # Filter out features with zero importance
            top_features = [f for f in top_features if importance_series[f] > 0]
            
            if len(top_features) < 5:
                logger.warning(f"Only {len(top_features)} features with non-zero importance. Using all features.")
                return X_train.columns.tolist()
            
            return top_features
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            return X_train.columns.tolist()
    
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} not found in prices")
    
    if config is None:
        config = get_default_config()
    
    enhanced_cfg = config.enhanced_model
    portfolio_cfg = config.enhanced_portfolio
    
    # Get rebalance dates
    all_dates = prices.index
    rebal_dates = get_rebalance_dates(all_dates, rebalance_frequency)
    
    # Filter to dates with enough history
    min_start_idx = min_train_periods + horizon_days
    if min_start_idx >= len(all_dates):
        raise ValueError("Insufficient data for backtest")
    
    min_start_date = all_dates[min_start_idx]
    rebal_dates = [d for d in rebal_dates if d >= min_start_date]
    
    if len(rebal_dates) < 2:
        raise ValueError("Not enough rebalance dates after warmup")
    
    # Determine model type description
    model_type_desc = enhanced_cfg.model_type
    target_mode_desc = enhanced_cfg.target_mode
    if enhanced_cfg.target_mode == "vol_norm":
        target_mode_desc = "vol_normalized_forward_return"
    
    logger.info(f"Running ENHANCED backtest with {len(rebal_dates)} rebalance dates")
    logger.info(f"  Model type: {model_type_desc}")
    logger.info(f"  Target mode: {target_mode_desc}")
    logger.info(f"  Weighting: {portfolio_cfg.weighting_scheme}")
    
    # Build enhanced feature config from enhanced_model config
    feature_config = EnhancedFeatureConfig(
        use_short_reversal=enhanced_cfg.use_short_reversal,
        use_residual_mom=enhanced_cfg.use_residual_momentum,
        use_idio_vol=enhanced_cfg.use_idio_vol,
        use_mad_metrics=enhanced_cfg.use_mad,
        use_sector_ohe=enhanced_cfg.use_sector_ohe,
        use_cross_sec_norm=enhanced_cfg.cross_sectional_normalize,
        cross_sec_norm_robust=enhanced_cfg.normalize_type == "robust",
    )
    
    # Build target config
    target_config = TargetConfig(
        mode=enhanced_cfg.target_mode,
        horizon_days=horizon_days,
    )
    
    # Create model based on type
    if enhanced_cfg.model_type == "regression":
        # Select objective based on robust_objective flag
        objective = "reg:pseudohubererror" if enhanced_cfg.robust_objective else "reg:squarederror"
        if enhanced_cfg.robust_objective:
            logger.info(f"  Using robust objective: {objective}")
        
        # Use configured hyperparameters or defaults
        reg_config = XGBRegressionConfig(
            objective=objective,
            n_estimators=enhanced_cfg.xgb_n_estimators or 300,
            max_depth=enhanced_cfg.xgb_max_depth or 4,
            learning_rate=enhanced_cfg.xgb_learning_rate or 0.05,
            reg_lambda=enhanced_cfg.xgb_reg_lambda or 2.0,
            reg_alpha=enhanced_cfg.xgb_reg_alpha or 0.1,
            subsample=enhanced_cfg.xgb_subsample or 0.8,
            colsample_bytree=enhanced_cfg.xgb_colsample_bytree or 0.8,
        )
        model = XGBRegressionModel(reg_config)
        use_regression_model = True
    elif enhanced_cfg.model_type in ("rank_pairwise", "rank_ndcg"):
        objective = "rank:pairwise" if enhanced_cfg.model_type == "rank_pairwise" else "rank:ndcg"
        xgb_config = XGBRankingConfig(
            objective=objective,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            reg_lambda=2.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model = XGBRankingModel(xgb_config)
        use_regression_model = False
    else:
        raise ValueError(f"Unknown model type: {enhanced_cfg.model_type}")
    
    # Get tickers excluding benchmark
    tradeable_tickers = [t for t in prices.columns if t != benchmark]
    
    # Build portfolio config
    port_config = PortfolioConfig(
        top_k=portfolio_cfg.top_k,
        weighting_scheme=portfolio_cfg.weighting_scheme,
        rank_tau=portfolio_cfg.rank_tau,
        neutralization=portfolio_cfg.neutralization,
        max_position_weight=portfolio_cfg.max_position_weight,
    )
    
    # Storage
    portfolio_returns = []
    weights_history = []
    ic_history = []
    scores_history = []
    
    # Rolling window configuration
    use_rolling_window = enhanced_cfg.rolling_window
    rolling_window_days = enhanced_cfg.rolling_window_days
    if use_rolling_window:
        logger.info(f"  Using rolling window: {rolling_window_days} days")
    else:
        logger.info(f"  Using expanding window (all available history)")
    
    # SHAP feature selection configuration
    if enhanced_cfg.use_shap_selection:
        logger.info(f"  Using SHAP feature selection: top {enhanced_cfg.shap_top_k_features} features")
    
    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]
        
        logger.debug(f"Rebalancing on {rebal_date}")
        
        # Get training data up to rebal_date (inclusive)
        train_end_idx = all_dates.get_loc(rebal_date)
        
        # Apply rolling window if enabled
        if use_rolling_window:
            # Use only the last rolling_window_days of data
            train_start_idx = max(0, train_end_idx - rolling_window_days + 1)
        else:
            # Use all available history (expanding window)
            train_start_idx = 0
        
        train_prices = prices.iloc[train_start_idx:train_end_idx + 1]
        
        # Compute enhanced features for training data
        try:
            # Compute enhanced features - pass benchmark string, not returns
            enhanced_features_wide = compute_all_enhanced_features(
                train_prices,  # Include benchmark for beta calculation
                benchmark=benchmark,
                config=feature_config,
            )
            
            # Reshape features from wide (date x (ticker, feature)) to long (date, ticker) x feature
            # Stack ticker level to rows
            enhanced_features = enhanced_features_wide.stack(level=0, future_stack=True)
            enhanced_features.index.names = ["date", "ticker"]
            
            # Build targets - pass prices WITH benchmark for benchmark forward returns
            targets_df = build_targets(
                train_prices,  # Keep benchmark for target computation
                target_config,
                benchmark=benchmark,
            )
            
            # Stack targets to match features index format
            targets_stacked = targets_df.stack()
            targets_stacked.index.names = ["date", "ticker"]
            
            # Find common index
            common_idx = enhanced_features.index.intersection(targets_stacked.index)
            
            # Filter both to common
            X_train = enhanced_features.loc[common_idx]
            y_train = targets_stacked.loc[common_idx]
            
            # Drop columns that are all NaN (e.g., 252-day features when insufficient history)
            valid_cols = X_train.columns[X_train.notna().any()]
            X_train = X_train[valid_cols]
            
            # Now drop rows with any NaN in remaining columns
            valid_rows = X_train.dropna().index
            X_train = X_train.loc[valid_rows]
            y_train = y_train.loc[valid_rows].dropna()
            
            # Align again after dropping NaN
            common_idx = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common_idx]
            y_train = y_train.loc[common_idx]
            
        except Exception as e:
            logger.warning(f"Failed to prepare enhanced training data for {rebal_date}: {e}")
            continue
        
        if len(X_train) < 50:
            logger.warning(f"Insufficient training data for {rebal_date}: {len(X_train)} samples")
            continue
        
        # Train model (both regression and ranking models have same fit API)
        try:
            # Feature selection based on importance
            if enhanced_cfg.use_shap_selection and use_regression_model:
                # First, train on all features to compute feature importance
                model.fit(X_train, y_train)
                
                # Get top features using XGBoost's gain-based importance
                selected_features = get_top_features_by_importance(
                    model, X_train, enhanced_cfg.shap_top_k_features
                )
                logger.debug(f"Feature selection: kept {len(selected_features)} features: {selected_features[:5]}...")
                
                # Re-train on selected features only
                X_train_selected = X_train[selected_features]
                model.fit(X_train_selected, y_train)
                train_columns = selected_features
            else:
                # Standard training on all features
                model.fit(X_train, y_train)
                train_columns = X_train.columns.tolist()
        except Exception as e:
            logger.warning(f"Failed to train model for {rebal_date}: {e}")
            continue
        
        # Get features for current date for prediction
        try:
            # Get full price data up to rebal_date for feature computation
            current_features_wide = compute_all_enhanced_features(
                prices.loc[:rebal_date],
                benchmark=benchmark,
                config=feature_config,
            )
            
            # Stack to (date, ticker) x features format
            current_features_long = current_features_wide.stack(level=0, future_stack=True)
            current_features_long.index.names = ["date", "ticker"]
            
            # Get just the features for the rebal_date
            if rebal_date in current_features_long.index.get_level_values("date"):
                current_features = current_features_long.loc[rebal_date]
            else:
                # Find closest date
                available_dates = current_features_long.index.get_level_values("date").unique()
                closest_date = available_dates[available_dates <= rebal_date].max()
                current_features = current_features_long.loc[closest_date]
            
            # Use only the columns that were used in training
            current_features = current_features[train_columns]
            
        except Exception as e:
            logger.warning(f"Failed to get features for {rebal_date}: {e}")
            continue
        
        # Filter to tradeable tickers (current_features now has ticker as index)
        current_features = current_features.loc[
            current_features.index.isin(tradeable_tickers)
        ]
        
        # Drop rows with NaN in the training columns
        current_features = current_features.dropna()
        
        if len(current_features) == 0:
            logger.warning(f"No valid features for {rebal_date}")
            continue
        
        # Predict scores (higher = better for both regression and ranking models)
        try:
            scores = model.predict_with_index(current_features)
            
            # Optional: Invert signal if configured and IC is consistently negative
            # (This would be checked based on training IC, not implemented in this loop)
        except Exception as e:
            logger.warning(f"Failed to predict for {rebal_date}: {e}")
            continue
        
        # Store scores
        scores_history.append((rebal_date, scores.copy()))
        
        # Build portfolio - use sector-neutral if configured
        if portfolio_cfg.sector_neutral_ranking:
            # Get sector assignments for all scored tickers
            ticker_sectors = get_ticker_sectors(scores.index.tolist())
            
            weights = build_sector_neutral_portfolio(
                scores=scores,
                top_k=portfolio_cfg.top_k,
                sectors=ticker_sectors,
                weighting_scheme=portfolio_cfg.weighting_scheme,
                rank_tau=portfolio_cfg.rank_tau,
            )
        else:
            # Standard rank-weighted portfolio
            weights = build_rank_weighted_portfolio(scores, port_config)
        
        weights_history.append((rebal_date, weights))
        
        # Compute holding period returns
        if next_rebal in prices.index and rebal_date in prices.index:
            start_prices = prices.loc[rebal_date, tradeable_tickers]
            end_prices = prices.loc[next_rebal, tradeable_tickers]
            period_returns = (end_prices / start_prices) - 1
            
            port_return = compute_portfolio_return(weights, period_returns)
            portfolio_returns.append((next_rebal, port_return))
            
            # Compute IC
            realized = period_returns.loc[scores.index]
            ic = compute_ic(scores, realized)
            ic_history.append((next_rebal, ic))
    
    # Convert to Series
    if portfolio_returns:
        dates, returns = zip(*portfolio_returns)
        port_ret_series = pd.Series(returns, index=pd.DatetimeIndex(dates), name="portfolio")
    else:
        port_ret_series = pd.Series(dtype=float, name="portfolio")
    
    # Compute benchmark returns
    benchmark_returns = []
    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]
        if next_rebal in prices.index and rebal_date in prices.index:
            start_price = prices.loc[rebal_date, benchmark]
            end_price = prices.loc[next_rebal, benchmark]
            bench_ret = (end_price / start_price) - 1
            benchmark_returns.append((next_rebal, bench_ret))
    
    if benchmark_returns:
        bench_dates, bench_rets = zip(*benchmark_returns)
        bench_ret_series = pd.Series(
            bench_rets, 
            index=pd.DatetimeIndex(bench_dates), 
            name="benchmark"
        )
    else:
        bench_ret_series = pd.Series(dtype=float, name="benchmark")
    
    # Align
    if len(port_ret_series) > 0 and len(bench_ret_series) > 0:
        common_dates = port_ret_series.index.intersection(bench_ret_series.index)
        port_ret_series = port_ret_series.loc[common_dates]
        bench_ret_series = bench_ret_series.loc[common_dates]
    
    # Compute metrics
    periods_per_year = 12 if rebalance_frequency == "M" else 52
    metrics = compute_all_metrics(
        port_ret_series,
        bench_ret_series,
        periods_per_year=periods_per_year,
    )
    
    # Add model metadata
    metrics["target_type"] = enhanced_cfg.target_mode
    metrics["model_type"] = enhanced_cfg.model_type
    
    # Add average IC and IC stats
    if ic_history:
        ic_values = [ic for _, ic in ic_history]
        metrics["average_ic"] = np.mean(ic_values)
        metrics["ic_std"] = np.std(ic_values)
        ic_mean = metrics["average_ic"]
        ic_std = metrics["ic_std"]
        metrics["ic_ir"] = ic_mean / ic_std if ic_std > 0 else 0.0
        metrics["ic_positive_rate"] = np.mean([ic > 0 for ic in ic_values])
    
    logger.info(f"Enhanced backtest complete. Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    
    return BacktestResult(
        portfolio_returns=port_ret_series,
        benchmark_returns=bench_ret_series,
        weights_history=weights_history,
        ic_history=ic_history,
        metrics=metrics,
        config=config,
        scores_history=scores_history if scores_history else None,
    )
