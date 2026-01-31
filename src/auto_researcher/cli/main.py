"""
CLI entry point for Auto-Researcher.

This module provides a command-line interface for running demo backtests
and exploring the research pipeline.
"""

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

from auto_researcher.config import (
    PipelineConfig,
    ResearchConfig,
    FeatureConfig,
    ModelConfig,
    EnhancedModelConfig,
    EnhancedPortfolioConfig,
    get_default_config,
)
from auto_researcher.data.universe import get_universe, get_sp500_universe
from auto_researcher.data.price_loader import download_price_history, get_adjusted_close
from auto_researcher.backtest.runner import run_backtest, run_enhanced_backtest, BacktestResult
from auto_researcher.backtest.metrics import compute_cumulative_returns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-Researcher: ML-based stock ranking and portfolio construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  auto-researcher                    # Run with default settings
  auto-researcher --top-k 5          # Select top 5 stocks
  auto-researcher --horizon 21       # Use 21-day (1 month) horizon
  auto-researcher --sp500            # Use S&P 500 universe (slow)
        """,
    )
    
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=None,
        help="List of tickers to analyze (default: AAPL, MSFT, GOOG, AMZN, META, NVDA, SPY)",
    )
    
    parser.add_argument(
        "--sp500",
        action="store_true",
        help="Use S&P 500 universe (takes longer to download)",
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-01-01",
        help="Start date for data (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data (YYYY-MM-DD, default: today)",
    )
    
    parser.add_argument(
        "--horizon",
        type=int,
        default=63,
        help="Forward return horizon in trading days (default: 63 ≈ 3 months)",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of stocks to select (default: 5)",
    )
    
    parser.add_argument(
        "--rebalance",
        type=str,
        choices=["M", "W"],
        default="M",
        help="Rebalance frequency: M=monthly, W=weekly (default: M)",
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Benchmark ticker (default: SPY)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    # Enhanced model options
    enhanced_group = parser.add_argument_group("Enhanced Model Options")
    
    enhanced_group.add_argument(
        "--enhanced-model",
        action="store_true",
        help="Enable enhanced rank-based model with all features",
    )
    
    enhanced_group.add_argument(
        "--model-type",
        type=str,
        choices=["regression", "rank_pairwise", "rank_ndcg"],
        default="regression",
        help="Model type (default: regression)",
    )
    
    enhanced_group.add_argument(
        "--target-mode",
        type=str,
        choices=["vol_norm", "raw", "rank", "ortho"],
        default="vol_norm",
        help="Target engineering mode (default: vol_norm)",
    )
    
    enhanced_group.add_argument(
        "--no-short-reversal",
        action="store_true",
        help="Disable short-term reversal features",
    )
    
    enhanced_group.add_argument(
        "--no-residual-momentum",
        action="store_true",
        help="Disable residual momentum features",
    )
    
    enhanced_group.add_argument(
        "--no-idio-vol",
        action="store_true",
        help="Disable idiosyncratic volatility features",
    )
    
    enhanced_group.add_argument(
        "--no-sector-ohe",
        action="store_true",
        help="Disable sector one-hot encoding",
    )
    
    enhanced_group.add_argument(
        "--no-cross-normalize",
        action="store_true",
        help="Disable cross-sectional normalization",
    )
    
    # Enhanced portfolio options
    portfolio_group = parser.add_argument_group("Enhanced Portfolio Options")
    
    portfolio_group.add_argument(
        "--weighting",
        type=str,
        choices=["equal", "rank", "score"],
        default="rank",
        help="Portfolio weighting scheme (default: rank)",
    )
    
    portfolio_group.add_argument(
        "--rank-tau",
        type=float,
        default=5.0,
        help="Temperature for rank weighting (default: 5.0)",
    )
    
    portfolio_group.add_argument(
        "--neutralization",
        type=str,
        choices=["none", "vol", "beta", "vol_beta"],
        default="none",
        help="Risk neutralization method (default: none)",
    )
    
    portfolio_group.add_argument(
        "--max-position",
        type=float,
        default=0.10,
        help="Maximum position weight (default: 0.10)",
    )
    
    portfolio_group.add_argument(
        "--sector-neutral",
        action="store_true",
        help="Enable sector-neutral ranking (rank within sectors first, then select)",
    )
    
    # Rolling window training arguments
    portfolio_group.add_argument(
        "--rolling-window",
        action="store_true",
        help="Use rolling window for training instead of expanding window. "
             "Rolling windows adapt to regime changes faster but have less training data.",
    )
    
    portfolio_group.add_argument(
        "--window-size",
        type=int,
        default=504,
        help="Size of rolling window in trading days (default: 504 = ~2 years). "
             "Only used when --rolling-window is enabled.",
    )
    
    portfolio_group.add_argument(
        "--robust-objective",
        action="store_true",
        help="Use robust regression objective (pseudohuber) instead of squared error. "
             "This is more robust to outliers in the target variable.",
    )
    
    portfolio_group.add_argument(
        "--shap-selection",
        action="store_true",
        help="Use SHAP values to select top features, reducing overfitting.",
    )
    
    portfolio_group.add_argument(
        "--shap-top-k",
        type=int,
        default=15,
        help="Number of top features to keep when using SHAP selection (default: 15).",
    )
    
    # XGBoost hyperparameter arguments
    xgb_group = parser.add_argument_group("XGBoost Hyperparameters")
    
    xgb_group.add_argument(
        "--xgb-max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (default: 4)",
    )
    
    xgb_group.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=None,
        help="Learning rate / step size (default: 0.05)",
    )
    
    xgb_group.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=None,
        help="Number of boosting rounds (default: 300)",
    )
    
    xgb_group.add_argument(
        "--xgb-subsample",
        type=float,
        default=None,
        help="Subsample ratio of training instances (default: 0.8)",
    )
    
    xgb_group.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=None,
        help="Subsample ratio of columns for each tree (default: 0.8)",
    )
    
    xgb_group.add_argument(
        "--xgb-reg-lambda",
        type=float,
        default=None,
        help="L2 regularization term (default: 2.0)",
    )
    
    xgb_group.add_argument(
        "--xgb-reg-alpha",
        type=float,
        default=None,
        help="L1 regularization term (default: 0.1)",
    )

    return parser.parse_args()


def print_banner() -> None:
    """Print application banner."""
    banner = """
+-----------------------------------------------------------+
|           AUTO-RESEARCHER FOR ASSET SELECTION             |
|       ML-based Stock Ranking & Portfolio Construction     |
+-----------------------------------------------------------+
    """
    print(banner)


def print_config(config: PipelineConfig, tickers: list[str]) -> None:
    """Print configuration summary."""
    print("\n[CONFIG] CONFIGURATION")
    print("-" * 40)
    print(f"  Universe: {len(tickers)} tickers")
    print(f"  Date Range: {config.research.start_date} to {config.research.end_date}")
    print(f"  Horizon: {config.research.horizon_days} days")
    print(f"  Top-K: {config.research.top_k}")
    print(f"  Rebalance: {'Monthly' if config.research.rebalance_frequency == 'M' else 'Weekly'}")
    print(f"  Benchmark: {config.research.benchmark}")
    print("-" * 40)


def print_results(result: BacktestResult) -> None:
    """Print backtest results."""
    print("\n[RESULTS] BACKTEST RESULTS")
    print("=" * 50)
    
    metrics = result.metrics
    
    # Performance metrics
    print("\n[PERF] Performance Metrics")
    print("-" * 40)
    print(f"  Total Return:          {metrics.get('total_return', 0) * 100:>8.2f}%")
    print(f"  Annualized Return:     {metrics.get('annualized_return', 0) * 100:>8.2f}%")
    print(f"  Annualized Volatility: {metrics.get('annualized_volatility', 0) * 100:>8.2f}%")
    print(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>8.2f}")
    print(f"  Sortino Ratio:         {metrics.get('sortino_ratio', 0):>8.2f}")
    print(f"  Max Drawdown:          {metrics.get('max_drawdown', 0) * 100:>8.2f}%")
    
    # Benchmark comparison
    if "benchmark_return" in metrics:
        print("\n[BENCH] Benchmark Comparison")
        print("-" * 40)
        print(f"  Benchmark Return:      {metrics.get('benchmark_return', 0) * 100:>8.2f}%")
        print(f"  Excess Return:         {metrics.get('excess_return', 0) * 100:>8.2f}%")
        print(f"  Hit Rate:              {metrics.get('hit_rate', 0) * 100:>8.2f}%")
        print(f"  Information Ratio:     {metrics.get('information_ratio', 0):>8.2f}")
    
    # Model performance
    if "average_ic" in metrics:
        print("\n[MODEL] Model Performance")
        print("-" * 40)
        target_type = "regression (vol-normalized)" if metrics.get("target_type") == "vol_norm" else "ranking"
        print(f"  Target type:           {target_type}")
        print(f"  Average IC:            {metrics.get('average_ic', 0):>8.4f}")
        if "ic_std" in metrics:
            print(f"  IC Std:                {metrics.get('ic_std', 0):>8.4f}")
        if "ic_ir" in metrics:
            print(f"  IC IR:                 {metrics.get('ic_ir', 0):>8.4f}")
    
    # Summary
    print("\n[SUMMARY] Backtest Summary")
    print("-" * 40)
    print(f"  Number of Rebalances: {len(result.weights_history)}")
    print(f"  Trading Periods: {len(result.portfolio_returns)}")
    
    if len(result.portfolio_returns) > 0:
        print(f"  First Date: {result.portfolio_returns.index.min().strftime('%Y-%m-%d')}")
        print(f"  Last Date: {result.portfolio_returns.index.max().strftime('%Y-%m-%d')}")
    
    print("=" * 50)


def print_sample_portfolio(result: BacktestResult) -> None:
    """Print sample portfolio from last rebalance."""
    if not result.weights_history:
        return
    
    last_date, last_weights = result.weights_history[-1]
    
    print("\n[PORTFOLIO] SAMPLE PORTFOLIO (Last Rebalance)")
    print("-" * 40)
    print(f"  Date: {last_date.strftime('%Y-%m-%d')}")
    print("\n  Holdings:")
    
    for ticker, weight in sorted(last_weights.items(), key=lambda x: -x[1]):
        print(f"    {ticker:<6}  {weight * 100:>6.2f}%")


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    # Determine universe
    if args.tickers:
        tickers = args.tickers
    elif args.sp500:
        tickers = get_sp500_universe()
    else:
        # Default: tech stocks + benchmark
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "SPY"]
    
    # Ensure benchmark is in tickers
    if args.benchmark not in tickers:
        tickers.append(args.benchmark)
    
    # End date
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    # Create configuration
    # Build enhanced model config if enabled
    if args.enhanced_model:
        enhanced_model_config = EnhancedModelConfig(
            enabled=True,
            model_type=args.model_type,
            target_mode=args.target_mode,
            use_short_reversal=not args.no_short_reversal,
            use_residual_momentum=not args.no_residual_momentum,
            use_idio_vol=not args.no_idio_vol,
            use_mad=True,
            use_sector_ohe=not args.no_sector_ohe,
            cross_sectional_normalize=not args.no_cross_normalize,
            rolling_window=args.rolling_window,
            rolling_window_days=args.window_size,
            robust_objective=args.robust_objective,
            use_shap_selection=args.shap_selection,
            shap_top_k_features=args.shap_top_k,
            # XGBoost hyperparameters
            xgb_max_depth=args.xgb_max_depth,
            xgb_learning_rate=args.xgb_learning_rate,
            xgb_n_estimators=args.xgb_n_estimators,
            xgb_subsample=args.xgb_subsample,
            xgb_colsample_bytree=args.xgb_colsample_bytree,
            xgb_reg_lambda=args.xgb_reg_lambda,
            xgb_reg_alpha=args.xgb_reg_alpha,
        )
        enhanced_portfolio_config = EnhancedPortfolioConfig(
            top_k=args.top_k,
            weighting_scheme=args.weighting,
            rank_tau=args.rank_tau,
            neutralization=args.neutralization,
            max_position_weight=args.max_position,
            sector_neutral_ranking=args.sector_neutral,
        )
        print("\n[ENHANCED] Enhanced Model Mode Enabled")
        print(f"  Model Type:   {args.model_type}")
        target_desc = "vol_normalized_forward_return" if args.target_mode == "vol_norm" else args.target_mode
        print(f"  Target Mode:  {target_desc}")
        print(f"  Weighting:    {args.weighting}")
        if args.sector_neutral:
            print(f"  Sector-Neutral: Yes (rank within sectors first)")
        if args.rolling_window:
            print(f"  Rolling Window: Yes ({args.window_size} trading days = ~{args.window_size // 252:.1f} years)")
        if args.robust_objective:
            print(f"  Robust Objective: Yes (pseudohuber loss instead of squared error)")
        if args.shap_selection:
            print(f"  SHAP Selection: Yes (top {args.shap_top_k} features)")
    else:
        enhanced_model_config = EnhancedModelConfig(enabled=False)
        enhanced_portfolio_config = EnhancedPortfolioConfig()
    
    config = PipelineConfig(
        research=ResearchConfig(
            universe=tuple(tickers),
            start_date=args.start_date,
            end_date=end_date,
            horizon_days=args.horizon,
            top_k=args.top_k,
            rebalance_frequency=args.rebalance,
            benchmark=args.benchmark,
            lookback_days=252,
        ),
        features=FeatureConfig(
            momentum_windows=(21, 63, 126),
            volatility_windows=(21, 63),
        ),
        model=ModelConfig(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            early_stopping_rounds=None,
        ),
        enhanced_model=enhanced_model_config,
        enhanced_portfolio=enhanced_portfolio_config,
    )
    
    print_config(config, tickers)
    
    # Download price data
    print("\n[DATA] Downloading price data...")
    try:
        prices_raw = download_price_history(
            tickers=tickers,
            start=args.start_date,
            end=end_date,
        )
        prices = get_adjusted_close(prices_raw)
        print(f"  [OK] Downloaded {len(prices)} days of data for {len(prices.columns)} tickers")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        print(f"  [ERR] Error: {e}")
        return 1
    
    # Validate data
    missing = prices.isna().sum()
    if missing.any():
        print(f"  [WARN] Some tickers have missing data:")
        for ticker, count in missing[missing > 0].items():
            print(f"    {ticker}: {count} missing days")
    
    # Run backtest
    print("\n[RUN] Running backtest...")
    try:
        if args.enhanced_model:
            # Use enhanced backtest with XGBoost ranking and enhanced features
            result = run_enhanced_backtest(
                prices=prices,
                horizon_days=args.horizon,
                top_k=args.top_k,
                benchmark=args.benchmark,
                rebalance_frequency=args.rebalance,
                min_train_periods=252,
                config=config,
            )
        else:
            # Use standard backtest
            result = run_backtest(
                prices=prices,
                horizon_days=args.horizon,
                top_k=args.top_k,
                benchmark=args.benchmark,
                rebalance_frequency=args.rebalance,
                min_train_periods=252,
                config=config,
            )
        print("  [OK] Backtest completed successfully")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"  [ERR] Error: {e}")
        return 1
    
    # Print results
    print_results(result)
    print_sample_portfolio(result)
    
    print("\n[DONE] Done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
