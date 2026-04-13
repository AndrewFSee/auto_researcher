"""
Target engineering for cross-sectional stock ranking models.

This module provides functions for creating targets for ML models:
- Raw returns (current behavior)
- Rank-based targets (more stable across regimes)
- Orthogonalized (residual) targets (remove market/sector effects)
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TargetConfig:
    """
    Configuration for target engineering.
    
    Attributes:
        mode: Target transformation mode.
            - "vol_norm": Volatility-normalized forward returns (default).
            - "raw": Use raw forward returns (current behavior).
            - "rank": Use cross-sectional rank of forward returns.
            - "ortho": Use orthogonalized (residual) forward returns.
        rank_normalize: If mode="rank", how to normalize ranks.
            - "percentile": Ranks in [0, 1] range.
            - "zscore": Z-score of ranks.
        ortho_remove_market: If mode="ortho", remove market component.
        ortho_remove_sector: If mode="ortho", remove sector means.
        horizon_days: Forward return horizon in trading days.
        vol_lookback: Lookback window for volatility calculation in vol_norm mode.
    """
    mode: Literal["vol_norm", "raw", "rank", "ortho"] = "vol_norm"
    rank_normalize: Literal["percentile", "zscore"] = "percentile"
    ortho_remove_market: bool = True
    ortho_remove_sector: bool = True
    horizon_days: int = 21
    vol_lookback: int | None = None  # If None, uses horizon_days


# Default configurations
DEFAULT_TARGET_CONFIG = TargetConfig()

RAW_TARGET_CONFIG = TargetConfig(mode="raw")
RANK_TARGET_CONFIG = TargetConfig(mode="rank", rank_normalize="percentile")
ORTHO_TARGET_CONFIG = TargetConfig(mode="ortho", ortho_remove_market=True, ortho_remove_sector=True)
VOL_NORM_TARGET_CONFIG = TargetConfig(mode="vol_norm")


# =============================================================================
# VOLATILITY-NORMALIZED RETURNS TARGET
# =============================================================================

def compute_realized_volatility(
    prices: pd.DataFrame,
    window: int,
    annualize: bool = False,
) -> pd.DataFrame:
    """
    Compute realized volatility from price data.
    
    Uses rolling standard deviation of daily log returns.
    
    Args:
        prices: Price DataFrame with tickers as columns.
        window: Rolling window size in trading days.
        annualize: If True, annualize the volatility (multiply by sqrt(252)).
    
    Returns:
        DataFrame of realized volatility, same shape as input.
    """
    # Compute daily log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # Rolling standard deviation
    vol = log_returns.rolling(window=window, min_periods=max(1, window // 2)).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def validate_forward_returns(
    forward_returns: pd.DataFrame,
    max_return: float = 5.0,
    min_return: float = -0.90,
    min_coverage: float = 0.1,
) -> pd.DataFrame:
    """
    Validate forward returns and flag suspicious values.

    Detects likely stock splits or data errors (extreme returns) and
    warns about tickers with very low data coverage.

    Args:
        forward_returns: DataFrame of forward returns.
        max_return: Maximum plausible return (default +500%). Returns above
            this are set to NaN (likely stock split or data error).
        min_return: Minimum plausible return (default -90%). Returns below
            this are set to NaN.
        min_coverage: Minimum fraction of non-NaN values per ticker.
            Tickers below this threshold trigger a warning.

    Returns:
        Cleaned DataFrame with extreme returns set to NaN.
    """
    result = forward_returns.copy()

    # Flag extreme returns (likely stock splits or data errors)
    extreme_mask = (result > max_return) | (result < min_return)
    n_extreme = extreme_mask.sum().sum()
    if n_extreme > 0:
        extreme_tickers = extreme_mask.any(axis=0)
        affected = list(extreme_tickers[extreme_tickers].index)
        logger.warning(
            f"Found {n_extreme} extreme forward returns (>{max_return:.0%} or "
            f"<{min_return:.0%}) in {len(affected)} tickers: {affected[:10]}. "
            f"Setting to NaN (likely stock splits or data errors)."
        )
        result[extreme_mask] = np.nan

    # Warn about low coverage tickers
    for col in result.columns:
        coverage = result[col].notna().mean()
        if coverage < min_coverage:
            logger.warning(
                f"Ticker {col} has very low forward return coverage "
                f"({coverage:.1%} non-NaN). Horizon may exceed available data."
            )

    return result


def compute_vol_normalized_targets(
    prices: pd.DataFrame,
    horizon_days: int,
    vol_lookback: int | None = None,
    benchmark: str | None = None,
    epsilon: float = 1e-8,
    winsorize_mode: Literal["expanding", "full"] = "expanding",
) -> pd.DataFrame:
    """
    Compute volatility-normalized forward returns.

    Target = forward_return / realized_volatility

    This approximates an "IR-like" return per unit of volatility, reducing
    the dominance of high-volatility names in the target.

    Args:
        prices: Price DataFrame with tickers as columns.
        horizon_days: Forward return horizon in trading days.
        vol_lookback: Lookback window for volatility. If None, uses horizon_days.
        benchmark: Benchmark ticker to exclude from returns (optional).
        epsilon: Small value to prevent division by zero.
        winsorize_mode: How to compute winsorization bounds.
            - "expanding": Use expanding-window stats (only past data at each point).
              Prevents look-ahead bias in target construction.
            - "full": Use full-sample stats (legacy behavior, has look-ahead bias).

    Returns:
        DataFrame of volatility-normalized targets.
    """
    if vol_lookback is None:
        vol_lookback = horizon_days

    # Get stock prices (exclude benchmark)
    if benchmark and benchmark in prices.columns:
        stock_prices = prices.drop(columns=[benchmark])
    else:
        stock_prices = prices

    # Compute forward returns: (price_{t+h} / price_t) - 1
    forward_returns = stock_prices.pct_change(periods=horizon_days).shift(-horizon_days)

    # Validate forward returns (flag stock splits, extreme outliers)
    forward_returns = validate_forward_returns(forward_returns)

    # Compute realized volatility at time t (uses past data)
    realized_vol = compute_realized_volatility(stock_prices, window=vol_lookback, annualize=False)

    # Scale volatility to horizon (approximate: multiply by sqrt(horizon_days))
    horizon_vol = realized_vol * np.sqrt(horizon_days)

    # Compute vol-normalized target
    vol_norm_target = forward_returns / (horizon_vol + epsilon)

    # Clip extreme values for stability (winsorize at 3 std)
    if winsorize_mode == "expanding":
        # Expanding-window: only use data up to each point in time
        # Prevents look-ahead bias (future data cannot influence past bounds)
        for col in vol_norm_target.columns:
            series = vol_norm_target[col]
            expanding_mean = series.expanding(min_periods=10).mean()
            expanding_std = series.expanding(min_periods=10).std()
            valid_mask = expanding_std > 0
            lower = expanding_mean - 3 * expanding_std
            upper = expanding_mean + 3 * expanding_std
            clipped = series.copy()
            mask = valid_mask & series.notna()
            clipped[mask] = series[mask].clip(
                lower=lower[mask], upper=upper[mask]
            )
            vol_norm_target[col] = clipped
    elif winsorize_mode == "full":
        # Legacy behavior: full-sample stats (has look-ahead bias)
        logger.warning(
            "Using winsorize_mode='full' -- this has look-ahead bias. "
            "Use 'expanding' for unbiased targets."
        )
        for col in vol_norm_target.columns:
            valid = vol_norm_target[col].dropna()
            if len(valid) > 10:
                mean_val = valid.mean()
                std_val = valid.std()
                if std_val > 0:
                    lower = mean_val - 3 * std_val
                    upper = mean_val + 3 * std_val
                    vol_norm_target[col] = vol_norm_target[col].clip(lower=lower, upper=upper)
    else:
        raise ValueError(f"Unknown winsorize_mode: {winsorize_mode}")

    logger.info(
        f"Computed vol-normalized targets with horizon={horizon_days}d, "
        f"vol_lookback={vol_lookback}d, winsorize_mode={winsorize_mode}"
    )

    return vol_norm_target


# =============================================================================
# RAW RETURNS TARGET
# =============================================================================

def compute_forward_returns(
    prices: pd.DataFrame,
    horizon_days: int,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """
    Compute forward returns for each stock.
    
    Args:
        prices: Price DataFrame with tickers as columns.
        horizon_days: Number of trading days forward.
        benchmark: Benchmark ticker to exclude from returns (optional).
    
    Returns:
        DataFrame of forward returns, same shape as input (shifted).
    """
    if benchmark and benchmark in prices.columns:
        stock_prices = prices.drop(columns=[benchmark])
    else:
        stock_prices = prices
    
    # Forward return: (price_{t+h} / price_t) - 1
    forward_ret = stock_prices.pct_change(periods=horizon_days).shift(-horizon_days)

    # Validate forward returns (flag stock splits, extreme outliers)
    forward_ret = validate_forward_returns(forward_ret)

    return forward_ret


# =============================================================================
# RANK-BASED TARGETS
# =============================================================================

def compute_rank_targets(
    forward_returns: pd.DataFrame,
    normalize: Literal["percentile", "zscore"] = "percentile",
) -> pd.DataFrame:
    """
    Convert forward returns to cross-sectional ranks.
    
    For each date, rank stocks by forward return and normalize.
    
    Args:
        forward_returns: DataFrame of forward returns.
        normalize: How to normalize ranks.
            - "percentile": Ranks in [0, 1] range (rank / N).
            - "zscore": Z-score of ranks (mean 0, std 1).
    
    Returns:
        DataFrame of rank-based targets.
    """
    result = pd.DataFrame(index=forward_returns.index, columns=forward_returns.columns, dtype=float)
    
    for date in forward_returns.index:
        row = forward_returns.loc[date]
        valid = row.dropna()
        
        if len(valid) < 2:
            continue
        
        # Rank stocks (higher return = higher rank)
        ranks = valid.rank(ascending=True, method='average')
        n = len(ranks)
        
        if normalize == "percentile":
            # Normalize to [0, 1]: (rank - 1) / (N - 1)
            normalized = (ranks - 1) / (n - 1) if n > 1 else ranks * 0 + 0.5
        elif normalize == "zscore":
            # Z-score of ranks
            normalized = (ranks - ranks.mean()) / ranks.std()
        else:
            raise ValueError(f"Unknown normalize mode: {normalize}")
        
        result.loc[date, normalized.index] = normalized.values
    
    return result


# =============================================================================
# ORTHOGONALIZED (RESIDUAL) TARGETS
# =============================================================================

def compute_ortho_targets(
    forward_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    sector_mapping: dict[str, str] | None = None,
    remove_market: bool = True,
    remove_sector: bool = True,
) -> pd.DataFrame:
    """
    Compute orthogonalized forward returns.
    
    Remove market and/or sector effects to isolate idiosyncratic returns.
    
    Args:
        forward_returns: DataFrame of forward returns.
        benchmark_returns: Forward returns of benchmark (for market removal).
        sector_mapping: Ticker -> Sector mapping (for sector removal).
        remove_market: If True, regress out market returns.
        remove_sector: If True, remove sector means.
    
    Returns:
        DataFrame of orthogonalized returns.
    """
    from auto_researcher.features.enhanced import get_sector_for_ticker, DEFAULT_SECTOR_MAPPING
    
    if sector_mapping is None:
        sector_mapping = DEFAULT_SECTOR_MAPPING
    
    result = forward_returns.copy()
    
    for date in forward_returns.index:
        row = forward_returns.loc[date].copy()
        valid_mask = row.notna()
        
        if valid_mask.sum() < 3:
            continue
        
        valid_returns = row[valid_mask]
        residuals = valid_returns.copy()
        
        # Step 1: Remove market component
        if remove_market and benchmark_returns is not None:
            market_ret = benchmark_returns.get(date, np.nan)
            if pd.notna(market_ret):
                # Simple approach: subtract market return
                # More sophisticated: regress and use residuals
                residuals = residuals - market_ret
        
        # Step 2: Remove sector means
        if remove_sector:
            sectors = {ticker: get_sector_for_ticker(ticker, sector_mapping) 
                      for ticker in residuals.index}
            
            # Group by sector and remove sector mean
            sector_means = {}
            for ticker, ret in residuals.items():
                sector = sectors.get(ticker, "Unknown")
                if sector not in sector_means:
                    sector_means[sector] = []
                sector_means[sector].append(ret)
            
            sector_avg = {s: np.nanmean(rets) for s, rets in sector_means.items()}
            
            for ticker in residuals.index:
                sector = sectors.get(ticker, "Unknown")
                sector_mean = sector_avg.get(sector, 0)
                residuals[ticker] = residuals[ticker] - sector_mean
        
        result.loc[date, residuals.index] = residuals.values
    
    return result


# =============================================================================
# MAIN TARGET BUILDER
# =============================================================================

def build_targets(
    prices: pd.DataFrame,
    config: TargetConfig | None = None,
    benchmark: str = "SPY",
    sector_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build targets according to configuration.
    
    Args:
        prices: Price DataFrame with tickers as columns.
        config: Target configuration.
        benchmark: Benchmark ticker.
        sector_mapping: Ticker -> Sector mapping.
    
    Returns:
        DataFrame of targets, aligned with prices index.
    """
    if config is None:
        config = DEFAULT_TARGET_CONFIG
    
    if config.mode == "vol_norm":
        return compute_vol_normalized_targets(
            prices,
            horizon_days=config.horizon_days,
            vol_lookback=config.vol_lookback,
            benchmark=benchmark,
        )
    
    # Compute raw forward returns for other modes
    forward_returns = compute_forward_returns(
        prices, 
        horizon_days=config.horizon_days,
        benchmark=benchmark,
    )
    
    if config.mode == "raw":
        return forward_returns
    
    elif config.mode == "rank":
        return compute_rank_targets(
            forward_returns,
            normalize=config.rank_normalize,
        )
    
    elif config.mode == "ortho":
        # Get benchmark forward returns
        benchmark_forward = None
        if config.ortho_remove_market and benchmark in prices.columns:
            benchmark_forward = prices[benchmark].pct_change(periods=config.horizon_days).shift(-config.horizon_days)
        
        return compute_ortho_targets(
            forward_returns,
            benchmark_returns=benchmark_forward,
            sector_mapping=sector_mapping,
            remove_market=config.ortho_remove_market,
            remove_sector=config.ortho_remove_sector,
        )
    
    else:
        raise ValueError(f"Unknown target mode: {config.mode}")


def prepare_training_targets(
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    config: TargetConfig | None = None,
    benchmark: str = "SPY",
) -> pd.Series:
    """
    Prepare training targets in flat format for model training.
    
    Args:
        prices: Price DataFrame.
        dates: Training dates.
        tickers: List of tradeable tickers.
        config: Target configuration.
        benchmark: Benchmark ticker.
    
    Returns:
        Series with MultiIndex (date, ticker) and target values.
    """
    targets_df = build_targets(
        prices, 
        config=config, 
        benchmark=benchmark,
    )
    
    # Filter to requested dates and tickers
    targets_df = targets_df.loc[dates, tickers]
    
    # Stack to get flat format
    targets = targets_df.stack()
    targets.index.names = ["date", "ticker"]
    targets.name = "target"
    
    return targets


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_target_mode_description(config: TargetConfig) -> str:
    """Get human-readable description of target mode."""
    if config.mode == "vol_norm":
        vol_window = config.vol_lookback or config.horizon_days
        return f"Vol-normalized forward returns ({config.horizon_days}d horizon, {vol_window}d vol)"
    elif config.mode == "raw":
        return f"Raw forward returns ({config.horizon_days}d)"
    elif config.mode == "rank":
        return f"Rank-based targets ({config.rank_normalize}, {config.horizon_days}d)"
    elif config.mode == "ortho":
        parts = []
        if config.ortho_remove_market:
            parts.append("market-adjusted")
        if config.ortho_remove_sector:
            parts.append("sector-adjusted")
        adj_str = " + ".join(parts) if parts else "no adjustment"
        return f"Orthogonalized returns ({adj_str}, {config.horizon_days}d)"
    else:
        return f"Unknown mode: {config.mode}"
