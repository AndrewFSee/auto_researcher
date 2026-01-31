"""
Market regime signal computation for regime-switching strategies.

This module provides functions to compute a market regime signal based on
volatility and trend conditions. The regime is used to switch between
different model types (e.g., Momentum vs Quality) at each rebalance date.

Regime Rules:
- MOMENTUM: Low volatility AND uptrend (good for momentum strategies)
- QUALITY: High volatility OR downtrend (risk-off, prefer quality/value)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class MarketRegime(str, Enum):
    """Market regime for strategy switching."""
    MOMENTUM = "MOMENTUM"
    QUALITY = "QUALITY"


@dataclass
class RegimeConfig:
    """
    Configuration for regime signal computation.
    
    Attributes:
        vol_window: Rolling window for realized volatility (default: 21 days).
        trend_window: Rolling window for moving average (default: 200 days).
        vol_threshold_method: How to set vol threshold:
            - "median": Use median of historical vol (default)
            - "fixed": Use fixed value from vol_threshold_value
        vol_threshold_value: Fixed volatility threshold (annualized, e.g., 0.15 = 15%).
            Only used if vol_threshold_method="fixed".
        annualize_factor: Factor to annualize daily vol (sqrt(252) ≈ 15.87).
    """
    vol_window: int = 21
    trend_window: int = 200
    vol_threshold_method: Literal["median", "fixed"] = "median"
    vol_threshold_value: float = 0.15  # 15% annualized vol
    annualize_factor: float = np.sqrt(252)


# =============================================================================
# Core Functions
# =============================================================================


def compute_realized_volatility(
    prices: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute rolling realized volatility from price series.
    
    Args:
        prices: Price series (e.g., SPY close prices).
        window: Rolling window in trading days.
        annualize: If True, annualize the volatility (multiply by sqrt(252)).
    
    Returns:
        Series of rolling volatility values.
    """
    returns = prices.pct_change()
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def compute_trend_signal(
    prices: pd.Series,
    window: int = 200,
) -> pd.Series:
    """
    Compute trend signal: whether price is above moving average.
    
    Args:
        prices: Price series.
        window: Moving average window in trading days.
    
    Returns:
        Boolean Series: True if price > MA, False otherwise.
    """
    ma = prices.rolling(window=window).mean()
    trend_up = prices > ma
    return trend_up


def compute_regime_series(
    spy_prices: pd.Series,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """
    Compute market regime series based on volatility and trend.
    
    Regime Rules:
    - MOMENTUM: vol < threshold AND trend_up (low vol, uptrend)
    - QUALITY: vol >= threshold OR NOT trend_up (high vol or downtrend)
    
    Args:
        spy_prices: SPY price series with DatetimeIndex.
        config: Regime configuration. Uses defaults if None.
    
    Returns:
        Series indexed by date with values "MOMENTUM" or "QUALITY".
    """
    if config is None:
        config = RegimeConfig()
    
    # Compute volatility
    vol = compute_realized_volatility(
        spy_prices,
        window=config.vol_window,
        annualize=True,
    )
    
    # Compute trend
    trend_up = compute_trend_signal(
        spy_prices,
        window=config.trend_window,
    )
    
    # Determine volatility threshold
    if config.vol_threshold_method == "median":
        # Use median of entire series (in-sample for backtest)
        vol_threshold = vol.median()
        logger.info(f"Using median vol threshold: {vol_threshold:.2%}")
    else:
        vol_threshold = config.vol_threshold_value
        logger.info(f"Using fixed vol threshold: {vol_threshold:.2%}")
    
    # Low volatility condition
    low_vol = vol < vol_threshold
    
    # Regime assignment
    # MOMENTUM when low vol AND uptrend, else QUALITY
    regime = pd.Series(
        index=spy_prices.index,
        data=MarketRegime.QUALITY.value,  # Default to QUALITY
        dtype=str,
    )
    
    momentum_mask = low_vol & trend_up
    regime[momentum_mask] = MarketRegime.MOMENTUM.value
    
    return regime


def get_regime_for_date(
    regime_series: pd.Series,
    date: pd.Timestamp,
) -> MarketRegime:
    """
    Get the regime for a specific date.
    
    Uses the most recent available regime value (for dates not in the series).
    
    Args:
        regime_series: Series from compute_regime_series().
        date: Target date.
    
    Returns:
        MarketRegime enum value.
    """
    # Find most recent date <= target date
    available_dates = regime_series.index[regime_series.index <= date]
    
    if len(available_dates) == 0:
        # Default to QUALITY if no historical data
        logger.warning(f"No regime data before {date}, defaulting to QUALITY")
        return MarketRegime.QUALITY
    
    latest_date = available_dates[-1]
    regime_str = regime_series[latest_date]
    
    return MarketRegime(regime_str)


def compute_regime_summary(
    regime_series: pd.Series,
    rebal_dates: list[pd.Timestamp] | None = None,
) -> dict[str, int]:
    """
    Compute summary statistics for regime series.
    
    Args:
        regime_series: Series from compute_regime_series().
        rebal_dates: If provided, count regimes only at these dates.
    
    Returns:
        Dict with counts: {"MOMENTUM": n, "QUALITY": m}
    """
    if rebal_dates is not None:
        # Count at rebalance dates only
        regimes_at_rebal = []
        for d in rebal_dates:
            regime = get_regime_for_date(regime_series, d)
            regimes_at_rebal.append(regime.value)
        counts = pd.Series(regimes_at_rebal).value_counts().to_dict()
    else:
        counts = regime_series.value_counts().to_dict()
    
    # Ensure both keys exist
    result = {
        MarketRegime.MOMENTUM.value: counts.get(MarketRegime.MOMENTUM.value, 0),
        MarketRegime.QUALITY.value: counts.get(MarketRegime.QUALITY.value, 0),
    }
    
    return result
