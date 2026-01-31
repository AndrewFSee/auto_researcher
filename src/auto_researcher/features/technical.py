"""
Technical feature engineering.

This module provides functions to compute technical indicators like
momentum, volatility, and other price-based features.
"""

import pandas as pd
import numpy as np


def compute_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute daily returns from price data.

    Args:
        prices: Price data with DatetimeIndex. Can be Series (single ticker)
                or DataFrame (multiple tickers as columns).

    Returns:
        Daily returns with same structure as input.

    Examples:
        >>> prices = pd.Series([100, 102, 101], index=pd.date_range('2023-01-01', periods=3))
        >>> returns = compute_returns(prices)
        >>> len(returns) == 3
        True
    """
    return prices.pct_change()


def compute_log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute log returns from price data.

    Log returns are more suitable for longer horizons and statistical analysis.

    Args:
        prices: Price data with DatetimeIndex.

    Returns:
        Log returns with same structure as input.
    """
    return np.log(prices / prices.shift(1))


def compute_momentum(
    returns: pd.DataFrame,
    window: int,
    skip_recent: int = 1,
) -> pd.DataFrame:
    """
    Compute momentum as cumulative return over a lookback window.

    Momentum is computed as the total return over the window, optionally
    skipping the most recent days to avoid short-term reversal.

    Args:
        returns: Daily returns DataFrame with tickers as columns.
        window: Lookback window in trading days.
        skip_recent: Number of recent days to skip (default 1 for reversal).

    Returns:
        Momentum scores with same structure as input.

    Examples:
        >>> import numpy as np
        >>> dates = pd.date_range('2023-01-01', periods=100)
        >>> returns = pd.DataFrame(np.random.randn(100, 3) * 0.02, index=dates, columns=['A', 'B', 'C'])
        >>> mom = compute_momentum(returns, window=21)
        >>> mom.shape == returns.shape
        True
    """
    # Compute cumulative return over window
    # Skip most recent 'skip_recent' days to avoid reversal effect
    if skip_recent > 0:
        shifted_cumret = (1 + returns).rolling(window=window).apply(
            lambda x: x[:-skip_recent].prod() - 1 if len(x) > skip_recent else np.nan,
            raw=False,
        )
    else:
        shifted_cumret = (1 + returns).rolling(window=window).apply(
            lambda x: x.prod() - 1,
            raw=False,
        )

    return shifted_cumret


def compute_momentum_simple(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute simple momentum as rolling sum of returns.

    This is a faster approximation suitable for short windows.

    Args:
        returns: Daily returns DataFrame.
        window: Lookback window in trading days.

    Returns:
        Rolling sum of returns.
    """
    return returns.rolling(window=window).sum()


def compute_volatility(
    returns: pd.DataFrame,
    window: int,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling volatility of returns.

    Args:
        returns: Daily returns DataFrame with tickers as columns.
        window: Lookback window in trading days.
        annualize: If True, annualize volatility (multiply by sqrt(252)).

    Returns:
        Volatility with same structure as input.

    Examples:
        >>> dates = pd.date_range('2023-01-01', periods=100)
        >>> returns = pd.DataFrame(np.random.randn(100, 3) * 0.02, index=dates, columns=['A', 'B', 'C'])
        >>> vol = compute_volatility(returns, window=21)
        >>> vol.shape == returns.shape
        True
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def compute_sharpe(
    returns: pd.DataFrame,
    window: int,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe ratio.

    Args:
        returns: Daily returns DataFrame.
        window: Lookback window in trading days.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Rolling Sharpe ratio.
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    mean_return = excess_returns.rolling(window=window).mean() * 252
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    
    return mean_return / volatility.replace(0, np.nan)


def compute_max_drawdown(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling maximum drawdown.

    Args:
        prices: Price DataFrame with tickers as columns.
        window: Lookback window in trading days.

    Returns:
        Rolling maximum drawdown (negative values).
    """
    rolling_max = prices.rolling(window=window, min_periods=1).max()
    drawdown = prices / rolling_max - 1
    return drawdown.rolling(window=window).min()


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    Args:
        prices: Price DataFrame with tickers as columns.
        window: RSI period (default 14).

    Returns:
        RSI values between 0 and 100.
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_moving_average_ratio(
    prices: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """
    Compute ratio of short to long moving average.

    This captures trend strength - values > 1 indicate uptrend.

    Args:
        prices: Price DataFrame.
        short_window: Short MA window.
        long_window: Long MA window.

    Returns:
        Ratio of short to long moving average.
    """
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    
    return short_ma / long_ma


def compute_all_technical_features(
    prices: pd.DataFrame,
    momentum_windows: tuple[int, ...] = (21, 63, 126, 252),
    volatility_windows: tuple[int, ...] = (21, 63),
) -> pd.DataFrame:
    """
    Compute all technical features for a price DataFrame.

    Args:
        prices: Price DataFrame with tickers as columns and DatetimeIndex.
        momentum_windows: Windows for momentum calculations.
        volatility_windows: Windows for volatility calculations.

    Returns:
        DataFrame with MultiIndex columns (ticker, feature_name).
    """
    returns = compute_returns(prices)
    
    features = {}
    
    # Momentum features
    for window in momentum_windows:
        mom = compute_momentum_simple(returns, window)
        for ticker in mom.columns:
            features[(ticker, f"mom_{window}")] = mom[ticker]
    
    # Volatility features
    for window in volatility_windows:
        vol = compute_volatility(returns, window)
        for ticker in vol.columns:
            features[(ticker, f"vol_{window}")] = vol[ticker]
    
    # RSI
    rsi = compute_rsi(prices)
    for ticker in rsi.columns:
        features[(ticker, "rsi_14")] = rsi[ticker]
    
    # MA ratios
    ma_ratio = compute_moving_average_ratio(prices, 20, 50)
    for ticker in ma_ratio.columns:
        features[(ticker, "ma_ratio_20_50")] = ma_ratio[ticker]
    
    # Combine into DataFrame
    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])
    
    return result
