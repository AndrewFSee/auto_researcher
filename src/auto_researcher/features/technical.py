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


def compute_short_term_reversal(
    returns: pd.DataFrame,
    windows: tuple[int, ...] = (1, 3, 5),
) -> dict[tuple[str, str], pd.Series]:
    """
    Compute short-term reversal signals.

    Short-term (1-5 day) returns tend to reverse, especially in liquid stocks.
    Negative past returns predict positive future returns at very short horizons.

    Args:
        returns: Daily returns DataFrame with tickers as columns.
        windows: Lookback windows for reversal signal.

    Returns:
        Dictionary of (ticker, feature_name) -> Series.
    """
    features = {}
    for window in windows:
        rev = -returns.rolling(window=window).sum()
        for ticker in rev.columns:
            features[(ticker, f"reversal_{window}d")] = rev[ticker]
    return features


def compute_abnormal_volume(
    volume: pd.DataFrame,
    window: int = 20,
) -> dict[tuple[str, str], pd.Series]:
    """
    Compute abnormal volume as ratio to rolling average.

    Abnormal volume often precedes or accompanies price moves.
    Values > 1 indicate above-average trading activity.

    Args:
        volume: Volume DataFrame with tickers as columns.
        window: Lookback window for average volume (default 20 days).

    Returns:
        Dictionary of (ticker, feature_name) -> Series.
    """
    features = {}
    avg_vol = volume.rolling(window=window, min_periods=max(1, window // 2)).mean()
    ratio = volume / avg_vol.replace(0, np.nan)

    for ticker in volume.columns:
        features[(ticker, "abnormal_volume")] = ratio[ticker]
        # Log volume ratio is more normally distributed
        features[(ticker, "log_abnormal_volume")] = np.log1p(ratio[ticker].clip(lower=0))
    return features


def compute_earnings_revision_signal(
    tickers: list[str],
) -> dict[tuple[str, str], float]:
    """
    Compute earnings revision momentum from yfinance analyst estimates.

    Earnings revisions (changes in consensus EPS estimates) are one of the
    strongest known alpha signals. Positive revisions predict positive returns.

    Features:
    - earnings_growth_0q: Current quarter expected EPS growth vs year-ago
    - earnings_surprise_avg: Average earnings surprise over recent quarters
    - earnings_revision_momentum: Current quarter growth minus next year growth
      (positive = accelerating expectations)

    Args:
        tickers: List of ticker symbols.

    Returns:
        Dictionary of (ticker, feature) -> scalar value.
    """
    import yfinance as yf

    features = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)

            # Current quarter estimate growth
            est = t.earnings_estimate
            if est is not None and not est.empty and "growth" in est.columns:
                # Current quarter growth (0q)
                if "0q" in est.index:
                    growth_0q = est.loc["0q", "growth"]
                    if pd.notna(growth_0q):
                        features[(ticker, "earnings_growth_0q")] = float(growth_0q)

                # Revision momentum: current quarter vs next year (acceleration)
                if "0q" in est.index and "+1y" in est.index:
                    g0 = est.loc["0q", "growth"]
                    g1y = est.loc["+1y", "growth"]
                    if pd.notna(g0) and pd.notna(g1y):
                        features[(ticker, "earnings_revision_momentum")] = float(g0 - g1y)

            # Historical surprise track record
            hist = t.earnings_history
            if hist is not None and not hist.empty and "surprisePercent" in hist.columns:
                surprises = hist["surprisePercent"].dropna()
                if len(surprises) >= 2:
                    features[(ticker, "earnings_surprise_avg")] = float(surprises.mean())
                    # Trend of surprises (improving or deteriorating beat record)
                    features[(ticker, "earnings_surprise_trend")] = float(
                        surprises.iloc[-1] - surprises.iloc[0]
                    )

        except (ValueError, TypeError, KeyError, IndexError):
            continue

    return features


def compute_analyst_momentum(
    tickers: list[str],
) -> dict[tuple[str, str], float]:
    """
    Compute analyst rating momentum from yfinance recommendations.

    Converts recommendation counts into a consensus score (-1 to +1) and
    computes the change vs prior month. Positive momentum = upgrades.

    Score = (strongBuy*2 + buy*1 + hold*0 + sell*-1 + strongSell*-2) / total

    Args:
        tickers: List of ticker symbols.

    Returns:
        Dictionary of (ticker, feature) -> scalar value (point-in-time snapshot).
    """
    import yfinance as yf

    features = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            recs = t.recommendations
            if recs is None or recs.empty or len(recs) < 2:
                continue

            def _consensus(row):
                total = row["strongBuy"] + row["buy"] + row["hold"] + row["sell"] + row["strongSell"]
                if total == 0:
                    return np.nan
                score = (
                    row["strongBuy"] * 2 + row["buy"] * 1
                    + row["sell"] * -1 + row["strongSell"] * -2
                ) / total
                return score

            current = _consensus(recs.iloc[0])
            previous = _consensus(recs.iloc[1])

            if pd.notna(current):
                features[(ticker, "analyst_consensus")] = current
            if pd.notna(current) and pd.notna(previous):
                features[(ticker, "analyst_momentum")] = current - previous
                # Coverage ratio: fraction of analysts with strong opinion
                row = recs.iloc[0]
                total = row["strongBuy"] + row["buy"] + row["hold"] + row["sell"] + row["strongSell"]
                if total > 0:
                    features[(ticker, "analyst_conviction")] = (
                        row["strongBuy"] + row["strongSell"]
                    ) / total
        except (ValueError, TypeError, KeyError, IndexError):
            continue

    return features


def compute_all_technical_features(
    prices: pd.DataFrame,
    momentum_windows: tuple[int, ...] = (21, 63, 126, 252),
    volatility_windows: tuple[int, ...] = (21, 63),
    volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute all technical features for a price DataFrame.

    Args:
        prices: Price DataFrame with tickers as columns and DatetimeIndex.
        momentum_windows: Windows for momentum calculations.
        volatility_windows: Windows for volatility calculations.
        volume: Optional volume DataFrame with same structure as prices.

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

    # Short-term reversal (1, 3, 5 day)
    features.update(compute_short_term_reversal(returns))

    # Abnormal volume
    if volume is not None:
        # Align volume columns to price columns
        common_tickers = prices.columns.intersection(volume.columns)
        if len(common_tickers) > 0:
            features.update(compute_abnormal_volume(volume[common_tickers]))

    # Combine into DataFrame
    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])

    return result
