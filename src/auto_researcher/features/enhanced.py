"""
Enhanced technical features for cross-sectional ranking models.

This module provides additional features for more robust stock ranking:
- Short-term reversal (5d, 10d momentum)
- Beta-adjusted (residual) momentum
- Idiosyncratic volatility
- Robust dispersion metrics (MAD)
- Cross-sectional normalization
- Sector encoding (one-hot)
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EnhancedFeatureConfig:
    """
    Configuration for enhanced feature engineering.
    
    Attributes:
        use_short_reversal: Include 5d and 10d momentum features.
        use_residual_mom: Include beta-adjusted residual momentum.
        use_idio_vol: Include idiosyncratic volatility features.
        use_mad_metrics: Include median absolute deviation metrics.
        use_trend_health: Include trend health (MA ratio) feature.
        use_sector_ohe: Include one-hot encoded sector features.
        use_cross_sec_norm: Apply cross-sectional normalization to features.
        cross_sec_norm_by_sector: Normalize within sector (requires sector data).
        cross_sec_norm_robust: Use robust normalization (median/MAD vs mean/std).
        
        beta_window: Rolling window for beta estimation (days).
        idio_vol_windows: Windows for idiosyncratic volatility.
        residual_mom_windows: Windows for residual momentum.
        mad_window: Window for MAD computation.
    """
    # Feature toggles
    use_short_reversal: bool = True
    use_residual_mom: bool = True
    use_idio_vol: bool = True
    use_mad_metrics: bool = True
    use_sector_ohe: bool = True
    use_cross_sec_norm: bool = True
    cross_sec_norm_by_sector: bool = False
    cross_sec_norm_robust: bool = True
    
    # Regime-aware scaling (dampen short-term reversal in downtrends)
    use_regime_scaling: bool = True
    regime_mom_feature: str = "tech_resid_mom_252"
    regime_threshold: float = 0.0
    regime_short_reversal_scale: float = 0.3
    regime_short_reversal_features: tuple[str, ...] = ("tech_mom_5d", "tech_mom_10d")
    
    # Window parameters
    beta_window: int = 60
    idio_vol_windows: tuple[int, ...] = (21, 63)
    residual_mom_windows: tuple[int, ...] = (63, 126, 252)
    mad_window: int = 63
    trend_health_windows: tuple[int, int] = (63, 252)

    # Trend health feature toggle
    use_trend_health: bool = True
    
    # Short reversal windows
    short_reversal_windows: tuple[int, ...] = (5, 10)


# Default configuration for enhanced model
DEFAULT_ENHANCED_CONFIG = EnhancedFeatureConfig()


# =============================================================================
# SHORT-TERM REVERSAL FEATURES
# =============================================================================

def compute_short_reversal_features(
    prices: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Compute short-term reversal (momentum) features.
    
    Short-term reversal captures mean reversion at very short horizons
    (typically 5-10 days).
    
    Args:
        prices: Price DataFrame with tickers as columns.
        windows: Windows for momentum calculation (e.g., 5, 10 days).
    
    Returns:
        DataFrame with MultiIndex columns (ticker, feature_name).
    """
    features = {}
    
    for window in windows:
        # Simple momentum: (price_t / price_{t-window}) - 1
        mom = prices.pct_change(periods=window)
        for ticker in mom.columns:
            features[(ticker, f"tech_mom_{window}d")] = mom[ticker]
    
    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])
    
    logger.debug(f"Computed short reversal features for windows {windows}")
    return result


# =============================================================================
# ROLLING BETA ESTIMATION
# =============================================================================

def compute_rolling_beta(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Compute rolling market beta for each stock.
    
    Beta is estimated via rolling OLS regression:
        stock_ret_t = alpha + beta * benchmark_ret_t + epsilon
    
    Args:
        stock_returns: Daily returns DataFrame with tickers as columns.
        benchmark_returns: Daily returns Series for benchmark (e.g., SPY).
        window: Rolling window in trading days.
        min_periods: Minimum periods for valid beta. Defaults to window//2.
    
    Returns:
        DataFrame with rolling beta estimates, same shape as stock_returns.
    """
    if min_periods is None:
        min_periods = max(window // 2, 20)
    
    betas = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)
    
    # Align benchmark to stock returns index
    benchmark_aligned = benchmark_returns.reindex(stock_returns.index)
    
    for ticker in stock_returns.columns:
        stock_ret = stock_returns[ticker]
        
        # Rolling covariance and variance
        cov = stock_ret.rolling(window=window, min_periods=min_periods).cov(benchmark_aligned)
        var = benchmark_aligned.rolling(window=window, min_periods=min_periods).var()
        
        # Beta = Cov(stock, market) / Var(market)
        beta = cov / var.replace(0, np.nan)
        betas[ticker] = beta
    
    logger.debug(f"Computed rolling betas with window={window}")
    return betas


def compute_rolling_residuals(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    betas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute residual returns after removing market exposure.
    
    residual_ret_t = stock_ret_t - beta_t * benchmark_ret_t
    
    Args:
        stock_returns: Daily returns DataFrame.
        benchmark_returns: Daily benchmark returns Series.
        betas: Rolling beta estimates DataFrame.
    
    Returns:
        DataFrame of residual returns.
    """
    benchmark_aligned = benchmark_returns.reindex(stock_returns.index)
    
    # Broadcast benchmark returns and multiply by betas
    market_component = betas.multiply(benchmark_aligned, axis=0)
    
    residuals = stock_returns - market_component
    
    return residuals


# =============================================================================
# RESIDUAL MOMENTUM FEATURES
# =============================================================================

def compute_residual_momentum_features(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    windows: tuple[int, ...] = (63, 126, 252),
    beta_window: int = 60,
) -> pd.DataFrame:
    """
    Compute beta-adjusted (residual) momentum features.
    
    For each stock, estimate rolling beta and compute residual returns.
    Then compute cumulative residual returns over various horizons.
    
    Args:
        prices: Price DataFrame with tickers as columns.
        benchmark_prices: Benchmark price Series.
        windows: Momentum windows for cumulative residual returns.
        beta_window: Window for beta estimation.
    
    Returns:
        DataFrame with MultiIndex columns (ticker, feature_name).
    """
    from auto_researcher.features.technical import compute_returns
    
    stock_returns = compute_returns(prices)
    benchmark_returns = compute_returns(benchmark_prices)
    
    # Compute rolling betas
    betas = compute_rolling_beta(stock_returns, benchmark_returns, window=beta_window)
    
    # Compute residual returns
    residuals = compute_rolling_residuals(stock_returns, benchmark_returns, betas)
    
    features = {}
    
    for window in windows:
        # Cumulative residual return over window
        cum_resid = residuals.rolling(window=window).sum()
        for ticker in cum_resid.columns:
            features[(ticker, f"tech_resid_mom_{window}")] = cum_resid[ticker]
    
    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])
    
    logger.debug(f"Computed residual momentum features for windows {windows}")
    return result


# =============================================================================
# IDIOSYNCRATIC VOLATILITY FEATURES
# =============================================================================

def compute_idio_volatility_features(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    windows: tuple[int, ...] = (21, 63),
    beta_window: int = 60,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute idiosyncratic volatility features.
    
    Idiosyncratic volatility is the standard deviation of residual returns
    after removing systematic market exposure.
    
    Args:
        prices: Price DataFrame with tickers as columns.
        benchmark_prices: Benchmark price Series.
        windows: Windows for volatility computation.
        beta_window: Window for beta estimation.
        annualize: If True, annualize volatility (multiply by sqrt(252)).
    
    Returns:
        DataFrame with MultiIndex columns (ticker, feature_name).
    """
    from auto_researcher.features.technical import compute_returns
    
    stock_returns = compute_returns(prices)
    benchmark_returns = compute_returns(benchmark_prices)
    
    # Compute rolling betas
    betas = compute_rolling_beta(stock_returns, benchmark_returns, window=beta_window)
    
    # Compute residual returns
    residuals = compute_rolling_residuals(stock_returns, benchmark_returns, betas)
    
    features = {}
    
    for window in windows:
        # Rolling std of residuals
        vol = residuals.rolling(window=window).std()
        if annualize:
            vol = vol * np.sqrt(252)
        
        for ticker in vol.columns:
            features[(ticker, f"tech_vol_idio_{window}")] = vol[ticker]
    
    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])
    
    logger.debug(f"Computed idiosyncratic volatility features for windows {windows}")
    return result


# =============================================================================
# TREND HEALTH FEATURES
# =============================================================================

def compute_trend_health_features(
    prices: pd.DataFrame,
    short_window: int = 63,
    long_window: int = 252,
) -> pd.DataFrame:
    """
    Compute trend health features using moving average ratios.
    
    Feature:
      tech_ma_ratio_{short}_{long}: (MA_short / MA_long) - 1
    """
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    ma_short = prices.rolling(window=short_window, min_periods=short_window // 2).mean()
    ma_long = prices.rolling(window=long_window, min_periods=long_window // 2).mean()
    ma_ratio = (ma_short / ma_long) - 1

    features = {}
    for ticker in ma_ratio.columns:
        features[(ticker, f"tech_ma_ratio_{short_window}_{long_window}")] = ma_ratio[ticker]

    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])
    logger.debug(f"Computed trend health features: MA ratio {short_window}/{long_window}")
    return result


# =============================================================================
# ROBUST DISPERSION METRICS (MAD)
# =============================================================================

def compute_mad_features(
    prices: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute Median Absolute Deviation (MAD) features.
    
    MAD is more robust to outliers than standard deviation.
    MAD = median(|x - median(x)|)
    
    Args:
        prices: Price DataFrame with tickers as columns.
        window: Rolling window for MAD computation.
    
    Returns:
        DataFrame with MultiIndex columns (ticker, feature_name).
    """
    from auto_researcher.features.technical import compute_returns
    
    returns = compute_returns(prices)
    
    features = {}
    
    def rolling_mad(x):
        """Compute MAD for a rolling window."""
        if len(x) < window // 2:
            return np.nan
        median = np.nanmedian(x)
        return np.nanmedian(np.abs(x - median))
    
    for ticker in returns.columns:
        # MAD of returns
        mad_ret = returns[ticker].rolling(window=window).apply(rolling_mad, raw=True)
        features[(ticker, f"tech_mad_ret_{window}")] = mad_ret
        
        # MAD of rolling volatility (more stable measure of vol dispersion)
        vol_21 = returns[ticker].rolling(window=21).std()
        mad_vol = vol_21.rolling(window=window).apply(rolling_mad, raw=True)
        features[(ticker, f"tech_mad_vol_{window}")] = mad_vol
    
    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])
    
    logger.debug(f"Computed MAD features with window={window}")
    return result


# =============================================================================
# SECTOR ENCODING (ONE-HOT)
# =============================================================================

# GICS Sector mapping (can be extended with actual data source)
DEFAULT_SECTOR_MAPPING = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOG": "Technology", "GOOGL": "Technology", "META": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "CRM": "Technology",
    "CSCO": "Technology", "ACN": "Technology", "ADBE": "Technology",
    "AMD": "Technology", "INTC": "Technology", "IBM": "Technology",
    "TXN": "Technology", "QCOM": "Technology", "INTU": "Technology",
    "NOW": "Technology", "AMAT": "Technology", "MU": "Technology",
    
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "PFE": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "MDT": "Healthcare",
    "GILD": "Healthcare", "CVS": "Healthcare", "ISRG": "Healthcare",
    "CI": "Healthcare", "ELV": "Healthcare", "SYK": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare",
    
    # Financials
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "C": "Financials", "AXP": "Financials", "SCHW": "Financials",
    "CB": "Financials", "PNC": "Financials", "USB": "Financials",
    "TFC": "Financials", "CME": "Financials", "ICE": "Financials",
    "AON": "Financials", "MMC": "Financials",
    
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "MAR": "Consumer Discretionary", "ORLY": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "F": "Consumer Discretionary",
    "ROST": "Consumer Discretionary", "DHI": "Consumer Discretionary",
    "LEN": "Consumer Discretionary", "YUM": "Consumer Discretionary",
    
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    "TGT": "Consumer Staples", "GIS": "Consumer Staples", "KMB": "Consumer Staples",
    "STZ": "Consumer Staples", "KHC": "Consumer Staples", "SYY": "Consumer Staples",
    
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",
    "OXY": "Energy", "PXD": "Energy", "HAL": "Energy", "DVN": "Energy",
    "KMI": "Energy", "WMB": "Energy", "HES": "Energy",
    
    # Industrials
    "CAT": "Industrials", "RTX": "Industrials", "UNP": "Industrials",
    "HON": "Industrials", "BA": "Industrials", "DE": "Industrials",
    "LMT": "Industrials", "UPS": "Industrials", "GE": "Industrials",
    "MMM": "Industrials", "ADP": "Industrials", "ITW": "Industrials",
    "ETN": "Industrials", "EMR": "Industrials", "NOC": "Industrials",
    "FDX": "Industrials", "CSX": "Industrials", "NSC": "Industrials",
    "WM": "Industrials", "GD": "Industrials",
    
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "FCX": "Materials", "ECL": "Materials", "NEM": "Materials",
    "DOW": "Materials", "NUE": "Materials", "DD": "Materials",
    "VMC": "Materials", "MLM": "Materials", "PPG": "Materials",
    
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "SRE": "Utilities",
    "EXC": "Utilities", "XEL": "Utilities", "ED": "Utilities",
    "PEG": "Utilities", "WEC": "Utilities", "ES": "Utilities",
    
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate",
    "CCI": "Real Estate", "PSA": "Real Estate", "O": "Real Estate",
    "SPG": "Real Estate", "WELL": "Real Estate", "DLR": "Real Estate",
    "VICI": "Real Estate", "AVB": "Real Estate", "EQR": "Real Estate",
    
    # Communication Services
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "NFLX": "Communication Services", "VZ": "Communication Services",
    "T": "Communication Services", "TMUS": "Communication Services",
    "CHTR": "Communication Services", "EA": "Communication Services",
    "ATVI": "Communication Services", "WBD": "Communication Services",
    
    # ETFs and Benchmarks
    "SPY": "ETF",
}


def get_sector_for_ticker(
    ticker: str,
    sector_mapping: dict[str, str] | None = None,
    default_sector: str = "Unknown",
) -> str:
    """
    Get sector for a ticker from mapping.
    
    Args:
        ticker: Stock ticker.
        sector_mapping: Ticker -> Sector mapping. Uses default if None.
        default_sector: Sector to use if ticker not found.
    
    Returns:
        Sector name.
    """
    if sector_mapping is None:
        sector_mapping = DEFAULT_SECTOR_MAPPING
    
    return sector_mapping.get(ticker, default_sector)


def compute_sector_features(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    sector_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Compute one-hot encoded sector features.
    
    Args:
        tickers: List of tickers in the universe.
        dates: DatetimeIndex for the feature matrix.
        sector_mapping: Ticker -> Sector mapping.
    
    Returns:
        DataFrame with MultiIndex columns (ticker, sector_feature).
    """
    if sector_mapping is None:
        sector_mapping = DEFAULT_SECTOR_MAPPING
    
    # Get all sectors (excluding ETF)
    all_sectors = sorted(set(
        s for s in sector_mapping.values() 
        if s not in ("ETF", "Unknown")
    ))
    
    features = {}
    
    for ticker in tickers:
        sector = get_sector_for_ticker(ticker, sector_mapping)
        
        for s in all_sectors:
            # One-hot encoding: 1 if ticker belongs to sector, 0 otherwise
            value = 1.0 if sector == s else 0.0
            # Create constant series for all dates
            features[(ticker, f"sector_{s.replace(' ', '_')}")] = pd.Series(
                value, index=dates, dtype=float
            )
    
    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])
    
    logger.debug(f"Computed sector OHE features for {len(tickers)} tickers, {len(all_sectors)} sectors")
    return result


# =============================================================================
# CROSS-SECTIONAL NORMALIZATION
# =============================================================================

def normalize_cross_sectionally(
    features: pd.DataFrame,
    robust: bool = True,
    by_sector: bool = False,
    sector_mapping: dict[str, str] | None = None,
    exclude_prefixes: tuple[str, ...] = ("sector_",),
) -> pd.DataFrame:
    """
    Apply cross-sectional normalization to features.
    
    For each date, transform features to z-scores within the cross-section.
    
    Args:
        features: Feature DataFrame with MultiIndex columns (ticker, feature).
        robust: If True, use median and MAD. If False, use mean and std.
        by_sector: If True, normalize within sector groups.
        sector_mapping: Ticker -> Sector mapping (required if by_sector=True).
        exclude_prefixes: Feature prefixes to exclude from normalization
            (e.g., sector dummies should not be normalized).
    
    Returns:
        Normalized feature DataFrame.
    """
    if by_sector and sector_mapping is None:
        sector_mapping = DEFAULT_SECTOR_MAPPING
    
    result = features.copy()
    
    # Get unique feature names (level 1 of MultiIndex)
    feature_names = features.columns.get_level_values(1).unique()
    
    for feat_name in feature_names:
        # Skip excluded features (like sector dummies)
        if any(feat_name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        # Get all columns for this feature across tickers
        feat_cols = [(t, feat_name) for t in features.columns.get_level_values(0).unique()
                     if (t, feat_name) in features.columns]
        
        if not feat_cols:
            continue
        
        # Extract cross-sectional data for this feature
        feat_data = features[feat_cols]
        
        if by_sector:
            # Normalize within sector groups
            normalized = _normalize_by_sector(feat_data, sector_mapping, robust)
        else:
            # Global cross-sectional normalization
            normalized = _normalize_global(feat_data, robust)
        
        # Update result
        for col in feat_cols:
            result[col] = normalized[col]
    
    return result


def _normalize_global(feat_data: pd.DataFrame, robust: bool) -> pd.DataFrame:
    """
    Apply global cross-sectional normalization.
    
    For each row (date), compute z-scores across all tickers.
    """
    if robust:
        # Robust: median and MAD
        center = feat_data.median(axis=1)

        def _safe_mad(row):
            """MAD with small-sample guard (scipy needs >= 3 values)."""
            vals = row.dropna()
            if len(vals) < 3:
                return np.nan
            return stats.median_abs_deviation(vals, nan_policy='omit')

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*small.*')
            # Also catch scipy's SmallSampleWarning if it exists
            try:
                from scipy.stats import SmallSampleWarning
                warnings.filterwarnings('ignore', category=SmallSampleWarning)
            except ImportError:
                pass
            scale = feat_data.apply(_safe_mad, axis=1)
        # Scale MAD to be consistent with std (MAD * 1.4826 ≈ std for normal dist)
        scale = scale * 1.4826
    else:
        # Standard: mean and std
        center = feat_data.mean(axis=1)
        scale = feat_data.std(axis=1)
    
    # Avoid division by zero
    scale = scale.replace(0, np.nan)
    
    # Normalize: (x - center) / scale
    normalized = feat_data.sub(center, axis=0).div(scale, axis=0)
    
    return normalized


def apply_regime_scaling(
    features: pd.DataFrame,
    regime_mom_feature: str,
    threshold: float,
    short_reversal_features: tuple[str, ...],
    short_reversal_scale: float,
) -> pd.DataFrame:
    """
    Damp short-term reversal features when long-term momentum is negative.
    
    For each ticker and date, if regime_mom_feature < threshold, scale the
    short_reversal_features by short_reversal_scale.
    """
    if regime_mom_feature not in features.columns.get_level_values(1):
        return features
    
    result = features.copy()
    
    # Regime signal: DataFrame with columns = tickers, index = dates
    regime = features.xs(regime_mom_feature, level=1, axis=1)
    
    for feat in short_reversal_features:
        if feat not in features.columns.get_level_values(1):
            continue
        short_feat = features.xs(feat, level=1, axis=1)
        scaled = short_feat.where(regime >= threshold, short_feat * short_reversal_scale)
        
        # Write back to MultiIndex columns
        for ticker in scaled.columns:
            result[(ticker, feat)] = scaled[ticker]
    
    return result


def _normalize_by_sector(
    feat_data: pd.DataFrame,
    sector_mapping: dict[str, str],
    robust: bool,
) -> pd.DataFrame:
    """
    Apply sector-relative cross-sectional normalization.
    
    For each row (date), compute z-scores within each sector group.
    """
    normalized = feat_data.copy()
    
    # Get tickers from column MultiIndex
    tickers = [col[0] for col in feat_data.columns]
    
    # Group columns by sector
    sector_groups = {}
    for ticker in tickers:
        sector = get_sector_for_ticker(ticker, sector_mapping)
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(ticker)
    
    # Normalize within each sector
    for sector, sector_tickers in sector_groups.items():
        sector_cols = [(t, feat_data.columns[0][1]) for t in sector_tickers 
                       if (t, feat_data.columns[0][1]) in feat_data.columns]
        
        if len(sector_cols) < 2:
            # Need at least 2 stocks to normalize
            continue
        
        sector_data = feat_data[sector_cols]
        sector_normalized = _normalize_global(sector_data, robust)
        
        for col in sector_cols:
            normalized[col] = sector_normalized[col]
    
    return normalized


# =============================================================================
# COMBINED ENHANCED FEATURE BUILDER
# =============================================================================

def compute_all_enhanced_features(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    config: EnhancedFeatureConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Compute all enhanced features based on configuration.
    
    Args:
        prices: Price DataFrame with tickers as columns and DatetimeIndex.
        benchmark: Benchmark ticker for beta/residual calculations.
        config: Enhanced feature configuration.
        sector_mapping: Ticker -> Sector mapping for sector features.
    
    Returns:
        DataFrame with MultiIndex columns (ticker, feature_name).
    """
    if config is None:
        config = DEFAULT_ENHANCED_CONFIG
    
    # Separate benchmark from stock prices
    if benchmark in prices.columns:
        benchmark_prices = prices[benchmark]
        stock_prices = prices.drop(columns=[benchmark])
    else:
        benchmark_prices = None
        stock_prices = prices
        logger.warning(f"Benchmark {benchmark} not in prices, skipping residual features")
    
    all_features: list[pd.DataFrame] = []
    
    # Short-term reversal
    if config.use_short_reversal:
        short_rev = compute_short_reversal_features(
            stock_prices,
            windows=config.short_reversal_windows,
        )
        all_features.append(short_rev)
        logger.info(f"Added {len(short_rev.columns)} short reversal features")
    
    # Residual momentum
    if config.use_residual_mom and benchmark_prices is not None:
        resid_mom = compute_residual_momentum_features(
            stock_prices,
            benchmark_prices,
            windows=config.residual_mom_windows,
            beta_window=config.beta_window,
        )
        all_features.append(resid_mom)
        logger.info(f"Added {len(resid_mom.columns)} residual momentum features")
    
    # Idiosyncratic volatility
    if config.use_idio_vol and benchmark_prices is not None:
        idio_vol = compute_idio_volatility_features(
            stock_prices,
            benchmark_prices,
            windows=config.idio_vol_windows,
            beta_window=config.beta_window,
        )
        all_features.append(idio_vol)
        logger.info(f"Added {len(idio_vol.columns)} idiosyncratic volatility features")
    
    # MAD features
    if config.use_mad_metrics:
        mad_feats = compute_mad_features(stock_prices, window=config.mad_window)
        all_features.append(mad_feats)
        logger.info(f"Added {len(mad_feats.columns)} MAD features")

    # Trend health features
    if config.use_trend_health:
        short_w, long_w = config.trend_health_windows
        trend_feats = compute_trend_health_features(
            stock_prices,
            short_window=short_w,
            long_window=long_w,
        )
        all_features.append(trend_feats)
        logger.info(f"Added {len(trend_feats.columns)} trend health features")
    
    # Sector encoding
    if config.use_sector_ohe:
        sector_feats = compute_sector_features(
            tickers=list(stock_prices.columns),
            dates=stock_prices.index,
            sector_mapping=sector_mapping,
        )
        all_features.append(sector_feats)
        logger.info(f"Added {len(sector_feats.columns)} sector OHE features")
    
    if not all_features:
        raise ValueError("No enhanced features computed - check configuration")
    
    # Combine all features - memory efficient approach
    import gc
    if len(all_features) == 1:
        result = all_features[0]
    else:
        result = all_features[0]
        for i, other in enumerate(all_features[1:], 1):
            # Merge on index, combine columns
            result = pd.concat([result, other], axis=1)
            # Clear the merged dataframe from list to free memory
            all_features[i] = None
        # Clear intermediate list
        del all_features
        gc.collect()
    
    # Apply cross-sectional normalization
    if config.use_cross_sec_norm:
        result = normalize_cross_sectionally(
            result,
            robust=config.cross_sec_norm_robust,
            by_sector=config.cross_sec_norm_by_sector,
            sector_mapping=sector_mapping,
            exclude_prefixes=("sector_",),  # Don't normalize sector dummies
        )
        logger.info("Applied cross-sectional normalization to features")
    
    # Regime-aware scaling of short-term reversal signals
    if config.use_regime_scaling:
        result = apply_regime_scaling(
            result,
            regime_mom_feature=config.regime_mom_feature,
            threshold=config.regime_threshold,
            short_reversal_features=config.regime_short_reversal_features,
            short_reversal_scale=config.regime_short_reversal_scale,
        )
        logger.info("Applied regime-aware scaling to short-term reversal features")
    
    return result


# =============================================================================
# ROLLING BETA AS STANDALONE FEATURE
# =============================================================================

def compute_beta_features(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling beta as a feature.
    
    Useful for beta-neutral weighting or as a model feature.
    
    Args:
        prices: Price DataFrame with tickers as columns.
        benchmark: Benchmark ticker.
        window: Rolling window for beta estimation.
    
    Returns:
        DataFrame with MultiIndex columns (ticker, 'tech_beta_{window}').
    """
    from auto_researcher.features.technical import compute_returns
    
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} not in prices")
    
    stock_prices = prices.drop(columns=[benchmark])
    benchmark_prices = prices[benchmark]
    
    stock_returns = compute_returns(stock_prices)
    benchmark_returns = compute_returns(benchmark_prices)
    
    betas = compute_rolling_beta(stock_returns, benchmark_returns, window=window)
    
    features = {}
    for ticker in betas.columns:
        features[(ticker, f"tech_beta_{window}")] = betas[ticker]
    
    result = pd.DataFrame(features)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])
    
    return result
