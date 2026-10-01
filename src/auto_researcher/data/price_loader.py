"""
Price data loading via yfinance.

This module provides a clean interface for downloading and caching price data.
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "price_cache"


def download_price_history(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    use_default_cache: bool = True,
) -> pd.DataFrame:
    """
    Download historical price data for a list of tickers.

    Uses yfinance to fetch OHLCV data. Results are returned with a DatetimeIndex
    and MultiIndex columns (field, ticker).

    Args:
        tickers: List of ticker symbols.
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.
        cache_dir: Optional directory to cache downloaded data.
        max_retries: Maximum number of retry attempts for failed downloads.
        retry_delay: Delay between retries in seconds.
        use_default_cache: If True and cache_dir is None, use default cache.

    Returns:
        DataFrame with DatetimeIndex and MultiIndex columns (field, ticker).
        Fields include: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.

    Raises:
        ValueError: If no data is returned for any ticker.

    Examples:
        >>> df = download_price_history(['AAPL', 'MSFT'], '2023-01-01', '2023-12-31')
        >>> isinstance(df.index, pd.DatetimeIndex)
        True
    """
    logger.info(f"Downloading price data for {len(tickers)} tickers from {start} to {end}")

    # Use default cache if none provided
    if cache_dir is None and use_default_cache:
        cache_dir = DEFAULT_CACHE_DIR

    # Check cache first if cache_dir is provided
    if cache_dir is not None:
        cache_path = cache_dir / f"prices_{start}_{end}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            try:
                cached_data = pd.read_parquet(cache_path)
                # Verify cached data has the tickers we need
                if isinstance(cached_data.columns, pd.MultiIndex):
                    cached_tickers = set(cached_data.columns.get_level_values(1))
                    requested_tickers = set(tickers)
                    if requested_tickers.issubset(cached_tickers):
                        logger.info(f"Using cached data with {len(cached_data)} rows")
                        return cached_data
                    else:
                        missing = requested_tickers - cached_tickers
                        logger.warning(f"Cache missing tickers: {missing}, re-downloading")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, re-downloading")

    # Download from yfinance with retries
    data = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            
            # Check if we got valid data
            if data is not None and not data.empty:
                break
            else:
                logger.warning(f"Attempt {attempt + 1} returned empty data")
                last_error = ValueError("Empty data returned")
                
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
        
        # Wait before retry (except on last attempt)
        if attempt < max_retries - 1:
            logger.info(f"Waiting {retry_delay}s before retry...")
            time.sleep(retry_delay)
            retry_delay *= 1.5  # Exponential backoff

    if data is None or data.empty:
        # Check if we have any cached data as fallback
        if cache_dir is not None:
            requested_start = pd.Timestamp(start)
            requested_end = pd.Timestamp(end)
            requested_days = (requested_end - requested_start).days

            best_fallback = None
            best_overlap = 0
            best_file_name = ""

            for cache_file in cache_dir.glob("prices_*.parquet"):
                # Parse date range from filename to validate overlap
                match = re.match(
                    r"prices_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet",
                    cache_file.name,
                )
                if match:
                    cached_start = pd.Timestamp(match.group(1))
                    cached_end = pd.Timestamp(match.group(2))
                    overlap_start = max(requested_start, cached_start)
                    overlap_end = min(requested_end, cached_end)
                    overlap_days = max(0, (overlap_end - overlap_start).days)

                    # Require at least 50% date range overlap
                    if requested_days > 0 and overlap_days < requested_days * 0.5:
                        logger.debug(
                            f"Skipping {cache_file.name}: only {overlap_days}/{requested_days} "
                            f"days overlap ({overlap_days / requested_days * 100:.0f}%)"
                        )
                        continue
                else:
                    # Can't parse dates from filename -- skip to avoid wrong-range data
                    logger.debug(f"Skipping {cache_file.name}: can't parse date range")
                    continue

                try:
                    cached_data = pd.read_parquet(cache_file)
                    if isinstance(cached_data.columns, pd.MultiIndex):
                        cached_tickers = set(cached_data.columns.get_level_values(1))
                        requested_tickers = set(tickers)
                        ticker_overlap = len(requested_tickers & cached_tickers)
                        if ticker_overlap > 0 and overlap_days > best_overlap:
                            best_overlap = overlap_days
                            best_fallback = cached_data
                            best_file_name = cache_file.name
                except Exception:
                    continue

            if best_fallback is not None:
                coverage_pct = best_overlap / requested_days * 100 if requested_days > 0 else 0
                logger.warning(
                    f"Using fallback cache from {best_file_name} "
                    f"({best_overlap}/{requested_days} days overlap, {coverage_pct:.0f}% coverage)"
                )
                return best_fallback

        raise ValueError(f"No data returned for tickers: {tickers}")

    # Ensure consistent MultiIndex structure even for single ticker
    if len(tickers) == 1 and not isinstance(data.columns, pd.MultiIndex):
        # yfinance returns flat columns for single ticker
        # Create tuples for MultiIndex
        new_columns = [(col, tickers[0]) for col in data.columns]
        data.columns = pd.MultiIndex.from_tuples(new_columns)

    # Ensure DatetimeIndex
    data.index = pd.to_datetime(data.index)

    # Cache if directory provided
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            data.to_parquet(cache_path)
            logger.info(f"Cached data to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

    logger.info(f"Downloaded {len(data)} rows of price data")
    return data


_CACHE_NAME = re.compile(r"prices_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet")


def load_cached_price_panel(
    cache_dir: Path | None = None,
    field: str = "Adj Close",
    tickers: list[str] | None = None,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """
    Stitch every cached download into one continuous, offline price panel.

    Cached files come from different download dates. Adjusted closes from
    different vintages are not comparable in *level* (each vintage is
    back-adjusted for the dividends paid before it was downloaded), but their
    daily *returns* are. The panel is therefore rebuilt by chaining daily
    returns, preferring the most recent vintage wherever files overlap. Prices
    are normalized (each series starts at 1.0), which is fine for any
    scale-invariant feature or return calculation.

    Args:
        cache_dir: Directory holding ``prices_<start>_<end>.parquet`` files.
        field: Price field to extract (``"Adj Close"`` or ``"Close"``).
        tickers: Optional subset of tickers to keep.
        min_coverage: Drop tickers with less than this fraction of non-missing days.

    Returns:
        Wide DataFrame (date x ticker). Dates on which a ticker had no price in
        any cached file (e.g. before its IPO) are NaN.

    Raises:
        FileNotFoundError: If no usable cache file is found.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    vintages: list[tuple[pd.Timestamp, float, pd.DataFrame]] = []
    for path in cache_dir.glob("prices_*.parquet"):
        match = _CACHE_NAME.fullmatch(path.name)
        if not match:
            continue
        try:
            raw = pd.read_parquet(path)
        except Exception as exc:  # corrupt/partial cache file
            logger.warning(f"Skipping unreadable cache file {path.name}: {exc}")
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            continue
        if field not in raw.columns.get_level_values(0):
            continue
        px = raw[field].copy()
        px.index = pd.DatetimeIndex(pd.to_datetime(px.index)).tz_localize(None)
        px = px.sort_index()
        px = px[~px.index.duplicated(keep="last")]
        vintages.append((pd.Timestamp(match.group(2)), path.stat().st_mtime, px))

    if not vintages:
        raise FileNotFoundError(f"No usable price cache files in {cache_dir}")

    # Newest download first so combine_first prefers the latest vintage.
    vintages.sort(key=lambda v: (v[0], v[1]), reverse=True)
    returns: pd.DataFrame | None = None
    has_price: list[pd.DataFrame] = []
    for _, _, px in vintages:
        if tickers is not None:
            px = px[[t for t in tickers if t in px.columns]]
        # ffill before pct_change so a return spanning a data gap lands on the
        # day trading resumed, then blank out the gap days themselves.
        r = px.ffill().pct_change(fill_method=None).where(px.notna())
        returns = r if returns is None else returns.combine_first(r)
        has_price.append(px.notna())

    assert returns is not None
    returns = returns.sort_index()
    available = (
        pd.concat(has_price)
        .astype(float)
        .groupby(level=0)
        .max()
        .reindex(index=returns.index, columns=returns.columns)
        .fillna(0.0)
        .astype(bool)
    )
    prices = (1.0 + returns.fillna(0.0)).cumprod().where(available)

    if min_coverage > 0:
        coverage = prices.notna().mean()
        prices = prices.loc[:, coverage >= min_coverage]

    logger.info(
        f"Loaded cached price panel: {prices.shape[1]} tickers, "
        f"{prices.index.min().date()} to {prices.index.max().date()} "
        f"from {len(vintages)} files"
    )
    return prices.sort_index(axis=1)


def get_adjusted_close(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Extract adjusted close prices from the full price DataFrame.

    Args:
        prices: DataFrame with MultiIndex columns (field, ticker).

    Returns:
        DataFrame with tickers as columns and dates as index.
    """
    if "Adj Close" in prices.columns.get_level_values(0):
        return prices["Adj Close"]
    elif "Close" in prices.columns.get_level_values(0):
        logger.warning("Adj Close not found, using Close prices")
        return prices["Close"]
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in price data")


def get_volume(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Extract volume from the full price DataFrame.

    Args:
        prices: DataFrame with MultiIndex columns (field, ticker).

    Returns:
        DataFrame with tickers as columns and dates as index.
    """
    if "Volume" in prices.columns.get_level_values(0):
        return prices["Volume"]
    else:
        raise KeyError("'Volume' not found in price data")


def validate_price_data(prices: pd.DataFrame, tickers: list[str]) -> dict[str, bool]:
    """
    Validate that price data exists for all requested tickers.

    Args:
        prices: DataFrame with price data.
        tickers: List of expected tickers.

    Returns:
        Dictionary mapping ticker to whether valid data exists.
    """
    adj_close = get_adjusted_close(prices)
    result = {}
    for ticker in tickers:
        if ticker in adj_close.columns:
            # Check for non-null data
            result[ticker] = adj_close[ticker].notna().sum() > 0
        else:
            result[ticker] = False
    return result
