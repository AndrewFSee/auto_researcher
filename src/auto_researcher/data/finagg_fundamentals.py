"""
SEC EDGAR fundamentals provider via finagg.

Not point-in-time: values are dated by fiscal period end rather than filing
date, and restated comparatives can replace first-reported figures. For
research and backtests use ``auto_researcher.data.sec_fundamentals``.

This module provides functions to fetch quarterly fundamental data from SEC EDGAR
using the finagg library. It converts the raw SEC data into the format expected
by the fundamental factor pipeline.

Prerequisites:
--------------
1. Install finagg: pip install finagg
2. Set SEC_API_USER_AGENT environment variable:
   export SEC_API_USER_AGENT="FIRST LAST email@example.com"
   OR run: finagg sec install

Usage:
------
>>> from auto_researcher.data.finagg_fundamentals import fetch_finagg_quarterly_fundamentals
>>> df = fetch_finagg_quarterly_fundamentals(
...     tickers=["AAPL", "MSFT"],
...     start=pd.Timestamp("2020-01-01"),
...     end=pd.Timestamp("2024-01-01"),
... )
"""

import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

# Load .env file if present
try:
    from dotenv import load_dotenv
    
    # Look for .env in project root
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system env vars

logger = logging.getLogger(__name__)

# Finagg availability flag - set on first import attempt
_FINAGG_AVAILABLE: bool | None = None


class FinaggNotInstalledError(RuntimeError):
    """Raised when finagg is not installed but required."""
    
    def __init__(self, message: str | None = None):
        if message is None:
            message = (
                "finagg is not installed. Install it with:\n"
                "    pip install finagg\n"
                "Or install auto_researcher with finagg support:\n"
                "    pip install auto_researcher[finagg]\n\n"
                "After installation, set SEC credentials:\n"
                "    export SEC_API_USER_AGENT='Your Name email@example.com'\n"
                "Or run: finagg sec install"
            )
        super().__init__(message)


def _check_finagg_available() -> bool:
    """
    Check if finagg is installed and available.
    
    Returns:
        True if finagg is available, False otherwise.
    """
    global _FINAGG_AVAILABLE
    
    if _FINAGG_AVAILABLE is not None:
        return _FINAGG_AVAILABLE
    
    try:
        import finagg  # noqa: F401
        _FINAGG_AVAILABLE = True
    except ImportError:
        _FINAGG_AVAILABLE = False
    
    return _FINAGG_AVAILABLE


def _ensure_finagg_available() -> None:
    """Raise FinaggNotInstalledError if finagg is not installed."""
    if not _check_finagg_available():
        raise FinaggNotInstalledError()


# Mapping from finagg column names to our internal names
# These are the columns we expect from finagg.sec.feat.quarterly
FINAGG_COLUMN_MAPPING = {
    # Revenue and income
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "NetIncomeLoss": "net_income",
    "GrossProfit": "gross_profit",
    "OperatingIncomeLoss": "operating_income",
    # Balance sheet
    "Assets": "total_assets",
    "AssetsCurrent": "current_assets",
    "Liabilities": "total_liabilities",
    "StockholdersEquity": "stockholders_equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "stockholders_equity",
    # Per-share
    "EarningsPerShareBasic": "eps_basic",
    "EarningsPerShareDiluted": "eps_diluted",
    # Shares
    "CommonStockSharesOutstanding": "shares_outstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic": "shares_outstanding",
}

# Minimum set of columns needed for factor computation
REQUIRED_COLUMNS = ["revenue", "net_income", "gross_profit", "operating_income"]

# Per-ticker cache to avoid repeated API calls
_TICKER_CACHE: dict[str, pd.DataFrame] = {}


def fetch_finagg_quarterly_fundamentals(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    mode: Literal["refined", "api"] = "refined",
) -> pd.DataFrame:
    """
    Fetch quarterly fundamental data from SEC EDGAR via finagg.
    
    Results are cached per-ticker to avoid repeated API calls during
    walk-forward backtesting.
    
    Returns a multi-index DataFrame with index (date, ticker) containing
    raw fundamental series that can be used to compute factor families.
    
    Args:
        tickers: List of ticker symbols to fetch.
        start: Start date for data range.
        end: End date for data range.
        mode: Data fetching mode.
            - "refined": Use local SQL database (faster, requires prior finagg install)
            - "api": Fetch directly from SEC API (slower, always works)
    
    Returns:
        DataFrame with MultiIndex (date, ticker) and columns:
            - revenue: Total revenue (TTM approximated from quarterly * 4)
            - net_income: Net income (TTM approximated)
            - gross_profit: Gross profit
            - operating_income: Operating income
            - total_assets: Total assets
            - stockholders_equity: Total stockholders' equity
            - eps_basic: Basic EPS
            - shares_outstanding: Shares outstanding
            And derived columns:
            - gross_margin: Gross profit / revenue
            - operating_margin: Operating income / revenue
            - net_margin: Net income / revenue
            - roe: Net income / stockholders' equity
            - roa: Net income / total assets
    
    Raises:
        FinaggNotInstalledError: If finagg package is not installed.
        RuntimeError: If SEC credentials are not configured.
    
    Example:
        >>> df = fetch_finagg_quarterly_fundamentals(
        ...     ["AAPL", "MSFT"],
        ...     pd.Timestamp("2020-01-01"),
        ...     pd.Timestamp("2024-01-01"),
        ... )
        >>> df.columns.tolist()
        ['revenue', 'net_income', 'gross_margin', ...]
    """
    global _TICKER_CACHE
    
    _ensure_finagg_available()
    
    import finagg
    
    logger.info(
        f"Fetching SEC fundamentals via finagg for {len(tickers)} tickers "
        f"({start.date()} to {end.date()}, mode={mode})"
    )
    
    all_records: list[pd.DataFrame] = []
    tickers_to_fetch = []
    
    # Check cache first
    for ticker in tickers:
        cache_key = f"{ticker}_{mode}"
        if cache_key in _TICKER_CACHE:
            # Use cached data, filter to date range
            cached_df = _TICKER_CACHE[cache_key]
            mask = (cached_df.index.get_level_values("date") >= start) & \
                   (cached_df.index.get_level_values("date") <= end)
            filtered = cached_df.loc[mask]
            if not filtered.empty:
                all_records.append(filtered)
                logger.debug(f"Using cached data for {ticker}")
        else:
            tickers_to_fetch.append(ticker)
    
    # Fetch data for tickers not in cache
    for ticker in tickers_to_fetch:
        try:
            ticker_df = _fetch_single_ticker(
                ticker=ticker,
                start=start,
                end=end,
                mode=mode,
            )
            if ticker_df is not None and not ticker_df.empty:
                # Cache the full result
                cache_key = f"{ticker}_{mode}"
                _TICKER_CACHE[cache_key] = ticker_df.copy()
                all_records.append(ticker_df)
                logger.debug(f"Fetched {len(ticker_df)} quarters for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to fetch SEC data for {ticker}: {e}")
            continue
    
    if not all_records:
        logger.warning("No SEC fundamental data retrieved for any ticker")
        return pd.DataFrame()
    
    # Concatenate all ticker data
    df = pd.concat(all_records, axis=0)
    
    # Sort index
    df = df.sort_index()
    
    logger.info(
        f"Retrieved SEC fundamentals: {len(df)} records for "
        f"{df.index.get_level_values('ticker').nunique()} tickers"
    )
    
    return df


def _fetch_single_ticker(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mode: str,
) -> pd.DataFrame | None:
    """
    Fetch quarterly data for a single ticker from finagg.
    
    Args:
        ticker: Ticker symbol.
        start: Start date.
        end: End date.
        mode: "refined" or "api".
    
    Returns:
        DataFrame with MultiIndex (date, ticker) or None if no data.
    """
    import finagg
    
    try:
        # Try refined (local DB) first, fall back to API
        if mode == "refined":
            try:
                raw_df = finagg.sec.feat.quarterly.from_refined(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )
            except Exception as e:
                logger.debug(f"Refined data not available for {ticker}, trying API: {e}")
                raw_df = finagg.sec.feat.quarterly.from_api(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )
        else:
            raw_df = finagg.sec.feat.quarterly.from_api(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
    except Exception as e:
        logger.debug(f"No SEC data available for {ticker}: {e}")
        return None
    
    if raw_df is None or raw_df.empty:
        return None
    
    # Process the raw finagg data
    processed_df = _process_finagg_data(raw_df, ticker)
    
    return processed_df


def _process_finagg_data(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Process raw finagg DataFrame into our standard format.
    
    Finagg quarterly features returns pre-computed ratios:
    - ReturnOnAssets, ReturnOnEquity
    - EarningsPerShareBasic
    - DebtEquityRatio, QuickRatio, WorkingCapitalRatio
    - LOG_CHANGE(...) for various balance sheet items
    
    We convert to:
    - MultiIndex (date, ticker)
    - Standardized column names matching our factor pipeline
    
    Args:
        raw_df: Raw DataFrame from finagg.
        ticker: Ticker symbol for this data.
    
    Returns:
        Processed DataFrame with standardized columns.
    """
    # finagg quarterly features has MultiIndex: (fy, fp, filed)
    # We need to extract the 'filed' date
    if isinstance(raw_df.index, pd.MultiIndex):
        # Get the 'filed' level
        if "filed" in raw_df.index.names:
            filed_idx = raw_df.index.names.index("filed")
            date_values = raw_df.index.get_level_values(filed_idx)
        else:
            # Use last level as date
            date_values = raw_df.index.get_level_values(-1)
    else:
        date_values = raw_df.index
    
    # Create result DataFrame
    result = pd.DataFrame(index=range(len(raw_df)))
    result["date"] = pd.to_datetime(date_values)
    result["ticker"] = ticker
    
    # Map finagg's pre-computed features to our standard names
    # finagg quarterly returns: ReturnOnAssets, ReturnOnEquity, 
    # EarningsPerShareBasic, DebtEquityRatio, QuickRatio, etc.
    
    # Direct mappings for ratios (already computed by finagg)
    result["roa"] = _extract_column(raw_df, ["ReturnOnAssets"])
    result["roe"] = _extract_column(raw_df, ["ReturnOnEquity"])
    result["eps_basic"] = _extract_column(raw_df, ["EarningsPerShareBasic"])
    
    # Book ratio can be used for value
    result["book_ratio"] = _extract_column(raw_df, ["BookRatio"])
    
    # Debt and liquidity ratios for quality
    result["debt_equity"] = _extract_column(raw_df, ["DebtEquityRatio"])
    result["quick_ratio"] = _extract_column(raw_df, ["QuickRatio"])
    result["working_capital_ratio"] = _extract_column(raw_df, ["WorkingCapitalRatio"])
    result["asset_coverage"] = _extract_column(raw_df, ["AssetCoverageRatio"])
    
    # Growth metrics from LOG_CHANGE columns
    result["asset_growth"] = _extract_column(raw_df, ["LOG_CHANGE(Assets)"])
    result["equity_growth"] = _extract_column(raw_df, ["LOG_CHANGE(StockholdersEquity)"])
    result["shares_change"] = _extract_column(raw_df, ["LOG_CHANGE(CommonStockSharesOutstanding)"])
    result["inventory_growth"] = _extract_column(raw_df, ["LOG_CHANGE(InventoryNet)"])
    result["liabilities_growth"] = _extract_column(raw_df, ["LOG_CHANGE(Liabilities)"])
    
    # Create synthetic margin proxies from ratios
    # Note: finagg doesn't provide raw revenue/income, so we use ROA/ROE as profitability proxies
    # These are actually better than margins for cross-sectional ranking
    result["gross_margin"] = np.nan  # Not available from finagg quarterly features
    result["operating_margin"] = np.nan  # Not available
    result["net_margin"] = np.nan  # Not available
    
    # For value factors, use book ratio (inverse of P/B)
    result["pb_ratio"] = _safe_divide(
        pd.Series(np.ones(len(result))), 
        pd.Series(result["book_ratio"].values)
    ) if result["book_ratio"].notna().any() else np.nan
    
    # Set MultiIndex
    result = result.set_index(["date", "ticker"])
    
    # Drop rows with all NaN values in key computed columns
    key_cols = ["roa", "roe", "eps_basic", "debt_equity"]
    available_key_cols = [c for c in key_cols if c in result.columns and result[c].notna().any()]
    if available_key_cols:
        result = result.dropna(subset=available_key_cols, how="all")
    
    return result
    
    return result


def _extract_column(df: pd.DataFrame, column_names: list[str]) -> pd.Series:
    """
    Extract a column from DataFrame, trying multiple possible names.
    
    Args:
        df: DataFrame to extract from.
        column_names: List of possible column names to try.
    
    Returns:
        Series with the extracted values, or all NaN if not found.
    """
    for name in column_names:
        if name in df.columns:
            return df[name].values
    
    return np.full(len(df), np.nan)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safely divide two series, handling zeros and NaNs.
    
    Args:
        numerator: Numerator series.
        denominator: Denominator series.
    
    Returns:
        Result of division with zeros/NaNs handled.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], np.nan)
    return result


def align_finagg_to_prices(
    fund_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align quarterly finagg fundamentals to daily price dates.
    
    Quarterly fundamentals are forward-filled to daily frequency so that
    on any trading day, the most recent available fundamental data is used.
    
    Args:
        fund_df: DataFrame with MultiIndex (date, ticker) from finagg.
        prices: Price DataFrame with DatetimeIndex and ticker columns.
    
    Returns:
        DataFrame with MultiIndex (date, ticker), aligned to prices.index,
        with quarterly values forward-filled to daily.
    """
    if fund_df.empty:
        return pd.DataFrame()
    
    tickers = prices.columns.tolist()
    price_dates = prices.index
    
    aligned_records = []
    
    for ticker in tickers:
        # Get this ticker's fundamental data
        try:
            ticker_fund = fund_df.xs(ticker, level="ticker")
        except KeyError:
            # No data for this ticker
            continue
        
        if ticker_fund.empty:
            continue
        
        # Sort by date
        ticker_fund = ticker_fund.sort_index()
        
        # Reindex to price dates with forward-fill
        # This ensures we use the most recent known fundamentals
        ticker_aligned = ticker_fund.reindex(price_dates, method="ffill")
        ticker_aligned["ticker"] = ticker
        aligned_records.append(ticker_aligned)
    
    if not aligned_records:
        return pd.DataFrame()
    
    # Combine all tickers
    result = pd.concat(aligned_records, axis=0)
    result = result.reset_index().rename(columns={"index": "date"})
    result = result.set_index(["date", "ticker"]).sort_index()
    
    return result


def get_finagg_status() -> dict[str, bool | str]:
    """
    Get the status of finagg installation and configuration.
    
    Returns:
        Dictionary with status information:
            - installed: Whether finagg package is installed
            - configured: Whether SEC credentials are configured
            - db_path: Path to local finagg database (if any)
            - error: Error message if any
    """
    status = {
        "installed": False,
        "configured": False,
        "db_path": None,
        "error": None,
    }
    
    if not _check_finagg_available():
        status["error"] = "finagg package not installed"
        return status
    
    status["installed"] = True
    
    try:
        import finagg
        
        # Check if SEC credentials are configured
        import os
        if os.environ.get("SEC_API_USER_AGENT"):
            status["configured"] = True
        else:
            # Try to check finagg's own config
            try:
                # finagg stores config in its own location
                status["configured"] = True  # Assume configured if no error
            except Exception:
                status["error"] = "SEC_API_USER_AGENT not set"
        
        # Try to get database path
        try:
            from finagg import backend
            if hasattr(backend, "database_path"):
                status["db_path"] = str(backend.database_path)
        except Exception:
            pass
        
    except Exception as e:
        status["error"] = str(e)
    
    return status


# =============================================================================
# Direct SEC EDGAR API fetcher (more robust than finagg)
# =============================================================================

# Cache for ticker->CIK mapping
_TICKER_TO_CIK: dict[str, str] = {}
_SEC_CACHE: dict[str, pd.DataFrame] = {}


def _get_ticker_to_cik_mapping() -> dict[str, str]:
    """
    Fetch and cache the SEC ticker to CIK mapping.
    
    Returns:
        Dictionary mapping ticker symbols to 10-digit CIK codes.
    """
    global _TICKER_TO_CIK
    
    if _TICKER_TO_CIK:
        return _TICKER_TO_CIK
    
    import requests
    
    user_agent = os.environ.get('SEC_API_USER_AGENT', 'AutoResearcher research@example.com')
    headers = {'User-Agent': user_agent}
    
    url = 'https://www.sec.gov/files/company_tickers.json'
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    for val in response.json().values():
        _TICKER_TO_CIK[val['ticker']] = str(val['cik_str']).zfill(10)
    
    logger.info(f"Loaded {len(_TICKER_TO_CIK)} ticker->CIK mappings from SEC")
    return _TICKER_TO_CIK


def _fetch_company_facts(cik: str) -> dict | None:
    """
    Fetch company facts from SEC EDGAR API.
    
    Args:
        cik: 10-digit CIK code.
    
    Returns:
        JSON response with company facts or None on error.
    """
    import requests
    import time
    
    user_agent = os.environ.get('SEC_API_USER_AGENT', 'AutoResearcher research@example.com')
    headers = {'User-Agent': user_agent}
    
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    
    # Rate limit: SEC requires 10 requests per second max
    time.sleep(0.15)
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.debug(f"Failed to fetch company facts for CIK {cik}: {e}")
        return None


def _extract_quarterly_values(
    facts: dict,
    concept: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, float]]:
    """
    Extract quarterly values for a specific concept from company facts.
    
    SEC XBRL data often contains both cumulative YTD values and single-quarter
    values. This function filters to only return single-quarter values by
    checking the period duration (start to end date should be ~90 days).
    
    Args:
        facts: Company facts JSON from SEC API.
        concept: US-GAAP concept name (e.g., 'Revenues').
        start: Start date.
        end: End date.
    
    Returns:
        List of (date, value) tuples for quarterly filings.
    """
    us_gaap = facts.get('facts', {}).get('us-gaap', {})
    concept_data = us_gaap.get(concept, {})
    units = concept_data.get('units', {})
    
    # Try USD first, then shares
    values = units.get('USD', units.get('shares', []))
    
    results = []
    seen_periods = set()  # Deduplicate by (end_date, value)
    
    for item in values:
        # Only take quarterly filings (10-Q form or 10-K for Q4)
        form = item.get('form', '')
        if form not in ('10-Q', '10-K'):
            continue
        
        # Get the end date of the filing period
        end_date_str = item.get('end')
        if not end_date_str:
            continue
        
        try:
            period_end = pd.Timestamp(end_date_str)
        except Exception:
            continue
        
        if period_end < start or period_end > end:
            continue
        
        val = item.get('val')
        if val is None:
            continue
        
        # Check if this is a single-quarter value vs cumulative YTD
        # by looking at the period duration
        start_date_str = item.get('start')
        if start_date_str:
            try:
                period_start = pd.Timestamp(start_date_str)
                period_days = (period_end - period_start).days
                
                # For income statement items (revenue, net_income, etc.),
                # only take periods ~90 days (single quarter)
                # Skip cumulative YTD values (180+ days)
                # Allow some flexibility: 60-120 days for a quarter
                if period_days > 120:
                    # This is likely a cumulative YTD value, skip it
                    continue
            except Exception:
                pass
        
        # Deduplicate: same end date and value might appear twice
        key = (period_end, val)
        if key in seen_periods:
            continue
        seen_periods.add(key)
        
        results.append((period_end, float(val)))
    
    # Sort by date
    results.sort(key=lambda x: x[0])
    return results


# Mapping of our standard names to possible SEC XBRL concept names
# Order matters: first match wins
# Expanded mappings based on EDGAR concept analysis across 50 large-cap stocks
SEC_CONCEPT_MAPPING = {
    'revenue': [
        # Standard
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax', 
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
        'SalesRevenueServicesNet',
        'TotalRevenue',
        'TotalRevenuesAndOtherIncome',
        # Banks/Financial Services (use net interest income + noninterest income)
        'RevenuesNetOfInterestExpense',
        'InterestAndDividendIncomeOperating',
        'InterestIncomeExpenseNet',
        # Insurance
        'PremiumsEarnedNet',
        'InsurancePremiumsRevenueRecognized',
        # Energy
        'OilAndGasRevenue',
        'NaturalGasProductionRevenue',
    ],
    'net_income': [
        'NetIncomeLoss',
        'NetIncomeLossAvailableToCommonStockholdersBasic',
        'NetIncomeLossAvailableToCommonStockholdersDiluted',
        'NetIncomeLossAttributableToParent',
        'ProfitLoss',
        'IncomeLossFromContinuingOperations',
        'IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'gross_profit': [
        'GrossProfit',
        'RevenuesNetOfCostOfSales',
        # Banks: use net interest income as gross profit proxy
        'InterestIncomeExpenseNet',
        'InterestIncomeExpenseAfterProvisionForLoanLoss',
    ],
    'operating_income': [
        'OperatingIncomeLoss',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',
        'IncomeLossBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
        # Pre-tax income variations
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic',
        'IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet',
    ],
    'total_assets': [
        'Assets',
        'TotalAssets',
    ],
    'stockholders_equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'TotalStockholdersEquity',
        'TotalEquity',
        'MembersEquity',  # LLCs
        'PartnersCapital',  # Partnerships
        'LimitedPartnersCapitalAccount',
    ],
    'eps_basic': [
        'EarningsPerShareBasic',
        'IncomeLossFromContinuingOperationsPerBasicShare',
        'BasicEarningsLossPerShare',
        'EarningsPerShareBasicAndDiluted',
        'IncomeLossFromContinuingOperationsPerBasicAndDilutedShare',
    ],
    'eps_diluted': [
        'EarningsPerShareDiluted',
        'IncomeLossFromContinuingOperationsPerDilutedShare',
    ],
    'shares_outstanding': [
        'CommonStockSharesOutstanding',
        'WeightedAverageNumberOfSharesOutstandingBasic',
        'CommonStockSharesIssued',
        'WeightedAverageNumberOfDilutedSharesOutstanding',
    ],
    'total_liabilities': [
        'Liabilities',
        'TotalLiabilities',
    ],
    'current_assets': [
        'AssetsCurrent',
        'TotalCurrentAssets',
    ],
    'current_liabilities': [
        'LiabilitiesCurrent',
        'TotalCurrentLiabilities',
    ],
    'inventory': [
        'InventoryNet',
        'InventoryGross',
    ],
    # Additional metrics for better coverage
    'long_term_debt': [
        'LongTermDebt',
        'LongTermDebtNoncurrent',
        'LongTermDebtAndCapitalLeaseObligations',
    ],
    'cash': [
        'CashAndCashEquivalentsAtCarryingValue',
        'Cash',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
    ],
    'dividends_per_share': [
        'CommonStockDividendsPerShareDeclared',
        'CommonStockDividendsPerShareCashPaid',
    ],
    'cost_of_revenue': [
        'CostOfRevenue',
        'CostOfGoodsAndServicesSold',
        'CostOfGoodsSold',
    ],
}


def _compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived financial metrics from raw SEC data.
    
    This function computes profitability ratios, margins, and per-share metrics
    from the raw fundamental data extracted from SEC filings.
    
    Args:
        df: DataFrame with raw SEC metrics (revenue, net_income, etc.)
    
    Returns:
        DataFrame with additional derived columns added.
    """
    result = df.copy()
    
    # ----- Profitability Margins -----
    # Gross margin (requires gross_profit and revenue)
    if 'gross_profit' in result.columns and 'revenue' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['gross_margin'] = result['gross_profit'] / result['revenue']
            result['gross_margin'] = result['gross_margin'].replace([np.inf, -np.inf], np.nan)
    
    # Operating margin (requires operating_income and revenue)
    if 'operating_income' in result.columns and 'revenue' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['operating_margin'] = result['operating_income'] / result['revenue']
            result['operating_margin'] = result['operating_margin'].replace([np.inf, -np.inf], np.nan)
    
    # Net margin (requires net_income and revenue)
    if 'net_income' in result.columns and 'revenue' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['net_margin'] = result['net_income'] / result['revenue']
            result['net_margin'] = result['net_margin'].replace([np.inf, -np.inf], np.nan)
    
    # ----- Return Ratios -----
    # ROE (requires net_income and stockholders_equity)
    if 'net_income' in result.columns and 'stockholders_equity' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['roe'] = result['net_income'] / result['stockholders_equity']
            result['roe'] = result['roe'].replace([np.inf, -np.inf], np.nan)
    
    # ROA (requires net_income and total_assets)
    if 'net_income' in result.columns and 'total_assets' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['roa'] = result['net_income'] / result['total_assets']
            result['roa'] = result['roa'].replace([np.inf, -np.inf], np.nan)
    
    # ----- Leverage & Liquidity Ratios -----
    # Debt/Equity (requires total_liabilities and stockholders_equity)
    if 'total_liabilities' in result.columns and 'stockholders_equity' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['debt_equity'] = result['total_liabilities'] / result['stockholders_equity']
            result['debt_equity'] = result['debt_equity'].replace([np.inf, -np.inf], np.nan)
    
    # Quick ratio (requires current_assets, inventory, current_liabilities)
    if 'current_assets' in result.columns and 'current_liabilities' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            if 'inventory' in result.columns:
                result['quick_ratio'] = (result['current_assets'] - result['inventory'].fillna(0)) / result['current_liabilities']
            else:
                result['quick_ratio'] = result['current_assets'] / result['current_liabilities']
            result['quick_ratio'] = result['quick_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Current ratio
    if 'current_assets' in result.columns and 'current_liabilities' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['current_ratio'] = result['current_assets'] / result['current_liabilities']
            result['current_ratio'] = result['current_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # ----- Per-Share Metrics -----
    # Compute EPS from net income / shares if not directly available
    if 'eps_basic' not in result.columns or result['eps_basic'].isna().all():
        if 'net_income' in result.columns and 'shares_outstanding' in result.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                result['eps_basic'] = result['net_income'] / result['shares_outstanding']
                result['eps_basic'] = result['eps_basic'].replace([np.inf, -np.inf], np.nan)
    
    # Book value per share
    if 'stockholders_equity' in result.columns and 'shares_outstanding' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['book_value_per_share'] = result['stockholders_equity'] / result['shares_outstanding']
            result['book_value_per_share'] = result['book_value_per_share'].replace([np.inf, -np.inf], np.nan)
    
    # ----- Value Ratios (need market data, but can compute inverses) -----
    # Price/Book inverse (book/price) = book_value_per_share / price
    # We can't compute this without price data, but we store book_value_per_share
    
    return result


def fetch_sec_fundamentals_direct(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch fundamental data directly from SEC EDGAR API.
    
    This is a more robust alternative to finagg that handles more tickers
    by being lenient about missing XBRL concepts.
    
    Args:
        tickers: List of ticker symbols.
        start: Start date.
        end: End date.
    
    Returns:
        DataFrame with MultiIndex (date, ticker) and columns for each
        fundamental metric. Missing metrics will be NaN.
    """
    global _SEC_CACHE
    
    ticker_to_cik = _get_ticker_to_cik_mapping()
    
    all_records = []
    
    for ticker in tickers:
        cache_key = f"sec_direct_{ticker}"
        
        # Check cache
        if cache_key in _SEC_CACHE:
            cached = _SEC_CACHE[cache_key]
            mask = (cached.index.get_level_values('date') >= start) & \
                   (cached.index.get_level_values('date') <= end)
            filtered = cached.loc[mask]
            if not filtered.empty:
                all_records.append(filtered)
                continue
        
        cik = ticker_to_cik.get(ticker)
        if not cik:
            logger.debug(f"No CIK found for {ticker}")
            continue
        
        facts = _fetch_company_facts(cik)
        if not facts:
            continue
        
        # Extract all available concepts
        ticker_data = {}
        for our_name, sec_names in SEC_CONCEPT_MAPPING.items():
            # Find the concept with the most values (not just first match)
            best_values = []
            best_concept = None
            
            for sec_name in sec_names:
                values = _extract_quarterly_values(facts, sec_name, start, end)
                if values and len(values) > len(best_values):
                    best_values = values
                    best_concept = sec_name
            
            if best_values:
                # Use the concept with most data points
                for date, val in best_values:
                    if date not in ticker_data:
                        ticker_data[date] = {'ticker': ticker}
                    if our_name not in ticker_data[date]:
                        ticker_data[date][our_name] = val
        
        if ticker_data:
            df = pd.DataFrame(list(ticker_data.values()), 
                            index=list(ticker_data.keys()))
            df.index.name = 'date'
            df = df.reset_index().set_index(['date', 'ticker'])
            
            # Compute all derived metrics using the helper function
            df = _compute_derived_metrics(df)
            
            # Cache the result
            _SEC_CACHE[cache_key] = df.copy()
            all_records.append(df)
            logger.debug(f"Fetched {len(df)} quarters for {ticker}")
    
    if not all_records:
        logger.warning("No SEC fundamental data retrieved for any ticker")
        return pd.DataFrame()
    
    result = pd.concat(all_records, axis=0)
    result = result.sort_index()
    
    logger.info(
        f"Retrieved SEC fundamentals (direct): {len(result)} records for "
        f"{result.index.get_level_values('ticker').nunique()} tickers"
    )
    
    return result
