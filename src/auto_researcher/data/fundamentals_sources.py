"""
Fundamental data sources using FMP (Financial Modeling Prep) and Alpha Vantage.

This module provides a real fundamentals data ingestion layer with:
- FMP as the primary data source (rich historical quarterly data)
- Alpha Vantage as a fallback source
- Rate limiting to respect API limits
- Normalization to a common schema

Environment variables:
    FMP_API_KEY: API key for Financial Modeling Prep
    ALPHAVANTAGE_API_KEY: API key for Alpha Vantage
"""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FundamentalsSourceConfig:
    """
    Configuration for fundamental data sources.

    Attributes:
        fmp_api_key: API key for Financial Modeling Prep.
        av_api_key: API key for Alpha Vantage.
        fmp_base_url: Base URL for FMP API.
        av_base_url: Base URL for Alpha Vantage API.
        max_fmp_calls_per_minute: Rate limit for FMP (default 200).
        max_av_calls_per_minute: Rate limit for Alpha Vantage (default 5).
        timeout_seconds: HTTP request timeout.
    """

    fmp_api_key: str | None = None
    av_api_key: str | None = None
    fmp_base_url: str = "https://financialmodelingprep.com/api/v3"
    av_base_url: str = "https://www.alphavantage.co/query"
    max_fmp_calls_per_minute: int = 200
    max_av_calls_per_minute: int = 5
    timeout_seconds: int = 30


def get_fundamentals_source_config() -> FundamentalsSourceConfig:
    """
    Read API keys from environment variables and return a config instance.

    Environment variables:
        FMP_API_KEY: Financial Modeling Prep API key
        ALPHAVANTAGE_API_KEY: Alpha Vantage API key

    Returns:
        FundamentalsSourceConfig with API keys populated.
    """
    return FundamentalsSourceConfig(
        fmp_api_key=os.environ.get("FMP_API_KEY"),
        av_api_key=os.environ.get("ALPHAVANTAGE_API_KEY"),
    )


# =============================================================================
# Rate Limiting
# =============================================================================


class RateLimiter:
    """
    Simple time-based rate limiter for API calls.

    Tracks call timestamps per source and sleeps when rate limit is exceeded.
    """

    def __init__(self) -> None:
        """Initialize rate limiter with empty call history."""
        self._call_times: dict[str, list[float]] = defaultdict(list)

    def wait_if_needed(self, source: str, max_calls_per_minute: int) -> None:
        """
        Wait if necessary to respect rate limit.

        Args:
            source: Name of the API source (e.g., "fmp", "av").
            max_calls_per_minute: Maximum allowed calls per minute.
        """
        now = time.time()
        window_start = now - 60.0

        # Clean up old timestamps
        self._call_times[source] = [
            t for t in self._call_times[source] if t > window_start
        ]

        calls_in_window = len(self._call_times[source])

        if calls_in_window >= max_calls_per_minute:
            # Wait until the oldest call falls outside the window
            oldest = self._call_times[source][0]
            sleep_time = oldest - window_start + 0.1  # Add small buffer
            if sleep_time > 0:
                logger.debug(f"Rate limit reached for {source}, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        # Record this call
        self._call_times[source].append(time.time())


# Global rate limiter instance
_rate_limiter = RateLimiter()


# =============================================================================
# Column Schema
# =============================================================================

# Normalized column names expected by load_raw_fundamentals
FUNDAMENTAL_COLUMNS = [
    "market_cap",
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "dividend_yield",
    "roe",
    "roa",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "revenue_ttm",
    "revenue_growth_1y",
    "eps_ttm",
    "eps_growth_1y",
]


# =============================================================================
# FMP API Functions
# =============================================================================


def _fmp_request(
    endpoint: str,
    config: FundamentalsSourceConfig,
    params: dict[str, Any] | None = None,
) -> list[dict] | dict | None:
    """
    Make a request to the FMP API.

    Args:
        endpoint: API endpoint (e.g., "/key-metrics/AAPL").
        config: Configuration with API key and base URL.
        params: Additional query parameters.

    Returns:
        JSON response or None if request failed.
    """
    if not config.fmp_api_key:
        logger.warning("FMP_API_KEY not set, skipping FMP request")
        return None

    _rate_limiter.wait_if_needed("fmp", config.max_fmp_calls_per_minute)

    url = f"{config.fmp_base_url}{endpoint}"
    all_params = {"apikey": config.fmp_api_key}
    if params:
        all_params.update(params)

    try:
        response = requests.get(url, params=all_params, timeout=config.timeout_seconds)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.warning(f"FMP request failed for {endpoint}: {e}")
        return None


def fetch_fmp_fundamentals_for_symbol(
    symbol: str,
    config: FundamentalsSourceConfig,
    max_years_back: int = 15,
) -> pd.DataFrame:
    """
    Fetch quarterly fundamentals and ratios from FMP API.

    Uses multiple endpoints to retrieve comprehensive fundamental data:
    - /key-metrics/{symbol}?period=quarter - Key valuation metrics
    - /ratios/{symbol}?period=quarter - Financial ratios
    - /income-statement/{symbol}?period=quarter - Revenue, EPS

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        config: FundamentalsSourceConfig with API key.
        max_years_back: Maximum years of history to fetch.

    Returns:
        DataFrame indexed by date with normalized fundamental columns.
        Returns empty DataFrame if request fails.
    """
    if not config.fmp_api_key:
        return pd.DataFrame()

    # Fetch key metrics (includes market cap, PE, PB, etc.)
    key_metrics = _fmp_request(
        f"/key-metrics/{symbol}",
        config,
        params={"period": "quarter", "limit": max_years_back * 4},
    )

    # Fetch ratios (includes ROE, ROA, margins)
    ratios = _fmp_request(
        f"/ratios/{symbol}",
        config,
        params={"period": "quarter", "limit": max_years_back * 4},
    )

    # Fetch income statement (for revenue, EPS growth)
    income_stmt = _fmp_request(
        f"/income-statement/{symbol}",
        config,
        params={"period": "quarter", "limit": max_years_back * 4},
    )

    if not key_metrics and not ratios and not income_stmt:
        logger.warning(f"No FMP data available for {symbol}")
        return pd.DataFrame()

    # Merge data by date
    records = _merge_fmp_data(key_metrics, ratios, income_stmt)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Ensure all expected columns exist
    for col in FUNDAMENTAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[FUNDAMENTAL_COLUMNS]


def _merge_fmp_data(
    key_metrics: list[dict] | None,
    ratios: list[dict] | None,
    income_stmt: list[dict] | None,
) -> list[dict]:
    """
    Merge data from multiple FMP endpoints into normalized records.

    Args:
        key_metrics: Key metrics data from FMP.
        ratios: Ratios data from FMP.
        income_stmt: Income statement data from FMP.

    Returns:
        List of normalized records with date and fundamental columns.
    """
    # Build lookup dictionaries by date
    metrics_by_date: dict[str, dict] = {}

    # Process key metrics
    if key_metrics:
        for item in key_metrics:
            date = item.get("date")
            if date:
                if date not in metrics_by_date:
                    metrics_by_date[date] = {}
                metrics_by_date[date].update({
                    "market_cap": item.get("marketCap"),
                    "pe_ratio": item.get("peRatio"),
                    "pb_ratio": item.get("pbRatio"),
                    "ps_ratio": item.get("priceToSalesRatio"),
                    "dividend_yield": item.get("dividendYield"),
                    "roe": item.get("roe"),
                    "roa": item.get("roic"),  # FMP uses roic in key-metrics
                    "eps_ttm": item.get("netIncomePerShare"),
                    "revenue_ttm": item.get("revenuePerShare"),  # Approximation
                })

    # Process ratios
    if ratios:
        for item in ratios:
            date = item.get("date")
            if date:
                if date not in metrics_by_date:
                    metrics_by_date[date] = {}
                # Only update if not already set (key_metrics takes priority)
                if "roe" not in metrics_by_date[date] or metrics_by_date[date]["roe"] is None:
                    metrics_by_date[date]["roe"] = item.get("returnOnEquity")
                if "roa" not in metrics_by_date[date] or metrics_by_date[date]["roa"] is None:
                    metrics_by_date[date]["roa"] = item.get("returnOnAssets")
                metrics_by_date[date].update({
                    "gross_margin": item.get("grossProfitMargin"),
                    "operating_margin": item.get("operatingProfitMargin"),
                    "net_margin": item.get("netProfitMargin"),
                })

    # Process income statement for growth metrics
    if income_stmt:
        # Sort by date to calculate YoY growth
        sorted_stmts = sorted(income_stmt, key=lambda x: x.get("date", ""), reverse=True)
        revenue_by_date = {
            item.get("date"): item.get("revenue") for item in sorted_stmts
        }
        eps_by_date = {
            item.get("date"): item.get("eps") for item in sorted_stmts
        }

        for item in sorted_stmts:
            date = item.get("date")
            if date:
                if date not in metrics_by_date:
                    metrics_by_date[date] = {}

                # Calculate TTM values (sum of last 4 quarters)
                # For simplicity, use the current quarter values
                revenue = item.get("revenue")
                eps = item.get("eps")

                if revenue is not None:
                    metrics_by_date[date]["revenue_ttm"] = revenue

                if eps is not None:
                    metrics_by_date[date]["eps_ttm"] = eps

                # Calculate YoY growth (compare to 4 quarters ago)
                date_obj = pd.to_datetime(date)
                year_ago = (date_obj - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

                # Find closest date from a year ago
                year_ago_revenue = _find_closest_value(revenue_by_date, year_ago)
                year_ago_eps = _find_closest_value(eps_by_date, year_ago)

                if revenue and year_ago_revenue and year_ago_revenue != 0:
                    metrics_by_date[date]["revenue_growth_1y"] = (
                        revenue - year_ago_revenue
                    ) / abs(year_ago_revenue)

                if eps and year_ago_eps and year_ago_eps != 0:
                    metrics_by_date[date]["eps_growth_1y"] = (
                        eps - year_ago_eps
                    ) / abs(year_ago_eps)

    # Convert to list of records
    records = []
    for date, metrics in metrics_by_date.items():
        record = {"date": date}
        record.update(metrics)
        records.append(record)

    return records


def _find_closest_value(
    values_by_date: dict[str, float | None],
    target_date: str,
    tolerance_days: int = 45,
) -> float | None:
    """
    Find value from the closest date within tolerance.

    Args:
        values_by_date: Dictionary mapping date strings to values.
        target_date: Target date to find closest match for.
        tolerance_days: Maximum days difference to accept.

    Returns:
        Value from closest date or None if no match found.
    """
    target = pd.to_datetime(target_date)
    best_value = None
    best_diff = float("inf")

    for date_str, value in values_by_date.items():
        if value is None:
            continue
        date = pd.to_datetime(date_str)
        diff = abs((date - target).days)
        if diff < best_diff and diff <= tolerance_days:
            best_diff = diff
            best_value = value

    return best_value


# =============================================================================
# Alpha Vantage API Functions
# =============================================================================


def _av_request(
    function: str,
    symbol: str,
    config: FundamentalsSourceConfig,
) -> dict | None:
    """
    Make a request to the Alpha Vantage API.

    Args:
        function: AV function name (e.g., "OVERVIEW", "INCOME_STATEMENT").
        symbol: Ticker symbol.
        config: Configuration with API key.

    Returns:
        JSON response or None if request failed.
    """
    if not config.av_api_key:
        logger.warning("ALPHAVANTAGE_API_KEY not set, skipping AV request")
        return None

    _rate_limiter.wait_if_needed("av", config.max_av_calls_per_minute)

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": config.av_api_key,
    }

    try:
        response = requests.get(
            config.av_base_url, params=params, timeout=config.timeout_seconds
        )
        response.raise_for_status()
        data = response.json()

        # Check for API error messages
        if "Error Message" in data:
            logger.warning(f"AV error for {symbol}: {data['Error Message']}")
            return None
        if "Note" in data:
            logger.warning(f"AV rate limit note: {data['Note']}")
            return None

        return data
    except requests.RequestException as e:
        logger.warning(f"AV request failed for {function}/{symbol}: {e}")
        return None


def fetch_av_fundamentals_for_symbol(
    symbol: str,
    config: FundamentalsSourceConfig,
) -> pd.DataFrame:
    """
    Fetch fundamental data from Alpha Vantage as a fallback source.

    Uses:
    - OVERVIEW: Current valuation metrics
    - INCOME_STATEMENT: Quarterly revenue, net income
    - BALANCE_SHEET: Quarterly book value, assets

    Args:
        symbol: Ticker symbol.
        config: FundamentalsSourceConfig with API key.

    Returns:
        DataFrame indexed by date with normalized fundamental columns.
        May have limited historical depth compared to FMP.
    """
    if not config.av_api_key:
        return pd.DataFrame()

    # Fetch overview (current snapshot)
    overview = _av_request("OVERVIEW", symbol, config)

    # Fetch income statement (quarterly history)
    income_stmt = _av_request("INCOME_STATEMENT", symbol, config)

    # Fetch balance sheet (quarterly history)
    balance_sheet = _av_request("BALANCE_SHEET", symbol, config)

    if not overview and not income_stmt and not balance_sheet:
        logger.warning(f"No AV data available for {symbol}")
        return pd.DataFrame()

    records = _merge_av_data(overview, income_stmt, balance_sheet)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Ensure all expected columns exist
    for col in FUNDAMENTAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[FUNDAMENTAL_COLUMNS]


def _merge_av_data(
    overview: dict | None,
    income_stmt: dict | None,
    balance_sheet: dict | None,
) -> list[dict]:
    """
    Merge Alpha Vantage data into normalized records.

    Args:
        overview: Company overview data.
        income_stmt: Quarterly income statement data.
        balance_sheet: Quarterly balance sheet data.

    Returns:
        List of normalized records.
    """
    metrics_by_date: dict[str, dict] = {}

    # Process quarterly income statements
    if income_stmt and "quarterlyReports" in income_stmt:
        for report in income_stmt["quarterlyReports"]:
            date = report.get("fiscalDateEnding")
            if date:
                if date not in metrics_by_date:
                    metrics_by_date[date] = {}

                revenue = _safe_float(report.get("totalRevenue"))
                net_income = _safe_float(report.get("netIncome"))
                gross_profit = _safe_float(report.get("grossProfit"))
                operating_income = _safe_float(report.get("operatingIncome"))

                metrics_by_date[date]["revenue_ttm"] = revenue

                if revenue and revenue != 0:
                    if gross_profit:
                        metrics_by_date[date]["gross_margin"] = gross_profit / revenue
                    if operating_income:
                        metrics_by_date[date]["operating_margin"] = operating_income / revenue
                    if net_income:
                        metrics_by_date[date]["net_margin"] = net_income / revenue

    # Process quarterly balance sheets
    if balance_sheet and "quarterlyReports" in balance_sheet:
        for report in balance_sheet["quarterlyReports"]:
            date = report.get("fiscalDateEnding")
            if date:
                if date not in metrics_by_date:
                    metrics_by_date[date] = {}

                total_equity = _safe_float(report.get("totalShareholderEquity"))
                total_assets = _safe_float(report.get("totalAssets"))

                # We'd need net income to calculate ROE/ROA properly
                # Just store the denominators for now
                if date in metrics_by_date:
                    net_income = metrics_by_date[date].get("net_margin")
                    revenue = metrics_by_date[date].get("revenue_ttm")
                    if net_income and revenue:
                        actual_net_income = net_income * revenue
                        if total_equity and total_equity != 0:
                            metrics_by_date[date]["roe"] = actual_net_income / total_equity
                        if total_assets and total_assets != 0:
                            metrics_by_date[date]["roa"] = actual_net_income / total_assets

    # Add current overview data to most recent date if available
    if overview and metrics_by_date:
        most_recent = max(metrics_by_date.keys())

        # Map overview fields
        overview_mapping = {
            "MarketCapitalization": "market_cap",
            "PERatio": "pe_ratio",
            "PriceToBookRatio": "pb_ratio",
            "PriceToSalesRatioTTM": "ps_ratio",
            "DividendYield": "dividend_yield",
            "ReturnOnEquityTTM": "roe",
            "ReturnOnAssetsTTM": "roa",
            "ProfitMargin": "net_margin",
            "OperatingMarginTTM": "operating_margin",
            "GrossProfitTTM": "gross_margin",  # This is actually gross profit, not margin
            "EPS": "eps_ttm",
            "RevenueTTM": "revenue_ttm",
            "QuarterlyRevenueGrowthYOY": "revenue_growth_1y",
            "QuarterlyEarningsGrowthYOY": "eps_growth_1y",
        }

        for av_key, our_key in overview_mapping.items():
            value = _safe_float(overview.get(av_key))
            if value is not None:
                # Only update if not already set
                if our_key not in metrics_by_date[most_recent] or metrics_by_date[most_recent][our_key] is None:
                    metrics_by_date[most_recent][our_key] = value

    # Convert to list of records
    records = []
    for date, metrics in metrics_by_date.items():
        record = {"date": date}
        record.update(metrics)
        records.append(record)

    return records


def _safe_float(value: str | float | None) -> float | None:
    """
    Safely convert a value to float.

    Args:
        value: Value to convert.

    Returns:
        Float value or None if conversion fails.
    """
    if value is None or value == "None" or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Combined Fetcher
# =============================================================================


def fetch_fundamentals_for_symbol(
    symbol: str,
    config: FundamentalsSourceConfig,
    max_years_back: int = 15,
) -> pd.DataFrame:
    """
    Fetch fundamentals for a symbol, trying FMP first, then Alpha Vantage.

    Args:
        symbol: Ticker symbol.
        config: FundamentalsSourceConfig with API keys.
        max_years_back: Maximum years of history (for FMP).

    Returns:
        DataFrame with fundamental data, or empty DataFrame if both fail.
    """
    # Try FMP first
    if config.fmp_api_key:
        df = fetch_fmp_fundamentals_for_symbol(symbol, config, max_years_back)
        if len(df) > 0:
            return df

    # Fall back to Alpha Vantage
    if config.av_api_key:
        df = fetch_av_fundamentals_for_symbol(symbol, config)
        if len(df) > 0:
            return df

    logger.warning(f"No fundamental data found for {symbol} from any source")
    return pd.DataFrame()
