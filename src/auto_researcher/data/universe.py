"""
Universe management for ticker selection.

This module provides functions to define and retrieve the investment universe.
"""

from auto_researcher.config import DEFAULT_UNIVERSE, ResearchConfig


def get_universe(config: ResearchConfig | None = None) -> list[str]:
    """
    Get the list of tickers in the investment universe.

    Args:
        config: Optional ResearchConfig to override defaults.

    Returns:
        List of ticker symbols.

    Examples:
        >>> tickers = get_universe()
        >>> len(tickers) > 0
        True
    """
    if config is not None:
        return list(config.universe)
    return DEFAULT_UNIVERSE.copy()


def get_sp500_universe(max_tickers: int | None = None) -> list[str]:
    """
    Get the S&P 500 ticker universe.

    Returns a curated list of ~100 liquid S&P 500 constituents.
    These are selected for liquidity and data availability.

    Args:
        max_tickers: If provided, truncate the list to this many symbols.

    Returns:
        List of S&P 500 ticker symbols.
    """
    # Curated list of ~100 liquid S&P 500 stocks by market cap
    # Excludes BRK-A (expensive), includes major sectors
    sp500_tickers = [
        # Technology (25)
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ADBE", "CRM", "CSCO", "ORCL",
        "AMD", "INTC", "TXN", "QCOM", "IBM", "NOW", "AMAT", "MU", "LRCX", "ADI",
        "SNPS", "CDNS", "KLAC", "MCHP", "FTNT",
        # Financials (15)
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
        "PNC", "TFC", "COF", "CME", "ICE",
        # Healthcare (15)
        "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "ISRG",
        # Consumer Discretionary (10)
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR",
        # Consumer Staples (8)
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL",
        # Industrials (10)
        "CAT", "HON", "UPS", "UNP", "RTX", "BA", "DE", "LMT", "GE", "MMM",
        # Energy (5)
        "XOM", "CVX", "COP", "SLB", "EOG",
        # Materials (4)
        "LIN", "APD", "SHW", "FCX",
        # Utilities (3)
        "NEE", "DUK", "SO",
        # Real Estate (3)
        "PLD", "AMT", "EQIX",
        # Communication Services (5)
        "GOOG", "NFLX", "DIS", "CMCSA", "VZ",
    ]
    
    if max_tickers is not None:
        return sp500_tickers[:max_tickers]
    return sp500_tickers


def get_large_cap_universe() -> list[str]:
    """
    Return a medium-sized universe of ~50 US large caps.
    
    Suitable for fast experimentation while being more representative
    than the tiny 4-stock tech universe.
    
    Includes major names from each sector for diversification.

    Returns:
        List of ~50 liquid large-cap ticker symbols.
    """
    return [
        # Technology (12)
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ADBE", "CRM", "CSCO", "AMD",
        "INTC", "TXN",
        # Financials (8)
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C",
        # Healthcare (8)
        "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT",
        # Consumer Discretionary (5)
        "AMZN", "TSLA", "HD", "MCD", "NKE",
        # Consumer Staples (5)
        "PG", "KO", "PEP", "COST", "WMT",
        # Industrials (5)
        "CAT", "HON", "UPS", "RTX", "BA",
        # Energy (3)
        "XOM", "CVX", "COP",
        # Communication/Other (4)
        "NFLX", "DIS", "VZ", "NEE",
    ]


def filter_universe(
    tickers: list[str],
    exclude: list[str] | None = None,
    include_only: list[str] | None = None,
) -> list[str]:
    """
    Filter the universe by exclusion or inclusion lists.

    Args:
        tickers: Base list of tickers.
        exclude: Tickers to exclude from the universe.
        include_only: If provided, only include these tickers.

    Returns:
        Filtered list of tickers.
    """
    if include_only is not None:
        tickers = [t for t in tickers if t in include_only]
    if exclude is not None:
        tickers = [t for t in tickers if t not in exclude]
    return tickers
