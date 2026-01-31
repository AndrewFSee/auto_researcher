"""
Fundamentals Coverage Utilities.

This module provides helpers for computing and filtering based on fundamental
data coverage across tickers and time periods. Used to identify tickers with
sufficient fundamental data quality for modeling.

Usage:
    from auto_researcher.features.coverage import (
        compute_fundamentals_coverage,
        select_good_fundamentals_tickers,
        identify_fundamental_columns,
        FUNDAMENTAL_PREFIXES,
    )
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Fundamental column prefixes to check (consistent with check_fundamentals_coverage.py)
FUNDAMENTAL_PREFIXES = ("quality_", "growth_", "prof_", "value_", "size_")


def identify_fundamental_columns(
    df: pd.DataFrame,
    prefixes: tuple[str, ...] = FUNDAMENTAL_PREFIXES,
) -> list[str]:
    """
    Identify fundamental feature columns by prefix.
    
    Args:
        df: Feature DataFrame with MultiIndex columns (ticker, feature) or flat columns.
        prefixes: Tuple of prefixes that identify fundamental columns.
    
    Returns:
        List of fundamental feature names (unique, sorted).
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Get unique feature names from level 1
        all_features = df.columns.get_level_values(1).unique().tolist()
    else:
        all_features = df.columns.tolist()
    
    fundamental_cols = sorted([
        col for col in all_features
        if any(col.startswith(prefix) for prefix in prefixes)
    ])
    return fundamental_cols


def compute_fundamentals_coverage(
    df: pd.DataFrame,
    fundamental_cols: list[str] | None = None,
    coverage_freq: str = "Y",
    benchmark: str | None = "SPY",
) -> pd.DataFrame:
    """
    Compute fundamental data coverage matrix by ticker and time period.
    
    Given a feature DataFrame with DateIndex and ticker column level,
    computes the fraction of non-NaN values across all fundamental columns
    for each (ticker, period) combination.
    
    Args:
        df: Feature DataFrame with MultiIndex columns (ticker, feature) and DatetimeIndex.
        fundamental_cols: List of fundamental column names to check.
            If None, auto-detected using identify_fundamental_columns().
        coverage_freq: Pandas frequency for grouping ("Y" for yearly, "Q" for quarterly).
        benchmark: Benchmark ticker to exclude from coverage (e.g., "SPY").
    
    Returns:
        DataFrame with:
            - index: ticker symbols
            - columns: period labels (e.g., 2012, 2013, ... for yearly)
            - values: coverage fraction in [0, 1] = non-NaN fraction across fundamental cols
    
    Example:
        >>> coverage = compute_fundamentals_coverage(feature_df)
        >>> print(coverage)
                  2012   2013   2014   2015
        AAPL     0.56   0.56   0.56   0.56
        MSFT     0.11   0.56   0.56   0.56
        ...
    """
    if fundamental_cols is None:
        fundamental_cols = identify_fundamental_columns(df)
    
    if not fundamental_cols:
        logger.warning("No fundamental columns found in DataFrame")
        return pd.DataFrame()
    
    # Determine tickers from DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique().tolist()
    else:
        logger.warning("DataFrame does not have MultiIndex columns; cannot compute per-ticker coverage")
        return pd.DataFrame()
    
    # Exclude benchmark if specified
    if benchmark and benchmark in tickers:
        tickers = [t for t in tickers if t != benchmark]
    
    # Determine period labels from frequency
    if coverage_freq == "Y":
        period_func = lambda idx: idx.year
    elif coverage_freq == "Q":
        period_func = lambda idx: f"{idx.year}Q{idx.quarter}"
    elif coverage_freq == "M":
        period_func = lambda idx: f"{idx.year}-{idx.month:02d}"
    else:
        # Default to yearly
        period_func = lambda idx: idx.year
    
    # Get unique periods
    periods = sorted(set(period_func(d) for d in df.index))
    
    coverage_data = {}
    
    for ticker in tickers:
        ticker_coverage = {}
        
        for period in periods:
            # Get mask for this period
            period_mask = df.index.to_series().apply(lambda d: period_func(d) == period)
            period_df = df.loc[period_mask]
            
            if len(period_df) == 0:
                ticker_coverage[period] = np.nan
                continue
            
            # Get fundamental columns for this ticker
            ticker_fundamental_data = []
            for col in fundamental_cols:
                if (ticker, col) in period_df.columns:
                    ticker_fundamental_data.append(period_df[(ticker, col)])
            
            if not ticker_fundamental_data:
                ticker_coverage[period] = 0.0
                continue
            
            ticker_df = pd.concat(ticker_fundamental_data, axis=1)
            
            # Calculate coverage: % non-NaN cells
            total_cells = ticker_df.size
            non_nan_cells = ticker_df.notna().sum().sum()
            coverage = non_nan_cells / total_cells if total_cells > 0 else 0.0
            ticker_coverage[period] = coverage
        
        coverage_data[ticker] = ticker_coverage
    
    coverage_df = pd.DataFrame(coverage_data).T
    coverage_df.index.name = "Ticker"
    
    return coverage_df


def select_good_fundamentals_tickers(
    coverage: pd.DataFrame,
    min_avg_coverage: float = 0.4,
    min_test_year: int | None = None,
) -> list[str]:
    """
    Select tickers with sufficiently good fundamentals coverage.
    
    This function filters tickers based on their average fundamental data
    coverage, optionally with additional constraints on more recent periods.
    
    Args:
        coverage: Coverage matrix DataFrame from compute_fundamentals_coverage().
            index = tickers, columns = years/periods, values = coverage [0, 1].
        min_avg_coverage: Minimum average coverage across all years (default 0.4 = 40%).
        min_test_year: If not None, also require that the mean coverage over
            years >= min_test_year is >= min_avg_coverage.
    
    Returns:
        List of ticker symbols that pass the coverage filter.
    
    Raises:
        ValueError: If zero tickers pass the coverage filter.
    
    Example:
        >>> good_tickers = select_good_fundamentals_tickers(
        ...     coverage, min_avg_coverage=0.5, min_test_year=2017
        ... )
        >>> print(f"Selected {len(good_tickers)} tickers with good coverage")
    """
    if coverage.empty:
        raise ValueError("Coverage DataFrame is empty; cannot select tickers.")
    
    # Compute average coverage across all years for each ticker
    avg_coverage = coverage.mean(axis=1)
    
    # Filter by overall average coverage
    good_overall = avg_coverage >= min_avg_coverage
    
    # Additional filter: test period coverage
    if min_test_year is not None:
        # Get columns (years) >= min_test_year
        test_cols = [c for c in coverage.columns if c >= min_test_year]
        
        if not test_cols:
            logger.warning(
                f"No years >= {min_test_year} found in coverage matrix. "
                f"Available: {list(coverage.columns)}"
            )
            good_test_period = pd.Series(True, index=coverage.index)
        else:
            test_coverage = coverage[test_cols].mean(axis=1)
            good_test_period = test_coverage >= min_avg_coverage
    else:
        good_test_period = pd.Series(True, index=coverage.index)
    
    # Combine filters
    passing = good_overall & good_test_period
    good_tickers = passing[passing].index.tolist()
    
    if not good_tickers:
        raise ValueError(
            f"Zero tickers pass the fundamentals coverage filter "
            f"(min_avg_coverage={min_avg_coverage:.1%}"
            + (f", min_test_year={min_test_year}" if min_test_year else "")
            + f"). Average coverage: {avg_coverage.mean():.1%}. "
            "Consider lowering --min-fundamentals-coverage or using --fundamentals-universe-mode full."
        )
    
    return good_tickers


def get_excluded_tickers(
    all_tickers: list[str],
    good_tickers: list[str],
) -> list[str]:
    """
    Get list of tickers that were excluded from the good list.
    
    Args:
        all_tickers: Full list of tickers.
        good_tickers: List of tickers that passed the filter.
    
    Returns:
        List of excluded tickers.
    """
    return [t for t in all_tickers if t not in good_tickers]


def get_tech_only_tickers(
    coverage: pd.DataFrame,
    min_avg_coverage: float = 0.4,
) -> list[str]:
    """
    Get list of tickers that should use tech-only features (hybrid mode).
    
    In hybrid mode, tickers below the coverage threshold will have their
    fundamental features set to NaN, effectively using only technical features.
    
    Args:
        coverage: Coverage matrix DataFrame from compute_fundamentals_coverage().
        min_avg_coverage: Minimum average coverage threshold.
    
    Returns:
        List of tickers that should use tech-only features.
    """
    if coverage.empty:
        return []
    
    avg_coverage = coverage.mean(axis=1)
    tech_only = avg_coverage < min_avg_coverage
    return tech_only[tech_only].index.tolist()


def mask_fundamentals_for_tech_only_tickers(
    feature_df: pd.DataFrame,
    tech_only_tickers: list[str],
    fundamental_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Set fundamental features to NaN for tech-only tickers (hybrid mode).
    
    This allows the model to effectively use only technical features for
    tickers with poor fundamental data coverage.
    
    Args:
        feature_df: Feature DataFrame with MultiIndex columns (ticker, feature).
        tech_only_tickers: List of tickers to mask fundamentals for.
        fundamental_cols: List of fundamental column names. If None, auto-detected.
    
    Returns:
        Modified DataFrame with fundamental columns set to NaN for tech-only tickers.
    """
    if not tech_only_tickers:
        return feature_df
    
    if fundamental_cols is None:
        fundamental_cols = identify_fundamental_columns(feature_df)
    
    if not fundamental_cols:
        return feature_df
    
    # Make a copy to avoid modifying the original
    result = feature_df.copy()
    
    for ticker in tech_only_tickers:
        for col in fundamental_cols:
            if (ticker, col) in result.columns:
                result[(ticker, col)] = np.nan
    
    return result


def format_coverage_summary(
    mode: Literal["full", "good_only", "hybrid"],
    min_coverage: float,
    n_fundamentals: int,
    n_tech_only: int,
    good_tickers: list[str] | None = None,
    excluded_tickers: list[str] | None = None,
) -> str:
    """
    Format a summary string for fundamentals coverage filtering.
    
    Args:
        mode: Fundamentals universe mode.
        min_coverage: Minimum coverage threshold.
        n_fundamentals: Number of tickers with fundamentals enabled.
        n_tech_only: Number of tech-only tickers.
        good_tickers: List of tickers with good coverage (for good_only mode).
        excluded_tickers: List of excluded tickers (for good_only mode).
    
    Returns:
        Formatted summary string for printing.
    """
    lines = [
        "",
        "FUNDAMENTALS COVERAGE SUMMARY",
        "-" * 35,
        f"Fundamentals universe mode : {mode}",
        f"Min coverage threshold     : {min_coverage:.0%}",
        f"Tickers with fundamentals  : {n_fundamentals}",
        f"Tickers tech-only          : {n_tech_only}",
    ]
    
    if mode == "good_only" and good_tickers:
        lines.append("")
        lines.append(f"Tickers used ({len(good_tickers)}):")
        # Format in rows of 10
        for i in range(0, len(good_tickers), 10):
            batch = good_tickers[i:i+10]
            lines.append("  " + ", ".join(batch))
    
    if mode == "good_only" and excluded_tickers:
        lines.append("")
        lines.append(f"Tickers excluded ({len(excluded_tickers)}):")
        for i in range(0, len(excluded_tickers), 10):
            batch = excluded_tickers[i:i+10]
            lines.append("  " + ", ".join(batch))
    
    if mode == "hybrid" and n_tech_only > 0:
        lines.append("")
        lines.append("Note: Tech-only tickers use only technical features;")
        lines.append("      fundamental columns are set to NaN for these tickers.")
    
    lines.append("-" * 35)
    
    return "\n".join(lines)
