#!/usr/bin/env python3
"""
Fundamental Data Coverage Validation Script.

Checks SEC fundamental data availability for each ticker and year in the
50-stock large-cap universe before any modeling occurs.

Usage:
    python scripts/check_fundamentals_coverage.py
    python scripts/check_fundamentals_coverage.py --fundamentals-source finagg
    python scripts/check_fundamentals_coverage.py --fundamentals-source csv --fundamentals-csv data/fundamentals.csv
"""

import sys
import argparse
from pathlib import Path
from typing import Literal

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from auto_researcher.data.price_loader import download_price_history
from auto_researcher.data.universe import get_large_cap_universe
from auto_researcher.features.feature_pipeline import build_feature_matrix
from auto_researcher.config import FeatureConfig


# Configuration
START_DATE = "2012-01-01"
END_DATE = "2024-01-01"
BENCHMARK = "SPY"

# Fundamental column prefixes to check
FUNDAMENTAL_PREFIXES = ("quality_", "growth_", "prof_", "value_", "size_")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check fundamental data coverage across tickers and years."
    )
    parser.add_argument(
        "--fundamentals-source",
        type=str,
        default="finagg",
        choices=["finagg", "csv", "yfinance"],
        help="Source for fundamental data (default: finagg)",
    )
    parser.add_argument(
        "--fundamentals-csv",
        type=str,
        default=None,
        help="Path to fundamentals CSV file (required if source=csv)",
    )
    parser.add_argument(
        "--low-coverage-threshold",
        type=float,
        default=0.4,
        help="Threshold for low coverage warnings (default: 0.4 = 40%%)",
    )
    return parser.parse_args()


def identify_fundamental_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify fundamental feature columns by prefix.
    
    Args:
        df: Feature DataFrame with MultiIndex columns (ticker, feature).
    
    Returns:
        List of fundamental feature names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Get unique feature names from level 1
        all_features = df.columns.get_level_values(1).unique().tolist()
    else:
        all_features = df.columns.tolist()
    
    fundamental_cols = [
        col for col in all_features
        if any(col.startswith(prefix) for prefix in FUNDAMENTAL_PREFIXES)
    ]
    return fundamental_cols


def compute_coverage_matrix(
    feature_df: pd.DataFrame,
    fundamental_cols: list[str],
    tickers: list[str],
    years: list[int],
) -> pd.DataFrame:
    """
    Compute coverage matrix: % non-NaN fundamentals per (ticker, year).
    
    Args:
        feature_df: Feature DataFrame with MultiIndex columns (ticker, feature).
        fundamental_cols: List of fundamental column names.
        tickers: List of tickers to check.
        years: List of years to check.
    
    Returns:
        DataFrame with tickers as rows, years as columns, values as coverage %.
    """
    coverage_data = {}
    
    for ticker in tickers:
        if ticker == BENCHMARK:
            continue
        
        ticker_coverage = {}
        for year in years:
            # Filter to this year
            year_mask = feature_df.index.year == year
            year_df = feature_df.loc[year_mask]
            
            if len(year_df) == 0:
                ticker_coverage[year] = np.nan
                continue
            
            # Get fundamental columns for this ticker
            if isinstance(feature_df.columns, pd.MultiIndex):
                ticker_fundamental_data = []
                for col in fundamental_cols:
                    if (ticker, col) in year_df.columns:
                        ticker_fundamental_data.append(year_df[(ticker, col)])
                
                if not ticker_fundamental_data:
                    ticker_coverage[year] = 0.0
                    continue
                
                ticker_df = pd.concat(ticker_fundamental_data, axis=1)
            else:
                # Flat columns
                ticker_cols = [c for c in fundamental_cols if c in year_df.columns]
                if not ticker_cols:
                    ticker_coverage[year] = 0.0
                    continue
                ticker_df = year_df[ticker_cols]
            
            # Calculate coverage: % non-NaN cells
            total_cells = ticker_df.size
            non_nan_cells = ticker_df.notna().sum().sum()
            coverage = non_nan_cells / total_cells if total_cells > 0 else 0.0
            ticker_coverage[year] = coverage
        
        coverage_data[ticker] = ticker_coverage
    
    coverage_df = pd.DataFrame(coverage_data).T
    coverage_df.index.name = "Ticker"
    return coverage_df


def print_coverage_matrix(coverage_df: pd.DataFrame) -> None:
    """Print the coverage matrix in a readable format."""
    print("\n" + "=" * 100)
    print("  FUNDAMENTAL DATA COVERAGE MATRIX (% non-NaN)")
    print("=" * 100)
    
    # Format as percentages
    formatted = coverage_df.map(lambda x: f"{x*100:5.1f}%" if pd.notna(x) else "  N/A")
    
    # Print header
    years = coverage_df.columns.tolist()
    header = f"{'Ticker':<8} |"
    for year in years:
        header += f" {year} |"
    print(header)
    print("-" * len(header))
    
    # Print each ticker
    for ticker in coverage_df.index:
        row = f"{ticker:<8} |"
        for year in years:
            val = formatted.loc[ticker, year]
            row += f" {val} |"
        print(row)
    
    # Print footer with averages
    print("-" * len(header))
    avg_row = f"{'AVG':<8} |"
    for year in years:
        avg = coverage_df[year].mean()
        avg_row += f" {avg*100:5.1f}% |"
    print(avg_row)


def print_summary_report(
    coverage_df: pd.DataFrame,
    low_threshold: float = 0.4,
) -> None:
    """
    Print summary report with warnings for low coverage.
    
    Args:
        coverage_df: Coverage matrix DataFrame.
        low_threshold: Threshold for low coverage warnings (0.4 = 40%).
    """
    print("\n" + "=" * 100)
    print("  COVERAGE SUMMARY REPORT")
    print("=" * 100)
    
    years = coverage_df.columns.tolist()
    tickers = coverage_df.index.tolist()
    
    # 1. Overall statistics
    global_avg = coverage_df.values[~np.isnan(coverage_df.values)].mean()
    print(f"\n  Overall average coverage: {global_avg*100:.1f}%")
    print(f"  Low coverage threshold: {low_threshold*100:.0f}%")
    print(f"  Tickers analyzed: {len(tickers)}")
    print(f"  Years analyzed: {years[0]} - {years[-1]}")
    
    # 2. Years with low global coverage
    print("\n  YEARS WITH LOW COVERAGE (<{:.0f}%):".format(low_threshold * 100))
    print("  " + "-" * 50)
    
    low_coverage_years = []
    for year in years:
        year_avg = coverage_df[year].mean()
        if year_avg < low_threshold:
            low_coverage_years.append((year, year_avg))
            print(f"    {year}: {year_avg*100:5.1f}% average coverage")
    
    if not low_coverage_years:
        print("    (none)")
    
    # 3. Consecutive years with low coverage
    if len(low_coverage_years) >= 2:
        print("\n  WARNING: Multiple consecutive years with low coverage:")
        sorted_low = sorted(low_coverage_years, key=lambda x: x[0])
        start_year = sorted_low[0][0]
        end_year = sorted_low[-1][0]
        avg_pct = np.mean([y[1] for y in sorted_low])
        print(f"    SEC fundamental coverage low for {start_year}-{end_year} ({avg_pct*100:.0f}% average)")
    
    # 4. Tickers with low average coverage
    print("\n  TICKERS WITH LOW AVERAGE COVERAGE (<{:.0f}%):".format(low_threshold * 100))
    print("  " + "-" * 50)
    
    ticker_avg = coverage_df.mean(axis=1)
    low_tickers = ticker_avg[ticker_avg < low_threshold].sort_values()
    
    if len(low_tickers) > 0:
        for ticker, avg in low_tickers.items():
            print(f"    {ticker}: {avg*100:5.1f}% average coverage")
        
        pct_low = len(low_tickers) / len(tickers) * 100
        print(f"\n  WARNING: {len(low_tickers)}/{len(tickers)} tickers ({pct_low:.0f}%) have low coverage")
    else:
        print("    (none)")
    
    # 5. Tickers with perfect coverage
    print("\n  TICKERS WITH PERFECT COVERAGE (100%):".format())
    print("  " + "-" * 50)
    
    perfect_tickers = ticker_avg[ticker_avg >= 0.99].sort_values(ascending=False)
    if len(perfect_tickers) > 0:
        perfect_list = ", ".join(perfect_tickers.index[:10])
        if len(perfect_tickers) > 10:
            perfect_list += f", ... (+{len(perfect_tickers)-10} more)"
        print(f"    {perfect_list}")
    else:
        print("    (none)")
    
    # 6. Recommend periods to use
    print("\n  RECOMMENDATIONS:")
    print("  " + "-" * 50)
    
    good_years = [y for y in years if coverage_df[y].mean() >= low_threshold]
    if good_years:
        print(f"    Use data from: {min(good_years)}-{max(good_years)}")
        print(f"    Good coverage years: {', '.join(map(str, good_years))}")
    
    bad_years = [y for y in years if coverage_df[y].mean() < low_threshold]
    if bad_years:
        print(f"    Avoid or supplement: {', '.join(map(str, bad_years))}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "=" * 100)
    print("  FUNDAMENTAL DATA COVERAGE VALIDATION")
    print("=" * 100)
    
    print(f"\n  Configuration:")
    print(f"    Date range: {START_DATE} to {END_DATE}")
    print(f"    Fundamentals source: {args.fundamentals_source}")
    print(f"    Low coverage threshold: {args.low_coverage_threshold*100:.0f}%")
    
    # 1. Get universe
    print("\n  [1/4] Loading universe...")
    universe = get_large_cap_universe()
    tickers = universe + [BENCHMARK]
    print(f"        Loaded {len(universe)} tickers + {BENCHMARK} benchmark")
    
    # 2. Download prices
    print("\n  [2/4] Downloading price data...")
    raw_prices = download_price_history(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
    )
    print(f"        Loaded {len(raw_prices)} trading days")
    
    # Extract adjusted close prices (simple columns for build_feature_matrix)
    if isinstance(raw_prices.columns, pd.MultiIndex):
        # MultiIndex columns - extract Adj Close
        adj_close = raw_prices.xs('Adj Close', axis=1, level=0)
    else:
        # Already simple columns (unlikely but handle it)
        adj_close = raw_prices
    
    print(f"        Adjusted close shape: {adj_close.shape}")
    
    # 3. Build feature matrix with fundamentals
    print("\n  [3/4] Building feature matrix with fundamentals...")
    
    # Configure fundamental factors
    fundamental_factor_families = {
        'value': True,
        'quality': True,
        'growth': True,
        'profitability': True,
        'size': False,  # Size requires market cap from prices
    }
    
    feature_config = FeatureConfig(
        momentum_windows=(21, 63, 126, 252),
        volatility_windows=(21, 63),
        include_fundamentals=True,
        include_sentiment=False,
        fundamentals_source=args.fundamentals_source,
        fundamentals_csv_path=args.fundamentals_csv,
        fundamental_factor_families=fundamental_factor_families,
    )
    
    feature_df = build_feature_matrix(adj_close, config=feature_config)
    print(f"        Built features: {feature_df.shape}")
    
    # 4. Identify fundamental columns
    fundamental_cols = identify_fundamental_columns(feature_df)
    print(f"        Found {len(fundamental_cols)} fundamental columns:")
    for col in fundamental_cols[:10]:
        print(f"          - {col}")
    if len(fundamental_cols) > 10:
        print(f"          ... and {len(fundamental_cols) - 10} more")
    
    # 5. Compute coverage matrix
    print("\n  [4/4] Computing coverage matrix...")
    years = list(range(2012, 2024))  # 2012-2023
    
    coverage_df = compute_coverage_matrix(
        feature_df=feature_df,
        fundamental_cols=fundamental_cols,
        tickers=universe,
        years=years,
    )
    
    # 6. Print results
    print_coverage_matrix(coverage_df)
    print_summary_report(coverage_df, low_threshold=args.low_coverage_threshold)
    
    print("\n" + "=" * 100)
    print("  Coverage validation complete.")
    print("=" * 100 + "\n")
    
    return coverage_df


if __name__ == "__main__":
    main()
