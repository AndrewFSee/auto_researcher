#!/usr/bin/env python
"""
Download fundamental data from FMP and Alpha Vantage.

This script fetches historical fundamental data for a given universe of tickers
and saves it to a CSV file compatible with load_raw_fundamentals().

Usage:
    python scripts/download_fundamentals.py \\
        --universe largecap \\
        --start-year 2005 \\
        --end-year 2024 \\
        --output-csv data/fundamentals_largecap.csv

Environment Variables:
    FMP_API_KEY: API key for Financial Modeling Prep (primary source)
    ALPHAVANTAGE_API_KEY: API key for Alpha Vantage (fallback)

Examples:
    # Download fundamentals for large-cap universe
    python scripts/download_fundamentals.py --universe largecap

    # Download fundamentals for S&P 500 universe
    python scripts/download_fundamentals.py --universe sp500 --output-csv data/fundamentals_sp500.csv

    # Download fundamentals for custom ticker list
    python scripts/download_fundamentals.py --universe file --universe-file tickers.txt
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_researcher.data.fundamentals_sources import (
    FUNDAMENTAL_COLUMNS,
    FundamentalsSourceConfig,
    fetch_fundamentals_for_symbol,
    get_fundamentals_source_config,
)
from auto_researcher.data.universe import (
    get_large_cap_universe,
    get_sp500_universe,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download fundamental data from FMP and Alpha Vantage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--universe",
        choices=["largecap", "sp500", "file"],
        default="largecap",
        help="Universe of tickers to download (default: largecap)",
    )

    parser.add_argument(
        "--universe-file",
        type=str,
        default=None,
        help="Path to file with tickers (one per line or CSV with 'ticker' column)",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2005,
        help="Start year for data (default: 2005)",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for data (default: current year)",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/fundamentals_largecap.csv",
        help="Output CSV path (default: data/fundamentals_largecap.csv)",
    )

    parser.add_argument(
        "--max-years-back",
        type=int,
        default=20,
        help="Maximum years of history to fetch per symbol (default: 20)",
    )

    return parser.parse_args()


def load_tickers_from_file(path: str) -> list[str]:
    """
    Load tickers from a file.

    Supports:
    - Plain text file with one ticker per line
    - CSV file with a 'ticker' column

    Args:
        path: Path to the file.

    Returns:
        List of ticker symbols.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
        if "ticker" in df.columns:
            return df["ticker"].dropna().str.strip().tolist()
        else:
            # Assume first column contains tickers
            return df.iloc[:, 0].dropna().str.strip().tolist()
    else:
        # Plain text file
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]


def get_tickers_for_universe(universe: str, universe_file: str | None) -> list[str]:
    """
    Get list of tickers based on universe selection.

    Args:
        universe: Universe type ("largecap", "sp500", or "file").
        universe_file: Path to file if universe == "file".

    Returns:
        List of ticker symbols.
    """
    if universe == "largecap":
        return get_large_cap_universe()
    elif universe == "sp500":
        return get_sp500_universe()
    elif universe == "file":
        if not universe_file:
            raise ValueError("--universe-file is required when --universe=file")
        return load_tickers_from_file(universe_file)
    else:
        raise ValueError(f"Unknown universe: {universe}")


def download_fundamentals(
    tickers: list[str],
    config: FundamentalsSourceConfig,
    start_year: int,
    end_year: int,
    max_years_back: int,
) -> pd.DataFrame:
    """
    Download fundamentals for all tickers.

    Args:
        tickers: List of ticker symbols.
        config: FundamentalsSourceConfig with API keys.
        start_year: Start year for filtering.
        end_year: End year for filtering.
        max_years_back: Maximum years of history per symbol.

    Returns:
        DataFrame with all fundamental data.
    """
    all_records: list[pd.DataFrame] = []
    successful = 0
    failed = 0
    failed_tickers: list[str] = []

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    logger.info(f"Downloading fundamentals for {len(tickers)} tickers...")
    logger.info(f"Date range: {start_date} to {end_date}")

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] Fetching {ticker}...")

        try:
            df = fetch_fundamentals_for_symbol(ticker, config, max_years_back)

            if len(df) == 0:
                logger.warning(f"  No data returned for {ticker}")
                failed += 1
                failed_tickers.append(ticker)
                continue

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            if len(df) == 0:
                logger.warning(f"  No data in date range for {ticker}")
                failed += 1
                failed_tickers.append(ticker)
                continue

            # Add ticker column
            df = df.reset_index()
            df["ticker"] = ticker
            df = df.rename(columns={"index": "date"})

            all_records.append(df)
            successful += 1
            logger.info(f"  Got {len(df)} records for {ticker}")

        except Exception as e:
            logger.error(f"  Error fetching {ticker}: {e}")
            failed += 1
            failed_tickers.append(ticker)

    logger.info("")
    logger.info(f"Download complete:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")

    if failed_tickers:
        logger.warning(f"  Failed tickers: {', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            logger.warning(f"    ... and {len(failed_tickers) - 10} more")

    if not all_records:
        return pd.DataFrame()

    # Combine all records
    combined = pd.concat(all_records, ignore_index=True)

    # Reorder columns: date, ticker, then fundamentals
    cols = ["date", "ticker"] + FUNDAMENTAL_COLUMNS
    for col in cols:
        if col not in combined.columns:
            combined[col] = pd.NA

    return combined[cols]


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set default end year
    if args.end_year is None:
        args.end_year = datetime.now().year

    logger.info("=" * 70)
    logger.info("  DOWNLOAD FUNDAMENTALS")
    logger.info("=" * 70)
    logger.info("")

    # Get configuration
    config = get_fundamentals_source_config()

    if not config.fmp_api_key and not config.av_api_key:
        logger.error(
            "No API keys found. Set FMP_API_KEY and/or ALPHAVANTAGE_API_KEY environment variables."
        )
        sys.exit(1)

    logger.info("Data sources:")
    if config.fmp_api_key:
        logger.info("  - FMP (Financial Modeling Prep): ✓ API key found")
    else:
        logger.info("  - FMP (Financial Modeling Prep): ✗ No API key")

    if config.av_api_key:
        logger.info("  - Alpha Vantage: ✓ API key found")
    else:
        logger.info("  - Alpha Vantage: ✗ No API key")

    logger.info("")

    # Get tickers
    tickers = get_tickers_for_universe(args.universe, args.universe_file)
    logger.info(f"Universe: {args.universe}")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info("")

    # Download fundamentals
    df = download_fundamentals(
        tickers=tickers,
        config=config,
        start_year=args.start_year,
        end_year=args.end_year,
        max_years_back=args.max_years_back,
    )

    if len(df) == 0:
        logger.error("No data downloaded. Check API keys and network connection.")
        sys.exit(1)

    # Ensure output directory exists
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info("")
    logger.info(f"Saved {len(df)} records to {output_path}")
    logger.info(f"Unique tickers: {df['ticker'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
