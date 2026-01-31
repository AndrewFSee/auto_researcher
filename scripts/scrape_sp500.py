"""
S&P 500 News Scraping Orchestrator.

This script manages the full scraping job for all S&P 500 tickers.
It can run in stages, resume from failures, and provides progress tracking.

Usage:
    # Start or resume the full scrape
    python scripts/scrape_sp500.py
    
    # Scrape specific sectors only
    python scripts/scrape_sp500.py --sector Technology
    
    # Quick update (only recent pages for existing tickers)
    python scripts/scrape_sp500.py --update
    
    # Show progress
    python scripts/scrape_sp500.py --status
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_researcher.data.news_scraper import NewsDatabase, get_sp500_tickers

# Use local async scraper
from scripts.async_scraper import AsyncBusinessInsiderScraper


def get_sp500_with_sectors() -> dict[str, list[str]]:
    """Get S&P 500 tickers organized by sector."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        
        # Group by GICS Sector
        sectors = {}
        for _, row in df.iterrows():
            sector = row.get('GICS Sector', 'Other')
            ticker = row['Symbol'].replace('.', '-')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)
        
        return sectors
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 with sectors: {e}")
        return {"All": get_sp500_tickers()}


def get_scrape_status(db: NewsDatabase) -> dict:
    """Get current scraping status."""
    import sqlite3
    
    stats = db.get_stats()
    
    # Get progress by ticker
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # Total progress
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM scrape_progress")
        tickers_started = cursor.fetchone()['cnt']
        
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM scrape_progress WHERE is_complete = 1")
        tickers_complete = cursor.fetchone()['cnt']
        
        # Recently updated
        yesterday = datetime.now() - timedelta(days=1)
        cursor = conn.execute(
            "SELECT ticker, last_scrape_date, total_articles FROM scrape_progress ORDER BY last_scrape_date DESC LIMIT 10"
        )
        recent = [dict(row) for row in cursor.fetchall()]
    
    return {
        'total_articles': stats['total_articles'],
        'unique_tickers': stats['unique_tickers'],
        'date_range': stats['date_range'],
        'tickers_started': tickers_started,
        'tickers_complete': tickers_complete,
        'recent_updates': recent,
    }


def print_status(db: NewsDatabase):
    """Print current scraping status."""
    status = get_scrape_status(db)
    
    print("\n" + "=" * 60)
    print("BUSINESS INSIDER NEWS DATABASE STATUS")
    print("=" * 60)
    print(f"\nDatabase: {db.db_path}")
    print(f"Total articles: {status['total_articles']:,}")
    print(f"Unique tickers: {status['unique_tickers']}")
    print(f"Tickers started: {status['tickers_started']}")
    print(f"Tickers complete: {status['tickers_complete']}")
    
    if status['date_range'][0]:
        print(f"Date range: {status['date_range'][0][:10]} to {status['date_range'][1][:10]}")
    
    print(f"\nRecent updates:")
    for row in status['recent_updates'][:5]:
        date_str = row['last_scrape_date'][:16] if row['last_scrape_date'] else 'N/A'
        print(f"  {row['ticker']}: {row['total_articles']:,} articles (updated {date_str})")
    
    # Estimate completion
    if status['unique_tickers'] > 0:
        avg_per_ticker = status['total_articles'] / status['unique_tickers']
        remaining = 500 - status['unique_tickers']
        est_remaining = remaining * avg_per_ticker
        print(f"\nEstimate for S&P 500:")
        print(f"  Avg articles per ticker: {avg_per_ticker:,.0f}")
        print(f"  Remaining tickers: {remaining}")
        print(f"  Est. remaining articles: {est_remaining:,.0f}")
        print(f"  Est. remaining time: {remaining * 0.5:.0f} minutes ({remaining * 0.5 / 60:.1f} hours)")


async def scrape_tickers(
    tickers: list[str],
    max_pages: int = 100,
    update_only: bool = False,
) -> dict[str, int]:
    """Scrape a list of tickers."""
    db = NewsDatabase()
    
    # If update only, filter to tickers we've already scraped
    if update_only:
        existing = set()
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute("SELECT ticker FROM scrape_progress")
            existing = {row[0] for row in cursor.fetchall()}
        
        tickers = [t for t in tickers if t in existing]
        max_pages = 5  # Only check recent pages
        logger.info(f"Update mode: checking {len(tickers)} existing tickers")
    
    if not tickers:
        logger.info("No tickers to scrape")
        return {}
    
    scraper = AsyncBusinessInsiderScraper(
        db=db,
        max_concurrent=5,
        delay_between_tickers=1.0,
    )
    
    results = await scraper.scrape_all(tickers, max_pages)
    return results


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape S&P 500 news")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--sector", type=str, help="Scrape specific sector only")
    parser.add_argument("--update", action="store_true", help="Update existing tickers only")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages per ticker")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scrape")
    parser.add_argument("--resume", action="store_true", help="Skip already-complete tickers")
    
    args = parser.parse_args()
    
    db = NewsDatabase()
    
    if args.status:
        print_status(db)
        return
    
    # Determine tickers to scrape
    if args.tickers:
        tickers = args.tickers
    elif args.sector:
        sectors = get_sp500_with_sectors()
        if args.sector not in sectors:
            print(f"Available sectors: {list(sectors.keys())}")
            return
        tickers = sectors[args.sector]
        logger.info(f"Scraping {len(tickers)} tickers in {args.sector}")
    else:
        tickers = get_sp500_tickers()
        logger.info(f"Scraping all {len(tickers)} S&P 500 tickers")
    
    # Resume mode: skip complete tickers
    if args.resume:
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute("SELECT ticker FROM scrape_progress WHERE is_complete = 1")
            complete = {row[0] for row in cursor.fetchall()}
        
        original = len(tickers)
        tickers = [t for t in tickers if t not in complete]
        logger.info(f"Resume mode: skipping {original - len(tickers)} complete tickers")
    
    # Start scraping
    start_time = time.time()
    results = await scrape_tickers(tickers, args.max_pages, args.update)
    elapsed = time.time() - start_time
    
    # Summary
    total_new = sum(v for v in results.values() if v >= 0)
    failed = sum(1 for v in results.values() if v < 0)
    
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Tickers scraped: {len(results)}")
    print(f"New articles: {total_new:,}")
    print(f"Failed tickers: {failed}")
    
    # Show final status
    print_status(db)


if __name__ == "__main__":
    asyncio.run(main())
