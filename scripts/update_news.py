"""
News Database Update Script.

One-command update that:
  1. Scrapes latest articles from Business Insider for all S&P 500 tickers
  2. Scores new articles with FinBERT sentiment
  3. Syncs new articles into the ChromaDB vector store for RAG

Designed to be run periodically (daily/weekly) to keep news fresh.

Usage:
    # Full update (scrape + score + sync vector store)
    python scripts/update_news.py

    # Quick update (only recent pages, ~30 min)
    python scripts/update_news.py --quick

    # Scrape only (no scoring or vector store sync)
    python scripts/update_news.py --scrape-only

    # Score + sync only (no scraping, useful after manual scrape)
    python scripts/update_news.py --no-scrape

    # Specific tickers only
    python scripts/update_news.py --tickers AAPL NVDA MSFT

    # Show current status
    python scripts/update_news.py --status
"""

import asyncio
import argparse
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

# Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.auto_researcher.data.news_scraper import NewsDatabase, get_sp500_tickers


def print_status(db: NewsDatabase):
    """Print current database status."""
    stats = db.get_stats()
    
    print("\n" + "=" * 65)
    print("NEWS DATABASE STATUS")
    print("=" * 65)
    print(f"  Database:       {db.db_path}")
    print(f"  Total articles: {stats['total_articles']:,}")
    print(f"  Unique tickers: {stats['unique_tickers']}")
    
    if stats["date_range"][0]:
        print(f"  Date range:     {stats['date_range'][0][:10]} to {stats['date_range'][1][:10]}")
    
    # Articles without sentiment
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE sentiment_score IS NULL"
        )
        unscored = cursor.fetchone()[0]
        print(f"  Unscored:       {unscored:,}")
        
        # Scrape progress
        cursor = conn.execute("SELECT COUNT(*) FROM scrape_progress")
        started = cursor.fetchone()[0]
        cursor = conn.execute(
            "SELECT COUNT(*) FROM scrape_progress WHERE is_complete = 1"
        )
        complete = cursor.fetchone()[0]
        print(f"  Tickers scraped: {started} ({complete} complete)")
        
        # Most recent scrape
        cursor = conn.execute(
            "SELECT ticker, last_scrape_date FROM scrape_progress "
            "ORDER BY last_scrape_date DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            print(f"  Last scraped:   {row[0]} at {row[1][:16] if row[1] else 'N/A'}")
    
    # ChromaDB status
    try:
        from auto_researcher.data.news_vectorstore import NewsVectorStore
        store = NewsVectorStore()
        count = store.get_index_count()
        print(f"  Vector store:   {count:,} articles indexed")
    except Exception:
        print("  Vector store:   not available")
    
    print("=" * 65)


async def step_scrape(
    tickers: list[str],
    max_pages: int = 10,
    quick: bool = False,
) -> int:
    """Step 1: Scrape latest articles from Business Insider."""
    from scripts.async_scraper import AsyncBusinessInsiderScraper
    
    db = NewsDatabase()
    
    if quick:
        # Only check first 3 pages per ticker (most recent articles)
        max_pages = 3
        logger.info(f"Quick mode: checking {max_pages} pages per ticker")
    
    logger.info(f"Scraping {len(tickers)} tickers (max {max_pages} pages each)...")
    
    scraper = AsyncBusinessInsiderScraper(
        db=db,
        max_concurrent=5,
        delay_between_tickers=0.8,
    )
    
    results = await scraper.scrape_all(tickers, max_pages)
    
    total_new = sum(v for v in results.values() if v >= 0)
    failed = sum(1 for v in results.values() if v < 0)
    
    logger.info(f"Scraping done: {total_new:,} new articles, {failed} failed tickers")
    return total_new


def step_score(limit: int = 10000) -> int:
    """Step 2: Score unscored articles with FinBERT."""
    db = NewsDatabase()
    
    # Check how many need scoring
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE sentiment_score IS NULL"
        )
        unscored = cursor.fetchone()[0]
    
    if unscored == 0:
        logger.info("All articles already scored")
        return 0
    
    logger.info(f"Scoring {min(unscored, limit):,} unscored articles with FinBERT...")
    
    try:
        from auto_researcher.agents.finbert_sentiment import create_analyzer
        analyzer = create_analyzer(prefer_finbert=True)
    except ImportError as e:
        logger.error(f"Failed to load FinBERT: {e}")
        return 0
    
    # Get unscored articles
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT id, title, snippet FROM articles "
            "WHERE sentiment_score IS NULL "
            "ORDER BY published_date DESC LIMIT ?",
            (limit,),
        )
        articles = [dict(row) for row in cursor.fetchall()]
    
    if not articles:
        return 0
    
    scored = 0
    batch_size = 32
    t0 = time.time()
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        texts = []
        for art in batch:
            text = art["title"]
            if art.get("snippet"):
                text += ". " + art["snippet"]
            texts.append(text)
        
        try:
            results = analyzer.analyze_batch(texts)
            for art, result in zip(batch, results):
                db.update_sentiment(art["id"], result.sentiment_score, result.label)
                scored += 1
        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
        
        if (i + batch_size) % 500 == 0:
            elapsed = time.time() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            logger.info(f"  Scored {scored:,}/{len(articles):,} ({rate:.0f}/sec)")
    
    elapsed = time.time() - t0
    logger.info(f"Scored {scored:,} articles in {elapsed:.1f}s")
    
    # Cleanup
    if hasattr(analyzer, "unload"):
        analyzer.unload()
    
    return scored


def step_sync_vectorstore() -> int:
    """Step 3: Sync new articles from news.db into ChromaDB vector store."""
    try:
        from auto_researcher.data.news_vectorstore import NewsVectorStore
    except ImportError as e:
        logger.error(f"NewsVectorStore not available: {e}")
        return 0
    
    store = NewsVectorStore()
    existing = store.get_index_count()
    
    logger.info(f"Vector store has {existing:,} articles. Syncing new ones from DB...")
    
    # build_from_db already handles deduplication via existing IDs
    new_count = store.build_from_db(rebuild=False)
    
    total = store.get_index_count()
    logger.info(f"Vector store sync done: {new_count:,} new, {total:,} total")
    return new_count


async def main():
    parser = argparse.ArgumentParser(
        description="Update news database (scrape + score + sync vector store)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current database status and exit",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick update: only 3 pages per ticker (~30 min)",
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Specific tickers to update (default: all S&P 500)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=10,
        help="Max pages to scrape per ticker (default: 10)",
    )
    parser.add_argument(
        "--scrape-only", action="store_true",
        help="Only scrape, skip scoring and vector store sync",
    )
    parser.add_argument(
        "--no-scrape", action="store_true",
        help="Skip scraping, only score and sync vector store",
    )
    parser.add_argument(
        "--no-score", action="store_true",
        help="Skip FinBERT scoring step",
    )
    parser.add_argument(
        "--no-sync", action="store_true",
        help="Skip vector store sync step",
    )
    parser.add_argument(
        "--score-limit", type=int, default=10000,
        help="Max articles to score per run (default: 10000)",
    )
    
    args = parser.parse_args()
    db = NewsDatabase()
    
    if args.status:
        print_status(db)
        return
    
    tickers = args.tickers or get_sp500_tickers()
    
    print("\n" + "=" * 65)
    print("NEWS DATABASE UPDATE")
    print(f"  Date:    {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Mode:    {'quick' if args.quick else 'standard'}")
    steps = []
    if not args.no_scrape:
        steps.append("scrape")
    if not args.scrape_only and not args.no_score:
        steps.append("score")
    if not args.scrape_only and not args.no_sync:
        steps.append("sync vectorstore")
    print(f"  Steps:   {' → '.join(steps)}")
    print("=" * 65 + "\n")
    
    t_total = time.time()
    
    # Step 1: Scrape
    new_articles = 0
    if not args.no_scrape:
        logger.info("━" * 50)
        logger.info("STEP 1/3: SCRAPING BUSINESS INSIDER")
        logger.info("━" * 50)
        new_articles = await step_scrape(
            tickers,
            max_pages=args.max_pages,
            quick=args.quick,
        )
    
    if args.scrape_only:
        print_status(db)
        return
    
    # Step 2: Score with FinBERT
    scored = 0
    if not args.no_score:
        logger.info("")
        logger.info("━" * 50)
        logger.info("STEP 2/3: SCORING WITH FINBERT")
        logger.info("━" * 50)
        scored = step_score(limit=args.score_limit)
    
    # Step 3: Sync to vector store
    synced = 0
    if not args.no_sync:
        logger.info("")
        logger.info("━" * 50)
        logger.info("STEP 3/3: SYNCING VECTOR STORE")
        logger.info("━" * 50)
        synced = step_sync_vectorstore()
    
    elapsed = time.time() - t_total
    
    print("\n" + "=" * 65)
    print("UPDATE COMPLETE")
    print(f"  Time:           {elapsed / 60:.1f} minutes")
    print(f"  New articles:   {new_articles:,}")
    print(f"  Scored:         {scored:,}")
    print(f"  Synced to vec:  {synced:,}")
    print("=" * 65)
    
    print_status(db)


if __name__ == "__main__":
    asyncio.run(main())
