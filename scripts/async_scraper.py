"""
Async News Scraper with concurrent requests.

Much faster than the synchronous version - can scrape all S&P 500 tickers
in a few hours instead of days.

Usage:
    python scripts/async_scraper.py --tickers AAPL NVDA MSFT --max-pages 50
    python scripts/async_scraper.py --sp500 --max-pages 100
"""

import asyncio
import aiohttp
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import re
import hashlib
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_researcher.data.news_scraper import NewsDatabase, ScrapedArticle, get_sp500_tickers


class AsyncBusinessInsiderScraper:
    """Async scraper for Business Insider stock news."""
    
    BASE_URL = "https://markets.businessinsider.com/news/{ticker}-stock"
    
    def __init__(
        self,
        db: Optional[NewsDatabase] = None,
        max_concurrent: int = 5,  # Be polite - don't hammer the server
        delay_between_tickers: float = 1.0,
    ):
        self.db = db or NewsDatabase()
        self.max_concurrent = max_concurrent
        self.delay = delay_between_tickers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date strings."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        formats = [
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %I:%M %p",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%b %d, %Y, %I:%M %p",
            "%b %d, %Y",
            "%B %d, %Y",
            "%Y-%m-%d",
            "%m/%d/%Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Handle relative dates
        if "ago" in date_str.lower():
            return datetime.now()
        if "yesterday" in date_str.lower():
            return datetime.now() - timedelta(days=1)
        
        match = re.match(r'^(\d+)d$', date_str.strip())
        if match:
            return datetime.now() - timedelta(days=int(match.group(1)))
        
        match = re.match(r'^(\d+)h$', date_str.strip())
        if match:
            return datetime.now() - timedelta(hours=int(match.group(1)))
        
        return None
    
    def _parse_articles(self, html: str, ticker: str) -> list[ScrapedArticle]:
        """Parse articles from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        seen_urls = set()
        
        for elem in soup.select('.latest-news__story'):
            try:
                link = elem.find('a', href=True)
                if not link:
                    continue
                
                title = link.get_text(strip=True)
                url = link['href']
                
                if not url.startswith('http'):
                    url = urljoin("https://markets.businessinsider.com", url)
                
                if '/news/' not in url or url in seen_urls:
                    continue
                seen_urls.add(url)
                
                date_elem = elem.select_one('.latest-news__meta time')
                date_str = date_elem.get('datetime') if date_elem else None
                published_date = self._parse_date(date_str) if date_str else None
                
                source = "Business Insider"
                source_elem = elem.select_one('.latest-news__source')
                if source_elem:
                    source = source_elem.get_text(strip=True)
                
                if title and len(title) > 10:
                    articles.append(ScrapedArticle(
                        ticker=ticker.upper(),
                        title=title,
                        url=url,
                        published_date=published_date or datetime.now(),
                        source=source,
                    ))
            except Exception:
                continue
        
        return articles
    
    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        page: int,
    ) -> tuple[int, list[ScrapedArticle]]:
        """Fetch and parse a single page."""
        url = f"https://markets.businessinsider.com/news/{ticker.lower()}-stock"
        if page > 1:
            url += f"?p={page}"
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return page, []
                html = await resp.text()
                return page, self._parse_articles(html, ticker)
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker} page {page}: {e}")
            return page, []
    
    async def scrape_ticker(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        max_pages: int = 100,
    ) -> int:
        """Scrape all pages for a ticker concurrently."""
        ticker = ticker.upper()
        all_articles = []
        
        # Scrape pages in batches
        batch_size = self.max_concurrent
        page = 1
        consecutive_empty = 0
        
        while page <= max_pages and consecutive_empty < 3:
            # Create tasks for a batch of pages
            tasks = []
            for p in range(page, min(page + batch_size, max_pages + 1)):
                tasks.append(self._fetch_page(session, ticker, p))
            
            results = await asyncio.gather(*tasks)
            
            # Process results
            batch_articles = []
            empty_in_batch = 0
            for p, articles in sorted(results):
                if articles:
                    batch_articles.extend(articles)
                    consecutive_empty = 0
                else:
                    empty_in_batch += 1
            
            if empty_in_batch == len(tasks):
                consecutive_empty += 1
            
            all_articles.extend(batch_articles)
            page += batch_size
            
            # Small delay between batches
            await asyncio.sleep(0.2)
        
        # Insert all articles
        new_count = self.db.insert_articles(all_articles)
        
        # Update progress
        self.db.update_scrape_progress(
            ticker, page,
            new_count,
            is_complete=(consecutive_empty >= 3)
        )
        
        return new_count
    
    async def scrape_all(
        self,
        tickers: list[str],
        max_pages_per_ticker: int = 100,
    ) -> dict[str, int]:
        """Scrape all tickers."""
        results = {}
        
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
        ) as session:
            for i, ticker in enumerate(tickers, 1):
                try:
                    logger.info(f"[{i}/{len(tickers)}] Scraping {ticker}...")
                    new_count = await self.scrape_ticker(session, ticker, max_pages_per_ticker)
                    results[ticker] = new_count
                    logger.info(f"  {ticker}: {new_count} new articles")
                except Exception as e:
                    logger.error(f"  {ticker}: Failed - {e}")
                    results[ticker] = -1
                
                # Delay between tickers
                if i < len(tickers):
                    await asyncio.sleep(self.delay)
        
        return results


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Async scrape Business Insider news")
    parser.add_argument("--tickers", nargs="+", help="Tickers to scrape")
    parser.add_argument("--sp500", action="store_true", help="Scrape all S&P 500")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages per ticker")
    parser.add_argument("--concurrent", type=int, default=5, help="Concurrent requests")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between tickers")
    
    args = parser.parse_args()
    
    if args.sp500:
        tickers = get_sp500_tickers()
        logger.info(f"Scraping {len(tickers)} S&P 500 tickers...")
    elif args.tickers:
        tickers = args.tickers
    else:
        parser.print_help()
        return
    
    scraper = AsyncBusinessInsiderScraper(
        max_concurrent=args.concurrent,
        delay_between_tickers=args.delay,
    )
    
    start_time = time.time()
    results = await scraper.scrape_all(tickers, args.max_pages)
    elapsed = time.time() - start_time
    
    # Summary
    total_new = sum(v for v in results.values() if v >= 0)
    failed = sum(1 for v in results.values() if v < 0)
    
    print("\n" + "=" * 50)
    print("SCRAPING COMPLETE")
    print("=" * 50)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"New articles: {total_new:,}")
    print(f"Failed tickers: {failed}")
    
    # Show DB stats
    db = NewsDatabase()
    stats = db.get_stats()
    print(f"\nTotal in database: {stats['total_articles']:,} articles")
    print(f"Unique tickers: {stats['unique_tickers']}")


if __name__ == "__main__":
    asyncio.run(main())
