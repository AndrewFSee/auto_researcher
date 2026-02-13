#!/usr/bin/env python
"""
Fill in missing historical articles for tickers that weren't fully scraped.

This script scrapes ONLY headlines (no full text) which is very fast.
Run with: python scripts/fill_missing_articles.py

Estimated time: ~10-15 minutes for all missing articles.
"""

import asyncio
import aiohttp
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "news.db"

# Tickers that need more historical data
TICKERS_TO_FILL = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'GOOGL', 'GOOG', 'BA', 'AMD', 'INTC']

# Max pages to scrape (each page = ~50 articles)
MAX_PAGES = 200


def get_existing_urls(ticker: str) -> set:
    """Get all existing URLs for a ticker."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT url FROM articles WHERE ticker = ?", (ticker,))
    urls = {row[0] for row in cur.fetchall()}
    conn.close()
    return urls


def parse_date(date_str: str) -> datetime | None:
    """Parse date string."""
    if not date_str:
        return None
    
    formats = [
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_articles(html: str, ticker: str, existing_urls: set) -> list[dict]:
    """Parse articles from HTML, skipping existing ones."""
    soup = BeautifulSoup(html, 'html.parser')
    articles = []
    
    for elem in soup.select('.latest-news__story'):
        try:
            link = elem.find('a', href=True)
            if not link:
                continue
            
            title = link.get_text(strip=True)
            url = link['href']
            
            if not url.startswith('http'):
                url = urljoin("https://markets.businessinsider.com", url)
            
            if '/news/' not in url:
                continue
            
            # Skip if we already have this article
            if url in existing_urls:
                continue
            
            date_elem = elem.select_one('.latest-news__meta time')
            date_str = date_elem.get('datetime') if date_elem else None
            published_date = parse_date(date_str) if date_str else datetime.now()
            
            source = "Business Insider"
            source_elem = elem.select_one('.latest-news__source')
            if source_elem:
                source = source_elem.get_text(strip=True)
            
            if title and len(title) > 10:
                articles.append({
                    'ticker': ticker.upper(),
                    'title': title,
                    'url': url,
                    'published_date': published_date,
                    'source': source,
                })
        except Exception:
            continue
    
    return articles


async def fetch_page(session: aiohttp.ClientSession, ticker: str, page: int) -> str:
    """Fetch a single page."""
    url = f"https://markets.businessinsider.com/news/{ticker.lower()}-stock"
    if page > 1:
        url += f"?p={page}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception as e:
        logger.debug(f"Error fetching {ticker} page {page}: {e}")
    return ""


async def scrape_ticker(session: aiohttp.ClientSession, ticker: str) -> int:
    """Scrape all missing articles for a ticker."""
    existing_urls = get_existing_urls(ticker)
    logger.info(f"{ticker}: Found {len(existing_urls):,} existing articles")
    
    all_new_articles = []
    consecutive_no_new = 0
    
    for page in range(1, MAX_PAGES + 1):
        html = await fetch_page(session, ticker, page)
        if not html:
            consecutive_no_new += 1
            if consecutive_no_new >= 3:
                break
            continue
        
        articles = parse_articles(html, ticker, existing_urls)
        
        if not articles:
            # Check if page had any articles at all
            soup = BeautifulSoup(html, 'html.parser')
            page_articles = soup.select('.latest-news__story')
            if not page_articles:
                # Empty page = end of content
                break
            # All articles were duplicates, continue
            consecutive_no_new += 1
            if consecutive_no_new >= 5:
                break
        else:
            consecutive_no_new = 0
            all_new_articles.extend(articles)
            # Add to existing to avoid duplicates within this run
            for a in articles:
                existing_urls.add(a['url'])
        
        if page % 20 == 0:
            logger.info(f"  {ticker}: Page {page}, {len(all_new_articles)} new articles so far")
        
        await asyncio.sleep(0.3)  # Be polite
    
    # Save to database
    if all_new_articles:
        save_articles(all_new_articles)
        logger.info(f"{ticker}: Added {len(all_new_articles):,} NEW articles")
    else:
        logger.info(f"{ticker}: No new articles found")
    
    return len(all_new_articles)


def save_articles(articles: list[dict]):
    """Save articles to database."""
    import hashlib
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    for a in articles:
        article_hash = hashlib.md5(a['url'].encode()).hexdigest()
        try:
            cur.execute("""
                INSERT OR IGNORE INTO articles 
                (article_hash, ticker, title, url, published_date, source, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                article_hash,
                a['ticker'],
                a['title'],
                a['url'],
                a['published_date'],
                a['source'],
                datetime.now(),
            ))
        except Exception as e:
            logger.debug(f"Error saving article: {e}")
    
    conn.commit()
    conn.close()


async def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Fill Missing Articles - Headlines Only (Fast)")
    logger.info("=" * 60)
    
    total_new = 0
    
    async with aiohttp.ClientSession() as session:
        for ticker in TICKERS_TO_FILL:
            new_count = await scrape_ticker(session, ticker)
            total_new += new_count
            await asyncio.sleep(1)  # Delay between tickers
    
    logger.info("=" * 60)
    logger.info(f"DONE! Added {total_new:,} new articles total")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
