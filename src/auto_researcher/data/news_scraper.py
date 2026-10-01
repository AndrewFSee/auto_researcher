"""
Business Insider News Scraper.

Scrapes historical news articles from Business Insider for stock tickers.
Stores in SQLite database for efficient querying and incremental updates.

Usage:
    python -m auto_researcher.data.news_scraper --tickers AAPL NVDA MSFT
    python -m auto_researcher.data.news_scraper --sp500  # All S&P 500
    python -m auto_researcher.data.news_scraper --update  # Update existing tickers
"""

import sqlite3
import logging
import time
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Generator
from urllib.parse import urljoin
import hashlib

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "news.db"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ScrapedArticle:
    """A scraped news article."""
    ticker: str
    title: str
    url: str
    published_date: datetime
    source: str = "Business Insider"
    snippet: Optional[str] = None
    full_text: Optional[str] = None
    scraped_at: datetime = None
    
    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.now()
    
    @property
    def article_hash(self) -> str:
        """Unique hash for deduplication."""
        return hashlib.md5(f"{self.url}".encode()).hexdigest()


# ==============================================================================
# DATABASE MANAGER
# ==============================================================================

class NewsDatabase:
    """SQLite database for storing scraped news articles."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_hash TEXT UNIQUE,
                    ticker TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    published_date TIMESTAMP,
                    source TEXT DEFAULT 'Business Insider',
                    snippet TEXT,
                    full_text TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment_score REAL,
                    sentiment_label TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker ON articles(ticker)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_published ON articles(published_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_date ON articles(ticker, published_date)
            """)
            
            # Track scraping progress per ticker
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scrape_progress (
                    ticker TEXT PRIMARY KEY,
                    last_page_scraped INTEGER DEFAULT 0,
                    last_scrape_date TIMESTAMP,
                    total_articles INTEGER DEFAULT 0,
                    is_complete BOOLEAN DEFAULT 0
                )
            """)
            
            conn.commit()
    
    def insert_article(self, article: ScrapedArticle) -> bool:
        """Insert article, return True if new, False if duplicate."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO articles 
                    (article_hash, ticker, title, url, published_date, source, snippet, full_text, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.article_hash,
                    article.ticker.upper(),
                    article.title,
                    article.url,
                    article.published_date,
                    article.source,
                    article.snippet,
                    article.full_text,
                    article.scraped_at,
                ))
                return conn.total_changes > 0
        except Exception as e:
            logger.error(f"Failed to insert article: {e}")
            return False
    
    def insert_articles(self, articles: list[ScrapedArticle]) -> int:
        """Bulk insert articles, return count of new articles."""
        new_count = 0
        with sqlite3.connect(self.db_path) as conn:
            for article in articles:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO articles 
                        (article_hash, ticker, title, url, published_date, source, snippet, full_text, scraped_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        article.article_hash,
                        article.ticker.upper(),
                        article.title,
                        article.url,
                        article.published_date,
                        article.source,
                        article.snippet,
                        article.full_text,
                        article.scraped_at,
                    ))
                    if conn.total_changes > 0:
                        new_count += 1
                except Exception as e:
                    logger.debug(f"Failed to insert: {e}")
            conn.commit()
        return new_count
    
    def get_articles(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get articles for a ticker."""
        query = "SELECT * FROM articles WHERE ticker = ?"
        params = [ticker.upper()]
        
        if start_date:
            query += " AND published_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND published_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY published_date DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_scrape_progress(self, ticker: str) -> dict:
        """Get scraping progress for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM scrape_progress WHERE ticker = ?",
                (ticker.upper(),)
            )
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def update_scrape_progress(
        self,
        ticker: str,
        last_page: int,
        total_articles: int,
        is_complete: bool = False,
    ):
        """Update scraping progress for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scrape_progress 
                (ticker, last_page_scraped, last_scrape_date, total_articles, is_complete)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker.upper(), last_page, datetime.now(), total_articles, is_complete))
            conn.commit()
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            stats['total_articles'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT ticker) FROM articles")
            stats['unique_tickers'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT MIN(published_date), MAX(published_date) FROM articles")
            row = cursor.fetchone()
            stats['date_range'] = (row[0], row[1])
            
            cursor = conn.execute("""
                SELECT ticker, COUNT(*) as cnt 
                FROM articles 
                GROUP BY ticker 
                ORDER BY cnt DESC 
                LIMIT 10
            """)
            stats['top_tickers'] = cursor.fetchall()
            
            return stats
    
    def update_sentiment(
        self,
        article_id: int,
        sentiment_score: float,
        sentiment_label: str,
    ):
        """Update headline sentiment for an article."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE articles 
                SET sentiment_score = ?, sentiment_label = ?
                WHERE id = ?
            """, (sentiment_score, sentiment_label, article_id))
            conn.commit()
    
    def update_fulltext_sentiment(
        self,
        article_id: int,
        sentiment_score: float,
        sentiment_label: str,
    ):
        """Update full-text sentiment for an article."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE articles 
                SET fulltext_sentiment_score = ?, fulltext_sentiment_label = ?
                WHERE id = ?
            """, (sentiment_score, sentiment_label, article_id))
            conn.commit()
    
    def update_fulltext_sentiment_batch(
        self,
        updates: list[tuple[int, float, str]],
    ):
        """Batch update full-text sentiment for multiple articles.
        
        Args:
            updates: List of (article_id, sentiment_score, sentiment_label) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                UPDATE articles 
                SET fulltext_sentiment_score = ?, fulltext_sentiment_label = ?
                WHERE id = ?
            """, [(score, label, aid) for aid, score, label in updates])
            conn.commit()
    
    def get_articles_without_fulltext_sentiment(
        self,
        limit: int = 1000,
    ) -> list[dict]:
        """Get articles that have full text but no full-text sentiment yet."""
        query = """
            SELECT id, ticker, title, full_text, published_date 
            FROM articles 
            WHERE full_text IS NOT NULL 
            AND full_text != ''
            AND fulltext_sentiment_score IS NULL
            ORDER BY published_date DESC 
            LIMIT ?
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def count_articles_without_fulltext_sentiment(self) -> int:
        """Count articles needing full-text sentiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE full_text IS NOT NULL 
                AND full_text != ''
                AND fulltext_sentiment_score IS NULL
            """)
            return cursor.fetchone()[0]

    def update_full_text(
        self,
        article_id: int,
        full_text: str,
    ):
        """Update full text for an article."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE articles 
                SET full_text = ?
                WHERE id = ?
            """, (full_text, article_id))
            conn.commit()
    
    def get_articles_without_full_text(
        self,
        limit: int = 1000,
        source: Optional[str] = None,
    ) -> list[dict]:
        """Get articles that don't have full text yet."""
        query = """
            SELECT id, ticker, title, url, published_date, source
            FROM articles 
            WHERE full_text IS NULL
        """
        params = []
        
        if source:
            query += " AND source LIKE ?"
            params.append(f"%{source}%")
        
        query += " ORDER BY published_date DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def count_articles_without_full_text(self) -> int:
        """Count articles without full text."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE full_text IS NULL"
            )
            return cursor.fetchone()[0]
    
    def count_articles_with_full_text(self) -> int:
        """Count articles with full text."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE full_text IS NOT NULL"
            )
            return cursor.fetchone()[0]
    
    def get_articles_without_sentiment(
        self,
        ticker: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Get articles that don't have sentiment scores yet."""
        query = """
            SELECT id, ticker, title, snippet, published_date 
            FROM articles 
            WHERE sentiment_score IS NULL
        """
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        
        query += " ORDER BY published_date DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_articles_for_backtesting(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Get articles with sentiment for backtesting."""
        query = """
            SELECT * FROM articles 
            WHERE ticker = ? 
            AND published_date >= ? 
            AND published_date <= ?
            AND sentiment_score IS NOT NULL
            ORDER BY published_date ASC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (ticker.upper(), start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_daily_sentiment(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """Get aggregated daily sentiment for a ticker."""
        query = """
            SELECT 
                DATE(published_date) as date,
                COUNT(*) as article_count,
                AVG(sentiment_score) as avg_sentiment,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
            FROM articles
            WHERE ticker = ?
            AND sentiment_score IS NOT NULL
        """
        params = [ticker.upper()]
        
        if start_date:
            query += " AND published_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND published_date <= ?"
            params.append(end_date)
        
        query += " GROUP BY DATE(published_date) ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


# ==============================================================================
# BUSINESS INSIDER SCRAPER
# ==============================================================================

class BusinessInsiderScraper:
    """Scraper for Business Insider stock news."""
    
    BASE_URL = "https://markets.businessinsider.com/news/{ticker}-stock"
    
    def __init__(
        self,
        db: Optional[NewsDatabase] = None,
        delay_between_requests: float = 1.0,
        max_retries: int = 3,
    ):
        self.db = db or NewsDatabase()
        self.delay = delay_between_requests
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def _get_page_url(self, ticker: str, page: int = 1) -> str:
        """Build URL for a ticker's news page."""
        base = self.BASE_URL.format(ticker=ticker.lower())
        if page > 1:
            return f"{base}?p={page}"
        return base
    
    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with retries."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats from Business Insider."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Try various formats (order matters - more specific first)
        formats = [
            "%m/%d/%Y %I:%M:%S %p",  # 12/31/2025 2:38:01 PM (Business Insider datetime attr)
            "%m/%d/%Y %I:%M %p",  # 12/31/2025 2:38 PM
            "%Y-%m-%dT%H:%M:%S",  # ISO format
            "%Y-%m-%dT%H:%M:%SZ",  # ISO with Z
            "%b %d, %Y, %I:%M %p",  # Jan 23, 2026, 3:45 PM
            "%b %d, %Y",  # Jan 23, 2026
            "%B %d, %Y",  # January 23, 2026
            "%Y-%m-%d",  # 2026-01-23
            "%m/%d/%Y",  # 01/23/2026
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Handle relative dates like "2 hours ago", "yesterday", "24d"
        if "ago" in date_str.lower():
            return datetime.now()
        if "yesterday" in date_str.lower():
            return datetime.now() - timedelta(days=1)
        
        # Handle "24d" format (days ago)
        match = re.match(r'^(\d+)d$', date_str.strip())
        if match:
            days = int(match.group(1))
            return datetime.now() - timedelta(days=days)
        
        # Handle "2h" format (hours ago)
        match = re.match(r'^(\d+)h$', date_str.strip())
        if match:
            hours = int(match.group(1))
            return datetime.now() - timedelta(hours=hours)
        
        logger.debug(f"Could not parse date: {date_str}")
        return None
    
    def _parse_articles(self, html: str, ticker: str) -> list[ScrapedArticle]:
        """Parse articles from HTML page."""
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        seen_urls = set()
        
        # Primary selector - Business Insider uses .latest-news__story
        article_elements = soup.select('.latest-news__story')
        
        if not article_elements:
            # Fallback selectors
            article_elements = soup.select('.news-item, article.teaser')
        
        for elem in article_elements:
            try:
                # Extract title and URL from link
                link = elem.find('a', href=True)
                if not link:
                    continue
                
                title = link.get_text(strip=True)
                url = link['href']
                
                # Make URL absolute
                if not url.startswith('http'):
                    url = urljoin("https://markets.businessinsider.com", url)
                
                # Skip if not a real article URL or already seen
                if '/news/' not in url and '/article/' not in url:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Extract date from .latest-news__meta or time element
                date_elem = elem.select_one('.latest-news__meta time, time, .date')
                date_str = None
                if date_elem:
                    # Try datetime attribute first, then text
                    date_str = date_elem.get('datetime') or date_elem.get_text(strip=True)
                
                published_date = self._parse_date(date_str) if date_str else None
                
                # Extract source/author from meta
                source = "Business Insider"
                meta = elem.select_one('.latest-news__meta')
                if meta:
                    source_span = meta.find('span')
                    if source_span:
                        src_text = source_span.get_text(strip=True)
                        if src_text and 'ago' not in src_text.lower():
                            source = src_text
                
                if title and len(title) > 10:  # Filter out navigation links
                    article = ScrapedArticle(
                        ticker=ticker.upper(),
                        title=title,
                        url=url,
                        published_date=published_date or datetime.now(),
                        source=source,
                        snippet=None,
                    )
                    articles.append(article)
                    
            except Exception as e:
                logger.debug(f"Failed to parse article element: {e}")
                continue
        
        return articles
    
    def _has_more_pages(self, html: str, current_page: int) -> bool:
        """Check if there are more pages to scrape."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for pagination
        pagination = soup.find(class_=lambda x: x and 'pagination' in x.lower() if x else False)
        if pagination:
            # Check for next page link
            next_link = pagination.find('a', href=lambda x: x and f'p={current_page + 1}' in x if x else False)
            return next_link is not None
        
        # Check if page has articles (if not, we've gone too far)
        articles = self._parse_articles(html, "")
        return len(articles) > 0
    
    def scrape_ticker(
        self,
        ticker: str,
        max_pages: int = 100,
        resume: bool = True,
    ) -> int:
        """
        Scrape all news for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            max_pages: Maximum pages to scrape.
            resume: If True, resume from last scraped page.
            
        Returns:
            Number of new articles scraped.
        """
        ticker = ticker.upper()
        total_new = 0
        
        # Check progress
        progress = self.db.get_scrape_progress(ticker)
        start_page = 1
        if resume and progress:
            if progress.get('is_complete'):
                # Already complete, just check for new articles on page 1
                start_page = 1
                max_pages = 3  # Just check recent pages
            else:
                start_page = progress.get('last_page_scraped', 0) + 1
        
        logger.info(f"Scraping {ticker} starting from page {start_page}")
        
        consecutive_empty = 0
        page = start_page
        
        while page <= max_pages + start_page - 1:
            url = self._get_page_url(ticker, page)
            logger.debug(f"Fetching {url}")
            
            html = self._fetch_page(url)
            if not html:
                logger.warning(f"Failed to fetch page {page} for {ticker}")
                break
            
            articles = self._parse_articles(html, ticker)
            
            if not articles:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.info(f"No more articles for {ticker} after page {page}")
                    self.db.update_scrape_progress(
                        ticker, page, 
                        self.db.get_scrape_progress(ticker).get('total_articles', 0) + total_new,
                        is_complete=True
                    )
                    break
            else:
                consecutive_empty = 0
                new_count = self.db.insert_articles(articles)
                total_new += new_count
                logger.info(f"{ticker} page {page}: {len(articles)} articles, {new_count} new")
            
            # Update progress
            self.db.update_scrape_progress(
                ticker, page,
                self.db.get_scrape_progress(ticker).get('total_articles', 0) + total_new
            )
            
            page += 1
            time.sleep(self.delay)
        
        logger.info(f"Completed {ticker}: {total_new} new articles")
        return total_new
    
    def scrape_tickers(
        self,
        tickers: list[str],
        max_pages_per_ticker: int = 100,
        resume: bool = True,
    ) -> dict[str, int]:
        """Scrape multiple tickers."""
        results = {}
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] Scraping {ticker}...")
            try:
                new_count = self.scrape_ticker(ticker, max_pages_per_ticker, resume)
                results[ticker] = new_count
            except Exception as e:
                logger.error(f"Failed to scrape {ticker}: {e}")
                results[ticker] = -1
            
            # Longer delay between tickers to be polite
            if i < len(tickers):
                time.sleep(self.delay * 2)
        
        return results
    
    def scrape_article_text(self, url: str) -> Optional[str]:
        """
        Scrape the full text content from a Business Insider article.
        
        Args:
            url: The article URL.
            
        Returns:
            The article full text, or None if scraping failed.
        """
        try:
            html = self._fetch_page(url)
            if not html:
                return None
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements first
            for element in soup.select('script, style, nav, header, footer, .ad, .advertisement, .related-articles, .newsletter-signup, .social-share, aside, .share-post-box'):
                element.decompose()
            
            # Try multiple selectors for article content
            article_text = None
            
            # Updated selectors for Business Insider Markets (2025 layout)
            content_selectors = [
                '.news-content',           # Main content div for markets.businessinsider.com
                '.single-article',         # Article wrapper
                '.article-body',           # Old article body selector
                '.content-lock-content',   # Paywall content
                'article .body',           # Generic article body
                '.post-content',           # Blog post style
                '.article-content',        # Alternative article container
                '[data-component="article-body"]',  # Component-based layout
            ]
            
            for selector in content_selectors:
                content = soup.select(selector)
                if content:
                    # For container selectors, get all text
                    text = content[0].get_text(separator='\n', strip=True)
                    if len(text) > 200:  # Must have substantial content
                        article_text = text
                        break
            
            # Fallback: try <main> tag
            if not article_text:
                main = soup.find('main')
                if main:
                    text = main.get_text(separator='\n', strip=True)
                    if len(text) > 200:
                        article_text = text
            
            if article_text:
                # Clean up the text
                lines = article_text.split('\n')
                lines = [line.strip() for line in lines if line.strip()]
                # Filter out very short lines (likely navigation/buttons)
                lines = [line for line in lines if len(line) > 20 or line.endswith(('.', ':', '?', '!'))]
                article_text = '\n'.join(lines)
                
                # Only return if we have substantial content
                if len(article_text) > 200:
                    return article_text[:50000]  # Cap at 50k chars to prevent huge entries
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to scrape article text from {url}: {e}")
            return None
    
    def backfill_full_text(
        self,
        batch_size: int = 100,
        max_articles: Optional[int] = None,
        delay_between_articles: float = 1.0,
    ) -> dict:
        """
        Backfill full text for articles that don't have it.
        
        Args:
            batch_size: Number of articles to process in each batch.
            max_articles: Maximum articles to process (None = all).
            delay_between_articles: Delay between article fetches.
            
        Returns:
            Stats dict with success/failure counts.
        """
        stats = {
            'total_processed': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
        }
        
        total_without = self.db.count_articles_without_full_text()
        logger.info(f"Found {total_without:,} articles without full text")
        
        if max_articles:
            total_to_process = min(total_without, max_articles)
        else:
            total_to_process = total_without
        
        processed = 0
        
        while processed < total_to_process:
            # Get next batch
            articles = self.db.get_articles_without_full_text(limit=batch_size)
            
            if not articles:
                break
            
            for article in articles:
                if processed >= total_to_process:
                    break
                
                url = article['url']
                article_id = article['id']
                
                # Skip non-Business Insider URLs (SeekingAlpha, etc. block scraping)
                if 'businessinsider' not in url.lower():
                    # Mark with empty string so we don't retry
                    self.db.update_full_text(article_id, "")
                    stats['skipped'] += 1
                    stats['total_processed'] += 1
                    processed += 1
                    logger.debug(f"[{processed}/{total_to_process}] Skipped external: {url[:50]}...")
                    continue
                
                logger.debug(f"[{processed+1}/{total_to_process}] Fetching: {url}")
                
                full_text = self.scrape_article_text(url)
                
                if full_text:
                    self.db.update_full_text(article_id, full_text)
                    stats['success'] += 1
                    logger.info(f"[{processed+1}/{total_to_process}] ✓ {article['ticker']}: {article['title'][:50]}... ({len(full_text)} chars)")
                else:
                    stats['failed'] += 1
                    # Mark as empty string so we don't retry forever
                    self.db.update_full_text(article_id, "")
                    logger.warning(f"[{processed+1}/{total_to_process}] ✗ Failed: {url}")
                
                stats['total_processed'] += 1
                processed += 1
                
                # Rate limiting
                time.sleep(delay_between_articles)
        
        logger.info(f"Backfill complete: {stats['success']} success, {stats['failed']} failed, {stats['skipped']} skipped")
        return stats


# ==============================================================================
# S&P 500 TICKERS
# ==============================================================================

# Full S&P 500 list (as of Jan 2026)
SP500_TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV",
    "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA",
    "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB", "BIO",
    "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX",
    "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD",
    "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI",
    "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB",
    "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTLT",
    "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", "DD",
    "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR",
    "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN",
    "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES", "ESS",
    "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG",
    "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB",
    "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE",
    "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG",
    "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA",
    "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL",
    "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX",
    "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM",
    "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ",
    "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC",
    "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV",
    "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP",
    "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX",
    "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK",
    "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU",
    "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW",
    "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW",
    "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW", "PODD",
    "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PXD", "PYPL",
    "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK",
    "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW",
    "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE",
    "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK",
    "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX",
    "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO",
    "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR",
    "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VFC", "VICI",
    "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ",
    "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB",
    "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM",
    "ZBH", "ZBRA", "ZTS",
]

def get_sp500_tickers() -> list[str]:
    """Get list of S&P 500 tickers."""
    # Use embedded list - more reliable than scraping Wikipedia
    return SP500_TICKERS.copy()


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Scrape Business Insider stock news")
    parser.add_argument("--tickers", nargs="+", help="Tickers to scrape")
    parser.add_argument("--sp500", action="store_true", help="Scrape all S&P 500 tickers")
    parser.add_argument("--update", action="store_true", help="Update existing tickers only")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages per ticker")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    parser.add_argument("--backfill-text", action="store_true", help="Backfill full text for existing articles")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles to backfill")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for backfill")
    
    args = parser.parse_args()
    
    db = NewsDatabase()
    
    if args.stats:
        stats = db.get_stats()
        print(f"\nDatabase: {db.db_path}")
        print(f"Total articles: {stats['total_articles']:,}")
        print(f"Unique tickers: {stats['unique_tickers']}")
        print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"\nFull text status:")
        print(f"  With full text: {db.count_articles_with_full_text():,}")
        print(f"  Without full text: {db.count_articles_without_full_text():,}")
        print("\nTop tickers:")
        for ticker, count in stats['top_tickers']:
            print(f"  {ticker}: {count:,} articles")
    elif args.backfill_text:
        print(f"\nBackfilling article full text...")
        print(f"  Articles without text: {db.count_articles_without_full_text():,}")
        print(f"  Max articles: {args.max_articles or 'all'}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Delay: {args.delay}s\n")
        
        scraper = BusinessInsiderScraper(db=db, delay_between_requests=args.delay)
        stats = scraper.backfill_full_text(
            batch_size=args.batch_size,
            max_articles=args.max_articles,
            delay_between_articles=args.delay,
        )
        
        print("\n" + "="*50)
        print("BACKFILL COMPLETE")
        print("="*50)
        print(f"Total processed: {stats['total_processed']:,}")
        print(f"Success: {stats['success']:,}")
        print(f"Failed: {stats['failed']:,}")
        print(f"Skipped: {stats['skipped']:,}")
        print(f"\nArticles with full text: {db.count_articles_with_full_text():,}")
    else:
        scraper = BusinessInsiderScraper(db=db, delay_between_requests=args.delay)
        
        if args.sp500:
            tickers = get_sp500_tickers()
            print(f"Scraping {len(tickers)} S&P 500 tickers...")
        elif args.update:
            # Get tickers already in DB
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT ticker FROM articles")
                tickers = [row[0] for row in cursor.fetchall()]
            print(f"Updating {len(tickers)} existing tickers...")
        elif args.tickers:
            tickers = args.tickers
        else:
            parser.print_help()
            exit(1)
        
        results = scraper.scrape_tickers(tickers, max_pages_per_ticker=args.max_pages)
        
        print("\n" + "="*50)
        print("SCRAPING COMPLETE")
        print("="*50)
        total = sum(v for v in results.values() if v >= 0)
        failed = sum(1 for v in results.values() if v < 0)
        print(f"New articles: {total:,}")
        print(f"Failed tickers: {failed}")
        
        stats = db.get_stats()
        print(f"\nTotal in database: {stats['total_articles']:,} articles")
