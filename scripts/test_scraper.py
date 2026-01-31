"""Quick test of the Business Insider scraper."""

import sys
sys.path.insert(0, r"C:\Users\Andrew\projects\auto_researcher")

from src.auto_researcher.data.news_scraper import BusinessInsiderScraper, NewsDatabase

# Quick test on 2 pages of NVDA
db = NewsDatabase()
scraper = BusinessInsiderScraper(db=db, delay_between_requests=1.0)

# Scrape just 3 pages to test
print("Scraping NVDA (3 pages)...")
new_count = scraper.scrape_ticker('NVDA', max_pages=3, resume=False)
print(f'New articles scraped: {new_count}')

# Check what we got
articles = db.get_articles('NVDA', limit=10)
print(f'\nLatest articles:')
for art in articles[:5]:
    date_str = art['published_date'][:10] if art['published_date'] else 'N/A'
    print(f"  [{date_str}] {art['title'][:60]}...")
    print(f"    Source: {art['source']}")

# Show stats
stats = db.get_stats()
print(f"\nDB Stats: {stats['total_articles']} articles, {stats['unique_tickers']} tickers")
