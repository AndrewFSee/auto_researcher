"""Check database stats."""
import sys
sys.path.insert(0, r"C:\Users\Andrew\projects\auto_researcher")

from src.auto_researcher.data.news_scraper import NewsDatabase
import os

db = NewsDatabase()
stats = db.get_stats()

print(f"Database: {db.db_path}")
print(f"Total articles: {stats['total_articles']:,}")
print(f"Date range: {stats['date_range'][0][:10]} to {stats['date_range'][1][:10]}")
print(f"Unique tickers: {stats['unique_tickers']}")

# Check DB file size
size_mb = os.path.getsize(db.db_path) / 1024 / 1024
print(f"Database size: {size_mb:.1f} MB")

# Estimate for S&P 500
print(f"\nEstimate for S&P 500:")
avg_per_ticker = stats['total_articles'] / max(stats['unique_tickers'], 1)
print(f"  Avg articles per ticker: {avg_per_ticker:,.0f}")
print(f"  Est. total for 500 tickers: {avg_per_ticker * 500:,.0f} articles")
print(f"  Est. DB size: {size_mb * 500 / max(stats['unique_tickers'], 1):.0f} MB")
