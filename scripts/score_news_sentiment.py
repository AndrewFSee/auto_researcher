"""
Score scraped news articles with FinBERT sentiment.

This script processes articles in our news database and adds sentiment scores.
This enables backtesting of news sentiment as an alpha signal.

Usage:
    # Score all articles without sentiment
    python scripts/score_news_sentiment.py
    
    # Score specific ticker
    python scripts/score_news_sentiment.py --ticker NVDA
    
    # Limit number of articles
    python scripts/score_news_sentiment.py --limit 1000
"""

import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_researcher.data.news_scraper import NewsDatabase


def score_articles_with_finbert(
    db: NewsDatabase,
    ticker: str = None,
    limit: int = 5000,
    batch_size: int = 32,
):
    """Score articles with FinBERT sentiment."""
    
    # Load FinBERT
    logger.info("Loading FinBERT model...")
    try:
        from src.auto_researcher.agents.finbert_sentiment import create_analyzer
        analyzer = create_analyzer(prefer_finbert=True)
        logger.info(f"Loaded {type(analyzer).__name__}")
    except ImportError as e:
        logger.error(f"Failed to load sentiment analyzer: {e}")
        return
    
    # Get articles without sentiment
    logger.info("Fetching articles without sentiment...")
    articles = db.get_articles_without_sentiment(ticker=ticker, limit=limit)
    
    if not articles:
        logger.info("No articles to process")
        return
    
    logger.info(f"Processing {len(articles)} articles...")
    
    # Process in batches
    start_time = time.time()
    processed = 0
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        
        # Prepare texts for batch processing
        texts = []
        for art in batch:
            # Use title + snippet if available
            text = art['title']
            if art.get('snippet'):
                text += ". " + art['snippet']
            texts.append(text)
        
        # Score batch
        try:
            results = analyzer.analyze_batch(texts)
            
            # Update database
            for art, result in zip(batch, results):
                # FinBERTResult uses properties, not dict keys
                score = result.sentiment_score  # -1 to +1
                label = result.label  # positive, negative, neutral
                db.update_sentiment(art['id'], score, label)
            
            processed += len(batch)
            
            # Progress update
            if processed % 500 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = (len(articles) - processed) / rate if rate > 0 else 0
                logger.info(
                    f"Processed {processed}/{len(articles)} "
                    f"({processed/len(articles)*100:.1f}%) "
                    f"- {rate:.1f} articles/sec, "
                    f"~{remaining/60:.1f} min remaining"
                )
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            continue
    
    elapsed = time.time() - start_time
    logger.info(f"Completed: {processed} articles in {elapsed/60:.1f} minutes ({processed/elapsed:.1f}/sec)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Score news articles with FinBERT")
    parser.add_argument("--ticker", type=str, help="Specific ticker to process")
    parser.add_argument("--limit", type=int, default=10000, help="Max articles to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--stats", action="store_true", help="Show sentiment stats")
    
    args = parser.parse_args()
    
    db = NewsDatabase()
    
    if args.stats:
        # Show sentiment stats
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN sentiment_score IS NOT NULL THEN 1 ELSE 0 END) as scored,
                    AVG(sentiment_score) as avg_score
                FROM articles
            """)
            row = cursor.fetchone()
            print(f"Total articles: {row[0]:,}")
            print(f"Scored articles: {row[1]:,} ({row[1]/row[0]*100:.1f}%)")
            print(f"Average sentiment: {row[2]:.3f}" if row[2] else "No scores yet")
            
            cursor = conn.execute("""
                SELECT ticker, COUNT(*) as cnt, AVG(sentiment_score) as avg
                FROM articles
                WHERE sentiment_score IS NOT NULL
                GROUP BY ticker
                ORDER BY cnt DESC
                LIMIT 10
            """)
            print("\nSentiment by ticker:")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]:,} articles, avg score: {row[2]:.3f}")
        return
    
    # Score articles
    score_articles_with_finbert(
        db=db,
        ticker=args.ticker,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
