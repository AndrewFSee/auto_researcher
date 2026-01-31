"""
Topic Sentiment Backtest.

Tests whether topic-specific sentiment has more predictive power 
than generic sentiment (FinBERT).

Hypothesis:
- Litigation + negative sentiment → stronger bearish signal
- Earnings + positive sentiment → stronger bullish signal  
- Management changes → asymmetric (departures more impactful)
- Generic sentiment averages across topics, diluting signal

Methodology:
1. Load news articles with FinBERT sentiment scores
2. Classify each article by topic
3. Compute topic-adjusted sentiment
4. Compare predictive power: FinBERT vs Topic-adjusted vs Topic-specific
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_researcher.models.topic_sentiment import TopicSentimentModel, HIGH_SIGNAL_TOPICS


def load_news_with_topics(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load news and add topic classifications."""
    db_path = db_path or Path(__file__).parent.parent / "data" / "news.db"
    
    logger.info(f"Loading news from {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("""
            SELECT 
                ticker,
                DATE(published_date) as date,
                sentiment_score as finbert_sentiment,
                sentiment_label as finbert_label,
                title,
                snippet
            FROM articles
            WHERE sentiment_score IS NOT NULL
            ORDER BY ticker, date
        """, conn)
    
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df):,} articles")
    
    # Classify topics
    model = TopicSentimentModel()
    
    logger.info("Classifying topics...")
    topics = []
    topic_sentiments = []
    
    for _, row in df.iterrows():
        text = row['title']
        if pd.notna(row.get('snippet')):
            text += ". " + str(row['snippet'])
        
        result = model.analyze_article(text)
        topics.append(result.topic.primary_topic)
        topic_sentiments.append(result.topic_adjusted_sentiment)
    
    df['topic'] = topics
    df['topic_sentiment'] = topic_sentiments
    
    logger.info(f"Topic distribution:\n{df['topic'].value_counts()}")
    
    return df


def load_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Load price data for tickers."""
    import yfinance as yf
    
    logger.info(f"Loading price data for {len(tickers)} tickers...")
    
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    
    if len(tickers) == 1:
        prices = data[['Close']].copy()
        prices.columns = [tickers[0]]
    else:
        prices = data['Close'].copy()
    
    return prices


def compute_forward_returns(prices: pd.DataFrame, horizons: list = [1, 5, 10]) -> dict:
    """Compute forward returns for each horizon."""
    returns = {}
    for h in horizons:
        ret = prices.pct_change(h).shift(-h)  # Forward returns
        returns[f'fwd_{h}d'] = ret
    return returns


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level sentiment to daily ticker-level features.
    
    Creates features for both FinBERT and topic-adjusted sentiment.
    """
    # Group by ticker-date
    grouped = df.groupby(['ticker', 'date'])
    
    # FinBERT aggregates
    finbert_agg = grouped.agg(
        finbert_mean=('finbert_sentiment', 'mean'),
        finbert_std=('finbert_sentiment', 'std'),
        news_count=('finbert_sentiment', 'count'),
    ).reset_index()
    
    # Topic sentiment aggregates  
    topic_agg = grouped.agg(
        topic_mean=('topic_sentiment', 'mean'),
        topic_std=('topic_sentiment', 'std'),
    ).reset_index()
    
    # High-signal topic sentiment (litigation, earnings, management, M&A)
    high_signal_df = df[df['topic'].isin(HIGH_SIGNAL_TOPICS)]
    if len(high_signal_df) > 0:
        high_signal_agg = high_signal_df.groupby(['ticker', 'date']).agg(
            high_signal_mean=('topic_sentiment', 'mean'),
            high_signal_count=('topic_sentiment', 'count'),
        ).reset_index()
    else:
        high_signal_agg = pd.DataFrame(columns=['ticker', 'date', 'high_signal_mean', 'high_signal_count'])
    
    # Specific topic sentiments
    topic_specific = {}
    for topic in ['litigation', 'earnings', 'management', 'mna']:
        topic_df = df[df['topic'] == topic]
        if len(topic_df) > 0:
            topic_agg_specific = topic_df.groupby(['ticker', 'date']).agg(
                **{f'{topic}_sentiment': ('topic_sentiment', 'mean')},
                **{f'{topic}_count': ('topic_sentiment', 'count')},
            ).reset_index()
            topic_specific[topic] = topic_agg_specific
    
    # Merge all
    daily = finbert_agg.merge(topic_agg[['ticker', 'date', 'topic_mean', 'topic_std']], 
                               on=['ticker', 'date'], how='left')
    daily = daily.merge(high_signal_agg, on=['ticker', 'date'], how='left')
    
    for topic, topic_df in topic_specific.items():
        daily = daily.merge(topic_df, on=['ticker', 'date'], how='left')
    
    # Fill NaN
    daily = daily.fillna(0)
    
    logger.info(f"Created {len(daily):,} ticker-day observations")
    
    return daily


def analyze_signal_strength(df: pd.DataFrame, prices: pd.DataFrame, forward_returns: dict):
    """
    Analyze predictive power of different sentiment signals.
    
    Metrics:
    - IC (Information Coefficient): Spearman correlation with forward returns
    - Hit rate: Fraction of correct sign predictions
    - Return spread: Long top quintile vs short bottom quintile
    """
    results = []
    
    # Features to test
    features = ['finbert_mean', 'topic_mean', 'high_signal_mean', 
                'litigation_sentiment', 'earnings_sentiment']
    
    for feature in features:
        if feature not in df.columns or df[feature].std() == 0:
            continue
            
        for horizon, ret_df in forward_returns.items():
            # Merge sentiment with returns
            test_data = []
            for ticker in df['ticker'].unique():
                ticker_sent = df[df['ticker'] == ticker].set_index('date')
                if ticker not in ret_df.columns:
                    continue
                ticker_ret = ret_df[ticker]
                
                merged = pd.concat([ticker_sent[feature], ticker_ret], axis=1, join='inner')
                merged.columns = ['signal', 'return']
                test_data.append(merged)
            
            if not test_data:
                continue
                
            combined = pd.concat(test_data).dropna()
            
            if len(combined) < 30:
                continue
            
            # IC (Spearman correlation)
            ic, ic_pval = stats.spearmanr(combined['signal'], combined['return'])
            
            # Hit rate (correct sign prediction)
            hits = ((combined['signal'] > 0) & (combined['return'] > 0)) | \
                   ((combined['signal'] < 0) & (combined['return'] < 0))
            hit_rate = hits.mean()
            
            # Quintile spread
            combined['quintile'] = pd.qcut(combined['signal'], 5, labels=False, duplicates='drop')
            q_returns = combined.groupby('quintile')['return'].mean()
            if len(q_returns) >= 2:
                spread = q_returns.iloc[-1] - q_returns.iloc[0]  # Top - Bottom
            else:
                spread = 0
            
            results.append({
                'feature': feature,
                'horizon': horizon,
                'ic': ic,
                'ic_pval': ic_pval,
                'hit_rate': hit_rate,
                'spread': spread,
                'n_obs': len(combined),
            })
    
    return pd.DataFrame(results)


def run_backtest():
    """Run the full topic sentiment backtest."""
    
    # Load news with topic classifications
    df = load_news_with_topics()
    
    # Get unique tickers
    tickers = df['ticker'].unique().tolist()
    logger.info(f"Testing {len(tickers)} tickers")
    
    # Date range
    start_date = (df['date'].min() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = (df['date'].max() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Load prices
    prices = load_price_data(tickers, start_date, end_date)
    
    # Compute forward returns
    forward_returns = compute_forward_returns(prices, horizons=[1, 5, 10])
    
    # Aggregate daily sentiment
    daily = aggregate_daily_sentiment(df)
    
    # Analyze signal strength
    results = analyze_signal_strength(daily, prices, forward_returns)
    
    # Display results
    print("\n" + "="*80)
    print("TOPIC SENTIMENT BACKTEST RESULTS")
    print("="*80)
    
    if len(results) == 0:
        print("No results - insufficient data")
        return
    
    # Format results
    results['ic_str'] = results.apply(
        lambda r: f"{r['ic']:.4f}" + ("*" if r['ic_pval'] < 0.05 else ""), axis=1
    )
    
    # Pivot by feature and horizon
    for feature in results['feature'].unique():
        feat_df = results[results['feature'] == feature]
        print(f"\n{feature.upper()}")
        print("-" * 60)
        for _, row in feat_df.iterrows():
            print(f"  {row['horizon']:8s}: IC={row['ic']:+.4f} (p={row['ic_pval']:.3f}), "
                  f"Hit={row['hit_rate']:.1%}, Spread={row['spread']:+.2%}, N={row['n_obs']}")
    
    # Compare FinBERT vs Topic
    print("\n" + "="*80)
    print("COMPARISON: FinBERT vs Topic-Adjusted Sentiment")
    print("="*80)
    
    fb_results = results[results['feature'] == 'finbert_mean']
    topic_results = results[results['feature'] == 'topic_mean']
    
    for horizon in ['fwd_1d', 'fwd_5d', 'fwd_10d']:
        fb = fb_results[fb_results['horizon'] == horizon]
        tp = topic_results[topic_results['horizon'] == horizon]
        
        if len(fb) > 0 and len(tp) > 0:
            fb_ic = fb['ic'].iloc[0]
            tp_ic = tp['ic'].iloc[0]
            improvement = (abs(tp_ic) - abs(fb_ic)) / abs(fb_ic) * 100 if fb_ic != 0 else 0
            
            print(f"\n{horizon}:")
            print(f"  FinBERT IC:       {fb_ic:+.4f}")
            print(f"  Topic IC:         {tp_ic:+.4f}")
            print(f"  Improvement:      {improvement:+.1f}%")
    
    # High-signal topics
    print("\n" + "="*80)
    print("HIGH-SIGNAL TOPICS (Litigation, Earnings, M&A, Management)")
    print("="*80)
    
    hs_results = results[results['feature'] == 'high_signal_mean']
    if len(hs_results) > 0:
        for _, row in hs_results.iterrows():
            sig = "*" if row['ic_pval'] < 0.05 else ""
            print(f"  {row['horizon']}: IC={row['ic']:+.4f}{sig}, N={row['n_obs']}")
    
    # Litigation-specific (should be most predictive for negative)
    print("\n" + "="*80)
    print("LITIGATION SENTIMENT (Expected: Strong negative predictor)")
    print("="*80)
    
    lit_results = results[results['feature'] == 'litigation_sentiment']
    if len(lit_results) > 0:
        for _, row in lit_results.iterrows():
            sig = "*" if row['ic_pval'] < 0.05 else ""
            print(f"  {row['horizon']}: IC={row['ic']:+.4f}{sig}, Hit={row['hit_rate']:.1%}, N={row['n_obs']}")
    else:
        print("  No litigation articles in dataset")
    
    return results


if __name__ == "__main__":
    results = run_backtest()
