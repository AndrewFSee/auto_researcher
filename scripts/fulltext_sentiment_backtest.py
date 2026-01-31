"""
Full-Text Sentiment Backtest
============================
Compare headline sentiment vs full-text sentiment signal strength.

This script:
1. Samples articles with both headline and full-text content
2. Runs FinBERT on full-text for the sample
3. Saves full-text sentiment scores to DB
4. Compares IC (information coefficient) of headline vs full-text sentiment
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.auto_researcher.data.news_scraper import NewsDatabase


def get_sample_articles(db_path: str, n_samples: int = 10000) -> pd.DataFrame:
    """Get a stratified sample of articles with full text."""
    conn = sqlite3.connect(db_path)
    
    # Get articles with full text - ensure they're old enough for forward returns
    # Articles must be at least 45 days old for 40-day forward returns
    query = """
        SELECT 
            id, ticker, title, full_text, published_date,
            sentiment_score as headline_sentiment,
            sentiment_label as headline_label
        FROM articles 
        WHERE full_text IS NOT NULL 
        AND full_text != ''
        AND LENGTH(full_text) > 100
        AND sentiment_score IS NOT NULL
        AND published_date < date('now', '-45 days')
        ORDER BY RANDOM()
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(n_samples,))
    conn.close()
    
    print(f"Loaded {len(df):,} articles with headline sentiment and full text")
    return df


def run_finbert_batch(texts: list[str], batch_size: int = 16) -> list[tuple[float, str]]:
    """Run FinBERT sentiment on a batch of texts."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    results = []
    labels_map = {0: "positive", 1: "negative", 2: "neutral"}
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Truncate to first 512 tokens worth of text (roughly first 2000 chars)
        batch_texts = [t[:2000] if t else "" for t in batch_texts]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
            # Convert to sentiment scores: positive - negative (range -1 to 1)
            scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
        
        for score, pred in zip(scores, predictions):
            results.append((float(score), labels_map[pred]))
    
    return results


def get_forward_returns(tickers: list[str], dates: list, horizons: list[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """Get forward returns for tickers from dates."""
    import yfinance as yf
    
    # Get unique tickers
    unique_tickers = list(set(tickers))
    
    # Determine date range
    min_date = min(dates) - timedelta(days=5)
    max_date = max(dates) + timedelta(days=max(horizons) + 10)
    
    print(f"Fetching price data for {len(unique_tickers)} tickers...")
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    
    # Download all price data
    prices = {}
    for ticker in tqdm(unique_tickers, desc="Downloading prices"):
        try:
            df = yf.download(ticker, start=min_date, end=max_date, progress=False)
            if len(df) > 0:
                # Handle both old and new yfinance column formats
                if 'Adj Close' in df.columns:
                    prices[ticker] = df['Adj Close']
                elif 'Close' in df.columns:
                    prices[ticker] = df['Close']
                elif ('Close', ticker) in df.columns:
                    # New multi-level column format
                    prices[ticker] = df[('Close', ticker)]
                else:
                    # Try to get the Close column from any structure
                    close_cols = [c for c in df.columns if 'Close' in str(c)]
                    if close_cols:
                        prices[ticker] = df[close_cols[0]]
        except Exception as e:
            pass
    
    print(f"Successfully downloaded data for {len(prices)} tickers")
    
    # Calculate forward returns for each article
    returns_data = []
    valid_count = 0
    for ticker, date in zip(tickers, dates):
        if ticker not in prices:
            returns_data.append({f'ret_{h}d': np.nan for h in horizons})
            continue
        
        price_series = prices[ticker]
        
        # Convert date to pandas Timestamp if needed
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Make date tz-naive for comparison
        if date.tz is not None:
            date = date.tz_localize(None)
        
        # Find the trading day on or after the article date
        future_prices = price_series[price_series.index >= date]
        if len(future_prices) < max(horizons) + 1:
            returns_data.append({f'ret_{h}d': np.nan for h in horizons})
            continue
        
        start_price = future_prices.iloc[0]
        row = {}
        for h in horizons:
            if len(future_prices) > h:
                row[f'ret_{h}d'] = (future_prices.iloc[h] / start_price - 1) * 100
            else:
                row[f'ret_{h}d'] = np.nan
        returns_data.append(row)
        valid_count += 1
    
    print(f"Calculated returns for {valid_count:,} articles")
    return pd.DataFrame(returns_data)


def calculate_ic(sentiment: pd.Series, returns: pd.Series) -> tuple[float, float]:
    """Calculate Spearman IC and p-value."""
    mask = sentiment.notna() & returns.notna()
    if mask.sum() < 30:
        return np.nan, np.nan
    
    # Convert to numpy arrays to avoid pandas issues with scipy
    sent_vals = np.array(sentiment[mask].tolist())
    ret_vals = np.array(returns[mask].tolist())
    
    ic, pval = stats.spearmanr(sent_vals, ret_vals)
    return ic, pval


def run_backtest(sample_size: int = 10000, batch_size: int = 16):
    """Run the full backtest comparing headline vs full-text sentiment."""
    
    db = NewsDatabase()
    
    # Step 1: Get sample articles
    print("\n" + "="*60)
    print("STEP 1: Loading sample articles")
    print("="*60)
    df = get_sample_articles(db.db_path, sample_size)
    
    # Step 2: Run FinBERT on full text
    print("\n" + "="*60)
    print("STEP 2: Running FinBERT on full text")
    print("="*60)
    
    texts = df['full_text'].tolist()
    sentiment_results = run_finbert_batch(texts, batch_size=batch_size)
    
    df['fulltext_sentiment'] = [r[0] for r in sentiment_results]
    df['fulltext_label'] = [r[1] for r in sentiment_results]
    
    # Step 3: Save to database
    print("\n" + "="*60)
    print("STEP 3: Saving full-text sentiment to database")
    print("="*60)
    
    updates = [
        (int(row['id']), row['fulltext_sentiment'], row['fulltext_label'])
        for _, row in df.iterrows()
    ]
    db.update_fulltext_sentiment_batch(updates)
    print(f"Saved {len(updates):,} full-text sentiment scores to database")
    
    # Step 4: Get forward returns
    print("\n" + "="*60)
    print("STEP 4: Fetching forward returns")
    print("="*60)
    
    df['published_date'] = pd.to_datetime(df['published_date'])
    returns_df = get_forward_returns(
        df['ticker'].tolist(), 
        df['published_date'].tolist(),
        horizons=[1, 5, 10, 20, 40]
    )
    
    # Flatten any multi-level columns from returns_df
    if isinstance(returns_df.columns, pd.MultiIndex):
        returns_df.columns = [col[0] if isinstance(col, tuple) else col for col in returns_df.columns]
    
    df = pd.concat([df.reset_index(drop=True), returns_df.reset_index(drop=True)], axis=1)
    
    # Step 5: Calculate ICs
    print("\n" + "="*60)
    print("STEP 5: Comparing Headline vs Full-Text Sentiment IC")
    print("="*60)
    
    # Debug: check what data we have
    print(f"\nData diagnostics:")
    print(f"  headline_sentiment non-null: {df['headline_sentiment'].notna().sum():,}")
    print(f"  fulltext_sentiment non-null: {df['fulltext_sentiment'].notna().sum():,}")
    for h in [1, 5, 10, 20, 40]:
        col = f'ret_{h}d'
        if col in df.columns:
            print(f"  {col} non-null: {df[col].notna().sum():,}")
        else:
            print(f"  {col} MISSING from dataframe!")
    
    horizons = [1, 5, 10, 20, 40]
    results = []
    
    for h in horizons:
        ret_col = f'ret_{h}d'
        
        headline_ic, headline_pval = calculate_ic(df['headline_sentiment'], df[ret_col])
        fulltext_ic, fulltext_pval = calculate_ic(df['fulltext_sentiment'], df[ret_col])
        
        results.append({
            'horizon': f'{h}d',
            'headline_ic': headline_ic,
            'headline_pval': headline_pval,
            'fulltext_ic': fulltext_ic,
            'fulltext_pval': fulltext_pval,
            'improvement': fulltext_ic - headline_ic if not np.isnan(fulltext_ic) else np.nan
        })
    
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "-"*70)
    print(f"{'Horizon':<10} {'Headline IC':<15} {'FullText IC':<15} {'Improvement':<15}")
    print("-"*70)
    
    for _, row in results_df.iterrows():
        h_sig = "***" if row['headline_pval'] < 0.01 else "**" if row['headline_pval'] < 0.05 else "*" if row['headline_pval'] < 0.1 else ""
        f_sig = "***" if row['fulltext_pval'] < 0.01 else "**" if row['fulltext_pval'] < 0.05 else "*" if row['fulltext_pval'] < 0.1 else ""
        
        print(f"{row['horizon']:<10} {row['headline_ic']:+.4f}{h_sig:<5} {row['fulltext_ic']:+.4f}{f_sig:<5} {row['improvement']:+.4f}")
    
    print("-"*70)
    print("Significance: *** p<0.01, ** p<0.05, * p<0.1")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nSample size: {len(df):,} articles")
    print(f"Date range: {df['published_date'].min().date()} to {df['published_date'].max().date()}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    
    # Correlation between headline and fulltext sentiment
    corr = df['headline_sentiment'].corr(df['fulltext_sentiment'])
    print(f"\nCorrelation between headline & fulltext sentiment: {corr:.3f}")
    
    # Label distribution comparison
    print("\nSentiment Label Distribution:")
    print("  Headline:", df['headline_label'].value_counts().to_dict())
    print("  FullText:", df['fulltext_label'].value_counts().to_dict())
    
    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    avg_improvement = results_df['improvement'].mean()
    sig_improvements = sum(1 for _, r in results_df.iterrows() 
                          if r['fulltext_pval'] < 0.05 and r['improvement'] > 0)
    
    if avg_improvement > 0.01 and sig_improvements >= 2:
        print("✓ Full-text sentiment shows MEANINGFUL improvement over headline sentiment")
        print("  → Recommend running full-text sentiment on all articles")
    elif avg_improvement > 0:
        print("~ Full-text sentiment shows MARGINAL improvement")
        print("  → May not be worth the computational cost")
    else:
        print("✗ Full-text sentiment does NOT improve over headline sentiment")
        print("  → Stick with headline sentiment")
    
    return results_df, df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare headline vs full-text sentiment")
    parser.add_argument("--samples", type=int, default=5000, help="Number of articles to sample")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for FinBERT")
    
    args = parser.parse_args()
    
    results, data = run_backtest(sample_size=args.samples, batch_size=args.batch_size)
