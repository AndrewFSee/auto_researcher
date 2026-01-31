"""
News Sentiment Backtesting Model v2.

Alternative approaches after baseline showed no signal:

1. EVENT-DRIVEN: Focus on unusual news volume spikes
   - High volume days may contain more material information
   - Filter to days with volume_zscore > 2 (unusual coverage)

2. LONGER LOOKBACK: Weekly/monthly aggregation
   - Daily is too noisy, aggregate over 5/10/20 day windows
   - Sentiment drift over weeks may be more predictive

3. EXTREME SENTIMENT: Focus on tail events
   - Very positive or very negative sentiment days only
   - Middle-of-distribution days are noise
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_news_sentiment(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load all news sentiment data from database."""
    db_path = db_path or Path(__file__).parent.parent / "data" / "news.db"
    
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("""
            SELECT 
                ticker,
                DATE(published_date) as date,
                sentiment_score,
                sentiment_label,
                title
            FROM articles
            WHERE sentiment_score IS NOT NULL
            ORDER BY ticker, date
        """, conn)
    
    df['date'] = pd.to_datetime(df['date'])
    return df


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily ticker-level."""
    daily = df.groupby(['ticker', 'date']).agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        news_count=('sentiment_score', 'count'),
        positive_ratio=('sentiment_label', lambda x: (x == 'positive').mean()),
        negative_ratio=('sentiment_label', lambda x: (x == 'negative').mean()),
    ).reset_index()
    
    daily['sentiment_std'] = daily['sentiment_std'].fillna(0)
    return daily


def add_features_with_lookbacks(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add features with multiple lookback windows.
    
    Key changes from v1:
    - Multiple lookback windows (5, 10, 20, 60 days)
    - Volume z-score for event detection
    - Sentiment acceleration (2nd derivative)
    - Cumulative sentiment over longer periods
    """
    dfs = []
    
    for ticker, gdf in daily.groupby('ticker'):
        gdf = gdf.sort_values('date').copy()
        
        # Multiple lookback windows for sentiment
        for window in [5, 10, 20, 60]:
            gdf[f'sentiment_ma{window}'] = gdf['sentiment_mean'].rolling(window, min_periods=max(3, window//3)).mean()
            gdf[f'news_count_ma{window}'] = gdf['news_count'].rolling(window, min_periods=max(3, window//3)).mean()
        
        # Sentiment momentum at different horizons
        gdf['sentiment_mom5'] = gdf['sentiment_mean'] - gdf['sentiment_mean'].shift(5)
        gdf['sentiment_mom10'] = gdf['sentiment_mean'] - gdf['sentiment_mean'].shift(10)
        gdf['sentiment_mom20'] = gdf['sentiment_mean'] - gdf['sentiment_mean'].shift(20)
        
        # Sentiment acceleration (change in momentum)
        gdf['sentiment_accel'] = gdf['sentiment_mom5'] - gdf['sentiment_mom5'].shift(5)
        
        # Z-scores for surprise detection
        for window in [20, 60]:
            rolling_mean = gdf['sentiment_mean'].rolling(window, min_periods=10).mean()
            rolling_std = gdf['sentiment_mean'].rolling(window, min_periods=10).std()
            gdf[f'sentiment_zscore{window}'] = (gdf['sentiment_mean'] - rolling_mean) / rolling_std.clip(lower=0.01)
            
            vol_mean = gdf['news_count'].rolling(window, min_periods=5).mean()
            vol_std = gdf['news_count'].rolling(window, min_periods=5).std()
            gdf[f'volume_zscore{window}'] = (gdf['news_count'] - vol_mean) / vol_std.clip(lower=0.1)
        
        # Cumulative sentiment (sum over window)
        for window in [5, 10, 20]:
            gdf[f'sentiment_sum{window}'] = gdf['sentiment_mean'].rolling(window, min_periods=1).sum()
        
        # Trend strength (R-squared of linear fit over 20 days)
        def rolling_trend_strength(series, window=20):
            result = []
            for i in range(len(series)):
                if i < window - 1:
                    result.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    if np.all(np.isfinite(y)):
                        x = np.arange(window)
                        slope, intercept, r_value, _, _ = stats.linregress(x, y)
                        result.append(slope)  # Direction and magnitude of trend
                    else:
                        result.append(np.nan)
            return result
        
        gdf['sentiment_trend'] = rolling_trend_strength(gdf['sentiment_mean'], 20)
        
        dfs.append(gdf)
    
    return pd.concat(dfs, ignore_index=True)


def fetch_returns(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical returns."""
    import yfinance as yf
    
    all_data = []
    batch_size = 100
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(name=batch[0])
            all_data.append(data)
        except Exception as e:
            logger.warning(f"Failed batch {i}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    prices = pd.concat(all_data, axis=1)
    
    # Calculate returns at multiple horizons
    returns = {}
    for col in prices.columns:
        returns[f'{col}_ret1d'] = prices[col].pct_change(1).shift(-1)
        returns[f'{col}_ret5d'] = prices[col].pct_change(5).shift(-5)
        returns[f'{col}_ret10d'] = prices[col].pct_change(10).shift(-10)
        returns[f'{col}_ret20d'] = prices[col].pct_change(20).shift(-20)  # Added longer horizon
    
    ret_df = pd.DataFrame(returns, index=prices.index)
    
    records = []
    for date in ret_df.index:
        for ticker in prices.columns:
            records.append({
                'date': date,
                'ticker': ticker,
                'ret1d': ret_df.loc[date, f'{ticker}_ret1d'],
                'ret5d': ret_df.loc[date, f'{ticker}_ret5d'],
                'ret10d': ret_df.loc[date, f'{ticker}_ret10d'],
                'ret20d': ret_df.loc[date, f'{ticker}_ret20d'],
            })
    
    result = pd.DataFrame(records)
    result['date'] = pd.to_datetime(result['date']).dt.tz_localize(None)
    return result


def analyze_event_driven(df: pd.DataFrame):
    """
    APPROACH 1: Event-driven - focus on high volume days.
    
    Hypothesis: Days with unusual news coverage contain more signal.
    """
    print("\n" + "="*60)
    print("APPROACH 1: EVENT-DRIVEN (High Volume Days)")
    print("="*60)
    
    for vol_threshold in [1.0, 1.5, 2.0, 2.5]:
        # Filter to high volume days
        high_vol = df[df['volume_zscore20'] > vol_threshold].copy()
        n_events = len(high_vol)
        pct_data = 100 * n_events / len(df)
        
        if n_events < 500:
            print(f"\nVolume Z > {vol_threshold}: Only {n_events} events, skipping")
            continue
        
        print(f"\nVolume Z > {vol_threshold}: {n_events:,} events ({pct_data:.1f}% of data)")
        
        # IC on filtered data
        for ret_col in ['ret5d', 'ret10d', 'ret20d']:
            valid = high_vol[['sentiment_mean', ret_col]].dropna()
            if len(valid) < 100:
                continue
            ic, pval = stats.spearmanr(valid['sentiment_mean'], valid[ret_col])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  IC(sentiment -> {ret_col}): {ic:+.4f} (p={pval:.4f}) {sig}")
        
        # Quintile spread on high volume days
        valid = high_vol[['sentiment_mean', 'ret10d']].dropna()
        if len(valid) >= 500:
            valid['quintile'] = pd.qcut(valid['sentiment_mean'], 5, labels=[1,2,3,4,5], duplicates='drop')
            q_stats = valid.groupby('quintile')['ret10d'].mean() * 100
            if 1 in q_stats.index and 5 in q_stats.index:
                spread = q_stats[5] - q_stats[1]
                print(f"  L/S Spread (Q5-Q1, ret10d): {spread:+.3f}%")


def analyze_longer_lookback(df: pd.DataFrame):
    """
    APPROACH 2: Longer lookback aggregation.
    
    Hypothesis: Weekly/monthly sentiment drift is more predictive than daily noise.
    """
    print("\n" + "="*60)
    print("APPROACH 2: LONGER LOOKBACK (Aggregated Sentiment)")
    print("="*60)
    
    # Test different lookback windows
    lookbacks = [
        ('sentiment_ma5', '5-day MA'),
        ('sentiment_ma10', '10-day MA'),
        ('sentiment_ma20', '20-day MA'),
        ('sentiment_ma60', '60-day MA'),
        ('sentiment_sum10', '10-day Sum'),
        ('sentiment_sum20', '20-day Sum'),
        ('sentiment_trend', '20-day Trend'),
    ]
    
    print("\nIC Analysis by Lookback Window:")
    print("-" * 50)
    
    results = []
    for col, name in lookbacks:
        if col not in df.columns:
            continue
            
        for ret_col in ['ret5d', 'ret10d', 'ret20d']:
            valid = df[[col, ret_col]].dropna()
            if len(valid) < 1000:
                continue
            ic, pval = stats.spearmanr(valid[col], valid[ret_col])
            results.append({
                'feature': name,
                'target': ret_col,
                'ic': ic,
                'pval': pval,
                'n': len(valid)
            })
    
    results_df = pd.DataFrame(results)
    
    # Pivot for cleaner display
    for ret_col in ['ret5d', 'ret10d', 'ret20d']:
        print(f"\nTarget: {ret_col}")
        subset = results_df[results_df['target'] == ret_col]
        for _, row in subset.iterrows():
            sig = "***" if row['pval'] < 0.001 else "**" if row['pval'] < 0.01 else "*" if row['pval'] < 0.05 else ""
            print(f"  {row['feature']:15}: IC={row['ic']:+.4f} (p={row['pval']:.4f}) {sig}")
    
    # Best performing feature quintile analysis
    best_feature = 'sentiment_ma20'
    if best_feature in df.columns:
        print(f"\nQuintile Analysis ({best_feature} -> ret20d):")
        valid = df[[best_feature, 'ret20d']].dropna()
        valid['quintile'] = pd.qcut(valid[best_feature], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        for q in [1, 2, 3, 4, 5]:
            q_data = valid[valid['quintile'] == q]['ret20d']
            if len(q_data) > 0:
                print(f"  Q{q}: {q_data.mean()*100:+.3f}% (n={len(q_data):,})")
        
        q_stats = valid.groupby('quintile')['ret20d'].mean() * 100
        if 1 in q_stats.index and 5 in q_stats.index:
            spread = q_stats[5] - q_stats[1]
            print(f"\n  Long-Short Spread: {spread:+.3f}%")


def analyze_extreme_sentiment(df: pd.DataFrame):
    """
    APPROACH 3: Extreme sentiment days only.
    
    Hypothesis: Very positive or negative days have signal, neutral days are noise.
    """
    print("\n" + "="*60)
    print("APPROACH 3: EXTREME SENTIMENT (Tail Events)")
    print("="*60)
    
    # Use z-score to define extremes
    for zscore_threshold in [1.0, 1.5, 2.0]:
        if 'sentiment_zscore20' not in df.columns:
            continue
            
        extreme = df[df['sentiment_zscore20'].abs() > zscore_threshold].copy()
        n_extreme = len(extreme)
        pct_data = 100 * n_extreme / len(df)
        
        if n_extreme < 500:
            print(f"\n|Z| > {zscore_threshold}: Only {n_extreme} events, skipping")
            continue
        
        print(f"\n|Sentiment Z| > {zscore_threshold}: {n_extreme:,} events ({pct_data:.1f}%)")
        
        # Split into positive and negative extremes
        positive_extreme = extreme[extreme['sentiment_zscore20'] > zscore_threshold]
        negative_extreme = extreme[extreme['sentiment_zscore20'] < -zscore_threshold]
        
        for ret_col in ['ret5d', 'ret10d', 'ret20d']:
            pos_ret = positive_extreme[ret_col].dropna()
            neg_ret = negative_extreme[ret_col].dropna()
            
            if len(pos_ret) < 50 or len(neg_ret) < 50:
                continue
            
            pos_mean = pos_ret.mean() * 100
            neg_mean = neg_ret.mean() * 100
            spread = pos_mean - neg_mean
            
            # T-test for significance
            t_stat, t_pval = stats.ttest_ind(pos_ret, neg_ret)
            sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else ""
            
            print(f"  {ret_col}: Pos={pos_mean:+.3f}% (n={len(pos_ret)}), Neg={neg_mean:+.3f}% (n={len(neg_ret)}), Spread={spread:+.3f}% {sig}")


def analyze_momentum_signal(df: pd.DataFrame):
    """
    APPROACH 4: Sentiment momentum/reversal.
    
    Hypothesis: Stocks with improving sentiment outperform, or
                extreme positive momentum mean reverts (contrarian).
    """
    print("\n" + "="*60)
    print("APPROACH 4: SENTIMENT MOMENTUM")
    print("="*60)
    
    momentum_cols = ['sentiment_mom5', 'sentiment_mom10', 'sentiment_mom20', 'sentiment_accel']
    
    for mom_col in momentum_cols:
        if mom_col not in df.columns:
            continue
        
        print(f"\n{mom_col}:")
        
        for ret_col in ['ret5d', 'ret10d', 'ret20d']:
            valid = df[[mom_col, ret_col]].dropna()
            if len(valid) < 1000:
                continue
            ic, pval = stats.spearmanr(valid[mom_col], valid[ret_col])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  IC -> {ret_col}: {ic:+.4f} (p={pval:.4f}) {sig}")
        
        # Quintile analysis
        valid = df[[mom_col, 'ret10d']].dropna()
        if len(valid) >= 1000:
            valid['quintile'] = pd.qcut(valid[mom_col], 5, labels=[1,2,3,4,5], duplicates='drop')
            q_stats = valid.groupby('quintile')['ret10d'].mean() * 100
            if 1 in q_stats.index and 5 in q_stats.index:
                spread = q_stats[5] - q_stats[1]
                print(f"  L/S Spread (ret10d): {spread:+.3f}%")


def analyze_by_market_regime(df: pd.DataFrame):
    """
    Analyze if signal works differently in different regimes.
    """
    print("\n" + "="*60)
    print("APPROACH 5: MARKET REGIME ANALYSIS")
    print("="*60)
    
    # Split by year
    df['year'] = df['date'].dt.year
    
    print("\nIC by Year (sentiment_ma20 -> ret10d):")
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        if 'sentiment_ma20' not in year_data.columns:
            continue
        valid = year_data[['sentiment_ma20', 'ret10d']].dropna()
        if len(valid) < 500:
            continue
        ic, pval = stats.spearmanr(valid['sentiment_ma20'], valid['ret10d'])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {year}: IC={ic:+.4f} (p={pval:.4f}, n={len(valid):,}) {sig}")


def main():
    # Load data
    logger.info("Loading news sentiment...")
    articles = load_news_sentiment()
    logger.info(f"Loaded {len(articles):,} articles")
    
    # Aggregate to daily
    daily = aggregate_daily_sentiment(articles)
    logger.info(f"Aggregated to {len(daily):,} ticker-days")
    
    # Add features
    daily = add_features_with_lookbacks(daily)
    
    # Get unique tickers and date range
    tickers = daily['ticker'].unique().tolist()
    date_range = (daily['date'].min().strftime('%Y-%m-%d'),
                  (daily['date'].max() + timedelta(days=30)).strftime('%Y-%m-%d'))
    
    # Fetch returns
    logger.info("Fetching returns...")
    returns = fetch_returns(tickers, date_range[0], date_range[1])
    logger.info(f"Returns: {len(returns):,} rows")
    
    # Merge
    df = daily.merge(returns, on=['ticker', 'date'], how='inner')
    
    # Drop rows with too many missing features
    feature_cols = [c for c in df.columns if 'sentiment' in c or 'volume' in c or 'news' in c]
    df = df.dropna(subset=['ret10d'] + feature_cols[:5])  # At least basic features
    
    logger.info(f"Final dataset: {len(df):,} rows")
    
    # Run all analyses
    analyze_event_driven(df)
    analyze_longer_lookback(df)
    analyze_extreme_sentiment(df)
    analyze_momentum_signal(df)
    analyze_by_market_regime(df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Key findings will show if any approach yields significant IC or spreads.
Look for:
- IC > 0.02 with p < 0.05 (weak but usable signal)
- IC > 0.04 with p < 0.01 (decent signal)
- Long-short spread > 0.5% (economically meaningful)
    """)


if __name__ == "__main__":
    main()
