"""
News Sentiment Backtesting Model.

Tests whether news sentiment predicts forward stock returns.

Approach:
- Pooled cross-sectional model (like earnings model)
- Features: sentiment level, momentum, surprise, volume
- Target: 1d, 5d, 10d forward returns
- Model: LightGBM (faster than XGBoost for larger datasets)

Key insight: The signal is likely in CHANGES in sentiment,
not absolute levels. Each stock has different baseline coverage.
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

# Path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_researcher.data.news_scraper import NewsDatabase


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
    logger.info(f"Loaded {len(df):,} articles")
    return df


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level sentiment to daily ticker-level features.
    
    Features:
    - sentiment_mean: Average daily sentiment
    - sentiment_std: Sentiment dispersion (conflicting views)
    - news_count: Volume of coverage (attention)
    - positive_ratio: Fraction of positive articles
    - negative_ratio: Fraction of negative articles
    """
    daily = df.groupby(['ticker', 'date']).agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        news_count=('sentiment_score', 'count'),
        positive_ratio=('sentiment_label', lambda x: (x == 'positive').mean()),
        negative_ratio=('sentiment_label', lambda x: (x == 'negative').mean()),
    ).reset_index()
    
    # Fill NaN std with 0 (single article days)
    daily['sentiment_std'] = daily['sentiment_std'].fillna(0)
    
    logger.info(f"Aggregated to {len(daily):,} ticker-days")
    return daily


def add_sentiment_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling and change-based sentiment features.
    
    Key features:
    - sentiment_ma5/ma20: Rolling average sentiment
    - sentiment_momentum: 5d change in sentiment
    - sentiment_zscore: Current vs 20d rolling mean/std (surprise)
    - volume_zscore: Current volume vs 20d rolling
    """
    dfs = []
    
    for ticker, gdf in daily.groupby('ticker'):
        gdf = gdf.sort_values('date').copy()
        
        # Rolling sentiment averages
        gdf['sentiment_ma5'] = gdf['sentiment_mean'].rolling(5, min_periods=1).mean()
        gdf['sentiment_ma20'] = gdf['sentiment_mean'].rolling(20, min_periods=5).mean()
        
        # Sentiment momentum (5-day change)
        gdf['sentiment_momentum'] = gdf['sentiment_mean'] - gdf['sentiment_mean'].shift(5)
        
        # Sentiment surprise (z-score vs rolling window)
        rolling_mean = gdf['sentiment_mean'].rolling(20, min_periods=10).mean()
        rolling_std = gdf['sentiment_mean'].rolling(20, min_periods=10).std()
        gdf['sentiment_zscore'] = (gdf['sentiment_mean'] - rolling_mean) / rolling_std.clip(lower=0.01)
        
        # Volume surprise (unusual coverage)
        vol_mean = gdf['news_count'].rolling(20, min_periods=5).mean()
        vol_std = gdf['news_count'].rolling(20, min_periods=5).std()
        gdf['volume_zscore'] = (gdf['news_count'] - vol_mean) / vol_std.clip(lower=0.1)
        
        # Cumulative sentiment over past 5 days
        gdf['sentiment_sum5'] = gdf['sentiment_mean'].rolling(5, min_periods=1).sum()
        
        dfs.append(gdf)
    
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Added features, {len(result):,} rows")
    return result


def fetch_returns(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical returns for all tickers."""
    import yfinance as yf
    
    logger.info(f"Fetching returns for {len(tickers)} tickers...")
    
    # Download in batches
    all_data = []
    batch_size = 100
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )['Close']
            
            if isinstance(data, pd.Series):
                data = data.to_frame(name=batch[0])
            
            all_data.append(data)
        except Exception as e:
            logger.warning(f"Failed batch {i}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    prices = pd.concat(all_data, axis=1)
    
    # Calculate returns
    returns = {}
    for col in prices.columns:
        returns[f'{col}_ret1d'] = prices[col].pct_change(1).shift(-1)  # 1-day forward
        returns[f'{col}_ret5d'] = prices[col].pct_change(5).shift(-5)  # 5-day forward
        returns[f'{col}_ret10d'] = prices[col].pct_change(10).shift(-10)  # 10-day forward
    
    ret_df = pd.DataFrame(returns, index=prices.index)
    
    # Reshape to long format
    records = []
    for date in ret_df.index:
        for ticker in prices.columns:
            records.append({
                'date': date,
                'ticker': ticker,
                'ret1d': ret_df.loc[date, f'{ticker}_ret1d'],
                'ret5d': ret_df.loc[date, f'{ticker}_ret5d'],
                'ret10d': ret_df.loc[date, f'{ticker}_ret10d'],
            })
    
    result = pd.DataFrame(records)
    result['date'] = pd.to_datetime(result['date']).dt.tz_localize(None)
    logger.info(f"Fetched returns: {len(result):,} rows")
    return result


def build_dataset(daily: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Merge sentiment features with forward returns."""
    # Merge on ticker and date
    merged = daily.merge(returns, on=['ticker', 'date'], how='inner')
    
    # Drop rows with missing values
    feature_cols = [
        'sentiment_mean', 'sentiment_std', 'news_count',
        'positive_ratio', 'negative_ratio',
        'sentiment_ma5', 'sentiment_ma20', 'sentiment_momentum',
        'sentiment_zscore', 'volume_zscore', 'sentiment_sum5'
    ]
    
    merged = merged.dropna(subset=feature_cols + ['ret5d'])
    logger.info(f"Final dataset: {len(merged):,} rows")
    return merged


def analyze_signal_quality(df: pd.DataFrame):
    """Analyze raw signal quality before modeling."""
    print("\n" + "="*60)
    print("SIGNAL QUALITY ANALYSIS")
    print("="*60)
    
    # IC (Information Coefficient) - correlation with returns
    for ret_col in ['ret1d', 'ret5d', 'ret10d']:
        for sig_col in ['sentiment_mean', 'sentiment_momentum', 'sentiment_zscore']:
            valid = df[[sig_col, ret_col]].dropna()
            if len(valid) < 100:
                continue
            ic, pval = stats.spearmanr(valid[sig_col], valid[ret_col])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"IC({sig_col[:15]:15} -> {ret_col}): {ic:+.4f} (p={pval:.4f}) {sig}")
    
    # Quintile analysis
    print("\nQuintile Analysis (sentiment_zscore -> ret5d):")
    df_valid = df[['sentiment_zscore', 'ret5d']].dropna()
    df_valid['quintile'] = pd.qcut(df_valid['sentiment_zscore'], 5, labels=[1,2,3,4,5])
    
    quintile_stats = df_valid.groupby('quintile')['ret5d'].agg(['mean', 'std', 'count'])
    for q in [1, 2, 3, 4, 5]:
        if q in quintile_stats.index:
            mean_ret = quintile_stats.loc[q, 'mean'] * 100
            std_ret = quintile_stats.loc[q, 'std'] * 100
            count = quintile_stats.loc[q, 'count']
            print(f"  Q{q}: {mean_ret:+.3f}% (std: {std_ret:.2f}%, n={count:,.0f})")
    
    spread = (quintile_stats.loc[5, 'mean'] - quintile_stats.loc[1, 'mean']) * 100
    print(f"\nLong-Short Spread (Q5-Q1): {spread:+.3f}%")


def train_model(df: pd.DataFrame, target: str = 'ret5d'):
    """Train LightGBM model with time-series cross-validation."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not installed, using XGBoost")
        import xgboost as xgb
        lgb = None
    
    feature_cols = [
        'sentiment_mean', 'sentiment_std', 'news_count',
        'positive_ratio', 'negative_ratio',
        'sentiment_ma5', 'sentiment_ma20', 'sentiment_momentum',
        'sentiment_zscore', 'volume_zscore', 'sentiment_sum5'
    ]
    
    # Sort by date for time-series split
    df = df.sort_values('date')
    
    # Use 80% for training, 20% for testing (time-based split)
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    print(f"\nTrain: {len(train):,} samples ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test):,} samples ({test['date'].min().date()} to {test['date'].max().date()})")
    
    X_train = train[feature_cols]
    y_train = train[target]
    X_test = test[feature_cols]
    y_test = test[target]
    
    if lgb:
        # LightGBM model
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    else:
        # XGBoost fallback
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    
    model.fit(X_train, y_train)
    
    # Predictions
    test = test.copy()
    test['pred'] = model.predict(X_test)
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL PERFORMANCE (Out-of-Sample)")
    print("="*60)
    
    # IC
    ic, pval = stats.spearmanr(test['pred'], test[target])
    print(f"IC (Spearman): {ic:.4f} (p={pval:.6f})")
    
    # Quintile analysis
    test['pred_quintile'] = pd.qcut(test['pred'], 5, labels=[1,2,3,4,5], duplicates='drop')
    
    print(f"\nQuintile Returns ({target}):")
    for q in [1, 2, 3, 4, 5]:
        mask = test['pred_quintile'] == q
        if mask.sum() > 0:
            mean_ret = test.loc[mask, target].mean() * 100
            count = mask.sum()
            print(f"  Q{q}: {mean_ret:+.3f}% (n={count:,})")
    
    q5_ret = test.loc[test['pred_quintile'] == 5, target].mean()
    q1_ret = test.loc[test['pred_quintile'] == 1, target].mean()
    spread = (q5_ret - q1_ret) * 100
    print(f"\nLong-Short Spread: {spread:+.3f}%")
    
    # Hit rate
    test['correct'] = (test['pred'] > 0) == (test[target] > 0)
    hit_rate = test['correct'].mean()
    print(f"Hit Rate: {hit_rate:.2%}")
    
    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']:25} {row['importance']:.3f}")
    
    return model, test


def main():
    print("="*60)
    print("NEWS SENTIMENT BACKTESTING")
    print("="*60)
    
    # Load news sentiment
    df = load_news_sentiment()
    
    # Aggregate to daily
    daily = aggregate_daily_sentiment(df)
    
    # Add features
    daily = add_sentiment_features(daily)
    
    # Get unique tickers
    tickers = daily['ticker'].unique().tolist()
    
    # Date range
    start_date = daily['date'].min().strftime('%Y-%m-%d')
    end_date = daily['date'].max().strftime('%Y-%m-%d')
    
    # Fetch returns
    returns = fetch_returns(tickers, start_date, end_date)
    
    # Build dataset
    dataset = build_dataset(daily, returns)
    
    # Analyze raw signal quality
    analyze_signal_quality(dataset)
    
    # Train model
    model, test_results = train_model(dataset, target='ret5d')
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "news_backtest_results.parquet"
    test_results.to_parquet(output_path)
    print(f"\nResults saved to {output_path}")
    
    return model, test_results


if __name__ == "__main__":
    model, results = main()
