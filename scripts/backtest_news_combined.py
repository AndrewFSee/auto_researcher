"""
Combined News Sentiment Model.

Two signals discovered in v2 backtest:

1. EVENT-DRIVEN (momentum): On high-volume days (volume_zscore > 2),
   positive sentiment predicts positive returns. Fresh news = real information.
   
2. CONTRARIAN (mean-reversion): Stocks with persistently positive sentiment
   (high sentiment_ma20) underperform. Accumulated hype = overvaluation.

Strategy:
- Combine both signals into a single alpha score
- Event signal: sentiment * volume_zscore (high weight on high-volume days)
- Contrarian signal: -sentiment_ma20 (negative = short the hyped stocks)
- Weight by signal quality

Target: 10-day returns (optimal for both signals)
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
                sentiment_label
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


def add_combined_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for both event-driven and contrarian signals.
    """
    dfs = []
    
    for ticker, gdf in daily.groupby('ticker'):
        gdf = gdf.sort_values('date').copy()
        
        # === CONTRARIAN FEATURES (longer lookback) ===
        gdf['sentiment_ma5'] = gdf['sentiment_mean'].rolling(5, min_periods=2).mean()
        gdf['sentiment_ma10'] = gdf['sentiment_mean'].rolling(10, min_periods=3).mean()
        gdf['sentiment_ma20'] = gdf['sentiment_mean'].rolling(20, min_periods=5).mean()
        gdf['sentiment_sum20'] = gdf['sentiment_mean'].rolling(20, min_periods=5).sum()
        
        # Sentiment persistence (how long has it been positive/negative)
        gdf['sentiment_sign'] = np.sign(gdf['sentiment_mean'])
        
        # === EVENT-DRIVEN FEATURES ===
        # Volume z-score (unusual coverage)
        vol_mean = gdf['news_count'].rolling(20, min_periods=5).mean()
        vol_std = gdf['news_count'].rolling(20, min_periods=5).std().clip(lower=0.1)
        gdf['volume_zscore'] = (gdf['news_count'] - vol_mean) / vol_std
        
        # Sentiment z-score (unusual sentiment)
        sent_mean = gdf['sentiment_mean'].rolling(20, min_periods=5).mean()
        sent_std = gdf['sentiment_mean'].rolling(20, min_periods=5).std().clip(lower=0.01)
        gdf['sentiment_zscore'] = (gdf['sentiment_mean'] - sent_mean) / sent_std
        
        # === COMBINED SIGNALS ===
        # Event signal: today's sentiment weighted by volume surprise
        # High volume + positive sentiment = strong buy signal
        gdf['event_signal'] = gdf['sentiment_mean'] * gdf['volume_zscore'].clip(lower=0)
        
        # Contrarian signal: negative of accumulated sentiment
        # High accumulated positive sentiment = sell signal
        gdf['contrarian_signal'] = -gdf['sentiment_ma20']
        
        # Combined alpha (will be tuned)
        # For now, simple average of normalized signals
        gdf['alpha_raw'] = gdf['event_signal'] + gdf['contrarian_signal']
        
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
    
    returns = {}
    for col in prices.columns:
        returns[f'{col}_ret5d'] = prices[col].pct_change(5).shift(-5)
        returns[f'{col}_ret10d'] = prices[col].pct_change(10).shift(-10)
        returns[f'{col}_ret20d'] = prices[col].pct_change(20).shift(-20)
    
    ret_df = pd.DataFrame(returns, index=prices.index)
    
    records = []
    for date in ret_df.index:
        for ticker in prices.columns:
            records.append({
                'date': date,
                'ticker': ticker,
                'ret5d': ret_df.loc[date, f'{ticker}_ret5d'],
                'ret10d': ret_df.loc[date, f'{ticker}_ret10d'],
                'ret20d': ret_df.loc[date, f'{ticker}_ret20d'],
            })
    
    result = pd.DataFrame(records)
    result['date'] = pd.to_datetime(result['date']).dt.tz_localize(None)
    return result


def cross_sectional_normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Normalize features cross-sectionally each day."""
    result = df.copy()
    
    for col in cols:
        if col in result.columns:
            # Group by date and z-score normalize
            result[f'{col}_xs'] = result.groupby('date')[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
    
    return result


def analyze_individual_signals(df: pd.DataFrame):
    """Analyze each signal component separately."""
    print("\n" + "="*60)
    print("INDIVIDUAL SIGNAL ANALYSIS")
    print("="*60)
    
    signals = {
        'event_signal': 'Event (sentiment * volume)',
        'contrarian_signal': 'Contrarian (-sentiment_ma20)',
        'alpha_raw': 'Combined (event + contrarian)',
    }
    
    for col, name in signals.items():
        if col not in df.columns:
            continue
        
        print(f"\n{name}:")
        
        for ret_col in ['ret5d', 'ret10d', 'ret20d']:
            valid = df[[col, ret_col]].dropna()
            if len(valid) < 1000:
                continue
            ic, pval = stats.spearmanr(valid[col], valid[ret_col])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  IC -> {ret_col}: {ic:+.4f} (p={pval:.4f}) {sig}")
        
        # Quintile analysis for ret10d
        valid = df[[col, 'ret10d']].dropna()
        if len(valid) >= 1000:
            try:
                valid['quintile'] = pd.qcut(valid[col].rank(method='first'), 5, labels=[1,2,3,4,5])
                q_stats = valid.groupby('quintile')['ret10d'].agg(['mean', 'count'])
                
                print(f"\n  Quintiles (ret10d):")
                for q in [1, 2, 3, 4, 5]:
                    if q in q_stats.index:
                        print(f"    Q{q}: {q_stats.loc[q, 'mean']*100:+.3f}% (n={q_stats.loc[q, 'count']:.0f})")
                
                if 1 in q_stats.index and 5 in q_stats.index:
                    spread = (q_stats.loc[5, 'mean'] - q_stats.loc[1, 'mean']) * 100
                    print(f"  L/S Spread: {spread:+.3f}%")
            except Exception:
                print(f"\n  (Could not compute quintiles)")


def optimize_signal_weights(df: pd.DataFrame, target: str = 'ret10d'):
    """Find optimal weights for combining signals."""
    print("\n" + "="*60)
    print("SIGNAL WEIGHT OPTIMIZATION")
    print("="*60)
    
    valid = df[['event_signal', 'contrarian_signal', target]].dropna()
    
    # Grid search over weights
    best_ic = -np.inf
    best_weights = (0.5, 0.5)
    
    results = []
    for event_w in np.arange(0, 1.05, 0.1):
        contra_w = 1 - event_w
        
        combined = event_w * valid['event_signal'] + contra_w * valid['contrarian_signal']
        ic, pval = stats.spearmanr(combined, valid[target])
        
        results.append({
            'event_weight': event_w,
            'contrarian_weight': contra_w,
            'ic': ic,
            'pval': pval
        })
        
        if ic > best_ic:
            best_ic = ic
            best_weights = (event_w, contra_w)
    
    print(f"\nWeight Optimization Results ({target}):")
    print("-" * 50)
    
    for r in results:
        sig = "***" if r['pval'] < 0.001 else "**" if r['pval'] < 0.01 else "*" if r['pval'] < 0.05 else ""
        marker = " <-- BEST" if (r['event_weight'], r['contrarian_weight']) == best_weights else ""
        print(f"  Event={r['event_weight']:.1f}, Contra={r['contrarian_weight']:.1f}: IC={r['ic']:+.4f} {sig}{marker}")
    
    print(f"\nOptimal weights: Event={best_weights[0]:.1f}, Contrarian={best_weights[1]:.1f}")
    print(f"Best IC: {best_ic:+.4f}")
    
    return best_weights


def train_combined_model(df: pd.DataFrame, event_weight: float, contrarian_weight: float):
    """Train LightGBM model with optimized combined signal."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Run: pip install lightgbm")
        return
    
    print("\n" + "="*60)
    print("LIGHTGBM MODEL TRAINING")
    print("="*60)
    
    # Create optimized combined signal
    df['alpha_optimized'] = event_weight * df['event_signal'] + contrarian_weight * df['contrarian_signal']
    
    # Features for model
    feature_cols = [
        'sentiment_mean', 'sentiment_std', 'news_count',
        'positive_ratio', 'negative_ratio',
        'sentiment_ma5', 'sentiment_ma10', 'sentiment_ma20',
        'volume_zscore', 'sentiment_zscore',
        'event_signal', 'contrarian_signal', 'alpha_optimized'
    ]
    
    # Filter to rows with all features
    available_features = [c for c in feature_cols if c in df.columns]
    model_df = df.dropna(subset=available_features + ['ret10d'])
    
    # Time-series split
    model_df = model_df.sort_values('date')
    split_date = model_df['date'].quantile(0.8)
    
    train = model_df[model_df['date'] < split_date]
    test = model_df[model_df['date'] >= split_date]
    
    print(f"\nTrain: {len(train):,} samples ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test):,} samples ({test['date'].min().date()} to {test['date'].max().date()})")
    
    X_train = train[available_features]
    y_train = train['ret10d']
    X_test = test[available_features]
    y_test = test['ret10d']
    
    # Train model
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    test['pred'] = model.predict(X_test)
    
    # Evaluate
    ic, pval = stats.spearmanr(test['pred'], test['ret10d'])
    
    print(f"\n" + "="*60)
    print("OUT-OF-SAMPLE PERFORMANCE")
    print("="*60)
    print(f"\nIC (Spearman): {ic:+.4f} (p={pval:.6f})")
    
    # Quintile analysis
    test['quintile'] = pd.qcut(test['pred'], 5, labels=[1,2,3,4,5], duplicates='drop')
    q_stats = test.groupby('quintile')['ret10d'].agg(['mean', 'std', 'count'])
    
    print(f"\nQuintile Returns (ret10d):")
    for q in [1, 2, 3, 4, 5]:
        if q in q_stats.index:
            mean_ret = q_stats.loc[q, 'mean'] * 100
            std_ret = q_stats.loc[q, 'std'] * 100
            count = q_stats.loc[q, 'count']
            print(f"  Q{q}: {mean_ret:+.3f}% (std={std_ret:.2f}%, n={count:.0f})")
    
    if 1 in q_stats.index and 5 in q_stats.index:
        spread = (q_stats.loc[5, 'mean'] - q_stats.loc[1, 'mean']) * 100
        print(f"\nLong-Short Spread (Q5-Q1): {spread:+.3f}%")
    
    # Hit rate
    test['correct'] = ((test['pred'] > test['pred'].median()) == 
                       (test['ret10d'] > test['ret10d'].median()))
    hit_rate = test['correct'].mean() * 100
    print(f"Hit Rate: {hit_rate:.2f}%")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in importance.head(8).iterrows():
        print(f"  {row['feature']:25} {row['importance']:.0f}")
    
    # Compare to simple signal
    print("\n" + "="*60)
    print("COMPARISON: MODEL vs SIMPLE SIGNAL")
    print("="*60)
    
    # Simple optimized signal IC
    simple_ic, simple_pval = stats.spearmanr(test['alpha_optimized'], test['ret10d'])
    print(f"\nSimple combined signal IC: {simple_ic:+.4f} (p={simple_pval:.4f})")
    print(f"LightGBM model IC:         {ic:+.4f} (p={pval:.4f})")
    print(f"Improvement:               {(ic - simple_ic):+.4f}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "news_combined_results.parquet"
    test.to_parquet(output_path)
    print(f"\nResults saved to {output_path}")
    
    return model, test


def analyze_signal_by_regime(df: pd.DataFrame, event_weight: float, contrarian_weight: float):
    """Analyze how signals perform in different market conditions."""
    print("\n" + "="*60)
    print("SIGNAL PERFORMANCE BY REGIME")
    print("="*60)
    
    df['alpha_optimized'] = event_weight * df['event_signal'] + contrarian_weight * df['contrarian_signal']
    
    # By year
    df['year'] = df['date'].dt.year
    
    print("\nIC by Year (alpha_optimized -> ret10d):")
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        valid = year_data[['alpha_optimized', 'ret10d']].dropna()
        if len(valid) < 500:
            continue
        ic, pval = stats.spearmanr(valid['alpha_optimized'], valid['ret10d'])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {year}: IC={ic:+.4f} (p={pval:.4f}, n={len(valid):,}) {sig}")
    
    # By volume regime (high vs low volume days)
    print("\nIC by Volume Regime:")
    for vol_label, condition in [('Low Volume (Z<1)', df['volume_zscore'] < 1),
                                   ('High Volume (Z>2)', df['volume_zscore'] > 2)]:
        regime_data = df[condition]
        valid = regime_data[['alpha_optimized', 'ret10d']].dropna()
        if len(valid) < 500:
            continue
        ic, pval = stats.spearmanr(valid['alpha_optimized'], valid['ret10d'])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {vol_label}: IC={ic:+.4f} (p={pval:.4f}, n={len(valid):,}) {sig}")


def main():
    logger.info("Loading news sentiment...")
    articles = load_news_sentiment()
    logger.info(f"Loaded {len(articles):,} articles")
    
    # Aggregate to daily
    daily = aggregate_daily_sentiment(articles)
    logger.info(f"Aggregated to {len(daily):,} ticker-days")
    
    # Add combined features
    daily = add_combined_features(daily)
    
    # Fetch returns
    tickers = daily['ticker'].unique().tolist()
    date_range = (daily['date'].min().strftime('%Y-%m-%d'),
                  (daily['date'].max() + timedelta(days=30)).strftime('%Y-%m-%d'))
    
    logger.info("Fetching returns...")
    returns = fetch_returns(tickers, date_range[0], date_range[1])
    logger.info(f"Returns: {len(returns):,} rows")
    
    # Merge
    df = daily.merge(returns, on=['ticker', 'date'], how='inner')
    df = df.dropna(subset=['sentiment_ma20', 'volume_zscore', 'ret10d'])
    
    logger.info(f"Final dataset: {len(df):,} rows")
    
    # Cross-sectional normalize key signals
    df = cross_sectional_normalize(df, ['event_signal', 'contrarian_signal'])
    
    # Analyze individual signals
    analyze_individual_signals(df)
    
    # Optimize weights
    best_weights = optimize_signal_weights(df, 'ret10d')
    
    # Train combined model
    model, test_results = train_combined_model(df, best_weights[0], best_weights[1])
    
    # Analyze by regime
    analyze_signal_by_regime(df, best_weights[0], best_weights[1])
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
Combined News Sentiment Model:
- Event Signal Weight:      {best_weights[0]:.1f}
- Contrarian Signal Weight: {best_weights[1]:.1f}

Key Insights:
- Event signal works on HIGH VOLUME days (fresh news = real info)
- Contrarian signal works on accumulated sentiment (mean reversion)
- Combined signal captures both momentum and mean-reversion effects
    """)


if __name__ == "__main__":
    main()
