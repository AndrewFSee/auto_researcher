"""
Backtest: Post-Earnings Announcement Drift (PEAD)

Tests whether EPS surprises predict future returns.
PEAD is a classic market anomaly - stocks that beat/miss earnings 
tend to drift in the same direction for weeks/months afterward.

Data source: DefeatBeta stock_historical_eps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_eps_data():
    """Load EPS surprise history from DefeatBeta."""
    from src.auto_researcher.data.defeatbeta import DefeatBetaDataLoader
    
    df = DefeatBetaDataLoader.load('stock_historical_eps')
    if df is None:
        raise ValueError("Failed to load EPS history")
    
    logger.info(f"Loaded {len(df):,} EPS records")
    logger.info(f"Unique tickers: {df['symbol'].nunique():,}")
    
    return df


def calculate_surprise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various EPS surprise metrics.
    """
    df = df.copy()
    
    # Convert to numeric
    df['eps_actual'] = pd.to_numeric(df['eps_actual'], errors='coerce')
    df['eps_estimate'] = pd.to_numeric(df['eps_estimate'], errors='coerce')
    df['surprise_percent'] = pd.to_numeric(df['surprise_percent'], errors='coerce')
    
    # Parse quarter date
    df['quarter_date'] = pd.to_datetime(df['quarter_date'], errors='coerce')
    
    # Drop missing
    df = df.dropna(subset=['eps_actual', 'eps_estimate', 'quarter_date'])
    
    # Calculate standardized surprise (SUE - Standardized Unexpected Earnings)
    # SUE = (Actual - Estimate) / |Estimate|
    df['sue'] = np.where(
        df['eps_estimate'].abs() > 0.01,
        (df['eps_actual'] - df['eps_estimate']) / df['eps_estimate'].abs(),
        np.nan
    )
    
    # Raw surprise
    df['surprise_raw'] = df['eps_actual'] - df['eps_estimate']
    
    # Binary beat/miss
    df['beat'] = (df['eps_actual'] > df['eps_estimate']).astype(int)
    df['miss'] = (df['eps_actual'] < df['eps_estimate']).astype(int)
    
    # Surprise magnitude buckets
    df['surprise_bucket'] = pd.cut(
        df['sue'],
        bins=[-np.inf, -0.10, -0.02, 0.02, 0.10, np.inf],
        labels=['big_miss', 'small_miss', 'inline', 'small_beat', 'big_beat']
    )
    
    # Calculate streak (consecutive beats/misses)
    df = df.sort_values(['symbol', 'quarter_date'])
    
    def calc_streak(group):
        beats = group['beat'].values
        streak = np.zeros(len(beats))
        for i in range(1, len(beats)):
            if beats[i-1] == 1:
                streak[i] = streak[i-1] + 1
            elif beats[i-1] == 0 and group['miss'].values[i-1] == 1:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = 0
        return pd.Series(streak, index=group.index)
    
    df['beat_streak'] = df.groupby('symbol', group_keys=False).apply(calc_streak)
    
    return df


def get_post_earnings_returns(eps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get returns after earnings announcements.
    We fetch prices and calculate returns for various windows after each earnings date.
    """
    # Get unique tickers
    tickers = eps_df['symbol'].unique().tolist()
    logger.info(f"Fetching returns for {len(tickers)} tickers...")
    
    # Get all unique dates we need
    min_date = eps_df['quarter_date'].min() - timedelta(days=5)
    max_date = eps_df['quarter_date'].max() + timedelta(days=90)
    
    # Download prices in batches
    all_data = []
    batch_size = 50
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                batch, 
                start=min_date.strftime('%Y-%m-%d'), 
                end='2026-01-20', 
                progress=False
            )['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(name=batch[0])
            all_data.append(data)
        except Exception as e:
            logger.warning(f"Failed batch {i}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    prices = pd.concat(all_data, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    
    # For each earnings record, calculate post-announcement returns
    results = []
    
    for _, row in eps_df.iterrows():
        ticker = row['symbol']
        earn_date = row['quarter_date']
        
        if ticker not in prices.columns:
            continue
        
        ticker_prices = prices[ticker].dropna()
        
        # Find the first trading day on or after earnings
        future_prices = ticker_prices[ticker_prices.index >= earn_date]
        if len(future_prices) < 2:
            continue
        
        # Get price on announcement day (or next trading day)
        t0_date = future_prices.index[0]
        t0_price = future_prices.iloc[0]
        
        # Calculate returns for various horizons
        record = {
            'ticker': ticker,
            'quarter_date': earn_date,
            'eps_actual': row['eps_actual'],
            'eps_estimate': row['eps_estimate'],
            'sue': row['sue'],
            'surprise_raw': row['surprise_raw'],
            'beat': row['beat'],
            'surprise_bucket': row['surprise_bucket'],
            'beat_streak': row['beat_streak'],
        }
        
        # Post-earnings returns
        for days, col_name in [(1, 'ret1d'), (5, 'ret5d'), (10, 'ret10d'), 
                                (20, 'ret20d'), (40, 'ret40d'), (60, 'ret60d')]:
            future = ticker_prices[ticker_prices.index > t0_date]
            if len(future) >= days:
                record[col_name] = (future.iloc[days-1] / t0_price) - 1
            else:
                record[col_name] = np.nan
        
        results.append(record)
    
    return pd.DataFrame(results)


def analyze_signal(df: pd.DataFrame, signal_col: str, return_col: str = 'ret20d'):
    """Analyze a signal's predictive power."""
    valid = df.dropna(subset=[signal_col, return_col])
    
    if len(valid) < 100:
        return None
    
    # Information Coefficient (Spearman)
    ic, pval = stats.spearmanr(valid[signal_col], valid[return_col])
    
    # Quintile analysis
    try:
        valid = valid.copy()
        valid['quintile'] = pd.qcut(valid[signal_col].rank(method='first'), 5, labels=[1,2,3,4,5])
    except:
        return None
    
    q_stats = valid.groupby('quintile')[return_col].agg(['mean', 'std', 'count'])
    
    spread = q_stats.loc[5, 'mean'] - q_stats.loc[1, 'mean']
    
    return {
        'ic': ic,
        'pval': pval,
        'n': len(valid),
        'q1_ret': q_stats.loc[1, 'mean'],
        'q5_ret': q_stats.loc[5, 'mean'],
        'spread': spread,
        'q_stats': q_stats
    }


def analyze_by_bucket(df: pd.DataFrame, return_col: str = 'ret20d'):
    """Analyze returns by surprise bucket."""
    valid = df.dropna(subset=['surprise_bucket', return_col])
    
    bucket_order = ['big_miss', 'small_miss', 'inline', 'small_beat', 'big_beat']
    stats_df = valid.groupby('surprise_bucket')[return_col].agg(['mean', 'std', 'count'])
    stats_df = stats_df.reindex(bucket_order)
    
    return stats_df


def main():
    print("="*70)
    print("POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) BACKTEST")
    print("="*70)
    
    # Load data
    eps_df = load_eps_data()
    
    # Calculate surprise metrics
    eps_df = calculate_surprise_metrics(eps_df)
    print(f"\nEPS records with valid data: {len(eps_df):,}")
    print(f"Date range: {eps_df['quarter_date'].min().date()} to {eps_df['quarter_date'].max().date()}")
    
    # Summary stats
    print("\n" + "="*70)
    print("SURPRISE STATISTICS")
    print("="*70)
    
    print(f"\nBeat rate: {eps_df['beat'].mean()*100:.1f}%")
    print(f"Miss rate: {eps_df['miss'].mean()*100:.1f}%")
    print(f"Inline rate: {(1 - eps_df['beat'].mean() - eps_df['miss'].mean())*100:.1f}%")
    
    print(f"\nSUE (Standardized Unexpected Earnings):")
    sue_valid = eps_df['sue'].dropna()
    print(f"  Mean: {sue_valid.mean()*100:+.2f}%")
    print(f"  Median: {sue_valid.median()*100:+.2f}%")
    print(f"  Std: {sue_valid.std()*100:.2f}%")
    
    print(f"\nSurprise distribution:")
    bucket_counts = eps_df['surprise_bucket'].value_counts()
    for bucket in ['big_miss', 'small_miss', 'inline', 'small_beat', 'big_beat']:
        if bucket in bucket_counts:
            print(f"  {bucket}: {bucket_counts[bucket]:,} ({bucket_counts[bucket]/len(eps_df)*100:.1f}%)")
    
    # Get post-earnings returns
    print("\n" + "="*70)
    print("FETCHING POST-EARNINGS RETURNS")
    print("="*70)
    
    pead_df = get_post_earnings_returns(eps_df)
    print(f"\nPEAD dataset: {len(pead_df):,} earnings events with returns")
    
    if len(pead_df) == 0:
        print("No data to analyze")
        return
    
    # Analyze SUE signal
    print("\n" + "="*70)
    print("PEAD SIGNAL ANALYSIS (SUE → Future Returns)")
    print("="*70)
    
    horizons = ['ret1d', 'ret5d', 'ret10d', 'ret20d', 'ret40d', 'ret60d']
    
    print("\nSUE (Standardized Unexpected Earnings):")
    for horizon in horizons:
        result = analyze_signal(pead_df, 'sue', horizon)
        if result:
            sig = '***' if result['pval'] < 0.001 else ('**' if result['pval'] < 0.01 else ('*' if result['pval'] < 0.05 else ''))
            print(f"  → {horizon}: IC={result['ic']:+.4f}{sig} (p={result['pval']:.4f}), L/S={result['spread']*100:+.2f}%")
    
    # Analyze by surprise bucket
    print("\n" + "="*70)
    print("RETURNS BY SURPRISE BUCKET")
    print("="*70)
    
    for horizon in ['ret5d', 'ret20d', 'ret60d']:
        print(f"\n{horizon}:")
        bucket_stats = analyze_by_bucket(pead_df, horizon)
        for bucket in bucket_stats.index:
            if pd.notna(bucket_stats.loc[bucket, 'mean']):
                ret = bucket_stats.loc[bucket, 'mean']
                n = bucket_stats.loc[bucket, 'count']
                print(f"  {bucket:12}: {ret*100:+.2f}% (n={int(n):,})")
    
    # Monotonicity test
    print("\n" + "="*70)
    print("QUINTILE DEEP DIVE (SUE → ret20d)")
    print("="*70)
    
    result = analyze_signal(pead_df, 'sue', 'ret20d')
    if result:
        print(f"\nIC: {result['ic']:+.4f} (p={result['pval']:.6f})")
        print(f"N: {result['n']:,}")
        print(f"\nQuintile returns:")
        for q in [1, 2, 3, 4, 5]:
            ret = result['q_stats'].loc[q, 'mean']
            n = result['q_stats'].loc[q, 'count']
            print(f"  Q{q}: {ret*100:+.2f}% (n={int(n):,})")
        print(f"\n  L/S Spread (Q5-Q1): {result['spread']*100:+.2f}%")
    
    # Beat streak analysis
    print("\n" + "="*70)
    print("BEAT STREAK ANALYSIS")
    print("="*70)
    
    print("\nDo consecutive beats/misses predict stronger drift?")
    result = analyze_signal(pead_df, 'beat_streak', 'ret20d')
    if result:
        sig = '***' if result['pval'] < 0.001 else ('**' if result['pval'] < 0.01 else ('*' if result['pval'] < 0.05 else ''))
        print(f"Beat streak → ret20d: IC={result['ic']:+.4f}{sig}")
    
    # Time stability
    print("\n" + "="*70)
    print("SIGNAL STABILITY BY YEAR")
    print("="*70)
    
    pead_df['year'] = pd.to_datetime(pead_df['quarter_date']).dt.year
    
    for year in sorted(pead_df['year'].unique()):
        subset = pead_df[pead_df['year'] == year]
        result = analyze_signal(subset, 'sue', 'ret20d')
        if result and result['n'] > 50:
            sig = '***' if result['pval'] < 0.001 else ('**' if result['pval'] < 0.01 else ('*' if result['pval'] < 0.05 else ''))
            print(f"  {year}: IC={result['ic']:+.4f}{sig} (n={result['n']:,}), L/S={result['spread']*100:+.2f}%")
    
    # Out-of-sample test
    print("\n" + "="*70)
    print("OUT-OF-SAMPLE TEST (2025+)")
    print("="*70)
    
    train = pead_df[pead_df['quarter_date'] < '2025-01-01']
    test = pead_df[pead_df['quarter_date'] >= '2025-01-01']
    
    print(f"\nTrain: {len(train):,} ({train['quarter_date'].min().date()} to {train['quarter_date'].max().date()})")
    print(f"Test:  {len(test):,} ({test['quarter_date'].min().date()} to {test['quarter_date'].max().date()})")
    
    for signal in ['sue', 'beat']:
        print(f"\n{signal}:")
        
        train_result = analyze_signal(train, signal, 'ret20d')
        if train_result:
            sig = '***' if train_result['pval'] < 0.001 else ('**' if train_result['pval'] < 0.01 else ('*' if train_result['pval'] < 0.05 else ''))
            print(f"  Train IC: {train_result['ic']:+.4f}{sig}, L/S: {train_result['spread']*100:+.2f}%")
        
        test_result = analyze_signal(test, signal, 'ret20d')
        if test_result:
            sig = '***' if test_result['pval'] < 0.001 else ('**' if test_result['pval'] < 0.01 else ('*' if test_result['pval'] < 0.05 else ''))
            print(f"  Test IC:  {test_result['ic']:+.4f}{sig}, L/S: {test_result['spread']*100:+.2f}%")
    
    # Combine with estimate revisions?
    print("\n" + "="*70)
    print("INTERACTION: SURPRISE + PRIOR REVISIONS")
    print("="*70)
    print("\nNote: To combine PEAD with estimate revisions,")
    print("merge this data with the revision backtest results.")
    
    # Save results
    output_path = 'data/pead_backtest_results.parquet'
    pead_df.to_parquet(output_path)
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Post-Earnings Announcement Drift (PEAD):
- Stocks that beat earnings tend to drift UP for 20-60 days
- Stocks that miss earnings tend to drift DOWN for 20-60 days
- This is one of the most robust anomalies in finance
- Signal strength: SUE (Standardized Unexpected Earnings) is the key metric
""")


if __name__ == "__main__":
    main()
