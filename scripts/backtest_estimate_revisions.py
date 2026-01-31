"""
Backtest: Analyst Estimate Revisions Signal

Tests whether upward/downward analyst estimate revisions predict future returns.
This is a classic alpha signal - analysts revising estimates up = bullish signal.

Data source: DefeatBeta stock_earning_estimates
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_estimate_data():
    """Load earnings estimate data from DefeatBeta."""
    from src.auto_researcher.data.defeatbeta import DefeatBetaDataLoader
    
    df = DefeatBetaDataLoader.load('stock_earning_estimates')
    if df is None:
        raise ValueError("Failed to load earnings estimates")
    
    logger.info(f"Loaded {len(df):,} estimate records")
    logger.info(f"Unique tickers: {df['symbol'].nunique():,}")
    logger.info(f"Date range: {df['report_date'].min()} to {df['report_date'].max()}")
    
    return df


def calculate_revisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate estimate revision signals.
    
    Revision = (current_estimate - prior_estimate) / |prior_estimate|
    Positive = analysts raising estimates = bullish
    """
    df = df.copy()
    
    # Convert to numeric, coercing errors
    numeric_cols = ['estimate_avg_eps', 'seven_days_ago_estimate_avg_eps', 
                    'thirty_days_ago_estimate_avg_eps', 'sixty_days_ago_estimate_avg_eps',
                    'ninety_days_ago_estimate_avg_eps', 'year_ago_estimate_avg_eps']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate revision percentages
    current = df['estimate_avg_eps']
    
    # Helper function for safe division
    def calc_revision(current, prior):
        mask = (prior.notna()) & (prior != 0) & (current.notna())
        result = pd.Series(np.nan, index=df.index)
        result.loc[mask] = (current.loc[mask] - prior.loc[mask]) / prior.loc[mask].abs()
        return result
    
    # 7-day revision
    df['revision_7d'] = calc_revision(current, df['seven_days_ago_estimate_avg_eps'])
    
    # 30-day revision
    df['revision_30d'] = calc_revision(current, df['thirty_days_ago_estimate_avg_eps'])
    
    # 60-day revision
    df['revision_60d'] = calc_revision(current, df['sixty_days_ago_estimate_avg_eps'])
    
    # 90-day revision
    df['revision_90d'] = calc_revision(current, df['ninety_days_ago_estimate_avg_eps'])
    
    # Year-over-year revision
    df['revision_yoy'] = calc_revision(current, df['year_ago_estimate_avg_eps'])
    
    # Composite revision score (average of available)
    revision_cols = ['revision_7d', 'revision_30d', 'revision_60d', 'revision_90d']
    df['revision_composite'] = df[revision_cols].mean(axis=1)
    
    # Revision acceleration (short-term vs long-term)
    df['revision_accel'] = df['revision_7d'] - df['revision_30d']
    
    # Direction flags
    df['revision_positive'] = (df['revision_30d'] > 0.01).astype(int)
    df['revision_negative'] = (df['revision_30d'] < -0.01).astype(int)
    
    return df


def get_returns(tickers: list, start_date: str = '2020-01-01') -> pd.DataFrame:
    """Fetch returns for tickers."""
    logger.info(f"Fetching returns for {len(tickers)} tickers...")
    
    # Download in batches
    all_data = []
    batch_size = 50
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, start=start_date, end='2026-01-20', progress=False)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(name=batch[0])
            all_data.append(data)
        except Exception as e:
            logger.warning(f"Failed batch {i}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    prices = pd.concat(all_data, axis=1)
    
    # Calculate forward returns
    returns = {}
    for col in prices.columns:
        returns[f'{col}_ret5d'] = prices[col].pct_change(5).shift(-5)
        returns[f'{col}_ret10d'] = prices[col].pct_change(10).shift(-10)
        returns[f'{col}_ret20d'] = prices[col].pct_change(20).shift(-20)
        returns[f'{col}_ret60d'] = prices[col].pct_change(60).shift(-60)
    
    ret_df = pd.DataFrame(returns, index=prices.index)
    
    # Reshape to long format
    records = []
    for date in ret_df.index:
        for ticker in prices.columns:
            records.append({
                'date': date,
                'ticker': ticker,
                'ret5d': ret_df.loc[date, f'{ticker}_ret5d'],
                'ret10d': ret_df.loc[date, f'{ticker}_ret10d'],
                'ret20d': ret_df.loc[date, f'{ticker}_ret20d'],
                'ret60d': ret_df.loc[date, f'{ticker}_ret60d'],
            })
    
    result = pd.DataFrame(records)
    result['date'] = pd.to_datetime(result['date']).dt.tz_localize(None)
    return result


def analyze_signal(df: pd.DataFrame, signal_col: str, return_col: str = 'ret20d'):
    """Analyze a signal's predictive power."""
    valid = df.dropna(subset=[signal_col, return_col])
    
    if len(valid) < 100:
        return None
    
    # Information Coefficient (Spearman)
    ic, pval = stats.spearmanr(valid[signal_col], valid[return_col])
    
    # Quintile analysis
    try:
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


def main():
    print("="*70)
    print("ANALYST ESTIMATE REVISIONS BACKTEST")
    print("="*70)
    
    # Load data
    estimates = load_estimate_data()
    
    # Filter to quarterly estimates only
    estimates = estimates[estimates['period_type'] == 'quarterly'].copy()
    print(f"\nQuarterly estimates: {len(estimates):,}")
    
    # Calculate revisions
    estimates = calculate_revisions(estimates)
    
    # Rename columns for merge
    estimates = estimates.rename(columns={'symbol': 'ticker', 'report_date': 'date'})
    estimates['date'] = pd.to_datetime(estimates['date'])
    
    # Get unique tickers
    tickers = estimates['ticker'].unique().tolist()
    print(f"Unique tickers: {len(tickers)}")
    
    # Fetch returns
    returns = get_returns(tickers)
    if returns.empty:
        print("Failed to get returns")
        return
    
    logger.info(f"Returns: {len(returns):,} rows")
    
    # Merge
    merged = estimates.merge(returns, on=['ticker', 'date'], how='inner')
    merged = merged.dropna(subset=['revision_30d', 'ret20d'])
    
    print(f"\nMerged dataset: {len(merged):,} rows")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    
    # Summary stats
    print("\n" + "="*70)
    print("REVISION STATISTICS")
    print("="*70)
    
    for col in ['revision_7d', 'revision_30d', 'revision_60d', 'revision_90d']:
        valid = merged[col].dropna()
        print(f"\n{col}:")
        print(f"  Mean: {valid.mean()*100:+.2f}%")
        print(f"  Median: {valid.median()*100:+.2f}%")
        print(f"  Std: {valid.std()*100:.2f}%")
        print(f"  % Positive: {(valid > 0).mean()*100:.1f}%")
    
    # Analyze each signal
    print("\n" + "="*70)
    print("SIGNAL ANALYSIS")
    print("="*70)
    
    signals = ['revision_7d', 'revision_30d', 'revision_60d', 'revision_90d', 
               'revision_composite', 'revision_accel']
    horizons = ['ret5d', 'ret10d', 'ret20d', 'ret60d']
    
    results = []
    
    for signal in signals:
        print(f"\n{signal}:")
        for horizon in horizons:
            result = analyze_signal(merged, signal, horizon)
            if result:
                sig = '***' if result['pval'] < 0.001 else ('**' if result['pval'] < 0.01 else ('*' if result['pval'] < 0.05 else ''))
                print(f"  → {horizon}: IC={result['ic']:+.4f}{sig} (p={result['pval']:.4f}), L/S={result['spread']*100:+.2f}%")
                results.append({
                    'signal': signal,
                    'horizon': horizon,
                    'ic': result['ic'],
                    'pval': result['pval'],
                    'spread': result['spread'],
                    'n': result['n']
                })
    
    # Find best signal
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['ic'].abs().idxmax()]
    
    print("\n" + "="*70)
    print("BEST SIGNAL DEEP DIVE")
    print("="*70)
    print(f"\nBest: {best['signal']} → {best['horizon']}")
    print(f"IC: {best['ic']:+.4f}, L/S Spread: {best['spread']*100:+.2f}%")
    
    # Quintile breakdown for best signal
    result = analyze_signal(merged, best['signal'], best['horizon'])
    if result:
        print(f"\nQuintile Analysis ({best['horizon']}):")
        for q in [1, 2, 3, 4, 5]:
            ret = result['q_stats'].loc[q, 'mean']
            n = result['q_stats'].loc[q, 'count']
            print(f"  Q{q}: {ret*100:+.2f}% (n={int(n):,})")
    
    # Time stability
    print("\n" + "="*70)
    print("SIGNAL STABILITY BY YEAR")
    print("="*70)
    
    merged['year'] = merged['date'].dt.year
    
    for year in sorted(merged['year'].unique()):
        subset = merged[merged['year'] == year]
        result = analyze_signal(subset, 'revision_30d', 'ret20d')
        if result and result['n'] > 50:
            sig = '***' if result['pval'] < 0.001 else ('**' if result['pval'] < 0.01 else ('*' if result['pval'] < 0.05 else ''))
            print(f"  {year}: IC={result['ic']:+.4f}{sig} (n={result['n']:,}), L/S={result['spread']*100:+.2f}%")
    
    # Out-of-sample test
    print("\n" + "="*70)
    print("OUT-OF-SAMPLE TEST (2025+)")
    print("="*70)
    
    train = merged[merged['date'] < '2025-01-01']
    test = merged[merged['date'] >= '2025-01-01']
    
    print(f"\nTrain: {len(train):,} ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test):,} ({test['date'].min().date()} to {test['date'].max().date()})")
    
    for signal in ['revision_30d', 'revision_composite']:
        print(f"\n{signal}:")
        
        train_result = analyze_signal(train, signal, 'ret20d')
        if train_result:
            sig = '***' if train_result['pval'] < 0.001 else ('**' if train_result['pval'] < 0.01 else ('*' if train_result['pval'] < 0.05 else ''))
            print(f"  Train IC: {train_result['ic']:+.4f}{sig}, L/S: {train_result['spread']*100:+.2f}%")
        
        test_result = analyze_signal(test, signal, 'ret20d')
        if test_result:
            sig = '***' if test_result['pval'] < 0.001 else ('**' if test_result['pval'] < 0.01 else ('*' if test_result['pval'] < 0.05 else ''))
            print(f"  Test IC:  {test_result['ic']:+.4f}{sig}, L/S: {test_result['spread']*100:+.2f}%")
    
    # Save results
    output_path = 'data/revision_backtest_results.parquet'
    merged.to_parquet(output_path)
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Analyst Estimate Revisions Signal:
- Upward revisions (analysts raising EPS estimates) should predict outperformance
- 30-day revision is the standard lookback
- Revision acceleration (short-term vs long-term) captures momentum shifts
""")


if __name__ == "__main__":
    main()
