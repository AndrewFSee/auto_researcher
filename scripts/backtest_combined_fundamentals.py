"""
Combined Fundamentals Alpha Model
=================================
Combines two proven signals:
1. Analyst Estimate Revisions (IC=+0.071***)
2. Post-Earnings Announcement Drift / SUE (IC=+0.173***)

Strategy: Use revisions for timing, PEAD for confirmation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from src.auto_researcher.data.defeatbeta import DefeatBetaDataLoader

# ============================================================================
# DATA LOADING
# ============================================================================

def load_revision_data():
    """Load estimate revision data"""
    logger.info("Loading estimate revision data...")
    df = DefeatBetaDataLoader.load('stock_earning_estimates')
    
    # Parse date
    df['date'] = pd.to_datetime(df['report_date'])
    df['ticker'] = df['symbol']
    
    # Calculate revisions using correct column names
    for col in ['seven_days_ago_estimate_avg_eps', 'thirty_days_ago_estimate_avg_eps', 
                'sixty_days_ago_estimate_avg_eps', 'ninety_days_ago_estimate_avg_eps']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['current_eps'] = pd.to_numeric(df['estimate_avg_eps'], errors='coerce')
    
    # Calculate revision signals
    df['revision_30d'] = (df['current_eps'] - df['thirty_days_ago_estimate_avg_eps']) / df['thirty_days_ago_estimate_avg_eps'].abs().replace(0, np.nan)
    df['revision_60d'] = (df['current_eps'] - df['sixty_days_ago_estimate_avg_eps']) / df['sixty_days_ago_estimate_avg_eps'].abs().replace(0, np.nan)
    
    # Winsorize
    for col in ['revision_30d', 'revision_60d']:
        df[col] = df[col].clip(-2, 2)
    
    return df[['ticker', 'date', 'current_eps', 'revision_30d', 'revision_60d']].dropna()


def load_pead_data():
    """Load PEAD/EPS surprise data"""
    logger.info("Loading PEAD data...")
    df = DefeatBetaDataLoader.load('stock_historical_eps')
    
    # Parse date - use quarter_date
    df['report_date'] = pd.to_datetime(df['quarter_date'], errors='coerce')
    df['ticker'] = df['symbol']
    
    # Calculate SUE
    df['eps_actual'] = pd.to_numeric(df['eps_actual'], errors='coerce')
    df['eps_estimate'] = pd.to_numeric(df['eps_estimate'], errors='coerce')
    
    df = df.dropna(subset=['eps_actual', 'eps_estimate', 'report_date'])
    
    df['sue'] = (df['eps_actual'] - df['eps_estimate']) / df['eps_estimate'].abs().replace(0, np.nan)
    df['beat'] = (df['eps_actual'] > df['eps_estimate']).astype(int)
    
    # Winsorize SUE
    df['sue'] = df['sue'].clip(-2, 2)
    
    return df[['ticker', 'report_date', 'sue', 'beat']].dropna()


def merge_signals(revision_df, pead_df):
    """
    Merge revision and PEAD signals.
    
    Key insight: Match on the SAME fiscal quarter.
    - revision_df['date'] = fiscal quarter end date (forward-looking estimates)
    - pead_df['report_date'] = quarter date when EPS was reported
    
    We match revision estimates with the actual EPS for that quarter.
    Signal: stocks where analysts RAISED estimates AND company BEAT those estimates.
    """
    print("\nMerging signals...")
    
    # Normalize quarter dates to quarter-end
    revision_df = revision_df.copy()
    pead_df = pead_df.copy()
    
    # Round to quarter end (Q1=Mar, Q2=Jun, Q3=Sep, Q4=Dec)
    revision_df['quarter'] = revision_df['date'].dt.to_period('Q').dt.end_time.dt.date
    pead_df['quarter'] = pead_df['report_date'].dt.to_period('Q').dt.end_time.dt.date
    
    # Merge on ticker + quarter
    merged = pd.merge(
        revision_df[['ticker', 'quarter', 'current_eps', 'revision_30d', 'revision_60d']],
        pead_df[['ticker', 'quarter', 'report_date', 'sue', 'beat']],
        on=['ticker', 'quarter'],
        how='inner'
    )
    
    # Use report_date as the event date for return calculation
    merged['earnings_date'] = merged['report_date']
    
    print(f"  Merged dataset: {len(merged):,} earnings events with revision data")
    print(f"  Date range: {merged['earnings_date'].min()} to {merged['earnings_date'].max()}")
    
    return merged


def fetch_post_earnings_returns(df, horizons=[5, 10, 20, 40, 60]):
    """Fetch returns after earnings for merged dataset"""
    print(f"\nFetching post-earnings returns for {len(df['ticker'].unique())} tickers...")
    
    tickers = df['ticker'].unique().tolist()
    
    # Get price data
    start_date = df['earnings_date'].min() - timedelta(days=5)
    end_date = df['earnings_date'].max() + timedelta(days=max(horizons) + 5)
    
    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True
    )['Close']
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    
    # Calculate returns for each earnings event
    results = []
    for _, row in df.iterrows():
        ticker = row['ticker']
        earn_date = row['earnings_date']
        
        if ticker not in prices.columns:
            continue
            
        ticker_prices = prices[ticker].dropna()
        
        # Find trading day on or after earnings
        valid_dates = ticker_prices.index[ticker_prices.index >= earn_date]
        if len(valid_dates) == 0:
            continue
            
        base_date = valid_dates[0]
        base_price = ticker_prices.loc[base_date]
        
        result = row.to_dict()
        
        # Calculate returns for each horizon
        for h in horizons:
            future_dates = ticker_prices.index[ticker_prices.index >= base_date + timedelta(days=h)]
            if len(future_dates) > 0:
                future_price = ticker_prices.loc[future_dates[0]]
                result[f'ret{h}d'] = (future_price / base_price - 1) * 100
        
        results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# SIGNAL CONSTRUCTION
# ============================================================================

def create_combined_signals(df):
    """Create combined alpha signals"""
    print("\nCreating combined signals...")
    
    # 1. Simple average of z-scored signals
    df['revision_z'] = (df['revision_60d'] - df['revision_60d'].mean()) / df['revision_60d'].std()
    df['sue_z'] = (df['sue'] - df['sue'].mean()) / df['sue'].std()
    
    df['combined_simple'] = (df['revision_z'] + df['sue_z']) / 2
    
    # 2. Weighted combination (PEAD has higher IC historically)
    df['combined_weighted'] = 0.4 * df['revision_z'] + 0.6 * df['sue_z']
    
    # 3. Interaction signal: revision * sue (double positive/negative)
    df['interaction'] = df['revision_z'] * df['sue_z']
    
    # 4. Categorical: revision_up + beat
    df['revision_up'] = (df['revision_60d'] > 0).astype(int)
    df['double_positive'] = (df['revision_up'] & df['beat']).astype(int)
    df['double_negative'] = ((~df['revision_up'].astype(bool)) & (~df['beat'].astype(bool))).astype(int)
    
    # 5. Confirmation score: how well does revision predict beat?
    df['confirmation'] = df['double_positive'] - df['double_negative']
    
    return df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_ic(df, signals, ret_col='ret20d'):
    """Calculate IC for each signal"""
    print(f"\n{'='*70}")
    print(f"INFORMATION COEFFICIENT ANALYSIS ({ret_col})")
    print('='*70)
    
    results = []
    for sig in signals:
        valid = df[[sig, ret_col]].dropna()
        if len(valid) < 100:
            continue
            
        ic, p = stats.spearmanr(valid[sig], valid[ret_col])
        
        # Long-short spread
        q1 = valid[valid[sig] <= valid[sig].quantile(0.2)][ret_col].mean()
        q5 = valid[valid[sig] >= valid[sig].quantile(0.8)][ret_col].mean()
        ls_spread = q5 - q1
        
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {sig:25s}: IC={ic:+.4f}{stars:3s} (p={p:.4f}), L/S={ls_spread:+.2f}%")
        
        results.append({
            'signal': sig,
            'ic': ic,
            'p_value': p,
            'ls_spread': ls_spread,
            'n': len(valid)
        })
    
    return pd.DataFrame(results)


def analyze_double_signals(df, ret_col='ret20d'):
    """Analyze the double positive/negative categories"""
    print(f"\n{'='*70}")
    print("DOUBLE SIGNAL ANALYSIS")
    print('='*70)
    
    # Create categories
    df['category'] = 'neutral'
    df.loc[df['double_positive'] == 1, 'category'] = 'double_positive'
    df.loc[df['double_negative'] == 1, 'category'] = 'double_negative'
    df.loc[(df['revision_up'] == 1) & (df['beat'] == 0), 'category'] = 'revision_up_miss'
    df.loc[(df['revision_up'] == 0) & (df['beat'] == 1), 'category'] = 'revision_down_beat'
    
    print(f"\nCategory analysis for {ret_col}:")
    print("-" * 50)
    
    for cat in ['double_negative', 'revision_up_miss', 'revision_down_beat', 'double_positive']:
        subset = df[df['category'] == cat]
        if len(subset) < 20:
            continue
        mean_ret = subset[ret_col].mean()
        n = len(subset)
        print(f"  {cat:25s}: {mean_ret:+.2f}% (n={n:,})")
    
    # Calculate spread
    dp = df[df['category'] == 'double_positive'][ret_col].mean()
    dn = df[df['category'] == 'double_negative'][ret_col].mean()
    print(f"\n  Double Positive - Double Negative Spread: {dp - dn:+.2f}%")
    
    return df


def quintile_analysis(df, signal, ret_col='ret20d'):
    """Detailed quintile breakdown"""
    print(f"\n{'='*70}")
    print(f"QUINTILE ANALYSIS: {signal} → {ret_col}")
    print('='*70)
    
    valid = df[[signal, ret_col]].dropna()
    valid['quintile'] = pd.qcut(valid[signal], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        subset = valid[valid['quintile'] == q]
        print(f"  {q}: {subset[ret_col].mean():+.2f}% (n={len(subset):,})")
    
    q1_ret = valid[valid['quintile'] == 'Q1'][ret_col].mean()
    q5_ret = valid[valid['quintile'] == 'Q5'][ret_col].mean()
    print(f"\n  L/S Spread (Q5-Q1): {q5_ret - q1_ret:+.2f}%")


def oos_test(df, signals, train_end='2024-12-31'):
    """Out-of-sample test"""
    print(f"\n{'='*70}")
    print(f"OUT-OF-SAMPLE TEST (Train ≤{train_end}, Test >{train_end})")
    print('='*70)
    
    train_end = pd.to_datetime(train_end)
    train = df[df['earnings_date'] <= train_end]
    test = df[df['earnings_date'] > train_end]
    
    print(f"\nTrain: {len(train):,} events ({train['earnings_date'].min().date()} to {train['earnings_date'].max().date()})")
    print(f"Test:  {len(test):,} events ({test['earnings_date'].min().date()} to {test['earnings_date'].max().date()})")
    
    ret_col = 'ret20d'
    
    print(f"\nSignal performance ({ret_col}):")
    print("-" * 60)
    
    for sig in signals:
        # Train IC
        train_valid = train[[sig, ret_col]].dropna()
        train_ic, train_p = stats.spearmanr(train_valid[sig], train_valid[ret_col])
        
        # Test IC
        test_valid = test[[sig, ret_col]].dropna()
        if len(test_valid) < 50:
            continue
        test_ic, test_p = stats.spearmanr(test_valid[sig], test_valid[ret_col])
        
        # Test L/S
        q1 = test_valid[test_valid[sig] <= test_valid[sig].quantile(0.2)][ret_col].mean()
        q5 = test_valid[test_valid[sig] >= test_valid[sig].quantile(0.8)][ret_col].mean()
        test_ls = q5 - q1
        
        train_stars = '***' if train_p < 0.001 else '**' if train_p < 0.01 else '*' if train_p < 0.05 else ''
        test_stars = '***' if test_p < 0.001 else '**' if test_p < 0.01 else '*' if test_p < 0.05 else ''
        
        print(f"  {sig:25s}: Train IC={train_ic:+.3f}{train_stars:3s}, Test IC={test_ic:+.3f}{test_stars:3s}, Test L/S={test_ls:+.2f}%")


def yearly_stability(df, signal='combined_weighted', ret_col='ret20d'):
    """Check signal stability by year"""
    print(f"\n{'='*70}")
    print(f"YEARLY STABILITY: {signal}")
    print('='*70)
    
    df['year'] = df['earnings_date'].dt.year
    
    for year in sorted(df['year'].unique()):
        subset = df[df['year'] == year][[signal, ret_col]].dropna()
        if len(subset) < 50:
            continue
            
        ic, p = stats.spearmanr(subset[signal], subset[ret_col])
        
        # L/S
        q1 = subset[subset[signal] <= subset[signal].quantile(0.2)][ret_col].mean()
        q5 = subset[subset[signal] >= subset[signal].quantile(0.8)][ret_col].mean()
        ls = q5 - q1
        
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {year}: IC={ic:+.4f}{stars:3s} (n={len(subset):,}), L/S={ls:+.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("COMBINED FUNDAMENTALS ALPHA MODEL")
    print("Estimate Revisions + PEAD")
    print("="*70)
    
    # Load data
    revision_df = load_revision_data()
    pead_df = load_pead_data()
    
    print(f"\nRevision data: {len(revision_df):,} records")
    print(f"PEAD data: {len(pead_df):,} records")
    
    # Merge signals
    merged = merge_signals(revision_df, pead_df)
    
    if len(merged) < 100:
        print("ERROR: Not enough merged data. Check date alignment.")
        return
    
    # Fetch returns
    df = fetch_post_earnings_returns(merged)
    print(f"\nFinal dataset: {len(df):,} events with returns")
    
    # Create combined signals
    df = create_combined_signals(df)
    
    # Define signals to test
    signals = [
        'revision_60d',     # Individual: revision
        'sue',              # Individual: PEAD
        'combined_simple',  # Equal weight
        'combined_weighted',# Weighted (favor PEAD)
        'interaction',      # Multiplication
        'confirmation'      # Categorical
    ]
    
    # IC analysis across horizons
    for ret_col in ['ret5d', 'ret10d', 'ret20d', 'ret40d', 'ret60d']:
        if ret_col in df.columns:
            analyze_ic(df, signals, ret_col)
    
    # Double signal analysis
    df = analyze_double_signals(df, 'ret20d')
    
    # Quintile analysis for best signal
    quintile_analysis(df, 'combined_weighted', 'ret20d')
    quintile_analysis(df, 'combined_weighted', 'ret60d')
    
    # Yearly stability
    yearly_stability(df, 'combined_weighted', 'ret20d')
    
    # OOS test
    oos_test(df, signals)
    
    # Save results
    output_path = Path(__file__).parent.parent / 'data' / 'combined_fundamentals_results.parquet'
    output_path.parent.mkdir(exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Combined Fundamentals Alpha:
- Merges analyst estimate revisions with post-earnings drift
- Key insight: Stocks where analysts RAISE estimates AND company BEATS 
  those raised estimates show the strongest drift
- 'double_positive' = revision_up + beat = strongest signal
- 'double_negative' = revision_down + miss = weakest returns

Next steps:
1. If combined signals work OOS, build a production model
2. Consider adding earnings call sentiment from transcripts
3. Implement with proper position sizing and risk management
""")


if __name__ == "__main__":
    main()
