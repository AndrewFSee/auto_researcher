"""
PEAD Signal Enhancement
========================
Exploring ways to improve the base PEAD signal (IC=+0.047**)

Potential improvements:
1. Better SUE calculation (standardized by historical volatility)
2. Focus on big surprises only (ignore small beats/misses)
3. Time-of-day filtering (after-hours vs pre-market)
4. Combine with volume surge on announcement
5. Sector neutralization
6. Recency/momentum of surprises
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# Larger sample for better statistics
SP500_SAMPLE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CMCSA',
    'NFLX', 'XOM', 'VZ', 'INTC', 'T', 'PFE', 'ABT', 'CRM', 'CSCO', 'PEP',
    'AVGO', 'TMO', 'COST', 'MRK', 'ACN', 'NKE', 'CVX', 'WMT', 'LLY', 'ABBV',
    'MDT', 'DHR', 'NEE', 'TXN', 'UNP', 'HON', 'QCOM', 'PM', 'BMY', 'LIN',
    'ORCL', 'IBM', 'AMD', 'AMGN', 'RTX', 'LOW', 'UPS', 'GS', 'SBUX', 'CAT',
    'BA', 'BLK', 'DE', 'MMM', 'GE', 'ISRG', 'AXP', 'NOW', 'SYK', 'GILD',
    'MDLZ', 'TGT', 'MO', 'BKNG', 'ZTS', 'SPGI', 'CVS', 'ADP', 'CB', 'LMT',
    'CI', 'TJX', 'SCHW', 'BDX', 'DUK', 'SO', 'PLD', 'CL', 'MMC', 'ITW',
    'CME', 'WM', 'SHW', 'APD', 'NSC', 'FIS', 'ICE', 'EOG', 'FISV', 'CCI',
]


def fetch_earnings_with_details(ticker: str) -> pd.DataFrame:
    """Fetch earnings data with additional details for signal enhancement."""
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_dates
        
        if earnings is None or len(earnings) == 0:
            return pd.DataFrame()
        
        # Filter to historical data only
        earnings = earnings[earnings['Reported EPS'].notna()].copy()
        
        if len(earnings) == 0:
            return pd.DataFrame()
        
        earnings = earnings.reset_index()
        earnings.columns = ['announcement_date', 'eps_estimate', 'eps_actual', 'surprise_pct']
        earnings['announcement_date'] = pd.to_datetime(earnings['announcement_date']).dt.tz_localize(None)
        earnings['ticker'] = ticker
        
        # Calculate SUE
        earnings['sue'] = np.where(
            earnings['eps_estimate'] != 0,
            (earnings['eps_actual'] - earnings['eps_estimate']) / abs(earnings['eps_estimate']),
            0
        )
        
        # Extract announcement time (for after-hours analysis)
        # yfinance includes time in the announcement_date
        earnings['hour'] = pd.to_datetime(earnings['announcement_date']).dt.hour
        earnings['is_after_hours'] = earnings['hour'] >= 16  # After 4pm = after market close
        earnings['is_pre_market'] = earnings['hour'] < 9     # Before 9am = pre-market
        
        earnings['beat'] = earnings['eps_actual'] > earnings['eps_estimate']
        
        # Classify surprise magnitude
        earnings['surprise_bucket'] = pd.cut(
            earnings['sue'],
            bins=[-np.inf, -0.20, -0.05, 0.05, 0.20, np.inf],
            labels=['big_miss', 'small_miss', 'inline', 'small_beat', 'big_beat']
        )
        
        return earnings
        
    except Exception as e:
        return pd.DataFrame()


def fetch_volume_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch volume data for volume surge analysis."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        return hist[['Volume']].reset_index()
    except:
        return pd.DataFrame()


def calculate_volume_surge(ticker: str, announcement_date: pd.Timestamp, 
                           volume_data: pd.DataFrame) -> float:
    """
    Calculate volume surge on announcement day vs 20-day average.
    High volume on earnings = more conviction in price move.
    """
    if volume_data.empty:
        return np.nan
    
    volume_data = volume_data.copy()
    volume_data['Date'] = pd.to_datetime(volume_data['Date']).dt.tz_localize(None)
    
    # Find announcement day volume
    ann_day = volume_data[volume_data['Date'].dt.date == announcement_date.date()]
    if ann_day.empty:
        # Try next trading day
        future = volume_data[volume_data['Date'] > announcement_date]
        if future.empty:
            return np.nan
        ann_day = future.iloc[:1]
    
    ann_volume = ann_day['Volume'].iloc[0]
    
    # Calculate 20-day average before announcement
    prior = volume_data[volume_data['Date'] < announcement_date].tail(20)
    if len(prior) < 10:
        return np.nan
    
    avg_volume = prior['Volume'].mean()
    
    if avg_volume > 0:
        return ann_volume / avg_volume
    return np.nan


def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data for all tickers."""
    print(f"Fetching prices for {len(tickers)} tickers...")
    
    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True
    )['Close']
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    
    return prices


def calculate_enhanced_returns(earnings_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate forward returns with enhanced signals."""
    results = []
    
    for _, row in earnings_df.iterrows():
        ticker = row['ticker']
        ann_date = row['announcement_date']
        
        if ticker not in prices.columns:
            continue
        
        ticker_prices = prices[ticker].dropna()
        
        # Find T+1 trading day
        future_dates = ticker_prices.index[ticker_prices.index >= ann_date]
        if len(future_dates) < 2:
            continue
        
        t1_idx = 1 if len(future_dates) > 1 else 0
        base_date = future_dates[t1_idx]
        base_price = ticker_prices.loc[base_date]
        
        result = {
            'ticker': ticker,
            'announcement_date': ann_date,
            'sue': row['sue'],
            'beat': row['beat'],
            'surprise_pct': row['surprise_pct'],
            'surprise_bucket': row['surprise_bucket'],
            'is_after_hours': row['is_after_hours'],
            'is_pre_market': row['is_pre_market'],
            'hour': row['hour'],
        }
        
        # Forward returns
        for days in [5, 10, 20, 40, 60]:
            target_date = base_date + timedelta(days=days)
            future_prices = ticker_prices[ticker_prices.index >= target_date]
            
            if len(future_prices) > 0:
                end_price = future_prices.iloc[0]
                ret = (end_price / base_price) - 1
                result[f'ret{days}d'] = ret
            else:
                result[f'ret{days}d'] = np.nan
        
        results.append(result)
    
    return pd.DataFrame(results)


def calculate_ic(signal: pd.Series, returns: pd.Series) -> tuple:
    """Calculate Spearman IC with p-value."""
    mask = signal.notna() & returns.notna() & np.isfinite(signal) & np.isfinite(returns)
    if mask.sum() < 30:
        return np.nan, np.nan
    
    corr, pval = stats.spearmanr(signal[mask], returns[mask])
    return corr, pval


def create_enhanced_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Create various enhanced PEAD signals."""
    df = df.copy()
    
    # 1. Big surprise only signal (ignore small moves)
    df['sue_big_only'] = np.where(
        df['surprise_bucket'].isin(['big_beat', 'big_miss']),
        df['sue'],
        np.nan
    )
    
    # 2. Extreme surprise signal (top/bottom decile)
    sue_90 = df['sue'].quantile(0.90)
    sue_10 = df['sue'].quantile(0.10)
    df['sue_extreme'] = np.where(
        (df['sue'] >= sue_90) | (df['sue'] <= sue_10),
        df['sue'],
        np.nan
    )
    
    # 3. Binary beat/miss (simpler signal)
    df['beat_binary'] = df['beat'].astype(float) - 0.5  # -0.5 or +0.5
    
    # 4. Capped SUE (reduce outlier impact)
    df['sue_capped'] = np.clip(df['sue'], -0.5, 0.5)
    
    # 5. Signed sqrt (reduce right-skew of SUE)
    df['sue_sqrt'] = np.sign(df['sue']) * np.sqrt(np.abs(df['sue']))
    
    # 6. Rank-based signal (robust to outliers)
    df['sue_rank'] = df['sue'].rank(pct=True) - 0.5  # Center at 0
    
    return df


def main():
    print("="*70)
    print("PEAD SIGNAL ENHANCEMENT")
    print("Finding ways to improve IC=+0.047** baseline")
    print("="*70)
    print()
    
    # Step 1: Fetch earnings data
    print("Step 1: Fetching earnings data with details...")
    all_earnings = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_earnings_with_details, t): t for t in SP500_SAMPLE}
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if len(df) > 0:
                    all_earnings.append(df)
            except:
                pass
    
    earnings_df = pd.concat(all_earnings, ignore_index=True)
    
    # Filter to 2020-2024
    earnings_df = earnings_df[
        (earnings_df['announcement_date'] >= '2020-01-01') &
        (earnings_df['announcement_date'] <= '2024-12-31')
    ]
    print(f"Total earnings events: {len(earnings_df):,}")
    
    # Step 2: Fetch prices
    print("\nStep 2: Fetching prices...")
    tickers = earnings_df['ticker'].unique().tolist()
    prices = fetch_price_data(tickers, '2020-01-01', '2025-12-31')
    
    # Step 3: Calculate returns
    print("\nStep 3: Calculating returns...")
    results_df = calculate_enhanced_returns(earnings_df, prices)
    print(f"Events with returns: {len(results_df):,}")
    
    # Step 4: Create enhanced signals
    print("\nStep 4: Creating enhanced signals...")
    results_df = create_enhanced_signals(results_df)
    
    # Step 5: Test all signals
    print("\n" + "="*70)
    print("SIGNAL COMPARISON (ret40d horizon)")
    print("="*70)
    print(f"{'Signal':<20} {'N':<8} {'IC':<10} {'p-value':<10} {'vs Base'}")
    print("-"*58)
    
    signals = [
        ('sue (baseline)', 'sue'),
        ('sue_capped', 'sue_capped'),
        ('sue_sqrt', 'sue_sqrt'),
        ('sue_rank', 'sue_rank'),
        ('beat_binary', 'beat_binary'),
        ('sue_big_only', 'sue_big_only'),
        ('sue_extreme', 'sue_extreme'),
    ]
    
    baseline_ic = None
    for name, col in signals:
        if col not in results_df.columns:
            continue
        
        valid = results_df[results_df[col].notna()]
        ic, pval = calculate_ic(valid[col], valid['ret40d'])
        
        if baseline_ic is None:
            baseline_ic = ic
            improvement = "baseline"
        else:
            if pd.notna(ic) and pd.notna(baseline_ic):
                improvement = f"{((ic/baseline_ic)-1)*100:+.0f}%" if baseline_ic != 0 else "N/A"
            else:
                improvement = "N/A"
        
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        n = valid[col].notna().sum()
        
        if pd.notna(ic):
            print(f"{name:<20} {n:<8} {ic:+.4f}    {pval:.4f}     {improvement}")
        else:
            print(f"{name:<20} {n:<8} N/A")
    
    # Step 6: Test time-of-day effect
    print("\n" + "="*70)
    print("TIME-OF-DAY ANALYSIS")
    print("="*70)
    
    after_hours = results_df[results_df['is_after_hours'] == True]
    pre_market = results_df[results_df['is_pre_market'] == True]
    during_hours = results_df[(results_df['is_after_hours'] == False) & (results_df['is_pre_market'] == False)]
    
    for name, subset in [('After-hours', after_hours), ('Pre-market', pre_market), ('During hours', during_hours)]:
        if len(subset) < 50:
            continue
        ic, pval = calculate_ic(subset['sue'], subset['ret40d'])
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"{name:<15} N={len(subset):<6} IC={ic:+.4f}{sig}")
    
    # Step 7: Surprise magnitude analysis
    print("\n" + "="*70)
    print("SURPRISE MAGNITUDE ANALYSIS")
    print("="*70)
    print(f"{'Bucket':<15} {'N':<8} {'Avg ret40d':<12} {'Avg ret60d'}")
    print("-"*45)
    
    for bucket in ['big_miss', 'small_miss', 'inline', 'small_beat', 'big_beat']:
        subset = results_df[results_df['surprise_bucket'] == bucket]
        if len(subset) > 10:
            ret40 = subset['ret40d'].mean() * 100
            ret60 = subset['ret60d'].mean() * 100
            print(f"{bucket:<15} {len(subset):<8} {ret40:+.2f}%        {ret60:+.2f}%")
    
    # Long-short for extremes only
    print("\n" + "="*70)
    print("EXTREME SURPRISES ONLY (big_beat vs big_miss)")
    print("="*70)
    
    big_beat = results_df[results_df['surprise_bucket'] == 'big_beat']
    big_miss = results_df[results_df['surprise_bucket'] == 'big_miss']
    
    print(f"{'Horizon':<12} {'Big Miss':<12} {'Big Beat':<12} {'L/S Spread'}")
    print("-"*48)
    
    for horizon in ['ret5d', 'ret10d', 'ret20d', 'ret40d', 'ret60d']:
        miss_ret = big_miss[horizon].mean()
        beat_ret = big_beat[horizon].mean()
        spread = beat_ret - miss_ret
        print(f"{horizon:<12} {miss_ret*100:+.2f}%       {beat_ret*100:+.2f}%       {spread*100:+.2f}%")
    
    # Step 8: Quintile analysis with enhanced signal
    print("\n" + "="*70)
    print("ENHANCED SIGNAL: SUE_CAPPED QUINTILE ANALYSIS")
    print("="*70)
    
    results_df['sue_capped_quintile'] = pd.qcut(
        results_df['sue_capped'].rank(method='first'),
        q=5,
        labels=[1, 2, 3, 4, 5]
    )
    
    print(f"{'Horizon':<12} {'Q1':<10} {'Q2':<10} {'Q3':<10} {'Q4':<10} {'Q5':<10} {'Q5-Q1'}")
    print("-"*72)
    
    for horizon in ['ret20d', 'ret40d', 'ret60d']:
        q_rets = []
        for q in [1, 2, 3, 4, 5]:
            r = results_df[results_df['sue_capped_quintile'] == q][horizon].mean() * 100
            q_rets.append(r)
        
        spread = q_rets[4] - q_rets[0]
        print(f"{horizon:<12} {q_rets[0]:+.2f}%    {q_rets[1]:+.2f}%    {q_rets[2]:+.2f}%    {q_rets[3]:+.2f}%    {q_rets[4]:+.2f}%    {spread:+.2f}%")
    
    # Step 9: Momentum filter - consecutive beats
    print("\n" + "="*70)
    print("CONSECUTIVE BEATS/MISSES ANALYSIS")
    print("="*70)
    
    # Sort by ticker and date to find streaks
    results_df = results_df.sort_values(['ticker', 'announcement_date'])
    results_df['prev_beat'] = results_df.groupby('ticker')['beat'].shift(1)
    
    # Consecutive beat (current beat AND previous beat)
    consec_beat = results_df[(results_df['beat'] == True) & (results_df['prev_beat'] == True)]
    consec_miss = results_df[(results_df['beat'] == False) & (results_df['prev_beat'] == False)]
    single_beat = results_df[(results_df['beat'] == True) & (results_df['prev_beat'] == False)]
    single_miss = results_df[(results_df['beat'] == False) & (results_df['prev_beat'] == True)]
    
    print(f"{'Type':<25} {'N':<8} {'ret40d':<10} {'ret60d'}")
    print("-"*50)
    print(f"{'Consecutive beats':<25} {len(consec_beat):<8} {consec_beat['ret40d'].mean()*100:+.2f}%     {consec_beat['ret60d'].mean()*100:+.2f}%")
    print(f"{'Single beat (reversal)':<25} {len(single_beat):<8} {single_beat['ret40d'].mean()*100:+.2f}%     {single_beat['ret60d'].mean()*100:+.2f}%")
    print(f"{'Single miss (reversal)':<25} {len(single_miss):<8} {single_miss['ret40d'].mean()*100:+.2f}%     {single_miss['ret60d'].mean()*100:+.2f}%")
    print(f"{'Consecutive misses':<25} {len(consec_miss):<8} {consec_miss['ret40d'].mean()*100:+.2f}%     {consec_miss['ret60d'].mean()*100:+.2f}%")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: BEST IMPROVEMENTS")
    print("="*70)
    print("""
Key findings for improving PEAD signal:

1. SUE_CAPPED (clip to ±50%): Reduces outlier noise
   - May improve IC slightly by reducing extreme value influence

2. EXTREME SURPRISES ONLY: Focus on big beats/misses (>20% surprise)
   - Fewer trades but potentially stronger signal
   - Big beat vs big miss spread is much larger

3. CONSECUTIVE BEATS: Momentum in earnings
   - Stocks that beat 2+ quarters in a row may drift more
   - Could be combined with SUE for enhanced signal

4. TIME-OF-DAY: After-hours announcements
   - Most announcements are after-hours
   - May have different dynamics than pre-market

Recommended enhanced signal:
  - Use SUE_CAPPED for robustness
  - Filter to |SUE| > 0.10 for stronger signals
  - Consider consecutive beat/miss as additional factor
""")
    
    # Save results
    output_path = Path(__file__).parent / 'pead_enhanced_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
