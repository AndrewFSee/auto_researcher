"""
PEAD Backtest - FIXED VERSION
==============================
Uses yfinance earnings_dates for ACTUAL announcement dates instead of
DefeatBeta quarter_date (which is fiscal period end).

This fixes the forward bias found in the original backtest.
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
import time

warnings.filterwarnings('ignore')

# S&P 500 sample for testing
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
    'CME', 'WM', 'APD', 'SHW', 'FIS', 'NSC', 'ICE', 'FISV', 'CCI', 'EOG'
]


def fetch_earnings_data(ticker: str) -> pd.DataFrame:
    """
    Fetch earnings announcement dates and EPS data from yfinance.
    Returns actual announcement dates (not fiscal period ends).
    """
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_dates
        
        if earnings is None or len(earnings) == 0:
            return pd.DataFrame()
        
        # Filter to historical data only (not future estimates)
        earnings = earnings[earnings['Reported EPS'].notna()].copy()
        
        if len(earnings) == 0:
            return pd.DataFrame()
        
        # Reset index to get announcement date as column
        earnings = earnings.reset_index()
        earnings.columns = ['announcement_date', 'eps_estimate', 'eps_actual', 'surprise_pct']
        
        # Convert timezone-aware to naive datetime
        earnings['announcement_date'] = pd.to_datetime(earnings['announcement_date']).dt.tz_localize(None)
        
        earnings['ticker'] = ticker
        
        # Calculate SUE (Standardized Unexpected Earnings)
        earnings['sue'] = np.where(
            earnings['eps_estimate'] != 0,
            (earnings['eps_actual'] - earnings['eps_estimate']) / abs(earnings['eps_estimate']),
            0
        )
        
        earnings['beat'] = earnings['eps_actual'] > earnings['eps_estimate']
        
        return earnings
        
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return pd.DataFrame()


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


def calculate_forward_returns(earnings_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forward returns AFTER the announcement date.
    This is the key fix - we only look at returns AFTER the earnings are known.
    """
    results = []
    
    for _, row in earnings_df.iterrows():
        ticker = row['ticker']
        ann_date = row['announcement_date']
        
        if ticker not in prices.columns:
            continue
        
        ticker_prices = prices[ticker].dropna()
        
        # Find the first trading day ON or AFTER announcement
        # (announcement could be after market close)
        future_dates = ticker_prices.index[ticker_prices.index >= ann_date]
        if len(future_dates) < 2:
            continue
        
        # Start from the day after announcement (T+1)
        # This ensures we're not trading before the market opens after earnings
        t1_idx = 1 if len(future_dates) > 1 else 0
        base_date = future_dates[t1_idx]
        base_price = ticker_prices.loc[base_date]
        
        # Calculate various forward returns
        result = {
            'ticker': ticker,
            'announcement_date': ann_date,
            'sue': row['sue'],
            'beat': row['beat'],
            'eps_actual': row['eps_actual'],
            'eps_estimate': row['eps_estimate'],
            'surprise_pct': row['surprise_pct'],
            'base_date': base_date,
            'base_price': base_price,
        }
        
        # Forward returns: 5, 10, 20, 40, 60 days
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
    mask = signal.notna() & returns.notna()
    if mask.sum() < 30:
        return np.nan, np.nan
    
    corr, pval = stats.spearmanr(signal[mask], returns[mask])
    return corr, pval


def main():
    print("="*70)
    print("PEAD BACKTEST - FIXED VERSION")
    print("Using yfinance earnings_dates for ACTUAL announcement dates")
    print("="*70)
    print()
    
    # Step 1: Fetch earnings data for all tickers
    print("Step 1: Fetching earnings announcement dates...")
    all_earnings = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_earnings_data, t): t for t in SP500_SAMPLE}
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if len(df) > 0:
                    all_earnings.append(df)
                    print(f"  {ticker}: {len(df)} earnings events")
            except Exception as e:
                print(f"  {ticker}: Error - {e}")
    
    if not all_earnings:
        print("ERROR: No earnings data fetched")
        return
    
    earnings_df = pd.concat(all_earnings, ignore_index=True)
    print(f"\nTotal earnings events: {len(earnings_df):,}")
    print(f"Date range: {earnings_df['announcement_date'].min()} to {earnings_df['announcement_date'].max()}")
    
    # Filter to 2020-2024 for backtest (need future returns)
    earnings_df = earnings_df[
        (earnings_df['announcement_date'] >= '2020-01-01') &
        (earnings_df['announcement_date'] <= '2024-12-31')
    ]
    print(f"After date filter (2020-2024): {len(earnings_df):,} events")
    
    # Step 2: Fetch price data
    print("\nStep 2: Fetching price data...")
    tickers_with_earnings = earnings_df['ticker'].unique().tolist()
    prices = fetch_price_data(tickers_with_earnings, '2020-01-01', '2025-12-31')
    print(f"Price data: {len(prices)} trading days, {len(prices.columns)} tickers")
    
    # Step 3: Calculate forward returns
    print("\nStep 3: Calculating forward returns (AFTER announcement)...")
    results_df = calculate_forward_returns(earnings_df, prices)
    print(f"Events with returns: {len(results_df):,}")
    
    # Step 4: Calculate ICs
    print("\n" + "="*70)
    print("INFORMATION COEFFICIENTS (SUE signal)")
    print("="*70)
    print(f"{'Horizon':<12} {'IC':<10} {'p-value':<12} {'Significant':<12}")
    print("-"*46)
    
    horizons = ['ret5d', 'ret10d', 'ret20d', 'ret40d', 'ret60d']
    ic_results = {}
    
    for horizon in horizons:
        ic, pval = calculate_ic(results_df['sue'], results_df[horizon])
        ic_results[horizon] = (ic, pval)
        
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"{horizon:<12} {ic:+.4f}    {pval:.4f}       {sig}")
    
    # Step 5: Long-Short Analysis
    print("\n" + "="*70)
    print("LONG-SHORT ANALYSIS (Q5 - Q1)")
    print("="*70)
    
    # Create quintiles based on SUE
    results_df['sue_quintile'] = pd.qcut(
        results_df['sue'].rank(method='first'), 
        q=5, 
        labels=[1, 2, 3, 4, 5]
    )
    
    print(f"\n{'Horizon':<12} {'Q1 (Miss)':<12} {'Q5 (Beat)':<12} {'L/S Spread':<12}")
    print("-"*48)
    
    for horizon in horizons:
        q1_ret = results_df[results_df['sue_quintile'] == 1][horizon].mean()
        q5_ret = results_df[results_df['sue_quintile'] == 5][horizon].mean()
        spread = q5_ret - q1_ret
        
        print(f"{horizon:<12} {q1_ret*100:+.2f}%       {q5_ret*100:+.2f}%       {spread*100:+.2f}%")
    
    # Step 6: Sample events
    print("\n" + "="*70)
    print("SAMPLE EVENTS (verifying correct timing)")
    print("="*70)
    
    sample = results_df.head(10)[['ticker', 'announcement_date', 'base_date', 'sue', 'ret20d']].copy()
    sample['announcement_date'] = pd.to_datetime(sample['announcement_date']).dt.strftime('%Y-%m-%d')
    sample['base_date'] = pd.to_datetime(sample['base_date']).dt.strftime('%Y-%m-%d')
    print(sample.to_string(index=False))
    
    print("\n" + "="*70)
    print("COMPARISON: BIASED vs FIXED")
    print("="*70)
    print("""
Original (BIASED) Results using DefeatBeta quarter_date:
  - ret5d IC:  +0.008  (quarter_date = fiscal period END)
  - ret40d IC: +0.148*** 
  - ret60d IC: +0.161***
  - Problem: Returns calculated ~30-45 days BEFORE announcement!

Fixed Results using yfinance announcement_date:
  - Returns calculated AFTER actual announcement date
  - See IC table above for true predictive power
""")
    
    # Save results
    output_path = Path(__file__).parent / 'pead_fixed_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
