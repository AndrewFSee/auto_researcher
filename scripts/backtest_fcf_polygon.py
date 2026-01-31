"""
FCF Yield Backtest - FIXED VERSION using Polygon API
======================================================
Uses Polygon's filing_date for ACTUAL public availability date.
Polygon provides years of historical financial data with proper timing.

This fixes the forward bias found in the original backtest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
import time

warnings.filterwarnings('ignore')
load_dotenv()

# S&P 500 sample
SP500_SAMPLE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CMCSA',
    'NFLX', 'XOM', 'VZ', 'INTC', 'T', 'PFE', 'ABT', 'CRM', 'CSCO', 'PEP',
    'AVGO', 'TMO', 'COST', 'MRK', 'ACN', 'NKE', 'CVX', 'WMT', 'LLY', 'ABBV',
    'MDT', 'DHR', 'NEE', 'TXN', 'UNP', 'HON', 'QCOM', 'PM', 'BMY', 'LIN',
    'ORCL', 'IBM', 'AMD', 'AMGN', 'RTX', 'LOW', 'UPS', 'GS', 'SBUX', 'CAT',
    'BA', 'BLK', 'DE', 'MMM', 'GE', 'ISRG', 'AXP', 'NOW', 'SYK', 'GILD',
]


def fetch_polygon_financials(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Fetch financial data from Polygon with filing dates.
    """
    url = f'https://api.polygon.io/vX/reference/financials'
    params = {
        'ticker': ticker,
        'limit': 50,  # Get many quarters
        'timeframe': 'quarterly',
        'apiKey': api_key,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        
        if resp.status_code != 200:
            return pd.DataFrame()
        
        data = resp.json()
        results = data.get('results', [])
        
        records = []
        for r in results:
            # Skip TTM and annual for quarterly analysis
            if r.get('fiscal_period') in ['TTM', 'FY']:
                continue
            
            filing_date = r.get('filing_date')
            if not filing_date:
                continue  # Skip if no filing date
            
            cf = r.get('financials', {}).get('cash_flow_statement', {})
            bs = r.get('financials', {}).get('balance_sheet', {})
            
            # Operating Cash Flow
            ocf = cf.get('net_cash_flow_from_operating_activities', {}).get('value')
            
            # CapEx (usually negative)
            capex = cf.get('net_cash_flow_from_investing_activities_continuing', {}).get('value')
            
            # If no direct capex, estimate from PP&E changes
            if capex is None:
                capex = cf.get('net_cash_flow_from_investing_activities', {}).get('value', 0)
            
            # Total Assets for normalization
            total_assets = bs.get('assets', {}).get('value')
            
            if ocf is not None:
                # FCF = OCF - CapEx (CapEx is usually negative in data, so we add)
                fcf = ocf + (capex if capex else 0)
                
                records.append({
                    'ticker': ticker,
                    'fiscal_year': r.get('fiscal_year'),
                    'fiscal_period': r.get('fiscal_period'),
                    'period_end': pd.to_datetime(r.get('end_date')),
                    'filing_date': pd.to_datetime(filing_date),
                    'ocf': ocf,
                    'capex': capex,
                    'fcf': fcf,
                    'total_assets': total_assets,
                })
        
        return pd.DataFrame(records)
        
    except Exception as e:
        print(f"  {ticker}: Polygon error - {e}")
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


def get_market_caps(tickers: list) -> dict:
    """Get current market caps for rough normalization."""
    caps = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            caps[ticker] = stock.info.get('marketCap', None)
        except:
            caps[ticker] = None
    return caps


def calculate_signals_and_returns(financials_df: pd.DataFrame, prices: pd.DataFrame, market_caps: dict) -> pd.DataFrame:
    """
    Calculate FCF Yield signals using FILING DATE (proper timing).
    """
    results = []
    
    for _, row in financials_df.iterrows():
        ticker = row['ticker']
        filing_date = row['filing_date']
        fcf = row['fcf']
        
        if ticker not in prices.columns:
            continue
        
        ticker_prices = prices[ticker].dropna()
        
        # Find trading day on or after filing date
        future_dates = ticker_prices.index[ticker_prices.index >= filing_date]
        if len(future_dates) < 2:
            continue
        
        # Use T+1 to ensure filing is fully public
        base_date = future_dates[1] if len(future_dates) > 1 else future_dates[0]
        base_price = ticker_prices.loc[base_date]
        
        # Get market cap for FCF Yield calculation
        market_cap = market_caps.get(ticker)
        if market_cap is None or market_cap <= 0:
            continue
        
        # FCF Yield = FCF (annualized) / Market Cap
        # Note: Using current market cap is a simplification; in production
        # would use historical market cap at filing_date
        fcf_annualized = fcf * 4  # Quarterly to annual
        fcf_yield = fcf_annualized / market_cap
        
        # Also calculate FCF/Assets as alternative signal
        fcf_assets = fcf / row['total_assets'] if row['total_assets'] and row['total_assets'] > 0 else None
        
        result = {
            'ticker': ticker,
            'fiscal_year': row['fiscal_year'],
            'fiscal_period': row['fiscal_period'],
            'period_end': row['period_end'],
            'filing_date': filing_date,
            'base_date': base_date,
            'fcf': fcf,
            'fcf_yield': fcf_yield,
            'fcf_assets': fcf_assets,
            'market_cap': market_cap,
            'lag_days': (filing_date - row['period_end']).days,
        }
        
        # Calculate forward returns
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


def main():
    print("="*70)
    print("FCF YIELD BACKTEST - FIXED VERSION (Polygon API)")
    print("Using Polygon's filing_date for ACTUAL public availability")
    print("="*70)
    print()
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set in .env")
        return
    
    # Step 1: Fetch financial data from Polygon
    print("Step 1: Fetching financial data from Polygon...")
    all_financials = []
    
    for ticker in SP500_SAMPLE:
        df = fetch_polygon_financials(ticker, api_key)
        if len(df) > 0:
            all_financials.append(df)
            print(f"  {ticker}: {len(df)} quarters")
        time.sleep(0.15)  # Rate limit (free tier: ~5 req/min)
    
    if not all_financials:
        print("ERROR: No financial data fetched")
        return
    
    financials_df = pd.concat(all_financials, ignore_index=True)
    print(f"\nTotal financial records: {len(financials_df):,}")
    print(f"Tickers: {financials_df['ticker'].nunique()}")
    print(f"Date range: {financials_df['filing_date'].min()} to {financials_df['filing_date'].max()}")
    
    # Filter to 2020-2024 for proper backtest
    financials_df = financials_df[
        (financials_df['filing_date'] >= '2020-01-01') &
        (financials_df['filing_date'] <= '2024-12-31')
    ]
    print(f"After date filter (2020-2024): {len(financials_df):,}")
    
    # Step 2: Fetch price data
    print("\nStep 2: Fetching price data...")
    tickers_with_data = financials_df['ticker'].unique().tolist()
    prices = fetch_price_data(tickers_with_data, '2020-01-01', '2025-12-31')
    print(f"Price data: {len(prices)} trading days, {len(prices.columns)} tickers")
    
    # Step 3: Get market caps
    print("\nStep 3: Fetching market caps...")
    market_caps = get_market_caps(tickers_with_data)
    valid_caps = sum(1 for v in market_caps.values() if v)
    print(f"Market caps available: {valid_caps}/{len(tickers_with_data)}")
    
    # Step 4: Calculate signals
    print("\nStep 4: Calculating FCF Yield signals (using filing dates)...")
    results_df = calculate_signals_and_returns(financials_df, prices, market_caps)
    print(f"Events with returns: {len(results_df):,}")
    
    # Filter extreme FCF yield values
    results_df = results_df[
        (results_df['fcf_yield'] > -0.5) &
        (results_df['fcf_yield'] < 0.5)
    ]
    print(f"After filtering extremes: {len(results_df):,}")
    
    if len(results_df) < 50:
        print("WARNING: Low sample size, results may not be reliable")
    
    # Step 5: Calculate ICs
    print("\n" + "="*70)
    print("INFORMATION COEFFICIENTS")
    print("="*70)
    
    signals = ['fcf_yield', 'fcf_assets']
    horizons = ['ret5d', 'ret10d', 'ret20d', 'ret40d', 'ret60d']
    
    for signal_name in signals:
        if signal_name not in results_df.columns:
            continue
        print(f"\n{signal_name.upper()} Signal:")
        print(f"{'Horizon':<12} {'IC':<10} {'p-value':<12} {'Significant':<12}")
        print("-"*46)
        
        for horizon in horizons:
            ic, pval = calculate_ic(results_df[signal_name], results_df[horizon])
            
            if pd.notna(ic):
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"{horizon:<12} {ic:+.4f}    {pval:.4f}       {sig}")
            else:
                print(f"{horizon:<12} N/A        N/A")
    
    # Step 6: Long-Short Analysis
    print("\n" + "="*70)
    print("LONG-SHORT ANALYSIS (Q5 - Q1 by FCF Yield)")
    print("="*70)
    
    if len(results_df) >= 25:
        results_df['fcf_quintile'] = pd.qcut(
            results_df['fcf_yield'].rank(method='first'),
            q=5,
            labels=[1, 2, 3, 4, 5]
        )
        
        print(f"\n{'Horizon':<12} {'Q1 (Low)':<12} {'Q5 (High)':<12} {'L/S Spread':<12}")
        print("-"*48)
        
        for horizon in horizons:
            q1_ret = results_df[results_df['fcf_quintile'] == 1][horizon].mean()
            q5_ret = results_df[results_df['fcf_quintile'] == 5][horizon].mean()
            spread = q5_ret - q1_ret
            
            print(f"{horizon:<12} {q1_ret*100:+.2f}%       {q5_ret*100:+.2f}%       {spread*100:+.2f}%")
    else:
        print("Insufficient data for quintile analysis")
    
    # Step 7: Filing lag analysis
    print("\n" + "="*70)
    print("FILING LAG DISTRIBUTION")
    print("="*70)
    
    lag_stats = results_df['lag_days'].describe()
    print(f"Min: {lag_stats['min']:.0f} days")
    print(f"25%: {lag_stats['25%']:.0f} days")
    print(f"50%: {lag_stats['50%']:.0f} days (median)")
    print(f"75%: {lag_stats['75%']:.0f} days")
    print(f"Max: {lag_stats['max']:.0f} days")
    
    # Sample events
    print("\n" + "="*70)
    print("SAMPLE EVENTS")
    print("="*70)
    
    sample = results_df.head(10)[['ticker', 'fiscal_year', 'fiscal_period', 'filing_date', 'lag_days', 'fcf_yield', 'ret20d']].copy()
    sample['filing_date'] = pd.to_datetime(sample['filing_date']).dt.strftime('%Y-%m-%d')
    sample['fcf_yield'] = sample['fcf_yield'].apply(lambda x: f"{x*100:.1f}%")
    sample['ret20d'] = sample['ret20d'].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "N/A")
    print(sample.to_string(index=False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: BIASED vs FIXED")
    print("="*70)
    print("""
Original (BIASED) Results using DefeatBeta report_date:
  - FCF Yield IC: +0.10** (report_date = fiscal period END)
  - December fiscal year-end drove fake IC of +0.41***
  - Problem: Trading on data 30-60 days before it was public!

Fixed Results using Polygon filing_date:
  - Returns calculated AFTER actual SEC filing date
  - See IC table above for true predictive power
  - If IC is near zero → original signal was entirely fake
  - If IC is positive but smaller → some real signal exists
""")
    
    # Save results
    output_path = Path(__file__).parent / 'fcf_polygon_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
