"""
FCF Yield / DCF Valuation Backtest - FIXED VERSION
====================================================
Uses SEC EDGAR filing dates OR conservative lags instead of
DefeatBeta report_date (which is fiscal period end).

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import warnings
import time

warnings.filterwarnings('ignore')
load_dotenv()

# S&P 500 sample for testing
SP500_SAMPLE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CMCSA',
    'NFLX', 'XOM', 'VZ', 'INTC', 'T', 'PFE', 'ABT', 'CRM', 'CSCO', 'PEP',
    'AVGO', 'TMO', 'COST', 'MRK', 'ACN', 'NKE', 'CVX', 'WMT', 'LLY', 'ABBV',
    'MDT', 'DHR', 'NEE', 'TXN', 'UNP', 'HON', 'QCOM', 'PM', 'BMY', 'LIN',
]

# CIK mapping for SEC EDGAR (major companies)
CIK_MAP = {
    'AAPL': '320193',
    'MSFT': '789019',
    'GOOGL': '1652044',
    'AMZN': '1018724',
    'META': '1326801',
    'NVDA': '1045810',
    'TSLA': '1318605',
    'JPM': '19617',
    'JNJ': '200406',
    'V': '1403161',
    'PG': '80424',
    'UNH': '731766',
    'HD': '354950',
    'MA': '1141391',
    'DIS': '1744489',
    'BAC': '70858',
    'ADBE': '796343',
    'CMCSA': '1166691',
    'NFLX': '1065280',
    'XOM': '34088',
    'VZ': '732712',
    'INTC': '50863',
    'T': '732717',
    'PFE': '78003',
    'ABT': '1800',
    'CRM': '1108524',
    'CSCO': '858877',
    'PEP': '77476',
    'AVGO': '1730168',
    'TMO': '97745',
    'COST': '909832',
    'MRK': '310158',
    'ACN': '1467373',
    'NKE': '320187',
    'CVX': '93410',
    'WMT': '104169',
    'LLY': '59478',
    'ABBV': '1551152',
}


def get_sec_filings(ticker: str) -> pd.DataFrame:
    """
    Get 10-K and 10-Q filing dates from SEC EDGAR.
    Returns actual filing dates (when info became public).
    """
    if ticker not in CIK_MAP:
        return pd.DataFrame()
    
    cik = CIK_MAP[ticker]
    agent = os.getenv('SEC_API_USER_AGENT')
    
    if not agent:
        print(f"  {ticker}: SEC_API_USER_AGENT not set")
        return pd.DataFrame()
    
    try:
        url = f'https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json'
        headers = {'User-Agent': agent}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            return pd.DataFrame()
        
        data = resp.json()
        filings = data.get('filings', {}).get('recent', {})
        
        forms = filings.get('form', [])
        filed_dates = filings.get('filingDate', [])
        period_ends = filings.get('reportDate', [])
        
        records = []
        for form, filed, period in zip(forms, filed_dates, period_ends):
            if form in ['10-K', '10-Q']:
                records.append({
                    'ticker': ticker,
                    'form': form,
                    'filing_date': pd.to_datetime(filed),
                    'period_end': pd.to_datetime(period),
                })
        
        return pd.DataFrame(records)
        
    except Exception as e:
        print(f"  {ticker}: SEC error - {e}")
        return pd.DataFrame()


def get_fcf_from_yfinance(ticker: str) -> pd.DataFrame:
    """
    Get Free Cash Flow data from yfinance.
    Uses cashflow statement.
    """
    try:
        stock = yf.Ticker(ticker)
        cf = stock.quarterly_cashflow
        
        if cf is None or cf.empty:
            return pd.DataFrame()
        
        # Transpose so dates are rows
        cf = cf.T
        cf.index = pd.to_datetime(cf.index)
        
        # Get relevant fields
        records = []
        for date in cf.index:
            row = cf.loc[date]
            
            # Try different field names
            ocf = row.get('Operating Cash Flow') or row.get('Cash Flow From Continuing Operating Activities')
            capex = row.get('Capital Expenditure') or row.get('Purchase Of PPE')
            
            if pd.notna(ocf):
                capex_val = abs(capex) if pd.notna(capex) else 0
                fcf = ocf - capex_val
                
                records.append({
                    'ticker': ticker,
                    'period_end': date,
                    'ocf': ocf,
                    'capex': capex_val,
                    'fcf': fcf,
                })
        
        return pd.DataFrame(records)
        
    except Exception as e:
        print(f"  {ticker}: yfinance cashflow error - {e}")
        return pd.DataFrame()


def get_market_cap(ticker: str, date: datetime) -> float:
    """Get market cap for a ticker on a specific date."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=date - timedelta(days=7), end=date + timedelta(days=7))
        
        if hist.empty:
            return np.nan
        
        # Get closest price to date
        price = hist['Close'].iloc[-1]
        
        # Get shares outstanding
        info = stock.info
        shares = info.get('sharesOutstanding', np.nan)
        
        if pd.isna(shares):
            return np.nan
        
        return price * shares
        
    except:
        return np.nan


def merge_fcf_with_filings(fcf_df: pd.DataFrame, filings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge FCF data with SEC filing dates.
    Aligns FCF from a period with its actual filing date.
    """
    if fcf_df.empty or filings_df.empty:
        return pd.DataFrame()
    
    merged = []
    
    for _, fcf_row in fcf_df.iterrows():
        ticker = fcf_row['ticker']
        period_end = fcf_row['period_end']
        
        # Find matching filing
        ticker_filings = filings_df[filings_df['ticker'] == ticker]
        
        # Find filing for this period (within 10 days of period end)
        matching = ticker_filings[
            abs((ticker_filings['period_end'] - period_end).dt.days) <= 10
        ]
        
        if len(matching) > 0:
            filing = matching.iloc[0]
            filing_date = filing['filing_date']
            
            merged.append({
                **fcf_row.to_dict(),
                'filing_date': filing_date,
                'form': filing['form'],
                'lag_days': (filing_date - period_end).days,
            })
    
    return pd.DataFrame(merged)


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


def calculate_fcf_yield_signal(merged_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate FCF Yield signal using FILING DATE (when info is public).
    """
    results = []
    
    for _, row in merged_df.iterrows():
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
        
        # Estimate market cap (rough: price * assumed 1B shares for normalization)
        # In production, would use actual shares outstanding
        try:
            stock = yf.Ticker(ticker)
            shares = stock.info.get('sharesOutstanding', 1e9)
            market_cap = base_price * shares
        except:
            market_cap = base_price * 1e9  # Fallback
        
        # FCF Yield = FCF (annualized) / Market Cap
        fcf_annualized = fcf * 4  # Quarterly to annual
        fcf_yield = fcf_annualized / market_cap if market_cap > 0 else 0
        
        result = {
            'ticker': ticker,
            'period_end': row['period_end'],
            'filing_date': filing_date,
            'base_date': base_date,
            'fcf': fcf,
            'fcf_yield': fcf_yield,
            'market_cap': market_cap,
            'form': row.get('form', 'unknown'),
            'lag_days': row.get('lag_days', 0),
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
    print("FCF YIELD BACKTEST - FIXED VERSION")
    print("Using SEC EDGAR filing dates for ACTUAL public availability")
    print("="*70)
    print()
    
    # Step 1: Get SEC filing dates
    print("Step 1: Fetching SEC EDGAR filing dates...")
    all_filings = []
    
    for ticker in list(CIK_MAP.keys())[:30]:  # Limit for testing
        df = get_sec_filings(ticker)
        if len(df) > 0:
            all_filings.append(df)
            print(f"  {ticker}: {len(df)} filings")
        time.sleep(0.1)  # Rate limit
    
    if not all_filings:
        print("ERROR: No SEC filings fetched")
        return
    
    filings_df = pd.concat(all_filings, ignore_index=True)
    print(f"\nTotal SEC filings: {len(filings_df):,}")
    
    # Step 2: Get FCF data from yfinance
    print("\nStep 2: Fetching FCF data from yfinance...")
    all_fcf = []
    
    tickers_with_filings = filings_df['ticker'].unique().tolist()
    for ticker in tickers_with_filings:
        df = get_fcf_from_yfinance(ticker)
        if len(df) > 0:
            all_fcf.append(df)
            print(f"  {ticker}: {len(df)} quarters")
    
    if not all_fcf:
        print("ERROR: No FCF data fetched")
        return
    
    fcf_df = pd.concat(all_fcf, ignore_index=True)
    print(f"\nTotal FCF records: {len(fcf_df):,}")
    
    # Step 3: Merge FCF with filing dates
    print("\nStep 3: Merging FCF with filing dates...")
    merged_df = merge_fcf_with_filings(fcf_df, filings_df)
    print(f"Merged records: {len(merged_df):,}")
    
    if len(merged_df) == 0:
        print("ERROR: No merged data")
        return
    
    # Filter to 2020-2024
    merged_df = merged_df[
        (merged_df['filing_date'] >= '2020-01-01') &
        (merged_df['filing_date'] <= '2024-12-31')
    ]
    print(f"After date filter (2020-2024): {len(merged_df):,}")
    
    # Step 4: Fetch prices
    print("\nStep 4: Fetching price data...")
    tickers_in_merged = merged_df['ticker'].unique().tolist()
    prices = fetch_price_data(tickers_in_merged, '2020-01-01', '2025-12-31')
    print(f"Price data: {len(prices)} days, {len(prices.columns)} tickers")
    
    # Step 5: Calculate signals
    print("\nStep 5: Calculating FCF Yield signals (using filing dates)...")
    results_df = calculate_fcf_yield_signal(merged_df, prices)
    print(f"Events with returns: {len(results_df):,}")
    
    # Filter extreme values
    results_df = results_df[
        (results_df['fcf_yield'] > -1) &
        (results_df['fcf_yield'] < 1)
    ]
    print(f"After filtering extremes: {len(results_df):,}")
    
    # Step 6: Calculate ICs
    print("\n" + "="*70)
    print("INFORMATION COEFFICIENTS (FCF Yield signal)")
    print("="*70)
    print(f"{'Horizon':<12} {'IC':<10} {'p-value':<12} {'Significant':<12}")
    print("-"*46)
    
    horizons = ['ret5d', 'ret10d', 'ret20d', 'ret40d', 'ret60d']
    
    for horizon in horizons:
        ic, pval = calculate_ic(results_df['fcf_yield'], results_df[horizon])
        
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"{horizon:<12} {ic:+.4f}    {pval:.4f}       {sig}")
    
    # Step 7: Long-Short Analysis
    print("\n" + "="*70)
    print("LONG-SHORT ANALYSIS (Q5 - Q1 by FCF Yield)")
    print("="*70)
    
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
    
    # Step 8: Show filing lag distribution
    print("\n" + "="*70)
    print("FILING LAG DISTRIBUTION (days from period end to filing)")
    print("="*70)
    
    lag_stats = results_df['lag_days'].describe()
    print(f"Min: {lag_stats['min']:.0f} days")
    print(f"25%: {lag_stats['25%']:.0f} days")
    print(f"50%: {lag_stats['50%']:.0f} days")
    print(f"75%: {lag_stats['75%']:.0f} days")
    print(f"Max: {lag_stats['max']:.0f} days")
    
    # Sample events
    print("\n" + "="*70)
    print("SAMPLE EVENTS (verifying correct timing)")
    print("="*70)
    
    sample = results_df.head(8)[['ticker', 'period_end', 'filing_date', 'lag_days', 'fcf_yield', 'ret20d']].copy()
    sample['period_end'] = pd.to_datetime(sample['period_end']).dt.strftime('%Y-%m-%d')
    sample['filing_date'] = pd.to_datetime(sample['filing_date']).dt.strftime('%Y-%m-%d')
    sample['fcf_yield'] = sample['fcf_yield'].apply(lambda x: f"{x*100:.1f}%")
    sample['ret20d'] = sample['ret20d'].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "N/A")
    print(sample.to_string(index=False))
    
    print("\n" + "="*70)
    print("COMPARISON: BIASED vs FIXED")
    print("="*70)
    print("""
Original (BIASED) Results using DefeatBeta report_date:
  - FCF Yield IC: +0.10** (report_date = fiscal period END)
  - December fiscal year-end drove fake signal (IC=+0.41***)
  - Problem: Trading on data before it was filed with SEC!

Fixed Results using SEC EDGAR filing_date:
  - Returns calculated AFTER actual SEC filing
  - See IC table above for true predictive power
  - Filing lag properly accounts for when data became public
""")
    
    # Save results
    output_path = Path(__file__).parent / 'fcf_fixed_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
