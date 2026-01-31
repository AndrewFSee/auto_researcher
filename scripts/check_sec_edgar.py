"""
Check SEC EDGAR for actual filing dates vs period end dates.
This is critical for avoiding forward bias in DCF/valuation backtests.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import requests
from dotenv import load_dotenv
load_dotenv()

def get_sec_filings(cik: str, ticker: str):
    """Get 10-K and 10-Q filings for a company."""
    agent = os.getenv('SEC_API_USER_AGENT')
    if not agent:
        print("ERROR: SEC_API_USER_AGENT not set in .env")
        return
    
    # Pad CIK to 10 digits
    cik_padded = cik.zfill(10)
    url = f'https://data.sec.gov/submissions/CIK{cik_padded}.json'
    
    headers = {'User-Agent': agent}
    resp = requests.get(url, headers=headers)
    
    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} for {ticker}")
        return
    
    data = resp.json()
    filings = data.get('filings', {}).get('recent', {})
    
    forms = filings.get('form', [])
    filed_dates = filings.get('filingDate', [])
    period_ends = filings.get('reportDate', [])
    accession_nums = filings.get('accessionNumber', [])
    
    print(f"\n{'='*60}")
    print(f"{ticker} SEC FILINGS (CIK: {cik})")
    print(f"{'='*60}")
    print(f"{'Form':<8} {'Filed':<12} {'Period End':<12} {'Lag (days)':<10}")
    print('-'*42)
    
    count = 0
    lags = []
    for f, filed, period in zip(forms, filed_dates, period_ends):
        if f in ['10-K', '10-Q']:
            # Calculate lag
            from datetime import datetime
            filed_dt = datetime.strptime(filed, '%Y-%m-%d')
            period_dt = datetime.strptime(period, '%Y-%m-%d')
            lag = (filed_dt - period_dt).days
            lags.append((f, lag))
            
            print(f"{f:<8} {filed:<12} {period:<12} {lag:<10}")
            count += 1
            if count >= 12:
                break
    
    # Summarize lags
    print(f"\n--- Filing Lag Summary for {ticker} ---")
    annual_lags = [l for (f, l) in lags if f == '10-K']
    quarterly_lags = [l for (f, l) in lags if f == '10-Q']
    
    if annual_lags:
        print(f"10-K (Annual):    {min(annual_lags)}-{max(annual_lags)} days after period end")
    if quarterly_lags:
        print(f"10-Q (Quarterly): {min(quarterly_lags)}-{max(quarterly_lags)} days after period end")

def main():
    print("="*60)
    print("SEC EDGAR FILING DATE ANALYSIS")
    print("Purpose: Find actual filing dates for bias-free backtesting")
    print("="*60)
    
    # Test with major companies
    companies = [
        ('320193', 'AAPL'),   # Apple - Large accelerated filer
        ('789019', 'MSFT'),   # Microsoft
        ('1652044', 'GOOGL'), # Alphabet
        ('1018724', 'AMZN'),  # Amazon
        ('1326801', 'META'),  # Meta
    ]
    
    for cik, ticker in companies:
        get_sec_filings(cik, ticker)
    
    print("\n" + "="*60)
    print("IMPLICATIONS FOR BACKTESTING")
    print("="*60)
    print("""
SEC Filing Deadlines by Filer Category:
- Large Accelerated ($700M+ float): 10-K within 60 days, 10-Q within 40 days
- Accelerated ($75M-$700M float):   10-K within 75 days, 10-Q within 40 days  
- Non-Accelerated (<$75M float):    10-K within 90 days, 10-Q within 45 days

Most S&P 500 companies are Large Accelerated filers.

For bias-free backtesting:
- Use the ACTUAL filingDate from SEC EDGAR, not reportDate (period end)
- OR apply conservative lags: 60 days for 10-K, 45 days for 10-Q
""")

if __name__ == "__main__":
    main()
