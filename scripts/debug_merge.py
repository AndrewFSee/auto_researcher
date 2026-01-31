"""Debug the merge between revision and PEAD data"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_researcher.data.defeatbeta import DefeatBetaDataLoader
import pandas as pd

# Load revision data
rev = DefeatBetaDataLoader.load('stock_earning_estimates')
rev['date'] = pd.to_datetime(rev['report_date'])
print('REVISION DATA:')
print(f'  Columns: {list(rev.columns)}')
print(f'  Date range: {rev["date"].min()} to {rev["date"].max()}')
print(f'  Sample tickers: {list(rev["symbol"].unique()[:10])}')

# Load PEAD data
pead = DefeatBetaDataLoader.load('stock_historical_eps')
pead['earnings_date'] = pd.to_datetime(pead['quarter_date'], errors='coerce')
print('\nPEAD DATA:')
print(f'  Columns: {list(pead.columns)}')
print(f'  Date range: {pead["earnings_date"].min()} to {pead["earnings_date"].max()}')
print(f'  Sample tickers: {list(pead["symbol"].unique()[:10])}')

# Check overlap
rev_tickers = set(rev['symbol'].unique())
pead_tickers = set(pead['symbol'].unique())
overlap = rev_tickers & pead_tickers
print(f'\nTicker overlap: {len(overlap)} tickers')

# Check date overlap for a common ticker
if overlap:
    ticker = list(overlap)[0]
    rev_dates = rev[rev['symbol']==ticker]['date']
    pead_dates = pead[pead['symbol']==ticker]['earnings_date']
    print(f'\nFor {ticker}:')
    print(f'  Rev dates: {rev_dates.min()} to {rev_dates.max()}')
    print(f'  PEAD dates: {pead_dates.min()} to {pead_dates.max()}')
    print(f'  Rev sample:\n{rev[rev["symbol"]==ticker][["date"]].head(5)}')
    print(f'  PEAD sample:\n{pead[pead["symbol"]==ticker][["earnings_date"]].head(5)}')
