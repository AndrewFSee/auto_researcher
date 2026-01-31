"""Audit estimate revisions backtest for forward bias."""
import pandas as pd
from scipy import stats

BASE_URL = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data'
df = pd.read_parquet(f'{BASE_URL}/stock_earning_estimates.parquet')

print('=== FORWARD BIAS AUDIT: ESTIMATE REVISIONS ===')
print()

# Check data structure
print('1. DATA STRUCTURE')
print(f'   Total rows: {len(df):,}')
print(f'   Unique tickers: {df["symbol"].nunique():,}')
print()

# Check rows per ticker-period
aapl = df[df['symbol']=='AAPL']
print(f'   AAPL total rows: {len(aapl)}')
aapl_unique = len(aapl.drop_duplicates(['symbol', 'report_date', 'period_type']))
print(f'   AAPL unique (ticker, period_end, type): {aapl_unique}')
print()

# If 1:1, this is a single snapshot, not historical time series
rows_per_period = df.groupby(['symbol', 'report_date', 'period_type']).size()
print('   Rows per ticker-period-type:')
print(rows_per_period.value_counts())
print()

# KEY QUESTION: When was this snapshot taken?
print('2. WHEN WAS THIS DATA CAPTURED?')
# The presence of future estimates suggests this is current/recent snapshot
future_dates = df[df['report_date'] > '2026-01-30']
print(f'   Rows with future report_date (after today): {len(future_dates):,}')
print(f'   Max report_date: {df["report_date"].max()}')
print()

# The data contains estimates for 2027 fiscal periods
# This means it was captured RECENTLY and contains forward-looking data
print('3. CRITICAL ISSUE')
print('   This dataset contains CURRENT estimates for FUTURE periods')
print('   The "X_days_ago" revisions show how estimates changed recently')
print('   This is NOT a historical time series of when estimates were made!')
print()

# What the backtest does:
print('4. WHAT THE BACKTEST DID:')
print('   - Matched estimate data by report_date to price data')
print('   - report_date = fiscal period end (e.g., 2024-03-31)')
print('   - But the ESTIMATE REVISION data is from TODAY, not historical!')
print()

# This is a MASSIVE forward bias problem
print('5. FORWARD BIAS SEVERITY: CRITICAL')
print('   The revision data (7d_ago, 30d_ago, etc.) is from CURRENT snapshot')
print('   When backtesting on 2023 data, we used 2026 revision information!')
print('   This is completely invalid!')
print()

# The only way to properly backtest would be to have:
# - Daily/weekly snapshots of analyst estimates
# - Historical record of when each estimate was published
# But DefeatBeta only provides current state

print('6. CONCLUSION')
print('   The estimate revisions backtest has SEVERE forward bias')
print('   All reported ICs and spreads are INVALID')
print('   Would need historical estimate data (e.g., from IBES, FactSet)')
print('   to properly test this signal')
