"""
Analyze Fold 1 (2018-2020) period - understand where returns are lost
"""
import yfinance as yf
import pandas as pd
import numpy as np

# Download SPY to see market context
spy_df = yf.download("SPY", start="2018-01-01", end="2020-07-01", progress=False)
if 'Close' in spy_df.columns:
    spy = spy_df['Close']
elif isinstance(spy_df.columns, pd.MultiIndex):
    spy = spy_df['Close']['SPY']
else:
    spy = spy_df.iloc[:, 0]
if isinstance(spy, pd.DataFrame):
    spy = spy.iloc[:, 0]

spy_monthly = spy.resample('M').last().pct_change().dropna()

print("=" * 60)
print("FOLD 1 ANALYSIS: 2018-01-01 to 2020-06-30")
print("=" * 60)

print("\n📊 SPY Monthly Returns (Market Context):")
print("-" * 60)

# Format nicely by year
for year in [2018, 2019, 2020]:
    year_data = spy_monthly[spy_monthly.index.year == year]
    print(f"\n{year}:")
    for date, ret in year_data.items():
        marker = "🔴" if ret < -0.05 else ("🟢" if ret > 0.05 else "⚪")
        print(f"  {date.strftime('%Y-%m')}: {ret*100:+6.2f}% {marker}")
    print(f"  Year Total: {(1+year_data).prod()-1:.1%}")

print("\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)
print("""
1. ROLLING WINDOW PROBLEM:
   - Your 504-day rolling window means trading starts ~May 2019
   - You MISS the 2018 Q4 downturn entirely (training on it)
   - You MISS the 2019 bull run (Jan-Apr) entirely

2. WHAT YOUR STRATEGY ACTUALLY SEES:
   - Trading period: May 2019 - June 2020 (only 13 months!)
   - This includes COVID crash (Feb-Mar 2020)
   
3. TIMING:
   - 2018-Q4: Down 14% - NOT TRADED (in training window)
   - 2019: Up 29% - MOSTLY MISSED (only May-Dec traded)
   - 2020-Q1: COVID crash - FULLY HIT (-20%)
   - 2020-Q2: Recovery - Captured some (+20%)

4. THE REAL ISSUE:
   - You started trading RIGHT BEFORE the worst crash since 2008
   - Only 8 rebalances total (monthly) - tiny sample size
   - Last rebalance was March 2020 (worst timing ever)
""")

# Calculate what portion of 2019 bull run was captured
may_2019_idx = spy.index >= '2019-05-01'
jan_2019_idx = spy.index >= '2019-01-01'

spy_full_2019 = (spy[spy.index.year == 2019].iloc[-1] / spy[spy.index.year == 2019].iloc[0]) - 1
spy_may_dec_2019 = (spy[spy.index >= '2019-05-01'][spy.index < '2020-01-01'].iloc[-1] / 
                    spy[spy.index >= '2019-05-01'][spy.index < '2020-01-01'].iloc[0]) - 1

print(f"\n📈 2019 Bull Run Analysis:")
print(f"   Full 2019 SPY return:     {spy_full_2019:.1%}")
print(f"   May-Dec 2019 SPY return:  {spy_may_dec_2019:.1%}")
print(f"   You MISSED:               {spy_full_2019 - spy_may_dec_2019:.1%}")

# Show COVID impact
feb_2020 = spy['2020-02-19']  # Peak
mar_2020 = spy['2020-03-23']  # Bottom
covid_crash = (mar_2020 / feb_2020) - 1
print(f"\n💥 COVID Crash Impact:")
print(f"   Feb 19 to Mar 23, 2020:   {covid_crash:.1%}")
print(f"   Your last rebalance was March 31, 2020 - right after bottom")
