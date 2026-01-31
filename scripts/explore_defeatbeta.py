"""Explore DefeatBeta datasets for alpha opportunities."""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE_URL = "https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data"

datasets = {
    'stock_earning_estimates': 'Analyst EPS estimates (consensus)',
    'stock_revenue_estimates': 'Analyst revenue estimates', 
    'stock_historical_eps': 'Historical EPS data with surprises',
    'stock_earning_call_transcripts': 'Earnings call transcripts',
}

for name, description in datasets.items():
    print('='*70)
    print(f'{name}: {description}')
    print('='*70)
    try:
        df = pd.read_parquet(f"{BASE_URL}/{name}.parquet")
        print(f'Shape: {df.shape}')
        print(f'Columns: {list(df.columns)}')
        if 'symbol' in df.columns:
            print(f'Tickers: {df["symbol"].nunique():,}')
        print()
        print('Sample (first 2 rows):')
        print(df.head(2).to_string())
    except Exception as e:
        print(f'Error: {e}')
    print()

# Check a sample company
print("\n" + "="*60)
print("SAMPLE: AAPL Income Statement (quarterly)")
print("="*60)
aapl = df[(df['symbol']=='AAPL') & 
          (df['finance_type']=='income_statement') & 
          (df['period_type']=='quarterly')]
aapl = aapl.sort_values('report_date', ascending=False)

# Pivot to show items as columns
if len(aapl) > 0:
    pivot = aapl.pivot_table(index='report_date', columns='item_name', values='item_value', aggfunc='first')
    key_items = ['TotalRevenue', 'GrossProfit', 'OperatingIncome', 'NetIncome', 'BasicEPS']
    available = [i for i in key_items if i in pivot.columns]
    if available:
        print(pivot[available].head(8).to_string())
    else:
        print("Key items not found. Available columns:")
        print(pivot.columns.tolist()[:20])
