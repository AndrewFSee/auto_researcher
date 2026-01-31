"""Check DefeatBeta stock_news dataset coverage."""

import pandas as pd

url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_news.parquet'
print('Loading stock_news from HuggingFace...')
df = pd.read_parquet(url)

print(f'Total articles: {len(df):,}')
print(f'Columns: {df.columns.tolist()}')
print()
print(f"Date range: {df['report_date'].min()} to {df['report_date'].max()}")
print()
print('Top publishers:')
print(df['publisher'].value_counts().head(10))
print()
print('Article types:')
print(df['type'].value_counts())
print()

# Check related symbols
symbols = df['related_symbols'].dropna()
print(f'Articles with stock symbols: {len(symbols):,}')

# Sample article
print()
print('Sample article:')
sample = df.iloc[0]
print(f"  Title: {sample['title']}")
print(f"  Publisher: {sample['publisher']}")
print(f"  Date: {sample['report_date']}")
print(f"  Symbols: {sample['related_symbols']}")
