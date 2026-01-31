"""Compare yfinance vs DefeatBeta news sources."""

import pandas as pd
import yfinance as yf

ticker = "NVDA"

# yfinance
print("=== YFINANCE ===")
stock = yf.Ticker(ticker)
news = stock.news
print(f"Articles: {len(news)}")
for item in news[:3]:
    content = item.get('content', item)
    title = content.get('title', 'N/A')
    provider = content.get('provider', {})
    source = provider.get('displayName', 'Unknown') if isinstance(provider, dict) else 'Unknown'
    print(f"  [{source}] {title[:50]}...")

# DefeatBeta
print("\n=== DEFEATBETA ===")
url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_news.parquet'
df = pd.read_parquet(url)

# Filter for NVDA
nvda = df[df['related_symbols'].apply(lambda x: 'NVDA' in str(x) if x is not None else False)]
print(f"Articles: {len(nvda)}")
print(f"Date range: {nvda['report_date'].min()} to {nvda['report_date'].max()}")

# Sample recent articles
nvda_sorted = nvda.sort_values('report_date', ascending=False)
print("\nMost recent:")
for _, row in nvda_sorted.head(5).iterrows():
    print(f"  [{row['publisher']}] {row['title'][:50]}...")
    print(f"    Date: {row['report_date']}")

# Check if DefeatBeta has full article text
print("\n=== ARTICLE CONTENT ===")
sample = nvda_sorted.iloc[0]
news_content = sample['news']
if news_content is not None and len(news_content) > 0:
    print(f"Has full text: YES ({len(news_content)} paragraphs)")
    # Show first paragraph
    if isinstance(news_content, list) and len(news_content) > 0:
        first_para = news_content[0]
        if isinstance(first_para, dict):
            print(f"First paragraph: {first_para.get('paragraph', 'N/A')[:100]}...")
else:
    print("Has full text: NO (titles only)")
