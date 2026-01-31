"""Test integrated news fetching from all sources."""
import sys
sys.path.insert(0, r"C:\Users\Andrew\projects\auto_researcher")

from src.auto_researcher.agents.sentiment_agent import SentimentAgent, SentimentAgentConfig

# Create agent with all sources enabled
config = SentimentAgentConfig(
    use_defeatbeta=True,
    use_scraped_db=True,
    scraped_db_lookback_days=30,
    max_news_items=20,
)

agent = SentimentAgent(config=config, finbert_only=True)

# Test on NVDA (we have scraped data for this)
print("Fetching news for NVDA from all sources...")
news = agent.fetch_all_news("NVDA")

print(f"\nTotal news items: {len(news)}")
print("\nSources breakdown:")
from collections import Counter
sources = Counter(n.source for n in news)
for src, cnt in sources.most_common(10):
    print(f"  {src}: {cnt}")

print("\nLatest 10 articles:")
for n in news[:10]:
    date_str = n.published.strftime("%Y-%m-%d") if hasattr(n.published, 'strftime') else str(n.published)[:10]
    print(f"  [{date_str}] [{n.source}] {n.title[:60]}...")
