"""
FIXED: Cross-company sentiment comparison with proper word matching.
Bug fix: "Microsoft" was matching "soft", "transmission" was matching "miss"
"""
import pandas as pd
import numpy as np
import re

url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
print("Loading transcripts...")
df = pd.read_parquet(url)

def get_full_text(transcript_array):
    if isinstance(transcript_array, np.ndarray):
        return " ".join([p.get('content', '') for p in transcript_array if p.get('content')])
    return str(transcript_array)

# FIXED: Use exact word matching, not substring matching
POSITIVE_WORDS = {
    'growth', 'grew', 'grow', 'growing',
    'increase', 'increased', 'increasing', 'increases',
    'strong', 'stronger', 'strongest', 'strength',
    'record', 'records',  # NOT 'recorded' - that's neutral
    'exceeded', 'exceeding', 'exceeds',
    'beat', 'beats', 'beating',
    'outperformed', 'outperforming', 'outperformance',
    'momentum', 'accelerating', 'accelerated', 'acceleration',
    'robust', 'solid', 'healthy',
    'confident', 'confidence',
    'excited', 'exciting', 'excitement',
    'pleased', 'pleased',
    'optimistic', 'optimism',
    'opportunity', 'opportunities',
    'upside',
    'raised', 'raising', 'raise',
    'improving', 'improved', 'improvement',
    'expansion', 'expanding', 'expanded',
    'tailwind', 'tailwinds',
    'exceptional', 'outstanding', 'remarkable', 'impressive',
}

NEGATIVE_WORDS = {
    'decline', 'declined', 'declining', 'declines',
    'decrease', 'decreased', 'decreasing',
    'weak', 'weaker', 'weakest', 'weakness', 'weakening',
    'miss', 'missed', 'missing', 'misses',  # Exact words only
    'below',
    'underperformed', 'underperforming',
    'slowdown', 'slowing', 'slower',
    'deceleration', 'decelerating', 'decelerated',
    'soft', 'softer', 'softness', 'softening',  # Exact words only - won't match "Microsoft"
    'concerned', 'concern', 'concerns',
    'worried', 'worry', 'worries',
    'cautious', 'caution',
    'challenging', 'challenge', 'challenges', 'challenged',
    'headwind', 'headwinds',
    'pressure', 'pressures', 'pressured',
    'lowered', 'lowering', 'lower',
    'downgrade', 'downgrades',
    'deteriorating', 'deterioration',
    'contraction', 'contracting',
    'difficult', 'difficulty', 'difficulties',
    'disappointing', 'disappointed', 'disappointment',
    'uncertain', 'uncertainty', 'uncertainties',
}

def simple_sentiment(text, max_len=15000):
    """Calculate sentiment with EXACT word matching."""
    text = text[:max_len].lower()
    # Split into words, removing punctuation
    words = re.findall(r'\b[a-z]+\b', text)
    
    # Exact word matching only
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos_count + neg_count
    
    if total == 0:
        return 0.5, pos_count, neg_count
    
    score = (pos_count - neg_count) / total
    return score, pos_count, neg_count

# Test on MSFT vs AAPL
print("\n" + "=" * 70)
print("FIXED SENTIMENT ANALYSIS (exact word matching)")
print("=" * 70)

for symbol in ['MSFT', 'AAPL', 'JPM', 'NVDA']:
    company_df = df[df['symbol'] == symbol].sort_values('report_date', ascending=False).head(8)
    scores = []
    
    print(f"\n{symbol}:")
    for _, row in company_df.iterrows():
        text = get_full_text(row['transcripts'])
        score, pos, neg = simple_sentiment(text)
        scores.append(score)
        q = row['fiscal_quarter']
        y = row['fiscal_year']
        print(f"   {q} {y}: score={score:+.2f} (pos={pos}, neg={neg})")
    
    print(f"   MEAN: {np.mean(scores):+.3f}, STD: {np.std(scores):.3f}")

# Now rerun the full comparison
print("\n" + "=" * 70)
print("FULL CROSS-COMPANY COMPARISON (FIXED)")
print("=" * 70)

test_companies = {
    'Tech': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN'],
    'Finance': ['JPM', 'GS', 'BAC', 'MS', 'WFC', 'C'],
}

results = []

for sector, symbols in test_companies.items():
    print(f"\n{sector}:")
    for symbol in symbols:
        company_df = df[df['symbol'] == symbol].sort_values('report_date', ascending=False).head(12)
        if len(company_df) == 0:
            continue
        
        scores = []
        for _, row in company_df.iterrows():
            text = get_full_text(row['transcripts'])
            score, _, _ = simple_sentiment(text)
            scores.append(score)
        
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results.append({'sector': sector, 'symbol': symbol, 'mean': mean_score, 'std': std_score})
            print(f"   {symbol}: mean={mean_score:+.3f}, std={std_score:.3f}")

results_df = pd.DataFrame(results)
print(f"\nOverall mean: {results_df['mean'].mean():+.3f}")
print(f"Between-company std: {results_df['mean'].std():.3f}")
print(f"Within-company std (avg): {results_df['std'].mean():.3f}")
