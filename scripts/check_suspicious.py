"""
Scan for suspiciously low sentiment companies that might have word matching bugs.
"""
import pandas as pd
import numpy as np
import re
from collections import Counter

url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
print("Loading transcripts...")
df = pd.read_parquet(url)

def get_full_text(transcript_array):
    if isinstance(transcript_array, np.ndarray):
        return " ".join([p.get('content', '') for p in transcript_array if p.get('content')])
    return str(transcript_array)

# Current word lists
POSITIVE_WORDS = {
    'growth', 'grew', 'grow', 'growing',
    'increase', 'increased', 'increasing', 'increases',
    'strong', 'stronger', 'strongest', 'strength',
    'record', 'records',
    'exceeded', 'exceeding', 'exceeds',
    'beat', 'beats', 'beating',
    'outperformed', 'outperforming', 'outperformance',
    'momentum', 'accelerating', 'accelerated', 'acceleration',
    'robust', 'solid', 'healthy',
    'confident', 'confidence',
    'excited', 'exciting', 'excitement',
    'pleased',
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
    'miss', 'missed', 'missing', 'misses',
    'below',
    'underperformed', 'underperforming',
    'slowdown', 'slowing', 'slower',
    'deceleration', 'decelerating', 'decelerated',
    'soft', 'softer', 'softness', 'softening',
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

def analyze_sentiment(text, max_len=15000):
    """Return score and matched words for debugging."""
    text = text[:max_len].lower()
    words = re.findall(r'\b[a-z]+\b', text)
    
    pos_matches = [w for w in words if w in POSITIVE_WORDS]
    neg_matches = [w for w in words if w in NEGATIVE_WORDS]
    
    total = len(pos_matches) + len(neg_matches)
    if total == 0:
        return 0.5, pos_matches, neg_matches
    
    score = (len(pos_matches) - len(neg_matches)) / total
    return score, pos_matches, neg_matches

# Test a broad set of major companies
test_symbols = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL',
    'IBM', 'CSCO', 'ADBE', 'NOW', 'SNOW', 'PLTR', 'UBER', 'ABNB', 'SHOP', 'SQ',
    # Finance  
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V', 'MA',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
    # Industrial
    'CAT', 'DE', 'GE', 'HON', 'MMM', 'UNP', 'UPS', 'BA', 'LMT', 'RTX',
    # Consumer
    'WMT', 'COST', 'HD', 'MCD', 'SBUX', 'NKE', 'DIS', 'NFLX', 'PG', 'KO', 'PEP',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO', 'MPC', 'HAL',
]

print("\nAnalyzing sentiment for", len(test_symbols), "companies...")
print("=" * 80)

results = []

for symbol in test_symbols:
    company_df = df[df['symbol'] == symbol].sort_values('report_date', ascending=False).head(8)
    if len(company_df) == 0:
        continue
    
    all_scores = []
    all_neg_words = []
    
    for _, row in company_df.iterrows():
        text = get_full_text(row['transcripts'])
        score, pos, neg = analyze_sentiment(text)
        all_scores.append(score)
        all_neg_words.extend(neg)
    
    mean_score = np.mean(all_scores)
    neg_word_counts = Counter(all_neg_words)
    
    results.append({
        'symbol': symbol,
        'mean': mean_score,
        'std': np.std(all_scores),
        'min': np.min(all_scores),
        'n_calls': len(all_scores),
        'top_neg_words': neg_word_counts.most_common(5),
        'total_neg': sum(neg_word_counts.values()),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mean')

print("\n" + "=" * 80)
print("COMPANIES SORTED BY SENTIMENT (lowest first)")
print("=" * 80)

print("\n🔴 POTENTIALLY SUSPICIOUS (mean < 0.5 or min < 0):")
print("-" * 80)
suspicious = results_df[(results_df['mean'] < 0.5) | (results_df['min'] < 0)]
for _, row in suspicious.iterrows():
    print(f"\n{row['symbol']}: mean={row['mean']:+.3f}, min={row['min']:+.2f}")
    print(f"   Top negative words: {row['top_neg_words']}")

print("\n\n🟢 NORMAL RANGE (0.5 - 0.9):")
print("-" * 80)
normal = results_df[(results_df['mean'] >= 0.5) & (results_df['mean'] <= 0.9)]
for _, row in normal.head(10).iterrows():
    print(f"   {row['symbol']}: mean={row['mean']:+.3f}")
print(f"   ... and {len(normal) - 10} more companies")

print("\n\n🔵 VERY POSITIVE (mean > 0.9):")
print("-" * 80)
very_pos = results_df[results_df['mean'] > 0.9]
for _, row in very_pos.iterrows():
    print(f"   {row['symbol']}: mean={row['mean']:+.3f}")

# Check for potential word matching issues
print("\n\n" + "=" * 80)
print("CHECKING FOR POTENTIAL WORD BUGS")
print("=" * 80)

# Look for negative words that appear way too frequently
all_neg_combined = Counter()
for r in results:
    for word, count in r['top_neg_words']:
        all_neg_combined[word] += count

print("\nMost common negative words across all transcripts:")
for word, count in all_neg_combined.most_common(15):
    avg_per_company = count / len(results)
    flag = "⚠️ CHECK" if avg_per_company > 20 else ""
    print(f"   {word}: {count} total ({avg_per_company:.1f} per company) {flag}")
