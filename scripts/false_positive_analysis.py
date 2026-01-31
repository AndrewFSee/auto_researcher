"""Deep dive into false positive patterns in sentiment analysis."""
import pandas as pd
import numpy as np
import re

url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
df = pd.read_parquet(url)

def get_full_text(t):
    if isinstance(t, np.ndarray):
        return ' '.join([p.get('content', '') for p in t if p.get('content')])
    return str(t)

# Check for known false positive patterns
FALSE_POSITIVE_PATTERNS = {
    'lower 48': 'Geographic term (oil/gas industry)',
    'lower cost': 'Actually positive - reducing costs',
    'lower costs': 'Actually positive - reducing costs', 
    'lower expense': 'Actually positive',
    'lower expenses': 'Actually positive',
    'lower price': 'Neutral/positive for retailers',
    'lower prices': 'Neutral/positive for retailers',
    'lower interest': 'Could be positive (lower interest expense)',
    'lower tax': 'Actually positive',
    'decline rate': 'Oil industry term - NOT decline in business',
    'declining balance': 'Accounting term',
    'pressure pumping': 'Oil industry service - NOT financial pressure',
    'pressure testing': 'Engineering term',
}

suspicious_companies = ['DE', 'COST', 'COP', 'PSX', 'UNP', 'CAT', 'UPS', 'CVX', 'VLO', 'HAL', 'XOM', 'SLB', 'EOG', 'OXY', 'MPC']

print("=" * 80)
print("FALSE POSITIVE ANALYSIS BY COMPANY")
print("=" * 80)

for symbol in suspicious_companies:
    company_df = df[df['symbol'] == symbol].sort_values('report_date', ascending=False).head(4)
    if len(company_df) == 0:
        continue
    
    print(f"\n{symbol}:")
    
    total_false_positives = 0
    total_lower = 0
    total_decline = 0
    total_pressure = 0
    
    pattern_counts = {}
    
    for _, row in company_df.iterrows():
        text = get_full_text(row['transcripts']).lower()
        
        # Count false positives
        for pattern, meaning in FALSE_POSITIVE_PATTERNS.items():
            count = len(re.findall(pattern, text))
            if count > 0:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + count
                total_false_positives += count
        
        total_lower += len(re.findall(r'\blower\b', text))
        total_decline += len(re.findall(r'\bdeclin', text))
        total_pressure += len(re.findall(r'\bpressure', text))
    
    if pattern_counts:
        print(f"   FALSE POSITIVES DETECTED:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"      '{pattern}': {count} occurrences - {FALSE_POSITIVE_PATTERNS[pattern]}")
    
    print(f"   Totals: lower={total_lower}, decline={total_decline}, pressure={total_pressure}")
    print(f"   Estimated false positives: ~{total_false_positives}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("""
WORDS TO REMOVE FROM NEGATIVE LIST:
1. 'lower' - Too ambiguous ("lower costs" is positive, "lower 48" is geographic)
2. 'pressure' - Too ambiguous ("pressure pumping" is oil service)

BETTER APPROACH - Use bigram/context matching:
Instead of: 'lower' (negative)
Use: 'lower revenue', 'lower sales', 'lower margin' (specific negatives)

Instead of: 'decline' (negative)
Check it's not: 'decline rate' (oil term), 'declining balance' (accounting)
""")

# Re-run sentiment WITHOUT the problematic words
print("\n" + "=" * 80)
print("FIXED SENTIMENT (removing 'lower' as standalone negative)")
print("=" * 80)

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

# FIXED: Remove ambiguous words
NEGATIVE_WORDS_FIXED = {
    'decline', 'declined', 'declining', 'declines',  # Keep but check context
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
    # REMOVED: 'pressure', 'pressures', 'pressured' - too ambiguous
    # REMOVED: 'lower', 'lowered', 'lowering' - too ambiguous
    'downgrade', 'downgrades',
    'deteriorating', 'deterioration',
    'contraction', 'contracting',
    'difficult', 'difficulty', 'difficulties',
    'disappointing', 'disappointed', 'disappointment',
    'uncertain', 'uncertainty', 'uncertainties',
}

def fixed_sentiment(text, max_len=15000):
    text = text[:max_len].lower()
    words = re.findall(r'\b[a-z]+\b', text)
    
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS_FIXED)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.5
    return (pos_count - neg_count) / total

print("\nRe-scored suspicious companies:")
for symbol in suspicious_companies:
    company_df = df[df['symbol'] == symbol].sort_values('report_date', ascending=False).head(8)
    if len(company_df) == 0:
        continue
    
    scores = []
    for _, row in company_df.iterrows():
        text = get_full_text(row['transcripts'])
        scores.append(fixed_sentiment(text))
    
    mean_score = np.mean(scores)
    print(f"   {symbol}: {mean_score:+.3f}")
