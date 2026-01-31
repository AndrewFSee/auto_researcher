"""
Test whether sentiment patterns are comparable across different companies.
Using simple word-based sentiment to avoid VADER/numpy issues.
"""
import pandas as pd
import numpy as np
import re

# Load transcripts
url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
print("Loading transcripts...")
df = pd.read_parquet(url)

def get_full_text(transcript_array):
    """Convert transcript array to full text."""
    if isinstance(transcript_array, np.ndarray):
        return " ".join([p.get('content', '') for p in transcript_array if p.get('content')])
    return str(transcript_array)

# Simple financial sentiment lexicon
POSITIVE_WORDS = {
    'growth', 'grew', 'increase', 'increased', 'strong', 'strength', 'record', 
    'exceeded', 'beat', 'outperformed', 'momentum', 'accelerat', 'robust',
    'confident', 'excited', 'pleased', 'optimistic', 'opportunity', 'upside',
    'raised', 'raising', 'upgrade', 'improving', 'expansion', 'tailwind',
    'best', 'highest', 'exceptional', 'outstanding', 'remarkable', 'impressive'
}

NEGATIVE_WORDS = {
    'decline', 'declined', 'decrease', 'decreased', 'weak', 'weakness', 'miss',
    'missed', 'below', 'underperformed', 'slowdown', 'decelerat', 'soft',
    'concerned', 'worried', 'cautious', 'challenging', 'headwind', 'pressure',
    'lowered', 'lowering', 'downgrade', 'deteriorat', 'contraction', 'difficult',
    'worst', 'lowest', 'disappointing', 'underwhelming', 'uncertain'
}

def simple_sentiment(text, max_len=15000):
    """Calculate simple positive/negative ratio."""
    text = text[:max_len].lower()
    words = re.findall(r'\b\w+\b', text)
    
    pos_count = sum(1 for w in words if any(p in w for p in POSITIVE_WORDS))
    neg_count = sum(1 for w in words if any(n in w for n in NEGATIVE_WORDS))
    total = pos_count + neg_count
    
    if total == 0:
        return 0.5
    
    # Score from -1 to +1
    score = (pos_count - neg_count) / total
    return score

# Test diverse companies across sectors
test_companies = {
    'Tech': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN'],
    'Finance': ['JPM', 'GS', 'BAC', 'MS', 'WFC', 'C'],
    'Industrial': ['CAT', 'GE', 'HON', 'MMM', 'DE', 'UNP'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'LLY'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY'],
}

print("\n" + "=" * 70)
print("CROSS-COMPANY SENTIMENT SCORE COMPARISON")
print("=" * 70)

results = []

for sector, symbols in test_companies.items():
    print(f"\n📊 {sector} Sector:")
    for symbol in symbols:
        company_df = df[df['symbol'] == symbol].sort_values('report_date', ascending=False)
        if len(company_df) == 0:
            continue
        
        # Get last 12 quarters (3 years)
        company_df = company_df.head(12)
        scores = []
        
        for _, row in company_df.iterrows():
            text = get_full_text(row['transcripts'])
            score = simple_sentiment(text)
            scores.append(score)
        
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            results.append({
                'sector': sector,
                'symbol': symbol,
                'mean': mean_score,
                'std': std_score,
                'min': min_score,
                'max': max_score,
                'n': len(scores)
            })
            print(f"   {symbol}: mean={mean_score:+.3f}, std={std_score:.3f}, range=[{min_score:+.2f}, {max_score:+.2f}]")

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("VARIANCE DECOMPOSITION")
print("=" * 70)

# Calculate between-company and within-company variance
overall_mean = results_df['mean'].mean()
between_company_var = results_df['mean'].var()
within_company_var = (results_df['std'] ** 2).mean()

total_var = between_company_var + within_company_var
between_pct = 100 * between_company_var / total_var if total_var > 0 else 0
within_pct = 100 * within_company_var / total_var if total_var > 0 else 0

print(f"""
Overall mean sentiment:     {overall_mean:+.3f}
Between-company variance:   {between_company_var:.4f} ({between_pct:.1f}% of total)
Within-company variance:    {within_company_var:.4f} ({within_pct:.1f}% of total)
""")

print("=" * 70)
print("SECTOR COMPARISON")
print("=" * 70)

sector_stats = results_df.groupby('sector').agg({
    'mean': ['mean', 'std'],
    'std': 'mean'
}).round(3)
sector_stats.columns = ['sector_mean', 'between_company_std', 'within_company_std']
print(sector_stats.to_string())

print("\n" + "=" * 70)
print("INTERPRETATION: CAN YOU POOL COMPANIES?")
print("=" * 70)

if between_pct > 50:
    answer = """
⚠️  NO - Companies differ significantly in baseline sentiment!

Between-company variance ({:.1f}%) > Within-company variance ({:.1f}%)

This means:
  • Company A's "0.5" is NOT the same as Company B's "0.5"
  • Some CEOs are perpetually more optimistic/pessimistic
  • A pooled model will confuse "optimistic CEO" with "good fundamentals"

SOLUTION: Use one of these approaches:

1. DELTA MODEL (Recommended):
   → Feature = sentiment_change = current_Q - prior_Q
   → This removes company-specific baseline
   → "Tim Cook got MORE positive" is comparable to "Jamie Dimon got MORE positive"

2. WITHIN-COMPANY Z-SCORE:
   → z_score = (sentiment - company_mean) / company_std
   → Normalizes each company to its own history

3. COMPANY FIXED EFFECTS:
   → Include company dummies in regression
   → Model learns each company's baseline

4. DEVIATION FROM COMPANY MEAN:
   → Feature = sentiment - company_rolling_mean(4Q)
   → Captures "unusual" tone for THIS company
""".format(between_pct, within_pct)
else:
    answer = """
✅ YES - Sentiment scores are relatively comparable across companies!

Within-company variance ({:.1f}%) > Between-company variance ({:.1f}%)

This means:
  • Most sentiment variation is over TIME, not across companies
  • A pooled model should work reasonably well
  • You can still improve with company fixed effects, but it's not critical
""".format(within_pct, between_pct)

print(answer)

# Show the most extreme companies
print("\n" + "=" * 70)
print("MOST EXTREME COMPANIES (baseline bias examples)")
print("=" * 70)

most_positive = results_df.nlargest(5, 'mean')[['symbol', 'sector', 'mean', 'std']]
most_negative = results_df.nsmallest(5, 'mean')[['symbol', 'sector', 'mean', 'std']]

print("\nMost consistently positive (optimistic CEOs?):")
print(most_positive.to_string(index=False))

print("\nMost consistently negative (cautious CEOs?):")
print(most_negative.to_string(index=False))

print("\n💡 Notice: The spread between most positive and most negative is {:.3f}".format(
    most_positive['mean'].mean() - most_negative['mean'].mean()
))
print("   This is the 'CEO personality effect' you're worried about!")
