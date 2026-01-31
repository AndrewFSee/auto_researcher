"""
Test whether FinBERT sentiment scores are comparable across different companies.
Key question: Does 0.8 for AAPL mean the same as 0.8 for JPM?
"""
import pandas as pd
import numpy as np
import warnings
import sys
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

# Load FinBERT
from src.auto_researcher.agents.finbert_sentiment import create_analyzer
analyzer = create_analyzer(prefer_finbert=True)

# Load transcripts
url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
df = pd.read_parquet(url)

def get_full_text(transcript_array):
    """Convert transcript array to full text."""
    if isinstance(transcript_array, np.ndarray):
        return " ".join([p.get('content', '') for p in transcript_array if p.get('content')])
    return str(transcript_array)

# Test diverse companies across sectors
test_companies = {
    'Tech': ['AAPL', 'MSFT', 'NVDA', 'GOOGL'],
    'Finance': ['JPM', 'GS', 'BAC', 'MS'],
    'Industrial': ['CAT', 'GE', 'HON', 'MMM'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK'],
}

print("=" * 70)
print("CROSS-COMPANY SENTIMENT SCORE COMPARISON")
print("=" * 70)

results = []

for sector, symbols in test_companies.items():
    print(f"\n📊 {sector} Sector:")
    for symbol in symbols:
        company_df = df[df['symbol'] == symbol].sort_values('report_date', ascending=False)
        if len(company_df) == 0:
            continue
        
        # Get last 8 quarters
        company_df = company_df.head(8)
        scores = []
        
        for _, row in company_df.iterrows():
            text = get_full_text(row['transcripts'])
            # Sample first 10k chars (intro + prepared remarks)
            text_sample = text[:10000]
            try:
                result = analyzer.analyze(text_sample)
                score = result.sentiment_score if hasattr(result, 'sentiment_score') else result.score
                scores.append(score)
            except:
                continue
        
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results.append({
                'sector': sector,
                'symbol': symbol,
                'mean': mean_score,
                'std': std_score,
                'n': len(scores)
            })
            print(f"   {symbol}: mean={mean_score:.3f}, std={std_score:.3f} (n={len(scores)})")

print("\n" + "=" * 70)
print("SECTOR-LEVEL SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
for sector in test_companies.keys():
    sector_data = results_df[results_df['sector'] == sector]
    if len(sector_data) > 0:
        sector_mean = sector_data['mean'].mean()
        sector_std = sector_data['mean'].std()  # Variation BETWEEN companies
        within_std = sector_data['std'].mean()  # Variation WITHIN companies
        print(f"\n{sector}:")
        print(f"   Cross-company mean: {sector_mean:.3f}")
        print(f"   Between-company std: {sector_std:.3f} (how much companies differ)")
        print(f"   Within-company std: {within_std:.3f} (how much one company varies)")

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

overall_mean = results_df['mean'].mean()
between_company_std = results_df['mean'].std()
within_company_std = results_df['std'].mean()

print(f"""
Overall mean sentiment: {overall_mean:.3f}
Between-company std:    {between_company_std:.3f}
Within-company std:     {within_company_std:.3f}

INTERPRETATION:
""")

if between_company_std > within_company_std:
    print("""
⚠️  BETWEEN > WITHIN: Companies have different "baseline" sentiment levels!
   → A pooled model may struggle
   → Consider: company fixed effects, or within-company normalization
   → Or: use sentiment CHANGE rather than absolute level
""")
else:
    print("""
✅ WITHIN > BETWEEN: Sentiment variation is mostly over TIME, not across companies
   → A pooled model should work well
   → Absolute sentiment scores are comparable across companies
""")

# Check if there's a sector effect
sector_means = results_df.groupby('sector')['mean'].mean()
print(f"\nSector average sentiments:")
for sector, mean in sector_means.items():
    print(f"   {sector}: {mean:.3f}")

sector_range = sector_means.max() - sector_means.min()
print(f"\nSector range: {sector_range:.3f}")
if sector_range > 0.1:
    print("   → Consider sector fixed effects in your model")
