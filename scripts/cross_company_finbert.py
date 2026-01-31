"""
Cross-company sentiment comparison using actual FinBERT.
This is the proper analysis, not the word-based approximation.
"""
import pandas as pd
import numpy as np
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Load FinBERT
print("Loading FinBERT...")
finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert')
print("FinBERT loaded!")

# Load transcripts
print("Loading transcripts...")
url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
df = pd.read_parquet(url)
print(f"Loaded {len(df):,} transcripts")

def get_full_text(transcript_array):
    if isinstance(transcript_array, np.ndarray):
        return " ".join([p.get('content', '') for p in transcript_array if p.get('content')])
    return str(transcript_array)

def analyze_with_finbert(text, max_len=10000):
    """
    Analyze text with FinBERT.
    FinBERT has 512 token limit, so we analyze first ~10k chars (intro + prepared remarks).
    Returns score from -1 (negative) to +1 (positive).
    """
    text_sample = text[:max_len]
    
    # FinBERT can handle up to 512 tokens, ~2000 chars is safe
    # Analyze multiple chunks and average
    chunk_size = 2000
    chunks = [text_sample[i:i+chunk_size] for i in range(0, len(text_sample), chunk_size)]
    
    scores = []
    for chunk in chunks[:5]:  # Max 5 chunks
        if len(chunk) < 50:  # Skip tiny chunks
            continue
        try:
            result = finbert(chunk[:512])[0]  # Truncate to 512 tokens
            label = result['label']
            score = result['score']
            
            # Convert to -1 to +1 scale
            if label == 'positive':
                scores.append(score)
            elif label == 'negative':
                scores.append(-score)
            else:  # neutral
                scores.append(0)
        except Exception as e:
            continue
    
    return np.mean(scores) if scores else 0

# Test diverse companies
test_companies = {
    'Tech': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN'],
    'Finance': ['JPM', 'GS', 'BAC', 'MS', 'WFC', 'C'],
    'Industrial': ['CAT', 'DE', 'GE', 'HON', 'UNP', 'UPS'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'LLY'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY'],
}

print("\n" + "=" * 70)
print("CROSS-COMPANY SENTIMENT ANALYSIS (FinBERT)")
print("=" * 70)

results = []

for sector, symbols in test_companies.items():
    print(f"\n📊 {sector}:")
    for symbol in symbols:
        company_df = df[df['symbol'] == symbol].sort_values('report_date', ascending=False).head(8)
        if len(company_df) == 0:
            print(f"   {symbol}: No data")
            continue
        
        scores = []
        for _, row in company_df.iterrows():
            text = get_full_text(row['transcripts'])
            score = analyze_with_finbert(text)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results.append({
            'sector': sector,
            'symbol': symbol,
            'mean': mean_score,
            'std': std_score,
            'min': np.min(scores),
            'max': np.max(scores),
        })
        print(f"   {symbol}: mean={mean_score:+.3f}, std={std_score:.3f}, range=[{np.min(scores):+.2f}, {np.max(scores):+.2f}]")

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("VARIANCE DECOMPOSITION (FinBERT)")
print("=" * 70)

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
print("INTERPRETATION")
print("=" * 70)

if between_pct > within_pct:
    print(f"""
⚠️  BETWEEN ({between_pct:.1f}%) > WITHIN ({within_pct:.1f}%)
    → Companies differ in baseline sentiment
    → Use DELTA features (sentiment_change) for cross-company modeling
    → Or use company fixed effects
""")
else:
    print(f"""
✅ WITHIN ({within_pct:.1f}%) > BETWEEN ({between_pct:.1f}%)
    → Most variation is over TIME, not across companies
    → Absolute sentiment scores are reasonably comparable
    → Pooled model should work well
""")

# Most extreme companies
print("Most positive companies:")
for _, row in results_df.nlargest(5, 'mean').iterrows():
    print(f"   {row['symbol']} ({row['sector']}): {row['mean']:+.3f}")

print("\nMost cautious companies:")
for _, row in results_df.nsmallest(5, 'mean').iterrows():
    print(f"   {row['symbol']} ({row['sector']}): {row['mean']:+.3f}")
