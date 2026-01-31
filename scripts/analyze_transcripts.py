"""Analyze earnings call transcripts for predictive modeling feasibility."""
import pandas as pd
import numpy as np

# Load dataset
url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
df = pd.read_parquet(url)

print("=" * 60)
print("EARNINGS CALL DATASET ANALYSIS FOR PREDICTIVE MODELING")
print("=" * 60)

# Basic stats
print(f"\n📊 DATASET SIZE:")
print(f"   Total transcripts: {len(df):,}")
print(f"   Unique companies: {df['symbol'].nunique():,}")
print(f"   Date range: {df['report_date'].min()} to {df['report_date'].max()}")
print(f"   ~{len(df) // df['symbol'].nunique()} calls per company on average")

# The transcripts are structured as arrays of dicts with speaker/content
def get_full_text(transcript_array):
    """Convert transcript array to full text."""
    if isinstance(transcript_array, np.ndarray):
        return " ".join([p.get('content', '') for p in transcript_array if p.get('content')])
    return str(transcript_array)

# Sample to check transcript structure
sample_df = df.sample(n=min(100, len(df)), random_state=42)
sample_df['full_text'] = sample_df['transcripts'].apply(get_full_text)
sample_df['text_len'] = sample_df['full_text'].str.len()

print(f"\n📝 TRANSCRIPT LENGTHS (sampled):")
print(f"   Mean: {sample_df['text_len'].mean():,.0f} chars (~{sample_df['text_len'].mean()/5:.0f} words)")
print(f"   Median: {sample_df['text_len'].median():,.0f} chars")
print(f"   Min: {sample_df['text_len'].min():,} | Max: {sample_df['text_len'].max():,}")

# Calls per company distribution
cps = df.groupby('symbol').size()
companies_with_20_plus = (cps >= 20).sum()
companies_with_40_plus = (cps >= 40).sum()

print(f"\n📈 COMPANIES BY CALL COUNT:")
print(f"   With 20+ calls (5+ years): {companies_with_20_plus:,} ({100*companies_with_20_plus/len(cps):.1f}%)")
print(f"   With 40+ calls (10+ years): {companies_with_40_plus:,} ({100*companies_with_40_plus/len(cps):.1f}%)")

# Check speaker patterns
def get_speakers(transcript_array):
    """Extract unique speakers from transcript."""
    if isinstance(transcript_array, np.ndarray):
        return [p.get('speaker', '').lower() for p in transcript_array if p.get('speaker')]
    return []

sample_speakers = sample_df['transcripts'].apply(get_speakers)
all_speakers = [s for speakers in sample_speakers for s in speakers]

print(f"\n🎤 SPEAKER ANALYSIS (sample of {len(sample_df)} calls):")
print(f"   Total speaker turns: {len(all_speakers):,}")

# Check for CFO/CEO mentions
cfo_count = sum(1 for s in all_speakers if 'cfo' in s or 'chief financial' in s)
ceo_count = sum(1 for s in all_speakers if 'ceo' in s or 'chief executive' in s)
analyst_count = sum(1 for s in all_speakers if 'analyst' in s or any(firm in s for firm in ['morgan', 'goldman', 'jpmorgan', 'bofa', 'citi']))

print(f"   CFO speaker turns: {cfo_count}")
print(f"   CEO speaker turns: {ceo_count}")
print(f"   Analyst mentions: {analyst_count}")

print("\n" + "=" * 60)
print("KEY FINDING: STRUCTURED DATA!")
print("=" * 60)
print("""
The transcripts are STRUCTURED as paragraph arrays with:
  - speaker: Who is talking (CEO, CFO, Analyst, etc.)
  - content: What they said
  - paragraph_number: Order in call

This is BETTER than raw text because we can:
  ✅ Isolate CFO remarks (most standardized language)
  ✅ Separate prepared remarks from Q&A
  ✅ Analyze analyst questions separately
  ✅ Track sentiment by speaker role, not individual
""")


print("\n" + "=" * 60)
print("MODELING CONSIDERATIONS")
print("=" * 60)

print("""
✅ ADVANTAGES FOR PREDICTIVE MODELING:

1. LARGE SAMPLE SIZE
   - 186k transcripts is substantial for training
   - 2,382 companies have 40+ calls (10+ years of data)
   - Enough for both cross-sectional and panel models

2. STRUCTURED FORMAT
   - All transcripts follow similar structure (prepared remarks → Q&A)
   - CFO sections are particularly standardized (financial metrics)
   - Analyst questions provide external validation

3. FinBERT NORMALIZATION
   - FinBERT trained on financial text normalizes language differences
   - Converts heterogeneous CEO speak → standardized sentiment scores
   - Score of 0.85 from Tim Cook ≈ 0.85 from Jensen Huang

⚠️  CHALLENGES & MITIGATIONS:

1. LANGUAGE HETEROGENEITY (Your concern)
   Problem: CEOs have different communication styles
   
   Mitigations:
   a) Use DELTA features (change from prior quarter) - "Did Tim Cook 
      get MORE positive?" matters more than absolute level
   b) Within-company normalization (z-score sentiment by company)
   c) Focus on CFO sections (more standardized financial language)
   d) Use entity-neutral features like FinBERT sentiment scores

2. CEO/CFO TURNOVER
   Problem: Leadership changes cause style shifts
   
   Mitigations:
   a) Include executive tenure as control variable
   b) Use panel models with company fixed effects
   c) Detect style breaks in time series

3. SECTOR DIFFERENCES  
   Problem: Tech vs Banks vs Industrials use different language
   
   Mitigations:
   a) Include sector fixed effects
   b) Train sector-specific models
   c) Use features like "guidance raised/lowered" that are universal

💡 RECOMMENDED MODELING APPROACH:

1. PRIMARY FEATURES (robust across companies):
   - FinBERT sentiment score (normalized)
   - Quarter-over-quarter sentiment CHANGE
   - Guidance direction (raised/maintained/lowered)
   - Management confidence level
   - Q&A vs prepared remarks sentiment gap

2. CONTROL VARIABLES:
   - Sector/Industry
   - Company size (market cap)
   - Earnings surprise (beat/miss)
   - Analyst coverage

3. TARGET VARIABLE:
   - 1-day post-earnings return (immediate reaction)
   - 5-day drift (delayed reaction)
   - 30-day alpha (medium-term signal)

4. MODEL ARCHITECTURE:
   - Panel regression with company and time fixed effects
   - Or: XGBoost with proper cross-validation (no leakage)
   - Or: LSTM on sentiment time series per company
""")
