"""Investigate MSFT's low sentiment scores."""
import pandas as pd
import numpy as np
import re

url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
df = pd.read_parquet(url)

def get_full_text(transcript_array):
    if isinstance(transcript_array, np.ndarray):
        return ' '.join([p.get('content', '') for p in transcript_array if p.get('content')])
    return str(transcript_array)

# Get MSFT calls
msft = df[df['symbol'] == 'MSFT'].sort_values('report_date', ascending=False).head(12)

POSITIVE_WORDS = {'growth', 'grew', 'increase', 'increased', 'strong', 'strength', 'record', 
    'exceeded', 'beat', 'outperformed', 'momentum', 'accelerat', 'robust',
    'confident', 'excited', 'pleased', 'optimistic', 'opportunity', 'upside',
    'raised', 'raising', 'upgrade', 'improving', 'expansion', 'tailwind',
    'best', 'highest', 'exceptional', 'outstanding', 'remarkable', 'impressive'}

NEGATIVE_WORDS = {'decline', 'declined', 'decrease', 'decreased', 'weak', 'weakness', 'miss',
    'missed', 'below', 'underperformed', 'slowdown', 'decelerat', 'soft',
    'concerned', 'worried', 'cautious', 'challenging', 'headwind', 'pressure',
    'lowered', 'lowering', 'downgrade', 'deteriorat', 'contraction', 'difficult',
    'worst', 'lowest', 'disappointing', 'underwhelming', 'uncertain'}

print('MSFT Earnings Call Analysis:')
print('=' * 60)

for _, row in msft.iterrows():
    text = get_full_text(row['transcripts'])
    text_lower = text[:15000].lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    pos_matches = [w for w in words if any(p in w for p in POSITIVE_WORDS)]
    neg_matches = [w for w in words if any(n in w for n in NEGATIVE_WORDS)]
    
    total = len(pos_matches) + len(neg_matches)
    score = (len(pos_matches) - len(neg_matches)) / total if total > 0 else 0
    
    q = row['fiscal_quarter']
    y = row['fiscal_year']
    pos_count = len(pos_matches)
    neg_count = len(neg_matches)
    
    print(f"{q} {y}: score={score:+.2f} (positive={pos_count}, negative={neg_count})")
    print(f"   Top positive words: {dict(pd.Series(pos_matches).value_counts().head(5))}")
    print(f"   Top negative words: {dict(pd.Series(neg_matches).value_counts().head(5))}")
    print()

# Compare to AAPL
print("\n" + "=" * 60)
print("COMPARISON: AAPL Earnings Call Analysis:")
print("=" * 60)

aapl = df[df['symbol'] == 'AAPL'].sort_values('report_date', ascending=False).head(4)

for _, row in aapl.iterrows():
    text = get_full_text(row['transcripts'])
    text_lower = text[:15000].lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    pos_matches = [w for w in words if any(p in w for p in POSITIVE_WORDS)]
    neg_matches = [w for w in words if any(n in w for n in NEGATIVE_WORDS)]
    
    total = len(pos_matches) + len(neg_matches)
    score = (len(pos_matches) - len(neg_matches)) / total if total > 0 else 0
    
    q = row['fiscal_quarter']
    y = row['fiscal_year']
    pos_count = len(pos_matches)
    neg_count = len(neg_matches)
    
    print(f"{q} {y}: score={score:+.2f} (positive={pos_count}, negative={neg_count})")
    print(f"   Top positive words: {dict(pd.Series(pos_matches).value_counts().head(5))}")
    print(f"   Top negative words: {dict(pd.Series(neg_matches).value_counts().head(5))}")
    print()

# Show sample of actual MSFT text
print("\n" + "=" * 60)
print("SAMPLE MSFT TRANSCRIPT TEXT (first 2000 chars):")
print("=" * 60)
sample_msft = msft.iloc[0]
sample_text = get_full_text(sample_msft['transcripts'])
print(sample_text[:2000])
