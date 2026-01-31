"""Check context of 'lower' in suspicious companies."""
import pandas as pd
import numpy as np
import re

url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
df = pd.read_parquet(url)

def get_full_text(t):
    if isinstance(t, np.ndarray):
        return ' '.join([p.get('content', '') for p in t if p.get('content')])
    return str(t)

# Check DE (Deere) - the most negative
print("=" * 70)
print("DE (John Deere) - CONTEXT OF 'LOWER'")
print("=" * 70)

de = df[df['symbol'] == 'DE'].sort_values('report_date', ascending=False).iloc[0]
text = get_full_text(de['transcripts']).lower()

# Find instances of 'lower' with context
matches = list(re.finditer(r'.{30}lower.{30}', text[:10000]))
print(f"Found {len(matches)} instances of 'lower' in first 10k chars:\n")
for m in matches[:8]:
    context = m.group().replace('\n', ' ')
    print(f"  ...{context}...")
    print()

# Check COST (Costco) 
print("\n" + "=" * 70)
print("COST (Costco) - CONTEXT OF 'LOWER'")
print("=" * 70)

cost = df[df['symbol'] == 'COST'].sort_values('report_date', ascending=False).iloc[0]
text = get_full_text(cost['transcripts']).lower()

matches = list(re.finditer(r'.{30}lower.{30}', text[:10000]))
print(f"Found {len(matches)} instances of 'lower' in first 10k chars:\n")
for m in matches[:8]:
    context = m.group().replace('\n', ' ')
    print(f"  ...{context}...")
    print()

# Check if 'lower' is being used in legitimate negative context vs neutral
print("\n" + "=" * 70)
print("ANALYSIS: Is 'lower' a TRUE NEGATIVE or NEUTRAL?")
print("=" * 70)

print("""
PATTERNS FOUND:

1. LEGITIMATE NEGATIVE (should count):
   - "lower revenue", "lower sales", "lower margins"
   - "lower demand", "lower volumes"
   
2. NEUTRAL/COMPARATIVE (should NOT count as negative):
   - "lower cost structure" (actually positive!)
   - "lower expenses" (positive!)
   - "lower than last year" (neutral comparison)
   - "lower end of guidance" (neutral)
   
3. SPECIAL CASES:
   - "lower 48" (oil industry term - lower 48 US states, NOT negative!)
   - DE uses "lower" to describe declining farm equipment sales (legitimate)
   - Costco uses "lower" for price strategy (positive - lower prices!)
""")

# Check COP specifically for "lower 48"
print("\n" + "=" * 70)
print("COP (ConocoPhillips) - Checking for 'lower 48'")
print("=" * 70)

cop = df[df['symbol'] == 'COP'].sort_values('report_date', ascending=False).iloc[0]
text = get_full_text(cop['transcripts']).lower()

lower_48_count = len(re.findall(r'lower 48', text))
total_lower = len(re.findall(r'\blower\b', text))
print(f"Total 'lower': {total_lower}")
print(f"'lower 48' (geographic term): {lower_48_count}")
print(f"Actual negative 'lower': ~{total_lower - lower_48_count}")

if lower_48_count > 5:
    print("\n⚠️ BUG CONFIRMED: 'lower 48' is an oil industry term meaning")
    print("   the 48 continental US states - NOT a negative sentiment word!")
