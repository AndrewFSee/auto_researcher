"""
Statistical Significance Tests for Earnings Model
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load backtest results
df = pd.read_parquet('data/backtest_results.parquet')
df = df.dropna(subset=['pred', 'ret_5d'])

# 1. HIT RATE SIGNIFICANCE
n = len(df)
hits = ((df['ret_5d'] > 0) == (df['pred'] > 0)).sum()
hit_rate = hits / n

# Binomial test: is hit rate significantly > 50%?
result = stats.binomtest(hits, n, 0.5, alternative='greater')
p_value_hit = result.pvalue
std_err = np.sqrt(0.5 * 0.5 / n)
z_score = (hit_rate - 0.5) / std_err

print('='*60)
print('STATISTICAL SIGNIFICANCE TESTS')
print('='*60)
print(f'\n1. HIT RATE TEST')
print(f'   Samples: {n:,}')
print(f'   Hit Rate: {hit_rate:.2%}')
print(f'   Standard Error: {std_err:.2%}')
print(f'   Z-score: {z_score:.2f}')
print(f'   P-value (one-sided): {p_value_hit:.4f}')
if p_value_hit < 0.05:
    print('   Significant? YES')
else:
    print('   Significant? NO (you are correct to be skeptical!)')

# 2. IC SIGNIFICANCE (more relevant for ranking)
ic = np.corrcoef(df['ret_5d'], df['pred'])[0,1]
# Fisher z-transform for correlation significance
z = 0.5 * np.log((1+ic)/(1-ic))
se_z = 1 / np.sqrt(n - 3)
p_value_ic = 2 * (1 - stats.norm.cdf(abs(z / se_z)))

print(f'\n2. INFORMATION COEFFICIENT TEST')
print(f'   IC: {ic:.4f}')
print(f'   P-value: {p_value_ic:.6f}')
if p_value_ic < 0.05:
    print('   Significant? YES')
else:
    print('   Significant? NO')

# 3. QUINTILE SPREAD T-TEST (most important!)
q1_rets = df[df['quintile'] == 1]['ret_5d']
q5_rets = df[df['quintile'] == 5]['ret_5d']

t_stat, p_value_spread = stats.ttest_ind(q5_rets, q1_rets)

print(f'\n3. LONG-SHORT SPREAD T-TEST (Q5 vs Q1)')
print(f'   Q5 mean: {q5_rets.mean():.4f} ({q5_rets.mean()*100:.2f}%)')
print(f'   Q1 mean: {q1_rets.mean():.4f} ({q1_rets.mean()*100:.2f}%)')
print(f'   Spread: {q5_rets.mean() - q1_rets.mean():.4f} ({(q5_rets.mean() - q1_rets.mean())*100:.2f}%)')
print(f'   T-statistic: {t_stat:.3f}')
print(f'   P-value (two-sided): {p_value_spread:.4f}')
if p_value_spread < 0.05:
    print('   Significant? YES')
else:
    print('   Significant? NO')

# 4. BOOTSTRAP TEST FOR SPREAD
print(f'\n4. BOOTSTRAP CONFIDENCE INTERVAL FOR SPREAD')
np.random.seed(42)
n_bootstrap = 10000
spreads = []
for _ in range(n_bootstrap):
    q5_sample = np.random.choice(q5_rets, size=len(q5_rets), replace=True)
    q1_sample = np.random.choice(q1_rets, size=len(q1_rets), replace=True)
    spreads.append(q5_sample.mean() - q1_sample.mean())

spreads = np.array(spreads)
ci_low, ci_high = np.percentile(spreads, [2.5, 97.5])
print(f'   Bootstrap spread mean: {np.mean(spreads):.4f}')
print(f'   95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
if ci_low > 0:
    print('   CI excludes zero? YES - SIGNIFICANT')
else:
    print('   CI excludes zero? NO - NOT SIGNIFICANT')

# 5. PERMUTATION TEST
print(f'\n5. PERMUTATION TEST (most rigorous)')
np.random.seed(42)
observed_spread = q5_rets.mean() - q1_rets.mean()
combined = np.concatenate([q5_rets.values, q1_rets.values])
n_q5 = len(q5_rets)
n_perms = 10000
perm_spreads = []
for _ in range(n_perms):
    np.random.shuffle(combined)
    perm_q5 = combined[:n_q5].mean()
    perm_q1 = combined[n_q5:].mean()
    perm_spreads.append(perm_q5 - perm_q1)

p_perm = np.mean(np.array(perm_spreads) >= observed_spread)
print(f'   Observed spread: {observed_spread:.4f}')
print(f'   Permutation p-value: {p_perm:.4f}')
if p_perm < 0.05:
    print('   Significant? YES')
else:
    print('   Significant? NO')

print('\n' + '='*60)
print('INTERPRETATION')
print('='*60)
print('''
Hit rate is NOT the right metric for this strategy.

Why? Because the model is trying to RANK, not predict direction.
- A stock going +5% and +0.1% are both "hits" but very different P&L
- A stock going +0.1% predicted as -0.1% is a "miss" but tiny error

The IC and quintile spread are the relevant metrics:
- IC measures ranking quality (correlation between predicted and actual)
- Spread measures the actual P&L difference between top and bottom buckets
''')
