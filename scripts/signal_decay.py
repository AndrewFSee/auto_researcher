"""Analyze how the signal decays across different holding periods."""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('data/backtest_results.parquet')

print('='*60)
print('SIGNAL DECAY ANALYSIS')
print('='*60)

results = []

for ret_col in ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d']:
    subset = df.dropna(subset=[ret_col, 'pred'])
    subset['quintile'] = pd.qcut(subset['pred'], 5, labels=['Q1','Q2','Q3','Q4','Q5'])
    
    q_returns = subset.groupby('quintile')[ret_col].mean()
    spread = q_returns['Q5'] - q_returns['Q1']
    
    # T-test
    q5 = subset[subset['quintile']=='Q5'][ret_col]
    q1 = subset[subset['quintile']=='Q1'][ret_col]
    t_stat, p_val = stats.ttest_ind(q5, q1)
    
    # IC
    ic = subset['pred'].corr(subset[ret_col])
    
    days = ret_col.split('_')[1]
    print(f'\n{days.upper()} RETURNS:')
    print(f'  Q1: {q_returns["Q1"]*100:+.2f}%  Q5: {q_returns["Q5"]*100:+.2f}%')
    print(f'  Spread: {spread*100:.2f}%  (p={p_val:.4f})')
    print(f'  IC: {ic:.4f}')
    print(f'  Significant: {"YES" if p_val < 0.05 else "NO"}')
    
    results.append({
        'horizon': days,
        'spread': spread,
        'p_value': p_val,
        'ic': ic
    })

print('\n' + '='*60)
print('SUMMARY')
print('='*60)

print('\nHorizon | Spread | P-value | IC')
print('-'*40)
for r in results:
    sig = '*' if r['p_value'] < 0.05 else ''
    print(f"  {r['horizon']:>3}   | {r['spread']*100:5.2f}% |  {r['p_value']:.4f} | {r['ic']:.4f} {sig}")

# Daily alpha calculation
print('\n' + '='*60)
print('DAILY ALPHA (spread / days)')
print('='*60)
for r in results:
    days = int(r['horizon'].replace('d',''))
    daily = r['spread'] / days
    annual = daily * 252
    print(f"  {r['horizon']:>3}: {daily*100:.3f}% per day = {annual*100:.1f}% annualized")
