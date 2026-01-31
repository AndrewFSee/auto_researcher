"""Test regularization to prevent overfitting."""
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('data/news_combined_results.parquet')
print(f'Dataset: {len(df):,} rows')

df = df.sort_values('date')
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

feature_cols = [
    'sentiment_mean', 'sentiment_std', 'news_count',
    'positive_ratio', 'negative_ratio',
    'sentiment_ma5', 'sentiment_ma10', 'sentiment_ma20',
    'volume_zscore', 'sentiment_zscore',
    'event_signal', 'contrarian_signal', 'alpha_optimized'
]
available = [c for c in feature_cols if c in df.columns]

X_train = train[available].fillna(0)
y_train = train['ret10d']
X_test = test[available].fillna(0)
y_test = test['ret10d']

print(f'Train: {len(train):,} | Test: {len(test):,}')
print()

configs = [
    {'name': 'Current',       'n_est': 100, 'depth': 4, 'leaves': 15, 'min_child': 100, 'reg_alpha': 0, 'reg_lambda': 0},
    {'name': 'More L2',       'n_est': 100, 'depth': 4, 'leaves': 15, 'min_child': 100, 'reg_alpha': 0, 'reg_lambda': 10},
    {'name': 'Heavy L1+L2',   'n_est': 100, 'depth': 4, 'leaves': 15, 'min_child': 100, 'reg_alpha': 5, 'reg_lambda': 10},
    {'name': 'Shallower',     'n_est': 50,  'depth': 2, 'leaves': 4,  'min_child': 500, 'reg_alpha': 5, 'reg_lambda': 10},
    {'name': 'Single tree',   'n_est': 1,   'depth': 2, 'leaves': 4,  'min_child': 500, 'reg_alpha': 0, 'reg_lambda': 0},
    {'name': 'Linear boost',  'n_est': 50,  'depth': 1, 'leaves': 2,  'min_child': 1000, 'reg_alpha': 10, 'reg_lambda': 10},
]

print('='*75)
print('REGULARIZATION EXPERIMENT')
print('='*75)
print(f"{'Config':<15} {'Train IC':>10} {'Test IC':>10} {'Test p':>10} {'Overfit':>10}")
print('-'*75)

for cfg in configs:
    model = lgb.LGBMRegressor(
        n_estimators=cfg['n_est'],
        max_depth=cfg['depth'],
        num_leaves=cfg['leaves'],
        min_child_samples=cfg['min_child'],
        reg_alpha=cfg['reg_alpha'],
        reg_lambda=cfg['reg_lambda'],
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_ic, _ = stats.spearmanr(train_pred, y_train)
    test_ic, test_p = stats.spearmanr(test_pred, y_test)
    overfit = train_ic - test_ic
    
    sig = '***' if test_p < 0.001 else ('**' if test_p < 0.01 else ('*' if test_p < 0.05 else ''))
    print(f"{cfg['name']:<15} {train_ic:>+10.4f} {test_ic:>+10.4f}{sig:>3} {test_p:>10.4f} {overfit:>+10.4f}")

print('-'*75)
simple_ic, simple_p = stats.spearmanr(test['alpha_optimized'], y_test)
sig = '***' if simple_p < 0.001 else ('**' if simple_p < 0.01 else ('*' if simple_p < 0.05 else ''))
print(f"{'Simple signal':<15} {'--':>10} {simple_ic:>+10.4f}{sig:>3} {simple_p:>10.4f}")

# Also try: just use contrarian_signal alone (no ML)
contra_ic, contra_p = stats.spearmanr(test['contrarian_signal'], y_test)
sig = '***' if contra_p < 0.001 else ('**' if contra_p < 0.01 else ('*' if contra_p < 0.05 else ''))
print(f"{'Contrarian only':<15} {'--':>10} {contra_ic:>+10.4f}{sig:>3} {contra_p:>10.4f}")

print()
print('='*75)
print('DIAGNOSIS: Why does the model overfit?')
print('='*75)

# Check if the test period is different from train
train_dates = train['date'].dt.year.value_counts().sort_index()
test_dates = test['date'].dt.year.value_counts().sort_index()
print(f"\nTrain years: {dict(train_dates)}")
print(f"Test years:  {dict(test_dates)}")

# Check signal stability across time
print("\nSignal IC by period (contrarian_signal -> ret10d):")
for year in sorted(df['date'].dt.year.unique()):
    subset = df[df['date'].dt.year == year]
    if len(subset) > 100:
        ic, p = stats.spearmanr(subset['contrarian_signal'], subset['ret10d'])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {year}: IC={ic:+.4f}{sig} (n={len(subset):,})")

# Check market regime
print("\nMarket regime in test period:")
test_ret = test['ret10d'].mean()
train_ret = train['ret10d'].mean()
print(f"  Avg 10d return in train: {train_ret*100:+.2f}%")
print(f"  Avg 10d return in test:  {test_ret*100:+.2f}%")
