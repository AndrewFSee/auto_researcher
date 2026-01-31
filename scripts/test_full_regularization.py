"""Full regularization test with complete data rebuild."""
import pandas as pd
import numpy as np
import sqlite3
import lightgbm as lgb
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("Loading news data from SQLite...")
with sqlite3.connect('data/news.db') as conn:
    news = pd.read_sql("""
        SELECT ticker, date(published_date) as date, sentiment_score 
        FROM articles 
        WHERE sentiment_score IS NOT NULL
    """, conn)

news['date'] = pd.to_datetime(news['date'])

# Aggregate by ticker-date
daily = news.groupby(['ticker', 'date']).agg(
    sentiment_mean=('sentiment_score', 'mean'),
    sentiment_std=('sentiment_score', 'std'),
    news_count=('sentiment_score', 'count'),
    positive_ratio=('sentiment_score', lambda x: (x > 0.1).mean()),
    negative_ratio=('sentiment_score', lambda x: (x < -0.1).mean()),
).reset_index()

# Add rolling features
daily = daily.sort_values(['ticker', 'date'])
for window in [5, 10, 20]:
    daily[f'sentiment_ma{window}'] = daily.groupby('ticker')['sentiment_mean'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

# Volume z-score (news count)
daily['news_count_ma20'] = daily.groupby('ticker')['news_count'].transform(
    lambda x: x.rolling(20, min_periods=5).mean()
)
daily['news_count_std20'] = daily.groupby('ticker')['news_count'].transform(
    lambda x: x.rolling(20, min_periods=5).std()
)
daily['volume_zscore'] = (daily['news_count'] - daily['news_count_ma20']) / daily['news_count_std20'].replace(0, 1)

# Create signals
daily['event_signal'] = daily['sentiment_mean'] * daily['volume_zscore'].clip(lower=0)
daily['contrarian_signal'] = -daily['sentiment_ma20']
daily['alpha_combined'] = 0.1 * daily['event_signal'] + 0.9 * daily['contrarian_signal']

print(f"News features: {len(daily):,} ticker-dates")

# Get returns for top tickers
top_tickers = daily.groupby('ticker').size().nlargest(100).index.tolist()
print(f"Fetching returns for top {len(top_tickers)} tickers...")

prices = yf.download(top_tickers, start='2020-01-01', end='2026-01-20', progress=False)['Close']
if isinstance(prices, pd.Series):
    prices = prices.to_frame()

# Calculate returns
returns_data = []
for ticker in prices.columns:
    df = pd.DataFrame({
        'date': prices.index,
        'ticker': ticker,
        'ret10d': prices[ticker].pct_change(10).shift(-10)
    })
    returns_data.append(df)

returns = pd.concat(returns_data)
returns['date'] = pd.to_datetime(returns['date']).dt.tz_localize(None)

# Merge
merged = daily.merge(returns, on=['ticker', 'date'], how='inner')
merged = merged.dropna(subset=['ret10d', 'sentiment_mean', 'contrarian_signal'])
print(f"Merged dataset: {len(merged):,} rows")

# Time split
merged = merged.sort_values('date')
split_date = merged['date'].quantile(0.8)
train = merged[merged['date'] <= split_date].copy()
test = merged[merged['date'] > split_date].copy()

print(f"Train: {len(train):,} ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Test:  {len(test):,} ({test['date'].min().date()} to {test['date'].max().date()})")

feature_cols = ['sentiment_mean', 'sentiment_std', 'news_count', 
                'positive_ratio', 'negative_ratio',
                'sentiment_ma5', 'sentiment_ma10', 'sentiment_ma20',
                'volume_zscore', 'event_signal', 'contrarian_signal', 'alpha_combined']
available = [c for c in feature_cols if c in merged.columns]

X_train = train[available].fillna(0)
y_train = train['ret10d']
X_test = test[available].fillna(0)
y_test = test['ret10d']

configs = [
    {'name': 'Default',       'n_est': 100, 'depth': 4, 'leaves': 15, 'min_child': 100, 'reg_alpha': 0, 'reg_lambda': 0},
    {'name': 'Heavy reg',     'n_est': 100, 'depth': 4, 'leaves': 15, 'min_child': 100, 'reg_alpha': 5, 'reg_lambda': 10},
    {'name': 'Shallow+reg',   'n_est': 50,  'depth': 2, 'leaves': 4,  'min_child': 500, 'reg_alpha': 5, 'reg_lambda': 10},
    {'name': 'Single tree',   'n_est': 1,   'depth': 2, 'leaves': 4,  'min_child': 500, 'reg_alpha': 0, 'reg_lambda': 0},
    {'name': 'Linear boost',  'n_est': 50,  'depth': 1, 'leaves': 2,  'min_child': 1000, 'reg_alpha': 10, 'reg_lambda': 10},
]

print()
print('='*80)
print('FULL DATA REGULARIZATION TEST')
print('='*80)
print(f"{'Config':<15} {'Train IC':>10} {'Test IC':>10} {'Test p':>12} {'Overfit':>10}")
print('-'*80)

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
    print(f"{cfg['name']:<15} {train_ic:>+10.4f} {test_ic:>+10.4f}{sig:>3} {test_p:>12.6f} {overfit:>+10.4f}")

print('-'*80)
simple_ic, simple_p = stats.spearmanr(test['alpha_combined'], y_test)
sig = '***' if simple_p < 0.001 else ('**' if simple_p < 0.01 else ('*' if simple_p < 0.05 else ''))
print(f"{'Simple signal':<15} {'--':>10} {simple_ic:>+10.4f}{sig:>3} {simple_p:>12.6f}")

contra_ic, contra_p = stats.spearmanr(test['contrarian_signal'], y_test)
sig = '***' if contra_p < 0.001 else ('**' if contra_p < 0.01 else ('*' if contra_p < 0.05 else ''))
print(f"{'Contrarian only':<15} {'--':>10} {contra_ic:>+10.4f}{sig:>3} {contra_p:>12.6f}")

# Quintile analysis for best model (single tree)
print()
print('='*80)
print('QUINTILE ANALYSIS: Single Tree Model')
print('='*80)

model = lgb.LGBMRegressor(n_estimators=1, max_depth=2, num_leaves=4, min_child_samples=500, verbose=-1)
model.fit(X_train, y_train)
test['pred'] = model.predict(X_test)

test['quintile'] = pd.qcut(test['pred'].rank(method='first'), 5, labels=[1,2,3,4,5])
for q in [1, 2, 3, 4, 5]:
    subset = test[test['quintile'] == q]
    print(f"  Q{q}: {subset['ret10d'].mean()*100:+.2f}% (n={len(subset):,})")

spread = test[test['quintile']==5]['ret10d'].mean() - test[test['quintile']==1]['ret10d'].mean()
print(f"\n  Long-Short Spread: {spread*100:+.2f}%")
