"""
Extended Walk-Forward Test: 10-15 years covering multiple market regimes
Tests 6, 12, 24, 36-month training windows across bull/bear markets
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'C:/Users/Andrew/projects/auto_researcher')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from recommend import UNIVERSES
from auto_researcher.features.enhanced import EnhancedFeatureConfig, compute_all_enhanced_features
from auto_researcher.features.targets import TargetConfig, build_targets
from auto_researcher.models.xgb_ranking_model import XGBRegressionModel, XGBRegressionConfig

print('Extended Walk-Forward: 15-Year Test Across Market Regimes')
print('=' * 70)
print()
print('Market Regimes to Cover:')
print('  2011-2012: European debt crisis / choppy')
print('  2013-2014: Bull market recovery')
print('  2015-2016: China fears / oil crash')
print('  2017:      Low vol bull market')
print('  2018:      Q4 correction (-20%)')
print('  2019:      Strong bull market')
print('  2020:      COVID crash & recovery')
print('  2021:      Meme stock / momentum era')
print('  2022:      Bear market (-25%)')
print('  2023-2024: AI-led recovery')
print('  2025:      Current regime')
print()

# Get S&P 100 tickers - use current constituents
tickers = UNIVERSES['sp100']()

# Fetch maximum available data (yfinance typically has 20+ years)
end = datetime.now()
start = datetime(2010, 1, 1)  # 15+ years

print(f'Fetching data from {start.date()} to {end.date()}...')
prices = yf.download(tickers + ['SPY'], start=start, end=end, progress=False)['Close']
prices = prices.dropna(axis=1, how='all')

# Remove tickers with less than 80% of data
min_days = len(prices) * 0.8
prices = prices.dropna(axis=1, thresh=int(min_days))

print(f'Got {len(prices)} trading days, {len(prices.columns)-1} stocks (after filtering)')
print(f'Date range: {prices.index[0].date()} to {prices.index[-1].date()}')
print()

# Compute features
feature_config = EnhancedFeatureConfig(
    use_short_reversal=True,
    use_residual_mom=True,
    use_idio_vol=True,
    use_mad_metrics=True,
    use_sector_ohe=False,
    use_cross_sec_norm=True,
)

print('Computing features...')
features = compute_all_enhanced_features(prices, benchmark='SPY', config=feature_config)
features_long = features.stack(level=0, future_stack=True).dropna()
features_long.index.names = ['date', 'ticker']

# Cap features at ±3 z-scores
for col in features_long.columns:
    features_long[col] = features_long[col].clip(-3, 3)

print('Computing targets...')
target_config = TargetConfig(mode='vol_norm', horizon_days=21)
targets = build_targets(prices, target_config, benchmark='SPY')
targets_long = targets.stack()
targets_long.index.names = ['date', 'ticker']

aligned = features_long.join(targets_long.rename('target'), how='inner').dropna()
dates = aligned.index.get_level_values('date').unique().sort_values()
print(f'Aligned data: {len(aligned)} samples, {len(dates)} dates')
print(f'Date range: {dates[0].date()} to {dates[-1].date()}')

config = XGBRegressionConfig(objective='reg:pseudohubererror', n_estimators=100, max_depth=3)

# Test multiple training windows
windows = [6, 12, 24, 36]
results = {w: [] for w in windows}

# Monthly rebalancing - need 36 months of training data first
min_start = dates[0] + pd.Timedelta(days=36*21 + 252)  # 36 months + 1 year buffer for features
rebal_dates = [d for d in dates[::21] if d >= min_start]
print(f'Testing on {len(rebal_dates)} rebalance dates from {rebal_dates[0].date()}')
print()

for i, test_date in enumerate(rebal_dates):
    test_mask = aligned.index.get_level_values('date') == test_date
    test_data = aligned[test_mask]
    
    if len(test_data) < 20:
        continue
    
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    for lookback_months in windows:
        lookback_days = lookback_months * 21
        train_start = test_date - pd.Timedelta(days=lookback_days)
        
        train_mask = (aligned.index.get_level_values('date') >= train_start) & \
                     (aligned.index.get_level_values('date') < test_date)
        train_data = aligned[train_mask]
        
        if len(train_data) < 300:  # Lower threshold for shorter windows
            continue
        
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        
        model = XGBRegressionModel(config)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        ic = pd.Series(preds).corr(y_test.reset_index(drop=True))
        
        # Quintile spread
        pred_series = pd.Series(preds, index=X_test.index.get_level_values('ticker'))
        actual_series = y_test.droplevel(0)
        n = len(pred_series)
        q = max(n // 5, 1)
        top = pred_series.nlargest(q).index
        bot = pred_series.nsmallest(q).index
        spread = actual_series[top].mean() - actual_series[bot].mean()
        
        results[lookback_months].append({
            'date': test_date, 
            'ic': ic, 
            'spread': spread,
            'year': test_date.year
        })
    
    if (i+1) % 20 == 0:
        print(f'  Processed {i+1}/{len(rebal_dates)} dates...')

print()
print('=' * 70)
print('OVERALL RESULTS (FULL 15-YEAR PERIOD)')
print('=' * 70)
print()
print('Training Window Comparison:')
print('-' * 50)
for w in windows:
    df = pd.DataFrame(results[w])
    if len(df) > 0:
        mean_ic = df['ic'].mean()
        ic_std = df['ic'].std()
        ic_tstat = mean_ic / (ic_std / np.sqrt(len(df)))
        mean_spread = df['spread'].mean()
        pct_pos = (df['ic'] > 0).mean() * 100
        print(f'{w:2d}-month window:')
        print(f'   Mean IC: {mean_ic:+.4f} (t={ic_tstat:.2f})')
        print(f'   IC > 0:  {pct_pos:.1f}%')
        print(f'   L/S Spread: {mean_spread:+.3f}')
        print()

# Year-by-year breakdown
print()
print('=' * 70)
print('PERFORMANCE BY YEAR')
print('=' * 70)
all_years = sorted(set(r['year'] for w in windows for r in results[w]))

print()
print(f"{'Year':<6}" + "".join([f'{w}m IC   ' for w in windows]) + "  Best Window")
print('-' * 60)

yearly_winner = {w: 0 for w in windows}

for year in all_years:
    row = f'{year:<6}'
    year_ics = {}
    for w in windows:
        df = pd.DataFrame(results[w])
        year_df = df[df['year'] == year]
        if len(year_df) > 0:
            ic = year_df['ic'].mean()
            year_ics[w] = ic
            row += f'{ic:+.3f}   '
        else:
            row += '  N/A    '
    
    # Find best window for this year
    if year_ics:
        best_w = max(year_ics, key=year_ics.get)
        row += f'  {best_w}m'
        yearly_winner[best_w] += 1
    
    print(row)

print()
print('Winner Count by Window:')
for w in windows:
    print(f'  {w}-month: won {yearly_winner[w]} years')

# Market regime analysis
print()
print('=' * 70)
print('PERFORMANCE BY MARKET REGIME')
print('=' * 70)

regimes = {
    'Bull (2013-14, 2017, 2019, 2021)': [2013, 2014, 2017, 2019, 2021],
    'Bear/Crisis (2015-16, 2018, 2020, 2022)': [2015, 2016, 2018, 2020, 2022],
    'Recovery (2023-25)': [2023, 2024, 2025],
}

print()
for regime_name, years in regimes.items():
    print(f'{regime_name}:')
    for w in windows:
        df = pd.DataFrame(results[w])
        regime_df = df[df['year'].isin(years)]
        if len(regime_df) > 0:
            mean_ic = regime_df['ic'].mean()
            print(f'  {w:2d}-month: IC = {mean_ic:+.3f}')
    print()

print()
print('=' * 70)
print('CONCLUSION')
print('=' * 70)

# Determine overall winner
overall_ics = {}
for w in windows:
    df = pd.DataFrame(results[w])
    if len(df) > 0:
        overall_ics[w] = df['ic'].mean()

best_overall = max(overall_ics, key=overall_ics.get)
print(f'\nBest overall window: {best_overall}-month (Mean IC = {overall_ics[best_overall]:+.3f})')
print(f'Wins most years: {max(yearly_winner, key=yearly_winner.get)}-month ({max(yearly_winner.values())} years)')
