"""
Analyze feature selection stability across rolling windows.
Checks if the same features are consistently selected each month.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from the package
from auto_researcher.data.price_loader import download_price_history
from auto_researcher.features.enhanced import compute_all_enhanced_features, EnhancedFeatureConfig
from auto_researcher.features.targets import build_targets, TargetConfig

# S&P 500 subset (50 tickers for speed)
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
           'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
           'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT',
           'CRM', 'DHR', 'NEE', 'NKE', 'LIN', 'TXN', 'PM', 'VZ', 'RTX', 'ORCL',
           'HON', 'BMY', 'UPS', 'UNP', 'AMGN', 'IBM', 'CAT', 'SBUX', 'GS', 'BA']


def main():
    print("Loading price data...")
    prices = download_price_history(TICKERS, start='2019-01-01', end='2025-12-31')
    
    # Get adjusted close prices
    if 'Adj Close' in prices.columns.get_level_values(0):
        adj_close = prices['Adj Close']
    else:
        adj_close = prices['Close']
    
    print(f"Loaded {len(adj_close.columns)} tickers")
    
    print("Computing features...")
    feature_config = EnhancedFeatureConfig()
    features_df = compute_all_enhanced_features(adj_close, config=feature_config)
    
    print("Computing targets...")
    target_config = TargetConfig(mode="vol_norm", horizon_days=63, vol_lookback=63)
    targets_df = build_targets(adj_close, config=target_config)
    
    # Track which features are selected each month
    feature_selections = []
    months_analyzed = []
    
    # Get monthly rebalance dates (handle both 'date' and 'Date' index names)
    date_level = 'date' if 'date' in features_df.index.names else 'Date'
    all_dates = sorted(features_df.index.get_level_values(date_level).unique())
    monthly_dates = []
    for i, d in enumerate(all_dates):
        if i == 0 or d.month != all_dates[i-1].month:
            if d >= pd.Timestamp('2022-01-01'):
                monthly_dates.append(d)
    
    print(f"Analyzing {min(24, len(monthly_dates))} monthly periods (2022-2024)...")
    
    for i, rebal_date in enumerate(monthly_dates[:24]):
        # Get training window (504 days before)
        train_start = rebal_date - pd.Timedelta(days=750)  # ~2 years calendar
        train_mask = (features_df.index.get_level_values(date_level) >= train_start) & \
                     (features_df.index.get_level_values(date_level) < rebal_date)
        
        X_train = features_df[train_mask].dropna()
        y_train = targets_df.loc[X_train.index] if hasattr(targets_df, 'loc') else targets_df[train_mask]
        
        # Align indices
        if hasattr(y_train, 'index'):
            common_idx = X_train.index.intersection(y_train.dropna().index)
        else:
            common_idx = X_train.index
            
        if len(common_idx) < 100:
            print(f"  Skipping {rebal_date.strftime('%Y-%m')}: insufficient data ({len(common_idx)} samples)")
            continue
            
        X_train = X_train.loc[common_idx]
        y_train = y_train.loc[common_idx] if hasattr(y_train, 'loc') else y_train
        
        # Flatten y if needed
        if hasattr(y_train, 'values'):
            y_vals = y_train.values.ravel() if y_train.ndim > 1 else y_train.values
        else:
            y_vals = np.array(y_train).ravel()
        
        # Train XGBoost
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train.values, y_vals)
        
        # Get top 15 features by gain importance
        importance = model.feature_importances_
        feature_names = X_train.columns.tolist()
        top_15_idx = np.argsort(importance)[-15:][::-1]
        top_15_names = [feature_names[i] for i in top_15_idx]
        
        feature_selections.append(top_15_names)
        months_analyzed.append(rebal_date.strftime('%Y-%m'))
        print(f"  {rebal_date.strftime('%Y-%m')}: top feature = {top_15_names[0]}")
    
    print('\n' + '='*60)
    print('FEATURE STABILITY ANALYSIS')
    print('='*60)
    
    # Count how often each feature appears in top 15
    all_selected = [f for month_features in feature_selections for f in month_features]
    feature_counts = Counter(all_selected)
    
    print(f'\nAnalyzed {len(months_analyzed)} months')
    print(f'\nFEATURES BY SELECTION FREQUENCY (appeared in top 15):')
    print('-'*60)
    for feat, count in feature_counts.most_common(25):
        pct = 100 * count / len(months_analyzed)
        bar = '█' * int(pct / 5)
        feat_str = str(feat) if isinstance(feat, tuple) else feat
        print(f'{feat_str:50s} {count:3d}/{len(months_analyzed)} ({pct:5.1f}%) {bar}')
    
    # Check overlap between consecutive months
    overlaps = []
    for i in range(1, len(feature_selections)):
        prev = set(feature_selections[i-1])
        curr = set(feature_selections[i])
        overlap = len(prev & curr) / 15 * 100
        overlaps.append(overlap)
    
    print(f'\nMONTH-TO-MONTH STABILITY:')
    print(f'  Avg overlap between consecutive months: {np.mean(overlaps):.1f}%')
    print(f'  Min overlap: {np.min(overlaps):.1f}%')
    print(f'  Max overlap: {np.max(overlaps):.1f}%')
    
    # Features that ALWAYS appear
    always_selected = [f for f, c in feature_counts.items() if c == len(months_analyzed)]
    print(f'\nFEATURES ALWAYS IN TOP 15: {len(always_selected)}')
    for f in always_selected:
        print(f'  - {f}')
    
    # Features appearing > 80% of the time
    usually_selected = [f for f, c in feature_counts.items() if c >= 0.8 * len(months_analyzed)]
    print(f'\nFEATURES IN TOP 15 >= 80% OF TIME: {len(usually_selected)}')
    for f in usually_selected:
        print(f'  - {f} ({feature_counts[f]}/{len(months_analyzed)})')


if __name__ == '__main__':
    main()
