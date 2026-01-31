"""
Random Feature Baseline Test (Null Hypothesis)

Compares our model's performance using:
1. Top 15 features by importance (actual strategy)
2. Random 15 features (null hypothesis baseline)

If the model works, selected features should significantly outperform random selection.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise

from auto_researcher.data.price_loader import download_price_history
from auto_researcher.features.enhanced import compute_all_enhanced_features, EnhancedFeatureConfig
from auto_researcher.features.targets import build_targets, TargetConfig

# S&P 500 subset
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
           'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
           'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT',
           'CRM', 'DHR', 'NEE', 'NKE', 'LIN', 'TXN', 'PM', 'VZ', 'RTX', 'ORCL',
           'HON', 'BMY', 'UPS', 'UNP', 'AMGN', 'IBM', 'CAT', 'SBUX', 'GS', 'BA']


def compute_strategy_return(features_df, targets_df, adj_close, selected_features, top_k_stocks=15, debug=False):
    """
    Compute strategy return using given features.
    Returns tuple of (total_return, sharpe_ratio)
    """
    date_level = 'date' if 'date' in features_df.index.names else 'Date'
    ticker_level = 'ticker' if 'ticker' in features_df.index.names else 'Ticker'
    
    # Handle MultiIndex columns in adj_close
    if isinstance(adj_close.columns, pd.MultiIndex):
        ticker_cols = adj_close.columns.get_level_values(-1).tolist()
    else:
        ticker_cols = adj_close.columns.tolist()
    
    all_dates = sorted(features_df.index.get_level_values(date_level).unique())
    monthly_dates = []
    for i, d in enumerate(all_dates):
        if i == 0 or d.month != all_dates[i-1].month:
            if d >= pd.Timestamp('2022-06-01') and d <= pd.Timestamp('2024-12-31'):
                monthly_dates.append(d)
    
    if debug:
        print(f"  Monthly dates: {len(monthly_dates)}")
    
    portfolio_returns = []
    
    for i in range(len(monthly_dates) - 1):
        rebal_date = monthly_dates[i]
        next_rebal = monthly_dates[i + 1]
        
        # Get training data (rolling window)
        train_start = rebal_date - pd.Timedelta(days=750)
        train_mask = (features_df.index.get_level_values(date_level) >= train_start) & \
                     (features_df.index.get_level_values(date_level) < rebal_date)
        
        X_train = features_df[train_mask][selected_features].dropna()
        y_train = targets_df.loc[X_train.index] if hasattr(targets_df, 'loc') else targets_df[train_mask]
        
        # Align
        common_idx = X_train.index.intersection(y_train.dropna().index)
        if len(common_idx) < 100:
            if debug:
                print(f"  {rebal_date.strftime('%Y-%m')}: skipping, only {len(common_idx)} samples")
            continue
            
        X_train = X_train.loc[common_idx]
        y_train = y_train.loc[common_idx]
        
        # Flatten y
        y_vals = y_train.values.ravel() if hasattr(y_train, 'values') else np.array(y_train).ravel()
        
        # Train model
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, 
                            random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_train.values, y_vals)
        
        # Get prediction date data
        pred_mask = features_df.index.get_level_values(date_level) == rebal_date
        X_pred = features_df[pred_mask][selected_features].dropna()
        
        if len(X_pred) < top_k_stocks:
            if debug:
                print(f"  {rebal_date.strftime('%Y-%m')}: skipping, only {len(X_pred)} stocks at prediction date")
            continue
            
        # Make predictions
        scores = model.predict(X_pred.values)
        
        # Get tickers - handle tuple index
        tickers_at_date = X_pred.index.get_level_values(ticker_level)
        if isinstance(tickers_at_date[0], tuple):
            tickers_at_date = [t[0] if isinstance(t, tuple) else t for t in tickers_at_date]
        
        pred_df = pd.DataFrame({
            'ticker': tickers_at_date,
            'score': scores
        })
        
        # Select top K stocks
        top_stocks = pred_df.nlargest(top_k_stocks, 'score')['ticker'].tolist()
        
        # Get returns for holding period
        hold_start_candidates = adj_close.index[adj_close.index >= rebal_date]
        if len(hold_start_candidates) == 0:
            continue
        hold_start = hold_start_candidates[0]
        
        hold_end_candidates = adj_close.index[adj_close.index >= next_rebal]
        hold_end = hold_end_candidates[0] if len(hold_end_candidates) > 0 else adj_close.index[-1]
        
        # Compute equal-weight portfolio return
        valid_stocks = [t for t in top_stocks if t in ticker_cols]
        if len(valid_stocks) < 5:
            if debug:
                print(f"  {rebal_date.strftime('%Y-%m')}: skipping, only {len(valid_stocks)} valid stocks")
            continue
        
        # Handle MultiIndex columns
        if isinstance(adj_close.columns, pd.MultiIndex):
            start_prices = adj_close.loc[hold_start].xs(valid_stocks[0], level=-1) if len(valid_stocks) == 1 else \
                          pd.Series([adj_close.loc[hold_start].xs(t, level=-1).values[0] for t in valid_stocks], index=valid_stocks)
            end_prices = pd.Series([adj_close.loc[hold_end].xs(t, level=-1).values[0] for t in valid_stocks], index=valid_stocks)
        else:
            start_prices = adj_close.loc[hold_start, valid_stocks]
            end_prices = adj_close.loc[hold_end, valid_stocks]
            
        stock_returns = (end_prices / start_prices - 1).values
        port_return = np.mean(stock_returns)
        
        portfolio_returns.append(port_return)
        if debug:
            print(f"  {rebal_date.strftime('%Y-%m')}: {len(valid_stocks)} stocks, return={port_return*100:.2f}%")
    
    if len(portfolio_returns) == 0:
        return 0.0, 0.0
        
    total_return = np.prod([1 + r for r in portfolio_returns]) - 1
    avg_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns) if len(portfolio_returns) > 1 else 0.01
    sharpe = (avg_return / std_return) * np.sqrt(12) if std_return > 0 else 0.0
    
    return total_return * 100, sharpe


def main():
    print("Loading price data...")
    prices = download_price_history(TICKERS, start='2019-01-01', end='2025-12-31')
    
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
    
    all_features = features_df.columns.tolist()
    print(f"Total features available: {len(all_features)}")
    
    # Get date level for training
    date_level = 'date' if 'date' in features_df.index.names else 'Date'
    
    # Train one model to get top 15 features by importance
    print("\nTraining model to identify top 15 features...")
    train_mask = features_df.index.get_level_values(date_level) < pd.Timestamp('2022-06-01')
    X_train = features_df[train_mask].dropna()
    y_train = targets_df.loc[X_train.index]
    common_idx = X_train.index.intersection(y_train.dropna().index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]
    y_vals = y_train.values.ravel()
    
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, 
                        random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train.values, y_vals)
    
    importance = model.feature_importances_
    top_15_idx = np.argsort(importance)[-15:][::-1]
    top_15_features = [all_features[i] for i in top_15_idx]
    
    print(f"\nTop 15 features by importance:")
    for i, feat in enumerate(top_15_features[:5]):
        print(f"  {i+1}. {feat}")
    print(f"  ...")
    
    # Test 1: Top 15 features (actual strategy)
    print("\n" + "="*60)
    print("TESTING TOP 15 FEATURES (ACTUAL STRATEGY)")
    print("="*60)
    ret_top15, sharpe_top15 = compute_strategy_return(features_df, targets_df, adj_close, top_15_features, debug=True)
    print(f"  Total Return: {ret_top15:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_top15:.2f}")
    
    # Test 2: Random 15 features (5 trials)
    print("\n" + "="*60)
    print("TESTING RANDOM 15 FEATURES (NULL HYPOTHESIS)")
    print("="*60)
    
    random_returns = []
    random_sharpes = []
    np.random.seed(42)
    
    # Convert to list for random sampling (features may be tuples)
    all_features_list = list(range(len(all_features)))
    
    for trial in range(10):
        random_idx = np.random.choice(all_features_list, size=15, replace=False)
        random_features = [all_features[i] for i in random_idx]
        ret, sharpe = compute_strategy_return(features_df, targets_df, adj_close, random_features)
        random_returns.append(ret)
        random_sharpes.append(sharpe)
        print(f"  Trial {trial+1}: Return={ret:6.2f}%, Sharpe={sharpe:.2f}")
    
    print(f"\n  Random Baseline Summary:")
    print(f"    Mean Return: {np.mean(random_returns):.2f}%")
    print(f"    Std Return:  {np.std(random_returns):.2f}%")
    print(f"    Mean Sharpe: {np.mean(random_sharpes):.2f}")
    print(f"    Best Random: {max(random_returns):.2f}% (Sharpe={max(random_sharpes):.2f})")
    print(f"    Worst Random: {min(random_returns):.2f}% (Sharpe={min(random_sharpes):.2f})")
    
    # Statistical significance
    print("\n" + "="*60)
    print("SIGNIFICANCE TEST")
    print("="*60)
    z_score = (ret_top15 - np.mean(random_returns)) / (np.std(random_returns) + 1e-6)
    print(f"  Top 15 Return:     {ret_top15:.2f}%")
    print(f"  Random Mean:       {np.mean(random_returns):.2f}%")
    print(f"  Random Std:        {np.std(random_returns):.2f}%")
    print(f"  Z-Score:           {z_score:.2f}")
    
    better_than_random = sum(1 for r in random_returns if ret_top15 > r) / len(random_returns) * 100
    print(f"  Beat Random Trials: {better_than_random:.0f}%")
    
    if z_score > 2:
        print(f"\n  ✓ Top 15 features SIGNIFICANTLY outperform random (z={z_score:.1f})")
    elif z_score > 1:
        print(f"\n  ~ Top 15 features moderately outperform random (z={z_score:.1f})")
    else:
        print(f"\n  ✗ Top 15 features do NOT significantly outperform random (z={z_score:.1f})")


if __name__ == '__main__':
    main()
