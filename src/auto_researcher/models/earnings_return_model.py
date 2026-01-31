"""
Earnings Call Sentiment → Stock Return Prediction Model

This outlines the recommended approach for predicting post-earnings returns
based on our analysis of 186k earnings call transcripts.
"""

# =============================================================================
# 1. PROBLEM FORMULATION
# =============================================================================
"""
TARGET VARIABLE OPTIONS:
------------------------
1. ret_1d: 1-day post-earnings return (next trading day)
   - Captures immediate market reaction
   - Most liquid, least noisy
   - Risk: Much already priced in from after-hours

2. ret_drift_5d: 5-day cumulative abnormal return  
   - Captures post-earnings drift (PEAD)
   - Well-documented anomaly
   - RECOMMENDED as primary target

3. ret_alpha_30d: 30-day market-adjusted return
   - Longer-term signal
   - More noise, but bigger moves

FEATURE CATEGORIES:
-------------------
1. Sentiment Features (from FinBERT)
2. Sentiment Change Features (QoQ deltas)
3. Guidance Features (from LLM extraction)
4. Fundamental Surprise (EPS beat/miss)
5. Control Variables (sector, size, etc.)
"""

# =============================================================================
# 2. DATA PIPELINE
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def build_dataset():
    """
    Build the modeling dataset by:
    1. Loading transcripts
    2. Computing FinBERT sentiment
    3. Getting stock returns post-earnings
    4. Merging everything
    """
    
    # Load transcripts
    url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
    transcripts = pd.read_parquet(url)
    
    # For each transcript, we need:
    # - FinBERT sentiment score
    # - Prior quarter sentiment (for delta)
    # - Stock return post-earnings
    # - EPS surprise (if available)
    
    # This would be computed offline and cached
    pass


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def compute_features(df):
    """
    Feature engineering for the prediction model.
    
    Key insight: Use BOTH absolute and delta features.
    """
    features = pd.DataFrame(index=df.index)
    
    # -----------------------------
    # A. RAW SENTIMENT FEATURES
    # -----------------------------
    features['finbert_score'] = df['finbert_sentiment']  # -1 to +1
    
    # Bucketed version for interaction effects
    features['sentiment_bucket'] = pd.cut(
        df['finbert_sentiment'], 
        bins=[-1, -0.3, 0.3, 1],
        labels=['negative', 'neutral', 'positive']
    )
    
    # -----------------------------
    # B. DELTA FEATURES (most important!)
    # -----------------------------
    # Quarter-over-quarter change in sentiment
    features['sentiment_change_qoq'] = (
        df.groupby('symbol')['finbert_sentiment']
        .diff()
    )
    
    # Year-over-year change (same quarter last year)
    features['sentiment_change_yoy'] = (
        df.groupby('symbol')['finbert_sentiment']
        .diff(periods=4)
    )
    
    # Deviation from company's 4-quarter rolling mean
    features['sentiment_deviation'] = (
        df['finbert_sentiment'] - 
        df.groupby('symbol')['finbert_sentiment']
        .rolling(4, min_periods=2).mean().reset_index(0, drop=True)
    )
    
    # Z-score within company (normalized)
    company_stats = df.groupby('symbol')['finbert_sentiment'].agg(['mean', 'std'])
    df_with_stats = df.merge(company_stats, on='symbol', how='left')
    features['sentiment_zscore'] = (
        (df['finbert_sentiment'] - df_with_stats['mean']) / 
        df_with_stats['std'].clip(lower=0.01)
    )
    
    # -----------------------------
    # C. GUIDANCE FEATURES
    # -----------------------------
    # Binary/categorical
    features['guidance_raised'] = (df['guidance_direction'] == 'raised').astype(int)
    features['guidance_lowered'] = (df['guidance_direction'] == 'lowered').astype(int)
    features['guidance_not_provided'] = (df['guidance_direction'] == 'not_provided').astype(int)
    
    # Guidance change from prior quarter
    features['guidance_improved'] = (
        (df['guidance_direction'] == 'raised') & 
        (df.groupby('symbol')['guidance_direction'].shift(1) != 'raised')
    ).astype(int)
    
    # -----------------------------
    # D. FUNDAMENTAL SURPRISE
    # -----------------------------
    # EPS surprise (if available from your data source)
    features['eps_surprise_pct'] = df.get('eps_surprise_pct', 0)
    features['eps_beat'] = (df.get('eps_surprise_pct', 0) > 0).astype(int)
    
    # -----------------------------
    # E. INTERACTION FEATURES
    # -----------------------------
    # Sentiment + Guidance interaction (very predictive!)
    features['positive_with_raise'] = (
        (df['finbert_sentiment'] > 0.3) & 
        (df['guidance_direction'] == 'raised')
    ).astype(int)
    
    features['negative_with_lower'] = (
        (df['finbert_sentiment'] < -0.1) & 
        (df['guidance_direction'] == 'lowered')
    ).astype(int)
    
    # Sentiment confirming earnings surprise
    features['sentiment_confirms_beat'] = (
        (df['finbert_sentiment'] > 0.3) & 
        (df.get('eps_surprise_pct', 0) > 2)
    ).astype(int)
    
    # -----------------------------
    # F. CONTROL VARIABLES
    # -----------------------------
    features['log_market_cap'] = np.log(df.get('market_cap', 1e9).clip(lower=1e6))
    
    # Sector dummies (for fixed effects)
    if 'sector' in df.columns:
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        features = pd.concat([features, sector_dummies], axis=1)
    
    return features


# =============================================================================
# 4. TARGET VARIABLE COMPUTATION
# =============================================================================

def compute_returns(df, price_data):
    """
    Compute post-earnings returns.
    
    Parameters:
    - df: earnings data with 'symbol' and 'report_date'
    - price_data: daily stock prices
    """
    
    returns = pd.DataFrame(index=df.index)
    
    for idx, row in df.iterrows():
        symbol = row['symbol']
        report_date = pd.to_datetime(row['report_date'])
        
        # Get price data for this stock
        stock_prices = price_data[price_data['symbol'] == symbol].copy()
        stock_prices = stock_prices.sort_values('date')
        
        # Find the next trading day after earnings
        future_prices = stock_prices[stock_prices['date'] > report_date]
        
        if len(future_prices) >= 5:
            # T+0 close (day of or day after earnings)
            p0 = future_prices.iloc[0]['adj_close']
            p1 = future_prices.iloc[1]['adj_close']  # T+1
            p5 = future_prices.iloc[4]['adj_close']  # T+5
            
            returns.loc[idx, 'ret_1d'] = (p1 - p0) / p0
            returns.loc[idx, 'ret_5d'] = (p5 - p0) / p0
            
            # Could also compute abnormal return vs SPY
    
    return returns


# =============================================================================
# 5. MODEL ARCHITECTURE
# =============================================================================
"""
RECOMMENDED: Gradient Boosting (XGBoost/LightGBM)

Why XGBoost over linear regression:
1. Captures non-linear relationships (e.g., extreme sentiment more predictive)
2. Handles feature interactions automatically
3. Robust to outliers
4. Built-in feature importance

Why NOT deep learning:
1. Tabular data - trees dominate
2. Not enough samples for complex neural nets
3. Interpretability matters for finance
"""

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def train_model(X, y, n_splits=5):
    """
    Train XGBoost with proper time-series cross-validation.
    
    CRITICAL: No data leakage! 
    - Train on past, predict future
    - Never use future data to predict past
    """
    
    # Time-series split (not random!)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Model parameters (start conservative)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,  # Shallow to prevent overfitting
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,  # Prevent fitting noise
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Cross-validation
    cv_scores = []
    feature_importance = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        y_pred = model.predict(X_test)
        
        # Use IC (Information Coefficient) not R² for finance
        ic = np.corrcoef(y_test, y_pred)[0, 1]
        cv_scores.append(ic)
        
        feature_importance.append(
            pd.Series(model.feature_importances_, index=X.columns)
        )
    
    print(f"Cross-validated IC: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    
    # Average feature importance
    avg_importance = pd.concat(feature_importance, axis=1).mean(axis=1)
    print("\nTop 10 Features:")
    print(avg_importance.sort_values(ascending=False).head(10))
    
    # Train final model on all data
    model.fit(X, y)
    
    return model, cv_scores


# =============================================================================
# 6. EVALUATION METRICS
# =============================================================================
"""
DON'T USE: R², MSE
- Stock returns are noisy, R² will be low
- Low R² doesn't mean model is useless!

DO USE:
1. Information Coefficient (IC): Correlation between predicted and actual
   - IC of 0.05-0.10 is excellent for daily returns
   
2. Hit Rate: % of correct direction predictions
   - Random = 50%, good model = 52-55%
   
3. Long-Short Return: Backtest performance
   - Go long top decile, short bottom decile
   - Measure Sharpe ratio
"""

def evaluate_model(y_true, y_pred):
    """Calculate finance-appropriate metrics."""
    
    # Information Coefficient
    ic = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Hit Rate (direction accuracy)
    hit_rate = np.mean((y_true > 0) == (y_pred > 0))
    
    # Long-short spread (top vs bottom quintile)
    quintiles = pd.qcut(y_pred, 5, labels=False)
    long_ret = y_true[quintiles == 4].mean()
    short_ret = y_true[quintiles == 0].mean()
    long_short_spread = long_ret - short_ret
    
    print(f"Information Coefficient: {ic:.4f}")
    print(f"Hit Rate: {hit_rate:.2%}")
    print(f"Long-Short Spread: {long_short_spread:.4f}")
    
    return {'ic': ic, 'hit_rate': hit_rate, 'spread': long_short_spread}


# =============================================================================
# 7. COMPLETE PIPELINE
# =============================================================================

def run_pipeline():
    """
    End-to-end pipeline for earnings sentiment prediction.
    """
    
    # 1. Load and prepare data
    print("Loading data...")
    # df = load_and_merge_data()
    
    # 2. Compute features
    print("Computing features...")
    # X = compute_features(df)
    
    # 3. Compute target (post-earnings returns)
    print("Computing returns...")
    # y = compute_returns(df)['ret_5d']
    
    # 4. Remove NaN rows
    # valid_mask = X.notna().all(axis=1) & y.notna()
    # X, y = X[valid_mask], y[valid_mask]
    
    # 5. Train model with time-series CV
    print("Training model...")
    # model, cv_scores = train_model(X, y)
    
    # 6. Analyze feature importance
    # print("\nFeature Importance Analysis:")
    # importance = pd.Series(model.feature_importances_, index=X.columns)
    # print(importance.sort_values(ascending=False).head(15))
    
    print("Pipeline complete!")


if __name__ == "__main__":
    # This is a template - need to fill in with actual data loading
    print(__doc__)
    print("\nKey Features to Use:")
    print("1. sentiment_change_qoq (most predictive!)")
    print("2. sentiment_deviation (unusual for THIS company)")
    print("3. guidance_raised / guidance_lowered")
    print("4. positive_with_raise (interaction)")
    print("5. eps_surprise_pct (if available)")
