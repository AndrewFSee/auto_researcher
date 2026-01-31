"""
Earnings Call Sentiment → Stock Return Prediction Model
========================================================
Builds and backtests a model predicting post-earnings returns
using FinBERT sentiment from 186k transcripts.

Note: DefeatBeta has ~1 week lag, so this is for alpha research only.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
import yfinance as yf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_transcripts():
    """Load DefeatBeta earnings transcripts."""
    print("Loading transcripts from HuggingFace...")
    url = 'https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet'
    df = pd.read_parquet(url)
    
    # Parse dates
    df['report_date'] = pd.to_datetime(df['report_date'])
    
    # Sort by company and date
    df = df.sort_values(['symbol', 'report_date']).reset_index(drop=True)
    
    print(f"Loaded {len(df):,} transcripts")
    return df


def get_liquid_universe(df, min_transcripts=20):
    """
    Filter to liquid stocks with enough history.
    Focus on S&P 500 type names for cleaner data.
    """
    # Count transcripts per symbol
    counts = df.groupby('symbol').size()
    valid_symbols = counts[counts >= min_transcripts].index.tolist()
    
    # Filter
    df_filtered = df[df['symbol'].isin(valid_symbols)].copy()
    
    print(f"Filtered to {len(valid_symbols)} stocks with {min_transcripts}+ transcripts")
    print(f"Total samples: {len(df_filtered):,}")
    
    return df_filtered


# =============================================================================
# 2. COMPUTE FINBERT SENTIMENT
# =============================================================================

def extract_text_for_sentiment(transcript_content, max_chars=4000):
    """
    Extract key text from transcript for sentiment analysis.
    Include all management remarks, skip operators and analyst questions.
    """
    if not isinstance(transcript_content, list):
        return ""
    
    text_parts = []
    for item in transcript_content:
        if isinstance(item, dict):
            content = item.get('content', '')
            speaker = str(item.get('speaker', '')).lower()
            
            # Skip headers, operators, and analyst sections
            skip_patterns = [
                'operator', 'analysts:', 'executives:', 
                'questions-and-answer', 'operator instruction'
            ]
            if any(pat in speaker for pat in skip_patterns):
                continue
            
            # Skip analyst questions (usually have firm name with dash or hyphen)
            if ' – ' in speaker or ' - ' in speaker:
                # Check if it's an analyst (usually has firm name like "Morgan Stanley")
                analyst_firms = ['capital', 'securities', 'partners', 'advisors', 
                                 'research', 'stanley', 'sachs', 'lynch', 'suisse']
                if any(firm in speaker for firm in analyst_firms):
                    continue
            
            # Include everything else (company executives)
            if content and len(content) > 20:  # Skip very short responses like "Yes" "No"
                text_parts.append(content)
    
    text = ' '.join(text_parts)
    
    # Truncate for FinBERT (max 512 tokens ~ 2000 chars for safety)
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return text


def compute_finbert_sentiment(df, batch_size=16):
    """
    Compute FinBERT sentiment for all transcripts.
    This is the expensive step - will cache results.
    """
    print("\nLoading FinBERT...")
    pipe = pipeline(
        'sentiment-analysis', 
        model='ProsusAI/finbert',
        device=-1,  # CPU, use 0 for GPU
        truncation=True,
        max_length=512
    )
    
    print("Computing sentiment for all transcripts...")
    sentiments = []
    
    # Process in batches
    texts = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text"):
        text = extract_text_for_sentiment(row['transcripts'])
        texts.append(text if text else "neutral")  # Fallback
    
    # Batch inference
    print("Running FinBERT inference...")
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT"):
        batch = texts[i:i+batch_size]
        batch_results = pipe(batch)
        results.extend(batch_results)
    
    # Convert to scores
    for r in results:
        label = r['label']
        score = r['score']
        if label == 'positive':
            sentiments.append(score)
        elif label == 'negative':
            sentiments.append(-score)
        else:
            sentiments.append(0)
    
    df['finbert_sentiment'] = sentiments
    
    print(f"Sentiment computed. Mean: {df['finbert_sentiment'].mean():.3f}")
    return df


# =============================================================================
# 3. GET STOCK RETURNS
# =============================================================================

def get_price_data(symbols, start_date='2015-01-01'):
    """
    Download historical price data from Yahoo Finance.
    """
    print(f"\nDownloading price data for {len(symbols)} stocks...")
    
    try:
        # Download all at once - yfinance handles batching internally
        data = yf.download(
            symbols, 
            start=start_date, 
            end=datetime.now().strftime('%Y-%m-%d'),
            progress=True,
            threads=True,
            ignore_tz=True
        )
        
        # Extract Adj Close prices
        if 'Adj Close' in data.columns:
            # Multiple stocks
            prices = data['Adj Close']
        elif isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close']
        else:
            # Single stock
            prices = data[['Adj Close']]
            prices.columns = [symbols[0]]
        
        # Drop any columns that are all NaN
        prices = prices.dropna(axis=1, how='all')
        
        print(f"Downloaded prices for {len(prices.columns)} stocks, {len(prices)} days")
        return prices
        
    except Exception as e:
        print(f"Error downloading prices: {e}")
        return pd.DataFrame()


def compute_post_earnings_returns(df, prices, forward_days=[1, 5, 10, 20]):
    """
    Compute forward returns after each earnings call.
    """
    print("\nComputing post-earnings returns...")
    
    # Initialize return columns
    for d in forward_days:
        df[f'ret_{d}d'] = np.nan
    
    # Also compute benchmark returns (SPY)
    try:
        spy = yf.download('SPY', start='2015-01-01', progress=False)['Adj Close']
    except:
        spy = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing returns"):
        symbol = row['symbol']
        report_date = row['report_date']
        
        if symbol not in prices.columns:
            continue
        
        stock_prices = prices[symbol].dropna()
        
        # Find first trading day on or after report date
        future_dates = stock_prices.index[stock_prices.index >= report_date]
        
        if len(future_dates) < max(forward_days) + 1:
            continue
        
        # Get prices
        try:
            p0 = stock_prices.loc[future_dates[0]]
            
            for d in forward_days:
                if len(future_dates) > d:
                    pd_ = stock_prices.loc[future_dates[d]]
                    ret = (pd_ - p0) / p0
                    
                    # Compute abnormal return (vs SPY)
                    if spy is not None:
                        spy_future = spy[spy.index >= report_date]
                        if len(spy_future) > d:
                            spy_ret = (spy_future.iloc[d] - spy_future.iloc[0]) / spy_future.iloc[0]
                            ret = ret - spy_ret  # Abnormal return
                    
                    df.loc[idx, f'ret_{d}d'] = ret
        except Exception as e:
            continue
    
    # Print stats
    for d in forward_days:
        valid = df[f'ret_{d}d'].notna().sum()
        print(f"  ret_{d}d: {valid:,} valid samples, mean={df[f'ret_{d}d'].mean():.4f}")
    
    return df


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Create predictive features from sentiment and other signals.
    """
    print("\nEngineering features...")
    
    features = pd.DataFrame(index=df.index)
    
    # ----- RAW SENTIMENT -----
    features['finbert_score'] = df['finbert_sentiment']
    
    # ----- DELTA FEATURES (most important!) -----
    # Sort by company and date first
    df = df.sort_values(['symbol', 'report_date'])
    
    # Quarter-over-quarter change
    features['sentiment_change_qoq'] = df.groupby('symbol')['finbert_sentiment'].diff()
    
    # Year-over-year change (4 quarters ago)
    features['sentiment_change_yoy'] = df.groupby('symbol')['finbert_sentiment'].diff(periods=4)
    
    # Rolling mean and deviation
    rolling_mean = df.groupby('symbol')['finbert_sentiment'].transform(
        lambda x: x.rolling(4, min_periods=2).mean()
    )
    features['sentiment_deviation'] = df['finbert_sentiment'] - rolling_mean
    
    # Z-score within company
    company_mean = df.groupby('symbol')['finbert_sentiment'].transform('mean')
    company_std = df.groupby('symbol')['finbert_sentiment'].transform('std').clip(lower=0.01)
    features['sentiment_zscore'] = (df['finbert_sentiment'] - company_mean) / company_std
    
    # ----- MOMENTUM FEATURES -----
    # 2-quarter sentiment momentum
    features['sentiment_momentum_2q'] = df.groupby('symbol')['finbert_sentiment'].transform(
        lambda x: x.diff() + x.diff().shift(1)
    )
    
    # Is this an improvement from negative territory?
    features['recovery_signal'] = (
        (df['finbert_sentiment'] > 0) & 
        (df.groupby('symbol')['finbert_sentiment'].shift(1) < 0)
    ).astype(int)
    
    # Is this a deterioration from positive?
    features['deterioration_signal'] = (
        (df['finbert_sentiment'] < 0) & 
        (df.groupby('symbol')['finbert_sentiment'].shift(1) > 0)
    ).astype(int)
    
    # ----- EXTREME SENTIMENT -----
    features['extreme_positive'] = (df['finbert_sentiment'] > 0.6).astype(int)
    features['extreme_negative'] = (df['finbert_sentiment'] < -0.2).astype(int)
    
    # ----- CONSISTENCY -----
    # Standard deviation of last 4 quarters (volatile vs stable sentiment)
    features['sentiment_volatility'] = df.groupby('symbol')['finbert_sentiment'].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )
    
    # ----- RELATIVE FEATURES -----
    # Cross-sectional rank (percentile vs other companies that quarter)
    features['sentiment_rank'] = df.groupby('report_date')['finbert_sentiment'].rank(pct=True)
    
    # ----- TIME FEATURES -----
    features['quarter'] = pd.to_datetime(df['report_date']).dt.quarter
    features['year'] = pd.to_datetime(df['report_date']).dt.year
    
    print(f"Created {len(features.columns)} features")
    
    return features


# =============================================================================
# 5. MODEL TRAINING
# =============================================================================

from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

def train_xgboost_model(X, y, n_splits=5):
    """
    Train XGBoost with proper time-series cross-validation.
    """
    print("\nTraining XGBoost model...")
    
    # Remove NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    print(f"Training samples: {len(X_clean):,}")
    
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Model parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.03,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    # Cross-validation
    cv_results = []
    all_predictions = pd.Series(index=X_clean.index, dtype=float)
    feature_importances = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        y_pred = model.predict(X_test)
        all_predictions.iloc[test_idx] = y_pred
        
        # Calculate metrics
        ic = np.corrcoef(y_test, y_pred)[0, 1]
        hit_rate = np.mean((y_test > 0) == (y_pred > 0))
        
        cv_results.append({
            'fold': fold + 1,
            'ic': ic,
            'hit_rate': hit_rate,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        })
        
        feature_importances.append(
            pd.Series(model.feature_importances_, index=X_clean.columns)
        )
        
        print(f"  Fold {fold+1}: IC={ic:.4f}, Hit Rate={hit_rate:.2%}")
    
    # Summary
    cv_df = pd.DataFrame(cv_results)
    print(f"\n{'='*50}")
    print(f"Cross-Validation Summary:")
    print(f"  Mean IC: {cv_df['ic'].mean():.4f} (+/- {cv_df['ic'].std():.4f})")
    print(f"  Mean Hit Rate: {cv_df['hit_rate'].mean():.2%}")
    
    # Feature importance
    avg_importance = pd.concat(feature_importances, axis=1).mean(axis=1)
    print(f"\nTop 10 Features:")
    for feat, imp in avg_importance.sort_values(ascending=False).head(10).items():
        print(f"  {feat}: {imp:.4f}")
    
    # Train final model on all data
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X_clean, y_clean, verbose=False)
    
    return final_model, all_predictions, cv_df, avg_importance


# =============================================================================
# 6. BACKTEST ANALYSIS
# =============================================================================

def backtest_analysis(df, predictions, target_col='ret_5d'):
    """
    Analyze backtest performance with long-short portfolios.
    """
    print("\n" + "="*60)
    print("BACKTEST ANALYSIS")
    print("="*60)
    
    # Merge predictions with actual data
    df_bt = df.copy()
    df_bt['prediction'] = predictions
    df_bt = df_bt.dropna(subset=['prediction', target_col])
    
    # Overall metrics
    ic = np.corrcoef(df_bt[target_col], df_bt['prediction'])[0, 1]
    hit_rate = np.mean((df_bt[target_col] > 0) == (df_bt['prediction'] > 0))
    
    print(f"\nOverall Metrics:")
    print(f"  Information Coefficient: {ic:.4f}")
    print(f"  Hit Rate: {hit_rate:.2%}")
    
    # Quintile analysis
    df_bt['quintile'] = pd.qcut(df_bt['prediction'], 5, labels=[1, 2, 3, 4, 5])
    
    quintile_returns = df_bt.groupby('quintile')[target_col].agg(['mean', 'std', 'count'])
    quintile_returns['sharpe'] = quintile_returns['mean'] / quintile_returns['std'] * np.sqrt(4)  # Quarterly
    
    print(f"\nQuintile Returns ({target_col}):")
    print(quintile_returns.round(4))
    
    # Long-short spread
    long_ret = df_bt[df_bt['quintile'] == 5][target_col].mean()
    short_ret = df_bt[df_bt['quintile'] == 1][target_col].mean()
    spread = long_ret - short_ret
    
    print(f"\nLong-Short Analysis:")
    print(f"  Long (Q5) Return: {long_ret:.4f} ({long_ret*100:.2f}%)")
    print(f"  Short (Q1) Return: {short_ret:.4f} ({short_ret*100:.2f}%)")
    print(f"  L-S Spread: {spread:.4f} ({spread*100:.2f}%)")
    
    # Annualized (assuming ~4 earnings per year per stock)
    # But we're looking at 5-day returns, so annualize differently
    # Roughly 50 earnings weeks per year
    annual_spread = spread * 50
    print(f"  Annualized Spread (approx): {annual_spread:.2%}")
    
    # Time series of spread
    df_bt['year_quarter'] = df_bt['report_date'].dt.to_period('Q')
    
    quarterly_spread = df_bt.groupby('year_quarter').apply(
        lambda x: x[x['quintile'] == 5][target_col].mean() - 
                  x[x['quintile'] == 1][target_col].mean()
    )
    
    print(f"\nQuarterly L-S Spread Statistics:")
    print(f"  Mean: {quarterly_spread.mean():.4f}")
    print(f"  Std: {quarterly_spread.std():.4f}")
    print(f"  Sharpe: {quarterly_spread.mean() / quarterly_spread.std() * 2:.2f}")  # Annualized
    print(f"  Hit Rate (positive quarters): {(quarterly_spread > 0).mean():.2%}")
    
    return df_bt, quintile_returns, quarterly_spread


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def run_full_pipeline(
    max_stocks=200,  # Limit for speed
    min_transcripts=20,
    target='ret_5d'
):
    """
    Run the complete pipeline from data loading to backtest.
    """
    print("="*60)
    print("EARNINGS SENTIMENT → RETURN PREDICTION MODEL")
    print("="*60)
    
    # 1. Load transcripts
    df = load_transcripts()
    
    # 2. Filter to liquid universe
    df = get_liquid_universe(df, min_transcripts=min_transcripts)
    
    # Limit stocks for speed (remove for full run)
    if max_stocks:
        top_symbols = df.groupby('symbol').size().nlargest(max_stocks).index.tolist()
        df = df[df['symbol'].isin(top_symbols)].copy()
        print(f"Limited to top {max_stocks} stocks by transcript count")
    
    # 3. Compute FinBERT sentiment
    df = compute_finbert_sentiment(df)
    
    # 4. Get price data
    symbols = df['symbol'].unique().tolist()
    prices = get_price_data(symbols)
    
    # 5. Compute returns
    df = compute_post_earnings_returns(df, prices)
    
    # 6. Engineer features
    features = engineer_features(df)
    
    # 7. Train model
    target_col = target
    y = df[target_col]
    
    model, predictions, cv_results, importance = train_xgboost_model(features, y)
    
    # 8. Backtest
    df_bt, quintile_returns, quarterly_spread = backtest_analysis(df, predictions, target_col)
    
    # Save results
    print("\nSaving results...")
    df_bt.to_parquet('data/earnings_model_results.parquet')
    importance.to_csv('data/feature_importance.csv')
    quarterly_spread.to_csv('data/quarterly_spread.csv')
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    return model, df_bt, importance


if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    # Run with limited stocks first for speed
    # Set max_stocks=None for full run
    model, results, importance = run_full_pipeline(
        max_stocks=100,  # Start with 100 stocks
        min_transcripts=20,
        target='ret_5d'
    )
