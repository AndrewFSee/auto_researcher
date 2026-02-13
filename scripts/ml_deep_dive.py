"""
ML Ranking Model Deep Dive Analysis

Investigates why the ML model is not predictive and identifies issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.WARNING)

from auto_researcher.features.enhanced import (
    EnhancedFeatureConfig,
    compute_all_enhanced_features,
)
from auto_researcher.features.targets import TargetConfig, build_targets


def analyze_model_training():
    """Deep dive into what the ML model is learning."""
    
    print("="*70)
    print("ML MODEL TRAINING ANALYSIS")
    print("="*70)
    
    # Fetch prices for a small universe
    tickers = ["BKNG", "GOOGL", "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "SPY"]
    
    end = datetime.now()
    start = end - timedelta(days=504)  # 2 years
    
    print("\n1. Fetching price data...")
    prices = yf.download(tickers, start=start, end=end, progress=False)['Close']
    print(f"   Got {len(prices)} days of data")
    
    # Compute features
    print("\n2. Computing features...")
    config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,
        use_cross_sec_norm=True,
        cross_sec_norm_robust=True,
    )
    
    features_wide = compute_all_enhanced_features(prices, benchmark='SPY', config=config)
    features_long = features_wide.stack(level=0, future_stack=True)
    features_long.index.names = ["date", "ticker"]
    
    # Get latest features for BKNG
    latest_date = features_long.index.get_level_values("date").max()
    bkng_features = features_long.loc[(latest_date, 'BKNG')]
    
    print(f"\n3. BKNG Feature Values (as of {latest_date.date()}):")
    print("-"*50)
    for col in bkng_features.index:
        val = bkng_features[col]
        if not pd.isna(val):
            print(f"   {col:<35}: {val:+.3f}")
    
    # Compute targets
    print("\n4. Computing forward returns...")
    target_config = TargetConfig(mode="vol_norm", horizon_days=21)
    targets = build_targets(prices.iloc[:-21], target_config, benchmark='SPY')
    targets_stacked = targets.stack()
    targets_stacked.index.names = ["date", "ticker"]
    
    # Check BKNG's recent target values
    bkng_targets = targets_stacked.xs('BKNG', level='ticker')
    print(f"\n5. BKNG Recent 21-day Forward Returns (target values):")
    print("-"*50)
    for date, val in bkng_targets.tail(10).items():
        print(f"   {date.date()}: {val:+.3f}")
    
    # Check feature importance interpretation
    print("\n6. Feature Interpretation for BKNG:")
    print("-"*50)
    
    # Short reversal: negative = recent underperformance = model expects bounce
    mom_5d = bkng_features.get('tech_mom_5d', 0)
    mom_10d = bkng_features.get('tech_mom_10d', 0)
    
    print(f"   5-day momentum: {mom_5d:+.3f}")
    print(f"   10-day momentum: {mom_10d:+.3f}")
    
    if mom_5d < 0:
        print("   -> BKNG has recent underperformance")
        print("   -> Short-reversal strategy EXPECTS a bounce")
        print("   -> This is why ML model scores it HIGH")
    else:
        print("   -> BKNG has recent outperformance")
    
    # Check residual momentum features
    resid_mom_cols = [c for c in bkng_features.index if 'resid_mom' in c]
    if resid_mom_cols:
        print(f"\n   Residual momentum features:")
        for col in resid_mom_cols:
            val = bkng_features[col]
            print(f"     {col}: {val:+.3f}")
    
    # Correlation analysis
    print("\n7. Feature-Target Correlations (using training data):")
    print("-"*50)
    
    # Align features and targets
    common_idx = features_long.dropna().index.intersection(targets_stacked.dropna().index)
    X = features_long.loc[common_idx]
    y = targets_stacked.loc[common_idx]
    
    # Calculate correlations for each feature
    correlations = {}
    for col in X.columns:
        if X[col].std() > 0:
            corr = X[col].corr(y)
            correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\n   Top 10 features by correlation with forward returns:")
    for feat, corr in sorted_corrs[:10]:
        print(f"     {feat:<35}: {corr:+.3f}")
    
    print("\n   Bottom 5 features (negative correlation):")
    for feat, corr in sorted_corrs[-5:]:
        print(f"     {feat:<35}: {corr:+.3f}")
    
    # Check if short reversal is predictive
    print("\n8. Short Reversal Feature Analysis:")
    print("-"*50)
    
    if 'tech_mom_5d' in X.columns:
        mom5_corr = X['tech_mom_5d'].corr(y)
        print(f"   5d momentum correlation with 21d fwd return: {mom5_corr:+.3f}")
        
        if mom5_corr < 0:
            print("   -> NEGATIVE correlation: Recent losers outperform!")
            print("   -> Short-reversal strategy SHOULD work")
        else:
            print("   -> POSITIVE correlation: Momentum continues!")
            print("   -> Short-reversal strategy may NOT work currently")
    
    if 'tech_mom_10d' in X.columns:
        mom10_corr = X['tech_mom_10d'].corr(y)
        print(f"   10d momentum correlation with 21d fwd return: {mom10_corr:+.3f}")


def analyze_regime():
    """Check if we're in a momentum or reversal regime."""
    
    print("\n" + "="*70)
    print("MARKET REGIME ANALYSIS")
    print("="*70)
    
    # Get broad market data
    tickers = ["SPY", "QQQ", "IWM"]
    end = datetime.now()
    start = end - timedelta(days=365)
    
    prices = yf.download(tickers, start=start, end=end, progress=False)['Close']
    
    # Calculate rolling correlations between past and future returns
    for ticker in tickers:
        p = prices[ticker]
        
        # Past 21d return
        past_ret = p.pct_change(21)
        
        # Future 21d return (shifted back)
        future_ret = p.pct_change(21).shift(-21)
        
        # Rolling correlation (60 day window)
        rolling_corr = past_ret.rolling(60).corr(future_ret.shift(21))
        
        recent_corr = rolling_corr.dropna().tail(20).mean()
        
        print(f"\n{ticker}:")
        print(f"   Recent past-future return correlation: {recent_corr:+.3f}")
        
        if recent_corr > 0.1:
            print(f"   -> MOMENTUM regime (trends continue)")
        elif recent_corr < -0.1:
            print(f"   -> REVERSAL regime (mean reversion)")
        else:
            print(f"   -> NEUTRAL regime (no clear pattern)")


if __name__ == "__main__":
    analyze_model_training()
    analyze_regime()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The ML model's poor performance is likely due to:

1. SHORT TRAINING PERIOD: Only ~22 days of training data
   - Model has very limited signal to learn from
   - Prone to overfitting to recent patterns

2. REGIME MISMATCH: Short-term reversal may not be working now
   - If we're in a momentum regime, reversal bets fail
   - BKNG's underperformance continues instead of bouncing

3. FEATURE-TARGET MISALIGNMENT: 
   - Model trained on vol-normalized returns
   - Predictions may not align with raw returns we're measuring

RECOMMENDATIONS:
1. Use longer training period (252+ days)
2. Add regime detection to switch between momentum/reversal
3. Reduce weight of ML model in composite scoring
4. Consider ensemble of reversal + momentum features
""")
