"""
ML Ranking Model Backtest and Analysis Script

Analyzes why certain stocks score high and validates model predictions.
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Add both src and project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_researcher.features.enhanced import (
    EnhancedFeatureConfig,
    compute_all_enhanced_features,
)


def analyze_stock_performance(ticker: str, days: int = 365):
    """Analyze a stock's historical performance."""
    print(f"\n{'='*60}")
    print(f"PRICE PERFORMANCE ANALYSIS: {ticker}")
    print('='*60)
    
    end = datetime.now()
    start = end - timedelta(days=days)
    
    df = yf.download([ticker, 'SPY'], start=start, end=end, progress=False)['Close']
    
    if df.empty:
        print("Failed to fetch data")
        return
    
    stock = df[ticker]
    spy = df['SPY']
    
    # Calculate returns at different horizons
    periods = [
        ("1 Year", 0, -1),
        ("6 Months", len(stock)//2, -1),
        ("3 Months", int(len(stock)*0.75), -1),
        ("1 Month", -22, -1),
        ("1 Week", -5, -1),
    ]
    
    print(f"\n{'Period':<12} | {ticker:>8} | {'SPY':>8} | {'Excess':>8}")
    print("-" * 45)
    
    for name, start_idx, end_idx in periods:
        try:
            stock_ret = (stock.iloc[end_idx] / stock.iloc[start_idx] - 1) * 100
            spy_ret = (spy.iloc[end_idx] / spy.iloc[start_idx] - 1) * 100
            excess = stock_ret - spy_ret
            print(f"{name:<12} | {stock_ret:>7.1f}% | {spy_ret:>7.1f}% | {excess:>+7.1f}%")
        except Exception as e:
            print(f"{name:<12} | Error: {e}")
    
    # Recent price action
    print(f"\nCurrent price: ${stock.iloc[-1]:.2f}")
    print(f"52-week high: ${stock.max():.2f}")
    print(f"52-week low: ${stock.min():.2f}")
    
    # Calculate short-term reversal (key feature)
    ret_1w = (stock.iloc[-1] / stock.iloc[-5] - 1) * 100
    ret_1m = (stock.iloc[-1] / stock.iloc[-22] - 1) * 100
    
    print(f"\nShort-term momentum:")
    print(f"  1-week return: {ret_1w:+.1f}%")
    print(f"  1-month return: {ret_1m:+.1f}%")
    
    if ret_1m < -5:
        print("  -> Recent UNDERPERFORMANCE may trigger short-reversal signal!")


def analyze_ml_features(tickers: list[str]):
    """Analyze ML features for given tickers."""
    print(f"\n{'='*60}")
    print("ML FEATURE ANALYSIS")
    print('='*60)
    
    # Fetch prices
    end = datetime.now()
    start = end - timedelta(days=504)  # 2 years
    
    df = yf.download(tickers + ['SPY'], start=start, end=end, progress=False)['Close']
    
    # Compute features
    config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,
        use_cross_sec_norm=True,
        cross_sec_norm_robust=True,
    )
    
    features_wide = compute_all_enhanced_features(df, benchmark='SPY', config=config)
    features_long = features_wide.stack(level=0, future_stack=True)
    features_long.index.names = ["date", "ticker"]
    
    # Get latest features
    latest_date = features_long.index.get_level_values("date").max()
    latest = features_long.loc[latest_date]
    
    print(f"\nLatest date: {latest_date}")
    print(f"\nFeature values for each ticker:")
    print("-" * 80)
    
    # Show key features
    key_features = [col for col in latest.columns if any(x in col.lower() for x in ['reversal', 'mom', 'vol', 'mad'])][:10]
    
    for ticker in tickers:
        if ticker in latest.index and ticker != 'SPY':
            print(f"\n{ticker}:")
            for feat in key_features:
                val = latest.loc[ticker, feat]
                print(f"  {feat}: {val:+.3f}")


def backtest_ml_predictions(universe: list[str], lookback_months: int = 6):
    """
    Backtest: Compare ML predictions vs actual forward returns.
    
    This simulates what the ML model would have predicted historically
    and checks if high-scoring stocks actually outperformed.
    """
    print(f"\n{'='*60}")
    print(f"ML MODEL BACKTEST ({lookback_months} months)")
    print('='*60)
    
    from recommend import generate_recommendations, fetch_prices
    from auto_researcher.features.targets import TargetConfig, build_targets
    
    # Get current predictions
    print("\n1. Getting current ML predictions...")
    recommendations, scores_df, prices = generate_recommendations(
        tickers=universe,
        top_k=len(universe),  # Get all
        explain=False,
    )
    
    # Get recent forward returns (21d)
    print("\n2. Computing recent forward returns...")
    target_config = TargetConfig(mode="raw", horizon_days=21)
    targets = build_targets(prices, target_config, benchmark='SPY')
    
    # Get returns from ~21 days ago
    check_date = targets.index[-22]  # ~1 month ago
    actual_returns = prices.iloc[-1] / prices.loc[check_date] - 1
    
    # Compare predictions vs actuals
    print("\n3. Comparing predictions to actual returns...")
    
    results = []
    for rec in recommendations:
        if rec.ticker in actual_returns.index:
            actual_ret = actual_returns[rec.ticker] * 100
            results.append({
                'ticker': rec.ticker,
                'ml_score': rec.score,
                'ml_rank': rec.rank,
                'actual_return_21d': actual_ret,
            })
    
    results_df = pd.DataFrame(results)
    
    # Sort by ML score
    results_df = results_df.sort_values('ml_score', ascending=False)
    
    print(f"\n{'Ticker':<8} | {'ML Score':>9} | {'ML Rank':>8} | {'Actual 21d':>10}")
    print("-" * 45)
    
    for _, row in results_df.head(10).iterrows():
        print(f"{row['ticker']:<8} | {row['ml_score']:>9.3f} | {row['ml_rank']:>8} | {row['actual_return_21d']:>+9.1f}%")
    
    # Correlation analysis
    corr = results_df['ml_score'].corr(results_df['actual_return_21d'])
    print(f"\nCorrelation (ML score vs 21d return): {corr:.3f}")
    
    # Top/bottom quintile analysis
    n = len(results_df)
    quintile = n // 5
    
    top_quintile = results_df.head(quintile)['actual_return_21d'].mean()
    bottom_quintile = results_df.tail(quintile)['actual_return_21d'].mean()
    spread = top_quintile - bottom_quintile
    
    print(f"\nQuintile Analysis:")
    print(f"  Top 20% avg return:    {top_quintile:+.2f}%")
    print(f"  Bottom 20% avg return: {bottom_quintile:+.2f}%")
    print(f"  Long/Short Spread:     {spread:+.2f}%")
    
    if spread > 0:
        print("  -> Model has POSITIVE predictive power!")
    else:
        print("  -> WARNING: Model may not be predictive!")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Model Backtest")
    parser.add_argument("--ticker", type=str, help="Analyze specific ticker")
    parser.add_argument("--backtest", action="store_true", help="Run full backtest")
    parser.add_argument("--universe", default="sp100", choices=["sp100", "core_tech"])
    
    args = parser.parse_args()
    
    if args.ticker:
        analyze_stock_performance(args.ticker)
    
    if args.backtest:
        from recommend import UNIVERSES
        universe = UNIVERSES[args.universe]() if callable(UNIVERSES[args.universe]) else UNIVERSES[args.universe]
        backtest_ml_predictions(universe[:50])  # Limit for speed
    
    if not args.ticker and not args.backtest:
        # Default: analyze BKNG which scored high
        analyze_stock_performance("BKNG")
        analyze_stock_performance("GOOGL")
        
        # Quick feature comparison
        print("\n" + "="*60)
        print("TOP 5 vs BOTTOM 5 from sp100 run")
        print("="*60)
        
        # Load the results
        results_file = Path("data/ranking_results/final_ranking_sp100_20260202_1210.json")
        if results_file.exists():
            import json
            with open(results_file) as f:
                results = json.load(f)
            
            top5 = [r['ticker'] for r in results[:5]]
            bottom5 = [r['ticker'] for r in results[-5:]]
            
            print(f"\nTop 5: {top5}")
            print(f"Bottom 5: {bottom5}")
            
            for ticker in top5 + bottom5:
                analyze_stock_performance(ticker)
