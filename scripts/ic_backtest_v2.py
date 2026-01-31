#!/usr/bin/env python3
"""
Backtest the Early Adopter Model to calculate Information Coefficient (IC).

IC = Spearman rank correlation between signal and forward returns.
"""

import sys
sys.path.insert(0, "c:/Users/Andrew/projects/auto_researcher/src")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import yfinance as yf
import logging

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

from auto_researcher.models.early_adopter import EarlyAdopterModel

# Tech universe for testing
TECH_UNIVERSE = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", 
    "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PLTR",
    "IBM", "CSCO", "DELL", "HPQ", "NET", "DDOG", "ZS",
    "CRWD", "PANW", "FTNT", "WDAY", "TEAM"
]


def run_ic_backtest():
    """Run the IC backtest for the Early Adopter model."""
    
    print("=" * 70)
    print("EARLY ADOPTER MODEL - INFORMATION COEFFICIENT BACKTEST")
    print("=" * 70)
    
    # Initialize model
    model = EarlyAdopterModel()
    
    # Get pioneer signals for each company
    print("\n📊 Generating pioneer signals...")
    
    signals = []
    for ticker in TECH_UNIVERSE:
        try:
            signal = model.analyze_company(ticker)
            if signal:
                signals.append({
                    "ticker": ticker,
                    "pioneer_score": signal.pioneer_score,
                    "techs_adopted": signal.total_techs_adopted,
                    "techs_early": signal.techs_adopted_early,
                    "avg_lead_months": signal.avg_lead_months,
                    "genai_score": signal.genai_score,
                    "signal": signal.signal,
                })
        except Exception as e:
            print(f"   Failed to analyze {ticker}: {e}")
    
    signals_df = pd.DataFrame(signals)
    print(f"   Got signals for {len(signals_df)} companies")
    
    # Get historical prices for multiple time periods
    print("\n📅 Fetching price data...")
    
    # Fetch all prices at once
    all_tickers = signals_df["ticker"].tolist()
    prices = yf.download(
        all_tickers,
        start="2023-01-01",
        end="2026-01-31",
        progress=False
    )
    
    # Handle MultiIndex columns
    if isinstance(prices.columns, pd.MultiIndex):
        adj_close = prices["Adj Close"]
    else:
        adj_close = prices
    
    print(f"   Got {len(adj_close)} trading days of data")
    
    # Calculate returns at different forward horizons
    horizons = {
        "1mo": 21,
        "3mo": 63,
        "6mo": 126,
        "12mo": 252,
    }
    
    # Collect IC at different horizons
    ic_results = {}
    
    for horizon_name, horizon_days in horizons.items():
        # Calculate forward returns
        fwd_returns = adj_close.shift(-horizon_days) / adj_close - 1
        
        # Use specific test dates (start of each quarter)
        test_dates = pd.date_range("2023-03-01", "2025-06-01", freq="QS")
        
        ics = []
        for test_date in test_dates:
            # Find nearest trading day
            nearest = adj_close.index[adj_close.index >= test_date]
            if len(nearest) == 0:
                continue
            date = nearest[0]
            
            if date not in fwd_returns.index:
                continue
            
            # Get forward returns at this date
            fwd = fwd_returns.loc[date]
            
            # Merge with signals
            merged = signals_df.set_index("ticker")[["pioneer_score"]].copy()
            merged["fwd_return"] = fwd
            merged = merged.dropna()
            
            if len(merged) < 10:
                continue
            
            # Calculate Spearman IC
            ic, pval = stats.spearmanr(merged["pioneer_score"], merged["fwd_return"])
            
            if not np.isnan(ic):
                ics.append(ic)
        
        if ics:
            ic_results[horizon_name] = {
                "mean": np.mean(ics),
                "std": np.std(ics),
                "n_obs": len(ics),
                "ics": ics,
            }
    
    # Print IC results
    print("\n" + "=" * 70)
    print("📈 INFORMATION COEFFICIENT RESULTS")
    print("=" * 70)
    
    print(f"\n{'Horizon':<10} {'Mean IC':>10} {'Std IC':>10} {'t-stat':>10} {'p-value':>10} {'IR':>10} {'N obs':>10}")
    print("-" * 70)
    
    for horizon_name in ["1mo", "3mo", "6mo", "12mo"]:
        if horizon_name not in ic_results:
            print(f"{horizon_name:<10} {'N/A':>10}")
            continue
            
        r = ic_results[horizon_name]
        mean_ic = r["mean"]
        std_ic = r["std"]
        n_obs = r["n_obs"]
        
        if std_ic > 0 and n_obs > 1:
            t_stat = mean_ic / (std_ic / np.sqrt(n_obs))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 1))
            ir = mean_ic / std_ic
        else:
            t_stat = 0
            p_value = 1
            ir = 0
        
        sig = "*" if p_value < 0.1 else ""
        print(f"{horizon_name:<10} {mean_ic:>+10.4f} {std_ic:>10.4f} {t_stat:>10.2f} {p_value:>10.4f} {ir:>10.3f} {n_obs:>10} {sig}")
    
    # Cross-sectional IC for most recent period
    print("\n" + "=" * 70)
    print("📊 CROSS-SECTIONAL IC (Latest 6-month returns)")
    print("=" * 70)
    
    # Get 6-month returns ending now
    latest_date = adj_close.index[-1]
    start_date = adj_close.index[adj_close.index <= latest_date - timedelta(days=126)][-1]
    
    returns_6mo = (adj_close.loc[latest_date] / adj_close.loc[start_date] - 1)
    
    merged = signals_df.set_index("ticker").copy()
    merged["return_6mo"] = returns_6mo
    merged = merged.dropna(subset=["return_6mo"])
    
    print(f"\nTest period: {start_date.date()} to {latest_date.date()}")
    print(f"N companies: {len(merged)}")
    
    print(f"\n{'Component':<20} {'IC (Spearman)':>15} {'p-value':>10} {'Sig':>6}")
    print("-" * 55)
    
    components = [
        ("pioneer_score", "Pioneer Score"),
        ("techs_adopted", "# Techs Adopted"),
        ("techs_early", "# Techs Early"),
        ("avg_lead_months", "Avg Lead Months"),
        ("genai_score", "GenAI Score"),
    ]
    
    for col, name in components:
        if col in merged.columns:
            subset = merged[[col, "return_6mo"]].dropna()
            if len(subset) >= 10:
                ic, pval = stats.spearmanr(subset[col], subset["return_6mo"])
                sig = "✓" if pval < 0.1 else ("·" if pval < 0.2 else "")
                print(f"{name:<20} {ic:>+15.4f} {pval:>10.4f} {sig:>6}")
    
    # Quintile analysis
    print("\n" + "=" * 70)
    print("📊 QUINTILE ANALYSIS (6-month returns)")
    print("=" * 70)
    
    try:
        merged["quintile"] = pd.qcut(
            merged["pioneer_score"].rank(method="first"), 
            5, 
            labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"]
        )
        
        print(f"\n{'Quintile':<12} {'Avg Return':>12} {'Avg Pioneer':>12} {'Count':>8}")
        print("-" * 48)
        
        for q in ["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"]:
            subset = merged[merged["quintile"] == q]
            if len(subset) > 0:
                avg_ret = subset["return_6mo"].mean()
                avg_pioneer = subset["pioneer_score"].mean()
                print(f"{q:<12} {avg_ret:>+12.1%} {avg_pioneer:>12.2f} {len(subset):>8}")
        
        # Long-short spread
        q5_ret = merged[merged["quintile"] == "Q5 (High)"]["return_6mo"].mean()
        q1_ret = merged[merged["quintile"] == "Q1 (Low)"]["return_6mo"].mean()
        spread = q5_ret - q1_ret
        
        print("-" * 48)
        print(f"{'Q5 - Q1 Spread:':<24} {spread:>+12.1%}")
        
    except Exception as e:
        print(f"   Quintile analysis failed: {e}")
        spread = 0
    
    # Top/Bottom performers
    print("\n" + "=" * 70)
    print("📊 TOP & BOTTOM PERFORMERS")
    print("=" * 70)
    
    merged_sorted = merged.sort_values("return_6mo", ascending=False)
    
    print("\n🏆 TOP 5 (by 6mo return):")
    print(f"{'Ticker':<8} {'Return':>10} {'Pioneer':>10} {'Signal':>12}")
    print("-" * 44)
    for _, row in merged_sorted.head(5).iterrows():
        print(f"{row.name:<8} {row['return_6mo']:>+10.1%} {row['pioneer_score']:>10.2f} {row['signal']:>12}")
    
    print("\n📉 BOTTOM 5 (by 6mo return):")
    print(f"{'Ticker':<8} {'Return':>10} {'Pioneer':>10} {'Signal':>12}")
    print("-" * 44)
    for _, row in merged_sorted.tail(5).iterrows():
        print(f"{row.name:<8} {row['return_6mo']:>+10.1%} {row['pioneer_score']:>10.2f} {row['signal']:>12}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    # Find best horizon
    if ic_results:
        best_horizon = max(ic_results.keys(), key=lambda x: abs(ic_results[x]["mean"]))
        best = ic_results[best_horizon]
        
        # Calculate p-value
        if best["std"] > 0 and best["n_obs"] > 1:
            t_stat = best["mean"] / (best["std"] / np.sqrt(best["n_obs"]))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), best["n_obs"] - 1))
        else:
            t_stat = 0
            p_val = 1
        
        print(f"""
Early Adopter Model - IC Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best horizon: {best_horizon}
  • Mean IC: {best['mean']:+.4f}
  • IC Std: {best['std']:.4f}
  • t-stat: {t_stat:.2f}
  • N observations: {best['n_obs']}

Quintile Spread (Q5-Q1): {spread:+.1%}

Benchmark:
  • IC ≥ 0.02 is considered useful
  • IC ≥ 0.05 is considered strong
  • t-stat ≥ 2 is statistically significant

Verdict: """)
        
        if best["mean"] >= 0.05 and p_val < 0.1:
            print("  ✅ STRONG, SIGNIFICANT ALPHA")
        elif best["mean"] >= 0.02:
            print("  ⚡ USEFUL SIGNAL (needs more data for significance)")
        elif best["mean"] > 0:
            print("  ⚠️ WEAK POSITIVE SIGNAL")
        else:
            print("  ❌ NO PREDICTIVE POWER")
    
    return ic_results


if __name__ == "__main__":
    run_ic_backtest()
