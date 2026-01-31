#!/usr/bin/env python3
"""
Simple IC calculation for Early Adopter Model.
Uses point-in-time returns vs pioneer scores.
"""

import sys
sys.path.insert(0, "c:/Users/Andrew/projects/auto_researcher/src")

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from auto_researcher.models.early_adopter import EarlyAdopterModel

# Tech universe
TICKERS = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", 
    "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", 
    "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PLTR",
    "IBM", "CSCO", "DELL", "NET", "DDOG", "ZS",
    "CRWD", "PANW", "FTNT", "WDAY", "TEAM"
]


def main():
    print("=" * 60)
    print("EARLY ADOPTER MODEL - IC ANALYSIS")
    print("=" * 60)
    
    # Step 1: Get pioneer signals
    print("\n📊 Getting pioneer signals...")
    model = EarlyAdopterModel()
    
    data = []
    for ticker in TICKERS:
        try:
            signal = model.analyze_company(ticker)
            if signal:
                data.append({
                    "ticker": ticker,
                    "pioneer_score": signal.pioneer_score,
                    "techs": signal.total_techs_adopted,
                    "genai": signal.genai_score,
                })
        except:
            pass
    
    df = pd.DataFrame(data).set_index("ticker")
    print(f"   Got {len(df)} companies")
    
    # Step 2: Get returns
    print("\n📅 Fetching 6-month returns...")
    
    prices = yf.download(
        list(df.index), 
        start="2024-07-01", 
        end="2026-01-31",
        progress=False
    )["Adj Close"]
    
    # Calculate 6-month return
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    ret_6mo = (end_price / start_price - 1)
    
    df["return_6mo"] = ret_6mo
    df = df.dropna()
    
    print(f"   {len(df)} companies with price data")
    print(f"   Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Step 3: Calculate IC
    print("\n" + "=" * 60)
    print("📈 INFORMATION COEFFICIENT")
    print("=" * 60)
    
    ic, pval = stats.spearmanr(df["pioneer_score"], df["return_6mo"])
    
    print(f"\n   Pioneer Score IC: {ic:+.4f} (p={pval:.4f})")
    
    # Also test subcomponents
    ic_techs, p_techs = stats.spearmanr(df["techs"], df["return_6mo"])
    ic_genai, p_genai = stats.spearmanr(df["genai"], df["return_6mo"])
    
    print(f"   Techs Adopted IC: {ic_techs:+.4f} (p={p_techs:.4f})")
    print(f"   GenAI Score IC:   {ic_genai:+.4f} (p={p_genai:.4f})")
    
    # Step 4: Quintile analysis
    print("\n" + "=" * 60)
    print("📊 QUINTILE SPREAD")
    print("=" * 60)
    
    df["q"] = pd.qcut(df["pioneer_score"].rank(method="first"), 5, labels=[1,2,3,4,5])
    
    print(f"\n{'Quintile':<10} {'Avg Return':>12} {'Avg Pioneer':>12} {'N':>5}")
    print("-" * 42)
    
    for q in [1, 2, 3, 4, 5]:
        sub = df[df["q"] == q]
        label = "Low" if q == 1 else ("High" if q == 5 else f"Q{q}")
        print(f"Q{q} ({label:<4}) {sub['return_6mo'].mean():>+12.1%} {sub['pioneer_score'].mean():>12.2f} {len(sub):>5}")
    
    q5 = df[df["q"] == 5]["return_6mo"].mean()
    q1 = df[df["q"] == 1]["return_6mo"].mean()
    spread = q5 - q1
    
    print("-" * 42)
    print(f"{'Q5-Q1 Spread:':<22} {spread:>+12.1%}")
    
    # Step 5: Show data
    print("\n" + "=" * 60)
    print("📊 COMPANY DETAILS")
    print("=" * 60)
    
    df_sorted = df.sort_values("pioneer_score", ascending=False)
    
    print(f"\n{'Ticker':<8} {'Pioneer':>8} {'Techs':>6} {'GenAI':>8} {'6mo Ret':>10}")
    print("-" * 44)
    for idx, row in df_sorted.iterrows():
        print(f"{idx:<8} {row['pioneer_score']:>8.2f} {row['techs']:>6.0f} {row['genai']:>8.0f} {row['return_6mo']:>+10.1%}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    print(f"""
Early Adopter Model IC Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • IC (Spearman): {ic:+.4f}
  • p-value: {pval:.4f}
  • Quintile Spread (Q5-Q1): {spread:+.1%}
  
Benchmarks:
  • IC ≥ 0.02 = useful signal
  • IC ≥ 0.05 = strong signal
  • p-value < 0.1 = statistically significant
""")
    
    if ic >= 0.05 and pval < 0.1:
        print("  ✅ STRONG, SIGNIFICANT ALPHA")
    elif ic >= 0.02:
        print(f"  ⚡ USEFUL SIGNAL (IC={ic:.3f})")
    elif ic > 0:
        print("  ⚠️ WEAK POSITIVE SIGNAL")
    else:
        print("  ❌ NO PREDICTIVE POWER")


if __name__ == "__main__":
    main()
