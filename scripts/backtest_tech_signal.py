"""
Backtest Emerging Tech Signals.

Combines:
1. Earnings call tech intensity (QoQ changes)
2. 10-K filing tech language (YoY changes)

Tests if companies with accelerating tech focus outperform.

Signal Hypothesis:
    - Companies increasing AI/ML language = investing ahead of curve
    - Companies with strong moat language = sustainable advantage
    - QoQ increases in tech intensity = momentum signal
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sqlite3

import pandas as pd
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_researcher.models.earnings_tech_signal import EarningsCallTechModel
from auto_researcher.models.filing_tech_signal import FilingTechSignal

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TechSignal:
    """Combined tech signal for a ticker."""
    ticker: str
    date: datetime
    
    # Earnings call metrics
    earnings_tech_intensity: float = 0.0
    earnings_sentiment: float = 0.0
    earnings_qoq_change: Optional[float] = None
    
    # Filing metrics
    filing_tech_intensity: float = 0.0
    filing_moat_strength: float = 0.0
    
    # Combined signal
    signal_score: float = 0.0
    signal_rank: str = "neutral"


def get_price_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Get price data for a ticker from Polygon."""
    import requests
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return None
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {"apiKey": api_key, "adjusted": "true", "limit": 50000}
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results"):
                df = pd.DataFrame(data["results"])
                df["date"] = pd.to_datetime(df["t"], unit="ms")
                df = df.rename(columns={"c": "close", "o": "open", "h": "high", "l": "low", "v": "volume"})
                return df[["date", "open", "high", "low", "close", "volume"]].set_index("date")
    except Exception as e:
        logger.error(f"Failed to get prices for {ticker}: {e}")
    
    return None


def calculate_forward_returns(prices: pd.DataFrame, signal_date: datetime, periods: List[int] = [5, 10, 21]) -> Dict[int, float]:
    """Calculate forward returns from signal date."""
    returns = {}
    
    # Find nearest trading day
    try:
        idx = prices.index.get_indexer([signal_date], method="nearest")[0]
        if idx < 0 or idx >= len(prices):
            return {}
        
        signal_price = prices.iloc[idx]["close"]
        
        for period in periods:
            if idx + period < len(prices):
                future_price = prices.iloc[idx + period]["close"]
                returns[period] = (future_price - signal_price) / signal_price
    except:
        pass
    
    return returns


def analyze_ticker(ticker: str, earnings_model: EarningsCallTechModel) -> Optional[TechSignal]:
    """Analyze a single ticker for tech signals."""
    
    try:
        # Get earnings call analysis
        earnings = earnings_model.analyze_company(ticker, quarters=4)
        
        if not earnings.current:
            return None
        
        # Get date - current.date is already a datetime object
        signal_date = earnings.current.date
        if isinstance(signal_date, str):
            signal_date = datetime.strptime(signal_date.split()[0], "%Y-%m-%d")
        
        # Build signal
        signal = TechSignal(
            ticker=ticker,
            date=signal_date,
            earnings_tech_intensity=earnings.current.tech_intensity,
            earnings_sentiment=earnings.current.tech_sentiment,
            earnings_qoq_change=earnings.qoq_tech_change,
        )
        
        # Calculate signal score
        score = 0.0
        
        # Tech intensity contribution
        if signal.earnings_tech_intensity > 20:
            score += 0.3
        elif signal.earnings_tech_intensity > 10:
            score += 0.1
        
        # Sentiment contribution
        if signal.earnings_sentiment > 0.5:
            score += 0.2
        elif signal.earnings_sentiment > 0:
            score += 0.1
        elif signal.earnings_sentiment < -0.3:
            score -= 0.2
        
        # QoQ change contribution (momentum)
        if signal.earnings_qoq_change is not None:
            if signal.earnings_qoq_change > 0.2:  # 20% increase
                score += 0.3
            elif signal.earnings_qoq_change > 0:
                score += 0.1
            elif signal.earnings_qoq_change < -0.2:  # 20% decrease
                score -= 0.2
        
        signal.signal_score = score
        
        if score > 0.4:
            signal.signal_rank = "strong_buy"
        elif score > 0.2:
            signal.signal_rank = "buy"
        elif score < -0.2:
            signal.signal_rank = "sell"
        else:
            signal.signal_rank = "neutral"
        
        return signal
        
    except Exception as e:
        logger.debug(f"Error analyzing {ticker}: {e}")
        return None


def run_backtest():
    """Run backtest on tech signals."""
    
    print("=" * 60)
    print("EMERGING TECH SIGNAL BACKTEST")
    print("=" * 60)
    
    # Test tickers - mix of tech and non-tech
    tickers = [
        # Tech leaders
        "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN",
        # Semiconductors
        "AMD", "INTC", "AVGO", "QCOM",
        # Software
        "CRM", "ADBE", "ORCL",
        # Non-tech for comparison
        "JPM", "BAC", "JNJ", "PG", "KO", "WMT",
    ]
    
    print(f"\nAnalyzing {len(tickers)} tickers...")
    print("This will download earnings call transcripts (may take a few minutes)...")
    print()
    
    earnings_model = EarningsCallTechModel()
    
    signals = []
    
    for ticker in tickers:
        print(f"  Analyzing {ticker}...", end=" ", flush=True)
        signal = analyze_ticker(ticker, earnings_model)
        if signal:
            signals.append(signal)
            print(f"Tech: {signal.earnings_tech_intensity:.1f}, Signal: {signal.signal_score:+.2f}")
        else:
            print("No data")
    
    if not signals:
        print("\nNo signals generated. Check data availability.")
        return
    
    print()
    print("=" * 60)
    print("SIGNAL SUMMARY")
    print("=" * 60)
    
    # Sort by signal score
    signals.sort(key=lambda x: -x.signal_score)
    
    print(f"\n{'Ticker':<8} {'Tech Int':>10} {'Sentiment':>10} {'QoQ Chg':>10} {'Score':>8} {'Rank':>12}")
    print("-" * 60)
    
    for s in signals:
        qoq = f"{s.earnings_qoq_change:+.1%}" if s.earnings_qoq_change else "N/A"
        print(f"{s.ticker:<8} {s.earnings_tech_intensity:>10.1f} {s.earnings_sentiment:>+10.2f} {qoq:>10} {s.signal_score:>+8.2f} {s.signal_rank:>12}")
    
    # Group analysis
    print()
    print("=" * 60)
    print("GROUP ANALYSIS")
    print("=" * 60)
    
    high_tech = [s for s in signals if s.earnings_tech_intensity > 15]
    low_tech = [s for s in signals if s.earnings_tech_intensity < 5]
    
    print(f"\nHigh Tech Intensity (>15): {len(high_tech)} companies")
    if high_tech:
        print(f"  Average Sentiment: {np.mean([s.earnings_sentiment for s in high_tech]):+.2f}")
        print(f"  Companies: {', '.join([s.ticker for s in high_tech])}")
    
    print(f"\nLow Tech Intensity (<5): {len(low_tech)} companies")
    if low_tech:
        print(f"  Average Sentiment: {np.mean([s.earnings_sentiment for s in low_tech]):+.2f}")
        print(f"  Companies: {', '.join([s.ticker for s in low_tech])}")
    
    # Strong signals
    strong_buy = [s for s in signals if s.signal_rank == "strong_buy"]
    sell = [s for s in signals if s.signal_rank == "sell"]
    
    print()
    print("=" * 60)
    print("TRADING SIGNALS")
    print("=" * 60)
    
    if strong_buy:
        print(f"\n🚀 STRONG BUY: {', '.join([s.ticker for s in strong_buy])}")
        for s in strong_buy:
            print(f"   {s.ticker}: Tech intensity {s.earnings_tech_intensity:.1f}, Sentiment {s.earnings_sentiment:+.2f}")
    
    if sell:
        print(f"\n📉 SELL/AVOID: {', '.join([s.ticker for s in sell])}")
    
    # Forward return analysis (if we have price data)
    print()
    print("=" * 60)
    print("FORWARD RETURN ANALYSIS")
    print("=" * 60)
    
    returns_data = []
    
    for s in signals[:10]:  # Top 10 by signal
        prices = get_price_data(s.ticker, s.date - timedelta(days=5), datetime.now())
        if prices is not None and len(prices) > 20:
            returns = calculate_forward_returns(prices, s.date)
            if returns:
                returns_data.append({
                    "ticker": s.ticker,
                    "signal_score": s.signal_score,
                    "tech_intensity": s.earnings_tech_intensity,
                    **{f"ret_{k}d": v for k, v in returns.items()}
                })
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        print(f"\nForward returns from signal date (top signals):")
        print(returns_df.to_string(index=False, float_format=lambda x: f"{x:.2%}" if abs(x) < 1 else f"{x:.2f}"))
        
        # Correlation analysis
        print("\n\nSignal Score vs Forward Returns Correlation:")
        for col in returns_df.columns:
            if col.startswith("ret_"):
                corr, pval = stats.pearsonr(returns_df["signal_score"], returns_df[col])
                print(f"  {col}: r={corr:.3f}, p={pval:.3f}")
    else:
        print("\nCould not fetch price data for return analysis.")


if __name__ == "__main__":
    run_backtest()
