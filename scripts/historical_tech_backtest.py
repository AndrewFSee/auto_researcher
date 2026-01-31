"""
Historical Backtest of Emerging Tech Signals.

Uses earnings call transcripts from 2022-2023 to test if
tech intensity predicts forward returns.

This is a proper historical backtest - we use past transcript dates
and measure actual forward returns that occurred.
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import tempfile
import os

import pandas as pd
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_researcher.models.earnings_tech_signal import (
    EarningsCallTechModel,
    TRANSCRIPT_CACHE_PATH,
)

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HistoricalSignal:
    """Signal from a specific earnings call."""
    ticker: str
    date: datetime
    quarter: str
    tech_intensity: float
    sentiment: float
    qoq_change: Optional[float]
    top_areas: List[str]


def get_price_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Get price data for a ticker from Polygon."""
    import requests
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


def get_forward_returns(prices: pd.DataFrame, signal_date: datetime) -> Dict[int, float]:
    """Calculate forward returns from signal date."""
    returns = {}
    periods = [5, 10, 21, 63]  # 1wk, 2wk, 1mo, 3mo
    
    try:
        # Find nearest trading day after signal
        future_dates = prices.index[prices.index >= signal_date]
        if len(future_dates) == 0:
            return {}
        
        start_idx = prices.index.get_loc(future_dates[0])
        signal_price = prices.iloc[start_idx]["close"]
        
        for period in periods:
            end_idx = start_idx + period
            if end_idx < len(prices):
                future_price = prices.iloc[end_idx]["close"]
                returns[period] = (future_price - signal_price) / signal_price
    except:
        pass
    
    return returns


def run_historical_backtest():
    """Run backtest using 2023 earnings calls."""
    
    print("=" * 70)
    print("HISTORICAL TECH SIGNAL BACKTEST (2023 Earnings Calls)")
    print("=" * 70)
    
    # Load transcript cache
    print("\nLoading transcript data...")
    
    if not os.path.exists(TRANSCRIPT_CACHE_PATH):
        print("No transcript cache found. Run the EarningsCallTechModel first.")
        return
    
    df = pd.read_parquet(
        TRANSCRIPT_CACHE_PATH,
        columns=["symbol", "report_date", "fiscal_quarter", "fiscal_year", "transcripts"],
    )
    print(f"Loaded {len(df):,} transcripts")
    
    # Filter to 2023 Q1-Q4 (real historical data)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df_2023 = df[
        (df["report_date"] >= "2023-01-01") & 
        (df["report_date"] < "2024-01-01")
    ].copy()
    print(f"Found {len(df_2023):,} transcripts from 2023")
    
    # Test tickers
    tickers = [
        # Tech 
        "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN",
        "AMD", "INTC", "AVGO", "QCOM",
        "CRM", "ADBE", "ORCL", "NOW", "SNOW",
        # Non-tech
        "JPM", "BAC", "GS", "MS",
        "JNJ", "PFE", "UNH",
        "PG", "KO", "PEP", "WMT", "TGT",
        # Industrials
        "CAT", "DE", "HON", "GE",
    ]
    
    # Initialize model
    model = EarningsCallTechModel()
    
    # Analyze each ticker's Q3 2023 call (typically Aug-Oct 2023)
    # This gives us enough time to measure 3-month returns
    print("\nAnalyzing Q3 2023 earnings calls...")
    print()
    
    signals = []
    
    for ticker in tickers:
        # Find Q3 2023 transcript
        ticker_df = df_2023[
            (df_2023["symbol"] == ticker) &
            (df_2023["fiscal_quarter"] == 3) &
            (df_2023["fiscal_year"].isin([2023, 2024]))  # Fiscal year might differ
        ].sort_values("report_date", ascending=False)
        
        if ticker_df.empty:
            # Try any 2023 Q3 or Q4 call
            ticker_df = df_2023[
                (df_2023["symbol"] == ticker) &
                (df_2023["report_date"] >= "2023-07-01") &
                (df_2023["report_date"] < "2023-11-01")
            ].sort_values("report_date", ascending=False)
        
        if ticker_df.empty:
            print(f"  {ticker}: No Q3 2023 transcript found")
            continue
        
        row = ticker_df.iloc[0]
        
        # Convert transcript to text
        raw = row.get("transcripts", [])
        if isinstance(raw, list):
            parts = []
            for item in raw:
                if isinstance(item, dict):
                    speaker = item.get("speaker", "")
                    text = item.get("content", "")
                    parts.append(f"{speaker}: {text}")
                else:
                    parts.append(str(item))
            content = "\n".join(parts)
        else:
            content = str(raw)
        
        # Analyze
        try:
            date = row["report_date"].to_pydatetime()
            quarter = f"Q{row['fiscal_quarter']} {row['fiscal_year']}"
            
            analysis = model.analyze_transcript(content, ticker, date, quarter)
            
            # Get QoQ change (need previous quarter)
            prev_ticker_df = df_2023[
                (df_2023["symbol"] == ticker) &
                (df_2023["report_date"] < row["report_date"]) &
                (df_2023["report_date"] >= row["report_date"] - timedelta(days=120))
            ].sort_values("report_date", ascending=False)
            
            qoq_change = None
            if not prev_ticker_df.empty:
                prev_row = prev_ticker_df.iloc[0]
                prev_raw = prev_row.get("transcripts", [])
                if isinstance(prev_raw, list):
                    prev_parts = []
                    for item in prev_raw:
                        if isinstance(item, dict):
                            prev_parts.append(f"{item.get('speaker', '')}: {item.get('content', '')}")
                    prev_content = "\n".join(prev_parts)
                else:
                    prev_content = str(prev_raw)
                
                prev_date = prev_row["report_date"].to_pydatetime()
                prev_quarter = f"Q{prev_row['fiscal_quarter']} {prev_row['fiscal_year']}"
                prev_analysis = model.analyze_transcript(prev_content, ticker, prev_date, prev_quarter)
                
                if prev_analysis.tech_intensity > 0:
                    qoq_change = (analysis.tech_intensity - prev_analysis.tech_intensity) / prev_analysis.tech_intensity
            
            # Top areas
            top_areas = [area for area, _ in sorted(analysis.mentions_by_area.items(), key=lambda x: -x[1])[:3]]
            
            sig = HistoricalSignal(
                ticker=ticker,
                date=date,
                quarter=quarter,
                tech_intensity=analysis.tech_intensity,
                sentiment=analysis.tech_sentiment,
                qoq_change=qoq_change,
                top_areas=top_areas,
            )
            signals.append(sig)
            
            qoq_str = f"{qoq_change:+.1%}" if qoq_change is not None else "N/A"
            print(f"  {ticker}: {date.strftime('%Y-%m-%d')} | Tech: {analysis.tech_intensity:.1f} | QoQ: {qoq_str} | {', '.join(top_areas[:2])}")
            
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
    
    if not signals:
        print("\nNo signals generated.")
        return
    
    # Now get forward returns
    print("\n" + "=" * 70)
    print("FORWARD RETURN ANALYSIS")
    print("=" * 70)
    
    results = []
    
    print("\nFetching price data...")
    for sig in signals:
        # Get prices from signal date to 4 months later
        prices = get_price_data(sig.ticker, sig.date - timedelta(days=5), sig.date + timedelta(days=100))
        
        if prices is None or len(prices) < 30:
            print(f"  {sig.ticker}: No price data")
            continue
        
        returns = get_forward_returns(prices, sig.date)
        
        if returns:
            results.append({
                "ticker": sig.ticker,
                "date": sig.date,
                "tech_intensity": sig.tech_intensity,
                "sentiment": sig.sentiment,
                "qoq_change": sig.qoq_change if sig.qoq_change else 0,
                "top_area": sig.top_areas[0] if sig.top_areas else "",
                **{f"ret_{k}d": v for k, v in returns.items()}
            })
    
    if not results:
        print("\nNo return data available.")
        return
    
    results_df = pd.DataFrame(results)
    
    # Sort by tech intensity
    results_df = results_df.sort_values("tech_intensity", ascending=False)
    
    print(f"\n{'Ticker':<6} {'Date':<12} {'Tech':<8} {'QoQ':<8} {'1wk':<8} {'2wk':<8} {'1mo':<8} {'3mo':<8}")
    print("-" * 74)
    
    for _, row in results_df.iterrows():
        qoq = f"{row['qoq_change']:+.0%}" if row['qoq_change'] != 0 else "N/A"
        r5 = f"{row.get('ret_5d', 0):+.1%}"
        r10 = f"{row.get('ret_10d', 0):+.1%}"
        r21 = f"{row.get('ret_21d', 0):+.1%}"
        r63 = f"{row.get('ret_63d', 0):+.1%}"
        print(f"{row['ticker']:<6} {row['date'].strftime('%Y-%m-%d'):<12} {row['tech_intensity']:>6.1f} {qoq:>8} {r5:>8} {r10:>8} {r21:>8} {r63:>8}")
    
    # Group analysis
    print("\n" + "=" * 70)
    print("QUINTILE ANALYSIS")
    print("=" * 70)
    
    # Split into high/low tech intensity
    median_tech = results_df["tech_intensity"].median()
    high_tech = results_df[results_df["tech_intensity"] >= median_tech]
    low_tech = results_df[results_df["tech_intensity"] < median_tech]
    
    print(f"\nHigh Tech Intensity (>= {median_tech:.1f}): {len(high_tech)} companies")
    for period in ["ret_5d", "ret_10d", "ret_21d", "ret_63d"]:
        if period in high_tech.columns:
            mean_ret = high_tech[period].mean()
            print(f"  {period}: {mean_ret:+.2%}")
    
    print(f"\nLow Tech Intensity (< {median_tech:.1f}): {len(low_tech)} companies")
    for period in ["ret_5d", "ret_10d", "ret_21d", "ret_63d"]:
        if period in low_tech.columns:
            mean_ret = low_tech[period].mean()
            print(f"  {period}: {mean_ret:+.2%}")
    
    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    print("\nTech Intensity vs Forward Returns:")
    for period in ["ret_5d", "ret_10d", "ret_21d", "ret_63d"]:
        if period in results_df.columns:
            corr, pval = stats.pearsonr(results_df["tech_intensity"], results_df[period])
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {period}: r={corr:+.3f}, p={pval:.3f} {sig}")
    
    print("\nQoQ Tech Change vs Forward Returns:")
    qoq_df = results_df[results_df["qoq_change"] != 0]
    if len(qoq_df) >= 5:
        for period in ["ret_5d", "ret_10d", "ret_21d", "ret_63d"]:
            if period in qoq_df.columns:
                corr, pval = stats.pearsonr(qoq_df["qoq_change"], qoq_df[period])
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"  {period}: r={corr:+.3f}, p={pval:.3f} {sig}")
    else:
        print("  Not enough data with QoQ changes")
    
    # Long-short analysis
    print("\n" + "=" * 70)
    print("LONG-SHORT STRATEGY")
    print("=" * 70)
    
    # Top 5 vs Bottom 5
    top5 = results_df.nlargest(5, "tech_intensity")
    bottom5 = results_df.nsmallest(5, "tech_intensity")
    
    print("\nTop 5 Tech Intensity (LONG):")
    print(f"  Tickers: {', '.join(top5['ticker'].tolist())}")
    for period in ["ret_5d", "ret_10d", "ret_21d", "ret_63d"]:
        if period in top5.columns:
            print(f"  {period}: {top5[period].mean():+.2%}")
    
    print("\nBottom 5 Tech Intensity (SHORT):")
    print(f"  Tickers: {', '.join(bottom5['ticker'].tolist())}")
    for period in ["ret_5d", "ret_10d", "ret_21d", "ret_63d"]:
        if period in bottom5.columns:
            print(f"  {period}: {bottom5[period].mean():+.2%}")
    
    print("\nLong-Short Spread:")
    for period in ["ret_5d", "ret_10d", "ret_21d", "ret_63d"]:
        if period in top5.columns and period in bottom5.columns:
            spread = top5[period].mean() - bottom5[period].mean()
            print(f"  {period}: {spread:+.2%}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Save results
    output_file = Path(__file__).parent.parent / "data" / "tech_backtest_results.csv"
    output_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_historical_backtest()
