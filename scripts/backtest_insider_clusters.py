"""
Insider Cluster Signal Backtesting.

Tests whether insider cluster buying/selling predicts forward stock returns.

Hypothesis (from academic literature):
- Cluster buying (3+ insiders buying in 30 days) strongly predicts positive returns
- CEO/CFO purchases are more informative than lower-level insiders
- Open market purchases matter more than options exercises
- Selling is less informative (insiders sell for many reasons)

References:
- Lakonishok & Lee (2001): "Are Insider Trades Informative?"
- Seyhun (1986): "Insiders' Profits, Costs of Trading, and Market Efficiency"
- Cohen, Malloy, Pomorski (2012): "Decoding Inside Information"

Data Source: SEC EDGAR Form 4 filings (clean, legally-mandated dates)
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_researcher.agents.insider_trading_agent import (
    SECEdgarClient,
    InsiderTransaction,
    ClusterEvent,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Universe for backtesting
SP100_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMZN", "AVGO",
    "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT", "CHTR",
    "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "DE",
    "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
    "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ",
    "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
    "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX",
    "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN",
    "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
]

# Use smaller subset for faster testing
TEST_TICKERS = SP100_TICKERS[:15]  # Start small, expand if signal looks good

# Cluster detection parameters
CLUSTER_WINDOW_DAYS = 30
MIN_CLUSTER_INSIDERS = 2  # Lowered since open market buys are rare


# ==============================================================================
# DATA COLLECTION
# ==============================================================================

@dataclass
class ClusterSignal:
    """A cluster event with computed signal."""
    ticker: str
    event_date: datetime  # End date of cluster (when signal fires)
    direction: str  # "buy" or "sell"
    num_insiders: int
    total_value: float
    avg_role_weight: float
    has_ceo_cfo: bool
    signal_strength: float  # Computed signal strength
    
    # Forward returns (to be filled)
    ret_5d: Optional[float] = None
    ret_10d: Optional[float] = None
    ret_20d: Optional[float] = None
    ret_40d: Optional[float] = None
    ret_60d: Optional[float] = None


def fetch_insider_data(
    tickers: list[str],
    lookback_years: int = 3,
) -> dict[str, list[InsiderTransaction]]:
    """
    Fetch Form 4 transactions for all tickers.
    
    Note: SEC EDGAR has rate limits (10 requests/second).
    """
    client = SECEdgarClient()
    all_transactions = {}
    
    lookback_days = lookback_years * 365
    
    logger.info(f"Fetching insider data for {len(tickers)} tickers (lookback: {lookback_years} years)")
    
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")
        
        try:
            # Get Form 4 filings
            filings = client.get_form4_filings(
                ticker, 
                days_back=lookback_days,
                max_filings=200,
            )
            
            transactions = []
            for filing in filings:
                time.sleep(0.15)  # Rate limiting
                txns = client.parse_form4(filing["url"], ticker)
                transactions.extend(txns)
            
            # Filter to meaningful transactions
            # Note: Open market purchases (type P) are rare but most informative
            # Options exercises (type M) are more common but still informative
            meaningful = []
            for t in transactions:
                # Skip 10b5-1 pre-planned
                if t.is_10b5_1:
                    continue
                # Skip tiny transactions
                if t.total_value and t.total_value < 1000:
                    continue
                # Include open market (P, S) and options exercises (M)
                # M = options exercise - executive choosing to exercise is somewhat informative
                if t.transaction_type in ('P', 'S', 'M'):
                    meaningful.append(t)
            
            all_transactions[ticker] = meaningful
            p_count = len([t for t in meaningful if t.transaction_type == 'P'])
            s_count = len([t for t in meaningful if t.transaction_type == 'S'])
            m_count = len([t for t in meaningful if t.transaction_type == 'M'])
            logger.info(f"  → {len(meaningful)} txns (P={p_count}, S={s_count}, M={m_count})")
            
        except Exception as e:
            logger.error(f"  → Failed: {e}")
            all_transactions[ticker] = []
        
        time.sleep(0.1)  # Additional rate limiting
    
    return all_transactions


def detect_clusters(
    transactions: list[InsiderTransaction],
    window_days: int = 30,
    min_insiders: int = 2,
) -> list[ClusterEvent]:
    """Detect cluster buying/selling events."""
    if not transactions:
        return []
    
    # Sort by date
    sorted_txns = sorted(transactions, key=lambda t: t.transaction_date)
    window = timedelta(days=window_days)
    
    clusters = []
    processed_dates = set()
    
    for i, start_txn in enumerate(sorted_txns):
        window_end = start_txn.transaction_date + window
        
        # Skip if we've already found a cluster starting near here
        date_key = start_txn.transaction_date.strftime("%Y-%m")
        if date_key in processed_dates:
            continue
        
        # Collect transactions in window
        window_txns = [t for t in sorted_txns 
                      if start_txn.transaction_date <= t.transaction_date <= window_end]
        
        # Count unique insiders by direction
        # is_purchase includes both P (open market) and M (options exercise) with AD=A
        buy_insiders = {}  # name -> list of transactions
        sell_insiders = {}
        
        for t in window_txns:
            # Classify as buy or sell based on acquisition_disposition
            if t.acquisition_disposition == 'A':  # Acquired shares
                if t.insider_name not in buy_insiders:
                    buy_insiders[t.insider_name] = []
                buy_insiders[t.insider_name].append(t)
            elif t.acquisition_disposition == 'D':  # Disposed shares
                if t.insider_name not in sell_insiders:
                    sell_insiders[t.insider_name] = []
                sell_insiders[t.insider_name].append(t)
        
        # Check for buy cluster
        if len(buy_insiders) >= min_insiders:
            buy_txns = [t for txns in buy_insiders.values() for t in txns]
            total_value = sum(t.total_value or 0 for t in buy_txns)
            avg_weight = np.mean([t.role_weight for t in buy_txns])
            
            clusters.append(ClusterEvent(
                ticker=start_txn.ticker,
                start_date=min(t.transaction_date for t in buy_txns),
                end_date=max(t.transaction_date for t in buy_txns),
                direction="buy",
                num_insiders=len(buy_insiders),
                total_value=total_value,
                transactions=buy_txns,
            ))
            processed_dates.add(date_key)
        
        # Check for sell cluster
        if len(sell_insiders) >= min_insiders:
            sell_txns = [t for txns in sell_insiders.values() for t in txns]
            total_value = sum(t.total_value or 0 for t in sell_txns)
            
            clusters.append(ClusterEvent(
                ticker=start_txn.ticker,
                start_date=min(t.transaction_date for t in sell_txns),
                end_date=max(t.transaction_date for t in sell_txns),
                direction="sell",
                num_insiders=len(sell_insiders),
                total_value=total_value,
                transactions=sell_txns,
            ))
            processed_dates.add(date_key)
    
    return clusters


def clusters_to_signals(clusters: list[ClusterEvent]) -> list[ClusterSignal]:
    """Convert cluster events to signals with computed strength."""
    signals = []
    
    for cluster in clusters:
        # Check if CEO/CFO participated
        has_ceo_cfo = any(
            any(title in t.insider_title.lower() 
                for title in ["ceo", "cfo", "chief executive", "chief financial"])
            for t in cluster.transactions
        )
        
        avg_weight = np.mean([t.role_weight for t in cluster.transactions])
        
        # Signal strength based on:
        # - Number of insiders (more = stronger)
        # - Role weight (C-suite = stronger)
        # - Total value (larger = stronger)
        insider_score = min(1.0, (cluster.num_insiders - 2) / 3)  # 3 insiders = 0.33, 5+ = 1.0
        role_score = avg_weight
        value_score = min(1.0, np.log10(cluster.total_value + 1) / 7)  # ~$10M = 1.0
        
        strength = 0.4 * insider_score + 0.3 * role_score + 0.3 * value_score
        
        # Selling is less informative
        if cluster.direction == "sell":
            strength *= 0.5
        
        signals.append(ClusterSignal(
            ticker=cluster.ticker,
            event_date=cluster.end_date,
            direction=cluster.direction,
            num_insiders=cluster.num_insiders,
            total_value=cluster.total_value,
            avg_role_weight=avg_weight,
            has_ceo_cfo=has_ceo_cfo,
            signal_strength=strength,
        ))
    
    return signals


# ==============================================================================
# RETURN CALCULATION
# ==============================================================================

def fetch_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch adjusted close prices for all tickers."""
    logger.info(f"Fetching prices for {len(tickers)} tickers...")
    
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,  # Use adjusted prices directly
    )
    
    # Handle yfinance multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        # New yfinance format: ('Close', 'AAPL'), ('Close', 'MSFT'), ...
        prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    else:
        # Single ticker case
        prices = data['Close'] if 'Close' in data.columns else data
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    
    return prices


def calculate_forward_returns(
    signals: list[ClusterSignal],
    prices: pd.DataFrame,
) -> list[ClusterSignal]:
    """Calculate forward returns for each signal."""
    
    for signal in signals:
        ticker = signal.ticker
        event_date = signal.event_date
        
        if ticker not in prices.columns:
            continue
        
        ticker_prices = prices[ticker].dropna()
        
        # Find the next trading day after event
        valid_dates = ticker_prices.index[ticker_prices.index >= pd.Timestamp(event_date)]
        if len(valid_dates) == 0:
            continue
        
        signal_date = valid_dates[0]
        signal_price = ticker_prices.loc[signal_date]
        
        # Calculate forward returns
        for days, attr in [(5, 'ret_5d'), (10, 'ret_10d'), (20, 'ret_20d'), 
                           (40, 'ret_40d'), (60, 'ret_60d')]:
            future_dates = ticker_prices.index[ticker_prices.index >= signal_date]
            if len(future_dates) > days:
                future_price = ticker_prices.iloc[ticker_prices.index.get_loc(signal_date) + days]
                ret = (future_price - signal_price) / signal_price
                setattr(signal, attr, ret)
    
    return signals


# ==============================================================================
# ANALYSIS
# ==============================================================================

def analyze_signals(signals: list[ClusterSignal]) -> dict:
    """Analyze signal predictive power."""
    
    # Convert to DataFrame
    data = []
    for s in signals:
        data.append({
            'ticker': s.ticker,
            'date': s.event_date,
            'direction': s.direction,
            'num_insiders': s.num_insiders,
            'total_value': s.total_value,
            'avg_role_weight': s.avg_role_weight,
            'has_ceo_cfo': s.has_ceo_cfo,
            'signal_strength': s.signal_strength,
            'ret_5d': s.ret_5d,
            'ret_10d': s.ret_10d,
            'ret_20d': s.ret_20d,
            'ret_40d': s.ret_40d,
            'ret_60d': s.ret_60d,
        })
    
    df = pd.DataFrame(data)
    df = df.dropna(subset=['ret_20d'])  # Require at least 20d return
    
    logger.info(f"\n{'='*70}")
    logger.info(f"INSIDER CLUSTER SIGNAL ANALYSIS")
    logger.info(f"{'='*70}")
    logger.info(f"Total signals: {len(df)}")
    logger.info(f"  Buy clusters: {len(df[df['direction']=='buy'])}")
    logger.info(f"  Sell clusters: {len(df[df['direction']=='sell'])}")
    logger.info(f"  With CEO/CFO: {len(df[df['has_ceo_cfo']])}")
    
    results = {}
    
    # 1. Overall IC by horizon
    logger.info(f"\n--- INFORMATION COEFFICIENT (Spearman) ---")
    
    # For buy clusters: higher strength should → higher returns
    buy_df = df[df['direction'] == 'buy'].copy()
    sell_df = df[df['direction'] == 'sell'].copy()
    
    # Create directional signal: +strength for buy, -strength for sell
    df['directional_signal'] = df.apply(
        lambda r: r['signal_strength'] if r['direction'] == 'buy' else -r['signal_strength'],
        axis=1
    )
    
    for horizon in ['ret_5d', 'ret_10d', 'ret_20d', 'ret_40d', 'ret_60d']:
        valid = df.dropna(subset=[horizon])
        if len(valid) < 20:
            continue
        
        ic, pval = stats.spearmanr(valid['directional_signal'], valid[horizon])
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        logger.info(f"  {horizon}: IC = {ic:+.3f} (p={pval:.3f}) {sig} [N={len(valid)}]")
        results[f'ic_{horizon}'] = ic
        results[f'pval_{horizon}'] = pval
    
    # 2. Buy vs Sell returns
    logger.info(f"\n--- BUY vs SELL CLUSTER RETURNS ---")
    
    for horizon in ['ret_20d', 'ret_40d', 'ret_60d']:
        buy_valid = buy_df.dropna(subset=[horizon])
        sell_valid = sell_df.dropna(subset=[horizon])
        
        if len(buy_valid) < 10 or len(sell_valid) < 10:
            continue
        
        buy_ret = buy_valid[horizon].mean()
        sell_ret = sell_valid[horizon].mean()
        spread = buy_ret - sell_ret
        
        # T-test for significance
        t_stat, t_pval = stats.ttest_ind(buy_valid[horizon], sell_valid[horizon])
        sig = "***" if t_pval < 0.01 else "**" if t_pval < 0.05 else "*" if t_pval < 0.10 else ""
        
        logger.info(f"  {horizon}: Buy={buy_ret:+.2%}, Sell={sell_ret:+.2%}, Spread={spread:+.2%} {sig}")
        results[f'buy_ret_{horizon}'] = buy_ret
        results[f'sell_ret_{horizon}'] = sell_ret
        results[f'spread_{horizon}'] = spread
    
    # 3. CEO/CFO Effect
    logger.info(f"\n--- CEO/CFO PARTICIPATION EFFECT ---")
    
    ceo_buy = buy_df[buy_df['has_ceo_cfo'] == True]
    non_ceo_buy = buy_df[buy_df['has_ceo_cfo'] == False]
    
    for horizon in ['ret_20d', 'ret_40d', 'ret_60d']:
        ceo_valid = ceo_buy.dropna(subset=[horizon])
        non_ceo_valid = non_ceo_buy.dropna(subset=[horizon])
        
        if len(ceo_valid) < 5 or len(non_ceo_valid) < 5:
            continue
        
        ceo_ret = ceo_valid[horizon].mean()
        non_ceo_ret = non_ceo_valid[horizon].mean()
        
        logger.info(f"  {horizon}: CEO/CFO Buy={ceo_ret:+.2%} [N={len(ceo_valid)}], "
                   f"Others={non_ceo_ret:+.2%} [N={len(non_ceo_valid)}]")
    
    # 4. Quintile Analysis
    logger.info(f"\n--- QUINTILE ANALYSIS (by signal strength) ---")
    
    buy_df = buy_df.dropna(subset=['ret_40d'])
    if len(buy_df) >= 25:
        buy_df['quintile'] = pd.qcut(buy_df['signal_strength'], 5, labels=[1,2,3,4,5])
        
        logger.info(f"  Buy Clusters by Strength (ret_40d):")
        for q in [1, 2, 3, 4, 5]:
            q_data = buy_df[buy_df['quintile'] == q]
            if len(q_data) > 0:
                logger.info(f"    Q{q}: {q_data['ret_40d'].mean():+.2%} [N={len(q_data)}]")
        
        # L/S spread
        q5_ret = buy_df[buy_df['quintile'] == 5]['ret_40d'].mean()
        q1_ret = buy_df[buy_df['quintile'] == 1]['ret_40d'].mean()
        logger.info(f"  Q5-Q1 Spread: {q5_ret - q1_ret:+.2%}")
        results['quintile_spread_40d'] = q5_ret - q1_ret
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run insider cluster backtesting."""
    
    # Configuration
    tickers = TEST_TICKERS  # Start with smaller set
    lookback_years = 3
    
    logger.info(f"{'='*70}")
    logger.info(f"INSIDER CLUSTER SIGNAL BACKTEST")
    logger.info(f"{'='*70}")
    logger.info(f"Universe: {len(tickers)} tickers")
    logger.info(f"Lookback: {lookback_years} years")
    logger.info(f"Cluster params: {MIN_CLUSTER_INSIDERS}+ insiders in {CLUSTER_WINDOW_DAYS} days")
    logger.info("")
    
    # Step 1: Fetch insider data
    logger.info("Step 1: Fetching insider transaction data...")
    all_transactions = fetch_insider_data(tickers, lookback_years)
    
    total_txns = sum(len(v) for v in all_transactions.values())
    logger.info(f"Total meaningful transactions: {total_txns}")
    
    # Step 2: Detect clusters
    logger.info("\nStep 2: Detecting cluster events...")
    all_clusters = []
    for ticker, transactions in all_transactions.items():
        clusters = detect_clusters(
            transactions, 
            CLUSTER_WINDOW_DAYS, 
            MIN_CLUSTER_INSIDERS
        )
        all_clusters.extend(clusters)
    
    logger.info(f"Total cluster events: {len(all_clusters)}")
    buy_clusters = len([c for c in all_clusters if c.direction == "buy"])
    sell_clusters = len([c for c in all_clusters if c.direction == "sell"])
    logger.info(f"  Buy clusters: {buy_clusters}")
    logger.info(f"  Sell clusters: {sell_clusters}")
    
    if len(all_clusters) == 0:
        logger.warning("No clusters found! Try increasing lookback or reducing min_insiders.")
        return
    
    # Step 3: Convert to signals
    logger.info("\nStep 3: Computing signal strength...")
    signals = clusters_to_signals(all_clusters)
    
    # Step 4: Get price data
    logger.info("\nStep 4: Fetching price data...")
    min_date = min(s.event_date for s in signals)
    max_date = max(s.event_date for s in signals) + timedelta(days=90)
    
    prices = fetch_prices(
        tickers,
        start_date=(min_date - timedelta(days=5)).strftime("%Y-%m-%d"),
        end_date=min(max_date, datetime.now()).strftime("%Y-%m-%d"),
    )
    
    # Step 5: Calculate returns
    logger.info("\nStep 5: Calculating forward returns...")
    signals = calculate_forward_returns(signals, prices)
    
    # Filter signals with returns
    valid_signals = [s for s in signals if s.ret_20d is not None]
    logger.info(f"Signals with valid returns: {len(valid_signals)}")
    
    # Step 6: Analyze
    logger.info("\nStep 6: Analyzing predictive power...")
    results = analyze_signals(valid_signals)
    
    # Save results
    output_path = Path(__file__).parent / "insider_cluster_results.csv"
    df = pd.DataFrame([{
        'ticker': s.ticker,
        'date': s.event_date,
        'direction': s.direction,
        'num_insiders': s.num_insiders,
        'total_value': s.total_value,
        'has_ceo_cfo': s.has_ceo_cfo,
        'signal_strength': s.signal_strength,
        'ret_5d': s.ret_5d,
        'ret_10d': s.ret_10d,
        'ret_20d': s.ret_20d,
        'ret_40d': s.ret_40d,
        'ret_60d': s.ret_60d,
    } for s in valid_signals])
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*70}")
    if 'ic_ret_40d' in results:
        ic = results.get('ic_ret_40d', 0)
        spread = results.get('spread_ret_40d', 0)
        logger.info(f"40-day IC: {ic:+.3f}")
        logger.info(f"40-day Buy-Sell Spread: {spread:+.2%}")
    
    return results


if __name__ == "__main__":
    results = main()
