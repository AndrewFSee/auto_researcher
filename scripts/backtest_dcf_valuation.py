"""
DCF Valuation Alpha Backtesting.

Tests whether DCF-based valuation signals predict forward stock returns.

Signals tested:
1. FCF Yield (FCF / Enterprise Value) - simplest valuation metric
2. DCF Gap (DCF implied value / Market Cap - 1) - undervaluation signal
3. FCF Growth - momentum in cash generation
4. DCF Revision - changes in intrinsic value over time

Key principles:
- Point-in-time data only (no look-ahead bias)
- Use quarterly data with appropriate lag for reporting delay
- Cross-sectional ranking (relative value, not absolute)
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Data URLs
BASE_URL = "https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data"

DATASETS = {
    "stock_statement": f"{BASE_URL}/stock_statement.parquet",
    "stock_summary": f"{BASE_URL}/stock_summary.parquet",
    "stock_prices": f"{BASE_URL}/stock_prices.parquet",
}

# Cache for loaded data
_cache = {}


def load_dataset(name: str) -> pd.DataFrame:
    """Load dataset with caching."""
    if name not in _cache:
        logger.info(f"Loading {name}...")
        _cache[name] = pd.read_parquet(DATASETS[name])
        logger.info(f"Loaded {name}: {len(_cache[name]):,} rows")
    return _cache[name]


# ==============================================================================
# DATA EXTRACTION
# ==============================================================================

def get_historical_financials(period_type: str = 'quarterly') -> pd.DataFrame:
    """
    Extract key financial metrics for all tickers over time.
    
    Returns DataFrame with one row per ticker-quarter with:
    - FCF, Operating Cash Flow, CapEx
    - Revenue, Net Income, EBITDA
    - Total Debt, Cash, Stockholders Equity
    """
    df = load_dataset("stock_statement")
    
    # Filter to period type
    df = df[df['period_type'] == period_type].copy()
    
    # Pivot to get items as columns per ticker-date
    logger.info("Pivoting financial data...")
    pivot = df.pivot_table(
        index=['symbol', 'report_date'],
        columns='item_name',
        values='item_value',
        aggfunc='first'
    )
    
    # Flatten column names if MultiIndex
    if isinstance(pivot.columns, pd.MultiIndex):
        pivot.columns = pivot.columns.get_level_values(0)
    
    pivot = pivot.reset_index()
    
    # Rename columns for clarity
    # Note: 'cash' column already exists in some data, so don't rename cash_and_cash_equivalents to 'cash'
    rename_map = {
        'symbol': 'ticker',
        'free_cash_flow': 'fcf',
        'operating_cash_flow': 'ocf',
        'capital_expenditure': 'capex',
        'total_revenue': 'revenue',
        'net_income': 'net_income',
        'ebitda': 'ebitda',
        'total_debt': 'total_debt',
        'stockholders_equity': 'equity',
        'total_assets': 'total_assets',
        'ordinary_shares_number': 'shares',
    }
    
    # Only rename columns that exist (and avoid creating duplicates)
    existing_renames = {k: v for k, v in rename_map.items() 
                       if k in pivot.columns and v not in pivot.columns}
    pivot = pivot.rename(columns=existing_renames)
    
    # Filter out TTM rows (trailing twelve months - not a date)
    pivot = pivot[pivot['report_date'] != 'TTM'].copy()
    
    # Convert report_date to datetime
    pivot['report_date'] = pd.to_datetime(pivot['report_date'])
    
    # Sort
    pivot = pivot.sort_values(['ticker', 'report_date'])
    
    logger.info(f"Extracted financials: {len(pivot):,} ticker-quarters, {pivot['ticker'].nunique():,} tickers")
    
    return pivot


def get_market_data() -> pd.DataFrame:
    """
    Get current market data (market cap, EV, beta, etc).
    
    Note: This is point-in-time current data, not historical.
    For proper backtesting we'll use historical prices.
    """
    summary = load_dataset("stock_summary")
    
    summary = summary.rename(columns={
        'symbol': 'ticker',
        'market_cap': 'market_cap',
        'enterprise_value': 'ev',
        'beta': 'beta',
    })
    
    return summary[['ticker', 'market_cap', 'ev', 'beta', 'shares_outstanding']].copy()


def get_historical_prices(tickers: list = None) -> pd.DataFrame:
    """
    Get historical price data for calculating returns and market caps.
    
    Args:
        tickers: Optional list of tickers to filter. If None, loads all.
    """
    prices = load_dataset("stock_prices")
    
    # Filter by tickers first to reduce memory
    if tickers:
        prices = prices[prices['symbol'].isin(tickers)]
    
    prices = prices.rename(columns={
        'symbol': 'ticker',
        'report_date': 'date',
        'close': 'price',
        'volume': 'volume',
    })
    
    # Convert date column efficiently
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date'])
    
    return prices[['ticker', 'date', 'price', 'volume']]


# ==============================================================================
# DCF VALUATION MODEL
# ==============================================================================

def calculate_fcf_growth_rate(fcf_history: list, min_periods: int = 3) -> Optional[float]:
    """
    Calculate FCF growth rate from historical data.
    
    Uses geometric mean of YoY growth rates.
    """
    # Convert to float and filter valid values
    clean_history = []
    for x in fcf_history:
        if x is None:
            continue
        try:
            val = float(x)
            if not np.isnan(val) and val > 0:
                clean_history.append(val)
        except (TypeError, ValueError):
            continue
    
    if len(clean_history) < min_periods:
        return None
    
    # Calculate YoY growth rates
    growth_rates = []
    for i in range(1, len(clean_history)):
        if clean_history[i-1] > 0:
            growth = (clean_history[i] - clean_history[i-1]) / clean_history[i-1]
            # Cap extreme growth rates
            growth = max(-0.5, min(1.0, growth))
            growth_rates.append(growth)
    
    if not growth_rates:
        return None
    
    # Use median growth rate (more robust than mean)
    return np.median(growth_rates)


def simple_dcf_valuation(
    current_fcf: float,
    growth_rate: float,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    projection_years: int = 5,
) -> float:
    """
    Simple two-stage DCF model.
    
    Stage 1: Project FCF for N years at growth_rate
    Stage 2: Terminal value using Gordon Growth Model
    
    Args:
        current_fcf: Most recent annual FCF
        growth_rate: Expected FCF growth rate for projection period
        discount_rate: WACC or required return (default 10%)
        terminal_growth: Perpetual growth rate (default 2.5%)
        projection_years: Years of explicit projection
        
    Returns:
        Intrinsic enterprise value
    """
    if current_fcf <= 0:
        return 0.0
    
    # Cap growth rate to reasonable bounds
    growth_rate = max(-0.1, min(0.3, growth_rate))
    
    # Stage 1: Project FCF
    projected_fcf = []
    fcf = current_fcf
    for year in range(1, projection_years + 1):
        fcf = fcf * (1 + growth_rate)
        pv = fcf / ((1 + discount_rate) ** year)
        projected_fcf.append(pv)
    
    stage1_value = sum(projected_fcf)
    
    # Stage 2: Terminal value
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / ((1 + discount_rate) ** projection_years)
    
    return stage1_value + terminal_pv


# ==============================================================================
# SIGNAL CONSTRUCTION
# ==============================================================================

def build_valuation_signals(
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    reporting_lag_days: int = 45,  # Assume ~45 days for quarterly report availability
) -> pd.DataFrame:
    """
    Build point-in-time valuation signals.
    
    For each ticker-date, calculate:
    1. FCF Yield = TTM FCF / EV
    2. DCF Gap = (DCF Value - Market Cap) / Market Cap
    3. FCF Growth = YoY FCF growth
    4. FCF Margin = FCF / Revenue
    
    Uses reporting lag to avoid look-ahead bias.
    """
    logger.info("Building valuation signals...")
    
    # Sort by ticker and date
    financials = financials.sort_values(['ticker', 'report_date']).reset_index(drop=True)
    
    # Calculate TTM metrics (sum of last 4 quarters) per ticker
    logger.info("Calculating TTM metrics...")
    for col in ['fcf', 'ocf', 'revenue', 'net_income', 'ebitda']:
        if col in financials.columns:
            financials[f'{col}_ttm'] = financials.groupby('ticker')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )
    
    # Calculate FCF growth (YoY = compare to 4 quarters ago)
    logger.info("Calculating growth rates...")
    if 'fcf_ttm' in financials.columns:
        financials['fcf_growth'] = financials.groupby('ticker')['fcf_ttm'].transform(
            lambda x: x.pct_change(4)
        )
    
    # Add signal date (when data becomes available)
    financials['signal_date'] = financials['report_date'] + pd.Timedelta(days=reporting_lag_days)
    
    # Merge with prices to get market cap at signal date
    logger.info("Merging with price data...")
    
    # Check if shares column already exists from the financials pivot
    if 'shares' not in financials.columns:
        # Try to get from statement data
        statements = load_dataset("stock_statement")
        shares_df = statements[
            (statements['item_name'] == 'ordinary_shares_number') &
            (statements['period_type'] == 'quarterly')
        ][['symbol', 'report_date', 'item_value']].copy()
        shares_df = shares_df.rename(columns={
            'symbol': 'ticker',
            'item_value': 'shares'
        })
        shares_df['report_date'] = pd.to_datetime(shares_df['report_date'])
        financials = financials.merge(shares_df, on=['ticker', 'report_date'], how='left')
    
    # Forward fill shares within each ticker
    financials = financials.sort_values(['ticker', 'report_date'])
    financials['shares'] = financials.groupby('ticker')['shares'].ffill()
    
    # Get price at signal date
    prices_sorted = prices.sort_values(['ticker', 'date'])
    
    # For each financial record, find price at signal_date
    signals = []
    
    # Group financials by ticker for efficiency
    for ticker, fin_group in financials.groupby('ticker'):
        price_group = prices_sorted[prices_sorted['ticker'] == ticker]
        if price_group.empty:
            continue
        
        for _, row in fin_group.iterrows():
            signal_date = row['signal_date']
            
            # Find closest price on or after signal date
            future_prices = price_group[price_group['date'] >= signal_date]
            if future_prices.empty:
                continue
            
            price_row = future_prices.iloc[0]
            price = price_row['price']
            
            # Calculate market cap
            shares = row['shares'] if 'shares' in row.index else None
            if pd.isna(shares) or shares is None or shares <= 0:
                continue
            
            market_cap = price * shares
            
            # Calculate signals - safely get values
            fcf_ttm = row['fcf_ttm'] if 'fcf_ttm' in row.index else None
            revenue_ttm = row['revenue_ttm'] if 'revenue_ttm' in row.index else None
            total_debt = row['total_debt'] if 'total_debt' in row.index else 0
            fcf_growth = row['fcf_growth'] if 'fcf_growth' in row.index else None
            
            # Get cash - try multiple column names
            cash = None
            for cash_col in ['cash_and_cash_equivalents', 'cash']:
                if cash_col in row.index:
                    val = row[cash_col]
                    # Handle case where multiple columns match (returns Series)
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    if not pd.isna(val):
                        cash = float(val)
                        break
            cash = 0 if cash is None else float(cash)
            
            # Handle NaN values and convert to float
            total_debt = 0 if pd.isna(total_debt) else float(total_debt)
            
            # Enterprise Value = Market Cap + Debt - Cash
            ev = float(market_cap) + total_debt - cash
            
            # Convert other values to float
            fcf_ttm = float(fcf_ttm) if fcf_ttm is not None and not pd.isna(fcf_ttm) else None
            revenue_ttm = float(revenue_ttm) if revenue_ttm is not None and not pd.isna(revenue_ttm) else None
            fcf_growth = float(fcf_growth) if fcf_growth is not None and not pd.isna(fcf_growth) else None
            
            signal = {
                'ticker': ticker,
                'report_date': row['report_date'],
                'signal_date': signal_date,
                'price_date': price_row['date'],
                'price': float(price),
                'shares': float(shares),
                'market_cap': float(market_cap),
                'ev': ev,
                'fcf_ttm': fcf_ttm,
                'revenue_ttm': revenue_ttm,
                'fcf_growth': fcf_growth,
            }
            
            # FCF Yield
            if fcf_ttm is not None and ev > 0:
                signal['fcf_yield'] = fcf_ttm / ev
            else:
                signal['fcf_yield'] = None
            
            # FCF Margin
            if fcf_ttm is not None and revenue_ttm is not None and revenue_ttm > 0:
                signal['fcf_margin'] = fcf_ttm / revenue_ttm
            else:
                signal['fcf_margin'] = None
            
            signals.append(signal)
    
    signals_df = pd.DataFrame(signals)
    logger.info(f"Built {len(signals_df):,} signal observations")
    
    return signals_df


def add_dcf_valuations(signals: pd.DataFrame, financials: pd.DataFrame) -> pd.DataFrame:
    """
    Add DCF-based valuations to signals.
    
    Uses historical FCF to estimate growth rate, then calculates DCF value.
    """
    logger.info("Calculating DCF valuations...")
    
    # Get historical FCF for each ticker
    fcf_history = financials.groupby('ticker').apply(
        lambda x: x.sort_values('report_date')['fcf'].dropna().tolist()
    ).to_dict()
    
    dcf_values = []
    dcf_gaps = []
    
    for _, row in signals.iterrows():
        ticker = row['ticker']
        fcf_ttm = row['fcf_ttm']
        market_cap = row['market_cap']
        
        if pd.isna(fcf_ttm) or fcf_ttm <= 0 or ticker not in fcf_history:
            dcf_values.append(None)
            dcf_gaps.append(None)
            continue
        
        # Get FCF history up to this report date
        hist = fcf_history[ticker]
        
        # Estimate growth rate
        growth_rate = calculate_fcf_growth_rate(hist)
        if growth_rate is None:
            growth_rate = 0.05  # Default 5% growth
        
        # Calculate DCF value
        dcf_value = simple_dcf_valuation(
            current_fcf=fcf_ttm,
            growth_rate=growth_rate,
            discount_rate=0.10,
            terminal_growth=0.025,
            projection_years=5,
        )
        
        dcf_values.append(dcf_value)
        
        # DCF Gap (undervaluation signal)
        if market_cap > 0:
            dcf_gaps.append((dcf_value - market_cap) / market_cap)
        else:
            dcf_gaps.append(None)
    
    signals['dcf_value'] = dcf_values
    signals['dcf_gap'] = dcf_gaps
    
    logger.info("DCF valuations complete")
    
    return signals


def add_forward_returns(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward returns for each signal date.
    """
    logger.info("Calculating forward returns...")
    
    horizons = [5, 10, 20, 40, 60]
    
    # Create price lookup - convert to float for calculations
    prices_sorted = prices.sort_values(['ticker', 'date']).copy()
    prices_sorted['price'] = prices_sorted['price'].astype(float)
    
    returns = {f'ret{h}d': [] for h in horizons}
    
    for _, row in signals.iterrows():
        ticker = row['ticker']
        price_date = row['price_date']
        price = float(row['price'])
        
        ticker_prices = prices_sorted[prices_sorted['ticker'] == ticker]
        
        for h in horizons:
            future_date = price_date + pd.Timedelta(days=h)
            future_prices = ticker_prices[ticker_prices['date'] >= future_date]
            
            if future_prices.empty or pd.isna(price) or price <= 0:
                returns[f'ret{h}d'].append(None)
            else:
                future_price = float(future_prices.iloc[0]['price'])
                ret = (future_price - price) / price
                returns[f'ret{h}d'].append(ret)
    
    for h in horizons:
        signals[f'ret{h}d'] = returns[f'ret{h}d']
    
    logger.info("Forward returns complete")
    
    return signals


# ==============================================================================
# BACKTESTING
# ==============================================================================

def calculate_ic(signal: pd.Series, returns: pd.Series) -> tuple:
    """Calculate Information Coefficient (Spearman correlation)."""
    valid = pd.DataFrame({'signal': signal, 'returns': returns}).dropna()
    if len(valid) < 30:
        return None, None
    
    ic, pval = stats.spearmanr(valid['signal'].values, valid['returns'].values)
    return ic, pval


def quintile_analysis(signals: pd.DataFrame, signal_col: str, return_col: str) -> pd.DataFrame:
    """Analyze returns by signal quintile."""
    valid = signals[[signal_col, return_col]].dropna()
    
    if len(valid) < 100:
        return None
    
    valid['quintile'] = pd.qcut(valid[signal_col], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    result = valid.groupby('quintile')[return_col].agg(['mean', 'std', 'count'])
    result['mean'] = result['mean'] * 100  # Convert to %
    result['std'] = result['std'] * 100
    result['sharpe'] = result['mean'] / result['std'] * np.sqrt(252/int(return_col.replace('ret','').replace('d','')))
    
    return result


def run_backtest(signals: pd.DataFrame):
    """Run full backtest analysis."""
    print("\n" + "="*70)
    print("DCF VALUATION ALPHA BACKTEST")
    print("="*70)
    
    print(f"\nDataset: {len(signals):,} observations")
    print(f"Tickers: {signals['ticker'].nunique():,}")
    print(f"Date range: {signals['signal_date'].min().date()} to {signals['signal_date'].max().date()}")
    
    # Signal columns to test
    signal_cols = ['fcf_yield', 'dcf_gap', 'fcf_growth', 'fcf_margin']
    return_cols = ['ret5d', 'ret10d', 'ret20d', 'ret40d', 'ret60d']
    
    # ==== IC Analysis ====
    print("\n" + "="*70)
    print("INFORMATION COEFFICIENT (Spearman Correlation)")
    print("="*70)
    print("\n" + " "*20 + "  ".join([f"{r:>8}" for r in return_cols]))
    print("-"*70)
    
    for sig in signal_cols:
        if sig not in signals.columns:
            continue
        
        ics = []
        for ret in return_cols:
            if ret not in signals.columns:
                ics.append("    N/A")
                continue
            
            ic, pval = calculate_ic(signals[sig], signals[ret])
            if ic is None:
                ics.append("    N/A")
            else:
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                ics.append(f"{ic:+.4f}{stars:3}")
        
        print(f"{sig:20}" + "  ".join(ics))
    
    # ==== Quintile Analysis for Best Signals ====
    print("\n" + "="*70)
    print("QUINTILE ANALYSIS")
    print("="*70)
    
    for sig in ['fcf_yield', 'dcf_gap']:
        if sig not in signals.columns:
            continue
        
        print(f"\n--- {sig.upper()} -> ret20d ---")
        
        if 'ret20d' not in signals.columns:
            print("No return data")
            continue
        
        result = quintile_analysis(signals, sig, 'ret20d')
        if result is None:
            print("Insufficient data")
            continue
        
        print(f"{'Quintile':>10} {'Mean Ret%':>12} {'Std%':>10} {'Count':>10} {'Sharpe':>10}")
        print("-"*55)
        for q in [1, 2, 3, 4, 5]:
            if q in result.index:
                row = result.loc[q]
                print(f"{q:>10} {row['mean']:>+11.3f}% {row['std']:>9.1f}% {row['count']:>10.0f} {row['sharpe']:>10.2f}")
        
        if 5 in result.index and 1 in result.index:
            spread = result.loc[5, 'mean'] - result.loc[1, 'mean']
            print(f"\nLong-Short Spread (Q5-Q1): {spread:+.3f}%")
    
    # ==== Time Series Analysis ====
    print("\n" + "="*70)
    print("IC BY YEAR (FCF_YIELD -> ret20d)")
    print("="*70)
    
    if 'fcf_yield' in signals.columns and 'ret20d' in signals.columns:
        signals['year'] = signals['signal_date'].dt.year
        
        for year in sorted(signals['year'].unique()):
            year_data = signals[signals['year'] == year]
            ic, pval = calculate_ic(year_data['fcf_yield'], year_data['ret20d'])
            if ic is not None:
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"  {year}: IC={ic:+.4f} (n={len(year_data):,}) {stars}")
    
    return signals


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    logger.info("Starting DCF Valuation Backtest...")
    
    # Load financials first to get list of tickers
    financials = get_historical_financials(period_type='quarterly')
    
    # Get unique tickers from financials
    tickers = financials['ticker'].unique().tolist()
    logger.info(f"Found {len(tickers):,} tickers with financial data")
    
    # Load prices only for those tickers (memory efficient)
    prices = get_historical_prices(tickers=tickers)
    
    # Build signals
    signals = build_valuation_signals(financials, prices)
    
    # Add DCF valuations
    signals = add_dcf_valuations(signals, financials)
    
    # Add forward returns
    signals = add_forward_returns(signals, prices)
    
    # Run backtest
    signals = run_backtest(signals)
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "dcf_backtest_results.parquet"
    signals.to_parquet(output_path)
    logger.info(f"Results saved to {output_path}")
    
    return signals


if __name__ == "__main__":
    results = main()
