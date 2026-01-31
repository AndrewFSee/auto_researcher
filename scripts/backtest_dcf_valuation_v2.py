"""
DCF Valuation Alpha Backtesting - Memory Efficient Version.

Tests whether DCF-based valuation signals predict forward stock returns.

Signals tested:
1. FCF Yield (FCF / Enterprise Value) - simplest valuation metric
2. DCF Gap (DCF implied value / Market Cap - 1) - undervaluation signal
3. FCF Growth - momentum in cash generation

This version:
- Uses sampled tickers (S&P 500) instead of all 10k
- Uses yfinance for price data instead of loading 33M row parquet
- Processes in chunks to avoid OOM
"""

import logging
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Data URL - just financials
BASE_URL = "https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data"
STATEMENT_URL = f"{BASE_URL}/stock_statement.parquet"

# S&P 500 tickers (subset for efficiency)
SP500_SAMPLE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'KO',
    'COST', 'AVGO', 'LLY', 'MCD', 'TMO', 'WMT', 'CSCO', 'ACN', 'ABT', 'DHR',
    'VZ', 'NEE', 'ADBE', 'NKE', 'TXN', 'PM', 'CRM', 'UNP', 'LIN', 'ORCL',
    'AMD', 'INTC', 'HON', 'IBM', 'LOW', 'QCOM', 'CAT', 'BA', 'GE', 'AMGN',
    'RTX', 'SPGI', 'INTU', 'DE', 'SBUX', 'PLD', 'ISRG', 'GILD', 'MDLZ', 'ADP',
    'BKNG', 'ADI', 'VRTX', 'MMC', 'CB', 'SYK', 'REGN', 'TJX', 'LRCX', 'ZTS',
    'MO', 'CVS', 'PGR', 'CI', 'NOW', 'SCHW', 'BDX', 'SO', 'DUK', 'CL',
    'CME', 'AON', 'ITW', 'EQIX', 'SHW', 'MU', 'SNPS', 'ICE', 'CDNS', 'BSX',
    'NOC', 'ATVI', 'WM', 'APD', 'HUM', 'MCK', 'COP', 'ETN', 'PXD', 'FCX',
]


def load_financials_for_tickers(tickers: list) -> pd.DataFrame:
    """Load only financial data for specified tickers."""
    logger.info(f"Loading financials for {len(tickers)} tickers...")
    
    # Load full dataset but filter immediately
    df = pd.read_parquet(STATEMENT_URL)
    df = df[df['symbol'].isin(tickers) & (df['period_type'] == 'quarterly')].copy()
    
    logger.info(f"Filtered to {len(df):,} rows")
    
    # Pivot to columns
    pivot = df.pivot_table(
        index=['symbol', 'report_date'],
        columns='item_name',
        values='item_value',
        aggfunc='first'
    ).reset_index()
    
    # Rename key columns
    rename_map = {
        'symbol': 'ticker',
        'free_cash_flow': 'fcf',
        'operating_cash_flow': 'ocf', 
        'total_revenue': 'revenue',
        'total_debt': 'total_debt',
        'ordinary_shares_number': 'shares',
        'cash_and_cash_equivalents': 'cash_equiv',
    }
    existing_renames = {k: v for k, v in rename_map.items() if k in pivot.columns}
    pivot = pivot.rename(columns=existing_renames)
    
    # Filter out TTM
    pivot = pivot[pivot['report_date'] != 'TTM'].copy()
    pivot['report_date'] = pd.to_datetime(pivot['report_date'])
    
    # Convert Decimal to float
    for col in pivot.columns:
        if col not in ['ticker', 'report_date']:
            pivot[col] = pd.to_numeric(pivot[col], errors='coerce')
    
    # Sort
    pivot = pivot.sort_values(['ticker', 'report_date']).reset_index(drop=True)
    
    logger.info(f"Processed financials: {len(pivot):,} ticker-quarters")
    
    # Clean up
    del df
    gc.collect()
    
    return pivot


def calculate_ttm_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """Calculate trailing twelve month metrics."""
    logger.info("Calculating TTM metrics...")
    
    financials = financials.copy()
    
    for col in ['fcf', 'ocf', 'revenue']:
        if col in financials.columns:
            financials[f'{col}_ttm'] = financials.groupby('ticker')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )
    
    # FCF YoY growth
    if 'fcf_ttm' in financials.columns:
        financials['fcf_growth'] = financials.groupby('ticker')['fcf_ttm'].transform(
            lambda x: x.pct_change(4)
        )
    
    return financials


def get_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data from yfinance."""
    logger.info(f"Fetching prices for {len(tickers)} tickers...")
    
    try:
        # Download all at once
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning("No price data returned")
            return pd.DataFrame()
        
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            # New yfinance format: ('Close', 'AAPL')
            close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
        else:
            close = data[['Close']]
            close.columns = [tickers[0]] if len(tickers) == 1 else tickers
        
        # Reshape to long format
        prices = close.reset_index().melt(
            id_vars='Date', 
            var_name='ticker', 
            value_name='price'
        )
        prices = prices.rename(columns={'Date': 'date'})
        prices = prices.dropna(subset=['price'])
        
        logger.info(f"Fetched {len(prices):,} price observations")
        return prices
        
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        return pd.DataFrame()


def build_signals(financials: pd.DataFrame, prices: pd.DataFrame, 
                  quarterly_lag_days: int = 45,
                  annual_lag_days: int = 75) -> pd.DataFrame:
    """Build valuation signals with point-in-time prices.
    
    Uses different lags for quarterly vs annual reports:
    - Quarterly (10-Q): 45 days for large accelerated filers
    - Annual (10-K): 75 days to be safe (60 for large accelerated, but buffer for safety)
    """
    logger.info("Building signals...")
    
    # Determine if report is annual (fiscal year end) vs quarterly
    # Most companies have Dec year-end, but some have June, March, Sep
    # We'll flag any report at quarter-end months and use longer lag
    financials['report_month'] = financials['report_date'].dt.month
    
    # Fiscal year-end is typically: Dec (most), Jun, Mar, Sep
    # Use longer lag for all to be safe, shorter for mid-fiscal-year quarters
    # Conservative approach: use annual lag for Dec/Mar/Jun/Sep quarter ends
    # since we don't know which is the actual fiscal year end
    financials['is_quarter_end'] = financials['report_month'].isin([3, 6, 9, 12])
    
    # Apply appropriate lag
    financials['signal_date'] = financials.apply(
        lambda row: row['report_date'] + pd.Timedelta(
            days=annual_lag_days if row['report_month'] in [12, 6] else quarterly_lag_days
        ),
        axis=1
    )
    
    # Forward fill shares
    financials['shares'] = financials.groupby('ticker')['shares'].ffill()
    
    signals = []
    
    for ticker in financials['ticker'].unique():
        fin_data = financials[financials['ticker'] == ticker]
        price_data = prices[prices['ticker'] == ticker].sort_values('date')
        
        if price_data.empty:
            continue
        
        for _, row in fin_data.iterrows():
            signal_date = row['signal_date']
            shares = row.get('shares')
            fcf_ttm = row.get('fcf_ttm')
            revenue_ttm = row.get('revenue_ttm')
            total_debt = row.get('total_debt', 0) or 0
            cash = row.get('cash_equiv', 0) or 0
            fcf_growth = row.get('fcf_growth')
            
            # Handle NaN
            if pd.isna(shares) or shares <= 0:
                continue
            if pd.isna(total_debt): total_debt = 0
            if pd.isna(cash): cash = 0
            
            # Find price at signal date
            future_prices = price_data[price_data['date'] >= signal_date]
            if future_prices.empty:
                continue
            
            price_row = future_prices.iloc[0]
            price = price_row['price']
            price_date = price_row['date']
            
            # Market cap and EV
            market_cap = float(price) * float(shares)
            ev = market_cap + float(total_debt) - float(cash)
            
            signal = {
                'ticker': ticker,
                'report_date': row['report_date'],
                'signal_date': signal_date,
                'price_date': price_date,
                'price': price,
                'market_cap': market_cap,
                'ev': ev,
                'fcf_ttm': fcf_ttm,
                'fcf_growth': fcf_growth,
            }
            
            # FCF Yield
            if fcf_ttm and not pd.isna(fcf_ttm) and ev > 0:
                signal['fcf_yield'] = float(fcf_ttm) / ev
            else:
                signal['fcf_yield'] = None
            
            # FCF Margin
            if fcf_ttm and revenue_ttm and not pd.isna(revenue_ttm) and revenue_ttm > 0:
                signal['fcf_margin'] = float(fcf_ttm) / float(revenue_ttm)
            else:
                signal['fcf_margin'] = None
            
            signals.append(signal)
    
    signals_df = pd.DataFrame(signals)
    logger.info(f"Built {len(signals_df):,} signals")
    return signals_df


def add_forward_returns(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate forward returns."""
    logger.info("Calculating forward returns...")
    
    horizons = [5, 10, 20, 40, 60]
    
    for h in horizons:
        signals[f'ret{h}d'] = None
    
    for idx, row in signals.iterrows():
        ticker = row['ticker']
        price_date = row['price_date']
        price = row['price']
        
        ticker_prices = prices[prices['ticker'] == ticker]
        
        for h in horizons:
            future_date = price_date + pd.Timedelta(days=h)
            future_prices = ticker_prices[ticker_prices['date'] >= future_date]
            
            if not future_prices.empty and price > 0:
                future_price = future_prices.iloc[0]['price']
                signals.at[idx, f'ret{h}d'] = (future_price - price) / price
    
    return signals


def add_dcf_valuations(signals: pd.DataFrame) -> pd.DataFrame:
    """Add simple DCF-based valuations."""
    logger.info("Calculating DCF valuations...")
    
    dcf_gaps = []
    
    for _, row in signals.iterrows():
        fcf_ttm = row.get('fcf_ttm')
        fcf_growth = row.get('fcf_growth')
        market_cap = row.get('market_cap')
        
        if pd.isna(fcf_ttm) or fcf_ttm <= 0 or pd.isna(market_cap) or market_cap <= 0:
            dcf_gaps.append(None)
            continue
        
        # Use historical growth, capped
        growth = 0.05  # Default 5%
        if fcf_growth and not pd.isna(fcf_growth):
            growth = max(-0.1, min(0.25, float(fcf_growth)))
        
        # Simple DCF: 5yr projection + terminal value
        discount_rate = 0.10
        terminal_growth = 0.025
        
        pv_fcf = 0
        fcf = float(fcf_ttm)
        for year in range(1, 6):
            fcf = fcf * (1 + growth)
            pv_fcf += fcf / ((1 + discount_rate) ** year)
        
        # Terminal value
        terminal_fcf = fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        terminal_pv = terminal_value / ((1 + discount_rate) ** 5)
        
        dcf_value = pv_fcf + terminal_pv
        dcf_gap = (dcf_value - market_cap) / market_cap
        dcf_gaps.append(dcf_gap)
    
    signals['dcf_gap'] = dcf_gaps
    return signals


def calculate_ic(signal: pd.Series, returns: pd.Series) -> tuple:
    """Calculate Information Coefficient."""
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
    result['mean'] = result['mean'] * 100
    result['std'] = result['std'] * 100
    
    return result


def run_backtest(signals: pd.DataFrame):
    """Run backtest analysis."""
    print("\n" + "="*70)
    print("DCF VALUATION ALPHA BACKTEST")
    print("="*70)
    
    print(f"\nDataset: {len(signals):,} observations")
    print(f"Tickers: {signals['ticker'].nunique():,}")
    print(f"Date range: {signals['signal_date'].min().date()} to {signals['signal_date'].max().date()}")
    
    signal_cols = ['fcf_yield', 'dcf_gap', 'fcf_growth', 'fcf_margin']
    return_cols = ['ret5d', 'ret10d', 'ret20d', 'ret40d', 'ret60d']
    
    # IC Analysis
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
    
    # Quintile Analysis
    print("\n" + "="*70)
    print("QUINTILE ANALYSIS (FCF_YIELD -> ret20d)")
    print("="*70)
    
    if 'fcf_yield' in signals.columns and 'ret20d' in signals.columns:
        result = quintile_analysis(signals, 'fcf_yield', 'ret20d')
        if result is not None:
            print(f"\n{'Quintile':>10} {'Mean Ret%':>12} {'Std%':>10} {'Count':>10}")
            print("-"*45)
            for q in [1, 2, 3, 4, 5]:
                if q in result.index:
                    row = result.loc[q]
                    print(f"{q:>10} {row['mean']:>+11.3f}% {row['std']:>9.1f}% {row['count']:>10.0f}")
            
            if 5 in result.index and 1 in result.index:
                spread = result.loc[5, 'mean'] - result.loc[1, 'mean']
                print(f"\nLong-Short Spread (Q5-Q1): {spread:+.3f}%")
    
    # DCF Gap Quintiles
    print("\n" + "="*70)
    print("QUINTILE ANALYSIS (DCF_GAP -> ret20d)")
    print("="*70)
    
    if 'dcf_gap' in signals.columns and 'ret20d' in signals.columns:
        result = quintile_analysis(signals, 'dcf_gap', 'ret20d')
        if result is not None:
            print(f"\n{'Quintile':>10} {'Mean Ret%':>12} {'Std%':>10} {'Count':>10}")
            print("-"*45)
            for q in [1, 2, 3, 4, 5]:
                if q in result.index:
                    row = result.loc[q]
                    print(f"{q:>10} {row['mean']:>+11.3f}% {row['std']:>9.1f}% {row['count']:>10.0f}")
            
            if 5 in result.index and 1 in result.index:
                spread = result.loc[5, 'mean'] - result.loc[1, 'mean']
                print(f"\nLong-Short Spread (Q5-Q1): {spread:+.3f}%")


def main():
    logger.info("Starting DCF Valuation Backtest (Memory-Efficient)...")
    
    # Use S&P 500 sample
    tickers = SP500_SAMPLE
    
    # Load financials
    financials = load_financials_for_tickers(tickers)
    
    # Calculate TTM
    financials = calculate_ttm_metrics(financials)
    
    # Get date range
    min_date = financials['report_date'].min()
    max_date = financials['report_date'].max() + pd.Timedelta(days=120)
    
    # Fetch prices
    prices = get_price_data(
        tickers, 
        min_date.strftime('%Y-%m-%d'),
        max_date.strftime('%Y-%m-%d')
    )
    
    if prices.empty:
        logger.error("No price data - aborting")
        return None
    
    # Build signals
    signals = build_signals(financials, prices)
    
    # Add DCF valuations
    signals = add_dcf_valuations(signals)
    
    # Add forward returns
    signals = add_forward_returns(signals, prices)
    
    # Run backtest
    run_backtest(signals)
    
    # Save
    output_path = Path(__file__).parent.parent / "data" / "dcf_backtest_results.parquet"
    signals.to_parquet(output_path)
    logger.info(f"Results saved to {output_path}")
    
    return signals


if __name__ == "__main__":
    results = main()
