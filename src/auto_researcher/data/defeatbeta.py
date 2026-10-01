"""
DefeatBeta Data Source.

Free financial data from HuggingFace covering 8,000+ stocks.
Includes financial statements, EPS history, estimates, and more.

Data is sourced from Yahoo Finance via the bwzheng2010/yahoo-finance-data dataset.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# DATASET URLs
# ==============================================================================

BASE_URL = "https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data"

DATASETS = {
    "stock_statement": f"{BASE_URL}/stock_statement.parquet",
    "stock_summary": f"{BASE_URL}/stock_summary.parquet",
    "stock_profile": f"{BASE_URL}/stock_profile.parquet",
    "stock_historical_eps": f"{BASE_URL}/stock_historical_eps.parquet",
    "stock_earning_estimates": f"{BASE_URL}/stock_earning_estimates.parquet",
    "stock_revenue_estimates": f"{BASE_URL}/stock_revenue_estimates.parquet",
    "stock_news": f"{BASE_URL}/stock_news.parquet",
    "stock_prices": f"{BASE_URL}/stock_prices.parquet",
    "stock_earning_call_transcripts": f"{BASE_URL}/stock_earning_call_transcripts.parquet",
}


# ==============================================================================
# CACHED DATA LOADER
# ==============================================================================

class DefeatBetaDataLoader:
    """
    Loads and caches DefeatBeta datasets from HuggingFace.
    
    Caches data in memory to avoid repeated downloads within a session.
    """
    
    _cache: dict[str, pd.DataFrame] = {}
    _cache_time: dict[str, datetime] = {}
    _cache_ttl_hours: int = 24
    
    @classmethod
    def load(cls, dataset_name: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Load a dataset from DefeatBeta.
        
        Args:
            dataset_name: One of the keys in DATASETS dict.
            force_refresh: If True, bypass cache and reload.
            
        Returns:
            DataFrame with the dataset, or None if failed.
        """
        if dataset_name not in DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None
        
        # Check cache
        now = datetime.now()
        if not force_refresh and dataset_name in cls._cache:
            cache_age = (now - cls._cache_time[dataset_name]).total_seconds() / 3600
            if cache_age < cls._cache_ttl_hours:
                logger.debug(f"Using cached {dataset_name} (age: {cache_age:.1f}h)")
                return cls._cache[dataset_name]
        
        # Load from HuggingFace
        try:
            url = DATASETS[dataset_name]
            logger.info(f"Loading {dataset_name} from HuggingFace...")
            df = pd.read_parquet(url)
            cls._cache[dataset_name] = df
            cls._cache_time[dataset_name] = now
            logger.info(f"Loaded {dataset_name}: {len(df):,} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            return None
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached data."""
        cls._cache.clear()
        cls._cache_time.clear()


# ==============================================================================
# FINANCIAL STATEMENT DATA
# ==============================================================================

@dataclass
class FinancialStatement:
    """Financial statement data for a company."""
    ticker: str
    period_type: str  # 'quarterly' or 'annual'
    report_date: str
    
    # Income Statement
    total_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None
    basic_eps: Optional[float] = None
    diluted_eps: Optional[float] = None
    
    # Balance Sheet
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    stockholders_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_cash_equivalents: Optional[float] = None
    
    # Cash Flow
    operating_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    free_cash_flow: Optional[float] = None


def get_financial_statements(
    ticker: str,
    period_type: str = "quarterly",
    limit: int = 8
) -> list[FinancialStatement]:
    """
    Get historical financial statements for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        period_type: 'quarterly' or 'annual'.
        limit: Maximum number of periods to return.
        
    Returns:
        List of FinancialStatement objects, most recent first.
    """
    df = DefeatBetaDataLoader.load("stock_statement")
    if df is None:
        return []
    
    # Filter for ticker and period
    mask = (df['symbol'] == ticker.upper()) & (df['period_type'] == period_type)
    ticker_df = df[mask].copy()
    
    if ticker_df.empty:
        logger.debug(f"No financial statements found for {ticker}")
        return []
    
    # Pivot to get items as columns
    pivot = ticker_df.pivot_table(
        index='report_date',
        columns='item_name',
        values='item_value',
        aggfunc='first'
    )
    
    # Sort by date descending
    pivot = pivot.sort_index(ascending=False)
    
    # Exclude TTM row for historical analysis
    pivot = pivot[pivot.index != 'TTM']
    
    # Limit results
    pivot = pivot.head(limit)
    
    results = []
    for report_date, row in pivot.iterrows():
        stmt = FinancialStatement(
            ticker=ticker,
            period_type=period_type,
            report_date=str(report_date),
            # Income Statement
            total_revenue=row.get('total_revenue'),
            gross_profit=row.get('gross_profit'),
            operating_income=row.get('operating_income'),
            net_income=row.get('net_income'),
            ebitda=row.get('ebitda'),
            basic_eps=row.get('basic_eps'),
            diluted_eps=row.get('diluted_eps'),
            # Balance Sheet
            total_assets=row.get('total_assets'),
            total_liabilities=row.get('total_liabilities_net_minority_interest'),
            stockholders_equity=row.get('stockholders_equity'),
            total_debt=row.get('total_debt'),
            cash_and_cash_equivalents=row.get('cash_and_cash_equivalents'),
            # Cash Flow
            operating_cash_flow=row.get('operating_cash_flow'),
            capital_expenditure=row.get('capital_expenditure'),
            free_cash_flow=row.get('free_cash_flow'),
        )
        results.append(stmt)
    
    return results


def get_financial_trends(ticker: str, periods: int = 4) -> dict:
    """
    Calculate financial trends over recent periods.
    
    Returns growth rates and trend indicators for key metrics.
    """
    statements = get_financial_statements(ticker, period_type='quarterly', limit=periods + 1)
    
    if len(statements) < 2:
        return {}
    
    trends = {}
    
    # Revenue trend
    revenues = [s.total_revenue for s in statements if s.total_revenue]
    if len(revenues) >= 2:
        # YoY growth (compare current to 4 quarters ago if available)
        if len(revenues) >= 5:
            yoy_growth = (revenues[0] - revenues[4]) / revenues[4] if revenues[4] else None
            trends['revenue_yoy_growth'] = yoy_growth
        # QoQ growth
        qoq_growth = (revenues[0] - revenues[1]) / revenues[1] if revenues[1] else None
        trends['revenue_qoq_growth'] = qoq_growth
        # Trend direction
        trends['revenue_trend'] = 'growing' if sum(1 for i in range(len(revenues)-1) if revenues[i] > revenues[i+1]) > len(revenues)/2 else 'declining'
    
    # Net income trend
    net_incomes = [s.net_income for s in statements if s.net_income is not None]
    if len(net_incomes) >= 2:
        if len(net_incomes) >= 5 and net_incomes[4]:
            yoy_growth = (net_incomes[0] - net_incomes[4]) / abs(net_incomes[4])
            trends['net_income_yoy_growth'] = yoy_growth
        trends['net_income_trend'] = 'growing' if sum(1 for i in range(len(net_incomes)-1) if net_incomes[i] > net_incomes[i+1]) > len(net_incomes)/2 else 'declining'
    
    # Margin trends
    if statements[0].total_revenue and statements[0].gross_profit:
        trends['gross_margin'] = statements[0].gross_profit / statements[0].total_revenue
    if statements[0].total_revenue and statements[0].operating_income:
        trends['operating_margin'] = statements[0].operating_income / statements[0].total_revenue
    if statements[0].total_revenue and statements[0].net_income:
        trends['net_margin'] = statements[0].net_income / statements[0].total_revenue
    
    # Debt trends
    if statements[0].total_debt and statements[0].stockholders_equity:
        trends['debt_to_equity'] = statements[0].total_debt / statements[0].stockholders_equity
    
    # Cash position
    if statements[0].cash_and_cash_equivalents:
        trends['cash_position'] = statements[0].cash_and_cash_equivalents
    
    return trends


# ==============================================================================
# EPS HISTORY AND ESTIMATES
# ==============================================================================

def get_eps_history(ticker: str) -> list[dict]:
    """
    Get EPS surprise history for a ticker.
    
    Returns list of quarters with actual vs estimate EPS and surprise %.
    """
    df = DefeatBetaDataLoader.load("stock_historical_eps")
    if df is None:
        return []
    
    ticker_df = df[df['symbol'] == ticker.upper()].copy()
    if ticker_df.empty:
        return []
    
    results = []
    for _, row in ticker_df.iterrows():
        results.append({
            'quarter': row.get('quarter_name'),
            'quarter_date': row.get('quarter_date'),
            'eps_actual': row.get('eps_actual'),
            'eps_estimate': row.get('eps_estimate'),
            'surprise_pct': row.get('surprise_percent'),
        })
    
    return results


def get_earnings_estimates(ticker: str) -> dict:
    """
    Get current earnings estimates and revisions for a ticker.
    """
    df = DefeatBetaDataLoader.load("stock_earning_estimates")
    if df is None:
        return {}
    
    ticker_df = df[df['symbol'] == ticker.upper()].copy()
    if ticker_df.empty:
        return {}
    
    # Get most recent estimates
    ticker_df = ticker_df.sort_values('report_date', ascending=False)
    latest = ticker_df.iloc[0]
    
    return {
        'estimate_avg_eps': latest.get('estimate_avg_eps'),
        'estimate_high_eps': latest.get('estimate_high_eps'),
        'estimate_low_eps': latest.get('estimate_low_eps'),
        'estimate_eps_growth': latest.get('estimate_eps_growth'),
        'num_analysts': latest.get('number_of_analysts'),
        '7d_ago_estimate': latest.get('seven_days_ago_estimate_avg_eps'),
        '30d_ago_estimate': latest.get('thirty_days_ago_estimate_avg_eps'),
        '90d_ago_estimate': latest.get('ninety_days_ago_estimate_avg_eps'),
        'period_type': latest.get('period_type'),
    }


def calculate_estimate_revisions(estimates: dict) -> dict:
    """
    Calculate estimate revision trends (a classic alpha signal).
    
    Positive revisions = analysts becoming more bullish = potential outperformance.
    """
    current = estimates.get('estimate_avg_eps')
    revisions = {}
    
    if current is None:
        return revisions
    
    # 7-day revision
    d7 = estimates.get('7d_ago_estimate')
    if d7 and d7 != 0:
        revisions['revision_7d'] = (current - d7) / abs(d7)
    
    # 30-day revision
    d30 = estimates.get('30d_ago_estimate')
    if d30 and d30 != 0:
        revisions['revision_30d'] = (current - d30) / abs(d30)
    
    # 90-day revision
    d90 = estimates.get('90d_ago_estimate')
    if d90 and d90 != 0:
        revisions['revision_90d'] = (current - d90) / abs(d90)
    
    # Overall revision direction
    if revisions:
        avg_revision = sum(revisions.values()) / len(revisions)
        if avg_revision > 0.02:
            revisions['revision_signal'] = 'positive'
        elif avg_revision < -0.02:
            revisions['revision_signal'] = 'negative'
        else:
            revisions['revision_signal'] = 'neutral'
    
    return revisions


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_fundamental_data(ticker: str) -> dict:
    """
    Get comprehensive fundamental data for a ticker.
    
    Combines financial statements, trends, EPS history, and estimates.
    """
    data = {
        'ticker': ticker,
        'statements': get_financial_statements(ticker, limit=4),
        'trends': get_financial_trends(ticker),
        'eps_history': get_eps_history(ticker),
        'estimates': get_earnings_estimates(ticker),
    }
    
    # Add estimate revisions
    if data['estimates']:
        data['estimate_revisions'] = calculate_estimate_revisions(data['estimates'])
    
    return data


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"\n{'='*60}")
    print(f"FUNDAMENTAL DATA FOR {ticker}")
    print(f"{'='*60}")
    
    # Get statements
    stmts = get_financial_statements(ticker)
    if stmts:
        print(f"\nLatest Financial Statement ({stmts[0].report_date}):")
        rev = float(stmts[0].total_revenue) if stmts[0].total_revenue else None
        ni = float(stmts[0].net_income) if stmts[0].net_income else None
        eps = float(stmts[0].basic_eps) if stmts[0].basic_eps else None
        print(f"  Revenue: ${rev/1e9:.2f}B" if rev else "  Revenue: N/A")
        print(f"  Net Income: ${ni/1e9:.2f}B" if ni else "  Net Income: N/A")
        print(f"  EPS: ${eps:.2f}" if eps else "  EPS: N/A")
    
    # Get trends
    trends = get_financial_trends(ticker)
    if trends:
        print(f"\nTrends:")
        for k, v in trends.items():
            if isinstance(v, float):
                print(f"  {k}: {v*100:.1f}%" if 'growth' in k or 'margin' in k else f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
    
    # Get EPS history
    eps = get_eps_history(ticker)
    if eps:
        print(f"\nEPS History (last 4 quarters):")
        for e in eps[:4]:
            surprise = e.get('surprise_pct', 'N/A')
            print(f"  {e['quarter']}: Actual={e['eps_actual']} vs Est={e['eps_estimate']} (Surprise: {surprise})")
    
    # Get estimates
    est = get_earnings_estimates(ticker)
    if est:
        print(f"\nCurrent Estimates:")
        print(f"  Avg EPS: {est.get('estimate_avg_eps')}")
        print(f"  EPS Growth: {est.get('estimate_eps_growth')}")
        print(f"  Analysts: {est.get('num_analysts')}")
        
        revisions = calculate_estimate_revisions(est)
        if revisions:
            print(f"\nEstimate Revisions:")
            for k, v in revisions.items():
                if isinstance(v, float):
                    print(f"  {k}: {v*100:+.1f}%")
                else:
                    print(f"  {k}: {v}")
