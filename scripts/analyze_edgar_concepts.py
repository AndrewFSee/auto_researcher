#!/usr/bin/env python3
"""
EDGAR XBRL Concept Analyzer.

Analyzes SEC EDGAR data to identify which XBRL concepts are available
for each ticker, helping us expand the concept mappings for better coverage.

Usage:
    python scripts/analyze_edgar_concepts.py
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import requests
import time

from auto_researcher.data.universe import get_large_cap_universe

# Configuration
START_DATE = pd.Timestamp("2012-01-01")
END_DATE = pd.Timestamp("2024-01-01")

# Target metrics we want to compute
TARGET_METRICS = {
    'revenue': 'For value factors (P/S) and profitability margins',
    'net_income': 'For ROE, ROA, EPS',
    'gross_profit': 'For gross margin',
    'operating_income': 'For operating margin',
    'total_assets': 'For ROA',
    'stockholders_equity': 'For ROE, book value',
    'book_value_per_share': 'For P/B ratio',
    'current_assets': 'For liquidity ratios',
    'current_liabilities': 'For liquidity ratios',
    'total_liabilities': 'For debt/equity',
    'eps_basic': 'For P/E ratio',
}

# Known XBRL concepts that could map to our targets
KNOWN_CONCEPTS = {
    'revenue': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
        'SalesRevenueServicesNet',
        'TotalRevenue',
        'TotalRevenuesAndOtherIncome',
        'RevenuesNetOfInterestExpense',  # Banks
        'InterestAndDividendIncomeOperating',  # Banks
        'InterestIncomeExpenseNet',  # Banks
        'NoninterestIncome',  # Banks
        'TotalNoninterestIncome',  # Banks
        'InsurancePremiumsRevenueRecognized',  # Insurance
        'PremiumsEarnedNet',  # Insurance
        'RealEstateRevenueNet',  # REITs
        'OilAndGasRevenue',  # Energy
        'NaturalGasProductionRevenue',  # Energy
    ],
    'net_income': [
        'NetIncomeLoss',
        'NetIncomeLossAvailableToCommonStockholdersBasic',
        'NetIncomeLossAvailableToCommonStockholdersDiluted',
        'ProfitLoss',
        'IncomeLossFromContinuingOperations',
        'ComprehensiveIncomeNetOfTax',
        'NetIncomeLossAttributableToParent',
    ],
    'gross_profit': [
        'GrossProfit',
        'RevenuesNetOfCostOfSales',
    ],
    'operating_income': [
        'OperatingIncomeLoss',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',
        'IncomeLossBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
        'OperatingExpenses',  # Can compute as revenue - operating_expenses
    ],
    'total_assets': [
        'Assets',
        'TotalAssets',
    ],
    'stockholders_equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'TotalEquity',
        'TotalStockholdersEquity',
        'CommonStockHeldBySubsidiary',
        'MembersEquity',  # LLCs
        'PartnersCapital',  # Partnerships
        'LimitedPartnersCapitalAccount',
    ],
    'book_value_per_share': [
        'BookValuePerShareDiluted',
        'TangibleBookValuePerShare',
    ],
    'current_assets': [
        'AssetsCurrent',
        'TotalCurrentAssets',
    ],
    'current_liabilities': [
        'LiabilitiesCurrent',
        'TotalCurrentLiabilities',
    ],
    'total_liabilities': [
        'Liabilities',
        'TotalLiabilities',
        'LiabilitiesAndStockholdersEquity',  # Would need to subtract equity
    ],
    'eps_basic': [
        'EarningsPerShareBasic',
        'IncomeLossFromContinuingOperationsPerBasicShare',
        'BasicEarningsLossPerShare',
        'EarningsPerShareBasicAndDiluted',
    ],
}

# Cache for ticker->CIK mapping
_TICKER_TO_CIK: dict[str, str] = {}


def get_ticker_to_cik_mapping() -> dict[str, str]:
    """Fetch and cache the SEC ticker to CIK mapping."""
    global _TICKER_TO_CIK
    
    if _TICKER_TO_CIK:
        return _TICKER_TO_CIK
    
    user_agent = os.environ.get('SEC_API_USER_AGENT', 'AutoResearcher research@example.com')
    headers = {'User-Agent': user_agent}
    
    url = 'https://www.sec.gov/files/company_tickers.json'
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    for val in response.json().values():
        _TICKER_TO_CIK[val['ticker']] = str(val['cik_str']).zfill(10)
    
    print(f"  Loaded {len(_TICKER_TO_CIK)} ticker->CIK mappings")
    return _TICKER_TO_CIK


def fetch_company_facts(ticker: str, ticker_to_cik: dict) -> dict | None:
    """Fetch company facts from SEC EDGAR API."""
    user_agent = os.environ.get('SEC_API_USER_AGENT', 'AutoResearcher research@example.com')
    headers = {'User-Agent': user_agent}
    
    cik = ticker_to_cik.get(ticker)
    if not cik:
        return None
    
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    
    # Rate limit
    time.sleep(0.15)
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    [Error] {ticker}: {e}")
        return None


def analyze_available_concepts(facts: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, list[str]]:
    """
    Analyze which XBRL concepts are available for a company in the date range.
    
    Returns:
        Dict mapping concept name to list of forms where it appears (10-K, 10-Q).
    """
    us_gaap = facts.get('facts', {}).get('us-gaap', {})
    
    available = {}
    
    for concept_name, concept_data in us_gaap.items():
        units = concept_data.get('units', {})
        values = units.get('USD', units.get('shares', []))
        
        forms = set()
        for item in values:
            form = item.get('form', '')
            if form not in ('10-Q', '10-K'):
                continue
            
            end_date = item.get('end')
            if not end_date:
                continue
            
            try:
                date = pd.Timestamp(end_date)
            except:
                continue
            
            if start <= date <= end:
                forms.add(form)
        
        if forms:
            available[concept_name] = sorted(forms)
    
    return available


def find_best_concept_for_target(available: dict[str, list[str]], target: str) -> str | None:
    """Find the best available concept that matches a target metric."""
    known = KNOWN_CONCEPTS.get(target, [])
    
    for concept in known:
        if concept in available:
            return concept
    
    return None


def find_new_candidate_concepts(available: dict[str, list[str]], target: str) -> list[str]:
    """
    Find candidate concepts that might map to a target metric.
    
    Uses keyword matching to find concepts that sound like they could work.
    """
    keywords = {
        'revenue': ['Revenue', 'Sales', 'Income'],
        'net_income': ['NetIncome', 'ProfitLoss', 'Earnings'],
        'gross_profit': ['Gross', 'Profit'],
        'operating_income': ['Operating', 'Income'],
        'total_assets': ['Asset'],
        'stockholders_equity': ['Equity', 'Capital'],
        'current_assets': ['Current', 'Asset'],
        'current_liabilities': ['Current', 'Liab'],
        'total_liabilities': ['Liabil'],
        'eps_basic': ['EarningsPerShare', 'EPS'],
    }
    
    target_keywords = keywords.get(target, [])
    known = set(KNOWN_CONCEPTS.get(target, []))
    
    candidates = []
    for concept in available.keys():
        if concept in known:
            continue
        
        for kw in target_keywords:
            if kw.lower() in concept.lower():
                candidates.append(concept)
                break
    
    return candidates


def main():
    """Main analysis function."""
    print("\n" + "=" * 100)
    print("  EDGAR XBRL CONCEPT ANALYZER")
    print("=" * 100)
    
    # Get universe
    print("\n  [1/3] Loading universe and CIK mappings...")
    universe = get_large_cap_universe()
    ticker_to_cik = get_ticker_to_cik_mapping()
    
    # Track results
    coverage_by_target = {target: {'found': 0, 'missing': 0, 'concepts': defaultdict(int)} 
                          for target in TARGET_METRICS.keys()}
    missing_by_ticker = {}
    new_candidates = defaultdict(set)
    
    # Analyze each ticker
    print(f"\n  [2/3] Analyzing {len(universe)} tickers...")
    
    for i, ticker in enumerate(universe):
        if (i + 1) % 10 == 0:
            print(f"        Progress: {i + 1}/{len(universe)}")
        
        facts = fetch_company_facts(ticker, ticker_to_cik)
        if not facts:
            continue
        
        available = analyze_available_concepts(facts, START_DATE, END_DATE)
        
        ticker_missing = []
        
        for target in TARGET_METRICS.keys():
            best = find_best_concept_for_target(available, target)
            
            if best:
                coverage_by_target[target]['found'] += 1
                coverage_by_target[target]['concepts'][best] += 1
            else:
                coverage_by_target[target]['missing'] += 1
                ticker_missing.append(target)
                
                # Look for new candidates
                candidates = find_new_candidate_concepts(available, target)
                for c in candidates:
                    new_candidates[target].add(c)
        
        if ticker_missing:
            missing_by_ticker[ticker] = ticker_missing
    
    # Print results
    print("\n" + "=" * 100)
    print("  ANALYSIS RESULTS")
    print("=" * 100)
    
    # Coverage summary
    print("\n  COVERAGE BY TARGET METRIC:")
    print("  " + "-" * 80)
    print(f"  {'Metric':<25} | {'Found':<8} | {'Missing':<8} | {'Coverage':>10}")
    print("  " + "-" * 80)
    
    for target, data in coverage_by_target.items():
        total = data['found'] + data['missing']
        pct = (data['found'] / total * 100) if total > 0 else 0
        print(f"  {target:<25} | {data['found']:>8} | {data['missing']:>8} | {pct:>9.1f}%")
    
    # Top concepts used
    print("\n  TOP CONCEPTS USED PER TARGET:")
    print("  " + "-" * 80)
    
    for target, data in coverage_by_target.items():
        concepts = data['concepts']
        if concepts:
            sorted_concepts = sorted(concepts.items(), key=lambda x: -x[1])
            top3 = sorted_concepts[:3]
            print(f"\n  {target}:")
            for concept, count in top3:
                print(f"    - {concept}: {count} tickers")
    
    # New candidate concepts
    print("\n  NEW CANDIDATE CONCEPTS (not in current mapping):")
    print("  " + "-" * 80)
    
    for target, candidates in new_candidates.items():
        if candidates:
            print(f"\n  {target}:")
            for c in sorted(candidates)[:10]:
                print(f"    + {c}")
    
    # Tickers with most missing
    print("\n  TICKERS WITH MOST MISSING METRICS:")
    print("  " + "-" * 80)
    
    sorted_missing = sorted(missing_by_ticker.items(), key=lambda x: -len(x[1]))
    for ticker, missing in sorted_missing[:15]:
        print(f"  {ticker}: missing {', '.join(missing)}")
    
    print("\n" + "=" * 100)
    print("  Analysis complete.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
