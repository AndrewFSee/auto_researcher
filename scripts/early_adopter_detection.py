"""
Early Adopter Detection System.

Identifies companies that were FIRST to discuss emerging technologies
in their earnings calls - potential signal for ahead-of-curve investment.

Key Signals:
1. First Mover Score: How early did they adopt new tech language vs peers?
2. Adoption Lead Time: Months ahead of industry average adoption
3. Technology Pioneer Index: Consistency of being early across multiple techs

Hypothesis: Companies that discuss new technologies early are investing
ahead of competitors and may have future competitive advantages.
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import tempfile
import os
import re

import pandas as pd
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_researcher.models.earnings_tech_signal import TRANSCRIPT_CACHE_PATH

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Emerging technologies to track - focus on recent/new terms
EMERGING_TECH_TERMS = {
    # Generative AI (emerged ~2022-2023)
    "generative_ai": {
        "terms": ["generative ai", "generative artificial intelligence", "gen ai", "genai"],
        "emerged": "2022-06",
        "category": "AI/ML"
    },
    "large_language_model": {
        "terms": ["large language model", "llm", "llms", "language model"],
        "emerged": "2022-01",
        "category": "AI/ML"
    },
    "chatgpt": {
        "terms": ["chatgpt", "chat gpt", "gpt-4", "gpt-3", "gpt4", "gpt3"],
        "emerged": "2022-11",
        "category": "AI/ML"
    },
    "copilot": {
        "terms": ["copilot", "github copilot", "ai copilot", "ai assistant"],
        "emerged": "2021-06",
        "category": "AI/ML"
    },
    "transformer": {
        "terms": ["transformer model", "transformer architecture", "attention mechanism"],
        "emerged": "2020-01",
        "category": "AI/ML"
    },
    
    # Cloud/Data (emerged at different times)
    "edge_computing": {
        "terms": ["edge computing", "edge ai", "edge inference"],
        "emerged": "2018-01",
        "category": "Cloud"
    },
    "multicloud": {
        "terms": ["multi-cloud", "multicloud", "hybrid cloud"],
        "emerged": "2018-01",
        "category": "Cloud"
    },
    
    # Autonomous
    "autonomous_vehicle": {
        "terms": ["autonomous vehicle", "self-driving", "robotaxi", "level 4", "level 5"],
        "emerged": "2016-01",
        "category": "Autonomous"
    },
    "fsd": {
        "terms": ["full self-driving", "fsd", "autopilot"],
        "emerged": "2019-01",
        "category": "Autonomous"
    },
    
    # Semiconductors
    "chiplet": {
        "terms": ["chiplet", "chiplets", "advanced packaging"],
        "emerged": "2020-01",
        "category": "Semiconductors"
    },
    "3nm": {
        "terms": ["3nm", "3 nanometer", "2nm", "2 nanometer"],
        "emerged": "2021-01",
        "category": "Semiconductors"
    },
    
    # Emerging/Frontier
    "quantum": {
        "terms": ["quantum computing", "quantum computer", "qubit", "quantum supremacy"],
        "emerged": "2019-01",
        "category": "Quantum"
    },
    "metaverse": {
        "terms": ["metaverse", "virtual world", "digital twin"],
        "emerged": "2021-10",
        "category": "Metaverse"
    },
    "web3": {
        "terms": ["web3", "web 3.0", "decentralized", "blockchain"],
        "emerged": "2021-01",
        "category": "Crypto/Web3"
    },
    
    # Biotech
    "mrna": {
        "terms": ["mrna", "messenger rna", "mrna technology"],
        "emerged": "2020-03",
        "category": "Biotech"
    },
    "crispr": {
        "terms": ["crispr", "gene editing", "gene therapy"],
        "emerged": "2017-01",
        "category": "Biotech"
    },
    
    # Clean Tech
    "carbon_neutral": {
        "terms": ["carbon neutral", "net zero", "carbon negative", "decarbonization"],
        "emerged": "2019-01",
        "category": "Clean Tech"
    },
    "ev_battery": {
        "terms": ["solid state battery", "ev battery", "battery technology", "lithium"],
        "emerged": "2018-01",
        "category": "Clean Tech"
    },
}


@dataclass
class TechAdoption:
    """Record of when a company first mentioned a technology."""
    ticker: str
    tech_id: str
    tech_name: str
    category: str
    first_mention_date: datetime
    first_mention_quarter: str
    context: str  # The actual quote
    industry_avg_adoption: Optional[datetime] = None
    lead_months: Optional[float] = None  # Months ahead of industry


@dataclass
class EarlyAdopterProfile:
    """Profile of a company's early adoption behavior."""
    ticker: str
    total_techs_adopted: int
    techs_adopted_early: int  # Before industry average
    avg_lead_months: float  # Average months ahead
    pioneer_score: float  # 0-1 score
    earliest_adoptions: List[TechAdoption]
    categories_pioneered: Dict[str, int]


def load_transcripts() -> pd.DataFrame:
    """Load all transcripts from cache."""
    if not os.path.exists(TRANSCRIPT_CACHE_PATH):
        raise FileNotFoundError(f"Transcript cache not found: {TRANSCRIPT_CACHE_PATH}")
    
    df = pd.read_parquet(
        TRANSCRIPT_CACHE_PATH,
        columns=["symbol", "report_date", "fiscal_quarter", "fiscal_year", "transcripts"],
    )
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def find_first_mentions(df: pd.DataFrame, tickers: List[str]) -> Dict[str, List[TechAdoption]]:
    """
    Find when each company first mentioned each emerging technology.
    
    Returns dict of ticker -> list of TechAdoption records
    """
    
    # Compile patterns
    patterns = {}
    for tech_id, config in EMERGING_TECH_TERMS.items():
        pattern = r'\b(' + '|'.join(re.escape(t) for t in config["terms"]) + r')\b'
        patterns[tech_id] = re.compile(pattern, re.IGNORECASE)
    
    # Track first mentions
    first_mentions = defaultdict(dict)  # ticker -> tech_id -> TechAdoption
    
    print(f"Scanning {len(tickers)} tickers for first mentions of {len(EMERGING_TECH_TERMS)} technologies...")
    
    for ticker in tickers:
        ticker_df = df[df["symbol"] == ticker].sort_values("report_date", ascending=True)
        
        if ticker_df.empty:
            continue
        
        for _, row in ticker_df.iterrows():
            # Convert transcript to text
            raw = row.get("transcripts", [])
            if isinstance(raw, list):
                parts = []
                for item in raw:
                    if isinstance(item, dict):
                        parts.append(f"{item.get('speaker', '')}: {item.get('content', '')}")
                    else:
                        parts.append(str(item))
                content = "\n".join(parts)
            else:
                content = str(raw)
            
            date = row["report_date"].to_pydatetime()
            quarter = f"Q{row['fiscal_quarter']} {row['fiscal_year']}"
            
            # Check for each technology
            for tech_id, pattern in patterns.items():
                # Skip if already found for this ticker
                if tech_id in first_mentions[ticker]:
                    continue
                
                match = pattern.search(content)
                if match:
                    # Get context
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end].strip()
                    
                    config = EMERGING_TECH_TERMS[tech_id]
                    
                    first_mentions[ticker][tech_id] = TechAdoption(
                        ticker=ticker,
                        tech_id=tech_id,
                        tech_name=tech_id.replace("_", " ").title(),
                        category=config["category"],
                        first_mention_date=date,
                        first_mention_quarter=quarter,
                        context=context,
                    )
    
    # Convert to list format
    result = {}
    for ticker, techs in first_mentions.items():
        result[ticker] = list(techs.values())
    
    return result


def calculate_industry_averages(all_mentions: Dict[str, List[TechAdoption]]) -> Dict[str, datetime]:
    """Calculate when each technology was adopted by the industry on average."""
    
    tech_dates = defaultdict(list)
    
    for ticker, adoptions in all_mentions.items():
        for adoption in adoptions:
            tech_dates[adoption.tech_id].append(adoption.first_mention_date)
    
    averages = {}
    for tech_id, dates in tech_dates.items():
        if dates:
            # Use median adoption date
            sorted_dates = sorted(dates)
            median_idx = len(sorted_dates) // 2
            averages[tech_id] = sorted_dates[median_idx]
    
    return averages


def calculate_lead_times(
    all_mentions: Dict[str, List[TechAdoption]], 
    industry_avgs: Dict[str, datetime]
) -> Dict[str, List[TechAdoption]]:
    """Calculate how many months each company was ahead of industry average."""
    
    for ticker, adoptions in all_mentions.items():
        for adoption in adoptions:
            if adoption.tech_id in industry_avgs:
                avg_date = industry_avgs[adoption.tech_id]
                adoption.industry_avg_adoption = avg_date
                
                # Calculate lead time in months
                delta = avg_date - adoption.first_mention_date
                adoption.lead_months = delta.days / 30.44
    
    return all_mentions


def build_pioneer_profiles(all_mentions: Dict[str, List[TechAdoption]]) -> Dict[str, EarlyAdopterProfile]:
    """Build early adopter profiles for each company."""
    
    profiles = {}
    
    for ticker, adoptions in all_mentions.items():
        if not adoptions:
            continue
        
        # Count early adoptions
        early_adoptions = [a for a in adoptions if a.lead_months and a.lead_months > 0]
        
        # Average lead time
        lead_times = [a.lead_months for a in adoptions if a.lead_months is not None]
        avg_lead = np.mean(lead_times) if lead_times else 0
        
        # Pioneer score (0-1): combination of breadth and earliness
        breadth_score = min(len(adoptions) / 10, 1.0)  # Max at 10 techs
        earliness_score = min(max(avg_lead / 12, 0), 1.0)  # Max at 12 months ahead
        pioneer_score = 0.5 * breadth_score + 0.5 * earliness_score
        
        # Categories pioneered
        categories = defaultdict(int)
        for a in early_adoptions:
            categories[a.category] += 1
        
        # Sort adoptions by lead time
        sorted_adoptions = sorted(adoptions, key=lambda x: -(x.lead_months or 0))
        
        profiles[ticker] = EarlyAdopterProfile(
            ticker=ticker,
            total_techs_adopted=len(adoptions),
            techs_adopted_early=len(early_adoptions),
            avg_lead_months=avg_lead,
            pioneer_score=pioneer_score,
            earliest_adoptions=sorted_adoptions[:5],
            categories_pioneered=dict(categories),
        )
    
    return profiles


def get_price_returns(ticker: str, start_date: datetime, months: int = 12) -> Optional[float]:
    """Get total return over period."""
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return None
    
    end_date = start_date + timedelta(days=months * 30)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {"apiKey": api_key, "adjusted": "true", "limit": 50000}
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results") and len(data["results"]) > 20:
                results = data["results"]
                start_price = results[0]["c"]
                end_price = results[-1]["c"]
                return (end_price - start_price) / start_price
    except:
        pass
    
    return None


def run_early_adopter_analysis():
    """Main analysis."""
    
    print("=" * 70)
    print("EARLY ADOPTER DETECTION SYSTEM")
    print("Finding companies that discussed emerging tech FIRST")
    print("=" * 70)
    
    # Load data
    print("\nLoading transcript data...")
    df = load_transcripts()
    print(f"Loaded {len(df):,} transcripts")
    
    # Filter to reasonable date range (2018-2024)
    df = df[(df["report_date"] >= "2018-01-01") & (df["report_date"] < "2025-01-01")]
    print(f"Filtered to {len(df):,} transcripts (2018-2024)")
    
    # Define universe of stocks to analyze
    tickers = [
        # Big Tech
        "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
        # Semiconductors
        "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "ASML",
        # Software/Cloud
        "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PLTR", "NET", "DDOG", "ZS",
        # Enterprise Tech
        "IBM", "HPQ", "DELL", "CSCO", "VMW",
        # Fintech
        "V", "MA", "PYPL", "SQ", "COIN",
        # Healthcare/Biotech
        "JNJ", "PFE", "MRNA", "LLY", "ABBV", "TMO", "DHR",
        # Auto
        "F", "GM", "RIVN",
        # Traditional (control group)
        "JPM", "BAC", "GS", "WMT", "TGT", "KO", "PEP", "PG", "XOM", "CVX",
    ]
    
    # Find first mentions
    all_mentions = find_first_mentions(df, tickers)
    
    # Calculate industry averages
    print("\nCalculating industry adoption timelines...")
    industry_avgs = calculate_industry_averages(all_mentions)
    
    # Calculate lead times
    all_mentions = calculate_lead_times(all_mentions, industry_avgs)
    
    # Build profiles
    profiles = build_pioneer_profiles(all_mentions)
    
    # Display results
    print("\n" + "=" * 70)
    print("TECHNOLOGY ADOPTION TIMELINE (Industry-Wide)")
    print("=" * 70)
    
    print(f"\n{'Technology':<25} {'Category':<15} {'First Adopter':<10} {'Date':<12} {'Industry Avg':<12}")
    print("-" * 74)
    
    # Find first adopter for each tech
    first_adopters = {}
    for ticker, adoptions in all_mentions.items():
        for a in adoptions:
            if a.tech_id not in first_adopters or a.first_mention_date < first_adopters[a.tech_id][1]:
                first_adopters[a.tech_id] = (ticker, a.first_mention_date)
    
    for tech_id in sorted(first_adopters.keys(), key=lambda x: first_adopters[x][1]):
        ticker, date = first_adopters[tech_id]
        config = EMERGING_TECH_TERMS[tech_id]
        avg = industry_avgs.get(tech_id)
        avg_str = avg.strftime("%Y-%m") if avg else "N/A"
        print(f"{tech_id:<25} {config['category']:<15} {ticker:<10} {date.strftime('%Y-%m'):<12} {avg_str:<12}")
    
    # Top pioneers
    print("\n" + "=" * 70)
    print("TOP TECHNOLOGY PIONEERS (Early Adopter Rankings)")
    print("=" * 70)
    
    sorted_profiles = sorted(profiles.values(), key=lambda x: -x.pioneer_score)
    
    print(f"\n{'Rank':<5} {'Ticker':<8} {'Score':<8} {'Techs':<8} {'Early':<8} {'Avg Lead':<10} {'Top Categories'}")
    print("-" * 80)
    
    for i, p in enumerate(sorted_profiles[:25], 1):
        cats = ", ".join([f"{k}({v})" for k, v in sorted(p.categories_pioneered.items(), key=lambda x: -x[1])[:2]])
        print(f"{i:<5} {p.ticker:<8} {p.pioneer_score:.2f}     {p.total_techs_adopted:<8} {p.techs_adopted_early:<8} {p.avg_lead_months:>+6.1f} mo   {cats}")
    
    # Specific technology leaders
    print("\n" + "=" * 70)
    print("FIRST MOVERS BY TECHNOLOGY")
    print("=" * 70)
    
    for tech_id, config in sorted(EMERGING_TECH_TERMS.items(), key=lambda x: x[1]["emerged"]):
        # Find top 3 earliest adopters
        adopters = []
        for ticker, adoptions in all_mentions.items():
            for a in adoptions:
                if a.tech_id == tech_id:
                    adopters.append((ticker, a.first_mention_date, a.lead_months or 0))
        
        if adopters:
            adopters.sort(key=lambda x: x[1])
            print(f"\n{tech_id.replace('_', ' ').upper()} ({config['category']}):")
            print(f"  Industry emerged: {config['emerged']}")
            for ticker, date, lead in adopters[:5]:
                lead_str = f"+{lead:.0f}mo" if lead > 0 else f"{lead:.0f}mo"
                print(f"    {ticker}: {date.strftime('%Y-%m')} ({lead_str} vs avg)")
    
    # Generative AI deep dive
    print("\n" + "=" * 70)
    print("GENERATIVE AI PIONEERS (Deep Dive)")
    print("=" * 70)
    
    genai_techs = ["generative_ai", "large_language_model", "chatgpt", "copilot"]
    genai_scores = defaultdict(float)
    genai_details = defaultdict(list)
    
    for ticker, adoptions in all_mentions.items():
        for a in adoptions:
            if a.tech_id in genai_techs:
                genai_scores[ticker] += (a.lead_months or 0)
                genai_details[ticker].append(f"{a.tech_id}: {a.first_mention_date.strftime('%Y-%m')}")
    
    print(f"\n{'Ticker':<8} {'GenAI Score':<12} {'Technologies Discussed'}")
    print("-" * 70)
    
    for ticker, score in sorted(genai_scores.items(), key=lambda x: -x[1])[:15]:
        details = ", ".join(genai_details[ticker])
        print(f"{ticker:<8} {score:>+8.1f} mo   {details}")
    
    # Backtest: Do early adopters outperform?
    print("\n" + "=" * 70)
    print("BACKTEST: Early Adopter Returns (2023)")
    print("=" * 70)
    
    # Get returns for profiles from Q3 2023
    returns_data = []
    
    print("\nFetching returns data...")
    for profile in sorted_profiles[:30]:
        ticker = profile.ticker
        # Get 12-month return from Oct 2023
        ret = get_price_returns(ticker, datetime(2023, 10, 1), months=12)
        if ret is not None:
            returns_data.append({
                "ticker": ticker,
                "pioneer_score": profile.pioneer_score,
                "avg_lead_months": profile.avg_lead_months,
                "techs_adopted": profile.total_techs_adopted,
                "return_12m": ret,
            })
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        
        # Split by pioneer score
        median_score = returns_df["pioneer_score"].median()
        high_pioneers = returns_df[returns_df["pioneer_score"] >= median_score]
        low_pioneers = returns_df[returns_df["pioneer_score"] < median_score]
        
        print(f"\nHigh Pioneer Score (>= {median_score:.2f}): {len(high_pioneers)} companies")
        print(f"  Average 12-month return: {high_pioneers['return_12m'].mean():+.1%}")
        print(f"  Companies: {', '.join(high_pioneers['ticker'].tolist())}")
        
        print(f"\nLow Pioneer Score (< {median_score:.2f}): {len(low_pioneers)} companies")
        print(f"  Average 12-month return: {low_pioneers['return_12m'].mean():+.1%}")
        print(f"  Companies: {', '.join(low_pioneers['ticker'].tolist())}")
        
        print(f"\nSpread (High - Low): {high_pioneers['return_12m'].mean() - low_pioneers['return_12m'].mean():+.1%}")
        
        # Correlation
        print("\nCorrelation Analysis:")
        corr, pval = stats.pearsonr(returns_df["pioneer_score"], returns_df["return_12m"])
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  Pioneer Score vs 12mo Return: r={corr:+.3f}, p={pval:.3f} {sig}")
        
        corr2, pval2 = stats.pearsonr(returns_df["avg_lead_months"], returns_df["return_12m"])
        sig2 = "***" if pval2 < 0.01 else "**" if pval2 < 0.05 else "*" if pval2 < 0.1 else ""
        print(f"  Avg Lead Months vs 12mo Return: r={corr2:+.3f}, p={pval2:.3f} {sig2}")
    
    # Save detailed results
    print("\n" + "=" * 70)
    print("SAMPLE EARLY ADOPTION QUOTES")
    print("=" * 70)
    
    # Show some example quotes from pioneers
    for ticker in ["NVDA", "MSFT", "GOOGL", "META"]:
        if ticker in profiles:
            p = profiles[ticker]
            print(f"\n{ticker} (Pioneer Score: {p.pioneer_score:.2f}):")
            for a in p.earliest_adoptions[:2]:
                print(f"  {a.tech_name} ({a.first_mention_date.strftime('%Y-%m')}, +{a.lead_months:.0f}mo ahead):")
                print(f"    \"{a.context[:150]}...\"")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_early_adopter_analysis()
