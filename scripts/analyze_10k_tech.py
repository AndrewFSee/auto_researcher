"""
Fetch and analyze SEC 10-K filings for technology signals.

Uses SEC EDGAR API to fetch 10-K filings and runs them through
FilingTechSignal to detect emerging technology adoption.

Usage:
    python scripts/analyze_10k_tech.py NVDA AAPL MSFT
"""

import os
import sys
import re
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import Optional, List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from auto_researcher.models.filing_tech_signal import FilingTechSignal, FilingTechProfile

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# SEC EDGAR CLIENT
# ==============================================================================

class SECFilingFetcher:
    """Fetch 10-K filings from SEC EDGAR."""
    
    BASE_URL = "https://www.sec.gov"  # Use www.sec.gov for Archives
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    
    def __init__(self):
        self.user_agent = os.getenv("SEC_API_USER_AGENT", "AutoResearcher research@example.com")
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._cik_cache = {}
    
    def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        
        # Fetch company tickers mapping
        url = "https://www.sec.gov/files/company_tickers.json"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for entry in data.values():
                    if entry.get("ticker", "").upper() == ticker.upper():
                        cik = str(entry["cik_str"]).zfill(10)
                        self._cik_cache[ticker] = cik
                        return cik
        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
        
        return None
    
    def get_10k_filings(
        self,
        ticker: str,
        years_back: int = 3,
    ) -> List[Dict]:
        """
        Get recent 10-K filings for a company.
        
        Returns list of dicts with: accession_number, filing_date, form_type
        """
        cik = self._get_cik(ticker)
        if not cik:
            logger.warning(f"Could not find CIK for {ticker}")
            return []
        
        # Fetch submissions
        url = self.SUBMISSIONS_URL.format(cik=cik)
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            time.sleep(0.15)  # Rate limiting
            
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch submissions for {ticker}: {resp.status_code}")
                return []
            
            data = resp.json()
            filings = data.get("filings", {}).get("recent", {})
            
            # Extract 10-K and 10-K/A filings
            results = []
            cutoff = datetime.now() - timedelta(days=365 * years_back)
            
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            
            for i, form in enumerate(forms):
                if form in ["10-K", "10-K/A"]:
                    filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
                    if filing_date >= cutoff:
                        results.append({
                            "ticker": ticker,
                            "cik": cik,
                            "form_type": form,
                            "filing_date": filing_date,
                            "accession_number": accessions[i],
                        })
            
            return results[:years_back + 1]  # Usually 1 per year
            
        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            return []
    
    def fetch_filing_text(
        self,
        ticker: str,
        accession_number: str,
        cik: str,
    ) -> Optional[str]:
        """
        Fetch the full text of a filing.
        
        Returns the text content (may be HTML).
        """
        # Format accession number for URL
        acc_clean = accession_number.replace("-", "")
        
        # First, get the filing index to find the main document
        index_url = f"{self.BASE_URL}/Archives/edgar/data/{cik.lstrip('0')}/{acc_clean}/index.json"
        
        try:
            resp = requests.get(index_url, headers=self.headers, timeout=15)
            time.sleep(0.15)
            
            if resp.status_code != 200:
                logger.warning(f"Failed to get filing index: {resp.status_code}")
                return None
            
            data = resp.json()
            items = data.get("directory", {}).get("item", [])
            
            # Find the main 10-K document using multiple strategies
            main_doc = None
            ticker_lower = ticker.lower()
            
            # Strategy 1: Look for ticker-YYYYMMDD.htm pattern (common format)
            for item in items:
                name = item.get("name", "")
                if name.startswith(f"{ticker_lower}-") and name.endswith(".htm"):
                    main_doc = name
                    break
            
            # Strategy 2: Look for explicit 10-k in filename
            if not main_doc:
                for item in items:
                    name = item.get("name", "")
                    if name.endswith(".htm") and ("10-k" in name.lower() or "10k" in name.lower()):
                        main_doc = name
                        break
            
            # Strategy 3: Find largest .htm file that's not an index
            if not main_doc:
                largest_size = 0
                for item in items:
                    name = item.get("name", "")
                    size = int(item.get("size", 0) or 0)
                    if (name.endswith(".htm") and 
                        not name.startswith("0001") and
                        "index" not in name.lower() and
                        size > largest_size):
                        largest_size = size
                        main_doc = name
            
            if not main_doc:
                logger.warning(f"Could not find main document in filing")
                return None
            
            # Fetch the document
            doc_url = f"{self.BASE_URL}/Archives/edgar/data/{cik.lstrip('0')}/{acc_clean}/{main_doc}"
            resp = requests.get(doc_url, headers=self.headers, timeout=30)
            time.sleep(0.15)
            
            if resp.status_code == 200:
                return resp.text
            
        except Exception as e:
            logger.error(f"Error fetching filing text: {e}")
        
        return None
    
    def extract_relevant_sections(self, html_text: str) -> str:
        """
        Extract relevant sections from 10-K HTML.
        
        Focuses on: Item 1 (Business), Item 7 (MD&A), Risk Factors
        """
        import html
        
        # Remove HTML tags but keep text
        text = re.sub(r'<[^>]+>', ' ', html_text)
        
        # Decode HTML entities (&#160; etc)
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Try to find key sections with more flexible patterns
        sections = []
        
        # Item 1 - Business (more flexible pattern)
        item1_match = re.search(
            r'Item\s*1\.?\s*Business\s*(.*?)(?=Item\s*1[AB]\.?|Item\s*2\.?)',
            text, re.IGNORECASE | re.DOTALL
        )
        if item1_match:
            content = item1_match.group(1).strip()
            if len(content) > 100:  # Ensure we got real content
                sections.append(("Item 1 - Business", content[:50000]))
        
        # Item 7 - MD&A (more flexible)
        item7_match = re.search(
            r'Item\s*7\.?\s*Management.{0,50}Discussion(.*?)(?=Item\s*7A\.?|Item\s*8\.?)',
            text, re.IGNORECASE | re.DOTALL
        )
        if item7_match:
            content = item7_match.group(1).strip()
            if len(content) > 100:
                sections.append(("Item 7 - MD&A", content[:50000]))
        
        # Risk Factors (Item 1A)
        risk_match = re.search(
            r'Item\s*1A\.?\s*Risk\s*Factors\s*(.*?)(?=Item\s*1B\.?|Item\s*2\.?)',
            text, re.IGNORECASE | re.DOTALL
        )
        if risk_match:
            content = risk_match.group(1).strip()
            if len(content) > 100:
                sections.append(("Risk Factors", content[:30000]))
        
        if sections:
            return "\n\n".join([f"=== {name} ===\n{content}" for name, content in sections])
        
        # If section extraction fails, return first 100K chars
        return text[:100000]


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def analyze_ticker(
    ticker: str,
    fetcher: SECFilingFetcher,
    model: FilingTechSignal,
    years_back: int = 2,
) -> Optional[Dict]:
    """Analyze a single ticker's 10-K filings."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker}")
    print('='*60)
    
    # Get filings
    filings = fetcher.get_10k_filings(ticker, years_back=years_back)
    
    if not filings:
        print(f"  No 10-K filings found for {ticker}")
        return None
    
    print(f"  Found {len(filings)} 10-K filings")
    
    profiles = []
    
    for filing in filings:
        print(f"  Processing {filing['form_type']} from {filing['filing_date'].strftime('%Y-%m-%d')}...")
        
        # Fetch filing text
        text = fetcher.fetch_filing_text(
            ticker,
            filing["accession_number"],
            filing["cik"],
        )
        
        if not text:
            print(f"    Failed to fetch filing text")
            continue
        
        # Extract relevant sections
        sections = fetcher.extract_relevant_sections(text)
        print(f"    Extracted {len(sections):,} characters of relevant text")
        
        # Analyze with model
        profile = model.analyze_text(
            sections,
            ticker=ticker,
            filing_date=filing["filing_date"],
            filing_type=filing["form_type"],
        )
        
        profiles.append(profile)
        
        print(f"    Tech Intensity: {profile.tech_intensity:.1f}")
        print(f"    Emerging Focus: {profile.emerging_focus:.1%}")
        print(f"    Tech Areas: {list(profile.mentions_by_area.keys())[:5]}")
    
    if not profiles:
        return None
    
    # Compare most recent to previous
    current = profiles[0]
    previous = profiles[1] if len(profiles) > 1 else None
    
    result = model.compare_filings(current, previous)
    
    print("\n" + model.get_summary(result))
    
    return {
        "ticker": ticker,
        "result": result,
        "profiles": profiles,
    }


def main():
    """Main entry point."""
    
    # Default tickers if none provided
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        # Mix of tech and non-tech for comparison
        tickers = ["NVDA", "AAPL", "JPM", "JNJ", "TSLA"]
    
    print("="*60)
    print("10-K TECHNOLOGY SIGNAL ANALYSIS")
    print("="*60)
    print(f"Analyzing tickers: {', '.join(tickers)}")
    
    fetcher = SECFilingFetcher()
    model = FilingTechSignal()
    
    results = []
    
    for ticker in tickers:
        try:
            result = analyze_ticker(ticker, fetcher, model)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(0.5)  # Be nice to SEC servers
    
    # Summary table
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Ticker':<8} {'Tech Intensity':>15} {'Emerging %':>12} {'Trajectory':>15} {'Signal':>10}")
        print("-"*60)
        
        for r in results:
            res = r["result"]
            prof = res.current_profile
            print(f"{r['ticker']:<8} {prof.tech_intensity:>15.1f} {prof.emerging_focus:>11.1%} {res.innovation_trajectory:>15} {res.signal_strength:>+10.2f}")
        
        # Rank by signal strength
        print("\n" + "-"*60)
        print("RANKING (by innovation signal):")
        ranked = sorted(results, key=lambda x: -x["result"].signal_strength)
        for i, r in enumerate(ranked, 1):
            emoji = "🚀" if r["result"].signal_strength > 0.3 else "📈" if r["result"].signal_strength > 0 else "📉"
            print(f"  {i}. {emoji} {r['ticker']}: {r['result'].signal_strength:+.2f}")


if __name__ == "__main__":
    main()
