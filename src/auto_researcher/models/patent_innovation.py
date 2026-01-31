"""
Patent-Based Innovation Signal.

Patents are filed 12-18 months before products launch and represent
genuine R&D investment, making them better signals than news for
detecting emerging technology adoption.

================================================================================
SIGNAL HYPOTHESIS
================================================================================

Patents provide legal, forward-looking signals:
    1. Patent filing = committed R&D investment
    2. Patent claims = specific technology direction
    3. Patent citations = building on prior art
    4. Patent velocity = acceleration in innovation

Academic basis:
    - Deng et al. (1999): R&D and patents predict stock returns
    - Hall et al. (2005): Patent citations predict firm value
    - Hirshleifer et al. (2013): Innovative efficiency predicts returns

Key insight: Patent filings are PUBLIC but underanalyzed.
Most investors look at news, not patent databases.

================================================================================

Usage:
    from auto_researcher.models.patent_innovation import PatentInnovationModel
    
    model = PatentInnovationModel()
    signal = await model.analyze_company("AAPL")
    print(f"Innovation Score: {signal.innovation_score}")
    print(f"Key Tech Areas: {signal.top_tech_areas}")
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)

# USPTO CPC (Cooperative Patent Classification) codes for key technologies
# See: https://www.uspto.gov/web/patents/classification/cpc/html/cpc.html

TECH_CPC_CODES = {
    # AI/ML
    "ai_machine_learning": {
        "codes": ["G06N3", "G06N20", "G06N5"],  # Neural networks, Machine learning, Knowledge processing
        "name": "AI/Machine Learning",
        "maturity": 2,
    },
    
    # Computer Vision
    "computer_vision": {
        "codes": ["G06V", "G06T7"],  # Image/video recognition
        "name": "Computer Vision",
        "maturity": 2,
    },
    
    # Natural Language Processing
    "nlp": {
        "codes": ["G06F40", "G10L15"],  # NLP, Speech recognition
        "name": "Natural Language Processing",
        "maturity": 2,
    },
    
    # Quantum Computing
    "quantum": {
        "codes": ["G06N10", "H03K19/195"],  # Quantum computing
        "name": "Quantum Computing",
        "maturity": 1,
    },
    
    # Autonomous Vehicles
    "autonomous_vehicles": {
        "codes": ["B60W60", "G05D1"],  # Automated driving, Control of vehicles
        "name": "Autonomous Vehicles",
        "maturity": 2,
    },
    
    # Robotics
    "robotics": {
        "codes": ["B25J9", "B25J13"],  # Robots, Sensors for robots
        "name": "Robotics",
        "maturity": 2,
    },
    
    # Batteries/Energy Storage
    "batteries": {
        "codes": ["H01M10", "H02J7"],  # Batteries, Charging
        "name": "Batteries/Energy Storage",
        "maturity": 2,
    },
    
    # Semiconductor
    "semiconductor": {
        "codes": ["H01L21", "H01L29"],  # Semiconductor manufacturing
        "name": "Semiconductors",
        "maturity": 2,
    },
    
    # Biotech/Gene
    "biotech": {
        "codes": ["C12N15", "C12Q1/68"],  # Genetic engineering, DNA analysis
        "name": "Biotechnology",
        "maturity": 2,
    },
    
    # Blockchain/Crypto
    "blockchain": {
        "codes": ["G06Q20/38", "H04L9/06"],  # Payment protocols, Cryptography
        "name": "Blockchain/Crypto",
        "maturity": 3,
    },
    
    # 5G/Wireless
    "wireless": {
        "codes": ["H04W72", "H04L5/00"],  # Wireless resource management
        "name": "5G/Wireless",
        "maturity": 2,
    },
    
    # AR/VR
    "ar_vr": {
        "codes": ["G02B27/01", "G06F3/01"],  # Head-mounted displays, VR interfaces
        "name": "AR/VR",
        "maturity": 2,
    },
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class PatentRecord:
    """A single patent record."""
    patent_number: str
    title: str
    abstract: str
    filing_date: datetime
    grant_date: Optional[datetime]
    assignee: str
    cpc_codes: List[str]
    claims_count: int
    citation_count: int = 0
    
    # Derived
    tech_areas: List[str] = field(default_factory=list)


@dataclass
class PatentPortfolio:
    """Summary of a company's patent portfolio."""
    assignee_name: str
    ticker: Optional[str]
    analysis_date: datetime
    
    # Counts
    total_patents: int
    recent_patents: int  # Last 2 years
    patents_by_tech: Dict[str, int]
    
    # Velocity metrics
    filing_velocity: float  # Patents per year
    velocity_trend: str  # "accelerating", "stable", "decelerating"
    
    # Quality metrics
    avg_claims: float
    avg_citations: float
    
    # Innovation score
    innovation_score: float  # 0-100
    emerging_tech_ratio: float  # % in emerging (maturity=1) tech


@dataclass
class PatentInnovationSignal:
    """Innovation signal based on patent analysis."""
    ticker: str
    assignee: str
    
    # Portfolio
    portfolio: PatentPortfolio
    
    # Peer comparison
    peer_percentile: Optional[float] = None
    peer_velocity_percentile: Optional[float] = None
    
    # Signal
    innovation_rank: str = "average"  # "laggard", "average", "leader", "pioneer"
    signal_strength: float = 0.0
    
    # Key findings
    top_tech_areas: List[str] = field(default_factory=list)
    accelerating_areas: List[str] = field(default_factory=list)
    emerging_investments: List[str] = field(default_factory=list)
    
    # Forward indicators
    patent_moat_score: float = 0.0  # Breadth and depth of IP
    
    # Caveats
    caveats: List[str] = field(default_factory=list)


# ==============================================================================
# USPTO API CLIENT
# ==============================================================================

class USPTOClient:
    """
    Client for USPTO Patent Data.
    
    Uses multiple sources:
    1. USPTO PatentCenter API (new)
    2. Google Patents (scraping as backup)
    3. USPTO bulk data (for historical)
    """
    
    # USPTO PatentCenter API (new endpoint)
    PATENT_CENTER_SEARCH = "https://ppubs.uspto.gov/dirsearch-public/searches/searchWithBeFamily"
    
    # Backup: Google Patents
    GOOGLE_PATENTS_URL = "https://patents.google.com"
    
    def __init__(self):
        """Initialize USPTO client."""
        self.session = None
        self.headers = {
            "User-Agent": "AutoResearcher/1.0 (research tool)",
            "Accept": "application/json",
        }
    
    def search_patents_sync(
        self,
        assignee: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 100,
    ) -> List[Dict]:
        """
        Search patents by assignee using USPTO PatentCenter API.
        """
        import requests
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*5)
        if end_date is None:
            end_date = datetime.now()
        
        # USPTO PatentCenter search payload
        payload = {
            "searchText": f'AN/"{assignee}"',  # Assignee Name search
            "fq": [
                f"appFilingDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
            ],
            "fl": "*",
            "mm": "100%",
            "df": "patentTitle",
            "facet": False,
            "sort": "datePublished desc",
            "start": 0,
            "rows": max_results,
        }
        
        try:
            response = requests.post(
                self.PATENT_CENTER_SEARCH,
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("patents", data.get("results", []))
            else:
                logger.debug(f"USPTO PatentCenter API returned: {response.status_code}")
                # Fall back to scraping Google Patents
                return self._search_google_patents(assignee, start_date, max_results)
                
        except Exception as e:
            logger.debug(f"USPTO search failed: {e}, trying Google Patents")
            return self._search_google_patents(assignee, start_date, max_results)
    
    def _search_google_patents(
        self,
        assignee: str,
        start_date: datetime,
        max_results: int = 50,
    ) -> List[Dict]:
        """
        Search Google Patents as a backup.
        
        Google Patents has a public interface but no official API.
        We scrape the search results page.
        """
        import requests
        from urllib.parse import quote
        
        # Construct search URL
        # Format: assignee:(NVIDIA) after:2020-01-01 country:US
        query = f'assignee:({assignee}) after:{start_date.strftime("%Y-%m-%d")} country:US'
        url = f"{self.GOOGLE_PATENTS_URL}/?q={quote(query)}&oq={quote(query)}"
        
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=20,
            )
            
            if response.status_code != 200:
                logger.debug(f"Google Patents returned {response.status_code}")
                return []
            
            # Parse the HTML to extract patent info
            patents = self._parse_google_patents_html(response.text, max_results)
            return patents
            
        except Exception as e:
            logger.error(f"Google Patents search failed: {e}")
            return []
    
    def _parse_google_patents_html(self, html: str, max_results: int) -> List[Dict]:
        """Parse Google Patents search results HTML."""
        import re
        
        patents = []
        
        # Find patent entries (simplified parsing)
        # Google Patents uses structured data we can extract
        
        # Look for patent numbers (US pattern)
        patent_pattern = r'US(\d{7,11})[A-Z]?\d*'
        title_pattern = r'<h3[^>]*>([^<]+)</h3>'
        
        patent_nums = re.findall(patent_pattern, html)
        titles = re.findall(title_pattern, html)
        
        for i, pnum in enumerate(patent_nums[:max_results]):
            title = titles[i] if i < len(titles) else f"US Patent {pnum}"
            patents.append({
                "patent_number": f"US{pnum}",
                "patent_title": title,
                "patent_abstract": "",
                "patent_date": datetime.now().strftime("%Y-%m-%d"),  # Approximate
                "patent_num_claims": 10,  # Default
                "cpc_category": [],
            })
        
        return patents
    
    def get_company_patents(
        self,
        company_name: str,
        ticker: Optional[str] = None,
        years_back: int = 5,
    ) -> List[PatentRecord]:
        """
        Get patents for a company.
        
        Note: Company names in patents may differ from common names.
        E.g., "APPLE INC" not "Apple", "NVIDIA CORP" not "NVIDIA"
        """
        # Common name mappings (patent assignee names)
        name_mappings = {
            "AAPL": "APPLE INC",
            "GOOGL": "GOOGLE LLC",
            "GOOG": "GOOGLE LLC",
            "MSFT": "MICROSOFT CORPORATION",
            "NVDA": "NVIDIA CORPORATION",
            "META": "META PLATFORMS",
            "AMZN": "AMAZON TECHNOLOGIES",
            "TSLA": "TESLA INC",
            "AMD": "ADVANCED MICRO DEVICES",
            "INTC": "INTEL CORPORATION",
            "IBM": "INTERNATIONAL BUSINESS MACHINES",
            "QCOM": "QUALCOMM",
            "AVGO": "BROADCOM",
            "CRM": "SALESFORCE",
            "ORCL": "ORACLE",
            "CSCO": "CISCO",
        }
        
        search_name = name_mappings.get(ticker, company_name.upper())
        
        start_date = datetime.now() - timedelta(days=365*years_back)
        raw_patents = self.search_patents_sync(search_name, start_date=start_date)
        
        # Convert to PatentRecord
        records = []
        for p in raw_patents:
            try:
                cpc_codes = []
                cpc_raw = p.get("cpc_category", p.get("cpcCodes", []))
                if cpc_raw:
                    for cpc in cpc_raw:
                        if isinstance(cpc, dict):
                            cpc_codes.append(cpc.get("cpc_category", cpc.get("code", "")))
                        else:
                            cpc_codes.append(str(cpc))
                
                # Handle different date formats
                date_str = p.get("patent_date", p.get("publicationDate", "2020-01-01"))
                if isinstance(date_str, str):
                    if len(date_str) == 8:  # YYYYMMDD format
                        filing_date = datetime.strptime(date_str, "%Y%m%d")
                    else:
                        filing_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                else:
                    filing_date = datetime.now()
                
                record = PatentRecord(
                    patent_number=p.get("patent_number", p.get("patentNumber", "")),
                    title=p.get("patent_title", p.get("inventionTitle", "")),
                    abstract=p.get("patent_abstract", p.get("abstract", "")) or "",
                    filing_date=filing_date,
                    grant_date=None,
                    assignee=search_name,
                    cpc_codes=cpc_codes,
                    claims_count=int(p.get("patent_num_claims", p.get("claimCount", 1))),
                )
                
                # Map CPC codes to tech areas
                for tech_id, config in TECH_CPC_CODES.items():
                    for cpc in record.cpc_codes:
                        if any(cpc.startswith(code) for code in config["codes"]):
                            if tech_id not in record.tech_areas:
                                record.tech_areas.append(tech_id)
                
                # Also extract tech from title/abstract if no CPC
                if not record.tech_areas:
                    text = f"{record.title} {record.abstract}".lower()
                    if any(kw in text for kw in ["neural", "machine learning", "artificial intelligence", "deep learning"]):
                        record.tech_areas.append("ai_machine_learning")
                    if any(kw in text for kw in ["quantum", "qubit"]):
                        record.tech_areas.append("quantum")
                    if any(kw in text for kw in ["autonomous", "self-driving"]):
                        record.tech_areas.append("autonomous_vehicles")
                    if any(kw in text for kw in ["gpu", "graphics", "accelerator", "tensor"]):
                        record.tech_areas.append("semiconductor")
                
                records.append(record)
                
            except Exception as e:
                logger.debug(f"Failed to parse patent: {e}")
                continue
        
        return records


# ==============================================================================
# PATENT INNOVATION MODEL
# ==============================================================================

class PatentInnovationModel:
    """
    Analyzes company innovation through patent filings.
    
    Patents are better than news for tech adoption because:
    1. Legal commitment (not just marketing)
    2. Filed 12-18 months before product launch
    3. Detailed technical claims
    4. Searchable by technology classification
    """
    
    def __init__(self):
        """Initialize the patent innovation model."""
        self.client = USPTOClient()
        self.tech_cpc = TECH_CPC_CODES
        logger.info("PatentInnovationModel initialized")
    
    def build_portfolio(
        self,
        patents: List[PatentRecord],
        assignee: str,
        ticker: Optional[str] = None,
    ) -> PatentPortfolio:
        """Build a patent portfolio summary."""
        if not patents:
            return PatentPortfolio(
                assignee_name=assignee,
                ticker=ticker,
                analysis_date=datetime.now(),
                total_patents=0,
                recent_patents=0,
                patents_by_tech={},
                filing_velocity=0,
                velocity_trend="unknown",
                avg_claims=0,
                avg_citations=0,
                innovation_score=0,
                emerging_tech_ratio=0,
            )
        
        # Count by tech area
        tech_counts = defaultdict(int)
        for p in patents:
            for tech in p.tech_areas:
                tech_counts[tech] += 1
        
        # Recent patents (last 2 years)
        two_years_ago = datetime.now() - timedelta(days=730)
        recent = [p for p in patents if p.filing_date > two_years_ago]
        
        # Filing velocity
        date_range = (datetime.now() - min(p.filing_date for p in patents)).days / 365
        velocity = len(patents) / max(date_range, 1)
        
        # Velocity trend (compare last year to year before)
        one_year_ago = datetime.now() - timedelta(days=365)
        last_year = [p for p in patents if p.filing_date > one_year_ago]
        prev_year = [p for p in patents 
                    if one_year_ago > p.filing_date > two_years_ago]
        
        if len(last_year) > len(prev_year) * 1.2:
            trend = "accelerating"
        elif len(last_year) < len(prev_year) * 0.8:
            trend = "decelerating"
        else:
            trend = "stable"
        
        # Quality metrics
        avg_claims = sum(p.claims_count for p in patents) / len(patents)
        avg_citations = sum(p.citation_count for p in patents) / len(patents)
        
        # Emerging tech ratio
        emerging_count = 0
        for tech_id, count in tech_counts.items():
            if self.tech_cpc.get(tech_id, {}).get("maturity") == 1:
                emerging_count += count
        total_tech = sum(tech_counts.values())
        emerging_ratio = emerging_count / total_tech if total_tech > 0 else 0
        
        # Innovation score (0-100)
        # Components: volume, velocity, quality, emerging focus
        volume_score = min(len(patents) / 100, 1) * 25
        velocity_score = min(velocity / 20, 1) * 25
        quality_score = min(avg_claims / 20, 1) * 25
        emerging_score = emerging_ratio * 25
        
        innovation_score = volume_score + velocity_score + quality_score + emerging_score
        
        return PatentPortfolio(
            assignee_name=assignee,
            ticker=ticker,
            analysis_date=datetime.now(),
            total_patents=len(patents),
            recent_patents=len(recent),
            patents_by_tech=dict(tech_counts),
            filing_velocity=velocity,
            velocity_trend=trend,
            avg_claims=avg_claims,
            avg_citations=avg_citations,
            innovation_score=innovation_score,
            emerging_tech_ratio=emerging_ratio,
        )
    
    def analyze_company(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        years_back: int = 5,
    ) -> PatentInnovationSignal:
        """
        Analyze a company's innovation through patents.
        
        Args:
            ticker: Stock ticker
            company_name: Company name (for patent search)
            years_back: Years of patent history to analyze
            
        Returns:
            PatentInnovationSignal
        """
        # Get patents
        patents = self.client.get_company_patents(
            company_name or ticker,
            ticker=ticker,
            years_back=years_back,
        )
        
        # Build portfolio
        portfolio = self.build_portfolio(patents, company_name or ticker, ticker)
        
        # Build signal
        signal = PatentInnovationSignal(
            ticker=ticker,
            assignee=portfolio.assignee_name,
            portfolio=portfolio,
        )
        
        # Determine rank
        if portfolio.innovation_score >= 75:
            signal.innovation_rank = "pioneer"
            signal.signal_strength = 0.8
        elif portfolio.innovation_score >= 50:
            signal.innovation_rank = "leader"
            signal.signal_strength = 0.5
        elif portfolio.innovation_score >= 25:
            signal.innovation_rank = "average"
            signal.signal_strength = 0.0
        else:
            signal.innovation_rank = "laggard"
            signal.signal_strength = -0.3
        
        # Top tech areas
        if portfolio.patents_by_tech:
            sorted_tech = sorted(portfolio.patents_by_tech.items(), key=lambda x: -x[1])
            signal.top_tech_areas = [self.tech_cpc.get(t, {}).get("name", t) 
                                     for t, _ in sorted_tech[:5]]
        
        # Emerging investments
        for tech_id, count in portfolio.patents_by_tech.items():
            if self.tech_cpc.get(tech_id, {}).get("maturity") == 1 and count >= 2:
                signal.emerging_investments.append(
                    self.tech_cpc[tech_id]["name"]
                )
        
        # Accelerating (based on velocity trend)
        if portfolio.velocity_trend == "accelerating":
            signal.accelerating_areas.append("Overall R&D")
        
        # Patent moat score
        # Breadth (number of tech areas) * Depth (average patents per area)
        if portfolio.patents_by_tech:
            breadth = len(portfolio.patents_by_tech)
            depth = sum(portfolio.patents_by_tech.values()) / breadth
            signal.patent_moat_score = min((breadth * np.sqrt(depth)) / 20, 1.0)
        
        # Caveats
        if len(patents) < 10:
            signal.caveats.append("Low patent count - may not be patent-focused company")
        if portfolio.velocity_trend == "decelerating":
            signal.caveats.append("Patent filing velocity is decreasing")
        
        return signal
    
    def get_summary(self, signal: PatentInnovationSignal) -> str:
        """Generate human-readable summary."""
        lines = [f"Patent Innovation Analysis for {signal.ticker}"]
        lines.append("=" * 50)
        
        p = signal.portfolio
        lines.append(f"Assignee: {p.assignee_name}")
        lines.append(f"Total Patents: {p.total_patents}")
        lines.append(f"Recent (2yr): {p.recent_patents}")
        lines.append(f"Filing Velocity: {p.filing_velocity:.1f}/year ({p.velocity_trend})")
        lines.append(f"Avg Claims: {p.avg_claims:.1f}")
        
        lines.append("")
        lines.append(f"Innovation Score: {p.innovation_score:.0f}/100")
        lines.append(f"Emerging Tech Ratio: {p.emerging_tech_ratio:.1%}")
        lines.append(f"Patent Moat: {signal.patent_moat_score:.2f}")
        
        lines.append("")
        lines.append(f"INNOVATION RANK: {signal.innovation_rank.upper()}")
        
        if signal.top_tech_areas:
            lines.append("")
            lines.append(f"Top Tech Areas: {', '.join(signal.top_tech_areas[:3])}")
        
        if signal.emerging_investments:
            lines.append(f"🚀 Emerging Investments: {', '.join(signal.emerging_investments)}")
        
        if signal.caveats:
            lines.append("")
            for caveat in signal.caveats:
                lines.append(f"⚠️  {caveat}")
        
        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def analyze_patent_innovation(ticker: str, company_name: Optional[str] = None) -> PatentInnovationSignal:
    """Quick function to analyze patent innovation."""
    model = PatentInnovationModel()
    return model.analyze_company(ticker, company_name)


if __name__ == "__main__":
    import numpy as np  # For moat score calculation
    
    # Demo
    model = PatentInnovationModel()
    
    print("Testing patent search for NVIDIA...")
    print()
    
    signal = model.analyze_company("NVDA", "NVIDIA")
    print(model.get_summary(signal))
