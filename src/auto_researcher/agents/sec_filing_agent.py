"""
SEC Filing Agent.

Fetches and analyzes SEC filings (10-K, 10-Q, 8-K) from EDGAR.
Uses LLM to extract key insights, risks, and business developments.

================================================================================
10-K TONE MODEL - STRENGTHS & WEAKNESSES
================================================================================

ACADEMIC BASIS:
    Loughran & McDonald (2011): Finance-specific sentiment dictionaries
    Li (2010): MD&A tone predicts future earnings
    Feldman et al. (2010): Tone CHANGES predict ~1.5% abnormal returns
    
STRENGTHS:
    ✅ Based on finance-specific word lists (not generic sentiment)
    ✅ Year-over-year CHANGES are more predictive than levels
    ✅ Full text analysis captures nuance missed by metrics
    ✅ Works well for detecting deteriorating fundamentals
    ✅ Objective and reproducible
    ✅ Captures uncertainty, litigation, and constraint language
    
WEAKNESSES:
    ⚠️ SIGNAL IS STALE - 10-K filed annually (up to 13 months old)
    ⚠️ Signal decays significantly after 60-90 days
    ⚠️ Large companies have PR-polished filings (less informative)
    ⚠️ Boilerplate language can dominate word counts
    ⚠️ Doesn't capture context (negation, sarcasm)
    ⚠️ SEC data can be slow to fetch
    
BEST USE CASES:
    • Detecting subtle deterioration in business language
    • Confirming red flags from other analysis
    • Screening for increased uncertainty/litigation risk
    
CAUTIONS:
    • Signal is freshest right after annual filing (Feb-Apr for Dec FYE)
    • Don't rely on tone alone - combine with fundamentals
    • Bullish tone change in weak company may be spin
    • Tone may lag actual business changes

Usage:
    from auto_researcher.agents.sec_filing_agent import SECFilingAgent
    
    agent = SECFilingAgent(model="gpt-4o-mini")
    results = agent.analyze_tickers(["GOOGL", "NVDA", "AAPL"])
    
    for ticker, analysis in results.items():
        print(f"{ticker}: {analysis.summary}")
"""

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Literal
from urllib.parse import urljoin

import pandas as pd
import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    litellm = None

# Import tone model
try:
    from ..models.filing_tone import FilingToneModel, ToneChangeSignal
    HAS_TONE_MODEL = True
except ImportError:
    HAS_TONE_MODEL = False
    FilingToneModel = None
    ToneChangeSignal = None


# ==============================================================================
# BASE AGENT CLASS
# ==============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for identification."""
        pass

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary task."""
        pass


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SECFiling:
    """A single SEC filing."""
    ticker: str
    cik: str
    form_type: str  # 10-K, 10-Q, 8-K
    filing_date: datetime
    accession_number: str
    document_url: str
    description: str = ""
    content_excerpt: str = ""  # First N chars of filing


@dataclass
class FilingAnalysis:
    """Analysis result from SEC filings."""
    ticker: str
    filings_analyzed: int
    latest_filing_date: Optional[datetime]
    latest_filing_type: Optional[str]
    
    # Risk analysis
    risk_score: float  # 0 (low risk) to 1 (high risk)
    risk_label: Literal["low", "moderate", "elevated", "high"]
    key_risks: list[str]
    
    # Business insights
    business_developments: list[str]
    strategic_initiatives: list[str]
    
    # Financial highlights
    revenue_trend: Optional[str]
    margin_trend: Optional[str]
    guidance_summary: Optional[str]
    
    # Red flags
    red_flags: list[str]
    material_changes: list[str]
    
    # Summary
    summary: str
    investment_implications: str
    
    # 10-K Tone Model signals (backtested) - all have defaults
    tone_net_current: Optional[float] = None       # Current 10-K net tone
    tone_net_prior: Optional[float] = None         # Prior 10-K net tone
    tone_change: Optional[float] = None            # YoY tone change
    tone_negative_pct: Optional[float] = None      # % negative words
    tone_positive_pct: Optional[float] = None      # % positive words
    tone_uncertainty_pct: Optional[float] = None   # % uncertainty words
    tone_direction: Optional[str] = None           # "bullish" or "bearish"
    tone_strength: Optional[str] = None            # "strong", "moderate", "weak", "none"
    tone_is_actionable: bool = False
    tone_expected_alpha: Optional[float] = None    # Expected excess return
    tone_signal_decay: Optional[float] = None      # 1.0 = fresh, 0.0 = stale
    tone_days_since_filing: Optional[int] = None   # Days since 10-K
    tone_summary: Optional[str] = None             # Human-readable signal
    
    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None


@dataclass
class SECAgentConfig:
    """Configuration for the SEC filing agent."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    timeout: int = 120
    max_filings: int = 3  # Number of recent filings to analyze
    filing_types: tuple = ("10-K", "10-Q", "8-K")
    excerpt_length: int = 15000  # Characters to extract from filing
    user_agent: str = "AutoResearcher research@example.com"  # Required by SEC


# ==============================================================================
# SEC EDGAR API
# ==============================================================================

class SECEdgarClient:
    """Client for fetching SEC filings from EDGAR."""
    
    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    COMPANY_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    # CIK lookup cache
    CIK_CACHE = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "GOOGL": "0001652044",
        "GOOG": "0001652044",
        "AMZN": "0001018724",
        "META": "0001326801",
        "NVDA": "0001045810",
        "TSLA": "0001318605",
        "BRK-B": "0001067983",
        "JPM": "0000019617",
        "JNJ": "0000200406",
        "V": "0001403161",
        "UNH": "0000731766",
        "XOM": "0000034088",
        "PG": "0000080424",
        "MA": "0001141391",
        "HD": "0000354950",
        "CVX": "0000093410",
        "MRK": "0000310158",
        "ABBV": "0001551152",
        "PFE": "0000078003",
        "KO": "0000021344",
        "PEP": "0000077476",
        "COST": "0000909832",
        "TMO": "0000097745",
        "AVGO": "0001730168",
        "MCD": "0000063908",
        "WMT": "0000104169",
        "CSCO": "0000858877",
        "ACN": "0001467373",
        "ABT": "0000001800",
        "LLY": "0000059478",
        "CRM": "0001108524",
        "AMD": "0000002488",
        "INTC": "0000050863",
        "ORCL": "0001341439",
        "NFLX": "0001065280",
        "ADBE": "0000796343",
    }
    
    def __init__(self, user_agent: str):
        """Initialize SEC EDGAR client."""
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in self.CIK_CACHE:
            return self.CIK_CACHE[ticker]
        
        # Try to fetch from SEC
        try:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&output=atom"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                # Parse CIK from response
                match = re.search(r'CIK=(\d{10})', response.text)
                if match:
                    cik = match.group(1)
                    self.CIK_CACHE[ticker] = cik
                    return cik
        except Exception as e:
            logger.warning(f"Failed to lookup CIK for {ticker}: {e}")
        
        return None
    
    def get_recent_filings(
        self,
        ticker: str,
        form_types: tuple = ("10-K", "10-Q", "8-K"),
        max_filings: int = 5,
    ) -> list[SECFiling]:
        """Get recent SEC filings for a ticker."""
        cik = self.get_cik(ticker)
        if not cik:
            logger.warning(f"Could not find CIK for {ticker}")
            return []
        
        # Fetch submissions
        try:
            url = self.SUBMISSIONS_URL.format(cik=cik.zfill(10))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch filings for {ticker}: {e}")
            return []
        
        # Parse filings
        filings = []
        recent = data.get("filings", {}).get("recent", {})
        
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        descs = recent.get("primaryDocDescription", [])
        
        for i in range(min(len(forms), 100)):  # Check last 100
            form_type = forms[i]
            
            if form_type in form_types:
                accession = accessions[i].replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{docs[i]}"
                
                try:
                    filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
                except:
                    filing_date = datetime.now()
                
                filings.append(SECFiling(
                    ticker=ticker,
                    cik=cik,
                    form_type=form_type,
                    filing_date=filing_date,
                    accession_number=accessions[i],
                    document_url=doc_url,
                    description=descs[i] if i < len(descs) else "",
                ))
                
                if len(filings) >= max_filings:
                    break
        
        return filings
    
    def fetch_filing_content(self, filing: SECFiling, max_chars: int = 15000) -> str:
        """Fetch the text content of a filing."""
        try:
            # Add delay to respect rate limits
            time.sleep(0.2)
            
            response = self.session.get(filing.document_url, timeout=30)
            response.raise_for_status()
            
            content = response.text
            
            # Clean HTML if present
            if "<html" in content.lower():
                # Simple HTML stripping
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'&nbsp;', ' ', content)
                content = re.sub(r'&[a-zA-Z]+;', ' ', content)
                content = re.sub(r'\s+', ' ', content)
            
            # Truncate
            return content[:max_chars].strip()
            
        except Exception as e:
            logger.warning(f"Failed to fetch filing content: {e}")
            return ""


# ==============================================================================
# SEC FILING AGENT
# ==============================================================================

class SECFilingAgent(BaseAgent):
    """
    Agent for analyzing SEC filings (10-K, 10-Q, 8-K).
    
    Fetches recent filings from SEC EDGAR and uses LLM to extract
    key insights about risks, business developments, and investment implications.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        config: Optional[SECAgentConfig] = None,
        use_tone_model: bool = True,
    ) -> None:
        """Initialize the SEC Filing Agent."""
        if not HAS_LITELLM:
            raise ImportError("litellm is required. Install with: pip install litellm")
        
        self.config = config or SECAgentConfig(model=model)
        self.model = self.config.model
        self.edgar_client = SECEdgarClient(user_agent=self.config.user_agent)
        
        # Initialize backtested 10-K tone model
        self.use_tone_model = use_tone_model and HAS_TONE_MODEL
        self._tone_model = None  # FilingToneModel or None
        if self.use_tone_model:
            try:
                self._tone_model = FilingToneModel(user_agent=self.config.user_agent)
                logger.info("FilingToneModel initialized for SECFilingAgent")
            except Exception as e:
                logger.warning(f"Failed to initialize FilingToneModel: {e}")
                self.use_tone_model = False
        
        logger.info(f"Initialized SECFilingAgent with model: {self.model}")
    
    @property
    def name(self) -> str:
        return "SECFilingAgent"
    
    def run(
        self,
        tickers: list[str],
        **kwargs,
    ) -> dict[str, FilingAnalysis]:
        """Execute SEC filing analysis for tickers."""
        return self.analyze_tickers(tickers)
    
    def _build_prompt(self, ticker: str, filings: list[SECFiling]) -> str:
        """Build the analysis prompt for the LLM."""
        
        filings_text = ""
        for filing in filings:
            filings_text += f"""
--- {filing.form_type} FILED {filing.filing_date.strftime('%Y-%m-%d')} ---
{filing.content_excerpt[:5000]}
...
"""
        
        prompt = f"""Analyze these SEC filings for {ticker} and provide investment insights.

{filings_text}

Provide your analysis in EXACTLY this format:

RISK_SCORE: [number from 0.0 to 1.0, where 0 is low risk and 1 is high risk]
RISK_LABEL: [one of: low, moderate, elevated, high]
KEY_RISKS: [comma-separated list of 3-5 key risk factors mentioned]
BUSINESS_DEVELOPMENTS: [comma-separated list of 2-4 significant business developments]
STRATEGIC_INITIATIVES: [comma-separated list of 2-3 strategic initiatives or priorities]
REVENUE_TREND: [brief description of revenue trend if mentioned]
MARGIN_TREND: [brief description of margin trends if mentioned]
GUIDANCE_SUMMARY: [any forward guidance mentioned, or "None provided"]
RED_FLAGS: [any concerning items that investors should note, or "None identified"]
MATERIAL_CHANGES: [significant changes from prior periods, or "None noted"]
SUMMARY: [2-3 sentence summary of overall filing insights]
INVESTMENT_IMPLICATIONS: [1-2 sentence investment thesis based on filings]

Focus on:
1. Risk factors - litigation, regulatory, competitive, operational
2. Business developments - new products, markets, acquisitions
3. Financial trends - revenue, margins, cash flow
4. Forward guidance and outlook
5. Any red flags or material changes"""

        return prompt
    
    def _parse_response(self, ticker: str, response: str, filings: list[SECFiling]) -> FilingAnalysis:
        """Parse the LLM response into a FilingAnalysis."""
        
        # Defaults
        risk_score = 0.5
        risk_label = "moderate"
        key_risks = []
        business_developments = []
        strategic_initiatives = []
        revenue_trend = None
        margin_trend = None
        guidance_summary = None
        red_flags = []
        material_changes = []
        summary = "Analysis not available."
        investment_implications = "Insufficient data for assessment."
        
        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("RISK_SCORE:"):
                    try:
                        risk_score = float(line.split(":", 1)[1].strip())
                        risk_score = max(0.0, min(1.0, risk_score))
                    except:
                        pass
                elif line.startswith("RISK_LABEL:"):
                    label = line.split(":", 1)[1].strip().lower()
                    if label in ["low", "moderate", "elevated", "high"]:
                        risk_label = label
                elif line.startswith("KEY_RISKS:"):
                    risks_str = line.split(":", 1)[1].strip()
                    key_risks = [r.strip() for r in risks_str.split(",") if r.strip()]
                elif line.startswith("BUSINESS_DEVELOPMENTS:"):
                    dev_str = line.split(":", 1)[1].strip()
                    business_developments = [d.strip() for d in dev_str.split(",") if d.strip()]
                elif line.startswith("STRATEGIC_INITIATIVES:"):
                    init_str = line.split(":", 1)[1].strip()
                    strategic_initiatives = [i.strip() for i in init_str.split(",") if i.strip()]
                elif line.startswith("REVENUE_TREND:"):
                    revenue_trend = line.split(":", 1)[1].strip()
                elif line.startswith("MARGIN_TREND:"):
                    margin_trend = line.split(":", 1)[1].strip()
                elif line.startswith("GUIDANCE_SUMMARY:"):
                    guidance_summary = line.split(":", 1)[1].strip()
                elif line.startswith("RED_FLAGS:"):
                    flags_str = line.split(":", 1)[1].strip()
                    if "none" not in flags_str.lower():
                        red_flags = [f.strip() for f in flags_str.split(",") if f.strip()]
                elif line.startswith("MATERIAL_CHANGES:"):
                    changes_str = line.split(":", 1)[1].strip()
                    if "none" not in changes_str.lower():
                        material_changes = [c.strip() for c in changes_str.split(",") if c.strip()]
                elif line.startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.startswith("INVESTMENT_IMPLICATIONS:"):
                    investment_implications = line.split(":", 1)[1].strip()
        except Exception as e:
            logger.warning(f"Failed to parse SEC response: {e}")
        
        latest_filing = filings[0] if filings else None
        
        return FilingAnalysis(
            ticker=ticker,
            filings_analyzed=len(filings),
            latest_filing_date=latest_filing.filing_date if latest_filing else None,
            latest_filing_type=latest_filing.form_type if latest_filing else None,
            risk_score=risk_score,
            risk_label=risk_label,
            key_risks=key_risks,
            business_developments=business_developments,
            strategic_initiatives=strategic_initiatives,
            revenue_trend=revenue_trend,
            margin_trend=margin_trend,
            guidance_summary=guidance_summary,
            red_flags=red_flags,
            material_changes=material_changes,
            summary=summary,
            investment_implications=investment_implications,
            raw_response=response,
        )
    
    def _add_tone_signals(self, result: FilingAnalysis) -> None:
        """
        Add backtested 10-K tone signals to the filing analysis.
        
        Based on Loughran & McDonald (2011) financial sentiment dictionaries
        and research showing tone changes predict future returns.
        
        Args:
            result: FilingAnalysis to enhance with tone signals.
        """
        if not self._tone_model:
            return
        
        try:
            signal = self._tone_model.get_signal(result.ticker)
            
            if signal.current_tone:
                result.tone_net_current = signal.current_tone.net_tone
                result.tone_negative_pct = signal.current_tone.negative_pct
                result.tone_positive_pct = signal.current_tone.positive_pct
                result.tone_uncertainty_pct = signal.current_tone.uncertainty_pct
            
            if signal.prior_tone:
                result.tone_net_prior = signal.prior_tone.net_tone
            
            result.tone_change = signal.net_tone_change
            result.tone_direction = signal.direction
            result.tone_strength = signal.strength
            result.tone_is_actionable = signal.is_actionable
            result.tone_expected_alpha = signal.expected_alpha
            result.tone_signal_decay = signal.signal_decay
            result.tone_days_since_filing = signal.days_since_filing
            result.tone_summary = signal.summary
            
            logger.info(
                f"{result.ticker}: 10-K tone {signal.direction or 'neutral'} "
                f"(change: {signal.net_tone_change:+.3f})"
            )
            
        except Exception as e:
            logger.warning(f"Failed to get tone signal for {result.ticker}: {e}")
            result.tone_summary = f"TONE: Error - {str(e)[:50]}"
    
    def analyze_ticker(self, ticker: str) -> FilingAnalysis:
        """Analyze SEC filings for a single ticker."""
        
        # Fetch recent filings
        filings = self.edgar_client.get_recent_filings(
            ticker=ticker,
            form_types=self.config.filing_types,
            max_filings=self.config.max_filings,
        )
        
        if not filings:
            return FilingAnalysis(
                ticker=ticker,
                filings_analyzed=0,
                latest_filing_date=None,
                latest_filing_type=None,
                risk_score=0.5,
                risk_label="moderate",
                key_risks=[],
                business_developments=[],
                strategic_initiatives=[],
                revenue_trend=None,
                margin_trend=None,
                guidance_summary=None,
                red_flags=[],
                material_changes=[],
                summary="No SEC filings found for analysis.",
                investment_implications="Unable to assess - no filings available.",
            )
        
        # Fetch content for each filing
        for filing in filings:
            filing.content_excerpt = self.edgar_client.fetch_filing_content(
                filing,
                max_chars=self.config.excerpt_length // len(filings),
            )
        
        # Build prompt
        prompt = self._build_prompt(ticker, filings)
        
        # Call LLM
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SEC filing analyst. Extract key insights from corporate filings for investors."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            
            response_text = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM call failed for {ticker}: {e}")
            return FilingAnalysis(
                ticker=ticker,
                filings_analyzed=len(filings),
                latest_filing_date=filings[0].filing_date if filings else None,
                latest_filing_type=filings[0].form_type if filings else None,
                risk_score=0.5,
                risk_label="moderate",
                key_risks=[],
                business_developments=[],
                strategic_initiatives=[],
                revenue_trend=None,
                margin_trend=None,
                guidance_summary=None,
                red_flags=[],
                material_changes=[],
                summary=f"Analysis failed: {str(e)[:50]}",
                investment_implications="Unable to assess.",
            )
        
        result = self._parse_response(ticker, response_text, filings)
        
        # Add backtested 10-K tone signals
        if self.use_tone_model:
            self._add_tone_signals(result)
        
        return result
    
    def analyze_tickers(self, tickers: list[str]) -> dict[str, FilingAnalysis]:
        """Analyze SEC filings for multiple tickers."""
        results = {}
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Analyzing SEC filings for {ticker} ({i}/{len(tickers)})...")
            try:
                result = self.analyze_ticker(ticker)
                results[ticker] = result
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
        
        return results
    
    def format_results(self, results: dict[str, FilingAnalysis]) -> str:
        """Format results as a readable string."""
        lines = []
        lines.append("=" * 70)
        lines.append("  SEC FILING ANALYSIS")
        lines.append("=" * 70)
        lines.append("")
        
        for ticker, r in results.items():
            # Risk indicator
            if r.risk_score >= 0.7:
                indicator = "🔴"
            elif r.risk_score >= 0.4:
                indicator = "🟡"
            else:
                indicator = "🟢"
            
            lines.append(f"{indicator} {ticker} - Risk: {r.risk_label.upper()} ({r.risk_score:.2f})")
            
            if r.latest_filing_date:
                lines.append(f"   Latest: {r.latest_filing_type} ({r.latest_filing_date.strftime('%Y-%m-%d')})")
            
            lines.append(f"   {r.summary}")
            
            # 10-K Tone Model (backtested)
            if r.tone_summary:
                lines.append(f"   📝 {r.tone_summary}")
                if r.tone_is_actionable and r.tone_expected_alpha:
                    lines.append(f"      Expected Alpha: {r.tone_expected_alpha:.1%} | Signal Decay: {r.tone_signal_decay:.0%}")
            
            if r.key_risks:
                lines.append(f"   ⚠️  Risks: {', '.join(r.key_risks[:3])}")
            
            if r.business_developments:
                lines.append(f"   📈 Developments: {', '.join(r.business_developments[:2])}")
            
            if r.red_flags:
                lines.append(f"   🚩 Red Flags: {', '.join(r.red_flags[:2])}")
            
            lines.append(f"   💡 {r.investment_implications}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dataframe(self, results: dict[str, FilingAnalysis]) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        for ticker, r in results.items():
            data.append({
                "ticker": ticker,
                "filings_analyzed": r.filings_analyzed,
                "latest_filing_type": r.latest_filing_type,
                "latest_filing_date": r.latest_filing_date,
                "risk_score": r.risk_score,
                "risk_label": r.risk_label,
                "key_risks": "; ".join(r.key_risks),
                "red_flags": "; ".join(r.red_flags),
                "summary": r.summary,
                # Tone model fields
                "tone_change": r.tone_change,
                "tone_direction": r.tone_direction,
                "tone_strength": r.tone_strength,
                "tone_is_actionable": r.tone_is_actionable,
                "tone_expected_alpha": r.tone_expected_alpha,
                "tone_days_since_filing": r.tone_days_since_filing,
            })
        return pd.DataFrame(data)


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze SEC filings")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to analyze")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    
    args = parser.parse_args()
    
    agent = SECFilingAgent(model=args.model)
    results = agent.analyze_tickers(args.tickers)
    print(agent.format_results(results))
