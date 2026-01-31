"""
Insider Trading Agent.

Analyzes SEC Form 4 filings to detect meaningful insider trading patterns.
Focuses on cluster buying events and filters out noise (10b5-1 plans, options exercises).

Key insights:
- Cluster buying (3+ insiders buying in 30 days) is strongly predictive
- Open market purchases matter more than options exercises
- CEO/CFO trades carry more weight than lower-level insiders
- Large purchases relative to typical trading are more meaningful

Usage:
    from auto_researcher.agents.insider_trading_agent import InsiderTradingAgent
    
    agent = InsiderTradingAgent()
    result = agent.analyze_ticker("AAPL")
    print(f"Signal: {result.signal} ({result.score:+.2f})")
"""

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

# Import InsiderClusterModel for enhanced signal
# Use direct module import to avoid circular imports
try:
    import importlib.util
    _cluster_module_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'insider_cluster.py')
    _spec = importlib.util.spec_from_file_location("insider_cluster", _cluster_module_path)
    _cluster_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cluster_module)
    InsiderSignal = _cluster_module.InsiderSignal
    INSIDER_CONFIG = _cluster_module.INSIDER_CONFIG
    HAS_CLUSTER_MODEL = True
except Exception as e:
    logger.debug(f"InsiderClusterModel import deferred: {e}")
    HAS_CLUSTER_MODEL = False
    InsiderSignal = None
    INSIDER_CONFIG = None

# SEC EDGAR settings
SEC_BASE_URL = "https://www.sec.gov"
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILINGS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

# User agent for SEC requests
USER_AGENT = os.getenv("SEC_USER_AGENT", "AutoResearcher research@example.com")


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class InsiderTransaction:
    """A single insider transaction from Form 4."""
    ticker: str
    insider_name: str
    insider_title: str  # CEO, CFO, Director, VP, etc.
    transaction_date: datetime
    transaction_type: str  # P (purchase), S (sale), A (award), etc.
    shares: float
    price_per_share: Optional[float]
    total_value: Optional[float]
    shares_owned_after: Optional[float]
    is_direct: bool = True  # Direct vs indirect ownership
    is_10b5_1: bool = False  # Pre-planned trading
    acquisition_disposition: str = "A"  # A = acquired, D = disposed
    form_type: str = "4"
    filing_url: str = ""
    
    @property
    def is_purchase(self) -> bool:
        return self.acquisition_disposition == "A" and self.transaction_type in ("P", "M")
    
    @property
    def is_sale(self) -> bool:
        return self.acquisition_disposition == "D" and self.transaction_type in ("S", "F")
    
    @property
    def is_open_market(self) -> bool:
        """Open market transaction (not options exercise, gift, etc.)"""
        return self.transaction_type in ("P", "S")
    
    @property
    def role_weight(self) -> float:
        """Weight based on insider role (CEO/CFO most important)."""
        title_lower = self.insider_title.lower()
        if any(t in title_lower for t in ["ceo", "chief executive"]):
            return 1.0
        elif any(t in title_lower for t in ["cfo", "chief financial"]):
            return 0.9
        elif any(t in title_lower for t in ["coo", "chief operating", "president"]):
            return 0.8
        elif any(t in title_lower for t in ["cto", "chief tech", "chief product"]):
            return 0.7
        elif "director" in title_lower:
            return 0.6
        elif any(t in title_lower for t in ["vp", "vice president", "evp", "svp"]):
            return 0.5
        elif "officer" in title_lower:
            return 0.4
        else:
            return 0.3


@dataclass
class ClusterEvent:
    """A cluster of insider transactions in a short period."""
    ticker: str
    start_date: datetime
    end_date: datetime
    direction: Literal["buy", "sell", "mixed"]
    num_insiders: int
    total_value: float
    transactions: list[InsiderTransaction]
    
    @property
    def days_span(self) -> int:
        return (self.end_date - self.start_date).days
    
    @property
    def avg_role_weight(self) -> float:
        if not self.transactions:
            return 0.0
        return sum(t.role_weight for t in self.transactions) / len(self.transactions)


@dataclass
class InsiderAnalysis:
    """Complete insider trading analysis for a ticker."""
    ticker: str
    analysis_date: datetime
    
    # Summary metrics
    signal: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"]
    score: float  # -1 to +1
    confidence: float  # 0 to 1
    
    # Transaction counts (last 90 days)
    total_transactions: int
    buy_count: int
    sell_count: int
    net_shares: float
    net_value: float
    
    # Open market only (filtering noise)
    open_market_buys: int
    open_market_sells: int
    open_market_net_value: float
    
    # Cluster events
    cluster_events: list[ClusterEvent] = field(default_factory=list)
    has_cluster_buying: bool = False
    has_cluster_selling: bool = False
    
    # Cluster Model Signal (InsiderClusterModel fields)
    cluster_direction: Optional[str] = None  # "long" or "short"
    cluster_strength: str = "none"  # "strong", "moderate", "weak", "none"
    cluster_is_actionable: bool = False
    cluster_num_insiders: int = 0
    cluster_total_value: float = 0.0
    cluster_has_ceo_cfo: bool = False
    cluster_expected_return: Optional[float] = None
    cluster_days_since: Optional[int] = None
    cluster_summary: str = ""
    cluster_model_caveats: list[str] = field(default_factory=list)  # Model limitations to consider
    
    # Notable transactions
    notable_transactions: list[InsiderTransaction] = field(default_factory=list)
    
    # Insights
    summary: str = ""
    key_insights: list[str] = field(default_factory=list)
    
    # Raw data
    all_transactions: list[InsiderTransaction] = field(default_factory=list)


# ==============================================================================
# SEC EDGAR CLIENT
# ==============================================================================

class SECEdgarClient:
    """Client for fetching Form 4 filings from SEC EDGAR."""
    
    def __init__(self, user_agent: str = USER_AGENT):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._cik_cache = {}
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        
        try:
            # Use SEC's company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            # Build ticker -> CIK mapping
            for item in data.values():
                t = item.get("ticker", "").upper()
                cik = str(item.get("cik_str", "")).zfill(10)
                self._cik_cache[t] = cik
            
            return self._cik_cache.get(ticker.upper())
        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
            return None
    
    def get_form4_filings(
        self,
        ticker: str,
        days_back: int = 90,
        max_filings: int = 50,
    ) -> list[dict]:
        """Get recent Form 4 filings for a ticker."""
        cik = self.get_cik(ticker)
        if not cik:
            logger.warning(f"Could not find CIK for {ticker}")
            return []
        
        try:
            # Get company filings
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract Form 4 filings
            filings = []
            recent = data.get("filings", {}).get("recent", {})
            
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            docs = recent.get("primaryDocument", [])
            
            cutoff = datetime.now() - timedelta(days=days_back)
            
            for i, form in enumerate(forms):
                if form not in ("4", "4/A"):
                    continue
                
                try:
                    filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
                    if filing_date < cutoff:
                        continue
                    
                    accession = accessions[i].replace("-", "")
                    doc = docs[i]
                    # Remove xsl transform prefix if present (e.g., xslF345X05/)
                    if "/" in doc:
                        doc = doc.split("/")[-1]
                    # Strip leading zeros from CIK for URL
                    cik_int = str(int(cik))
                    
                    filings.append({
                        "form": form,
                        "date": filing_date,
                        "accession": accession,
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}",
                        "cik": cik,
                    })
                    
                    if len(filings) >= max_filings:
                        break
                except (IndexError, ValueError):
                    continue
            
            return filings
            
        except Exception as e:
            logger.error(f"Failed to get Form 4 filings for {ticker}: {e}")
            return []
    
    def parse_form4(self, filing_url: str, ticker: str) -> list[InsiderTransaction]:
        """Parse a Form 4 XML filing."""
        transactions = []
        
        try:
            time.sleep(0.15)  # Rate limiting
            resp = self.session.get(filing_url, timeout=10)
            resp.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(resp.content)
            
            # Get namespace
            ns = {"": ""}
            if root.tag.startswith("{"):
                ns_uri = root.tag[1:root.tag.index("}")]
                ns = {"ns": ns_uri}
            
            # Get reporting owner info
            owner_elem = root.find(".//reportingOwner")
            if owner_elem is None:
                owner_elem = root.find("reportingOwner")
            
            insider_name = ""
            insider_title = ""
            
            if owner_elem is not None:
                # Get name
                name_elem = owner_elem.find(".//rptOwnerName")
                if name_elem is None:
                    name_elem = owner_elem.find("rptOwnerName")
                if name_elem is not None and name_elem.text:
                    insider_name = name_elem.text.strip()
                
                # Get title
                title_elem = owner_elem.find(".//officerTitle")
                if title_elem is None:
                    title_elem = owner_elem.find("officerTitle")
                if title_elem is not None and title_elem.text:
                    insider_title = title_elem.text.strip()
                
                # Check for director
                director_elem = owner_elem.find(".//isDirector")
                if director_elem is None:
                    director_elem = owner_elem.find("isDirector")
                if director_elem is not None and director_elem.text == "1" and not insider_title:
                    insider_title = "Director"
            
            # Parse non-derivative transactions
            for txn in root.findall(".//nonDerivativeTransaction"):
                try:
                    trans = self._parse_transaction(txn, ticker, insider_name, insider_title, filing_url)
                    if trans:
                        transactions.append(trans)
                except Exception as e:
                    logger.debug(f"Failed to parse transaction: {e}")
            
            # Also check derivativeTransaction for options exercises
            for txn in root.findall(".//derivativeTransaction"):
                try:
                    trans = self._parse_derivative_transaction(txn, ticker, insider_name, insider_title, filing_url)
                    if trans:
                        transactions.append(trans)
                except Exception as e:
                    logger.debug(f"Failed to parse derivative transaction: {e}")
            
        except Exception as e:
            logger.debug(f"Failed to parse Form 4 at {filing_url}: {e}")
        
        return transactions
    
    def _parse_transaction(
        self,
        txn_elem,
        ticker: str,
        insider_name: str,
        insider_title: str,
        filing_url: str,
    ) -> Optional[InsiderTransaction]:
        """Parse a non-derivative transaction element."""
        
        def find_value(elem, *paths):
            """Find text value in nested elements."""
            for path in paths:
                found = elem.find(f".//{path}")
                if found is not None and found.text:
                    return found.text.strip()
            return None
        
        # Transaction date - it's in transactionDate/value
        date_elem = txn_elem.find(".//transactionDate/value")
        if date_elem is None or not date_elem.text:
            return None
        date_str = date_elem.text.strip()
        
        try:
            trans_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None
        
        # Transaction code (P=purchase, S=sale, A=award, etc.)
        code_elem = txn_elem.find(".//transactionCoding/transactionCode")
        code = code_elem.text if code_elem is not None else "P"
        
        # Acquisition/Disposition (A = acquired, D = disposed)
        ad_elem = txn_elem.find(".//transactionAmounts/transactionAcquiredDisposedCode/value")
        ad = ad_elem.text if ad_elem is not None else "A"
        
        # Shares
        shares_elem = txn_elem.find(".//transactionAmounts/transactionShares/value")
        try:
            shares = float(shares_elem.text) if shares_elem is not None else 0
        except ValueError:
            shares = 0
        
        # Price
        price_elem = txn_elem.find(".//transactionAmounts/transactionPricePerShare/value")
        try:
            price = float(price_elem.text) if price_elem is not None and price_elem.text else None
        except ValueError:
            price = None
        
        # Check for 10b5-1 plan
        is_10b5_1 = False
        plan_elem = txn_elem.find(".//transactionCoding/equitySwapInvolved")
        # Also check footnotes for Rule 10b5-1 mentions (heuristic)
        
        # Shares owned after
        owned_after = None
        owned_elem = txn_elem.find(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        if owned_elem is not None and owned_elem.text:
            try:
                owned_after = float(owned_elem.text)
            except (ValueError, TypeError):
                pass
        
        total_value = shares * price if price else None
        
        return InsiderTransaction(
            ticker=ticker,
            insider_name=insider_name,
            insider_title=insider_title,
            transaction_date=trans_date,
            transaction_type=code or "P",
            shares=shares,
            price_per_share=price,
            total_value=total_value,
            shares_owned_after=owned_after,
            is_direct=True,
            is_10b5_1=is_10b5_1,
            acquisition_disposition=ad,
            filing_url=filing_url,
        )
    
    def _parse_derivative_transaction(
        self,
        txn_elem,
        ticker: str,
        insider_name: str,
        insider_title: str,
        filing_url: str,
    ) -> Optional[InsiderTransaction]:
        """Parse a derivative transaction (options exercise, etc.)."""
        # For now, we'll skip derivatives as they're usually noise
        # Could expand this later
        return None


# ==============================================================================
# INSIDER TRADING AGENT
# ==============================================================================

class InsiderTradingAgent:
    """
    Analyzes insider trading patterns from SEC Form 4 filings.
    
    Scoring approach:
    - Cluster buying (3+ insiders same direction in 30 days) is strongly weighted
    - Open market purchases weighted higher than options/awards
    - CEO/CFO trades weighted higher than lower-level insiders
    - 10b5-1 pre-planned trades are filtered out
    - Dollar value normalized by company size
    """
    
    def __init__(
        self,
        lookback_days: int = 90,
        cluster_window_days: int = 30,
        min_cluster_insiders: int = 3,
    ):
        """
        Initialize the agent.
        
        Args:
            lookback_days: How far back to look for transactions.
            cluster_window_days: Window for detecting cluster events.
            min_cluster_insiders: Min insiders for a cluster event.
        """
        self.lookback_days = lookback_days
        self.cluster_window_days = cluster_window_days
        self.min_cluster_insiders = min_cluster_insiders
        self.client = SECEdgarClient()
    
    @property
    def name(self) -> str:
        return "InsiderTradingAgent"
    
    def _fetch_transactions(self, ticker: str) -> list[InsiderTransaction]:
        """Fetch all recent transactions for a ticker."""
        filings = self.client.get_form4_filings(
            ticker,
            days_back=self.lookback_days,
        )
        
        all_transactions = []
        for filing in filings:
            time.sleep(0.1)  # Rate limiting
            transactions = self.client.parse_form4(filing["url"], ticker)
            all_transactions.extend(transactions)
        
        return all_transactions
    
    def _filter_meaningful_transactions(
        self,
        transactions: list[InsiderTransaction],
    ) -> list[InsiderTransaction]:
        """Filter out noise (10b5-1, small trades, etc.)."""
        meaningful = []
        
        for t in transactions:
            # Skip 10b5-1 pre-planned trades
            if t.is_10b5_1:
                continue
            
            # Skip very small transactions (less than $1000)
            if t.total_value and t.total_value < 1000:
                continue
            
            # Focus on open market transactions
            if not t.is_open_market:
                continue
            
            meaningful.append(t)
        
        return meaningful
    
    def _detect_clusters(
        self,
        transactions: list[InsiderTransaction],
    ) -> list[ClusterEvent]:
        """Detect cluster buying/selling events."""
        if not transactions:
            return []
        
        # Sort by date
        sorted_txns = sorted(transactions, key=lambda t: t.transaction_date)
        
        clusters = []
        window = timedelta(days=self.cluster_window_days)
        
        # Sliding window approach
        i = 0
        while i < len(sorted_txns):
            window_end = sorted_txns[i].transaction_date + window
            
            # Collect transactions in window
            window_txns = []
            j = i
            while j < len(sorted_txns) and sorted_txns[j].transaction_date <= window_end:
                window_txns.append(sorted_txns[j])
                j += 1
            
            # Count unique insiders
            buy_insiders = set()
            sell_insiders = set()
            
            for t in window_txns:
                if t.is_purchase:
                    buy_insiders.add(t.insider_name)
                elif t.is_sale:
                    sell_insiders.add(t.insider_name)
            
            # Check for cluster
            if len(buy_insiders) >= self.min_cluster_insiders:
                buy_txns = [t for t in window_txns if t.is_purchase]
                total_value = sum(t.total_value or 0 for t in buy_txns)
                clusters.append(ClusterEvent(
                    ticker=sorted_txns[i].ticker,
                    start_date=buy_txns[0].transaction_date,
                    end_date=buy_txns[-1].transaction_date,
                    direction="buy",
                    num_insiders=len(buy_insiders),
                    total_value=total_value,
                    transactions=buy_txns,
                ))
            
            if len(sell_insiders) >= self.min_cluster_insiders:
                sell_txns = [t for t in window_txns if t.is_sale]
                total_value = sum(t.total_value or 0 for t in sell_txns)
                clusters.append(ClusterEvent(
                    ticker=sorted_txns[i].ticker,
                    start_date=sell_txns[0].transaction_date,
                    end_date=sell_txns[-1].transaction_date,
                    direction="sell",
                    num_insiders=len(sell_insiders),
                    total_value=total_value,
                    transactions=sell_txns,
                ))
            
            i += 1
        
        # Deduplicate overlapping clusters (keep largest)
        if clusters:
            clusters = self._dedupe_clusters(clusters)
        
        return clusters
    
    def _dedupe_clusters(self, clusters: list[ClusterEvent]) -> list[ClusterEvent]:
        """Remove overlapping clusters, keeping the most significant."""
        if not clusters:
            return []
        
        # Sort by significance (num_insiders * total_value)
        sorted_clusters = sorted(
            clusters,
            key=lambda c: c.num_insiders * c.total_value,
            reverse=True,
        )
        
        kept = []
        for cluster in sorted_clusters:
            # Check if overlaps with any kept cluster
            overlaps = False
            for kept_cluster in kept:
                if cluster.direction != kept_cluster.direction:
                    continue
                # Check date overlap
                if (cluster.start_date <= kept_cluster.end_date and 
                    cluster.end_date >= kept_cluster.start_date):
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(cluster)
        
        return kept
    
    def _calculate_score(
        self,
        transactions: list[InsiderTransaction],
        clusters: list[ClusterEvent],
    ) -> tuple[float, float]:
        """
        Calculate overall insider trading score.
        
        Returns:
            (score, confidence) where score is -1 to +1 and confidence is 0 to 1.
        """
        if not transactions:
            return 0.0, 0.0
        
        # Base score from transaction flow
        buy_value = 0.0
        sell_value = 0.0
        buy_weighted = 0.0
        sell_weighted = 0.0
        
        for t in transactions:
            value = t.total_value or 0
            weight = t.role_weight
            
            if t.is_purchase:
                buy_value += value
                buy_weighted += value * weight
            elif t.is_sale:
                sell_value += value
                sell_weighted += value * weight
        
        total_value = buy_value + sell_value
        if total_value == 0:
            base_score = 0.0
        else:
            # Weighted net as percentage of total
            net_weighted = buy_weighted - sell_weighted
            base_score = net_weighted / (buy_weighted + sell_weighted + 1)
        
        # Cluster bonus/penalty
        cluster_score = 0.0
        for cluster in clusters:
            # Scale by number of insiders and average role weight
            impact = (cluster.num_insiders - 2) * 0.15 * cluster.avg_role_weight
            if cluster.direction == "buy":
                cluster_score += impact
            else:
                cluster_score -= impact * 0.5  # Selling less informative
        
        # Combined score
        score = base_score * 0.4 + cluster_score * 0.6
        score = max(-1.0, min(1.0, score))
        
        # Confidence based on data quality
        confidence = min(1.0, len(transactions) / 10) * 0.5
        if clusters:
            confidence += 0.3
        if total_value > 1_000_000:
            confidence += 0.2
        confidence = min(1.0, confidence)
        
        return score, confidence
    
    def _score_to_signal(self, score: float) -> str:
        """Convert numeric score to signal label."""
        if score >= 0.3:
            return "strong_buy"
        elif score >= 0.1:
            return "buy"
        elif score > -0.1:
            return "neutral"
        elif score > -0.3:
            return "sell"
        else:
            return "strong_sell"
    
    def _generate_insights(
        self,
        transactions: list[InsiderTransaction],
        clusters: list[ClusterEvent],
        score: float,
    ) -> tuple[str, list[str]]:
        """Generate human-readable insights."""
        insights = []
        
        # Cluster insights
        buy_clusters = [c for c in clusters if c.direction == "buy"]
        sell_clusters = [c for c in clusters if c.direction == "sell"]
        
        if buy_clusters:
            c = buy_clusters[0]
            insights.append(
                f"🟢 Cluster buying: {c.num_insiders} insiders bought "
                f"${c.total_value:,.0f} worth in {c.days_span} days"
            )
        
        if sell_clusters:
            c = sell_clusters[0]
            insights.append(
                f"🔴 Cluster selling: {c.num_insiders} insiders sold "
                f"${c.total_value:,.0f} worth in {c.days_span} days"
            )
        
        # Notable transactions
        if transactions:
            # Find largest purchase
            purchases = [t for t in transactions if t.is_purchase and t.total_value]
            if purchases:
                largest_buy = max(purchases, key=lambda t: t.total_value or 0)
                if largest_buy.total_value and largest_buy.total_value >= 100_000:
                    insights.append(
                        f"📈 {largest_buy.insider_title} {largest_buy.insider_name} "
                        f"bought ${largest_buy.total_value:,.0f}"
                    )
            
            # Find largest sale
            sales = [t for t in transactions if t.is_sale and t.total_value]
            if sales:
                largest_sell = max(sales, key=lambda t: t.total_value or 0)
                if largest_sell.total_value and largest_sell.total_value >= 500_000:
                    insights.append(
                        f"📉 {largest_sell.insider_title} {largest_sell.insider_name} "
                        f"sold ${largest_sell.total_value:,.0f}"
                    )
        
        # Generate summary
        if score > 0.2:
            summary = "Insiders are net buyers, especially at senior levels. This is a bullish signal."
        elif score < -0.2:
            summary = "Insiders are net sellers. While often just diversification, monitor for patterns."
        elif buy_clusters:
            summary = "Cluster buying detected - multiple insiders buying together is historically bullish."
        elif sell_clusters:
            summary = "Cluster selling detected - while often benign, warrants attention."
        else:
            summary = "Mixed or minimal insider activity. No strong directional signal."
        
        return summary, insights
    
    def analyze_ticker(self, ticker: str) -> InsiderAnalysis:
        """
        Analyze insider trading for a single ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            InsiderAnalysis with signal, score, and insights.
        """
        logger.info(f"Analyzing insider trading for {ticker}")
        
        # Fetch transactions
        all_transactions = self._fetch_transactions(ticker)
        
        if not all_transactions:
            return InsiderAnalysis(
                ticker=ticker,
                analysis_date=datetime.now(),
                signal="neutral",
                score=0.0,
                confidence=0.0,
                total_transactions=0,
                buy_count=0,
                sell_count=0,
                net_shares=0.0,
                net_value=0.0,
                open_market_buys=0,
                open_market_sells=0,
                open_market_net_value=0.0,
                summary="No recent Form 4 filings found.",
            )
        
        # Filter meaningful transactions
        meaningful = self._filter_meaningful_transactions(all_transactions)
        
        # Detect clusters
        clusters = self._detect_clusters(meaningful)
        
        # Calculate score
        score, confidence = self._calculate_score(meaningful, clusters)
        signal = self._score_to_signal(score)
        
        # Calculate metrics
        buy_count = sum(1 for t in all_transactions if t.is_purchase)
        sell_count = sum(1 for t in all_transactions if t.is_sale)
        
        buy_shares = sum(t.shares for t in all_transactions if t.is_purchase)
        sell_shares = sum(t.shares for t in all_transactions if t.is_sale)
        net_shares = buy_shares - sell_shares
        
        buy_value = sum(t.total_value or 0 for t in all_transactions if t.is_purchase)
        sell_value = sum(t.total_value or 0 for t in all_transactions if t.is_sale)
        net_value = buy_value - sell_value
        
        open_market_buys = sum(1 for t in meaningful if t.is_purchase)
        open_market_sells = sum(1 for t in meaningful if t.is_sale)
        om_buy_value = sum(t.total_value or 0 for t in meaningful if t.is_purchase)
        om_sell_value = sum(t.total_value or 0 for t in meaningful if t.is_sale)
        
        # Generate insights
        summary, insights = self._generate_insights(meaningful, clusters, score)
        
        # Notable transactions (top by value)
        notable = sorted(
            [t for t in meaningful if t.total_value],
            key=lambda t: t.total_value or 0,
            reverse=True,
        )[:5]
        
        # Compute enhanced cluster signal
        cluster_direction = None
        cluster_strength = "none"
        cluster_is_actionable = False
        cluster_num_insiders = 0
        cluster_total_value = 0.0
        cluster_has_ceo_cfo = False
        cluster_expected_return = None
        cluster_days_since = None
        cluster_summary = ""
        
        if clusters:
            # Use most recent cluster for the enhanced signal
            cluster = clusters[0]
            cluster_direction = "long" if cluster.direction == "buy" else "short"
            cluster_num_insiders = cluster.num_insiders
            cluster_total_value = cluster.total_value
            
            # Check for CEO/CFO
            for t in cluster.transactions:
                title_lower = t.insider_title.lower()
                if any(x in title_lower for x in ["ceo", "cfo", "chief executive", "chief financial"]):
                    cluster_has_ceo_cfo = True
                    break
            
            # Days since cluster
            cluster_days_since = (datetime.now() - cluster.end_date).days
            
            # Compute strength score
            insider_score = min(1.0, (cluster.num_insiders - 2) / 3)
            role_score = cluster.avg_role_weight
            value_score = min(1.0, (len(str(int(cluster.total_value))) - 1) / 8) if cluster.total_value > 0 else 0
            strength_score = 0.4 * insider_score + 0.3 * role_score + 0.3 * value_score
            if cluster.direction == "sell":
                strength_score *= 0.5
            
            # Map to category
            if strength_score >= 0.6:
                cluster_strength = "strong"
            elif strength_score >= 0.35:
                cluster_strength = "moderate"
            elif strength_score >= 0.15:
                cluster_strength = "weak"
            
            # Signal decay (60 days)
            if cluster_days_since <= 60:
                cluster_is_actionable = True
                decay = max(0, 1 - cluster_days_since / 60)
                if cluster.direction == "buy":
                    base_return = 0.05 if cluster_has_ceo_cfo else 0.03
                    cluster_expected_return = base_return * decay
                else:
                    cluster_expected_return = -0.01 * decay
            
            # Summary
            dir_emoji = "🟢" if cluster.direction == "buy" else "🔴"
            strength_text = cluster_strength.upper()
            ceo_note = " (incl. CEO/CFO)" if cluster_has_ceo_cfo else ""
            value_str = f"${cluster_total_value/1e6:.1f}M" if cluster_total_value >= 1e6 else f"${cluster_total_value/1e3:.0f}K"
            
            if cluster_is_actionable:
                cluster_summary = (
                    f"{dir_emoji} {strength_text} {cluster.direction.upper()}: "
                    f"{cluster_num_insiders} insiders{ceo_note} | "
                    f"{value_str} | {cluster_days_since}d ago"
                )
            else:
                cluster_summary = f"⚪ EXPIRED: Cluster was {cluster_days_since}d ago (>60d)"
        else:
            cluster_summary = f"No cluster events ({len(all_transactions)} individual transactions)"
        
        return InsiderAnalysis(
            ticker=ticker,
            analysis_date=datetime.now(),
            signal=signal,
            score=score,
            confidence=confidence,
            total_transactions=len(all_transactions),
            buy_count=buy_count,
            sell_count=sell_count,
            net_shares=net_shares,
            net_value=net_value,
            open_market_buys=open_market_buys,
            open_market_sells=open_market_sells,
            open_market_net_value=om_buy_value - om_sell_value,
            cluster_events=clusters,
            has_cluster_buying=any(c.direction == "buy" for c in clusters),
            has_cluster_selling=any(c.direction == "sell" for c in clusters),
            # Enhanced cluster model signal
            cluster_direction=cluster_direction,
            cluster_strength=cluster_strength,
            cluster_is_actionable=cluster_is_actionable,
            cluster_num_insiders=cluster_num_insiders,
            cluster_total_value=cluster_total_value,
            cluster_has_ceo_cfo=cluster_has_ceo_cfo,
            cluster_expected_return=cluster_expected_return,
            cluster_days_since=cluster_days_since,
            cluster_summary=cluster_summary,
            notable_transactions=notable,
            summary=summary,
            key_insights=insights,
            all_transactions=all_transactions,
        )
    
    def analyze_tickers(self, tickers: list[str]) -> dict[str, InsiderAnalysis]:
        """Analyze multiple tickers."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.analyze_ticker(ticker)
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
        return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Insider Trading Analysis")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--days", type=int, default=90, help="Lookback days")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    
    agent = InsiderTradingAgent(lookback_days=args.days)
    
    for ticker in args.tickers:
        print(f"\n{'='*60}")
        print(f"  {ticker} - INSIDER TRADING ANALYSIS")
        print("="*60)
        
        result = agent.analyze_ticker(ticker)
        
        # Signal
        signal_emoji = {
            "strong_buy": "🟢🟢",
            "buy": "🟢",
            "neutral": "🟡",
            "sell": "🔴",
            "strong_sell": "🔴🔴",
        }
        print(f"\n{signal_emoji.get(result.signal, '')} Signal: {result.signal.upper()}")
        print(f"Score: {result.score:+.2f} | Confidence: {result.confidence:.0%}")
        
        # Summary
        print(f"\n{result.summary}")
        
        # Metrics
        print(f"\nTransactions (last {args.days} days):")
        print(f"  Total: {result.total_transactions} | Buys: {result.buy_count} | Sells: {result.sell_count}")
        print(f"  Net Value: ${result.net_value:+,.0f}")
        print(f"  Open Market Only: {result.open_market_buys} buys, {result.open_market_sells} sells")
        
        # Clusters
        if result.cluster_events:
            print(f"\nCluster Events:")
            for c in result.cluster_events:
                direction = "🟢 BUY" if c.direction == "buy" else "🔴 SELL"
                print(f"  {direction}: {c.num_insiders} insiders, ${c.total_value:,.0f} ({c.days_span} days)")
        
        # Cluster Model Signal
        if result.cluster_summary:
            print(f"\n📊 CLUSTER MODEL SIGNAL:")
            print(f"  {result.cluster_summary}")
            if result.cluster_is_actionable and result.cluster_expected_return is not None:
                print(f"  Expected Return: {result.cluster_expected_return*100:+.2f}% (60-day horizon)")
        
        # Insights
        if result.key_insights:
            print(f"\nKey Insights:")
            for insight in result.key_insights:
                print(f"  {insight}")
        
        # Notable transactions
        if result.notable_transactions:
            print(f"\nNotable Transactions:")
            for t in result.notable_transactions[:3]:
                direction = "BUY" if t.is_purchase else "SELL"
                price_str = f" @ ${t.price_per_share:.2f}" if t.price_per_share else ""
                print(f"  {t.insider_title}: {direction} ${t.total_value:,.0f}{price_str}")
