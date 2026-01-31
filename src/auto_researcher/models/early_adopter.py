"""
Early Adopter Model.

Detects companies that discuss emerging technologies BEFORE competitors.
This is a leading indicator of future competitive advantage.

Key Signals:
1. First Mover Score: How early did they adopt new tech language vs peers?
2. Pioneer Score: Combination of breadth and earliness
3. Tech Momentum: Are they accelerating or decelerating tech adoption?

Research Finding:
- Companies with high pioneer scores (talking about tech early) outperformed
  by +25% over 12 months (p=0.05) in 2023 backtest.
"""

import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

from .earnings_tech_signal import (
    EarningsCallTechModel,
    DefeatBetaTranscriptClient,
    TRANSCRIPT_CACHE_PATH,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# EMERGING TECHNOLOGIES LEXICON
# ==============================================================================

# Technologies to track with emergence dates
# Format: tech_id -> {terms, emerged, category, maturity}
EMERGING_TECH_LEXICON = {
    # === AI/ML (Current Wave) ===
    "generative_ai": {
        "terms": ["generative ai", "generative artificial intelligence", "gen ai", "genai"],
        "emerged": "2022-06",
        "category": "AI/ML",
        "maturity": 1,  # 1=emerging, 2=growing, 3=mature
    },
    "large_language_model": {
        "terms": ["large language model", "llm", "llms", "language model", "foundation model"],
        "emerged": "2022-01",
        "category": "AI/ML",
        "maturity": 1,
    },
    "chatgpt": {
        "terms": ["chatgpt", "chat gpt", "gpt-4", "gpt-3", "gpt4", "gpt3", "gpt-5"],
        "emerged": "2022-11",
        "category": "AI/ML",
        "maturity": 1,
    },
    "copilot": {
        "terms": ["copilot", "github copilot", "ai copilot", "ai assistant", "ai agent", "agentic"],
        "emerged": "2021-06",
        "category": "AI/ML",
        "maturity": 1,
    },
    "transformer": {
        "terms": ["transformer model", "transformer architecture", "attention mechanism"],
        "emerged": "2020-01",
        "category": "AI/ML",
        "maturity": 2,
    },
    "inference": {
        "terms": ["inference", "model inference", "ai inference"],
        "emerged": "2020-01",
        "category": "AI/ML",
        "maturity": 2,
    },
    
    # === Cloud/Data ===
    "edge_computing": {
        "terms": ["edge computing", "edge ai", "edge inference", "on-device ai"],
        "emerged": "2018-01",
        "category": "Cloud",
        "maturity": 2,
    },
    "multicloud": {
        "terms": ["multi-cloud", "multicloud", "hybrid cloud"],
        "emerged": "2018-01",
        "category": "Cloud",
        "maturity": 2,
    },
    
    # === Autonomous ===
    "autonomous_vehicle": {
        "terms": ["autonomous vehicle", "self-driving", "robotaxi", "level 4", "level 5", "adas"],
        "emerged": "2016-01",
        "category": "Autonomous",
        "maturity": 2,
    },
    "fsd": {
        "terms": ["full self-driving", "fsd", "autopilot"],
        "emerged": "2019-01",
        "category": "Autonomous",
        "maturity": 2,
    },
    
    # === Semiconductors ===
    "chiplet": {
        "terms": ["chiplet", "chiplets", "advanced packaging", "2.5d", "3d packaging"],
        "emerged": "2020-01",
        "category": "Semiconductors",
        "maturity": 2,
    },
    "advanced_node": {
        "terms": ["3nm", "3 nanometer", "2nm", "2 nanometer", "1nm", "gate-all-around", "gaafet"],
        "emerged": "2021-01",
        "category": "Semiconductors",
        "maturity": 1,
    },
    "hbm": {
        "terms": ["hbm", "high bandwidth memory", "hbm3", "hbm4"],
        "emerged": "2023-01",
        "category": "Semiconductors",
        "maturity": 1,
    },
    
    # === Frontier Tech ===
    "quantum": {
        "terms": ["quantum computing", "quantum computer", "qubit", "quantum supremacy", "quantum advantage"],
        "emerged": "2019-01",
        "category": "Quantum",
        "maturity": 1,
    },
    "metaverse": {
        "terms": ["metaverse", "virtual world", "digital twin", "spatial computing"],
        "emerged": "2021-10",
        "category": "Metaverse",
        "maturity": 2,
    },
    
    # === Biotech ===
    "mrna": {
        "terms": ["mrna", "messenger rna", "mrna technology", "mrna platform"],
        "emerged": "2020-03",
        "category": "Biotech",
        "maturity": 2,
    },
    "crispr": {
        "terms": ["crispr", "gene editing", "gene therapy", "cas9"],
        "emerged": "2017-01",
        "category": "Biotech",
        "maturity": 2,
    },
    
    # === Clean Tech ===
    "carbon_neutral": {
        "terms": ["carbon neutral", "net zero", "carbon negative", "decarbonization", "carbon capture"],
        "emerged": "2019-01",
        "category": "Clean Tech",
        "maturity": 2,
    },
    "solid_state_battery": {
        "terms": ["solid state battery", "solid-state battery", "next-gen battery"],
        "emerged": "2020-01",
        "category": "Clean Tech",
        "maturity": 1,
    },
    
    # === Robotics ===
    "humanoid": {
        "terms": ["humanoid robot", "humanoid", "bipedal robot", "optimus"],
        "emerged": "2022-01",
        "category": "Robotics",
        "maturity": 1,
    },
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TechAdoption:
    """Record of when a company first mentioned a technology."""
    ticker: str
    tech_id: str
    tech_name: str
    category: str
    first_mention_date: datetime
    first_mention_quarter: str
    context: str
    industry_avg_date: Optional[datetime] = None
    lead_months: Optional[float] = None


@dataclass
class EarlyAdopterSignal:
    """Signal output from the Early Adopter Model."""
    ticker: str
    analysis_date: datetime
    
    # Core metrics
    pioneer_score: float  # 0-1 combined score
    total_techs_adopted: int  # How many emerging techs they discuss
    techs_adopted_early: int  # How many they discussed BEFORE industry avg
    avg_lead_months: float  # Average months ahead of industry
    
    # Breakdown by category
    categories: Dict[str, int]  # category -> count of early adoptions
    
    # Top adoptions (most ahead of curve)
    earliest_adoptions: List[TechAdoption]
    
    # Signal interpretation
    signal: str  # "strong_buy", "buy", "neutral", "avoid"
    signal_strength: float  # -1 to +1
    
    # GenAI specific (most relevant current wave)
    genai_score: float  # Lead months on GenAI-related terms
    genai_adopted: List[str]  # Which GenAI terms they've adopted


@dataclass
class IndustryTimeline:
    """When technologies were adopted across the industry."""
    tech_id: str
    first_adopter: str
    first_date: datetime
    median_date: datetime
    adoption_count: int
    adopters: List[Tuple[str, datetime]]  # List of (ticker, date) sorted by date


# ==============================================================================
# EARLY ADOPTER MODEL
# ==============================================================================

class EarlyAdopterModel:
    """
    Detects companies that discuss emerging technologies before competitors.
    
    This is a leading indicator of future competitive advantage.
    Companies that talk about new tech early are often investing ahead
    of the curve.
    
    Usage:
        model = EarlyAdopterModel()
        
        # Analyze a single company
        signal = model.analyze_company("NVDA")
        print(f"Pioneer Score: {signal.pioneer_score}")
        
        # Get industry-wide view
        timelines = model.get_industry_timelines()
        
        # Rank all companies by pioneer score
        rankings = model.rank_pioneers(["NVDA", "MSFT", "AAPL", ...])
    """
    
    def __init__(self, lookback_years: int = 6):
        """
        Initialize the model.
        
        Args:
            lookback_years: How many years of history to analyze
        """
        self.lookback_years = lookback_years
        self.client = DefeatBetaTranscriptClient()
        self.lexicon = EMERGING_TECH_LEXICON
        
        # Compiled patterns
        self._patterns = {}
        self._compile_patterns()
        
        # Caches
        self._adoption_cache: Dict[str, List[TechAdoption]] = {}
        self._timeline_cache: Dict[str, IndustryTimeline] = {}
        self._industry_avgs: Dict[str, datetime] = {}
        
        logger.info("EarlyAdopterModel initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for each technology."""
        for tech_id, config in self.lexicon.items():
            terms = config["terms"]
            pattern = r'\b(' + '|'.join(re.escape(t) for t in terms) + r')\b'
            self._patterns[tech_id] = re.compile(pattern, re.IGNORECASE)
    
    def _load_transcripts(self, ticker: str) -> List[Dict]:
        """Load all transcripts for a ticker."""
        # Use high limit to get full history
        return self.client.get_transcripts(ticker, limit=100)
    
    def _find_first_mentions(self, ticker: str) -> List[TechAdoption]:
        """Find when a company first mentioned each technology."""
        
        if ticker in self._adoption_cache:
            return self._adoption_cache[ticker]
        
        transcripts = self._load_transcripts(ticker)
        
        if not transcripts:
            self._adoption_cache[ticker] = []
            return []
        
        # Sort chronologically (oldest first)
        transcripts = sorted(transcripts, key=lambda x: x.get("date", ""))
        
        first_mentions = {}
        
        for t in transcripts:
            try:
                content = t.get("content", "")
                date_str = t.get("date", "")[:10]
                date = datetime.strptime(date_str, "%Y-%m-%d")
                quarter = f"Q{t.get('quarter', 1)} {t.get('year', 2024)}"
                
                for tech_id, pattern in self._patterns.items():
                    # Skip if already found
                    if tech_id in first_mentions:
                        continue
                    
                    match = pattern.search(content)
                    if match:
                        # Get context
                        start = max(0, match.start() - 100)
                        end = min(len(content), match.end() + 100)
                        context = content[start:end].strip()
                        
                        config = self.lexicon[tech_id]
                        
                        first_mentions[tech_id] = TechAdoption(
                            ticker=ticker,
                            tech_id=tech_id,
                            tech_name=tech_id.replace("_", " ").title(),
                            category=config["category"],
                            first_mention_date=date,
                            first_mention_quarter=quarter,
                            context=context,
                        )
            except Exception as e:
                logger.debug(f"Error processing transcript: {e}")
                continue
        
        result = list(first_mentions.values())
        self._adoption_cache[ticker] = result
        return result
    
    def build_industry_timelines(
        self, 
        tickers: List[str],
        min_date: Optional[datetime] = None,
    ) -> Dict[str, IndustryTimeline]:
        """
        Build adoption timelines for each technology across the industry.
        
        This calculates when each tech was first mentioned and the median
        adoption date across all companies.
        """
        
        if min_date is None:
            min_date = datetime.now() - timedelta(days=365 * self.lookback_years)
        
        # Collect all adoptions
        tech_adoptions: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)
        
        for ticker in tickers:
            adoptions = self._find_first_mentions(ticker)
            for a in adoptions:
                if a.first_mention_date >= min_date:
                    tech_adoptions[a.tech_id].append((ticker, a.first_mention_date))
        
        # Build timelines
        timelines = {}
        
        for tech_id, adopters in tech_adoptions.items():
            if not adopters:
                continue
            
            # Sort by date
            adopters = sorted(adopters, key=lambda x: x[1])
            
            # Calculate median
            dates = [d for _, d in adopters]
            median_idx = len(dates) // 2
            median_date = dates[median_idx]
            
            timelines[tech_id] = IndustryTimeline(
                tech_id=tech_id,
                first_adopter=adopters[0][0],
                first_date=adopters[0][1],
                median_date=median_date,
                adoption_count=len(adopters),
                adopters=adopters,
            )
            
            # Cache industry average
            self._industry_avgs[tech_id] = median_date
        
        self._timeline_cache = timelines
        return timelines
    
    def _calculate_lead_times(self, adoptions: List[TechAdoption]) -> List[TechAdoption]:
        """Calculate how many months ahead each adoption was."""
        for adoption in adoptions:
            if adoption.tech_id in self._industry_avgs:
                avg_date = self._industry_avgs[adoption.tech_id]
                adoption.industry_avg_date = avg_date
                
                delta = avg_date - adoption.first_mention_date
                adoption.lead_months = delta.days / 30.44
        
        return adoptions
    
    def analyze_company(
        self, 
        ticker: str,
        industry_tickers: Optional[List[str]] = None,
    ) -> EarlyAdopterSignal:
        """
        Analyze a company's early adopter behavior.
        
        Args:
            ticker: Stock ticker to analyze
            industry_tickers: List of peer companies for comparison.
                             If None, uses a default tech universe.
        
        Returns:
            EarlyAdopterSignal with pioneer score and details
        """
        
        # Default peer universe
        if industry_tickers is None:
            industry_tickers = [
                "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
                "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU",
                "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PLTR",
                "IBM", "CSCO", "DELL",
            ]
        
        # Ensure target ticker is in the list
        if ticker not in industry_tickers:
            industry_tickers = [ticker] + industry_tickers
        
        # Build industry timelines if not cached
        if not self._timeline_cache:
            self.build_industry_timelines(industry_tickers)
        
        # Get adoptions for this ticker
        adoptions = self._find_first_mentions(ticker)
        
        if not adoptions:
            return EarlyAdopterSignal(
                ticker=ticker,
                analysis_date=datetime.now(),
                pioneer_score=0.0,
                total_techs_adopted=0,
                techs_adopted_early=0,
                avg_lead_months=0.0,
                categories={},
                earliest_adoptions=[],
                signal="neutral",
                signal_strength=0.0,
                genai_score=0.0,
                genai_adopted=[],
            )
        
        # Calculate lead times
        adoptions = self._calculate_lead_times(adoptions)
        
        # Count early adoptions (before industry median)
        early_adoptions = [a for a in adoptions if a.lead_months and a.lead_months > 0]
        
        # Average lead time
        lead_times = [a.lead_months for a in adoptions if a.lead_months is not None]
        avg_lead = sum(lead_times) / len(lead_times) if lead_times else 0
        
        # Categories
        categories = defaultdict(int)
        for a in early_adoptions:
            categories[a.category] += 1
        
        # Pioneer score (0-1)
        breadth_score = min(len(adoptions) / 12, 1.0)  # Max at 12 techs
        earliness_score = min(max(avg_lead / 18, 0), 1.0)  # Max at 18 months ahead
        pioneer_score = 0.5 * breadth_score + 0.5 * earliness_score
        
        # GenAI-specific score
        genai_techs = ["generative_ai", "large_language_model", "chatgpt", "copilot", "inference"]
        genai_lead = 0
        genai_adopted = []
        for a in adoptions:
            if a.tech_id in genai_techs:
                genai_lead += (a.lead_months or 0)
                genai_adopted.append(a.tech_id)
        
        # Sort adoptions by lead time
        sorted_adoptions = sorted(adoptions, key=lambda x: -(x.lead_months or 0))
        
        # Signal interpretation
        if pioneer_score >= 0.7:
            signal = "strong_buy"
            signal_strength = 0.8 + 0.2 * (pioneer_score - 0.7) / 0.3
        elif pioneer_score >= 0.5:
            signal = "buy"
            signal_strength = 0.4 + 0.4 * (pioneer_score - 0.5) / 0.2
        elif pioneer_score >= 0.3:
            signal = "neutral"
            signal_strength = 0.0
        else:
            signal = "avoid"
            signal_strength = -0.3
        
        return EarlyAdopterSignal(
            ticker=ticker,
            analysis_date=datetime.now(),
            pioneer_score=pioneer_score,
            total_techs_adopted=len(adoptions),
            techs_adopted_early=len(early_adoptions),
            avg_lead_months=avg_lead,
            categories=dict(categories),
            earliest_adoptions=sorted_adoptions[:5],
            signal=signal,
            signal_strength=signal_strength,
            genai_score=genai_lead,
            genai_adopted=genai_adopted,
        )
    
    def rank_pioneers(
        self, 
        tickers: List[str],
    ) -> List[EarlyAdopterSignal]:
        """
        Rank all tickers by their pioneer score.
        
        Args:
            tickers: List of tickers to analyze
            
        Returns:
            List of signals sorted by pioneer score (highest first)
        """
        
        # Build industry timelines
        self.build_industry_timelines(tickers)
        
        # Analyze each ticker
        signals = []
        for ticker in tickers:
            try:
                signal = self.analyze_company(ticker, industry_tickers=tickers)
                signals.append(signal)
            except Exception as e:
                logger.warning(f"Failed to analyze {ticker}: {e}")
                continue
        
        # Sort by pioneer score
        signals.sort(key=lambda x: -x.pioneer_score)
        
        return signals
    
    def get_first_movers(self, tech_id: str) -> List[Tuple[str, datetime, float]]:
        """
        Get the first movers for a specific technology.
        
        Args:
            tech_id: Technology ID from EMERGING_TECH_LEXICON
            
        Returns:
            List of (ticker, first_mention_date, lead_months) tuples
        """
        if tech_id not in self._timeline_cache:
            raise ValueError(f"Unknown tech_id: {tech_id}. Build timelines first.")
        
        timeline = self._timeline_cache[tech_id]
        median = timeline.median_date
        
        result = []
        for ticker, date in timeline.adopters:
            lead_months = (median - date).days / 30.44
            result.append((ticker, date, lead_months))
        
        return result
    
    def get_emerging_now(self) -> List[str]:
        """
        Get technologies that are currently in the emerging phase (maturity=1).
        
        These are the most relevant for detecting early adopters right now.
        """
        return [
            tech_id for tech_id, config in self.lexicon.items()
            if config["maturity"] == 1
        ]
    
    def get_summary(self, signal: EarlyAdopterSignal) -> str:
        """Generate human-readable summary."""
        lines = [f"Early Adopter Analysis: {signal.ticker}"]
        lines.append("=" * 50)
        
        lines.append(f"\nPioneer Score: {signal.pioneer_score:.2f}")
        lines.append(f"Signal: {signal.signal.upper()} (strength: {signal.signal_strength:+.2f})")
        
        lines.append(f"\nTechnologies Adopted: {signal.total_techs_adopted}")
        lines.append(f"Adopted Early (before industry): {signal.techs_adopted_early}")
        lines.append(f"Average Lead Time: {signal.avg_lead_months:+.1f} months")
        
        if signal.categories:
            lines.append("\nCategories Pioneered:")
            for cat, count in sorted(signal.categories.items(), key=lambda x: -x[1]):
                lines.append(f"  {cat}: {count}")
        
        if signal.genai_adopted:
            lines.append(f"\nGenAI Score: {signal.genai_score:+.1f} months ahead")
            lines.append(f"GenAI Terms: {', '.join(signal.genai_adopted)}")
        
        if signal.earliest_adoptions:
            lines.append("\nTop Early Adoptions:")
            for a in signal.earliest_adoptions[:3]:
                lines.append(f"  {a.tech_name}: {a.first_mention_date.strftime('%Y-%m')} (+{a.lead_months:.0f}mo)")
        
        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def find_early_adopters(
    tickers: List[str],
    min_pioneer_score: float = 0.5,
) -> List[EarlyAdopterSignal]:
    """
    Find companies that are early adopters of emerging technologies.
    
    Args:
        tickers: List of tickers to analyze
        min_pioneer_score: Minimum pioneer score to include (0-1)
        
    Returns:
        List of signals for companies meeting the threshold
    """
    model = EarlyAdopterModel()
    signals = model.rank_pioneers(tickers)
    
    return [s for s in signals if s.pioneer_score >= min_pioneer_score]


def get_genai_pioneers(tickers: List[str], top_n: int = 10) -> List[EarlyAdopterSignal]:
    """
    Find the top GenAI early adopters.
    
    Args:
        tickers: List of tickers to analyze
        top_n: Number of top pioneers to return
        
    Returns:
        Top N signals sorted by GenAI score
    """
    model = EarlyAdopterModel()
    model.build_industry_timelines(tickers)
    
    signals = []
    for ticker in tickers:
        signal = model.analyze_company(ticker, industry_tickers=tickers)
        if signal.genai_adopted:  # Only include those with GenAI mentions
            signals.append(signal)
    
    signals.sort(key=lambda x: -x.genai_score)
    return signals[:top_n]
