"""
Emerging Technology Detection Model.

Attempts to detect early signals of technology adoption that could
provide competitive moats, by tracking:

1. Technology keyword mentions in news/filings
2. CHANGE in mention frequency (sudden increase = signal)
3. Relative adoption vs. sector peers (first-mover advantage)
4. Technology maturity stages (emerging → growth → mature)

================================================================================
SIGNAL HYPOTHESIS
================================================================================

The challenge: By the time tech is widely discussed, it's priced in.

Our edge: Track WHO mentions tech FIRST relative to peers.
    - First-movers in adopting tech often outperform
    - Sudden increase in mentions may precede announcements
    - Companies ahead of sector curve on tech investment

Limitations:
    - Can't predict truly novel breakthroughs
    - Management may hype without substance
    - Hard to distinguish R&D from marketing
    - Requires peer group comparison for context

================================================================================

Usage:
    from auto_researcher.models.emerging_tech import EmergingTechModel
    
    model = EmergingTechModel()
    signal = model.analyze_company("NVDA", articles, sector="Technology")
    print(f"Tech Adoption: {signal.adoption_stage}")
    print(f"Vs Peers: {signal.peer_percentile}th percentile")
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# EMERGING TECHNOLOGY LEXICON
# ==============================================================================

# Technologies organized by maturity stage and category
# Maturity: 1 = Emerging (high potential), 2 = Growth, 3 = Mature (priced in)

TECH_LEXICON = {
    # =========================================================================
    # ARTIFICIAL INTELLIGENCE & MACHINE LEARNING
    # =========================================================================
    "ai_core": {
        "maturity": 2,  # Now growth stage (was emerging 2022-2023)
        "keywords": [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "ai model", "ai system", "ai-powered",
            "ai platform", "ai solution", "ai capabilities",
        ],
        "moat_potential": "high",
    },
    
    "generative_ai": {
        "maturity": 2,  # Growth - GPT hype priced in, but differentiation still matters
        "keywords": [
            "generative ai", "large language model", "llm", "gpt",
            "chatgpt", "copilot", "claude", "gemini", "foundation model",
            "transformer model", "text generation", "image generation",
            "diffusion model", "stable diffusion", "midjourney",
        ],
        "moat_potential": "medium",  # Commoditizing fast
    },
    
    "ai_agents": {
        "maturity": 1,  # EMERGING - high potential
        "keywords": [
            "ai agent", "autonomous agent", "agentic ai", "agent framework",
            "multi-agent", "tool use", "function calling", "agent orchestration",
            "reasoning agent", "planning agent",
        ],
        "moat_potential": "high",
    },
    
    "ai_infrastructure": {
        "maturity": 2,
        "keywords": [
            "ai chip", "ai accelerator", "gpu cluster", "ai datacenter",
            "tensor core", "tpu", "inference engine", "training infrastructure",
            "ai hardware", "neural processing unit", "npu",
        ],
        "moat_potential": "very_high",  # Huge capex barrier
    },
    
    "edge_ai": {
        "maturity": 1,  # EMERGING
        "keywords": [
            "edge ai", "on-device ai", "edge inference", "tiny ml",
            "embedded ai", "ai at the edge", "local ai", "offline ai",
        ],
        "moat_potential": "high",
    },
    
    # =========================================================================
    # QUANTUM COMPUTING
    # =========================================================================
    "quantum_computing": {
        "maturity": 1,  # Still emerging - very long horizon
        "keywords": [
            "quantum computing", "quantum computer", "qubit", "quantum supremacy",
            "quantum advantage", "quantum algorithm", "quantum processor",
            "quantum error correction", "quantum simulation", "quantum annealing",
        ],
        "moat_potential": "very_high",  # Massive barrier if it works
    },
    
    "quantum_applications": {
        "maturity": 1,
        "keywords": [
            "quantum cryptography", "quantum sensing", "quantum network",
            "post-quantum", "quantum-safe", "quantum key distribution",
            "quantum machine learning", "variational quantum",
        ],
        "moat_potential": "high",
    },
    
    # =========================================================================
    # AUTONOMOUS SYSTEMS
    # =========================================================================
    "autonomous_vehicles": {
        "maturity": 2,  # Growth, but slower than expected
        "keywords": [
            "autonomous vehicle", "self-driving", "autonomous driving",
            "full self-driving", "fsd", "level 4", "level 5",
            "robotaxi", "driverless", "waymo", "cruise",
        ],
        "moat_potential": "high",
    },
    
    "robotics": {
        "maturity": 2,
        "keywords": [
            "humanoid robot", "industrial robot", "robot automation",
            "robotic process", "cobots", "collaborative robot",
            "warehouse robot", "delivery robot", "surgical robot",
        ],
        "moat_potential": "high",
    },
    
    "drones_uav": {
        "maturity": 2,
        "keywords": [
            "drone delivery", "uav", "unmanned aerial", "drone fleet",
            "autonomous drone", "drone inspection", "aerial robotics",
        ],
        "moat_potential": "medium",
    },
    
    # =========================================================================
    # BIOTECH & HEALTH
    # =========================================================================
    "gene_editing": {
        "maturity": 2,
        "keywords": [
            "crispr", "gene editing", "gene therapy", "genetic engineering",
            "cas9", "base editing", "prime editing", "gene modification",
        ],
        "moat_potential": "very_high",
    },
    
    "synthetic_biology": {
        "maturity": 1,  # EMERGING
        "keywords": [
            "synthetic biology", "synbio", "engineered biology",
            "cell programming", "biofoundry", "dna synthesis",
            "protein engineering", "metabolic engineering",
        ],
        "moat_potential": "very_high",
    },
    
    "ai_drug_discovery": {
        "maturity": 1,  # EMERGING but growing fast
        "keywords": [
            "ai drug discovery", "ai-designed drug", "computational drug",
            "alphafold", "protein folding", "ai pharma", "drug design ai",
            "molecular simulation", "virtual screening",
        ],
        "moat_potential": "high",
    },
    
    "longevity": {
        "maturity": 1,
        "keywords": [
            "longevity", "anti-aging", "age reversal", "senolytics",
            "lifespan extension", "biological age", "aging research",
        ],
        "moat_potential": "very_high",
    },
    
    # =========================================================================
    # ENERGY & SUSTAINABILITY
    # =========================================================================
    "nuclear_advanced": {
        "maturity": 1,  # EMERGING - renewed interest
        "keywords": [
            "small modular reactor", "smr", "nuclear fusion", "fusion energy",
            "advanced nuclear", "thorium reactor", "molten salt reactor",
            "iter", "fusion power",
        ],
        "moat_potential": "very_high",
    },
    
    "energy_storage": {
        "maturity": 2,
        "keywords": [
            "solid state battery", "sodium ion battery", "battery technology",
            "energy storage", "grid storage", "battery breakthrough",
            "lithium sulfur", "battery density",
        ],
        "moat_potential": "high",
    },
    
    "hydrogen": {
        "maturity": 1,  # Still emerging
        "keywords": [
            "green hydrogen", "hydrogen fuel", "hydrogen economy",
            "hydrogen production", "electrolyzer", "fuel cell",
            "hydrogen storage", "hydrogen infrastructure",
        ],
        "moat_potential": "high",
    },
    
    "carbon_capture": {
        "maturity": 1,
        "keywords": [
            "carbon capture", "carbon removal", "direct air capture",
            "carbon sequestration", "net zero", "carbon negative",
            "carbon offset", "decarbonization",
        ],
        "moat_potential": "medium",
    },
    
    # =========================================================================
    # SPACE & AEROSPACE
    # =========================================================================
    "space_commercial": {
        "maturity": 2,
        "keywords": [
            "space launch", "satellite constellation", "starlink",
            "reusable rocket", "space tourism", "orbital", "leo satellite",
            "space station", "space infrastructure",
        ],
        "moat_potential": "high",
    },
    
    "space_manufacturing": {
        "maturity": 1,  # Very early
        "keywords": [
            "space manufacturing", "in-space production", "asteroid mining",
            "space resources", "microgravity manufacturing", "lunar",
        ],
        "moat_potential": "very_high",
    },
    
    # =========================================================================
    # WEB3 & CRYPTO (Mostly priced in or deflated)
    # =========================================================================
    "blockchain_enterprise": {
        "maturity": 3,  # Mature/declining hype
        "keywords": [
            "blockchain", "distributed ledger", "smart contract",
            "tokenization", "web3", "defi", "nft",
        ],
        "moat_potential": "low",  # Mostly hype
    },
    
    # =========================================================================
    # SEMICONDUCTORS & COMPUTING
    # =========================================================================
    "advanced_chips": {
        "maturity": 2,
        "keywords": [
            "3nm", "2nm", "chiplet", "advanced packaging", "tsmc",
            "euv lithography", "gaafet", "high na euv", "chip manufacturing",
        ],
        "moat_potential": "very_high",
    },
    
    "neuromorphic": {
        "maturity": 1,  # EMERGING
        "keywords": [
            "neuromorphic", "brain-inspired", "spiking neural network",
            "neuromorphic computing", "memristor", "analog computing",
        ],
        "moat_potential": "high",
    },
    
    "photonics": {
        "maturity": 1,
        "keywords": [
            "silicon photonics", "optical computing", "photonic chip",
            "optical interconnect", "lidar chip", "photonic integration",
        ],
        "moat_potential": "high",
    },
}

# Categories for grouping
TECH_CATEGORIES = {
    "AI/ML": ["ai_core", "generative_ai", "ai_agents", "ai_infrastructure", "edge_ai"],
    "Quantum": ["quantum_computing", "quantum_applications"],
    "Autonomous": ["autonomous_vehicles", "robotics", "drones_uav"],
    "Biotech": ["gene_editing", "synthetic_biology", "ai_drug_discovery", "longevity"],
    "Energy": ["nuclear_advanced", "energy_storage", "hydrogen", "carbon_capture"],
    "Space": ["space_commercial", "space_manufacturing"],
    "Computing": ["advanced_chips", "neuromorphic", "photonics"],
    "Web3": ["blockchain_enterprise"],
}

# Sectors where each tech category is most relevant
SECTOR_TECH_RELEVANCE = {
    "Technology": ["AI/ML", "Computing", "Quantum", "Web3"],
    "Healthcare": ["Biotech", "AI/ML"],
    "Industrials": ["Autonomous", "AI/ML", "Energy"],
    "Energy": ["Energy", "AI/ML"],
    "Consumer Discretionary": ["Autonomous", "AI/ML", "Space"],
    "Financials": ["AI/ML", "Web3", "Quantum"],
    "Communication Services": ["AI/ML", "Computing", "Space"],
    "Materials": ["Energy", "Biotech"],
    "Utilities": ["Energy", "AI/ML"],
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TechMention:
    """A single technology mention in text."""
    tech_id: str
    tech_name: str
    category: str
    maturity: int  # 1=emerging, 2=growth, 3=mature
    keyword_matched: str
    moat_potential: str
    context: str  # Surrounding text


@dataclass
class CompanyTechProfile:
    """Technology adoption profile for a company."""
    ticker: str
    analysis_date: datetime
    
    # Mention counts by tech
    tech_mentions: Dict[str, int]
    category_mentions: Dict[str, int]
    
    # Maturity distribution
    emerging_mentions: int  # Maturity 1
    growth_mentions: int    # Maturity 2
    mature_mentions: int    # Maturity 3
    
    # Key metrics
    tech_intensity: float  # Total tech mentions / article count
    emerging_ratio: float  # Emerging / total tech mentions
    
    # Top technologies
    top_techs: List[Tuple[str, int]]  # (tech_id, count)
    top_emerging: List[Tuple[str, int]]  # Emerging tech only
    
    # Moat indicators
    high_moat_mentions: int  # Mentions of high/very_high moat potential tech
    moat_score: float  # 0-1 score based on moat potential


@dataclass
class TechAdoptionSignal:
    """Trading signal based on technology adoption."""
    ticker: str
    sector: Optional[str]
    
    # Profile
    profile: CompanyTechProfile
    
    # Peer comparison (if sector provided)
    peer_percentile: Optional[float] = None  # 0-100
    peer_emerging_percentile: Optional[float] = None
    ahead_of_peers: bool = False
    
    # Signal
    adoption_stage: str = "average"  # "laggard", "average", "leader", "pioneer"
    signal_strength: float = 0.0  # -1 to +1
    
    # Specific findings
    notable_emerging_tech: List[str] = field(default_factory=list)
    tech_advantage_areas: List[str] = field(default_factory=list)
    
    # Caveats
    caveats: List[str] = field(default_factory=list)


@dataclass 
class TechTrendSignal:
    """Signal based on technology mention trend over time."""
    ticker: str
    tech_id: str
    tech_name: str
    
    # Trend data
    mention_history: List[Tuple[datetime, int]]  # (period, count)
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_magnitude: float  # % change
    
    # Acceleration detection
    is_accelerating: bool = False
    acceleration_score: float = 0.0
    
    # Signal
    signal: str = "neutral"  # "bullish", "bearish", "neutral"


# ==============================================================================
# EMERGING TECH MODEL
# ==============================================================================

class EmergingTechModel:
    """
    Detects and scores emerging technology adoption from text.
    
    Key features:
    1. Classifies tech mentions by category and maturity
    2. Computes company tech intensity and emerging ratio
    3. Compares to sector peers (if peer data provided)
    4. Generates adoption signals (laggard/average/leader/pioneer)
    """
    
    def __init__(
        self,
        tech_lexicon: Optional[Dict] = None,
        min_mentions_for_signal: int = 2,
    ):
        """
        Initialize the emerging tech model.
        
        Args:
            tech_lexicon: Custom lexicon (uses defaults if None)
            min_mentions_for_signal: Minimum tech mentions to generate signal
        """
        self.lexicon = tech_lexicon or TECH_LEXICON
        self.min_mentions = min_mentions_for_signal
        
        # Build category lookup
        self.tech_to_category = {}
        for category, techs in TECH_CATEGORIES.items():
            for tech_id in techs:
                self.tech_to_category[tech_id] = category
        
        # Compile patterns
        self._compile_patterns()
        
        logger.info(f"EmergingTechModel initialized with {len(self.lexicon)} tech areas")
    
    def _compile_patterns(self):
        """Compile regex patterns for tech detection."""
        self.patterns = {}
        for tech_id, config in self.lexicon.items():
            keywords = config["keywords"]
            # Create pattern with word boundaries
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.patterns[tech_id] = re.compile(pattern, re.IGNORECASE)
    
    def detect_tech_mentions(self, text: str) -> List[TechMention]:
        """
        Detect technology mentions in text.
        
        Args:
            text: Article or filing text
            
        Returns:
            List of TechMention objects
        """
        mentions = []
        text_lower = text.lower()
        
        for tech_id, pattern in self.patterns.items():
            matches = pattern.finditer(text_lower)
            config = self.lexicon[tech_id]
            category = self.tech_to_category.get(tech_id, "Other")
            
            for match in matches:
                # Get context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                mentions.append(TechMention(
                    tech_id=tech_id,
                    tech_name=tech_id.replace("_", " ").title(),
                    category=category,
                    maturity=config["maturity"],
                    keyword_matched=match.group(),
                    moat_potential=config.get("moat_potential", "medium"),
                    context=context,
                ))
        
        return mentions
    
    def build_company_profile(
        self,
        articles: List[Dict],
        ticker: str,
        text_field: str = "title",
    ) -> CompanyTechProfile:
        """
        Build a technology profile for a company from articles.
        
        Args:
            articles: List of article dicts
            ticker: Stock ticker
            text_field: Field containing text to analyze
            
        Returns:
            CompanyTechProfile
        """
        tech_mentions = defaultdict(int)
        category_mentions = defaultdict(int)
        maturity_counts = {1: 0, 2: 0, 3: 0}
        moat_counts = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
        
        for article in articles:
            text = article.get(text_field, "") or ""
            if article.get("snippet"):
                text += " " + article.get("snippet", "")
            
            mentions = self.detect_tech_mentions(text)
            
            for m in mentions:
                tech_mentions[m.tech_id] += 1
                category_mentions[m.category] += 1
                maturity_counts[m.maturity] += 1
                moat_counts[m.moat_potential] += 1
        
        # Compute metrics
        total_tech = sum(tech_mentions.values())
        article_count = len(articles)
        
        tech_intensity = total_tech / article_count if article_count > 0 else 0
        emerging_ratio = maturity_counts[1] / total_tech if total_tech > 0 else 0
        
        # Top techs
        top_techs = sorted(tech_mentions.items(), key=lambda x: -x[1])[:5]
        top_emerging = [(t, c) for t, c in top_techs 
                       if self.lexicon.get(t, {}).get("maturity") == 1]
        
        # Moat score (weighted by moat potential)
        moat_weights = {"low": 0.25, "medium": 0.5, "high": 0.75, "very_high": 1.0}
        weighted_moat = sum(moat_counts[m] * moat_weights[m] for m in moat_counts)
        moat_score = weighted_moat / total_tech if total_tech > 0 else 0
        
        return CompanyTechProfile(
            ticker=ticker,
            analysis_date=datetime.now(),
            tech_mentions=dict(tech_mentions),
            category_mentions=dict(category_mentions),
            emerging_mentions=maturity_counts[1],
            growth_mentions=maturity_counts[2],
            mature_mentions=maturity_counts[3],
            tech_intensity=tech_intensity,
            emerging_ratio=emerging_ratio,
            top_techs=top_techs,
            top_emerging=top_emerging[:3],
            high_moat_mentions=moat_counts["high"] + moat_counts["very_high"],
            moat_score=moat_score,
        )
    
    def compare_to_peers(
        self,
        profile: CompanyTechProfile,
        peer_profiles: List[CompanyTechProfile],
    ) -> Tuple[float, float]:
        """
        Compare company tech profile to peers.
        
        Returns:
            (tech_intensity_percentile, emerging_ratio_percentile)
        """
        if not peer_profiles:
            return 50.0, 50.0
        
        # Compute percentiles
        intensities = [p.tech_intensity for p in peer_profiles] + [profile.tech_intensity]
        emerging_ratios = [p.emerging_ratio for p in peer_profiles] + [profile.emerging_ratio]
        
        # Simple percentile calculation
        intensity_percentile = (sum(1 for x in intensities if x < profile.tech_intensity) 
                               / len(intensities)) * 100
        emerging_percentile = (sum(1 for x in emerging_ratios if x < profile.emerging_ratio)
                              / len(emerging_ratios)) * 100
        
        return intensity_percentile, emerging_percentile
    
    def analyze_company(
        self,
        ticker: str,
        articles: List[Dict],
        sector: Optional[str] = None,
        peer_profiles: Optional[List[CompanyTechProfile]] = None,
        text_field: str = "title",
    ) -> TechAdoptionSignal:
        """
        Analyze a company's technology adoption and generate signal.
        
        Args:
            ticker: Stock ticker
            articles: Company's articles
            sector: Company's sector (for relevance filtering)
            peer_profiles: Profiles of peer companies for comparison
            text_field: Field containing article text
            
        Returns:
            TechAdoptionSignal
        """
        # Build profile
        profile = self.build_company_profile(articles, ticker, text_field)
        
        # Initialize signal
        signal = TechAdoptionSignal(
            ticker=ticker,
            sector=sector,
            profile=profile,
        )
        
        # Check minimum mentions
        total_mentions = sum(profile.tech_mentions.values())
        if total_mentions < self.min_mentions:
            signal.caveats.append(f"Low tech mentions ({total_mentions}), signal may be noisy")
        
        # Peer comparison if available
        if peer_profiles:
            intensity_pct, emerging_pct = self.compare_to_peers(profile, peer_profiles)
            signal.peer_percentile = intensity_pct
            signal.peer_emerging_percentile = emerging_pct
            signal.ahead_of_peers = intensity_pct > 60 and emerging_pct > 60
        
        # Determine adoption stage
        if profile.emerging_ratio > 0.4 and profile.tech_intensity > 0.3:
            signal.adoption_stage = "pioneer"
            signal.signal_strength = 0.8
        elif profile.emerging_ratio > 0.25 or (signal.peer_percentile and signal.peer_percentile > 75):
            signal.adoption_stage = "leader"
            signal.signal_strength = 0.5
        elif profile.emerging_ratio < 0.1 and profile.tech_intensity < 0.1:
            signal.adoption_stage = "laggard"
            signal.signal_strength = -0.3
        else:
            signal.adoption_stage = "average"
            signal.signal_strength = 0.0
        
        # Notable findings
        if profile.top_emerging:
            signal.notable_emerging_tech = [t[0].replace("_", " ").title() 
                                            for t in profile.top_emerging]
        
        # Tech advantage areas (high intensity in specific categories)
        for cat, count in profile.category_mentions.items():
            if count >= 3:
                signal.tech_advantage_areas.append(cat)
        
        # Sector relevance check
        if sector and sector in SECTOR_TECH_RELEVANCE:
            relevant_cats = SECTOR_TECH_RELEVANCE[sector]
            irrelevant_mentions = sum(c for cat, c in profile.category_mentions.items()
                                     if cat not in relevant_cats)
            if irrelevant_mentions > total_mentions * 0.5:
                signal.caveats.append("High mentions of sector-irrelevant tech (may be noise)")
        
        return signal
    
    def get_summary(self, signal: TechAdoptionSignal) -> str:
        """Generate human-readable summary."""
        lines = [f"Emerging Tech Analysis for {signal.ticker}"]
        lines.append("=" * 50)
        
        p = signal.profile
        lines.append(f"Tech Mentions: {sum(p.tech_mentions.values())}")
        lines.append(f"Tech Intensity: {p.tech_intensity:.2f} per article")
        lines.append(f"Emerging Ratio: {p.emerging_ratio:.1%}")
        lines.append(f"Moat Score: {p.moat_score:.2f}")
        
        if signal.peer_percentile is not None:
            lines.append("")
            lines.append(f"Peer Comparison:")
            lines.append(f"  Tech Intensity: {signal.peer_percentile:.0f}th percentile")
            lines.append(f"  Emerging Focus: {signal.peer_emerging_percentile:.0f}th percentile")
        
        lines.append("")
        lines.append(f"ADOPTION STAGE: {signal.adoption_stage.upper()}")
        lines.append(f"Signal Strength: {signal.signal_strength:+.2f}")
        
        if signal.notable_emerging_tech:
            lines.append("")
            lines.append(f"Notable Emerging Tech: {', '.join(signal.notable_emerging_tech)}")
        
        if signal.tech_advantage_areas:
            lines.append(f"Tech Advantage Areas: {', '.join(signal.tech_advantage_areas)}")
        
        if signal.caveats:
            lines.append("")
            for caveat in signal.caveats:
                lines.append(f"⚠️  {caveat}")
        
        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def analyze_tech_adoption(
    articles: List[Dict],
    ticker: str,
    sector: Optional[str] = None,
) -> TechAdoptionSignal:
    """Quick function to analyze tech adoption."""
    model = EmergingTechModel()
    return model.analyze_company(ticker, articles, sector=sector)


def get_emerging_tech_lexicon() -> Dict:
    """Return the emerging tech lexicon for customization."""
    return TECH_LEXICON.copy()


# ==============================================================================
# CONFIGURATION
# ==============================================================================

EMERGING_TECH_CONFIG = {
    "tech_categories": list(TECH_CATEGORIES.keys()),
    "sector_relevance": SECTOR_TECH_RELEVANCE,
    "adoption_stages": ["laggard", "average", "leader", "pioneer"],
    "maturity_levels": {1: "emerging", 2: "growth", 3: "mature"},
}


if __name__ == "__main__":
    # Demo
    model = EmergingTechModel()
    
    test_articles = [
        {"title": "NVIDIA announces new AI chips with 2x inference performance"},
        {"title": "NVIDIA partners with OpenAI on next-gen GPT training infrastructure"},
        {"title": "NVIDIA stock rises on strong datacenter demand"},
        {"title": "NVIDIA enters quantum computing with new quantum-classical hybrid platform"},
        {"title": "NVIDIA unveils neuromorphic computing research breakthrough"},
    ]
    
    signal = model.analyze_company("NVDA", test_articles, sector="Technology")
    print(model.get_summary(signal))
