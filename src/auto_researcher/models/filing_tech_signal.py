"""
Filing-Based Technology Signal.

Extracts technology/innovation signals from 10-K and 10-Q filings.
Uses the same SEC filing infrastructure already in place.

================================================================================
SIGNAL HYPOTHESIS
================================================================================

10-K filings contain forward-looking R&D discussion that:
    1. Reveals technology investment priorities
    2. Mentions specific emerging technologies
    3. Shows changes in strategic tech focus YoY
    
Key sections:
    - Item 1: Business Description (includes R&D discussion)
    - Item 7: MD&A (Management Discussion & Analysis)
    - Risk Factors (often mention tech dependencies)

This works better than news because:
    1. Companies MUST disclose material R&D investments
    2. Language is more specific/technical
    3. Annual cadence allows for YoY comparison
    
================================================================================

Usage:
    from auto_researcher.models.filing_tech_signal import FilingTechSignal
    
    model = FilingTechSignal()
    signal = await model.analyze_filing(ticker="NVDA")
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==============================================================================
# TECHNOLOGY LEXICON FOR FILINGS
# ==============================================================================

# More specific terms that appear in 10-K filings
# Organized by technology area with maturity levels

FILING_TECH_LEXICON = {
    # AI/ML - More formal language used in filings
    "ai_ml": {
        "terms": [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "natural language processing", "computer vision",
            "generative AI", "large language model", "foundation model",
            "inference", "training", "reinforcement learning",
            "AI-powered", "AI-enabled", "AI-driven", "ML model",
            "generative artificial intelligence",
        ],
        "name": "AI/Machine Learning",
        "maturity": 2,  # Growth phase
        "moat_weight": 0.9,
    },
    
    # Quantum
    "quantum": {
        "terms": [
            "quantum computing", "quantum processor", "qubit",
            "quantum advantage", "quantum supremacy", "quantum annealing",
            "quantum algorithm", "quantum machine learning",
        ],
        "name": "Quantum Computing",
        "maturity": 1,  # Emerging
        "moat_weight": 1.0,
    },
    
    # Cloud/Infrastructure
    "cloud": {
        "terms": [
            "cloud computing", "cloud infrastructure", "cloud services",
            "data center", "edge computing", "hybrid cloud", "multi-cloud",
            "infrastructure as a service", "platform as a service",
            "containerization", "kubernetes", "microservices",
        ],
        "name": "Cloud Infrastructure",
        "maturity": 3,  # Mature
        "moat_weight": 0.5,
    },
    
    # Autonomous Systems
    "autonomous": {
        "terms": [
            "autonomous vehicle", "self-driving", "autonomous driving",
            "ADAS", "advanced driver assistance", "autonomous mobility",
            "robotaxi", "level 4 autonomy", "level 5 autonomy",
            "autonomous systems", "unmanned aerial", "drone",
        ],
        "name": "Autonomous Systems",
        "maturity": 2,
        "moat_weight": 0.9,
    },
    
    # Semiconductor
    "semiconductor": {
        "terms": [
            "semiconductor", "chip", "processor", "GPU", "ASIC", "FPGA",
            "fabrication", "nanometer", "process node", "wafer",
            "packaging", "chiplet", "system on chip", "SoC",
            "accelerator", "tensor core",
        ],
        "name": "Semiconductors",
        "maturity": 2,
        "moat_weight": 0.8,
    },
    
    # Biotech/Life Sciences
    "biotech": {
        "terms": [
            "gene therapy", "CRISPR", "genomics", "proteomics",
            "mRNA", "immunotherapy", "CAR-T", "antibody",
            "precision medicine", "personalized medicine",
            "clinical trial", "FDA approval", "drug candidate",
            "therapeutic", "biologics", "biomarker",
        ],
        "name": "Biotechnology",
        "maturity": 2,
        "moat_weight": 0.9,
    },
    
    # Clean Energy
    "clean_energy": {
        "terms": [
            "renewable energy", "solar", "wind power", "battery storage",
            "electric vehicle", "EV", "hydrogen fuel", "fuel cell",
            "carbon neutral", "net zero", "sustainability",
            "energy storage", "grid scale", "lithium ion",
        ],
        "name": "Clean Energy",
        "maturity": 2,
        "moat_weight": 0.7,
    },
    
    # 5G/Connectivity
    "connectivity": {
        "terms": [
            "5G", "wireless", "spectrum", "mmWave", "network infrastructure",
            "IoT", "internet of things", "connected devices",
            "low latency", "network slicing",
        ],
        "name": "5G/Connectivity",
        "maturity": 3,
        "moat_weight": 0.5,
    },
    
    # Cybersecurity
    "cybersecurity": {
        "terms": [
            "cybersecurity", "cyber security", "zero trust",
            "encryption", "authentication", "threat detection",
            "security operations", "vulnerability", "ransomware",
            "security platform", "identity management",
        ],
        "name": "Cybersecurity",
        "maturity": 2,
        "moat_weight": 0.7,
    },
    
    # Blockchain/Web3
    "blockchain": {
        "terms": [
            "blockchain", "distributed ledger", "smart contract",
            "cryptocurrency", "digital asset", "tokenization",
            "decentralized", "web3",
        ],
        "name": "Blockchain/Web3",
        "maturity": 3,  # Mature/stabilizing
        "moat_weight": 0.4,
    },
    
    # AR/VR/Metaverse
    "spatial": {
        "terms": [
            "augmented reality", "virtual reality", "mixed reality",
            "metaverse", "spatial computing", "XR", "head-mounted display",
            "immersive", "3D rendering", "haptic",
        ],
        "name": "AR/VR/Spatial",
        "maturity": 2,
        "moat_weight": 0.6,
    },
}

# R&D investment language
RD_INVESTMENT_TERMS = [
    "research and development", "R&D", "R&D expenses", "R&D investment",
    "product development", "technology development", "innovation",
    "engineering expenses", "research costs",
]

# Competition/moat language
MOAT_LANGUAGE = {
    "strong_moat": [
        "proprietary technology", "competitive advantage", "barrier to entry",
        "intellectual property", "patents", "trade secrets",
        "first mover", "market leader", "industry leader",
        "unique technology", "differentiated", "defensible",
    ],
    "weak_moat": [
        "intense competition", "commoditized", "price pressure",
        "low switching costs", "many competitors",
        "competitive landscape", "competitive threats",
    ],
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TechMention:
    """A technology mention in a filing."""
    tech_area: str
    term: str
    context: str  # Surrounding text
    section: str  # Item 1, Item 7, etc.
    sentiment: str = "neutral"  # positive, negative, neutral


@dataclass 
class FilingTechProfile:
    """Technology profile from a single filing."""
    ticker: str
    filing_date: datetime
    filing_type: str  # 10-K, 10-Q
    
    # Mentions
    mentions: List[TechMention]
    mentions_by_area: Dict[str, int]
    
    # Scores
    tech_intensity: float  # Total mentions normalized
    emerging_focus: float  # % emerging tech
    moat_strength: float  # Strong moat language - weak moat language
    
    # R&D
    rd_mentions: int
    rd_growth_mentioned: bool


@dataclass
class FilingTechSignalResult:
    """Technology signal from filing analysis."""
    ticker: str
    analysis_date: datetime
    
    # Current profile
    current_profile: Optional[FilingTechProfile]
    
    # Year over year
    yoy_tech_change: Optional[float]  # % change in tech intensity
    yoy_emerging_change: Optional[float]  # Change in emerging focus
    
    # Key findings
    top_tech_areas: List[str]
    new_tech_mentions: List[str]  # Tech mentioned this year but not last
    dropped_tech: List[str]  # Tech mentioned last year but not this
    
    # Signal
    innovation_trajectory: str  # "accelerating", "stable", "decelerating"
    signal_strength: float
    tradeable: bool
    expected_alpha: float
    
    # Evidence
    key_quotes: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


# ==============================================================================
# FILING TECH SIGNAL MODEL
# ==============================================================================

class FilingTechSignal:
    """
    Extracts technology/innovation signals from SEC filings.
    
    Works with existing filing infrastructure.
    """
    
    def __init__(self):
        """Initialize filing tech signal model."""
        self.tech_lexicon = FILING_TECH_LEXICON
        self.rd_terms = RD_INVESTMENT_TERMS
        self.moat_language = MOAT_LANGUAGE
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info("FilingTechSignal model initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.tech_patterns = {}
        for tech_id, config in self.tech_lexicon.items():
            terms = config["terms"]
            # Create pattern that matches any term (case-insensitive)
            pattern = r'\b(' + '|'.join(re.escape(t) for t in terms) + r')\b'
            self.tech_patterns[tech_id] = re.compile(pattern, re.IGNORECASE)
        
        # R&D pattern
        rd_pattern = r'\b(' + '|'.join(re.escape(t) for t in self.rd_terms) + r')\b'
        self.rd_pattern = re.compile(rd_pattern, re.IGNORECASE)
        
        # Moat patterns
        strong_pattern = r'\b(' + '|'.join(re.escape(t) for t in self.moat_language["strong_moat"]) + r')\b'
        weak_pattern = r'\b(' + '|'.join(re.escape(t) for t in self.moat_language["weak_moat"]) + r')\b'
        self.strong_moat_pattern = re.compile(strong_pattern, re.IGNORECASE)
        self.weak_moat_pattern = re.compile(weak_pattern, re.IGNORECASE)
    
    def analyze_text(
        self,
        text: str,
        ticker: str,
        filing_date: datetime,
        filing_type: str = "10-K",
    ) -> FilingTechProfile:
        """
        Analyze a filing text for technology signals.
        
        Args:
            text: Full text or relevant sections of filing
            ticker: Stock ticker
            filing_date: Filing date
            filing_type: 10-K or 10-Q
            
        Returns:
            FilingTechProfile
        """
        mentions = []
        mentions_by_area = defaultdict(int)
        
        # Find all tech mentions
        for tech_id, pattern in self.tech_patterns.items():
            for match in pattern.finditer(text):
                term = match.group()
                
                # Get context (50 chars before and after)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                mentions.append(TechMention(
                    tech_area=tech_id,
                    term=term.lower(),
                    context=context.strip(),
                    section="",  # Would need section parsing
                ))
                
                mentions_by_area[tech_id] += 1
        
        # Calculate tech intensity (per 10K characters)
        text_len = len(text) / 10000
        tech_intensity = sum(mentions_by_area.values()) / max(text_len, 1)
        
        # Emerging focus
        emerging_count = 0
        total_weighted = 0
        for tech_id, count in mentions_by_area.items():
            maturity = self.tech_lexicon[tech_id]["maturity"]
            if maturity == 1:  # Emerging
                emerging_count += count
            total_weighted += count
        
        emerging_focus = emerging_count / max(total_weighted, 1)
        
        # Moat strength
        strong_matches = len(self.strong_moat_pattern.findall(text))
        weak_matches = len(self.weak_moat_pattern.findall(text))
        moat_strength = (strong_matches - weak_matches) / max(strong_matches + weak_matches, 1)
        
        # R&D mentions
        rd_matches = self.rd_pattern.findall(text)
        rd_mentions = len(rd_matches)
        
        # Check for R&D growth language
        rd_growth = bool(re.search(
            r'(increas|grow|expand|invest).{0,20}(R&D|research and development)',
            text, re.IGNORECASE
        ))
        
        return FilingTechProfile(
            ticker=ticker,
            filing_date=filing_date,
            filing_type=filing_type,
            mentions=mentions,
            mentions_by_area=dict(mentions_by_area),
            tech_intensity=tech_intensity,
            emerging_focus=emerging_focus,
            moat_strength=moat_strength,
            rd_mentions=rd_mentions,
            rd_growth_mentioned=rd_growth,
        )
    
    def compare_filings(
        self,
        current: FilingTechProfile,
        previous: Optional[FilingTechProfile],
    ) -> FilingTechSignalResult:
        """
        Compare current filing to previous for YoY changes.
        
        Args:
            current: Current filing profile
            previous: Previous year's filing profile (optional)
            
        Returns:
            FilingTechSignalResult
        """
        # Top tech areas
        sorted_tech = sorted(
            current.mentions_by_area.items(),
            key=lambda x: -x[1]
        )
        top_areas = [self.tech_lexicon[t]["name"] for t, _ in sorted_tech[:5]]
        
        # YoY comparison
        yoy_tech_change = None
        yoy_emerging_change = None
        new_tech = []
        dropped_tech = []
        
        if previous:
            # Tech intensity change
            if previous.tech_intensity > 0:
                yoy_tech_change = (
                    (current.tech_intensity - previous.tech_intensity) 
                    / previous.tech_intensity
                )
            
            # Emerging focus change
            yoy_emerging_change = current.emerging_focus - previous.emerging_focus
            
            # New vs dropped tech areas
            current_areas = set(current.mentions_by_area.keys())
            previous_areas = set(previous.mentions_by_area.keys())
            
            new_areas = current_areas - previous_areas
            dropped_areas = previous_areas - current_areas
            
            new_tech = [self.tech_lexicon[t]["name"] for t in new_areas]
            dropped_tech = [self.tech_lexicon[t]["name"] for t in dropped_areas]
        
        # Determine trajectory
        if yoy_tech_change is not None:
            if yoy_tech_change > 0.2:
                trajectory = "accelerating"
            elif yoy_tech_change < -0.2:
                trajectory = "decelerating"
            else:
                trajectory = "stable"
        else:
            trajectory = "unknown"
        
        # Signal strength and expected alpha
        signal_strength = 0.0
        expected_alpha = 0.0
        tradeable = False
        
        # Positive signals
        if trajectory == "accelerating":
            signal_strength += 0.3
        if current.emerging_focus > 0.2:
            signal_strength += 0.2
        if current.moat_strength > 0.3:
            signal_strength += 0.2
        if current.rd_growth_mentioned:
            signal_strength += 0.1
        if new_tech:
            signal_strength += 0.1
        
        # Negative signals
        if trajectory == "decelerating":
            signal_strength -= 0.3
        if current.moat_strength < -0.3:
            signal_strength -= 0.2
        if dropped_tech and not new_tech:
            signal_strength -= 0.1
        
        # Expected alpha based on signal
        if signal_strength > 0.4:
            expected_alpha = 0.02  # 2% alpha
            tradeable = True
        elif signal_strength < -0.3:
            expected_alpha = -0.015
            tradeable = True
        
        # Get key quotes (high context mentions)
        key_quotes = []
        for m in current.mentions[:5]:
            if len(m.context) > 50:
                key_quotes.append(f"...{m.context}...")
        
        # Build result
        result = FilingTechSignalResult(
            ticker=current.ticker,
            analysis_date=datetime.now(),
            current_profile=current,
            yoy_tech_change=yoy_tech_change,
            yoy_emerging_change=yoy_emerging_change,
            top_tech_areas=top_areas,
            new_tech_mentions=new_tech,
            dropped_tech=dropped_tech,
            innovation_trajectory=trajectory,
            signal_strength=signal_strength,
            tradeable=tradeable,
            expected_alpha=expected_alpha,
            key_quotes=key_quotes,
        )
        
        # Caveats
        if len(current.mentions) < 10:
            result.caveats.append("Low tech mention count - may not be tech-focused")
        if not previous:
            result.caveats.append("No prior filing for comparison")
        
        return result
    
    def get_summary(self, result: FilingTechSignalResult) -> str:
        """Generate human-readable summary."""
        lines = [f"Filing Tech Analysis: {result.ticker}"]
        lines.append("=" * 50)
        
        if result.current_profile:
            p = result.current_profile
            lines.append(f"Filing: {p.filing_type} ({p.filing_date.strftime('%Y-%m-%d')})")
            lines.append(f"Tech Intensity: {p.tech_intensity:.1f} mentions/10K chars")
            lines.append(f"Emerging Focus: {p.emerging_focus:.1%}")
            lines.append(f"Moat Language: {p.moat_strength:+.2f}")
            lines.append(f"R&D Mentions: {p.rd_mentions}")
        
        lines.append("")
        lines.append(f"Innovation Trajectory: {result.innovation_trajectory.upper()}")
        
        if result.yoy_tech_change is not None:
            lines.append(f"YoY Tech Change: {result.yoy_tech_change:+.1%}")
        
        if result.top_tech_areas:
            lines.append(f"\nTop Tech Areas: {', '.join(result.top_tech_areas[:3])}")
        
        if result.new_tech_mentions:
            lines.append(f"🚀 NEW Tech Focus: {', '.join(result.new_tech_mentions)}")
        
        if result.dropped_tech:
            lines.append(f"📉 Dropped: {', '.join(result.dropped_tech)}")
        
        lines.append("")
        lines.append(f"Signal Strength: {result.signal_strength:+.2f}")
        lines.append(f"Tradeable: {'YES' if result.tradeable else 'No'}")
        if result.tradeable:
            lines.append(f"Expected Alpha: {result.expected_alpha:+.1%}")
        
        if result.caveats:
            lines.append("")
            for caveat in result.caveats:
                lines.append(f"⚠️  {caveat}")
        
        return "\n".join(lines)


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    # Sample 10-K excerpt (would normally come from SEC EDGAR)
    sample_10k = """
    NVIDIA Corporation designs and manufactures graphics processing units (GPUs) 
    and related software. Our artificial intelligence and machine learning 
    platforms power data centers, autonomous vehicles, and gaming systems.
    
    We continue to invest heavily in research and development, with R&D expenses 
    increasing 25% year over year as we expand our deep learning and neural network 
    capabilities. Our CUDA platform and proprietary technology create significant 
    barriers to entry in the accelerator market.
    
    Our generative AI solutions, including large language model inference 
    optimization, represent the fastest-growing segment of our business. We have 
    established ourselves as the industry leader in AI computing infrastructure.
    
    We are also exploring quantum computing applications and have increased our 
    investment in autonomous vehicle systems and robotics platforms.
    
    Competition in the semiconductor industry remains intense, with many competitors 
    seeking to develop alternative accelerator solutions.
    """
    
    model = FilingTechSignal()
    profile = model.analyze_text(
        sample_10k,
        ticker="NVDA",
        filing_date=datetime.now(),
    )
    
    result = model.compare_filings(profile, None)
    print(model.get_summary(result))
