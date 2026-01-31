"""
Thematic Analysis Agent.

Provides forward-looking analysis including:
- Sector rotation signals (which sectors are gaining momentum)
- Economic moat scoring (sustainable competitive advantages)
- Theme exposure (AI, energy transition, cybersecurity, etc.)
- Macro environment assessment

================================================================================
SECTOR MOMENTUM MODEL - STRENGTHS & WEAKNESSES
================================================================================

ACADEMIC BASIS:
    Moskowitz & Grinblatt (1999): Sector momentum explains ~50% of stock momentum
    Hong et al. (2007): Information diffuses slowly across industries
    
STRENGTHS:
    ✅ Sector effects explain large portion of stock returns
    ✅ Clear, objective ranking (1-11 sectors)
    ✅ Multiple timeframes (1M, 3M, 6M) reduce noise
    ✅ Easy to implement - just ETF prices
    ✅ Identifies macro regime (risk-on, risk-off, rotating)
    ✅ Works well for sector allocation decisions
    
WEAKNESSES:
    ⚠️ MOMENTUM CAN REVERSE - what goes up can crash
    ⚠️ Sector classification is imperfect (AMZN = retail or tech?)
    ⚠️ Small sample (only 11 sectors)
    ⚠️ Doesn't capture intra-sector dispersion
    ⚠️ May lag - momentum persists until it doesn't
    ⚠️ Sector rotation is slow - signal changes infrequently
    
BEST USE CASES:
    • Sector allocation and rotation strategies
    • Identifying headwinds/tailwinds for stock picks
    • Understanding market regime (risk-on vs defensive)
    
CAUTIONS:
    • Top sector may be due for mean reversion
    • Bottom sector may be contrarian opportunity
    • Individual stock factors matter more than sector
    • Use as ONE input, not sole decision driver

Usage:
    from auto_researcher.agents.thematic_agent import ThematicAnalysisAgent
    
    agent = ThematicAnalysisAgent()
    result = agent.analyze_ticker("NVDA")
    print(f"Moat: {result.moat_rating} | Themes: {result.theme_exposures}")
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal

import yfinance as yf
import pandas as pd
import numpy as np

from ..models.sector_momentum import SectorMomentumModel, StockSectorSignal
from ..models.early_adopter import EarlyAdopterModel, EarlyAdopterSignal

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTOR DEFINITIONS
# ==============================================================================

# Sector ETFs for rotation analysis
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Ticker to sector mapping (major stocks)
TICKER_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary", "META": "Communication Services",
    "NVDA": "Technology", "TSLA": "Consumer Discretionary", "AMD": "Technology",
    "INTC": "Technology", "CRM": "Technology", "ORCL": "Technology",
    "ADBE": "Technology", "NFLX": "Communication Services", "PYPL": "Financials",
    "V": "Financials", "MA": "Financials", "JPM": "Financials",
    "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "CAT": "Industrials", "BA": "Industrials", "UPS": "Industrials",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
}

# Investment themes and associated keywords/tickers
INVESTMENT_THEMES = {
    "artificial_intelligence": {
        "name": "Artificial Intelligence",
        "keywords": ["AI", "machine learning", "neural network", "LLM", "GPT", 
                    "generative AI", "deep learning", "NLP", "computer vision"],
        "pure_plays": ["NVDA", "AMD", "GOOGL", "MSFT", "META", "PLTR", "AI", "PATH"],
        "etfs": ["BOTZ", "ROBO", "AIQ"],
    },
    "energy_transition": {
        "name": "Energy Transition / Clean Energy",
        "keywords": ["renewable", "solar", "wind", "EV", "electric vehicle", 
                    "battery", "hydrogen", "carbon neutral", "sustainability"],
        "pure_plays": ["TSLA", "ENPH", "SEDG", "FSLR", "NEE", "PLUG"],
        "etfs": ["ICLN", "TAN", "QCLN"],
    },
    "cybersecurity": {
        "name": "Cybersecurity",
        "keywords": ["cybersecurity", "data breach", "ransomware", "zero trust",
                    "endpoint security", "cloud security", "identity management"],
        "pure_plays": ["CRWD", "PANW", "ZS", "FTNT", "S", "OKTA"],
        "etfs": ["HACK", "CIBR", "BUG"],
    },
    "cloud_computing": {
        "name": "Cloud Computing",
        "keywords": ["cloud", "SaaS", "PaaS", "IaaS", "AWS", "Azure", "GCP",
                    "cloud migration", "hybrid cloud", "multi-cloud"],
        "pure_plays": ["AMZN", "MSFT", "GOOGL", "CRM", "NOW", "SNOW", "MDB"],
        "etfs": ["SKYY", "CLOU", "WCLD"],
    },
    "digital_payments": {
        "name": "Digital Payments / Fintech",
        "keywords": ["digital payments", "fintech", "BNPL", "cryptocurrency",
                    "mobile payments", "blockchain", "neobank"],
        "pure_plays": ["V", "MA", "PYPL", "SQ", "COIN", "AFRM"],
        "etfs": ["IPAY", "FINX", "ARKF"],
    },
    "healthcare_innovation": {
        "name": "Healthcare Innovation",
        "keywords": ["biotech", "gene therapy", "CRISPR", "mRNA", "precision medicine",
                    "telehealth", "digital health", "medical devices"],
        "pure_plays": ["MRNA", "REGN", "VRTX", "ISRG", "DXCM", "TDOC"],
        "etfs": ["XBI", "IBB", "ARKG"],
    },
    "semiconductor": {
        "name": "Semiconductors",
        "keywords": ["semiconductor", "chip", "fab", "foundry", "GPU", "CPU",
                    "memory", "DRAM", "NAND", "EUV", "lithography"],
        "pure_plays": ["NVDA", "AMD", "INTC", "TSM", "ASML", "MU", "QCOM"],
        "etfs": ["SMH", "SOXX", "XSD"],
    },
    "defense_aerospace": {
        "name": "Defense & Aerospace",
        "keywords": ["defense", "military", "aerospace", "space", "satellite",
                    "missile", "fighter jet", "government contract"],
        "pure_plays": ["LMT", "RTX", "NOC", "GD", "BA", "LHX"],
        "etfs": ["ITA", "XAR", "PPA"],
    },
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SectorRotation:
    """Sector rotation analysis."""
    sector: str
    relative_strength_1m: float  # vs SPY
    relative_strength_3m: float
    momentum_score: float  # -1 to +1
    rotation_signal: Literal["overweight", "neutral", "underweight"]
    rank: int  # 1 = strongest


@dataclass 
class ThemeExposure:
    """A company's exposure to an investment theme."""
    theme_id: str
    theme_name: str
    exposure_score: float  # 0 to 1
    is_pure_play: bool
    rationale: str


@dataclass
class MoatAssessment:
    """Economic moat analysis."""
    moat_rating: Literal["wide", "narrow", "none"]
    moat_score: float  # 0 to 1
    moat_sources: list[str]
    moat_stability: Literal["stable", "positive", "negative"]
    competitive_threats: list[str]
    
    # Supporting metrics
    gross_margin_avg: Optional[float] = None
    gross_margin_stability: Optional[float] = None  # std dev
    roic_avg: Optional[float] = None
    market_position: str = ""


@dataclass
class ThematicAnalysis:
    """Complete thematic analysis for a ticker."""
    ticker: str
    analysis_date: datetime
    
    # Sector analysis
    sector: str
    sector_rotation: Optional[SectorRotation] = None
    sector_outlook: str = ""
    
    # Sector Momentum Model signals (backtested)
    sm_sector_rank: Optional[int] = None           # 1-11 (1 = strongest)
    sm_rs_1m: Optional[float] = None               # 1-month relative strength vs SPY
    sm_rs_3m: Optional[float] = None               # 3-month relative strength vs SPY
    sm_rotation_signal: Optional[str] = None       # "overweight", "neutral", "underweight"
    sm_sector_boost: Optional[float] = None        # -0.25 to +0.30 impact
    sm_direction: Optional[str] = None             # "tailwind" or "headwind"
    sm_strength: Optional[str] = None              # "strong", "moderate", "weak", "none"
    sm_is_actionable: bool = False
    sm_expected_alpha: Optional[float] = None      # Expected annual alpha from sector
    sm_market_regime: Optional[str] = None         # "risk_on", "risk_off", "neutral", "rotating"
    sm_summary: Optional[str] = None               # Human-readable signal
    
    # Early Adopter Model signals (detects tech pioneers)
    ea_pioneer_score: Optional[float] = None       # 0-1 pioneer score
    ea_techs_adopted: Optional[int] = None         # Total emerging techs discussed
    ea_techs_early: Optional[int] = None           # Techs adopted BEFORE industry
    ea_avg_lead_months: Optional[float] = None     # Avg months ahead of peers
    ea_genai_score: Optional[float] = None         # GenAI-specific lead score
    ea_signal: Optional[str] = None                # "strong_buy", "buy", "neutral", "avoid"
    ea_top_adoptions: Optional[list] = None        # Top early tech adoptions
    ea_summary: Optional[str] = None               # Human-readable signal
    
    # Moat analysis
    moat: Optional[MoatAssessment] = None
    
    # Theme exposures
    theme_exposures: list[ThemeExposure] = field(default_factory=list)
    primary_themes: list[str] = field(default_factory=list)
    
    # Forward-looking signals
    forward_score: float = 0.0  # -1 to +1
    forward_signal: Literal["strong_tailwinds", "tailwinds", "neutral", "headwinds", "strong_headwinds"] = "neutral"
    
    # Summary
    summary: str = ""
    key_catalysts: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)


# ==============================================================================
# THEMATIC ANALYSIS AGENT
# ==============================================================================

class ThematicAnalysisAgent:
    """
    Analyzes forward-looking thematic factors for stocks.
    
    Provides:
    - Sector rotation signals
    - Economic moat assessment
    - Theme/trend exposure
    - Macro environment fit
    - Early adopter detection (tech pioneers)
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        use_sector_momentum_model: bool = True,
        use_early_adopter_model: bool = True,
    ):
        self.model = model
        self._sector_data_cache = None
        self._sector_cache_time = None
        
        # Backtested sector momentum model
        self.use_sector_momentum_model = use_sector_momentum_model
        self._sector_momentum_model: Optional[SectorMomentumModel] = None
        if use_sector_momentum_model:
            try:
                self._sector_momentum_model = SectorMomentumModel()
                logger.info("SectorMomentumModel initialized for ThematicAnalysisAgent")
            except Exception as e:
                logger.warning(f"Failed to initialize SectorMomentumModel: {e}")
        
        # Early adopter model (detects companies discussing tech before peers)
        self.use_early_adopter_model = use_early_adopter_model
        self._early_adopter_model: Optional[EarlyAdopterModel] = None
        if use_early_adopter_model:
            try:
                self._early_adopter_model = EarlyAdopterModel()
                logger.info("EarlyAdopterModel initialized for ThematicAnalysisAgent")
            except Exception as e:
                logger.warning(f"Failed to initialize EarlyAdopterModel: {e}")
    
    @property
    def name(self) -> str:
        return "ThematicAnalysisAgent"
    
    def _get_sector_rotation_data(self) -> dict[str, SectorRotation]:
        """Calculate sector rotation signals from ETF data."""
        # Cache for 1 hour
        if (self._sector_data_cache and self._sector_cache_time and 
            datetime.now() - self._sector_cache_time < timedelta(hours=1)):
            return self._sector_data_cache
        
        try:
            # Download sector ETF data
            tickers = list(SECTOR_ETFS.keys()) + ["SPY"]
            data = yf.download(tickers, period="6mo", progress=False)["Close"]
            
            if data.empty:
                return {}
            
            # Calculate returns
            returns_1m = data.pct_change(21).iloc[-1]  # ~1 month
            returns_3m = data.pct_change(63).iloc[-1]  # ~3 months
            
            spy_1m = returns_1m.get("SPY", 0)
            spy_3m = returns_3m.get("SPY", 0)
            
            rotations = {}
            for etf, sector in SECTOR_ETFS.items():
                if etf not in returns_1m.index:
                    continue
                    
                rs_1m = returns_1m[etf] - spy_1m
                rs_3m = returns_3m[etf] - spy_3m
                
                # Momentum score: weighted average of relative strength
                momentum = 0.4 * rs_1m + 0.6 * rs_3m
                
                # Signal based on momentum
                if momentum > 0.03:
                    signal = "overweight"
                elif momentum < -0.03:
                    signal = "underweight"
                else:
                    signal = "neutral"
                
                rotations[sector] = SectorRotation(
                    sector=sector,
                    relative_strength_1m=rs_1m,
                    relative_strength_3m=rs_3m,
                    momentum_score=momentum,
                    rotation_signal=signal,
                    rank=0,  # Will be set below
                )
            
            # Set ranks
            sorted_sectors = sorted(rotations.values(), 
                                   key=lambda x: x.momentum_score, reverse=True)
            for i, sr in enumerate(sorted_sectors):
                sr.rank = i + 1
            
            self._sector_data_cache = rotations
            self._sector_cache_time = datetime.now()
            
            return rotations
            
        except Exception as e:
            logger.error(f"Failed to get sector rotation data: {e}")
            return {}
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """Get sector for a ticker."""
        if ticker in TICKER_SECTORS:
            return TICKER_SECTORS[ticker]
        
        # Try yfinance
        try:
            info = yf.Ticker(ticker).info
            return info.get("sector", "Unknown")
        except:
            return "Unknown"
    
    def _add_sector_momentum_signals(self, result: ThematicAnalysis) -> None:
        """
        Add backtested sector momentum signals to the thematic analysis.
        
        Based on Moskowitz & Grinblatt (1999) showing sector momentum explains
        ~50% of individual stock momentum. Uses 11 SPDR sector ETFs with
        momentum = 0.3×RS_1M + 0.5×RS_3M + 0.2×RS_6M.
        
        Args:
            result: ThematicAnalysis to enhance with sector momentum signals.
        """
        if not self._sector_momentum_model:
            return
        
        try:
            # Get stock-level sector momentum signal
            signal = self._sector_momentum_model.get_stock_signal(result.ticker)
            
            if signal and signal.sector_signal:
                sector_sig = signal.sector_signal
                
                result.sm_sector_rank = sector_sig.rank
                result.sm_rs_1m = sector_sig.rs_1m
                result.sm_rs_3m = sector_sig.rs_3m
                result.sm_rotation_signal = sector_sig.rotation_signal
                result.sm_sector_boost = signal.sector_boost
                result.sm_direction = signal.direction
                result.sm_strength = signal.strength
                result.sm_is_actionable = signal.is_actionable
                result.sm_expected_alpha = signal.expected_alpha
                
                # Get market regime from snapshot
                snapshot = self._sector_momentum_model.get_rotation_snapshot()
                if snapshot:
                    result.sm_market_regime = snapshot.regime
                
                # Generate summary
                if signal.is_actionable:
                    boost_pct = signal.sector_boost * 100 if signal.sector_boost else 0
                    result.sm_summary = (
                        f"SECTOR MOMENTUM: {signal.direction.upper()} - {result.sector} "
                        f"ranks #{sector_sig.rank}/11 with {boost_pct:+.0f}% sector impact. "
                        f"Market regime: {result.sm_market_regime or 'unknown'}. "
                        f"Expected alpha: {signal.expected_alpha:.1%}." if signal.expected_alpha else
                        f"SECTOR MOMENTUM: {signal.direction.upper()} - {result.sector} "
                        f"ranks #{sector_sig.rank}/11 with {boost_pct:+.0f}% sector impact."
                    )
                else:
                    result.sm_summary = (
                        f"SECTOR MOMENTUM: Neutral - {result.sector} ranks "
                        f"#{sector_sig.rank}/11 (mid-pack). No actionable signal."
                    )
                
                logger.info(
                    f"{result.ticker}: Sector momentum {signal.direction or 'neutral'} "
                    f"(rank #{sector_sig.rank}, boost {signal.sector_boost:+.1%})"
                )
            else:
                result.sm_summary = f"SECTOR MOMENTUM: Unable to classify sector for {result.ticker}"
                
        except Exception as e:
            logger.warning(f"Failed to get sector momentum signal for {result.ticker}: {e}")
            result.sm_summary = f"SECTOR MOMENTUM: Error - {str(e)}"
    
    def _add_early_adopter_signals(self, result: ThematicAnalysis) -> None:
        """Add early adopter technology signals to the analysis result."""
        if not self._early_adopter_model:
            return
            
        try:
            # Get early adopter signal for this company
            signal = self._early_adopter_model.analyze_company(result.ticker)
            
            if signal is None:
                result.ea_summary = f"EARLY ADOPTER: No transcript data available for {result.ticker}"
                return
            
            # Populate all early adopter fields (using correct EarlyAdopterSignal field names)
            result.ea_pioneer_score = signal.pioneer_score
            result.ea_techs_adopted = signal.total_techs_adopted
            result.ea_techs_early = signal.techs_adopted_early
            result.ea_avg_lead_months = signal.avg_lead_months
            result.ea_genai_score = signal.genai_score
            result.ea_signal = signal.signal
            
            # Format top adoptions for display
            if signal.earliest_adoptions:
                top_3 = signal.earliest_adoptions[:3]
                result.ea_top_adoptions = [
                    f"{a.tech_name}: {a.lead_months:+.0f}mo" 
                    for a in top_3 if a.lead_months is not None
                ]
            
            # Generate human-readable summary based on signal from model
            # Model signals: "strong_buy", "buy", "neutral", "avoid"
            if signal.signal == "strong_buy":
                result.ea_summary = (
                    f"EARLY ADOPTER: STRONG PIONEER (score={signal.pioneer_score:.2f}) - "
                    f"{result.ticker} adopted {signal.total_techs_adopted} emerging techs, "
                    f"{signal.techs_adopted_early} ahead of industry "
                    f"(avg {signal.avg_lead_months:+.1f}mo lead). "
                )
                if signal.genai_score and signal.genai_score > 0:
                    result.ea_summary += f"GenAI pioneer: +{signal.genai_score:.0f}mo ahead."
            elif signal.signal == "buy":
                result.ea_summary = (
                    f"EARLY ADOPTER: PIONEER (score={signal.pioneer_score:.2f}) - "
                    f"{result.ticker} shows early adoption of {signal.total_techs_adopted} techs, "
                    f"{signal.techs_adopted_early} ahead of peers."
                )
            elif signal.signal == "neutral":
                result.ea_summary = (
                    f"EARLY ADOPTER: Fast Follower (score={signal.pioneer_score:.2f}) - "
                    f"{result.ticker} adopts tech around industry average "
                    f"({signal.total_techs_adopted} techs, {signal.avg_lead_months:+.1f}mo vs industry)."
                )
            else:  # "avoid"
                result.ea_summary = (
                    f"EARLY ADOPTER: Laggard (score={signal.pioneer_score:.2f}) - "
                    f"{result.ticker} shows limited emerging tech adoption "
                    f"({signal.total_techs_adopted} techs)."
                )
            
            logger.info(
                f"{result.ticker}: Early adopter {signal.signal} "
                f"(pioneer={signal.pioneer_score:.2f}, techs={signal.total_techs_adopted})"
            )
            
            # Adjust forward score based on pioneer status
            # Based on backtest: high pioneers +49% vs low +23.7% = +25.3% spread
            # Pioneer score explains ~12.5% incremental alpha (r=0.36)
            if signal.pioneer_score > 0:
                # Scale: pioneer_score 0.5+ = strong alpha potential
                ea_boost = (signal.pioneer_score - 0.3) * 0.3  # Max ~0.21 boost
                result.forward_score = max(-1, min(1, result.forward_score + ea_boost))
                
                # Update signal if boosted significantly
                if result.forward_score >= 0.3:
                    result.forward_signal = "strong_tailwinds"
                elif result.forward_score >= 0.1:
                    result.forward_signal = "tailwinds"
            
        except Exception as e:
            logger.warning(f"Failed to get early adopter signal for {result.ticker}: {e}")
            result.ea_summary = f"EARLY ADOPTER: Error - {str(e)}"
    
    def _calculate_moat(self, ticker: str) -> Optional[MoatAssessment]:
        """Assess economic moat from financial metrics."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get historical financials for stability analysis
            financials = stock.quarterly_financials
            
            # Moat indicators
            moat_sources = []
            moat_score = 0.0
            threats = []
            
            # 1. Gross Margin (pricing power)
            gross_margin = info.get("grossMargins", 0)
            if gross_margin:
                if gross_margin > 0.6:
                    moat_score += 0.2
                    moat_sources.append("high gross margins (pricing power)")
                elif gross_margin > 0.4:
                    moat_score += 0.1
                    moat_sources.append("solid gross margins")
            
            # 2. Return on Equity (capital efficiency)
            roe = info.get("returnOnEquity", 0)
            if roe:
                if roe > 0.25:
                    moat_score += 0.2
                    moat_sources.append("exceptional return on equity")
                elif roe > 0.15:
                    moat_score += 0.1
                    moat_sources.append("strong return on equity")
            
            # 3. Market Cap (scale advantages)
            market_cap = info.get("marketCap", 0)
            if market_cap:
                if market_cap > 500e9:  # >$500B
                    moat_score += 0.15
                    moat_sources.append("massive scale advantages")
                elif market_cap > 100e9:  # >$100B
                    moat_score += 0.1
                    moat_sources.append("significant scale")
            
            # 4. Profit Margin (overall efficiency)
            profit_margin = info.get("profitMargins", 0)
            if profit_margin:
                if profit_margin > 0.25:
                    moat_score += 0.15
                    moat_sources.append("excellent profit margins")
                elif profit_margin > 0.15:
                    moat_score += 0.08
            
            # 5. Revenue Growth (demand durability)
            revenue_growth = info.get("revenueGrowth", 0)
            if revenue_growth:
                if revenue_growth > 0.2:
                    moat_score += 0.1
                    moat_sources.append("strong demand growth")
                elif revenue_growth < 0:
                    moat_score -= 0.1
                    threats.append("declining revenue")
            
            # 6. Operating Margin (operational efficiency)
            op_margin = info.get("operatingMargins", 0)
            if op_margin:
                if op_margin > 0.3:
                    moat_score += 0.1
                    moat_sources.append("operational excellence")
            
            # 7. Debt levels (financial strength)
            debt_to_equity = info.get("debtToEquity", 0)
            if debt_to_equity:
                if debt_to_equity > 200:
                    moat_score -= 0.1
                    threats.append("high debt burden")
                elif debt_to_equity < 50:
                    moat_score += 0.1
                    moat_sources.append("strong balance sheet")
            
            # Determine moat rating
            moat_score = max(0, min(1, moat_score))
            if moat_score >= 0.6:
                rating = "wide"
            elif moat_score >= 0.35:
                rating = "narrow"
            else:
                rating = "none"
            
            # Stability assessment (simplified)
            stability = "stable"
            if revenue_growth and revenue_growth > 0.15:
                stability = "positive"
            elif revenue_growth and revenue_growth < -0.05:
                stability = "negative"
            
            # Market position
            position = ""
            if market_cap and market_cap > 200e9:
                position = "market leader"
            elif market_cap and market_cap > 50e9:
                position = "major player"
            else:
                position = "competitor"
            
            return MoatAssessment(
                moat_rating=rating,
                moat_score=moat_score,
                moat_sources=moat_sources,
                moat_stability=stability,
                competitive_threats=threats,
                gross_margin_avg=gross_margin,
                roic_avg=roe,
                market_position=position,
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate moat for {ticker}: {e}")
            return None
    
    def _analyze_theme_exposure(self, ticker: str, info: dict) -> list[ThemeExposure]:
        """Analyze exposure to investment themes."""
        exposures = []
        
        # Get business description for keyword matching
        description = (info.get("longBusinessSummary", "") or "").lower()
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        
        for theme_id, theme_data in INVESTMENT_THEMES.items():
            exposure_score = 0.0
            is_pure_play = ticker in theme_data.get("pure_plays", [])
            
            # Pure play bonus
            if is_pure_play:
                exposure_score = 0.8
            
            # Keyword matching in description
            keywords = theme_data.get("keywords", [])
            keyword_matches = sum(1 for kw in keywords if kw.lower() in description)
            keyword_score = min(0.5, keyword_matches * 0.1)
            exposure_score = max(exposure_score, keyword_score)
            
            # Sector/industry matching
            theme_name = theme_data["name"].lower()
            if sector.lower() in theme_name or theme_name in sector.lower():
                exposure_score = max(exposure_score, 0.4)
            if industry.lower() in theme_name or theme_name in industry.lower():
                exposure_score = max(exposure_score, 0.5)
            
            # Only include if meaningful exposure
            if exposure_score >= 0.3:
                rationale = []
                if is_pure_play:
                    rationale.append("pure play")
                if keyword_matches > 0:
                    rationale.append(f"{keyword_matches} keyword matches")
                    
                exposures.append(ThemeExposure(
                    theme_id=theme_id,
                    theme_name=theme_data["name"],
                    exposure_score=exposure_score,
                    is_pure_play=is_pure_play,
                    rationale=", ".join(rationale) if rationale else "sector alignment",
                ))
        
        # Sort by exposure score
        exposures.sort(key=lambda x: x.exposure_score, reverse=True)
        
        return exposures
    
    def _generate_forward_score(
        self,
        sector_rotation: Optional[SectorRotation],
        moat: Optional[MoatAssessment],
        theme_exposures: list[ThemeExposure],
    ) -> tuple[float, str]:
        """
        Generate forward-looking score combining all factors.
        
        Returns:
            (score, signal) where score is -1 to +1
        """
        score = 0.0
        
        # Sector rotation contribution (30% weight)
        if sector_rotation:
            sector_score = sector_rotation.momentum_score * 2  # Scale to ~-0.3 to +0.3
            score += sector_score * 0.3
        
        # Moat contribution (30% weight)
        if moat:
            # Moat score is 0-1, convert to -0.3 to +0.3
            moat_contrib = (moat.moat_score - 0.3) * 0.6
            score += moat_contrib * 0.3
        
        # Theme exposure contribution (40% weight)
        if theme_exposures:
            # Hot themes get bonus
            hot_themes = ["artificial_intelligence", "cybersecurity", "energy_transition"]
            theme_score = 0.0
            
            for exp in theme_exposures[:3]:  # Top 3 themes
                base = exp.exposure_score * 0.3
                if exp.theme_id in hot_themes:
                    base *= 1.5  # Bonus for hot themes
                theme_score += base
            
            theme_score = min(0.4, theme_score)
            score += theme_score
        
        # Normalize to -1 to +1
        score = max(-1, min(1, score))
        
        # Determine signal
        if score >= 0.3:
            signal = "strong_tailwinds"
        elif score >= 0.1:
            signal = "tailwinds"
        elif score > -0.1:
            signal = "neutral"
        elif score > -0.3:
            signal = "headwinds"
        else:
            signal = "strong_headwinds"
        
        return score, signal
    
    def _generate_summary(
        self,
        ticker: str,
        sector: str,
        sector_rotation: Optional[SectorRotation],
        moat: Optional[MoatAssessment],
        theme_exposures: list[ThemeExposure],
        forward_signal: str,
    ) -> tuple[str, list[str], list[str]]:
        """Generate summary and extract catalysts/risks."""
        parts = []
        catalysts = []
        risks = []
        
        # Sector outlook
        if sector_rotation:
            if sector_rotation.rotation_signal == "overweight":
                parts.append(f"{sector} sector showing strong momentum (rank #{sector_rotation.rank}/11).")
                catalysts.append(f"Sector tailwinds: {sector} outperforming market")
            elif sector_rotation.rotation_signal == "underweight":
                parts.append(f"{sector} sector underperforming (rank #{sector_rotation.rank}/11).")
                risks.append(f"Sector headwinds: {sector} lagging market")
            else:
                parts.append(f"{sector} sector performing in-line with market.")
        
        # Moat assessment
        if moat:
            if moat.moat_rating == "wide":
                parts.append(f"Company has a wide economic moat from {', '.join(moat.moat_sources[:2])}.")
                catalysts.append("Strong competitive position with wide moat")
            elif moat.moat_rating == "narrow":
                parts.append(f"Company has a narrow moat based on {', '.join(moat.moat_sources[:2])}.")
            else:
                parts.append("Limited competitive advantages identified.")
                risks.append("No clear economic moat")
            
            for threat in moat.competitive_threats:
                risks.append(threat)
        
        # Theme exposures
        if theme_exposures:
            top_themes = [e.theme_name for e in theme_exposures[:2]]
            pure_plays = [e.theme_name for e in theme_exposures if e.is_pure_play]
            
            if pure_plays:
                parts.append(f"Pure play on {', '.join(pure_plays)}.")
                for theme in pure_plays:
                    catalysts.append(f"Pure play exposure to {theme}")
            elif top_themes:
                parts.append(f"Exposure to {', '.join(top_themes)} themes.")
        
        # Forward outlook
        outlook_text = {
            "strong_tailwinds": "Strong forward-looking tailwinds from sector momentum and thematic exposure.",
            "tailwinds": "Moderate tailwinds from favorable positioning.",
            "neutral": "Neutral forward outlook.",
            "headwinds": "Some headwinds from sector rotation or competitive position.",
            "strong_headwinds": "Significant headwinds from unfavorable positioning.",
        }
        parts.append(outlook_text.get(forward_signal, ""))
        
        summary = " ".join(parts)
        
        return summary, catalysts[:5], risks[:5]
    
    def analyze_ticker(self, ticker: str) -> ThematicAnalysis:
        """
        Perform complete thematic analysis for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            ThematicAnalysis with all forward-looking signals.
        """
        logger.info(f"Running thematic analysis for {ticker}")
        
        # Get basic info
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
        except Exception as e:
            logger.error(f"Failed to get info for {ticker}: {e}")
            info = {}
        
        # Get sector
        sector = self._get_ticker_sector(ticker)
        
        # Sector rotation analysis
        sector_rotations = self._get_sector_rotation_data()
        sector_rotation = sector_rotations.get(sector)
        
        # Moat analysis
        moat = self._calculate_moat(ticker)
        
        # Theme exposure
        theme_exposures = self._analyze_theme_exposure(ticker, info)
        primary_themes = [e.theme_name for e in theme_exposures[:3]]
        
        # Forward-looking score
        forward_score, forward_signal = self._generate_forward_score(
            sector_rotation, moat, theme_exposures
        )
        
        # Generate summary
        summary, catalysts, risks = self._generate_summary(
            ticker, sector, sector_rotation, moat, theme_exposures, forward_signal
        )
        
        # Sector outlook
        sector_outlook = ""
        if sector_rotation:
            if sector_rotation.rotation_signal == "overweight":
                sector_outlook = f"Bullish - {sector} outperforming (+{sector_rotation.relative_strength_3m:.1%} vs SPY over 3m)"
            elif sector_rotation.rotation_signal == "underweight":
                sector_outlook = f"Bearish - {sector} underperforming ({sector_rotation.relative_strength_3m:.1%} vs SPY over 3m)"
            else:
                sector_outlook = f"Neutral - {sector} in-line with market"
        
        result = ThematicAnalysis(
            ticker=ticker,
            analysis_date=datetime.now(),
            sector=sector,
            sector_rotation=sector_rotation,
            sector_outlook=sector_outlook,
            moat=moat,
            theme_exposures=theme_exposures,
            primary_themes=primary_themes,
            forward_score=forward_score,
            forward_signal=forward_signal,
            summary=summary,
            key_catalysts=catalysts,
            key_risks=risks,
        )
        
        # Add backtested sector momentum signals
        if self.use_sector_momentum_model:
            self._add_sector_momentum_signals(result)
        
        # Add early adopter technology signals
        if self.use_early_adopter_model:
            self._add_early_adopter_signals(result)
        
        return result
    
    def get_sector_rankings(self) -> list[SectorRotation]:
        """Get current sector rotation rankings."""
        rotations = self._get_sector_rotation_data()
        return sorted(rotations.values(), key=lambda x: x.rank)
    
    def get_hot_themes(self) -> list[str]:
        """Get list of currently hot investment themes."""
        # This could be enhanced with news/trend analysis
        return ["artificial_intelligence", "cybersecurity", "energy_transition", "semiconductor"]


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Thematic Analysis")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--sectors", action="store_true", help="Show sector rankings")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    agent = ThematicAnalysisAgent()
    
    # Show sector rankings
    if args.sectors:
        print("\n" + "="*60)
        print("  SECTOR ROTATION RANKINGS")
        print("="*60)
        rankings = agent.get_sector_rankings()
        print(f"\n{'Rank':<6}{'Sector':<25}{'1M RS':<10}{'3M RS':<10}{'Signal':<12}")
        print("-"*60)
        for sr in rankings:
            signal_emoji = "🟢" if sr.rotation_signal == "overweight" else "🔴" if sr.rotation_signal == "underweight" else "🟡"
            print(f"{sr.rank:<6}{sr.sector:<25}{sr.relative_strength_1m:+.1%}     {sr.relative_strength_3m:+.1%}     {signal_emoji} {sr.rotation_signal}")
        print()
    
    # Analyze tickers
    for ticker in args.tickers:
        print(f"\n{'='*60}")
        print(f"  {ticker} - THEMATIC ANALYSIS")
        print("="*60)
        
        result = agent.analyze_ticker(ticker)
        
        # Forward signal
        signal_emoji = {
            "strong_tailwinds": "🟢🟢",
            "tailwinds": "🟢",
            "neutral": "🟡",
            "headwinds": "🔴",
            "strong_headwinds": "🔴🔴",
        }
        print(f"\n{signal_emoji.get(result.forward_signal, '')} Forward Signal: {result.forward_signal.upper()}")
        print(f"Score: {result.forward_score:+.2f}")
        
        # Sector
        print(f"\nSector: {result.sector}")
        if result.sector_rotation:
            sr = result.sector_rotation
            print(f"  Rotation: {sr.rotation_signal} (rank #{sr.rank}/11)")
            print(f"  3M Relative Strength: {sr.relative_strength_3m:+.1%}")
        
        # Sector Momentum Model (backtested)
        if result.sm_sector_rank is not None:
            sm_emoji = "🟢" if result.sm_direction == "tailwind" else "🔴" if result.sm_direction == "headwind" else "🟡"
            print(f"\n{sm_emoji} SECTOR MOMENTUM MODEL (Backtested):")
            print(f"  Sector Rank: #{result.sm_sector_rank}/11")
            print(f"  1M RS: {result.sm_rs_1m:+.1%} | 3M RS: {result.sm_rs_3m:+.1%}")
            print(f"  Signal: {result.sm_rotation_signal} | Direction: {result.sm_direction}")
            if result.sm_sector_boost:
                print(f"  Sector Boost: {result.sm_sector_boost:+.1%}")
            if result.sm_expected_alpha:
                print(f"  Expected Alpha: {result.sm_expected_alpha:.1%}")
            print(f"  Market Regime: {result.sm_market_regime}")
            print(f"  Actionable: {'Yes' if result.sm_is_actionable else 'No'}")
        
        # Moat
        if result.moat:
            m = result.moat
            print(f"\nMoat: {m.moat_rating.upper()} (score: {m.moat_score:.2f})")
            if m.moat_sources:
                print(f"  Sources: {', '.join(m.moat_sources[:3])}")
            print(f"  Stability: {m.moat_stability}")
        
        # Themes
        if result.theme_exposures:
            print(f"\nTheme Exposures:")
            for exp in result.theme_exposures[:4]:
                pure = "⭐" if exp.is_pure_play else ""
                print(f"  {pure}{exp.theme_name}: {exp.exposure_score:.0%} ({exp.rationale})")
        
        # Summary
        print(f"\n{result.summary}")
        
        # Catalysts & Risks
        if result.key_catalysts:
            print(f"\n🚀 Catalysts: {', '.join(result.key_catalysts[:3])}")
        if result.key_risks:
            print(f"⚠️  Risks: {', '.join(result.key_risks[:3])}")
