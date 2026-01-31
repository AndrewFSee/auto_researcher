"""
Earnings Call Transcript Analysis for Technology Signals.

Earnings calls are where management discusses:
    - New technology investments
    - R&D priorities  
    - Competitive positioning
    - Forward-looking tech strategy

Uses Financial Modeling Prep API for transcripts.

================================================================================
SIGNAL HYPOTHESIS
================================================================================

Earnings calls provide:
    1. Management's own words about technology priorities
    2. Analyst questions reveal what market is focused on
    3. Quarter-over-quarter changes in tech language
    4. Specific product/technology mentions before launches

Key insight: Changes in language matter more than absolute counts.
If a company suddenly starts mentioning "AI" 5x more, that's a signal.

================================================================================

Usage:
    from auto_researcher.models.earnings_tech_signal import EarningsCallTechModel
    
    model = EarningsCallTechModel()
    signal = model.analyze_company("NVDA")
"""

import os
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import requests

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ==============================================================================
# TECH LEXICON (tuned for earnings calls)
# ==============================================================================

# Executives use different language than 10-Ks
EARNINGS_TECH_LEXICON = {
    "ai_ml": {
        "terms": [
            "artificial intelligence", "AI", "machine learning", "ML",
            "deep learning", "neural network", "generative AI", "gen AI",
            "large language model", "LLM", "GPT", "ChatGPT", "Copilot",
            "AI infrastructure", "AI workload", "inference", "training",
            "foundation model", "transformer", "AI-driven", "AI-powered",
        ],
        "name": "AI/Machine Learning",
        "maturity": 2,
    },
    
    "cloud_data": {
        "terms": [
            "cloud", "data center", "hyperscale", "Azure", "AWS", "GCP",
            "cloud computing", "hybrid cloud", "multi-cloud",
            "cloud revenue", "cloud growth", "cloud transition",
        ],
        "name": "Cloud/Data Center",
        "maturity": 3,
    },
    
    "autonomous": {
        "terms": [
            "autonomous", "self-driving", "ADAS", "robotaxi",
            "autonomous vehicle", "AV", "level 4", "FSD",
            "full self-driving", "autopilot",
        ],
        "name": "Autonomous Systems",
        "maturity": 2,
    },
    
    "semiconductors": {
        "terms": [
            "GPU", "chip", "processor", "semiconductor", "silicon",
            "data center GPU", "accelerator", "ASIC", "TPU",
            "nanometer", "process node", "wafer", "fab",
        ],
        "name": "Semiconductors",
        "maturity": 2,
    },
    
    "software_platform": {
        "terms": [
            "platform", "SaaS", "subscription", "ARR", "recurring revenue",
            "software", "ecosystem", "developer", "API",
        ],
        "name": "Software/Platform",
        "maturity": 3,
    },
    
    "clean_tech": {
        "terms": [
            "EV", "electric vehicle", "battery", "renewable",
            "sustainability", "carbon", "green", "clean energy",
            "solar", "wind", "energy storage",
        ],
        "name": "Clean Tech",
        "maturity": 2,
    },
    
    "biotech": {
        "terms": [
            "drug", "therapy", "clinical trial", "FDA", "pipeline",
            "oncology", "immunotherapy", "gene therapy", "mRNA",
            "biologics", "therapeutic",
        ],
        "name": "Biotech/Pharma",
        "maturity": 2,
    },
    
    "quantum": {
        "terms": [
            "quantum", "qubit", "quantum computing", "quantum advantage",
        ],
        "name": "Quantum Computing",
        "maturity": 1,
    },
}

# Sentiment modifiers for tech mentions
POSITIVE_MODIFIERS = [
    "growth", "growing", "strong", "accelerating", "record",
    "exceeded", "beat", "outperformed", "momentum", "demand",
    "opportunity", "excited", "transformative", "breakthrough",
]

NEGATIVE_MODIFIERS = [
    "decline", "declining", "weak", "slowing", "challenging",
    "headwind", "pressure", "concern", "uncertainty", "delayed",
]


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TranscriptMention:
    """A technology mention in an earnings call."""
    tech_area: str
    term: str
    context: str
    speaker: str  # "management" or "analyst"
    sentiment: str = "neutral"


@dataclass
class TranscriptAnalysis:
    """Analysis of a single earnings call transcript."""
    ticker: str
    date: datetime
    quarter: str  # "Q1 2024"
    
    # Mentions
    mentions: List[TranscriptMention]
    mentions_by_area: Dict[str, int]
    
    # Metrics
    tech_intensity: float  # mentions per 1K words
    emerging_focus: float  # % emerging tech mentions
    
    # Sentiment
    tech_sentiment: float  # -1 to +1
    management_emphasis: Dict[str, int]  # What management talks about most
    analyst_focus: Dict[str, int]  # What analysts ask about


@dataclass
class EarningsCallTechSignal:
    """Technology signal from earnings call analysis."""
    ticker: str
    analysis_date: datetime
    
    # Current quarter
    current: Optional[TranscriptAnalysis]
    
    # Trend (QoQ)
    qoq_tech_change: Optional[float]
    qoq_sentiment_change: Optional[float]
    
    # Key findings
    top_tech_areas: List[str]
    emerging_mentions: List[str]  # Tech areas with maturity=1
    new_this_quarter: List[str]  # Mentioned now, not before
    
    # Signal
    tech_momentum: str  # "accelerating", "stable", "decelerating"
    signal_strength: float
    
    # Key quotes
    bullish_quotes: List[str] = field(default_factory=list)
    bearish_quotes: List[str] = field(default_factory=list)


# ==============================================================================
# DEFEATBETA TRANSCRIPT CLIENT (HuggingFace)
# ==============================================================================

# HuggingFace URL for earnings transcripts
TRANSCRIPT_URL = "https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet"

# Local cache path
import tempfile
import os as _os
TRANSCRIPT_CACHE_PATH = _os.path.join(tempfile.gettempdir(), "earnings_transcripts_cache.parquet")


class DefeatBetaTranscriptClient:
    """
    Client for earnings call transcripts from DefeatBeta/HuggingFace.
    
    Downloads the parquet file once to a local cache, then filters
    for the requested ticker to avoid repeated downloads.
    """
    
    _ticker_cache: Dict[str, List[Dict]] = {}  # Cache by ticker
    _df = None  # Full dataframe (only symbol column for filtering)
    _download_attempted = False
    
    def __init__(self):
        """Initialize client."""
        pass
    
    def _ensure_downloaded(self) -> bool:
        """Ensure the parquet file is downloaded to local cache."""
        import os
        
        # Already have it in memory
        if DefeatBetaTranscriptClient._df is not None:
            return True
        
        # Check if we have cached file
        if os.path.exists(TRANSCRIPT_CACHE_PATH):
            try:
                import pandas as pd
                logger.info(f"Loading transcripts from cache: {TRANSCRIPT_CACHE_PATH}")
                DefeatBetaTranscriptClient._df = pd.read_parquet(
                    TRANSCRIPT_CACHE_PATH,
                    columns=["symbol", "report_date", "fiscal_quarter", "fiscal_year", "transcripts"],
                )
                logger.info(f"Loaded {len(DefeatBetaTranscriptClient._df):,} transcripts from cache")
                return True
            except Exception as e:
                logger.warning(f"Cache file corrupted, re-downloading: {e}")
                os.remove(TRANSCRIPT_CACHE_PATH)
        
        # Download if not already attempted
        if DefeatBetaTranscriptClient._download_attempted:
            return False
        
        DefeatBetaTranscriptClient._download_attempted = True
        
        try:
            import requests
            import pandas as pd
            
            logger.info("Downloading earnings transcripts from HuggingFace (this may take a few minutes)...")
            
            response = requests.get(TRANSCRIPT_URL, stream=True, timeout=300, allow_redirects=True)
            response.raise_for_status()
            
            # Write to cache file
            with open(TRANSCRIPT_CACHE_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            
            file_size = os.path.getsize(TRANSCRIPT_CACHE_PATH)
            logger.info(f"Downloaded {file_size/1024/1024:.1f} MB to cache")
            
            # Load into memory
            DefeatBetaTranscriptClient._df = pd.read_parquet(
                TRANSCRIPT_CACHE_PATH,
                columns=["symbol", "report_date", "fiscal_quarter", "fiscal_year", "transcripts"],
            )
            logger.info(f"Loaded {len(DefeatBetaTranscriptClient._df):,} transcripts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download transcripts: {e}")
            return False
    
    def _load_ticker_transcripts(self, ticker: str) -> List[Dict]:
        """
        Load transcripts for a specific ticker from cached data.
        """
        ticker = ticker.upper()
        
        # Check cache
        if ticker in DefeatBetaTranscriptClient._ticker_cache:
            return DefeatBetaTranscriptClient._ticker_cache[ticker]
        
        # Ensure data is downloaded
        if not self._ensure_downloaded():
            return []
        
        try:
            import pandas as pd
            
            df = DefeatBetaTranscriptClient._df
            
            # Filter by ticker
            ticker_df = df[df["symbol"] == ticker].copy()
            
            if ticker_df.empty:
                logger.debug(f"No transcripts found for {ticker}")
                DefeatBetaTranscriptClient._ticker_cache[ticker] = []
                return []
            
            ticker_df = ticker_df.sort_values("report_date", ascending=False)
            
            logger.info(f"Found {len(ticker_df)} transcripts for {ticker}")
            
            # Convert to list of dicts
            results = []
            for _, row in ticker_df.iterrows():
                raw_transcript = row.get("transcripts", [])
                
                # Convert to single text string
                if isinstance(raw_transcript, list):
                    content_parts = []
                    for item in raw_transcript:
                        if isinstance(item, dict):
                            speaker = item.get("speaker", "")
                            text = item.get("content", "")
                            content_parts.append(f"{speaker}: {text}")
                        else:
                            content_parts.append(str(item))
                    content = "\n".join(content_parts)
                else:
                    content = str(raw_transcript)
                
                results.append({
                    "date": str(row.get("report_date", "")),
                    "quarter": int(row.get("fiscal_quarter", 0)) if pd.notna(row.get("fiscal_quarter")) else 0,
                    "year": int(row.get("fiscal_year", 0)) if pd.notna(row.get("fiscal_year")) else 0,
                    "content": content,
                })
            
            # Cache results
            DefeatBetaTranscriptClient._ticker_cache[ticker] = results
            return results
            
        except Exception as e:
            logger.error(f"Failed to filter transcripts for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_transcripts(
        self,
        ticker: str,
        limit: int = 4,  # Last 4 quarters
    ) -> List[Dict]:
        """
        Fetch earnings call transcripts for a ticker.
        
        Returns list of transcripts with: date, quarter, year, content
        """
        results = self._load_ticker_transcripts(ticker)
        return results[:limit]
    
    def get_transcript_by_quarter(
        self,
        ticker: str,
        year: int,
        quarter: int,
    ) -> Optional[Dict]:
        """Fetch a specific quarter's transcript."""
        results = self._load_ticker_transcripts(ticker)
        
        for r in results:
            if r["year"] == year and r["quarter"] == quarter:
                return r
        
        return None


# ==============================================================================
# EARNINGS CALL TECH MODEL
# ==============================================================================

class EarningsCallTechModel:
    """
    Analyzes earnings call transcripts for technology signals.
    
    Uses DefeatBeta/HuggingFace dataset with 223K+ transcripts.
    
    Key signals:
        1. Tech mention frequency (and changes over time)
        2. Management vs analyst focus differences
        3. Sentiment around tech mentions
        4. New technology areas appearing
    """
    
    def __init__(self):
        """Initialize the model."""
        self.client = DefeatBetaTranscriptClient()
        self.tech_lexicon = EARNINGS_TECH_LEXICON
        
        # Compile patterns
        self._compile_patterns()
        
        logger.info("EarningsCallTechModel initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.tech_patterns = {}
        for tech_id, config in self.tech_lexicon.items():
            # Create word boundary pattern
            terms = config["terms"]
            pattern = r'\b(' + '|'.join(re.escape(t) for t in terms) + r')\b'
            self.tech_patterns[tech_id] = re.compile(pattern, re.IGNORECASE)
        
        # Sentiment patterns
        self.positive_pattern = re.compile(
            r'\b(' + '|'.join(POSITIVE_MODIFIERS) + r')\b', re.IGNORECASE
        )
        self.negative_pattern = re.compile(
            r'\b(' + '|'.join(NEGATIVE_MODIFIERS) + r')\b', re.IGNORECASE
        )
    
    def _identify_speaker(self, context: str) -> str:
        """Try to identify if management or analyst is speaking."""
        # Common analyst question patterns
        analyst_patterns = [
            r'\?',  # Questions
            r'can you (talk|discuss|comment|elaborate)',
            r'what (is|are) your (thoughts|view|outlook)',
            r'how should we think about',
        ]
        
        for pattern in analyst_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return "analyst"
        
        return "management"
    
    def _analyze_sentiment(self, context: str) -> str:
        """Analyze sentiment of context around a tech mention."""
        pos = len(self.positive_pattern.findall(context))
        neg = len(self.negative_pattern.findall(context))
        
        if pos > neg + 1:
            return "positive"
        elif neg > pos + 1:
            return "negative"
        return "neutral"
    
    def analyze_transcript(
        self,
        text: str,
        ticker: str,
        date: datetime,
        quarter: str,
    ) -> TranscriptAnalysis:
        """
        Analyze a single transcript.
        
        Args:
            text: Full transcript text
            ticker: Stock ticker
            date: Earnings call date
            quarter: e.g., "Q1 2024"
            
        Returns:
            TranscriptAnalysis
        """
        mentions = []
        mentions_by_area = defaultdict(int)
        management_emphasis = defaultdict(int)
        analyst_focus = defaultdict(int)
        
        # Find all tech mentions
        for tech_id, pattern in self.tech_patterns.items():
            for match in pattern.finditer(text):
                term = match.group()
                
                # Get context
                start = max(0, match.start() - 150)
                end = min(len(text), match.end() + 150)
                context = text[start:end]
                
                speaker = self._identify_speaker(context)
                sentiment = self._analyze_sentiment(context)
                
                mentions.append(TranscriptMention(
                    tech_area=tech_id,
                    term=term,
                    context=context,
                    speaker=speaker,
                    sentiment=sentiment,
                ))
                
                mentions_by_area[tech_id] += 1
                
                if speaker == "management":
                    management_emphasis[tech_id] += 1
                else:
                    analyst_focus[tech_id] += 1
        
        # Calculate metrics
        word_count = len(text.split()) / 1000
        tech_intensity = sum(mentions_by_area.values()) / max(word_count, 1)
        
        # Emerging focus
        emerging_count = 0
        total = sum(mentions_by_area.values())
        for tech_id, count in mentions_by_area.items():
            if self.tech_lexicon[tech_id]["maturity"] == 1:
                emerging_count += count
        emerging_focus = emerging_count / max(total, 1)
        
        # Tech sentiment
        pos_count = sum(1 for m in mentions if m.sentiment == "positive")
        neg_count = sum(1 for m in mentions if m.sentiment == "negative")
        total_sentiment = pos_count + neg_count
        tech_sentiment = (pos_count - neg_count) / max(total_sentiment, 1)
        
        return TranscriptAnalysis(
            ticker=ticker,
            date=date,
            quarter=quarter,
            mentions=mentions,
            mentions_by_area=dict(mentions_by_area),
            tech_intensity=tech_intensity,
            emerging_focus=emerging_focus,
            tech_sentiment=tech_sentiment,
            management_emphasis=dict(management_emphasis),
            analyst_focus=dict(analyst_focus),
        )
    
    def analyze_company(
        self,
        ticker: str,
        quarters: int = 4,
    ) -> EarningsCallTechSignal:
        """
        Analyze a company's earnings calls.
        
        Args:
            ticker: Stock ticker
            quarters: Number of quarters to analyze
            
        Returns:
            EarningsCallTechSignal
        """
        # Fetch transcripts
        transcripts = self.client.get_transcripts(ticker, limit=quarters)
        
        if not transcripts:
            logger.warning(f"No transcripts found for {ticker}")
            return EarningsCallTechSignal(
                ticker=ticker,
                analysis_date=datetime.now(),
                current=None,
                qoq_tech_change=None,
                qoq_sentiment_change=None,
                top_tech_areas=[],
                emerging_mentions=[],
                new_this_quarter=[],
                tech_momentum="unknown",
                signal_strength=0,
            )
        
        # Analyze each transcript
        analyses = []
        for t in transcripts:
            try:
                content = t.get("content", "")
                date_str = t.get("date", "2024-01-01")
                date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                quarter = f"Q{t.get('quarter', 1)} {t.get('year', 2024)}"
                
                analysis = self.analyze_transcript(content, ticker, date, quarter)
                analyses.append(analysis)
                
            except Exception as e:
                logger.debug(f"Error analyzing transcript: {e}")
                continue
        
        if not analyses:
            return EarningsCallTechSignal(
                ticker=ticker,
                analysis_date=datetime.now(),
                current=None,
                qoq_tech_change=None,
                qoq_sentiment_change=None,
                top_tech_areas=[],
                emerging_mentions=[],
                new_this_quarter=[],
                tech_momentum="unknown",
                signal_strength=0,
            )
        
        current = analyses[0]
        previous = analyses[1] if len(analyses) > 1 else None
        
        # QoQ changes
        qoq_tech_change = None
        qoq_sentiment_change = None
        new_this_quarter = []
        
        if previous:
            if previous.tech_intensity > 0:
                qoq_tech_change = (
                    (current.tech_intensity - previous.tech_intensity) 
                    / previous.tech_intensity
                )
            qoq_sentiment_change = current.tech_sentiment - previous.tech_sentiment
            
            # New tech areas
            current_areas = set(current.mentions_by_area.keys())
            previous_areas = set(previous.mentions_by_area.keys())
            new_areas = current_areas - previous_areas
            new_this_quarter = [self.tech_lexicon[t]["name"] for t in new_areas]
        
        # Top tech areas
        sorted_tech = sorted(current.mentions_by_area.items(), key=lambda x: -x[1])
        top_areas = [self.tech_lexicon[t]["name"] for t, _ in sorted_tech[:5]]
        
        # Emerging mentions
        emerging = [
            self.tech_lexicon[t]["name"] 
            for t in current.mentions_by_area 
            if self.tech_lexicon[t]["maturity"] == 1
        ]
        
        # Momentum
        if qoq_tech_change is not None:
            if qoq_tech_change > 0.2:
                momentum = "accelerating"
            elif qoq_tech_change < -0.2:
                momentum = "decelerating"
            else:
                momentum = "stable"
        else:
            momentum = "unknown"
        
        # Signal strength
        signal = 0.0
        if momentum == "accelerating":
            signal += 0.3
        if current.tech_sentiment > 0.3:
            signal += 0.2
        if emerging:
            signal += 0.2
        if new_this_quarter:
            signal += 0.1
        if momentum == "decelerating":
            signal -= 0.3
        if current.tech_sentiment < -0.2:
            signal -= 0.2
        
        # Key quotes
        bullish = []
        bearish = []
        for m in current.mentions[:20]:
            if m.sentiment == "positive" and len(bullish) < 3:
                bullish.append(f"...{m.context.strip()}...")
            elif m.sentiment == "negative" and len(bearish) < 3:
                bearish.append(f"...{m.context.strip()}...")
        
        return EarningsCallTechSignal(
            ticker=ticker,
            analysis_date=datetime.now(),
            current=current,
            qoq_tech_change=qoq_tech_change,
            qoq_sentiment_change=qoq_sentiment_change,
            top_tech_areas=top_areas,
            emerging_mentions=emerging,
            new_this_quarter=new_this_quarter,
            tech_momentum=momentum,
            signal_strength=signal,
            bullish_quotes=bullish,
            bearish_quotes=bearish,
        )
    
    def get_summary(self, signal: EarningsCallTechSignal) -> str:
        """Generate human-readable summary."""
        lines = [f"Earnings Call Tech Analysis: {signal.ticker}"]
        lines.append("=" * 50)
        
        if signal.current:
            c = signal.current
            lines.append(f"Quarter: {c.quarter}")
            lines.append(f"Tech Intensity: {c.tech_intensity:.1f} mentions/1K words")
            lines.append(f"Tech Sentiment: {c.tech_sentiment:+.2f}")
            lines.append(f"Emerging Focus: {c.emerging_focus:.1%}")
            
            if c.management_emphasis:
                top_mgmt = sorted(c.management_emphasis.items(), key=lambda x: -x[1])[:3]
                mgmt_focus = [self.tech_lexicon[t]["name"] for t, _ in top_mgmt]
                lines.append(f"\nManagement Focus: {', '.join(mgmt_focus)}")
            
            if c.analyst_focus:
                top_analyst = sorted(c.analyst_focus.items(), key=lambda x: -x[1])[:3]
                analyst_focus = [self.tech_lexicon[t]["name"] for t, _ in top_analyst]
                lines.append(f"Analyst Questions: {', '.join(analyst_focus)}")
        
        lines.append("")
        lines.append(f"Tech Momentum: {signal.tech_momentum.upper()}")
        
        if signal.qoq_tech_change is not None:
            lines.append(f"QoQ Tech Change: {signal.qoq_tech_change:+.1%}")
        
        if signal.top_tech_areas:
            lines.append(f"\nTop Tech Areas: {', '.join(signal.top_tech_areas[:3])}")
        
        if signal.new_this_quarter:
            lines.append(f"🆕 New This Quarter: {', '.join(signal.new_this_quarter)}")
        
        if signal.emerging_mentions:
            lines.append(f"🚀 Emerging Tech: {', '.join(signal.emerging_mentions)}")
        
        lines.append("")
        lines.append(f"Signal Strength: {signal.signal_strength:+.2f}")
        
        if signal.bullish_quotes:
            lines.append("\n📈 Bullish Quote:")
            lines.append(f'  "{signal.bullish_quotes[0][:200]}"')
        
        if signal.bearish_quotes:
            lines.append("\n📉 Bearish Quote:")
            lines.append(f'  "{signal.bearish_quotes[0][:200]}"')
        
        return "\n".join(lines)


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    
    print(f"Analyzing earnings calls for {ticker}...")
    print()
    
    model = EarningsCallTechModel()
    signal = model.analyze_company(ticker)
    print(model.get_summary(signal))
