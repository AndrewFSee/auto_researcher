"""
Earnings Call Agent.

Fetches and analyzes earnings call transcripts.
Uses LLM to extract key insights, management tone, and forward guidance.

Note: Earnings transcripts require a data source. This agent supports:
1. Local transcript files
2. API integration (requires API key from providers like Seeking Alpha, Financial Modeling Prep)
3. Manual transcript input

Usage:
    from auto_researcher.agents.earnings_agent import EarningsCallAgent
    
    agent = EarningsCallAgent(model="gpt-4o-mini")
    results = agent.analyze_tickers(["GOOGL", "NVDA"])
    
    for ticker, analysis in results.items():
        print(f"{ticker}: {analysis.management_tone} - {analysis.summary}")
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Literal

import pandas as pd
import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Import Enhanced PEAD Model for bias-free earnings drift signal
# Use direct module import to avoid loading models/__init__.py which requires xgboost
try:
    import importlib.util
    import os as _os
    _pead_module_path = _os.path.join(_os.path.dirname(__file__), '..', 'models', 'pead_enhanced.py')
    _spec = importlib.util.spec_from_file_location("pead_enhanced", _pead_module_path)
    _pead_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pead_module)
    EnhancedPEADModel = _pead_module.EnhancedPEADModel
    PEADSignal = _pead_module.PEADSignal
    HAS_PEAD_MODEL = True
except Exception as e:
    logger.warning(f"Could not load PEAD model: {e}")
    HAS_PEAD_MODEL = False
    EnhancedPEADModel = None
    PEADSignal = None

# Try to import optional dependencies
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    litellm = None


# ==============================================================================
# BASE AGENT CLASS
# ==============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        pass


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class EarningsTranscript:
    """An earnings call transcript."""
    ticker: str
    quarter: str  # e.g., "Q4 2024"
    fiscal_year: int
    call_date: datetime
    transcript_text: str
    source: str = "unknown"


@dataclass
class EarningsAnalysis:
    """Analysis of earnings call transcript."""
    ticker: str
    quarter: str
    call_date: Optional[datetime]
    
    # Management tone
    management_tone: Literal["very_negative", "negative", "neutral", "positive", "very_positive"]
    confidence_level: float  # 0 to 1
    
    # Key metrics discussed
    revenue_commentary: Optional[str]
    margin_commentary: Optional[str]
    growth_drivers: list[str]
    
    # Forward guidance
    guidance_direction: Literal["lowered", "maintained", "raised", "not_provided"]
    guidance_details: Optional[str]
    
    # Q&A insights
    analyst_concerns: list[str]
    management_evasions: list[str]  # Topics management avoided or deflected
    
    # Key themes
    key_themes: list[str]
    strategic_priorities: list[str]
    
    # Risks and challenges
    challenges_mentioned: list[str]
    competitive_dynamics: Optional[str]
    
    # Summary
    summary: str
    key_takeaways: list[str]
    
    # FinBERT sentiment (additional ML-based analysis)
    finbert_sentiment_score: Optional[float] = None  # -1 to +1 scale
    finbert_sentiment_label: Optional[str] = None  # positive/neutral/negative
    llm_finbert_agreement: Optional[bool] = None  # Whether LLM and FinBERT agree
    
    # Alpha signal metrics (from trained XGBoost model)
    # Based on 10,000 earnings calls, 500 stocks, statistically significant (p<0.0001)
    days_since_earnings: Optional[int] = None  # Days since the earnings call
    signal_strength: Optional[float] = None  # 0-1, decays with time
    expected_quintile: Optional[int] = None  # 1-5, which return bucket the model predicts
    alpha_decay_note: Optional[str] = None  # Human-readable note about signal freshness
    
    # ==========================================================================
    # PEAD Signal (Post-Earnings Announcement Drift)
    # ==========================================================================
    # Based on bias-free backtest using yfinance earnings_dates (2026-01-30)
    # Uses ACTUAL announcement dates, not fiscal period ends
    #
    # STRENGTHS:
    #   - IC=+0.152*** for big surprises (>20%), 3x improvement over base
    #   - L/S spread: +3.02% at 40d, +3.55% at 60d
    #   - Works best 0-40 days after earnings
    #   - Consecutive beat/miss patterns add momentum
    #
    # WEAKNESSES:
    #   - Weak for small surprises (<10%): IC~0.05 only
    #   - Signal decays after 60 days (drift exhausted)
    #   - Sample size ~334 events/year for big surprises
    #   - Does NOT include analyst estimate revisions (that data was invalid)
    #
    # HOW TO USE:
    #   - Strong signal (is_big_surprise=True): Consider as primary factor
    #   - Weak signal (is_big_surprise=False): Use as secondary confirmation only
    #   - Expired signal (days >60): Ignore, drift already occurred
    # ==========================================================================
    pead_sue: Optional[float] = None  # Standardized Unexpected Earnings
    pead_is_big_surprise: bool = False  # |SUE| > 20% = strong signal
    pead_is_consecutive: bool = False  # Beat/miss streak adds momentum
    pead_direction: Optional[str] = None  # "long", "short", or "neutral"
    pead_strength: Optional[str] = None  # "strong", "moderate", or "weak"
    pead_is_actionable: bool = False  # Whether signal is fresh enough to trade
    pead_expected_return: Optional[float] = None  # Expected L/S spread
    pead_recommended_days: Optional[int] = None  # Optimal holding period
    pead_summary: Optional[str] = None  # Human-readable PEAD signal summary
    
    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None


@dataclass
class EarningsAgentConfig:
    """Configuration for the earnings call agent."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    timeout: int = 120
    max_transcript_chars: int = 50000  # Increased to capture full CFO guidance sections
    fmp_api_key: Optional[str] = None  # Financial Modeling Prep API key
    finnhub_api_key: Optional[str] = None  # Finnhub API key (free tier available)


# ==============================================================================
# TRANSCRIPT SOURCES
# ==============================================================================

class TranscriptSource(ABC):
    """Abstract base for transcript sources."""
    
    @abstractmethod
    def fetch_transcript(self, ticker: str) -> Optional[EarningsTranscript]:
        """Fetch the most recent earnings transcript."""
        pass


class DefeatBetaTranscriptSource(TranscriptSource):
    """
    Fetch transcripts from DefeatBeta's HuggingFace dataset.
    
    This is a FREE source with 180,000+ earnings call transcripts covering 4,700+ tickers.
    Data is updated weekly and includes full speaker attribution.
    
    Dataset: https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data
    """
    
    PARQUET_URL = "https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet"
    
    _cached_df: Optional[pd.DataFrame] = None
    _cache_time: Optional[datetime] = None
    _cache_ttl_hours: int = 24  # Cache for 24 hours
    
    def __init__(self, cache_ttl_hours: int = 24):
        """Initialize the DefeatBeta transcript source.
        
        Args:
            cache_ttl_hours: How long to cache the transcript data (default 24 hours).
        """
        self._cache_ttl_hours = cache_ttl_hours
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load transcript data from HuggingFace, with caching."""
        now = datetime.now()
        
        # Check if we have valid cached data
        if (DefeatBetaTranscriptSource._cached_df is not None and 
            DefeatBetaTranscriptSource._cache_time is not None):
            cache_age = (now - DefeatBetaTranscriptSource._cache_time).total_seconds() / 3600
            if cache_age < self._cache_ttl_hours:
                return DefeatBetaTranscriptSource._cached_df
        
        try:
            logger.info("Loading DefeatBeta transcripts from HuggingFace (this may take a moment)...")
            df = pd.read_parquet(self.PARQUET_URL)
            DefeatBetaTranscriptSource._cached_df = df
            DefeatBetaTranscriptSource._cache_time = now
            logger.info(f"Loaded {len(df):,} transcripts covering {df['symbol'].nunique():,} tickers")
            return df
        except Exception as e:
            logger.warning(f"Failed to load DefeatBeta transcripts: {e}")
            return None
    
    def fetch_transcript(self, ticker: str) -> Optional[EarningsTranscript]:
        """Fetch the most recent earnings transcript for a ticker."""
        try:
            df = self._load_data()
            if df is None:
                return None
            
            # Filter for ticker and sort by date
            ticker_df = df[df['symbol'] == ticker.upper()].copy()
            if ticker_df.empty:
                logger.debug(f"No DefeatBeta transcripts found for {ticker}")
                return None
            
            # Sort by date descending to get most recent
            ticker_df = ticker_df.sort_values('report_date', ascending=False)
            latest = ticker_df.iloc[0]
            
            # Extract transcript paragraphs
            transcripts = latest['transcripts']
            if hasattr(transcripts, 'tolist'):
                transcripts = transcripts.tolist()
            
            if not transcripts:
                return None
            
            # Build transcript text with speaker attribution
            text_parts = []
            for para in transcripts:
                if isinstance(para, dict):
                    speaker = para.get('speaker', 'Unknown')
                    content = para.get('content', '')
                    text_parts.append(f"\n{speaker}: {content}")
                else:
                    text_parts.append(str(para))
            
            transcript_text = "\n".join(text_parts)
            
            if not transcript_text.strip():
                return None
            
            # Parse date
            report_date = latest['report_date']
            if isinstance(report_date, str):
                try:
                    call_date = datetime.strptime(report_date[:10], '%Y-%m-%d')
                except:
                    call_date = datetime.now()
            else:
                call_date = pd.Timestamp(report_date).to_pydatetime()
            
            return EarningsTranscript(
                ticker=ticker,
                quarter=f"Q{latest['fiscal_quarter']} {latest['fiscal_year']}",
                fiscal_year=int(latest['fiscal_year']),
                call_date=call_date,
                transcript_text=transcript_text,
                source="DefeatBeta (HuggingFace)",
            )
            
        except Exception as e:
            logger.debug(f"DefeatBeta transcript fetch failed for {ticker}: {e}")
            return None


class FMPTranscriptSource(TranscriptSource):
    """Fetch transcripts from Financial Modeling Prep API."""
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def fetch_transcript(self, ticker: str) -> Optional[EarningsTranscript]:
        """Fetch latest earnings transcript from FMP."""
        try:
            url = f"{self.BASE_URL}/earning_call_transcript/{ticker}"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return None
            
            latest = data[0]  # Most recent
            
            return EarningsTranscript(
                ticker=ticker,
                quarter=f"Q{latest.get('quarter', '?')} {latest.get('year', '?')}",
                fiscal_year=latest.get('year', 0),
                call_date=datetime.strptime(latest.get('date', '')[:10], '%Y-%m-%d') if latest.get('date') else datetime.now(),
                transcript_text=latest.get('content', ''),
                source="Financial Modeling Prep",
            )
            
        except Exception as e:
            # 403 errors are expected for FMP free tier - just log at debug level
            if "403" in str(e):
                logger.debug(f"FMP transcript not available for {ticker} (requires paid tier)")
            else:
                logger.warning(f"FMP transcript fetch failed for {ticker}: {e}")
            return None


class FinnhubTranscriptSource(TranscriptSource):
    """
    Fetch transcripts from Finnhub API.
    Free tier: 60 API calls/minute, includes earnings call transcripts.
    Get free API key at: https://finnhub.io/register
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def fetch_transcript(self, ticker: str) -> Optional[EarningsTranscript]:
        """Fetch latest earnings transcript from Finnhub."""
        try:
            # First get list of available transcripts
            url = f"{self.BASE_URL}/stock/transcripts/list"
            params = {"symbol": ticker, "token": self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            transcripts = response.json()
            
            if not transcripts or 'transcripts' not in transcripts:
                logger.debug(f"No transcripts found for {ticker} on Finnhub")
                return None
            
            transcript_list = transcripts.get('transcripts', [])
            if not transcript_list:
                return None
            
            # Get the most recent transcript
            latest = transcript_list[0]  # Already sorted by date desc
            transcript_id = latest.get('id')
            
            if not transcript_id:
                return None
            
            # Fetch the actual transcript content
            content_url = f"{self.BASE_URL}/stock/transcripts"
            content_params = {"id": transcript_id, "token": self.api_key}
            
            content_response = requests.get(content_url, params=content_params, timeout=30)
            content_response.raise_for_status()
            
            transcript_data = content_response.json()
            
            # Build transcript text from segments
            segments = transcript_data.get('transcript', [])
            if not segments:
                return None
            
            text_parts = []
            for segment in segments:
                speaker = segment.get('name', 'Unknown')
                speech = segment.get('speech', [])
                if speech:
                    text_parts.append(f"\n{speaker}:")
                    text_parts.extend(speech)
            
            transcript_text = "\n".join(text_parts)
            
            if not transcript_text.strip():
                return None
            
            # Parse quarter info
            quarter = latest.get('quarter', 0)
            year = latest.get('year', datetime.now().year)
            
            return EarningsTranscript(
                ticker=ticker,
                quarter=f"Q{quarter} {year}",
                fiscal_year=year,
                call_date=datetime.strptime(latest.get('time', '')[:10], '%Y-%m-%d') if latest.get('time') else datetime.now(),
                transcript_text=transcript_text,
                source="Finnhub",
            )
            
        except Exception as e:
            logger.debug(f"Finnhub transcript fetch failed for {ticker}: {e}")
            return None


class YahooEarningsSource(TranscriptSource):
    """
    Fetch earnings information from Yahoo Finance.
    Uses income statement, financial metrics, and news to build context.
    """
    
    def fetch_transcript(self, ticker: str) -> Optional[EarningsTranscript]:
        """Fetch earnings info from Yahoo Finance."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            
            text_parts = []
            text_parts.append(f"EARNINGS AND FINANCIAL DATA FOR {ticker}")
            text_parts.append(f"=" * 50)
            
            # Get company info
            info = stock.info
            if info:
                text_parts.append(f"\nCompany: {info.get('longName', ticker)}")
                text_parts.append(f"Sector: {info.get('sector', 'N/A')}")
                text_parts.append(f"Industry: {info.get('industry', 'N/A')}")
            
            # Get quarterly income statement
            try:
                quarterly_income = stock.quarterly_income_stmt
                if quarterly_income is not None and not quarterly_income.empty:
                    text_parts.append(f"\nQUARTERLY INCOME STATEMENT (Last 4 quarters):")
                    
                    # Key metrics
                    metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EBITDA']
                    for metric in metrics:
                        if metric in quarterly_income.index:
                            row = quarterly_income.loc[metric]
                            values = [f"{v/1e9:.2f}B" if pd.notna(v) else "N/A" for v in row.head(4)]
                            text_parts.append(f"  {metric}: {' | '.join(values)}")
            except Exception as e:
                logger.debug(f"Could not get income statement: {e}")
            
            # Get key financial metrics
            if info:
                text_parts.append(f"\nKEY METRICS:")
                metrics_to_show = [
                    ('Revenue Growth', 'revenueGrowth'),
                    ('Earnings Growth', 'earningsGrowth'),
                    ('Profit Margin', 'profitMargins'),
                    ('Operating Margin', 'operatingMargins'),
                    ('ROE', 'returnOnEquity'),
                    ('P/E Ratio', 'trailingPE'),
                    ('Forward P/E', 'forwardPE'),
                ]
                for label, key in metrics_to_show:
                    val = info.get(key)
                    if val is not None:
                        if 'Margin' in label or 'Growth' in label or 'ROE' in label:
                            text_parts.append(f"  {label}: {val*100:.1f}%")
                        else:
                            text_parts.append(f"  {label}: {val:.2f}")
            
            # Get recent news for earnings context
            try:
                news = stock.news
                if news:
                    text_parts.append("\nRECENT NEWS:")
                    for item in news[:8]:
                        content = item.get('content', item)
                        if isinstance(content, dict):
                            title = content.get('title', 'No title')
                            summary = content.get('summary', '')[:200]
                        else:
                            title = item.get('title', 'No title')
                            summary = ''
                        text_parts.append(f"  - {title}")
                        if summary:
                            text_parts.append(f"    {summary}...")
            except Exception as e:
                logger.debug(f"Could not get news: {e}")
            
            # Get earnings dates if available
            try:
                calendar = stock.calendar
                if calendar is not None and not calendar.empty:
                    text_parts.append(f"\nEARNINGS CALENDAR:")
                    for col in calendar.columns[:2]:
                        text_parts.append(f"  {col}: {calendar[col].iloc[0] if len(calendar) > 0 else 'N/A'}")
            except:
                pass
            
            # Determine quarter
            import datetime as dt
            now = datetime.now()
            quarter = f"Q{(now.month - 1) // 3 + 1} {now.year}"
            
            if len(text_parts) > 5:  # We got some data
                return EarningsTranscript(
                    ticker=ticker,
                    quarter=quarter,
                    fiscal_year=now.year,
                    call_date=now,
                    transcript_text="\n".join(text_parts),
                    source="Yahoo Finance",
                )
            
        except Exception as e:
            logger.warning(f"Yahoo earnings fetch failed for {ticker}: {e}")
        
        return None


class ManualTranscriptSource(TranscriptSource):
    """Load transcripts from local files."""
    
    def __init__(self, transcript_dir: str = "transcripts"):
        self.transcript_dir = transcript_dir
    
    def fetch_transcript(self, ticker: str) -> Optional[EarningsTranscript]:
        """Load transcript from local file."""
        import os
        
        # Look for file like AAPL_Q4_2024.txt
        if not os.path.exists(self.transcript_dir):
            return None
        
        # Find most recent transcript
        pattern = f"{ticker}_Q"
        files = [f for f in os.listdir(self.transcript_dir) if f.startswith(pattern)]
        
        if not files:
            return None
        
        # Sort and get most recent
        files.sort(reverse=True)
        latest_file = files[0]
        
        with open(os.path.join(self.transcript_dir, latest_file), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse filename for quarter info
        match = re.match(r'(\w+)_Q(\d)_(\d{4})\.txt', latest_file)
        if match:
            quarter = f"Q{match.group(2)} {match.group(3)}"
            year = int(match.group(3))
        else:
            quarter = "Unknown"
            year = datetime.now().year
        
        return EarningsTranscript(
            ticker=ticker,
            quarter=quarter,
            fiscal_year=year,
            call_date=datetime.now(),
            transcript_text=content,
            source="local_file",
        )


# ==============================================================================
# EARNINGS CALL AGENT
# ==============================================================================

# ==============================================================================
# ALPHA DECAY CONSTANTS
# ==============================================================================

# Based on 10,000 earnings calls across 500 stocks (statistically significant, p<0.0001)
# Signal decays as time passes since earnings call
ALPHA_DECAY_CONFIG = {
    'optimal_holding_days': 10,  # Signal is strongest in first 10 days
    'max_signal_days': 20,  # Beyond this, signal becomes unreliable
    'spreads': {'1d': 0.0032, '5d': 0.0109, '10d': 0.0178, '20d': 0.0200},
    'daily_alpha': {'1d': 0.00316, '5d': 0.00219, '10d': 0.00178, '20d': 0.00100},
    'quintile_returns': {
        1: -0.0036,  # Q1 (worst sentiment) averages -0.36% in 5 days
        2: 0.0038,   # Q2 averages +0.38%
        3: 0.0065,   # Q3 averages +0.65%
        4: 0.0067,   # Q4 averages +0.67%
        5: 0.0074,   # Q5 (best sentiment) averages +0.74%
    },
}


def compute_alpha_decay(days_since_earnings: int) -> tuple[float, str]:
    """
    Compute signal strength and decay note based on days since earnings.
    
    Returns:
        (signal_strength, decay_note) where signal_strength is 0-1
    """
    if days_since_earnings is None:
        return None, "Unknown earnings date"
    
    if days_since_earnings <= 0:
        return 1.0, "🟢 Fresh signal - earnings just released"
    elif days_since_earnings <= 5:
        # Peak signal period
        strength = 1.0
        return strength, f"🟢 Strong signal - {days_since_earnings}d since earnings (optimal: 5-10d)"
    elif days_since_earnings <= 10:
        # Still optimal
        strength = 1.0
        return strength, f"🟢 Strong signal - {days_since_earnings}d since earnings (within optimal window)"
    elif days_since_earnings <= 20:
        # Decaying
        decay = (days_since_earnings - 10) / 10  # 0 at day 10, 1 at day 20
        strength = 1.0 - (decay * 0.5)  # Decays to 0.5 by day 20
        return strength, f"🟡 Decaying signal - {days_since_earnings}d since earnings ({strength:.0%} strength)"
    else:
        # Beyond useful window
        strength = max(0.1, 0.5 - ((days_since_earnings - 20) / 40))  # Continues to decay
        return strength, f"🔴 Stale signal - {days_since_earnings}d since earnings (signal may not be actionable)"


def estimate_quintile_from_sentiment(finbert_score: Optional[float]) -> Optional[int]:
    """
    Estimate expected return quintile based on FinBERT sentiment score.
    
    Based on the trained XGBoost model, sentiment is the top feature.
    """
    if finbert_score is None:
        return None
    
    # Map sentiment score (-1 to +1) to quintile (1-5)
    # Thresholds based on typical FinBERT score distribution
    if finbert_score < -0.3:
        return 1  # Very negative -> expect underperformance
    elif finbert_score < 0.0:
        return 2
    elif finbert_score < 0.3:
        return 3
    elif finbert_score < 0.6:
        return 4
    else:
        return 5  # Very positive -> expect outperformance


class EarningsCallAgent(BaseAgent):
    """
    Agent for analyzing earnings call transcripts.
    
    Uses LLM to extract management tone, guidance, key themes,
    and investment implications from earnings calls.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        config: Optional[EarningsAgentConfig] = None,
        use_finbert: bool = True,  # Enable FinBERT by default for transcript analysis
    ) -> None:
        """Initialize the Earnings Call Agent."""
        if not HAS_LITELLM:
            raise ImportError("litellm is required. Install with: pip install litellm")
        
        self.config = config or EarningsAgentConfig(model=model)
        self.model = self.config.model
        self.use_finbert = use_finbert
        
        # Initialize FinBERT analyzer for sentiment scoring
        self._finbert = None
        if use_finbert:
            try:
                from .finbert_sentiment import create_analyzer
                self._finbert = create_analyzer(prefer_finbert=True)
                logger.info(f"Loaded {type(self._finbert).__name__} for earnings sentiment analysis")
            except Exception as e:
                logger.warning(f"Could not load FinBERT analyzer: {e}")
                self._finbert = None
        
        # Initialize transcript sources (in order of preference)
        self.sources: list[TranscriptSource] = []
        
        # DefeatBeta FIRST - free, comprehensive, 180k+ transcripts, 4700+ tickers
        self.sources.append(DefeatBetaTranscriptSource())
        logger.info("Added DefeatBeta transcript source (primary)")
        
        # Finnhub as backup (free tier may have transcripts for some tickers)
        finnhub_key = self.config.finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        if finnhub_key:
            self.sources.append(FinnhubTranscriptSource(finnhub_key))
            logger.debug("Added Finnhub transcript source (backup)")
        
        # Try FMP if API key available (requires paid tier for transcripts)
        fmp_key = self.config.fmp_api_key or os.getenv("FMP_API_KEY")
        if fmp_key:
            self.sources.append(FMPTranscriptSource(fmp_key))
        
        # Add Yahoo as fallback (metrics only, no transcripts)
        self.sources.append(YahooEarningsSource())
        
        # Add local files
        self.sources.append(ManualTranscriptSource())
        
        # Initialize Enhanced PEAD Model for drift signal
        self._pead_model = None
        if HAS_PEAD_MODEL:
            try:
                self._pead_model = EnhancedPEADModel(cache_ttl_hours=24)
                logger.info("Loaded EnhancedPEADModel for PEAD drift signals")
            except Exception as e:
                logger.warning(f"Could not load PEAD model: {e}")
        
        logger.info(f"Initialized EarningsCallAgent with {len(self.sources)} sources")
    
    @property
    def name(self) -> str:
        return "EarningsCallAgent"
    
    def run(
        self,
        tickers: list[str],
        **kwargs,
    ) -> dict[str, EarningsAnalysis]:
        """Execute earnings analysis for tickers."""
        return self.analyze_tickers(tickers)
    
    def _analyze_finbert_sentiment(self, transcript: EarningsTranscript) -> tuple[Optional[float], Optional[str]]:
        """
        Analyze transcript sentiment using FinBERT.
        
        Returns:
            Tuple of (sentiment_score, sentiment_label) or (None, None) if unavailable.
            Score ranges from -1 (very negative) to +1 (very positive).
        """
        if self._finbert is None:
            return None, None
        
        try:
            # Get key sections for sentiment analysis (executive remarks are most important)
            text = transcript.transcript_text
            
            # For long transcripts, analyze in chunks and aggregate
            # Focus on first ~10k chars (opening remarks) and last ~5k chars (closing/guidance)
            if len(text) > 15000:
                intro_section = text[:10000]
                closing_section = text[-5000:]
                combined_text = intro_section + "\n\n" + closing_section
            else:
                combined_text = text
            
            # Analyze sentiment
            result = self._finbert.analyze(combined_text[:15000])  # Cap to avoid memory issues
            
            score = result.sentiment_score if hasattr(result, 'sentiment_score') else result.score
            label = result.label
            
            logger.debug(f"FinBERT sentiment for {transcript.ticker}: {label} ({score:.3f})")
            
            return score, label
            
        except Exception as e:
            logger.debug(f"FinBERT analysis failed for {transcript.ticker}: {e}")
            return None, None
    
    def _fetch_transcript(self, ticker: str) -> Optional[EarningsTranscript]:
        """Try to fetch transcript from available sources."""
        for source in self.sources:
            transcript = source.fetch_transcript(ticker)
            if transcript and transcript.transcript_text:
                return transcript
        return None
    
    def _build_prompt(self, transcript: EarningsTranscript) -> str:
        """Build the analysis prompt for the LLM."""
        
        # Truncate transcript if too long
        text = transcript.transcript_text[:self.config.max_transcript_chars]
        
        prompt = f"""Analyze this earnings call/earnings information for {transcript.ticker} ({transcript.quarter}):

{text}

Provide your analysis in EXACTLY this format:

MANAGEMENT_TONE: [one of: very_negative, negative, neutral, positive, very_positive]
CONFIDENCE_LEVEL: [number from 0.0 to 1.0 indicating management confidence]
REVENUE_COMMENTARY: [brief summary of revenue discussion]
MARGIN_COMMENTARY: [brief summary of margin trends]
GROWTH_DRIVERS: [comma-separated list of key growth drivers mentioned]
GUIDANCE_DIRECTION: [one of: lowered, maintained, raised, not_provided]
GUIDANCE_DETAILS: [specific guidance details if provided]
ANALYST_CONCERNS: [comma-separated list of main analyst concerns from Q&A]
MANAGEMENT_EVASIONS: [topics management avoided or deflected, or "None noted"]
KEY_THEMES: [comma-separated list of 3-5 key themes]
STRATEGIC_PRIORITIES: [comma-separated list of strategic priorities mentioned]
CHALLENGES_MENTIONED: [comma-separated list of challenges/headwinds discussed]
COMPETITIVE_DYNAMICS: [brief summary of competitive positioning]
SUMMARY: [2-3 sentence overall summary]
KEY_TAKEAWAYS: [comma-separated list of 3-4 key takeaways for investors]

Focus on:
1. Management tone and confidence
2. Forward-looking statements and guidance
3. Key business drivers and challenges
4. What wasn't said or was deflected
5. Investment implications"""

        return prompt
    
    def _parse_response(self, ticker: str, response: str, transcript: EarningsTranscript) -> EarningsAnalysis:
        """Parse the LLM response into an EarningsAnalysis."""
        
        # Defaults
        management_tone = "neutral"
        confidence_level = 0.5
        revenue_commentary = None
        margin_commentary = None
        growth_drivers = []
        guidance_direction = "not_provided"
        guidance_details = None
        analyst_concerns = []
        management_evasions = []
        key_themes = []
        strategic_priorities = []
        challenges_mentioned = []
        competitive_dynamics = None
        summary = "Analysis not available."
        key_takeaways = []
        
        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("MANAGEMENT_TONE:"):
                    tone = line.split(":", 1)[1].strip().lower()
                    if tone in ["very_negative", "negative", "neutral", "positive", "very_positive"]:
                        management_tone = tone
                elif line.startswith("CONFIDENCE_LEVEL:"):
                    try:
                        confidence_level = float(line.split(":", 1)[1].strip())
                        confidence_level = max(0.0, min(1.0, confidence_level))
                    except:
                        pass
                elif line.startswith("REVENUE_COMMENTARY:"):
                    revenue_commentary = line.split(":", 1)[1].strip()
                elif line.startswith("MARGIN_COMMENTARY:"):
                    margin_commentary = line.split(":", 1)[1].strip()
                elif line.startswith("GROWTH_DRIVERS:"):
                    drivers_str = line.split(":", 1)[1].strip()
                    growth_drivers = [d.strip() for d in drivers_str.split(",") if d.strip()]
                elif line.startswith("GUIDANCE_DIRECTION:"):
                    direction = line.split(":", 1)[1].strip().lower()
                    if direction in ["lowered", "maintained", "raised", "not_provided"]:
                        guidance_direction = direction
                elif line.startswith("GUIDANCE_DETAILS:"):
                    guidance_details = line.split(":", 1)[1].strip()
                elif line.startswith("ANALYST_CONCERNS:"):
                    concerns_str = line.split(":", 1)[1].strip()
                    analyst_concerns = [c.strip() for c in concerns_str.split(",") if c.strip()]
                elif line.startswith("MANAGEMENT_EVASIONS:"):
                    evasions_str = line.split(":", 1)[1].strip()
                    if "none" not in evasions_str.lower():
                        management_evasions = [e.strip() for e in evasions_str.split(",") if e.strip()]
                elif line.startswith("KEY_THEMES:"):
                    themes_str = line.split(":", 1)[1].strip()
                    key_themes = [t.strip() for t in themes_str.split(",") if t.strip()]
                elif line.startswith("STRATEGIC_PRIORITIES:"):
                    priorities_str = line.split(":", 1)[1].strip()
                    strategic_priorities = [p.strip() for p in priorities_str.split(",") if p.strip()]
                elif line.startswith("CHALLENGES_MENTIONED:"):
                    challenges_str = line.split(":", 1)[1].strip()
                    challenges_mentioned = [c.strip() for c in challenges_str.split(",") if c.strip()]
                elif line.startswith("COMPETITIVE_DYNAMICS:"):
                    competitive_dynamics = line.split(":", 1)[1].strip()
                elif line.startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.startswith("KEY_TAKEAWAYS:"):
                    takeaways_str = line.split(":", 1)[1].strip()
                    key_takeaways = [t.strip() for t in takeaways_str.split(",") if t.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse earnings response: {e}")
        
        return EarningsAnalysis(
            ticker=ticker,
            quarter=transcript.quarter,
            call_date=transcript.call_date,
            management_tone=management_tone,
            confidence_level=confidence_level,
            revenue_commentary=revenue_commentary,
            margin_commentary=margin_commentary,
            growth_drivers=growth_drivers,
            guidance_direction=guidance_direction,
            guidance_details=guidance_details,
            analyst_concerns=analyst_concerns,
            management_evasions=management_evasions,
            key_themes=key_themes,
            strategic_priorities=strategic_priorities,
            challenges_mentioned=challenges_mentioned,
            competitive_dynamics=competitive_dynamics,
            summary=summary,
            key_takeaways=key_takeaways,
            raw_response=response,
        )
    
    def analyze_ticker(self, ticker: str) -> EarningsAnalysis:
        """Analyze earnings call for a single ticker."""
        
        # Fetch transcript
        transcript = self._fetch_transcript(ticker)
        
        if not transcript:
            return EarningsAnalysis(
                ticker=ticker,
                quarter="Unknown",
                call_date=None,
                management_tone="neutral",
                confidence_level=0.0,
                revenue_commentary=None,
                margin_commentary=None,
                growth_drivers=[],
                guidance_direction="not_provided",
                guidance_details=None,
                analyst_concerns=[],
                management_evasions=[],
                key_themes=[],
                strategic_priorities=[],
                challenges_mentioned=[],
                competitive_dynamics=None,
                summary="No earnings transcript available.",
                key_takeaways=["Earnings data not available"],
            )
        
        # Run FinBERT sentiment analysis in parallel with LLM
        finbert_score, finbert_label = self._analyze_finbert_sentiment(transcript)
        
        # Build prompt
        prompt = self._build_prompt(transcript)
        
        # Call LLM
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst specializing in earnings call analysis. Extract key insights for investors."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            
            response_text = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM call failed for {ticker}: {e}")
            return EarningsAnalysis(
                ticker=ticker,
                quarter=transcript.quarter,
                call_date=transcript.call_date,
                management_tone="neutral",
                confidence_level=0.0,
                revenue_commentary=None,
                margin_commentary=None,
                growth_drivers=[],
                guidance_direction="not_provided",
                guidance_details=None,
                analyst_concerns=[],
                management_evasions=[],
                key_themes=[],
                strategic_priorities=[],
                challenges_mentioned=[],
                competitive_dynamics=None,
                summary=f"Analysis failed: {str(e)[:50]}",
                key_takeaways=[],
                finbert_sentiment_score=finbert_score,
                finbert_sentiment_label=finbert_label,
            )
        
        # Parse LLM response
        analysis = self._parse_response(ticker, response_text, transcript)
        
        # Add FinBERT results
        analysis.finbert_sentiment_score = finbert_score
        analysis.finbert_sentiment_label = finbert_label
        
        # Check LLM/FinBERT agreement
        if finbert_label and analysis.management_tone:
            # Map LLM tones to positive/negative/neutral
            llm_positive = analysis.management_tone in ["positive", "very_positive"]
            llm_negative = analysis.management_tone in ["negative", "very_negative"]
            llm_neutral = analysis.management_tone == "neutral"
            
            finbert_positive = finbert_label == "positive"
            finbert_negative = finbert_label == "negative"
            finbert_neutral = finbert_label == "neutral"
            
            # Agreement if both are same polarity (or both neutral)
            analysis.llm_finbert_agreement = (
                (llm_positive and finbert_positive) or
                (llm_negative and finbert_negative) or
                (llm_neutral and finbert_neutral)
            )
        
        # Compute alpha decay and signal strength
        if transcript.call_date:
            days_since = (datetime.now() - transcript.call_date).days
            analysis.days_since_earnings = days_since
            signal_strength, decay_note = compute_alpha_decay(days_since)
            analysis.signal_strength = signal_strength
            analysis.alpha_decay_note = decay_note
            analysis.expected_quintile = estimate_quintile_from_sentiment(finbert_score)
        
        # =======================================================================
        # PEAD Signal (Post-Earnings Announcement Drift)
        # =======================================================================
        # This uses yfinance for ACTUAL announcement dates (bias-free)
        # IC=+0.152*** for big surprises, much stronger than transcript sentiment
        if self._pead_model:
            try:
                pead_signal = self._pead_model.get_signal(ticker)
                
                analysis.pead_sue = pead_signal.sue
                analysis.pead_is_big_surprise = pead_signal.is_big_surprise
                analysis.pead_is_consecutive = pead_signal.is_consecutive
                analysis.pead_direction = pead_signal.direction
                analysis.pead_strength = pead_signal.strength
                analysis.pead_is_actionable = pead_signal.is_actionable
                analysis.pead_expected_return = pead_signal.expected_return
                analysis.pead_recommended_days = pead_signal.recommended_days
                analysis.pead_summary = pead_signal.summary
                
                # Log if we have a strong PEAD signal
                if pead_signal.is_big_surprise and pead_signal.is_actionable:
                    logger.info(
                        f"{ticker} PEAD: {pead_signal.direction.upper()} "
                        f"(SUE={pead_signal.sue*100:+.1f}%, strength={pead_signal.strength})"
                    )
            except Exception as e:
                logger.debug(f"PEAD signal failed for {ticker}: {e}")
        
        return analysis
    
    def analyze_tickers(self, tickers: list[str]) -> dict[str, EarningsAnalysis]:
        """Analyze earnings calls for multiple tickers."""
        results = {}
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Analyzing earnings for {ticker} ({i}/{len(tickers)})...")
            try:
                result = self.analyze_ticker(ticker)
                results[ticker] = result
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
        
        return results
    
    def format_results(self, results: dict[str, EarningsAnalysis]) -> str:
        """Format results as a readable string."""
        lines = []
        lines.append("=" * 70)
        lines.append("  EARNINGS CALL ANALYSIS")
        lines.append("=" * 70)
        lines.append("")
        
        for ticker, r in results.items():
            # Tone indicator
            if r.management_tone in ["positive", "very_positive"]:
                indicator = "🟢"
            elif r.management_tone in ["negative", "very_negative"]:
                indicator = "🔴"
            else:
                indicator = "🟡"
            
            tone_display = r.management_tone.replace("_", " ").title()
            lines.append(f"{indicator} {ticker} ({r.quarter}) - {tone_display}")
            
            lines.append(f"   {r.summary}")
            
            # Alpha signal section
            if r.finbert_sentiment_score is not None:
                lines.append(f"   📊 Sentiment Signal:")
                lines.append(f"      FinBERT: {r.finbert_sentiment_label} ({r.finbert_sentiment_score:+.2f})")
                if r.expected_quintile:
                    quintile_returns = ALPHA_DECAY_CONFIG['quintile_returns']
                    expected_ret = quintile_returns.get(r.expected_quintile, 0)
                    lines.append(f"      Expected Quintile: Q{r.expected_quintile} (avg 5d return: {expected_ret*100:+.2f}%)")
                if r.alpha_decay_note:
                    lines.append(f"      {r.alpha_decay_note}")
            
            if r.guidance_direction != "not_provided":
                guidance_emoji = "📈" if r.guidance_direction == "raised" else "📉" if r.guidance_direction == "lowered" else "➡️"
                lines.append(f"   {guidance_emoji} Guidance: {r.guidance_direction.title()}")
                if r.guidance_details:
                    lines.append(f"      {r.guidance_details}")
            
            if r.key_themes:
                lines.append(f"   🎯 Themes: {', '.join(r.key_themes[:3])}")
            
            if r.growth_drivers:
                lines.append(f"   📈 Drivers: {', '.join(r.growth_drivers[:2])}")
            
            if r.challenges_mentioned:
                lines.append(f"   ⚠️  Challenges: {', '.join(r.challenges_mentioned[:2])}")
            
            # PEAD Signal section (bias-free earnings drift)
            if r.pead_summary:
                lines.append(f"   📉 PEAD Drift Signal:")
                lines.append(f"      {r.pead_summary}")
                if r.pead_is_actionable and r.pead_expected_return:
                    lines.append(f"      Expected L/S spread: {r.pead_expected_return*100:+.1f}% over {r.pead_recommended_days}d")
                    if r.pead_is_big_surprise:
                        lines.append(f"      ⭐ BIG SURPRISE - High confidence signal (IC=0.15***)")
            
            if r.key_takeaways:
                lines.append(f"   💡 Takeaways:")
                for takeaway in r.key_takeaways[:3]:
                    lines.append(f"      • {takeaway}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dataframe(self, results: dict[str, EarningsAnalysis]) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        for ticker, r in results.items():
            data.append({
                "ticker": ticker,
                "quarter": r.quarter,
                "call_date": r.call_date,
                "management_tone": r.management_tone,
                "confidence": r.confidence_level,
                "guidance_direction": r.guidance_direction,
                "finbert_score": r.finbert_sentiment_score,
                "finbert_label": r.finbert_sentiment_label,
                "days_since_earnings": r.days_since_earnings,
                "signal_strength": r.signal_strength,
                "expected_quintile": r.expected_quintile,
                "alpha_decay_note": r.alpha_decay_note,
                # PEAD model fields
                "pead_sue": r.pead_sue,
                "pead_direction": r.pead_direction,
                "pead_is_big_surprise": r.pead_is_big_surprise,
                "pead_expected_return": r.pead_expected_return,
                "pead_hold_days": r.pead_hold_days,
                "pead_days_elapsed": r.pead_days_elapsed,
                "pead_actionable": r.pead_actionable,
                "pead_consecutive_pattern": r.pead_consecutive_pattern,
                "key_themes": "; ".join(r.key_themes),
                "growth_drivers": "; ".join(r.growth_drivers),
                "challenges": "; ".join(r.challenges_mentioned),
                "summary": r.summary,
            })
        return pd.DataFrame(data)


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze earnings calls")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to analyze")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    
    args = parser.parse_args()
    
    agent = EarningsCallAgent(model=args.model)
    results = agent.analyze_tickers(args.tickers)
    print(agent.format_results(results))
