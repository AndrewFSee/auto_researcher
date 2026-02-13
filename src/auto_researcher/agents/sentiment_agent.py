"""
Sentiment Agent.

This agent analyzes news sentiment for stocks using LLM, FinBERT, 
topic modeling, and earnings-focused models.

================================================================================
SENTIMENT MODELS - STRENGTHS & WEAKNESSES
================================================================================

TOPIC SENTIMENT MODEL:
    Academic Basis: Garcia (2013), Boudoukh et al. (2019)
    
    STRENGTHS:
        ✅ Topic classification adds context to sentiment
        ✅ Topic-specific multipliers (litigation hits harder)
        ✅ Detects litigation/management alerts
        ✅ Separates high-signal from low-signal news
    
    WEAKNESSES:
        ⚠️ Keyword-based topic detection is imperfect
        ⚠️ General news dominates (~70% of articles)
        ⚠️ Topic-adjusted sentiment still has weak overall IC
        
EARNINGS TOPIC MODEL (STRONGEST SIGNAL):
    Academic Basis: Tetlock et al. (2008), backtest validation
    Empirical Finding: IC = +0.02 at 10-day horizon (p < 0.001)
    
    STRENGTHS:
        ✅ BEST PERFORMING: 3-4x IC vs generic sentiment
        ✅ Strong statistical significance (p < 0.001)
        ✅ Threshold signals tradeable (+0.65% spread, p=0.0003)
        ✅ Multi-article days have 26% higher IC
        ✅ Topic + FinBERT agreement boosts confidence
    
    WEAKNESSES:
        ⚠️ Only works on earnings-related news
        ⚠️ Requires earnings articles to generate signal
        ⚠️ Signal decays quickly (5-day half-life)
        ⚠️ Post-earnings drift already partially priced
    
    BEST USE CASES:
        • Filter to earnings articles before computing sentiment
        • Weight by article count (more = higher conviction)
        • Focus on extreme signals (>0.2 or <-0.2)
        • Use when topic and FinBERT agree
    
    CAUTIONS:
        • Earnings sentiment is freshest right after earnings
        • Don't over-trade on single-article signals
        • Combine with fundamental analysis

================================================================================

Usage:
    from auto_researcher.agents.sentiment_agent import SentimentAgent
    
    agent = SentimentAgent(model="gpt-4o-mini")
    results = agent.analyze_tickers(["GOOGL", "NVDA", "AAPL"])
    
    for ticker, analysis in results.items():
        print(f"{ticker}: {analysis.sentiment_score:.2f} - {analysis.summary}")
        
        # Check earnings signal (strongest predictor)
        if analysis.earnings_topic_tradeable:
            print(f"  Earnings Signal: {analysis.earnings_topic_signal}")
            print(f"  Expected Alpha: {analysis.earnings_expected_alpha:.2%}")
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal

import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars set another way

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    litellm = None

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class NewsItem:
    """A single news article."""
    title: str
    source: str
    published: datetime
    url: str
    snippet: Optional[str] = None


@dataclass
class SentimentResult:
    """Sentiment analysis result for a ticker."""
    ticker: str
    sentiment_score: float  # -1 (bearish) to +1 (bullish)
    sentiment_label: Literal["very_bearish", "bearish", "neutral", "bullish", "very_bullish"]
    confidence: float  # 0 to 1
    summary: str  # One-line summary
    key_themes: list[str]  # Main themes from news
    news_count: int
    analysis_date: datetime
    raw_response: Optional[str] = None
    # FinBERT fields (for hybrid mode)
    finbert_score: Optional[float] = None  # FinBERT sentiment (-1 to +1)
    finbert_label: Optional[str] = None  # FinBERT label
    llm_agrees_with_finbert: Optional[bool] = None  # Whether LLM agrees with FinBERT
    method: str = "llm"  # "llm", "finbert", or "hybrid"
    # Topic sentiment fields (new)
    topic_sentiment: Optional[dict] = None  # sentiment by topic
    topic_counts: Optional[dict] = None  # article count by topic
    topic_signals: Optional[dict] = None  # signal by topic
    topic_composite_score: Optional[float] = None  # -1 to +1
    topic_composite_signal: Optional[str] = None  # bullish/bearish/neutral
    topic_litigation_alert: bool = False  # Multiple negative litigation articles
    topic_management_alert: bool = False  # Negative management news
    topic_earnings_surprise: bool = False  # Strong earnings sentiment
    # Earnings topic model fields (strongest signal from backtest)
    earnings_topic_signal: Optional[str] = None  # bullish/bearish/neutral
    earnings_topic_score: Optional[float] = None  # Topic-adjusted sentiment
    earnings_topic_confidence: Optional[float] = None  # 0 to 1
    earnings_topic_tradeable: bool = False  # Is signal actionable?
    earnings_articles_count: int = 0  # Number of earnings-related articles
    earnings_expected_alpha: Optional[float] = None  # Expected 5d alpha
    sentiment_score_base: Optional[float] = None  # Pre-adjustment score
    topic_ic_score: Optional[float] = None  # IC-weighted topic score
    topic_ic_confidence: Optional[float] = None  # 0 to 1
    topic_ic_alpha: Optional[float] = None  # Blend weight
    topic_ic_applied: Optional[dict] = None  # Per-topic IC application details


@dataclass
class SentimentAgentConfig:
    """Configuration for the sentiment agent."""
    model: str = "gpt-4o-mini"  # Cheap and fast
    temperature: float = 0.1  # Low for consistent analysis
    max_news_items: int = 10  # Max news per ticker
    news_lookback_days: int = 7  # Days of news to consider
    timeout: int = 30  # API timeout in seconds
    use_finbert: bool = False  # Use FinBERT for baseline sentiment
    hybrid_mode: bool = False  # Use FinBERT + LLM hybrid
    finbert_only: bool = False  # Use only FinBERT (no LLM)
    use_defeatbeta: bool = True  # Use DefeatBeta for historical news (updated weekly)
    defeatbeta_lookback_days: int = 14  # How far back to look in DefeatBeta (skip last 2 days for freshness)
    use_scraped_db: bool = True  # Use our scraped Business Insider news database
    scraped_db_lookback_days: int = 30  # How far back to look in scraped DB
    use_topic_model: bool = True  # Use topic-based sentiment model
    use_topic_ic_adjustment: bool = True  # Adjust sentiment by topic ICs
    topic_ic_path: Optional[str] = "data/topic_ic.json"  # Optional JSON path to topic IC weights
    topic_ic_weights: dict[str, float] = field(default_factory=dict)  # topic -> IC
    topic_ic_min_abs: float = 0.01  # Ignore weak ICs
    topic_ic_alpha_max: float = 0.6  # Max blend weight for IC-adjusted score
    # RAG settings
    use_rag: bool = True  # Enable RAG retrieval from vector store
    rag_n_results: int = 10  # Number of articles to retrieve per ticker
    rag_lookback_days: int = 30  # How far back to look for articles
    rag_include_topic_ic: bool = True  # Include topic IC context in prompt


# ==============================================================================
# SENTIMENT AGENT
# ==============================================================================

class SentimentAgent:
    """
    Agent for analyzing news sentiment using an LLM, FinBERT, or hybrid approach.
    
    Modes:
    - LLM only: Uses LLM for sentiment + themes + explanations
    - FinBERT only: Fast, free, deterministic financial sentiment
    - Hybrid: FinBERT baseline + LLM validation (best of both worlds)
    
    The hybrid mode is recommended - it uses FinBERT for fast baseline
    and has the LLM review/validate, noting any disagreements.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        config: Optional[SentimentAgentConfig] = None,
        use_finbert: bool = False,
        hybrid_mode: bool = False,
        finbert_only: bool = False,
    ) -> None:
        """
        Initialize the Sentiment Agent.
        
        Args:
            model: LLM model to use (via litellm). Options:
                - "gpt-4o-mini" (OpenAI, cheap)
                - "gpt-4o" (OpenAI, better)
                - "claude-3-haiku-20240307" (Anthropic, cheap)
                - "claude-3-5-sonnet-20241022" (Anthropic, better)
            config: Optional configuration object.
            use_finbert: Add FinBERT scores alongside LLM.
            hybrid_mode: Use FinBERT + LLM validation (recommended).
            finbert_only: Use only FinBERT (no LLM calls).
        """
        self.config = config or SentimentAgentConfig(model=model)
        self.model = self.config.model
        
        # Mode flags (config overrides params)
        self.use_finbert = use_finbert or self.config.use_finbert
        self.hybrid_mode = hybrid_mode or self.config.hybrid_mode
        self.finbert_only = finbert_only or self.config.finbert_only
        
        # If hybrid or finbert_only, we need FinBERT
        if self.hybrid_mode or self.finbert_only:
            self.use_finbert = True
        
        # Initialize FinBERT or VADER fallback if needed
        self._finbert = None
        if self.use_finbert:
            try:
                from .finbert_sentiment import create_analyzer
                self._finbert = create_analyzer(prefer_finbert=True)
                logger.info(f"Loaded {type(self._finbert).__name__} for sentiment analysis")
            except ImportError as e:
                logger.warning(f"No sentiment analyzer available: {e}")
                self.use_finbert = False
                self.hybrid_mode = False
                self.finbert_only = False
        
        # Check LLM availability
        if not self.finbert_only and not HAS_LITELLM:
            raise ImportError("litellm is required. Install with: pip install litellm")
        
        # Initialize Topic Sentiment Model
        self._topic_model = None
        if self.config.use_topic_model:
            try:
                from ..models.topic_sentiment import TopicSentimentModel
                self._topic_model = TopicSentimentModel()
                logger.info("Loaded TopicSentimentModel for topic-based analysis")
            except ImportError as e:
                logger.warning(f"TopicSentimentModel not available: {e}")

        if self.config.use_topic_ic_adjustment and self.config.topic_ic_path and not self.config.topic_ic_weights:
            if os.path.exists(self.config.topic_ic_path):
                try:
                    with open(self.config.topic_ic_path, "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        self.config.topic_ic_weights = {str(k): float(v) for k, v in data.items()}
                except Exception as e:
                    logger.warning(f"Failed to load topic IC weights: {e}")
        
        # Initialize Earnings Topic Model (strongest signal from backtest)
        self._earnings_model = None
        try:
            from ..models.earnings_topic_model import EarningsTopicModel
            self._earnings_model = EarningsTopicModel()
            logger.info("Loaded EarningsTopicModel for earnings-focused analysis")
        except ImportError as e:
            logger.warning(f"EarningsTopicModel not available: {e}")
        
        # Initialize RAG vector store
        self._vectorstore = None
        if self.config.use_rag:
            try:
                from ..data.news_vectorstore import NewsVectorStore
                self._vectorstore = NewsVectorStore()
                count = self._vectorstore.get_index_count()
                if count > 0:
                    logger.info(f"Loaded NewsVectorStore with {count:,} articles for RAG")
                else:
                    logger.warning("NewsVectorStore is empty - run 'python -m auto_researcher.data.news_vectorstore build' first")
                    self._vectorstore = None
            except Exception as e:
                logger.warning(f"NewsVectorStore not available: {e}")
                self._vectorstore = None
        
        mode = "finbert_only" if self.finbert_only else "hybrid" if self.hybrid_mode else "llm"
        logger.info(f"Initialized SentimentAgent with mode: {mode}, model: {self.model}")
    
    @property
    def name(self) -> str:
        return "SentimentAgent"
    
    # ==========================================================================
    # NEWS FETCHING
    # ==========================================================================
    
    def fetch_news(
        self,
        ticker: str,
        max_items: Optional[int] = None,
    ) -> list[NewsItem]:
        """
        Fetch recent news for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            max_items: Maximum number of news items to return.
            
        Returns:
            List of NewsItem objects.
        """
        max_items = max_items or self.config.max_news_items
        
        if not HAS_YFINANCE:
            logger.warning("yfinance not available, returning empty news list")
            return []
        
        try:
            stock = yf.Ticker(ticker)
            news_data = stock.news
            
            if not news_data:
                logger.debug(f"No news found for {ticker}")
                return []
            
            news_items = []
            for item in news_data[:max_items]:
                # Parse the news item - handle nested 'content' structure
                try:
                    # New yfinance format has content nested
                    content = item.get("content", item)
                    
                    # Get title
                    title = content.get("title", "")
                    
                    # Get publisher/source
                    provider = content.get("provider", {})
                    source = provider.get("displayName", "Unknown") if isinstance(provider, dict) else "Unknown"
                    
                    # Get publish date
                    pub_date_str = content.get("pubDate", "")
                    if pub_date_str:
                        try:
                            # Parse ISO format: 2026-01-21T19:10:36Z
                            published = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        except:
                            published = datetime.now()
                    else:
                        # Fallback to old format
                        pub_time = item.get("providerPublishTime", 0)
                        if isinstance(pub_time, int) and pub_time > 0:
                            published = datetime.fromtimestamp(pub_time)
                        else:
                            published = datetime.now()
                    
                    # Get URL
                    canonical = content.get("canonicalUrl", {})
                    url = canonical.get("url", "") if isinstance(canonical, dict) else item.get("link", "")
                    
                    # Get summary/snippet
                    snippet = content.get("summary", content.get("description", None))
                    
                    if title:  # Only add if we have a title
                        news_item = NewsItem(
                            title=title,
                            source=source,
                            published=published,
                            url=url,
                            snippet=snippet,
                        )
                        news_items.append(news_item)
                except Exception as e:
                    logger.debug(f"Failed to parse news item: {e}")
                    continue
            
            logger.debug(f"Fetched {len(news_items)} news items for {ticker}")
            return news_items
            
        except Exception as e:
            logger.warning(f"Failed to fetch news for {ticker}: {e}")
            return []
    
    def fetch_defeatbeta_news(
        self,
        ticker: str,
        max_items: Optional[int] = None,
        lookback_days: Optional[int] = None,
    ) -> list[NewsItem]:
        """
        Fetch historical news from DefeatBeta (HuggingFace dataset).
        
        DefeatBeta is updated weekly, so this is best for historical context
        (last 7-30 days), not breaking news. Use yfinance for latest news.
        
        Args:
            ticker: Stock ticker symbol.
            max_items: Maximum number of news items to return.
            lookback_days: How many days back to look.
            
        Returns:
            List of NewsItem objects.
        """
        max_items = max_items or self.config.max_news_items
        lookback_days = lookback_days or self.config.defeatbeta_lookback_days
        
        try:
            from ..data.defeatbeta import DefeatBetaDataLoader
            
            df = DefeatBetaDataLoader.load("stock_news")
            if df is None:
                return []
            
            # Filter for ticker (check if ticker is in related_symbols array)
            mask = df['related_symbols'].apply(
                lambda x: ticker.upper() in str(x) if x is not None else False
            )
            ticker_df = df[mask].copy()
            
            if ticker_df.empty:
                logger.debug(f"No DefeatBeta news for {ticker}")
                return []
            
            # Filter by date - skip last 2 days (may not be updated yet)
            cutoff_recent = datetime.now() - timedelta(days=2)
            cutoff_old = datetime.now() - timedelta(days=lookback_days)
            
            ticker_df['report_date_dt'] = pd.to_datetime(ticker_df['report_date'], errors='coerce')
            ticker_df = ticker_df[
                (ticker_df['report_date_dt'] <= cutoff_recent) &
                (ticker_df['report_date_dt'] >= cutoff_old)
            ]
            
            # Sort by date descending
            ticker_df = ticker_df.sort_values('report_date_dt', ascending=False)
            
            # Convert to NewsItem objects
            news_items = []
            for _, row in ticker_df.head(max_items).iterrows():
                try:
                    # Extract full text from news paragraphs if available
                    snippet = None
                    news_content = row.get('news')
                    if news_content is not None and len(news_content) > 0:
                        paragraphs = []
                        for para in news_content[:3]:  # First 3 paragraphs
                            if isinstance(para, dict):
                                paragraphs.append(para.get('paragraph', ''))
                        snippet = ' '.join(paragraphs)[:500]  # Cap at 500 chars
                    
                    news_item = NewsItem(
                        title=row['title'] or "",
                        source=row['publisher'] or "Unknown",
                        published=row['report_date_dt'] if pd.notna(row['report_date_dt']) else datetime.now(),
                        url=row.get('link', ''),
                        snippet=snippet,
                    )
                    if news_item.title:
                        news_items.append(news_item)
                except Exception as e:
                    logger.debug(f"Failed to parse DefeatBeta news item: {e}")
                    continue
            
            logger.debug(f"Fetched {len(news_items)} DefeatBeta news items for {ticker}")
            return news_items
            
        except ImportError:
            logger.debug("DefeatBeta module not available")
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch DefeatBeta news for {ticker}: {e}")
            return []
    
    def fetch_scraped_news(
        self,
        ticker: str,
        max_items: Optional[int] = None,
        lookback_days: Optional[int] = None,
    ) -> list[NewsItem]:
        """
        Fetch news from our scraped Business Insider database.
        
        This database contains years of historical news scraped from 
        Business Insider, providing deep historical context.
        
        Args:
            ticker: Stock ticker symbol.
            max_items: Maximum number of news items to return.
            lookback_days: How many days back to look.
            
        Returns:
            List of NewsItem objects.
        """
        max_items = max_items or self.config.max_news_items
        lookback_days = lookback_days or self.config.scraped_db_lookback_days
        
        try:
            from ..data.news_scraper import NewsDatabase
            
            db = NewsDatabase()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Fetch from database
            articles = db.get_articles(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                limit=max_items * 2,  # Get extra for deduplication
            )
            
            if not articles:
                logger.debug(f"No scraped news for {ticker}")
                return []
            
            # Convert to NewsItem objects
            news_items = []
            for art in articles:
                try:
                    # Parse published date
                    pub_date_str = art.get('published_date')
                    if pub_date_str:
                        try:
                            published = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        except:
                            published = datetime.now()
                    else:
                        published = datetime.now()
                    
                    news_item = NewsItem(
                        title=art.get('title', ''),
                        source=art.get('source', 'Business Insider'),
                        published=published,
                        url=art.get('url', ''),
                        snippet=art.get('snippet'),
                    )
                    if news_item.title:
                        news_items.append(news_item)
                except Exception as e:
                    logger.debug(f"Failed to parse scraped news item: {e}")
                    continue
            
            logger.debug(f"Fetched {len(news_items)} scraped news items for {ticker}")
            return news_items[:max_items]
            
        except ImportError:
            logger.debug("News scraper module not available")
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch scraped news for {ticker}: {e}")
            return []
    
    def fetch_all_news(
        self,
        ticker: str,
        max_items: Optional[int] = None,
    ) -> list[NewsItem]:
        """
        Fetch news from all sources (yfinance + DefeatBeta + scraped DB).
        
        Strategy:
        - yfinance: Most recent news (last 1-2 days) - always fresh
        - Scraped DB: Historical context (up to 30 days) - deep coverage
        - DefeatBeta: Backup historical (3-14 days ago) - updated weekly
        
        Returns combined list sorted by date.
        """
        max_items = max_items or self.config.max_news_items
        all_news = []
        
        # Get recent news from yfinance (always fresh)
        yf_news = self.fetch_news(ticker, max_items=max_items // 2)
        all_news.extend(yf_news)
        
        # Get historical news from our scraped database (deep coverage)
        if self.config.use_scraped_db:
            scraped_news = self.fetch_scraped_news(ticker, max_items=max_items)
            all_news.extend(scraped_news)
        
        # Get historical news from DefeatBeta if enabled (backup source)
        if self.config.use_defeatbeta:
            db_news = self.fetch_defeatbeta_news(ticker, max_items=max_items // 2)
            all_news.extend(db_news)
        
        # Deduplicate by title similarity (remove exact matches)
        seen_titles = set()
        unique_news = []
        for item in all_news:
            title_key = item.title.lower()[:50]  # First 50 chars as key
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(item)
        
        # Sort by date descending (normalize to naive datetime for comparison)
        def get_timestamp(item):
            dt = item.published
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt
        unique_news.sort(key=get_timestamp, reverse=True)
        
        # Limit to max_items
        return unique_news[:max_items * 2]  # Return more since we have richer data
    
    # ==========================================================================
    # LLM ANALYSIS
    # ==========================================================================
    
    def _build_prompt(self, ticker: str, news_items: list[NewsItem]) -> str:
        """Build the analysis prompt for the LLM."""
        
        news_text = "\n".join([
            f"- [{item.source}] {item.title}"
            for item in news_items
        ])
        
        # Retrieve articles from vector store for grounded analysis
        rag_context = ""
        if self._vectorstore:
            try:
                rag_context = self._vectorstore.format_context_for_prompt(
                    ticker=ticker,
                    n_results=self.config.rag_n_results,
                    lookback_days=self.config.rag_lookback_days,
                    include_topic_ic=self.config.rag_include_topic_ic,
                )
            except Exception as e:
                logger.debug(f"RAG retrieval failed for {ticker}: {e}")
        
        prompt = f"""Analyze the sentiment of recent news for {ticker} stock.

RECENT NEWS HEADLINES:
{news_text}
"""
        
        if rag_context:
            prompt += f"""
HISTORICAL NEWS CONTEXT (retrieved from database):
{rag_context}
"""
        
        prompt += """
Provide your analysis in the following exact format:

SENTIMENT_SCORE: [number from -1.0 to 1.0, where -1 is very bearish, 0 is neutral, 1 is very bullish]
CONFIDENCE: [number from 0.0 to 1.0 indicating how confident you are]
LABEL: [one of: very_bearish, bearish, neutral, bullish, very_bullish]
SUMMARY: [one sentence summarizing the overall sentiment and why]
THEMES: [comma-separated list of 2-4 key themes from the news]

Be objective and base your analysis on the headlines and retrieved articles provided.
If articles relate to high-IC topics (earnings, M&A, analyst), weight those more heavily.
If headlines are mixed, reflect that in a more neutral score."""

        return prompt
    
    def _parse_response(self, ticker: str, response: str, news_count: int) -> SentimentResult:
        """Parse the LLM response into a SentimentResult."""
        
        # Default values
        score = 0.0
        confidence = 0.5
        label = "neutral"
        summary = "Unable to parse sentiment."
        themes = []
        
        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("SENTIMENT_SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(-1.0, min(1.0, score))
                    except:
                        pass
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except:
                        pass
                elif line.startswith("LABEL:"):
                    label = line.split(":", 1)[1].strip().lower()
                    if label not in ["very_bearish", "bearish", "neutral", "bullish", "very_bullish"]:
                        label = "neutral"
                elif line.startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.startswith("THEMES:"):
                    themes_str = line.split(":", 1)[1].strip()
                    themes = [t.strip() for t in themes_str.split(",") if t.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        return SentimentResult(
            ticker=ticker,
            sentiment_score=score,
            sentiment_label=label,
            confidence=confidence,
            summary=summary,
            key_themes=themes,
            news_count=news_count,
            analysis_date=datetime.now(),
            raw_response=response,
        )
    
    def _run_finbert_analysis(self, ticker: str, news_items: list[NewsItem]) -> tuple[float, str]:
        """Run FinBERT analysis on news items."""
        if not self._finbert or not news_items:
            return 0.0, "neutral"
        
        # Build texts from news
        texts = []
        for item in news_items:
            text = item.title
            if item.snippet:
                text += ". " + item.snippet
            texts.append(text)
        
        # Run FinBERT
        results = self._finbert.analyze_batch(texts)
        agg = self._finbert.get_aggregate_sentiment(results)
        
        # Map to label
        score = agg["mean_sentiment"]
        if score > 0.3:
            label = "positive"
        elif score < -0.3:
            label = "negative"
        else:
            label = "neutral"
        
        return score, label
    
    def _run_topic_analysis(self, ticker: str, news_items: list[NewsItem]) -> dict:
        """
        Run topic-based sentiment analysis on news items.
        
        Returns dict with topic analysis results to add to SentimentResult.
        """
        if not self._topic_model or not news_items:
            return {}
        
        try:
            # Convert NewsItem to dicts for topic model
            articles = []
            for item in news_items:
                text = item.title
                if item.snippet:
                    text += ". " + item.snippet
                articles.append({
                    "title": text,
                    "published_date": item.published,
                })
            
            # Analyze with topic model
            signal = self._topic_model.analyze_articles(articles, ticker, text_field="title")
            
            results = {
                "topic_sentiment": signal.topic_sentiment,
                "topic_counts": signal.topic_counts,
                "topic_signals": signal.topic_signals,
                "topic_composite_score": signal.composite_score,
                "topic_composite_signal": signal.composite_signal,
                "topic_litigation_alert": signal.litigation_alert,
                "topic_management_alert": signal.management_alert,
                "topic_earnings_surprise": signal.earnings_surprise_detected,
            }
            
            # Also run earnings-focused model (strongest signal from backtest)
            if self._earnings_model:
                earnings_signal = self._earnings_model.analyze_news(articles, ticker, text_field="title")
                results.update({
                    "earnings_topic_signal": earnings_signal.direction,
                    "earnings_topic_score": earnings_signal.topic_sentiment,
                    "earnings_topic_confidence": earnings_signal.confidence,
                    "earnings_topic_tradeable": earnings_signal.is_tradeable,
                    "earnings_articles_count": earnings_signal.earnings_articles,
                    "earnings_expected_alpha": earnings_signal.expected_alpha,
                })
            
            return results
        except Exception as e:
            logger.warning(f"Topic analysis failed for {ticker}: {e}")
            return {}

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score > 0.3:
            return "bullish"
        if score > 0.1:
            return "neutral"
        if score > -0.1:
            return "neutral"
        if score > -0.3:
            return "bearish"
        return "very_bearish"

    def _compute_topic_ic_score(
        self,
        topic_sentiment: dict,
        topic_counts: dict,
    ) -> tuple[float, float, dict]:
        if not topic_sentiment or not self.config.topic_ic_weights:
            return 0.0, 0.0, {}

        total_weight = 0.0
        weighted = 0.0
        applied: dict[str, dict] = {}

        for topic, sent in topic_sentiment.items():
            if topic not in self.config.topic_ic_weights:
                continue
            ic = float(self.config.topic_ic_weights.get(topic, 0.0))
            if abs(ic) < self.config.topic_ic_min_abs:
                continue
            count = float(topic_counts.get(topic, 1) or 1)
            weight = abs(ic) * math.sqrt(count)
            adjusted = sent * (1 if ic >= 0 else -1)
            weighted += adjusted * weight
            total_weight += weight
            applied[topic] = {
                "ic": round(ic, 4),
                "weight": round(weight, 4),
                "sentiment": round(float(sent), 4),
                "adjusted": round(float(adjusted), 4),
            }

        if total_weight <= 0:
            return 0.0, 0.0, applied

        score = weighted / total_weight
        confidence = min(1.0, total_weight / 0.3)
        return float(score), float(confidence), applied

    def _apply_topic_ic_adjustment(
        self,
        result: SentimentResult,
        topic_results: dict,
    ) -> None:
        if not self.config.use_topic_ic_adjustment or not topic_results:
            return

        topic_sentiment = topic_results.get("topic_sentiment") or {}
        topic_counts = topic_results.get("topic_counts") or {}
        ic_score, ic_conf, applied = self._compute_topic_ic_score(topic_sentiment, topic_counts)
        if ic_conf <= 0:
            return

        base_score = result.sentiment_score
        alpha = min(self.config.topic_ic_alpha_max, 0.2 + 0.4 * ic_conf)
        blended = (1 - alpha) * base_score + alpha * ic_score

        result.sentiment_score_base = base_score
        result.sentiment_score = max(-1.0, min(1.0, blended))
        result.sentiment_label = self._label_from_score(result.sentiment_score)
        result.confidence = max(result.confidence, ic_conf * 0.8)
        result.topic_ic_score = ic_score
        result.topic_ic_confidence = ic_conf
        result.topic_ic_alpha = alpha
        result.topic_ic_applied = applied
    
    def _build_hybrid_prompt(
        self,
        ticker: str,
        news_items: list[NewsItem],
        finbert_score: float,
        finbert_label: str,
    ) -> str:
        """Build prompt for hybrid mode with FinBERT context."""
        
        # Format news
        news_text = ""
        for i, item in enumerate(news_items, 1):
            news_text += f"\n{i}. {item.title}"
            if item.snippet:
                news_text += f"\n   {item.snippet}"
            news_text += f"\n   Source: {item.source} | {item.published.strftime('%Y-%m-%d')}"
        
        # Retrieve articles from vector store for grounded analysis
        rag_context = ""
        if self._vectorstore:
            try:
                rag_context = self._vectorstore.format_context_for_prompt(
                    ticker=ticker,
                    n_results=self.config.rag_n_results,
                    lookback_days=self.config.rag_lookback_days,
                    include_topic_ic=self.config.rag_include_topic_ic,
                )
            except Exception as e:
                logger.debug(f"RAG retrieval failed for {ticker}: {e}")
        
        prompt = f"""Analyze the sentiment for {ticker} stock based on these news items:

{news_text}

A FinBERT model (specialized for financial text) scored the sentiment as:
- FinBERT Score: {finbert_score:+.2f} (scale: -1 bearish to +1 bullish)
- FinBERT Label: {finbert_label}
"""
        
        if rag_context:
            prompt += f"""
HISTORICAL NEWS CONTEXT (retrieved from database):
{rag_context}
"""
        
        prompt += """
Review the news, FinBERT assessment, and historical context. Provide your analysis:

1. Do you AGREE or DISAGREE with FinBERT's assessment? Why?
2. What nuances might FinBERT miss?
3. What are the key themes?
4. How do high-IC topics (earnings, M&A, analyst coverage) affect the outlook?

Respond in this EXACT format:
SENTIMENT_SCORE: [your score from -1.0 to +1.0]
SENTIMENT_LABEL: [very_bearish, bearish, neutral, bullish, or very_bullish]
CONFIDENCE: [0.0 to 1.0 confidence in your assessment]
AGREES_WITH_FINBERT: [yes or no]
AGREEMENT_REASON: [brief explanation of agreement/disagreement]
KEY_THEMES: [comma-separated list of 3-5 themes]
SUMMARY: [one sentence summary of overall sentiment]

Focus on what matters for investment decisions. Weight high-IC topics more heavily."""

        return prompt
    
    def _parse_hybrid_response(
        self,
        ticker: str,
        response: str,
        news_count: int,
        finbert_score: float,
        finbert_label: str,
    ) -> SentimentResult:
        """Parse LLM response for hybrid mode."""
        
        # Defaults
        sentiment_score = 0.0
        sentiment_label = "neutral"
        confidence = 0.5
        summary = "Analysis not available."
        key_themes = []
        agrees_with_finbert = True
        
        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("SENTIMENT_SCORE:"):
                    try:
                        sentiment_score = float(line.split(":", 1)[1].strip())
                        sentiment_score = max(-1.0, min(1.0, sentiment_score))
                    except:
                        pass
                elif line.startswith("SENTIMENT_LABEL:"):
                    label = line.split(":", 1)[1].strip().lower()
                    if label in ["very_bearish", "bearish", "neutral", "bullish", "very_bullish"]:
                        sentiment_label = label
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except:
                        pass
                elif line.startswith("AGREES_WITH_FINBERT:"):
                    agrees_str = line.split(":", 1)[1].strip().lower()
                    agrees_with_finbert = agrees_str in ["yes", "true", "agree"]
                elif line.startswith("KEY_THEMES:"):
                    themes_str = line.split(":", 1)[1].strip()
                    key_themes = [t.strip() for t in themes_str.split(",") if t.strip()]
                elif line.startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
        except Exception as e:
            logger.warning(f"Failed to parse hybrid response: {e}")
        
        return SentimentResult(
            ticker=ticker,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            summary=summary,
            key_themes=key_themes,
            news_count=news_count,
            analysis_date=datetime.now(),
            raw_response=response,
            finbert_score=finbert_score,
            finbert_label=finbert_label,
            llm_agrees_with_finbert=agrees_with_finbert,
            method="hybrid",
        )
    
    def analyze_ticker(self, ticker: str) -> SentimentResult:
        """
        Analyze sentiment for a single ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            SentimentResult with sentiment analysis.
        """
        # Fetch news from all sources (yfinance + DefeatBeta)
        news_items = self.fetch_all_news(ticker)
        
        # Add freshly fetched articles to vector store for RAG retrieval
        if self._vectorstore and news_items:
            try:
                live_articles = [
                    {
                        "title": item.title,
                        "snippet": item.snippet or "",
                        "source": item.source,
                        "published": item.published,
                        "url": item.url,
                    }
                    for item in news_items
                ]
                added = self._vectorstore.add_articles(live_articles, ticker=ticker)
                if added > 0:
                    logger.info(f"Indexed {added} live articles for {ticker} into vector store")
            except Exception as e:
                logger.debug(f"Failed to index live articles for {ticker}: {e}")
        
        # Run topic analysis if enabled (runs in parallel with other analysis)
        topic_results = self._run_topic_analysis(ticker, news_items) if news_items else {}
        
        if not news_items:
            return SentimentResult(
                ticker=ticker,
                sentiment_score=0.0,
                sentiment_label="neutral",
                confidence=0.0,
                summary="No recent news found.",
                key_themes=[],
                news_count=0,
                analysis_date=datetime.now(),
                method="none",
            )
        
        # FinBERT-only mode
        if self.finbert_only:
            finbert_score, finbert_label = self._run_finbert_analysis(ticker, news_items)
            
            # Map FinBERT label to our labels
            if finbert_score > 0.3:
                sentiment_label = "bullish"
            elif finbert_score > 0.1:
                sentiment_label = "neutral"
            elif finbert_score > -0.1:
                sentiment_label = "neutral"
            elif finbert_score > -0.3:
                sentiment_label = "bearish"
            else:
                sentiment_label = "very_bearish"
            
            result = SentimentResult(
                ticker=ticker,
                sentiment_score=finbert_score,
                sentiment_label=sentiment_label,
                confidence=abs(finbert_score),
                summary=f"FinBERT analysis: {finbert_label} sentiment based on {len(news_items)} news items.",
                key_themes=[],
                news_count=len(news_items),
                analysis_date=datetime.now(),
                finbert_score=finbert_score,
                finbert_label=finbert_label,
                method="finbert",
                **topic_results,
            )
            if topic_results:
                self._apply_topic_ic_adjustment(result, topic_results)
            return result
        
        # Get FinBERT baseline if using hybrid mode
        finbert_score = None
        finbert_label = None
        if self.hybrid_mode or self.use_finbert:
            finbert_score, finbert_label = self._run_finbert_analysis(ticker, news_items)
        
        # Build prompt (hybrid or standard)
        if self.hybrid_mode and finbert_score is not None:
            prompt = self._build_hybrid_prompt(ticker, news_items, finbert_score, finbert_label)
        else:
            prompt = self._build_prompt(ticker, news_items)
        
        # Call LLM
        try:
            system_msg = (
                "You are a financial analyst specializing in stock sentiment analysis. "
                "Be objective and concise. When provided with historical article context "
                "and topic IC (Information Coefficient) data, use it to ground your analysis. "
                "Topics with higher absolute IC are more predictive of future stock returns. "
                "Pay special attention to earnings (IC=+0.020), M&A (IC=-0.027, contrarian), "
                "and analyst coverage (IC=+0.012) topics."
            )
            
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            
            response_text = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM call failed for {ticker}: {e}")
            # Fall back to FinBERT if available
            if finbert_score is not None:
                result = SentimentResult(
                    ticker=ticker,
                    sentiment_score=finbert_score,
                    sentiment_label="bullish" if finbert_score > 0.2 else "bearish" if finbert_score < -0.2 else "neutral",
                    confidence=abs(finbert_score),
                    summary=f"FinBERT fallback (LLM failed): {finbert_label}",
                    key_themes=[],
                    news_count=len(news_items),
                    analysis_date=datetime.now(),
                    finbert_score=finbert_score,
                    finbert_label=finbert_label,
                    method="finbert_fallback",
                )
                if topic_results:
                    self._apply_topic_ic_adjustment(result, topic_results)
                return result
            return SentimentResult(
                ticker=ticker,
                sentiment_score=0.0,
                sentiment_label="neutral",
                confidence=0.0,
                summary=f"Analysis failed: {str(e)[:50]}",
                key_themes=[],
                news_count=len(news_items),
                analysis_date=datetime.now(),
            )
        
        # Parse response
        if self.hybrid_mode and finbert_score is not None:
            result = self._parse_hybrid_response(ticker, response_text, len(news_items), finbert_score, finbert_label)
        else:
            result = self._parse_response(ticker, response_text, len(news_items))
            # Add FinBERT scores if we ran it
            if finbert_score is not None:
                result.finbert_score = finbert_score
                result.finbert_label = finbert_label
        
        # Add topic analysis results
        if topic_results:
            result.topic_sentiment = topic_results.get("topic_sentiment")
            result.topic_counts = topic_results.get("topic_counts")
            result.topic_signals = topic_results.get("topic_signals")
            result.topic_composite_score = topic_results.get("topic_composite_score")
            result.topic_composite_signal = topic_results.get("topic_composite_signal")
            result.topic_litigation_alert = topic_results.get("topic_litigation_alert", False)
            result.topic_management_alert = topic_results.get("topic_management_alert", False)
            result.topic_earnings_surprise = topic_results.get("topic_earnings_surprise", False)
            # Earnings topic model (strongest signal)
            result.earnings_topic_signal = topic_results.get("earnings_topic_signal")
            result.earnings_topic_score = topic_results.get("earnings_topic_score")
            result.earnings_topic_confidence = topic_results.get("earnings_topic_confidence")
            result.earnings_topic_tradeable = topic_results.get("earnings_topic_tradeable", False)
            result.earnings_articles_count = topic_results.get("earnings_articles_count", 0)
            result.earnings_expected_alpha = topic_results.get("earnings_expected_alpha")

            self._apply_topic_ic_adjustment(result, topic_results)
        
        return result
    
    def analyze_tickers(
        self,
        tickers: list[str],
        parallel: bool = False,
    ) -> dict[str, SentimentResult]:
        """
        Analyze sentiment for multiple tickers.
        
        Args:
            tickers: List of ticker symbols.
            parallel: Whether to run in parallel (not yet implemented).
            
        Returns:
            Dictionary mapping ticker to SentimentResult.
        """
        results = {}
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Analyzing {ticker} ({i}/{len(tickers)})...")
            try:
                result = self.analyze_ticker(ticker)
                results[ticker] = result
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
                results[ticker] = SentimentResult(
                    ticker=ticker,
                    sentiment_score=0.0,
                    sentiment_label="neutral",
                    confidence=0.0,
                    summary=f"Analysis failed: {str(e)[:50]}",
                    key_themes=[],
                    news_count=0,
                    analysis_date=datetime.now(),
                )
        
        return results
    
    # ==========================================================================
    # OUTPUT FORMATTING
    # ==========================================================================
    
    def format_results(self, results: dict[str, SentimentResult]) -> str:
        """Format results as a readable string."""
        
        lines = []
        lines.append("=" * 70)
        lines.append("  SENTIMENT ANALYSIS RESULTS")
        lines.append("=" * 70)
        lines.append("")
        
        # Sort by sentiment score (most bullish first)
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.sentiment_score,
            reverse=True,
        )
        
        lines.append(f"{'Ticker':<8} {'Score':>7} {'Label':<14} {'Conf':>5}   Summary")
        lines.append("-" * 70)
        
        for r in sorted_results:
            score_str = f"{r.sentiment_score:+.2f}"
            conf_str = f"{r.confidence:.0%}"
            summary = r.summary[:40] + "..." if len(r.summary) > 43 else r.summary
            lines.append(f"{r.ticker:<8} {score_str:>7} {r.sentiment_label:<14} {conf_str:>5}   {summary}")
        
        lines.append("-" * 70)
        lines.append("")
        
        # Detailed view
        lines.append("DETAILED ANALYSIS:")
        lines.append("")
        
        for r in sorted_results:
            emoji = "🟢" if r.sentiment_score > 0.2 else "🔴" if r.sentiment_score < -0.2 else "🟡"
            lines.append(f"{emoji} {r.ticker} ({r.sentiment_label}, score: {r.sentiment_score:+.2f})")
            lines.append(f"   {r.summary}")
            if r.key_themes:
                lines.append(f"   Themes: {', '.join(r.key_themes)}")
            lines.append(f"   News items analyzed: {r.news_count}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dataframe(self, results: dict[str, SentimentResult]) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        
        data = []
        for ticker, r in results.items():
            data.append({
                "ticker": ticker,
                "sentiment_score": r.sentiment_score,
                "sentiment_label": r.sentiment_label,
                "confidence": r.confidence,
                "summary": r.summary,
                "themes": ", ".join(r.key_themes),
                "news_count": r.news_count,
                "analysis_date": r.analysis_date,
            })
        
        return pd.DataFrame(data)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def analyze_sentiment(
    tickers: list[str],
    model: str = "gpt-4o-mini",
) -> dict[str, SentimentResult]:
    """
    Convenience function to analyze sentiment for tickers.
    
    Args:
        tickers: List of ticker symbols.
        model: LLM model to use.
        
    Returns:
        Dictionary mapping ticker to SentimentResult.
    """
    agent = SentimentAgent(model=model)
    return agent.analyze_tickers(tickers)


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze stock sentiment")
    parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbols to analyze",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    
    args = parser.parse_args()
    
    # Run analysis
    agent = SentimentAgent(model=args.model)
    results = agent.analyze_tickers(args.tickers)
    
    # Print formatted results
    print(agent.format_results(results))
