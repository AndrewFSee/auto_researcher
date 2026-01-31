"""
Earnings-Focused Topic Sentiment Model.

Based on backtesting that found earnings-related news sentiment has 
significantly stronger predictive power than generic sentiment.

================================================================================
EMPIRICAL FINDINGS (from backtest)
================================================================================
    - Earnings sentiment IC = +0.02 at 10-day horizon (p < 0.001)
    - Strong positive (>0.2) vs negative (<-0.2): +0.65% spread (p=0.0003)
    - Multi-article days: IC improves 26% 
    - When Topic + FinBERT agree: IC = 0.038 (p=0.003)
    - When they disagree: IC ≈ 0

SIGNAL RULES:
    1. Filter to earnings-related news only
    2. Use topic-adjusted sentiment (not raw FinBERT)
    3. Weight by article count (more articles = higher conviction)
    4. Combine with FinBERT for agreement signal
    5. Focus on extreme signals (±0.2 threshold)

================================================================================

Usage:
    from auto_researcher.models.earnings_topic_model import EarningsTopicModel
    
    model = EarningsTopicModel()
    signal = model.analyze_news(articles, ticker="AAPL")
    
    if signal.is_tradeable:
        print(f"Signal: {signal.direction} with confidence {signal.confidence}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np

from .topic_sentiment import TopicSentimentModel, TopicSentiment

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

EARNINGS_MODEL_CONFIG = {
    # Topics considered earnings-related
    "earnings_topics": ["earnings", "analyst"],
    
    # Sentiment thresholds
    "strong_positive_threshold": 0.2,
    "strong_negative_threshold": -0.2,
    
    # Minimum articles for tradeable signal
    "min_articles_for_signal": 1,
    "high_conviction_articles": 2,  # 26% IC improvement with 2+ articles
    
    # Expected returns (from backtest)
    "expected_return_strong_positive": 0.0068,  # +0.68% 5-day
    "expected_return_strong_negative": 0.0003,  # +0.03% 5-day
    "expected_return_neutral": 0.0038,  # +0.38% 5-day (base rate)
    
    # Agreement bonus
    "agreement_ic_multiplier": 1.4,  # When topic + finbert agree, signal is stronger
    
    # Signal decay
    "signal_half_life_days": 5,  # Signal is strongest immediately
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class EarningsTopicSignal:
    """Signal from earnings-focused topic sentiment analysis."""
    ticker: str
    
    # Raw inputs
    total_articles: int
    earnings_articles: int
    
    # Sentiment
    topic_sentiment: float  # -1 to +1 (topic-adjusted)
    finbert_sentiment: Optional[float] = None  # Raw FinBERT
    
    # Agreement analysis
    models_agree: Optional[bool] = None
    agreement_confidence_boost: float = 1.0
    
    # Signal
    is_tradeable: bool = False
    direction: str = "neutral"  # "bullish", "bearish", "neutral"
    raw_score: float = 0.0  # Unbounded score
    confidence: float = 0.0  # 0 to 1
    
    # Expected outcome
    expected_5d_return: float = 0.0
    expected_alpha: float = 0.0  # vs base rate
    
    # Context
    high_conviction: bool = False  # 2+ articles
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    # Alerts
    earnings_surprise: bool = False  # Extreme sentiment detected


@dataclass
class EarningsArticleResult:
    """Analysis result for a single article."""
    text: str
    is_earnings_related: bool
    topic: str
    topic_sentiment: float
    keywords_matched: List[str]


# ==============================================================================
# EARNINGS TOPIC MODEL
# ==============================================================================

class EarningsTopicModel:
    """
    Earnings-focused topic sentiment model.
    
    Based on empirical finding that earnings-related news sentiment
    has 3-4x the predictive power of generic sentiment.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        include_analyst: bool = True,
    ):
        """
        Initialize the earnings topic model.
        
        Args:
            config: Custom configuration (uses defaults if None)
            include_analyst: Whether to include analyst-related news as earnings
        """
        self.config = config or EARNINGS_MODEL_CONFIG.copy()
        self._topic_model = TopicSentimentModel()
        
        # Topics to consider earnings-related
        self.earnings_topics = set(self.config.get("earnings_topics", ["earnings"]))
        if include_analyst:
            self.earnings_topics.add("analyst")
        
        logger.info(f"EarningsTopicModel initialized. Earnings topics: {self.earnings_topics}")
    
    def analyze_article(self, text: str) -> EarningsArticleResult:
        """
        Analyze a single article.
        
        Args:
            text: Article text or headline
            
        Returns:
            EarningsArticleResult
        """
        result = self._topic_model.analyze_article(text)
        
        is_earnings = result.topic.primary_topic in self.earnings_topics
        
        return EarningsArticleResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            is_earnings_related=is_earnings,
            topic=result.topic.primary_topic,
            topic_sentiment=result.topic_adjusted_sentiment,
            keywords_matched=result.topic.keywords_matched.get(result.topic.primary_topic, []),
        )
    
    def analyze_news(
        self,
        articles: List[Dict],
        ticker: str,
        text_field: str = "title",
        finbert_field: Optional[str] = "sentiment_score",
    ) -> EarningsTopicSignal:
        """
        Analyze a batch of news articles for earnings sentiment.
        
        Args:
            articles: List of article dicts
            ticker: Stock ticker
            text_field: Field containing article text
            finbert_field: Field containing pre-computed FinBERT score (optional)
            
        Returns:
            EarningsTopicSignal with trading signal
        """
        if not articles:
            return EarningsTopicSignal(
                ticker=ticker,
                total_articles=0,
                earnings_articles=0,
                topic_sentiment=0.0,
            )
        
        # Analyze each article
        earnings_sentiments = []
        finbert_scores = []
        
        for article in articles:
            text = article.get(text_field, "") or ""
            if not text:
                continue
            
            result = self.analyze_article(text)
            
            if result.is_earnings_related:
                earnings_sentiments.append(result.topic_sentiment)
                
                # Get FinBERT if available
                if finbert_field and finbert_field in article:
                    fb = article[finbert_field]
                    if fb is not None:
                        finbert_scores.append(fb)
        
        n_earnings = len(earnings_sentiments)
        
        if n_earnings == 0:
            return EarningsTopicSignal(
                ticker=ticker,
                total_articles=len(articles),
                earnings_articles=0,
                topic_sentiment=0.0,
            )
        
        # Aggregate sentiment
        topic_sentiment = np.mean(earnings_sentiments)
        finbert_sentiment = np.mean(finbert_scores) if finbert_scores else None
        
        # Agreement analysis
        models_agree = None
        agreement_boost = 1.0
        if finbert_sentiment is not None:
            # Check if they agree on direction
            topic_dir = "pos" if topic_sentiment > 0.05 else "neg" if topic_sentiment < -0.05 else "neutral"
            fb_dir = "pos" if finbert_sentiment > 0.05 else "neg" if finbert_sentiment < -0.05 else "neutral"
            
            models_agree = (topic_dir == fb_dir) or (topic_dir == "neutral" or fb_dir == "neutral")
            
            # When they agree, IC is 40% higher
            if topic_dir == fb_dir and topic_dir != "neutral":
                agreement_boost = self.config.get("agreement_ic_multiplier", 1.4)
        
        # Compute signal
        pos_threshold = self.config.get("strong_positive_threshold", 0.2)
        neg_threshold = self.config.get("strong_negative_threshold", -0.2)
        min_articles = self.config.get("min_articles_for_signal", 1)
        
        is_tradeable = n_earnings >= min_articles
        high_conviction = n_earnings >= self.config.get("high_conviction_articles", 2)
        
        # Direction
        if topic_sentiment > pos_threshold:
            direction = "bullish"
        elif topic_sentiment < neg_threshold:
            direction = "bearish"
        else:
            direction = "neutral"
            is_tradeable = False  # Only trade extreme signals
        
        # Confidence (0 to 1)
        # Based on: sentiment magnitude, article count, agreement
        sentiment_factor = min(abs(topic_sentiment) / 0.5, 1.0)
        count_factor = min(n_earnings / 3, 1.0)  # Saturates at 3 articles
        
        confidence = sentiment_factor * 0.5 + count_factor * 0.3 + (0.2 if models_agree else 0.0)
        confidence = confidence * agreement_boost
        confidence = min(confidence, 1.0)
        
        # Expected returns (from backtest)
        base_return = self.config.get("expected_return_neutral", 0.0038)
        if direction == "bullish":
            expected_5d = self.config.get("expected_return_strong_positive", 0.0068)
        elif direction == "bearish":
            expected_5d = self.config.get("expected_return_strong_negative", 0.0003)
        else:
            expected_5d = base_return
        
        expected_alpha = expected_5d - base_return
        
        # Earnings surprise detection
        earnings_surprise = abs(topic_sentiment) > 0.4 and n_earnings >= 2
        
        # Raw score (unbounded, for ranking)
        raw_score = topic_sentiment * np.sqrt(n_earnings) * agreement_boost
        
        return EarningsTopicSignal(
            ticker=ticker,
            total_articles=len(articles),
            earnings_articles=n_earnings,
            topic_sentiment=topic_sentiment,
            finbert_sentiment=finbert_sentiment,
            models_agree=models_agree,
            agreement_confidence_boost=agreement_boost,
            is_tradeable=is_tradeable,
            direction=direction,
            raw_score=raw_score,
            confidence=confidence,
            expected_5d_return=expected_5d,
            expected_alpha=expected_alpha,
            high_conviction=high_conviction,
            earnings_surprise=earnings_surprise,
        )
    
    def get_summary(self, signal: EarningsTopicSignal) -> str:
        """Generate human-readable summary."""
        lines = [f"Earnings Topic Signal for {signal.ticker}"]
        lines.append("=" * 50)
        
        lines.append(f"Articles: {signal.earnings_articles} earnings / {signal.total_articles} total")
        lines.append(f"Topic Sentiment: {signal.topic_sentiment:+.3f}")
        
        if signal.finbert_sentiment is not None:
            lines.append(f"FinBERT Sentiment: {signal.finbert_sentiment:+.3f}")
            lines.append(f"Models Agree: {'Yes' if signal.models_agree else 'No'}")
        
        lines.append("")
        lines.append(f"SIGNAL: {signal.direction.upper()}")
        lines.append(f"Tradeable: {'Yes' if signal.is_tradeable else 'No'}")
        lines.append(f"Confidence: {signal.confidence:.0%}")
        lines.append(f"High Conviction: {'Yes' if signal.high_conviction else 'No'}")
        
        if signal.is_tradeable:
            lines.append("")
            lines.append(f"Expected 5d Return: {signal.expected_5d_return:.2%}")
            lines.append(f"Expected Alpha: {signal.expected_alpha:+.2%}")
        
        if signal.earnings_surprise:
            lines.append("")
            lines.append("⚠️  EARNINGS SURPRISE: Extreme sentiment detected")
        
        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_earnings_signal(
    articles: List[Dict],
    ticker: str,
    text_field: str = "title",
) -> EarningsTopicSignal:
    """Quick function to get earnings topic signal."""
    model = EarningsTopicModel()
    return model.analyze_news(articles, ticker, text_field=text_field)


if __name__ == "__main__":
    # Demo
    model = EarningsTopicModel()
    
    test_articles = [
        {"title": "Apple beats Q4 earnings expectations, revenue up 12%", "sentiment_score": 0.8},
        {"title": "Analysts raise Apple price targets after strong earnings", "sentiment_score": 0.6},
        {"title": "Apple guidance exceeds Wall Street estimates", "sentiment_score": 0.7},
    ]
    
    signal = model.analyze_news(test_articles, "AAPL")
    print(model.get_summary(signal))
