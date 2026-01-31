"""
Topic-Based Sentiment Model.

Classifies news into financial topics and computes topic-specific sentiment.
Research shows certain topics (litigation, M&A, earnings) have stronger
predictive power than generic sentiment.

================================================================================
ACADEMIC BASIS
================================================================================
    Garcia (2013): Media sentiment predicts returns, especially during recessions
    Boudoukh et al. (2019): News topics have different return predictability
    Tetlock et al. (2008): Negative words in firm-specific news predict earnings
    Loughran & McDonald (2011): Finance-specific sentiment dictionaries
    
KEY INSIGHT:
    Generic sentiment averages signal across topics, diluting predictive power.
    "Lawsuit filed" (litigation + negative) has different implications than
    "Sales below expectations" (earnings + negative) or "CEO departure" (management).

TOPICS WITH STRONGEST SIGNAL (from literature):
    1. Litigation/Regulatory: Highly predictive, asymmetric (negative stronger)
    2. Earnings/Guidance: Immediate impact, rapid incorporation
    3. M&A: Acquirer vs target have opposite reactions
    4. Management Changes: CEO departures especially impactful
    5. Product/Innovation: Longer-term impact, harder to price

================================================================================

Usage:
    from auto_researcher.models.topic_sentiment import TopicSentimentModel
    
    model = TopicSentimentModel()
    result = model.analyze_article("Apple faces antitrust lawsuit from DOJ")
    print(f"Topic: {result.topic}, Sentiment: {result.sentiment}")
    print(f"Signal: {result.trading_signal}")
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# TOPIC DEFINITIONS
# ==============================================================================

# Each topic has keywords and sentiment modifiers
# The modifiers adjust how sentiment impacts returns

TOPIC_DEFINITIONS = {
    "litigation": {
        "keywords": [
            # Lawsuits
            "lawsuit", "sued", "sues", "suing", "litigation", "litigate",
            "plaintiff", "defendant", "settlement", "settles", "settled",
            "verdict", "jury", "court", "judge", "ruling", "appeal",
            # Regulatory
            "sec", "ftc", "doj", "antitrust", "investigation", "investigated",
            "probe", "probing", "subpoena", "indictment", "charged", "charges",
            "violation", "penalty", "fine", "fined", "sanctions", "compliance",
            "regulatory", "regulators", "enforcement", "consent decree",
            # Fraud
            "fraud", "fraudulent", "scandal", "misconduct", "whistleblower",
            "accounting irregularities", "restatement", "restated",
        ],
        "sentiment_multiplier": 1.5,  # Negative sentiment hits harder
        "base_impact": -0.02,  # Litigation news is inherently negative
        "decay_days": 30,  # Effect persists
    },
    
    "earnings": {
        "keywords": [
            "earnings", "revenue", "profit", "loss", "eps", "quarterly results",
            "beat", "beats", "miss", "missed", "guidance", "outlook", "forecast",
            "revenue growth", "margin", "margins", "gross margin", "operating margin",
            "same-store sales", "comparable sales", "comps", "revenue decline",
            "profit warning", "lowered guidance", "raised guidance", "preannounce",
            "fiscal quarter", "fiscal year", "annual report", "quarterly report",
        ],
        "sentiment_multiplier": 1.2,  # Standard earnings effect
        "base_impact": 0.0,  # Neutral - depends on beat/miss
        "decay_days": 5,  # Rapid incorporation
    },
    
    "mna": {  # Mergers & Acquisitions
        "keywords": [
            "merger", "acquisition", "acquire", "acquired", "acquires", "acquiring",
            "takeover", "bid", "bidding", "bidder", "buyout", "lbo",
            "target", "deal", "transaction", "combine", "combination",
            "spinoff", "spin-off", "divestiture", "divest", "divesting",
            "ipo", "spac", "going private", "take private", "strategic review",
            "activist investor", "proxy fight", "board seat",
        ],
        "sentiment_multiplier": 1.0,  # Complex - depends on role
        "base_impact": 0.0,  # Depends on whether acquirer or target
        "decay_days": 10,
    },
    
    "management": {
        "keywords": [
            "ceo", "chief executive", "cfo", "chief financial",
            "coo", "chief operating", "cto", "chief technology",
            "chairman", "board of directors", "director", "executive",
            "resign", "resigned", "resignation", "steps down", "stepping down",
            "retire", "retired", "retirement", "depart", "departure",
            "appoint", "appointed", "appointment", "hire", "hired", "hiring",
            "promoted", "promotion", "succession", "successor",
            "founder", "leadership", "management change", "restructuring",
        ],
        "sentiment_multiplier": 1.3,  # Management changes are impactful
        "base_impact": -0.005,  # Slight negative bias (departures are news)
        "decay_days": 20,
    },
    
    "product": {
        "keywords": [
            "product", "launch", "launches", "launched", "announce", "announced",
            "unveil", "unveiled", "reveal", "revealed", "introduce", "introduced",
            "release", "released", "rollout", "new model", "new version",
            "innovation", "innovative", "patent", "patented", "technology",
            "breakthrough", "disruption", "disruptive", "fda approval", "approved",
            "recall", "recalled", "defect", "defective", "safety issue",
            "clinical trial", "trial results", "phase 3", "phase 2",
        ],
        "sentiment_multiplier": 0.8,  # Harder to price, less immediate
        "base_impact": 0.01,  # Product news often positive
        "decay_days": 60,  # Slow incorporation
    },
    
    "analyst": {
        "keywords": [
            "upgrade", "upgraded", "downgrade", "downgraded", "rating",
            "buy rating", "sell rating", "hold rating", "outperform", "underperform",
            "price target", "target price", "analyst", "analysts",
            "wall street", "initiate", "initiated", "coverage",
            "overweight", "underweight", "equal weight", "neutral",
            "recommendation", "maintains", "reiterate", "reiterates",
        ],
        "sentiment_multiplier": 0.7,  # Already priced in often
        "base_impact": 0.0,
        "decay_days": 3,  # Very rapid
    },
    
    "macro": {
        "keywords": [
            "fed", "federal reserve", "interest rate", "rate hike", "rate cut",
            "inflation", "cpi", "ppi", "gdp", "unemployment", "jobs report",
            "tariff", "tariffs", "trade war", "trade deal", "sanctions",
            "recession", "recovery", "stimulus", "fiscal policy", "monetary policy",
            "treasury", "bond yields", "yield curve", "inversion",
            "oil prices", "commodity", "commodities", "supply chain",
        ],
        "sentiment_multiplier": 0.5,  # Affects all stocks similarly
        "base_impact": 0.0,
        "decay_days": 5,
    },
    
    "labor": {
        "keywords": [
            "layoff", "layoffs", "laid off", "job cuts", "workforce reduction",
            "restructuring", "downsizing", "furlough", "furloughed",
            "strike", "strikes", "union", "labor dispute", "workers",
            "hiring freeze", "cost cutting", "efficiency", "headcount",
        ],
        "sentiment_multiplier": 1.1,
        "base_impact": -0.01,  # Usually negative but can signal discipline
        "decay_days": 10,
    },
    
    "partnership": {
        "keywords": [
            "partnership", "partner", "partners", "partnered", "alliance",
            "collaboration", "collaborate", "joint venture", "jv",
            "licensing", "license", "licensed", "agreement", "contract",
            "deal", "signed", "multi-year", "exclusive", "strategic",
            "supplier", "customer", "client", "wins contract",
        ],
        "sentiment_multiplier": 0.9,
        "base_impact": 0.005,  # Usually positive
        "decay_days": 15,
    },
    
    "competitive": {
        "keywords": [
            "competitor", "competition", "market share", "pricing pressure",
            "price war", "undercutting", "rival", "rivals", "losing share",
            "gaining share", "disrupted", "threat", "threatens", "challenged",
            "dominant", "dominance", "moat", "advantage", "disadvantage",
        ],
        "sentiment_multiplier": 1.0,
        "base_impact": 0.0,
        "decay_days": 20,
    },
}

# High-priority topics (stronger signal from literature)
HIGH_SIGNAL_TOPICS = {"litigation", "earnings", "management", "mna"}


# ==============================================================================
# SENTIMENT DICTIONARIES (Loughran-McDonald inspired)
# ==============================================================================

POSITIVE_WORDS = {
    "beat", "beats", "beating", "exceeded", "exceeds", "surpass", "surpassed",
    "record", "records", "growth", "growing", "grew", "strong", "stronger",
    "profit", "profitable", "gains", "gained", "rally", "rallied", "surge",
    "surged", "soar", "soared", "bullish", "optimistic", "positive", "upbeat",
    "outperform", "outperformed", "upgrade", "upgraded", "success", "successful",
    "innovation", "innovative", "breakthrough", "expand", "expansion", "expanded",
    "robust", "momentum", "accelerate", "accelerated", "improve", "improved",
    "improvement", "better", "best", "exceed", "excellent", "exceptional",
    "win", "wins", "won", "winning", "approval", "approved", "launch", "launched",
    "partnership", "deal", "agreement", "contract", "milestone", "achievement",
}

NEGATIVE_WORDS = {
    "miss", "missed", "misses", "fell", "fall", "falls", "decline", "declined",
    "declining", "drop", "dropped", "dropping", "plunge", "plunged", "crash",
    "crashed", "loss", "losses", "losing", "weak", "weaker", "weakness",
    "bearish", "pessimistic", "negative", "downgrade", "downgraded",
    "underperform", "underperformed", "warning", "concern", "concerns", "worried",
    "risk", "risks", "risky", "threat", "threatens", "challenged", "struggle",
    "struggling", "disappointing", "disappointed", "disappointment", "trouble",
    "layoff", "layoffs", "lawsuit", "sued", "investigation", "probe", "fraud",
    "scandal", "recall", "recalled", "defect", "bankruptcy", "default", "debt",
    "downside", "lower", "lowered", "cut", "cuts", "slashed", "slashing",
    "failed", "failure", "terminate", "terminated", "delay", "delayed",
    "slowing", "slowdown", "slow",  # Added slowing
}

UNCERTAINTY_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "uncertain", "uncertainty",
    "unclear", "unknown", "question", "questions", "questioning", "doubt",
    "doubts", "volatile", "volatility", "unpredictable", "speculation",
    "speculate", "rumor", "rumors", "unconfirmed", "pending", "awaiting",
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TopicClassification:
    """Classification of an article into topic(s)."""
    primary_topic: str
    secondary_topics: List[str]
    topic_scores: Dict[str, float]  # Confidence per topic
    keywords_matched: Dict[str, List[str]]  # Which keywords matched


@dataclass
class TopicSentiment:
    """Sentiment analysis result with topic context."""
    text: str
    topic: TopicClassification
    
    # Raw sentiment
    positive_count: int
    negative_count: int
    uncertainty_count: int
    word_count: int
    
    # Computed scores
    raw_sentiment: float  # -1 to +1
    topic_adjusted_sentiment: float  # Adjusted by topic multiplier
    uncertainty_score: float  # 0 to 1
    
    # Trading signal
    trading_signal: str  # "bullish", "bearish", "neutral"
    signal_strength: float  # 0 to 1
    expected_return: float  # Expected return contribution
    
    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedTopicSignal:
    """Aggregated topic signals for a ticker over a time period."""
    ticker: str
    start_date: datetime
    end_date: datetime
    
    # Per-topic aggregates
    topic_sentiment: Dict[str, float]  # Average sentiment by topic
    topic_counts: Dict[str, int]  # Article count by topic
    topic_signals: Dict[str, str]  # Signal by topic
    
    # Overall signal
    composite_signal: str  # "bullish", "bearish", "neutral"
    composite_score: float  # -1 to +1
    confidence: float  # 0 to 1
    
    # High-signal topic alerts
    litigation_alert: bool = False
    management_alert: bool = False
    earnings_surprise_detected: bool = False


# ==============================================================================
# TOPIC SENTIMENT MODEL
# ==============================================================================

class TopicSentimentModel:
    """
    Analyzes news sentiment with topic-specific adjustments.
    
    Key features:
    1. Classifies news into financial topics
    2. Applies topic-specific sentiment multipliers
    3. Weights high-signal topics more heavily
    4. Detects topic-specific alerts (litigation, management changes)
    """
    
    def __init__(
        self,
        topic_definitions: Optional[Dict] = None,
        use_high_signal_weighting: bool = True,
    ):
        """
        Initialize the topic sentiment model.
        
        Args:
            topic_definitions: Custom topic definitions (uses defaults if None)
            use_high_signal_weighting: Whether to weight high-signal topics more
        """
        self.topics = topic_definitions or TOPIC_DEFINITIONS
        self.use_high_signal_weighting = use_high_signal_weighting
        
        # Precompile keyword patterns for efficiency
        self._compile_patterns()
        
        logger.info(f"TopicSentimentModel initialized with {len(self.topics)} topics")
    
    def _compile_patterns(self):
        """Compile regex patterns for keyword matching."""
        self.topic_patterns = {}
        for topic, config in self.topics.items():
            # Create word boundary pattern for each keyword
            keywords = config["keywords"]
            # Escape special chars and join with OR
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.topic_patterns[topic] = re.compile(pattern, re.IGNORECASE)
    
    def classify_topic(self, text: str) -> TopicClassification:
        """
        Classify text into financial topics.
        
        Args:
            text: Article text or headline
            
        Returns:
            TopicClassification with primary and secondary topics
        """
        text_lower = text.lower()
        topic_scores = {}
        keywords_matched = {}
        
        for topic, pattern in self.topic_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                # Score based on match count, normalized by typical keyword density
                keywords_matched[topic] = matches
                # Weight by number of unique keywords matched
                unique_matches = len(set(matches))
                total_matches = len(matches)
                topic_scores[topic] = unique_matches + (total_matches - unique_matches) * 0.5
        
        if not topic_scores:
            return TopicClassification(
                primary_topic="general",
                secondary_topics=[],
                topic_scores={"general": 1.0},
                keywords_matched={},
            )
        
        # Sort by score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: -x[1])
        primary = sorted_topics[0][0]
        secondary = [t for t, s in sorted_topics[1:] if s >= 1.0]
        
        return TopicClassification(
            primary_topic=primary,
            secondary_topics=secondary,
            topic_scores=topic_scores,
            keywords_matched=keywords_matched,
        )
    
    def compute_sentiment(self, text: str) -> Tuple[float, int, int, int, int]:
        """
        Compute raw sentiment from text.
        
        Returns:
            (sentiment_score, positive_count, negative_count, uncertainty_count, word_count)
        """
        words = text.lower().split()
        word_count = len(words)
        
        positive_count = sum(1 for w in words if w.strip('.,!?;:') in POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w.strip('.,!?;:') in NEGATIVE_WORDS)
        uncertainty_count = sum(1 for w in words if w.strip('.,!?;:') in UNCERTAINTY_WORDS)
        
        # Net sentiment normalized by word count
        if word_count > 0:
            sentiment = (positive_count - negative_count) / np.sqrt(word_count)
            # Clip to -1, +1
            sentiment = np.clip(sentiment, -1, 1)
        else:
            sentiment = 0.0
        
        return sentiment, positive_count, negative_count, uncertainty_count, word_count
    
    def analyze_article(self, text: str) -> TopicSentiment:
        """
        Analyze a single article for topic and sentiment.
        
        Args:
            text: Article text or headline
            
        Returns:
            TopicSentiment with full analysis
        """
        # Classify topic
        topic = self.classify_topic(text)
        
        # Compute raw sentiment
        raw_sentiment, pos, neg, unc, wc = self.compute_sentiment(text)
        
        # Get topic config
        topic_config = self.topics.get(topic.primary_topic, {
            "sentiment_multiplier": 1.0,
            "base_impact": 0.0,
        })
        
        # Apply topic adjustment
        multiplier = topic_config.get("sentiment_multiplier", 1.0)
        base_impact = topic_config.get("base_impact", 0.0)
        
        # Topic-adjusted sentiment
        # Negative sentiment is amplified by multiplier, positive less so
        if raw_sentiment < 0:
            adjusted = raw_sentiment * multiplier + base_impact
        else:
            adjusted = raw_sentiment * (2 - multiplier) + base_impact
        adjusted = np.clip(adjusted, -1, 1)
        
        # Uncertainty score
        uncertainty = unc / np.sqrt(wc) if wc > 0 else 0.0
        uncertainty = min(uncertainty, 1.0)
        
        # Trading signal
        if adjusted > 0.15:
            signal = "bullish"
            strength = min(adjusted, 1.0)
        elif adjusted < -0.15:
            signal = "bearish"
            strength = min(abs(adjusted), 1.0)
        else:
            signal = "neutral"
            strength = 0.0
        
        # Expected return (rough estimate based on literature)
        expected_return = adjusted * 0.02  # 2% max for extreme sentiment
        
        return TopicSentiment(
            text=text[:200] + "..." if len(text) > 200 else text,
            topic=topic,
            positive_count=pos,
            negative_count=neg,
            uncertainty_count=unc,
            word_count=wc,
            raw_sentiment=raw_sentiment,
            topic_adjusted_sentiment=adjusted,
            uncertainty_score=uncertainty,
            trading_signal=signal,
            signal_strength=strength,
            expected_return=expected_return,
        )
    
    def analyze_articles(
        self,
        articles: List[Dict],
        ticker: str,
        date_field: str = "published_date",
        text_field: str = "title",
    ) -> AggregatedTopicSignal:
        """
        Analyze multiple articles and aggregate by topic.
        
        Args:
            articles: List of article dicts with date and text fields
            ticker: Stock ticker
            date_field: Field name for article date
            text_field: Field name for article text (can be 'title' or 'full_text')
            
        Returns:
            AggregatedTopicSignal with topic-level and overall signals
        """
        if not articles:
            return AggregatedTopicSignal(
                ticker=ticker,
                start_date=datetime.now(),
                end_date=datetime.now(),
                topic_sentiment={},
                topic_counts={},
                topic_signals={},
                composite_signal="neutral",
                composite_score=0.0,
                confidence=0.0,
            )
        
        # Analyze each article
        results = []
        for article in articles:
            text = article.get(text_field, "") or ""
            if not text:
                continue
            result = self.analyze_article(text)
            result.article = article  # Attach for date access
            results.append(result)
        
        if not results:
            return AggregatedTopicSignal(
                ticker=ticker,
                start_date=datetime.now(),
                end_date=datetime.now(),
                topic_sentiment={},
                topic_counts={},
                topic_signals={},
                composite_signal="neutral",
                composite_score=0.0,
                confidence=0.0,
            )
        
        # Date range - handle timezone-naive/aware comparison
        dates = []
        for r in results:
            if hasattr(r, 'article') and date_field in r.article:
                d = r.article[date_field]
                if isinstance(d, str):
                    try:
                        d = datetime.fromisoformat(d.replace('Z', '+00:00'))
                    except:
                        d = datetime.now()
                # Make timezone-naive for comparison
                if hasattr(d, 'tzinfo') and d.tzinfo is not None:
                    d = d.replace(tzinfo=None)
                dates.append(d)
        
        start_date = min(dates) if dates else datetime.now()
        end_date = max(dates) if dates else datetime.now()
        
        # Aggregate by topic
        topic_sentiments = defaultdict(list)
        for r in results:
            topic = r.topic.primary_topic
            topic_sentiments[topic].append(r.topic_adjusted_sentiment)
            # Also count secondary topics
            for sec_topic in r.topic.secondary_topics:
                topic_sentiments[sec_topic].append(r.topic_adjusted_sentiment * 0.5)
        
        topic_sentiment = {t: np.mean(s) for t, s in topic_sentiments.items()}
        topic_counts = {t: len(s) for t, s in topic_sentiments.items()}
        
        # Per-topic signals
        topic_signals = {}
        for topic, sent in topic_sentiment.items():
            if sent > 0.15:
                topic_signals[topic] = "bullish"
            elif sent < -0.15:
                topic_signals[topic] = "bearish"
            else:
                topic_signals[topic] = "neutral"
        
        # Composite signal with weighting
        total_weight = 0
        weighted_sentiment = 0
        
        for topic, sent in topic_sentiment.items():
            count = topic_counts.get(topic, 1)
            
            # Weight by article count
            weight = np.sqrt(count)
            
            # Boost high-signal topics
            if self.use_high_signal_weighting and topic in HIGH_SIGNAL_TOPICS:
                weight *= 1.5
            
            weighted_sentiment += sent * weight
            total_weight += weight
        
        composite_score = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Composite signal
        if composite_score > 0.1:
            composite_signal = "bullish"
        elif composite_score < -0.1:
            composite_signal = "bearish"
        else:
            composite_signal = "neutral"
        
        # Confidence based on article count and agreement
        confidence = min(len(results) / 10, 1.0)  # More articles = more confidence
        
        # Alerts
        litigation_alert = (
            topic_counts.get("litigation", 0) >= 2 and 
            topic_sentiment.get("litigation", 0) < -0.2
        )
        management_alert = (
            topic_counts.get("management", 0) >= 1 and
            topic_sentiment.get("management", 0) < -0.1
        )
        earnings_surprise = (
            topic_counts.get("earnings", 0) >= 1 and
            abs(topic_sentiment.get("earnings", 0)) > 0.3
        )
        
        return AggregatedTopicSignal(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            topic_sentiment=topic_sentiment,
            topic_counts=topic_counts,
            topic_signals=topic_signals,
            composite_signal=composite_signal,
            composite_score=composite_score,
            confidence=confidence,
            litigation_alert=litigation_alert,
            management_alert=management_alert,
            earnings_surprise_detected=earnings_surprise,
        )
    
    def get_topic_summary(self, signal: AggregatedTopicSignal) -> str:
        """Generate a human-readable summary of topic signals."""
        lines = [f"Topic Sentiment Analysis for {signal.ticker}"]
        lines.append("=" * 50)
        
        if not signal.topic_counts:
            lines.append("No news articles analyzed.")
            return "\n".join(lines)
        
        lines.append(f"Period: {signal.start_date.date()} to {signal.end_date.date()}")
        lines.append(f"Total Topics Detected: {len(signal.topic_counts)}")
        lines.append("")
        
        # Sort by count
        for topic, count in sorted(signal.topic_counts.items(), key=lambda x: -x[1]):
            sent = signal.topic_sentiment.get(topic, 0)
            sig = signal.topic_signals.get(topic, "neutral")
            high = "⚡" if topic in HIGH_SIGNAL_TOPICS else ""
            lines.append(f"  {topic.upper()}{high}: {count} articles, sentiment={sent:.2f} ({sig})")
        
        lines.append("")
        lines.append(f"COMPOSITE: {signal.composite_signal.upper()} (score={signal.composite_score:.2f})")
        lines.append(f"Confidence: {signal.confidence:.0%}")
        
        # Alerts
        if signal.litigation_alert:
            lines.append("⚠️  LITIGATION ALERT: Multiple negative litigation articles")
        if signal.management_alert:
            lines.append("⚠️  MANAGEMENT ALERT: Negative management news")
        if signal.earnings_surprise_detected:
            lines.append("📊 EARNINGS SURPRISE: Strong sentiment on earnings news")
        
        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def analyze_news_by_topic(
    articles: List[Dict],
    ticker: str,
    text_field: str = "title",
) -> AggregatedTopicSignal:
    """
    Convenience function to analyze news articles by topic.
    
    Args:
        articles: List of article dicts
        ticker: Stock ticker
        text_field: Field containing article text
        
    Returns:
        AggregatedTopicSignal
    """
    model = TopicSentimentModel()
    return model.analyze_articles(articles, ticker, text_field=text_field)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

TOPIC_MODEL_CONFIG = {
    "high_signal_topics": list(HIGH_SIGNAL_TOPICS),
    "all_topics": list(TOPIC_DEFINITIONS.keys()),
    "sentiment_thresholds": {
        "bullish": 0.15,
        "bearish": -0.15,
    },
    "alert_thresholds": {
        "litigation_count": 2,
        "litigation_sentiment": -0.2,
        "management_sentiment": -0.1,
        "earnings_surprise": 0.3,
    },
}


if __name__ == "__main__":
    # Demo
    model = TopicSentimentModel()
    
    test_headlines = [
        "Apple beats earnings expectations, revenue surges 15%",
        "Tesla faces SEC investigation over autopilot claims",
        "Microsoft CEO Satya Nadella to step down in 2025",
        "Google announces partnership with OpenAI rival Anthropic",
        "Amazon to lay off 18,000 workers amid slowing growth",
        "Nvidia unveils new AI chip, stock rallies on breakthrough",
        "Meta faces antitrust lawsuit from FTC, could be forced to divest Instagram",
        "Ford recalls 500,000 vehicles due to brake defect",
    ]
    
    print("=" * 60)
    print("TOPIC SENTIMENT MODEL DEMO")
    print("=" * 60)
    
    for headline in test_headlines:
        result = model.analyze_article(headline)
        print(f"\n📰 {headline[:60]}...")
        print(f"   Topic: {result.topic.primary_topic.upper()}")
        print(f"   Sentiment: {result.topic_adjusted_sentiment:+.2f} ({result.trading_signal})")
        if result.topic.secondary_topics:
            print(f"   Also: {', '.join(result.topic.secondary_topics)}")
