"""
FinBERT Sentiment Analyzer.

Uses the FinBERT model for fast, deterministic financial sentiment analysis.
Falls back to VADER + financial lexicon if torch is unavailable.
Can be used standalone or as a baseline for LLM validation.

Usage:
    from auto_researcher.agents.finbert_sentiment import FinBERTAnalyzer
    
    analyzer = FinBERTAnalyzer()
    result = analyzer.analyze("Apple reported record quarterly earnings, beating expectations.")
    print(f"Sentiment: {result['label']} ({result['score']:.3f})")
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Check for transformers
HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    logger.info("transformers/torch not available. Using VADER fallback.")
except Exception as e:
    logger.warning(f"torch loading failed ({e}). Using VADER fallback.")

# VADER fallback
HAS_VADER = False
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    logger.info("VADER not available (install nltk).")


# Financial lexicon for enhancing VADER
FINANCIAL_POSITIVE = {
    "beat", "beats", "beating", "exceeded", "exceeds", "surpass", "surpassed",
    "upgrade", "upgraded", "outperform", "outperformed", "bullish", "growth",
    "profit", "profitable", "gains", "gained", "rally", "rallied", "soar",
    "soared", "surge", "surged", "record", "breakthrough", "innovation",
    "optimistic", "strong", "robust", "momentum", "accelerate", "accelerated",
    "dividend", "buyback", "acquisition", "partnership", "expansion", "upside",
}

FINANCIAL_NEGATIVE = {
    "miss", "missed", "misses", "fell", "fall", "falls", "decline", "declined",
    "downgrade", "downgraded", "underperform", "underperformed", "bearish",
    "loss", "losses", "drop", "dropped", "plunge", "plunged", "crash", "crashed",
    "warning", "concern", "concerns", "risk", "risks", "weak", "weakness",
    "disappointing", "disappointed", "layoff", "layoffs", "lawsuit", "fraud",
    "investigation", "recall", "bankruptcy", "default", "debt", "downside",
}


@dataclass
class FinBERTResult:
    """Result from FinBERT analysis."""
    text: str
    label: str  # positive, negative, neutral
    score: float  # confidence 0-1
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    
    @property
    def sentiment_score(self) -> float:
        """Convert to -1 to +1 scale."""
        return self.positive_prob - self.negative_prob


class FinBERTAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.
    
    FinBERT is a BERT model fine-tuned on financial text for sentiment analysis.
    It's fast, free, and specifically designed for financial language.
    """
    
    # Model options
    MODELS = {
        "finbert": "ProsusAI/finbert",
        "finbert-tone": "yiyanghkust/finbert-tone",
        "distilbert-financial": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    }
    
    def __init__(
        self,
        model_name: str = "finbert",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize FinBERT analyzer.
        
        Args:
            model_name: Which FinBERT variant to use.
            device: Device to run on ('cuda', 'cpu', or None for auto).
            max_length: Maximum token length.
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
        
        self.model_path = self.MODELS.get(model_name, model_name)
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        logger.info(f"Loading FinBERT model: {self.model_path} on {self.device}")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Create pipeline for convenience
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_length=self.max_length,
            truncation=True,
        )
        
        logger.info("FinBERT model loaded successfully")
    
    def analyze(self, text: str) -> FinBERTResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            FinBERTResult with sentiment scores.
        """
        if not text or not text.strip():
            return FinBERTResult(
                text=text,
                label="neutral",
                score=1.0,
                positive_prob=0.0,
                negative_prob=0.0,
                neutral_prob=1.0,
            )
        
        # Truncate if too long
        text = text[:self.max_length * 4]  # Rough char limit
        
        # Get prediction with full scores
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Get label mapping (varies by model)
        id2label = self.model.config.id2label
        
        # Build probability dict
        prob_dict = {}
        for idx, label in id2label.items():
            prob_dict[label.lower()] = probs[idx].item()
        
        # Normalize label names
        positive_prob = prob_dict.get("positive", prob_dict.get("pos", 0.0))
        negative_prob = prob_dict.get("negative", prob_dict.get("neg", 0.0))
        neutral_prob = prob_dict.get("neutral", prob_dict.get("neu", 0.0))
        
        # Get top label
        top_idx = probs.argmax().item()
        top_label = id2label[top_idx].lower()
        top_score = probs[top_idx].item()
        
        return FinBERTResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            label=top_label,
            score=top_score,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
        )
    
    def analyze_batch(self, texts: list[str]) -> list[FinBERTResult]:
        """
        Analyze sentiment of multiple texts efficiently.
        
        Args:
            texts: List of texts to analyze.
            
        Returns:
            List of FinBERTResult objects.
        """
        results = []
        
        # Process in batches
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Filter empty texts
            valid_texts = [(j, t) for j, t in enumerate(batch) if t and t.strip()]
            
            if not valid_texts:
                # All empty
                for _ in batch:
                    results.append(FinBERTResult(
                        text="",
                        label="neutral",
                        score=1.0,
                        positive_prob=0.0,
                        negative_prob=0.0,
                        neutral_prob=1.0,
                    ))
                continue
            
            # Batch inference
            with torch.no_grad():
                inputs = self.tokenizer(
                    [t for _, t in valid_texts],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                all_probs = torch.softmax(outputs.logits, dim=-1)
            
            # Map results back
            id2label = self.model.config.id2label
            valid_idx = 0
            
            for j, text in enumerate(batch):
                if not text or not text.strip():
                    results.append(FinBERTResult(
                        text="",
                        label="neutral",
                        score=1.0,
                        positive_prob=0.0,
                        negative_prob=0.0,
                        neutral_prob=1.0,
                    ))
                else:
                    probs = all_probs[valid_idx]
                    valid_idx += 1
                    
                    prob_dict = {}
                    for idx, label in id2label.items():
                        prob_dict[label.lower()] = probs[idx].item()
                    
                    positive_prob = prob_dict.get("positive", prob_dict.get("pos", 0.0))
                    negative_prob = prob_dict.get("negative", prob_dict.get("neg", 0.0))
                    neutral_prob = prob_dict.get("neutral", prob_dict.get("neu", 0.0))
                    
                    top_idx = probs.argmax().item()
                    top_label = id2label[top_idx].lower()
                    top_score = probs[top_idx].item()
                    
                    results.append(FinBERTResult(
                        text=text[:200] + "..." if len(text) > 200 else text,
                        label=top_label,
                        score=top_score,
                        positive_prob=positive_prob,
                        negative_prob=negative_prob,
                        neutral_prob=neutral_prob,
                    ))
        
        return results
    
    def analyze_news_items(self, news_items: list[dict]) -> list[FinBERTResult]:
        """
        Analyze a list of news items (with title/summary).
        
        Args:
            news_items: List of dicts with 'title' and optionally 'summary' keys.
            
        Returns:
            List of FinBERTResult objects.
        """
        texts = []
        for item in news_items:
            title = item.get("title", "")
            summary = item.get("summary", "")
            text = f"{title}. {summary}".strip()
            texts.append(text)
        
        return self.analyze_batch(texts)
    
    def get_aggregate_sentiment(self, results: list[FinBERTResult]) -> dict:
        """
        Aggregate sentiment across multiple results.
        
        Args:
            results: List of FinBERTResult objects.
            
        Returns:
            Dict with aggregate statistics.
        """
        if not results:
            return {
                "mean_sentiment": 0.0,
                "median_sentiment": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "count": 0,
            }
        
        sentiments = [r.sentiment_score for r in results]
        labels = [r.label for r in results]
        
        import statistics
        
        return {
            "mean_sentiment": statistics.mean(sentiments),
            "median_sentiment": statistics.median(sentiments),
            "positive_ratio": sum(1 for l in labels if l == "positive") / len(labels),
            "negative_ratio": sum(1 for l in labels if l == "negative") / len(labels),
            "neutral_ratio": sum(1 for l in labels if l == "neutral") / len(labels),
            "count": len(results),
            "std_sentiment": statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0,
        }


class VADERFinancialAnalyzer:
    """
    Fallback sentiment analyzer using VADER + financial lexicon.
    
    Used when PyTorch/transformers are unavailable (e.g., Windows DLL issues).
    Not as accurate as FinBERT but good enough for a baseline.
    """
    
    def __init__(self):
        """Initialize VADER analyzer."""
        if not HAS_VADER:
            raise ImportError("VADER not available. Install with: pip install nltk")
        
        # Download VADER lexicon if needed
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Downloading VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        
        self.vader = SentimentIntensityAnalyzer()
        
        # Add financial terms to VADER lexicon
        for word in FINANCIAL_POSITIVE:
            self.vader.lexicon[word] = 2.0
        for word in FINANCIAL_NEGATIVE:
            self.vader.lexicon[word] = -2.0
        
        logger.info("VADER Financial analyzer initialized")
    
    def analyze(self, text: str) -> FinBERTResult:
        """Analyze sentiment using VADER."""
        if not text or not text.strip():
            return FinBERTResult(
                text=text,
                label="neutral",
                score=1.0,
                positive_prob=0.0,
                negative_prob=0.0,
                neutral_prob=1.0,
            )
        
        scores = self.vader.polarity_scores(text)
        
        # Convert compound score to probabilities
        compound = scores['compound']
        
        # Map to label
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        # Estimate probabilities from compound
        if compound > 0:
            positive_prob = min(1.0, 0.5 + compound / 2)
            negative_prob = max(0.0, 0.5 - compound)
            neutral_prob = 1.0 - positive_prob - negative_prob
        elif compound < 0:
            negative_prob = min(1.0, 0.5 - compound / 2)
            positive_prob = max(0.0, 0.5 + compound)
            neutral_prob = 1.0 - positive_prob - negative_prob
        else:
            positive_prob = 0.25
            negative_prob = 0.25
            neutral_prob = 0.5
        
        # Ensure non-negative
        positive_prob = max(0.0, positive_prob)
        negative_prob = max(0.0, negative_prob)
        neutral_prob = max(0.0, neutral_prob)
        
        return FinBERTResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            label=label,
            score=max(positive_prob, negative_prob, neutral_prob),
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
        )
    
    def analyze_batch(self, texts: list[str]) -> list[FinBERTResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def analyze_news_items(self, news_items: list[dict]) -> list[FinBERTResult]:
        """Analyze news items."""
        texts = []
        for item in news_items:
            title = item.get("title", "")
            summary = item.get("summary", "")
            text = f"{title}. {summary}".strip()
            texts.append(text)
        return self.analyze_batch(texts)
    
    def get_aggregate_sentiment(self, results: list[FinBERTResult]) -> dict:
        """Aggregate sentiment across results."""
        if not results:
            return {
                "mean_sentiment": 0.0,
                "median_sentiment": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "count": 0,
            }
        
        sentiments = [r.sentiment_score for r in results]
        labels = [r.label for r in results]
        
        import statistics
        
        return {
            "mean_sentiment": statistics.mean(sentiments),
            "median_sentiment": statistics.median(sentiments),
            "positive_ratio": sum(1 for l in labels if l == "positive") / len(labels),
            "negative_ratio": sum(1 for l in labels if l == "negative") / len(labels),
            "neutral_ratio": sum(1 for l in labels if l == "neutral") / len(labels),
            "count": len(results),
            "std_sentiment": statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0,
        }


def create_analyzer(prefer_finbert: bool = True) -> "FinBERTAnalyzer | VADERFinancialAnalyzer":
    """
    Create best available sentiment analyzer.
    
    Args:
        prefer_finbert: Try FinBERT first, fall back to VADER.
        
    Returns:
        Sentiment analyzer instance.
    """
    if prefer_finbert and HAS_TRANSFORMERS:
        try:
            return FinBERTAnalyzer()
        except Exception as e:
            logger.warning(f"FinBERT failed ({e}), falling back to VADER")
    
    if HAS_VADER:
        return VADERFinancialAnalyzer()
    
    raise ImportError(
        "No sentiment analyzer available. Install either:\n"
        "  - transformers + torch (for FinBERT)\n"
        "  - nltk (for VADER fallback)"
    )


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

_analyzer = None

def get_analyzer() -> FinBERTAnalyzer:
    """Get or create singleton analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FinBERTAnalyzer()
    return _analyzer


def analyze_text(text: str) -> FinBERTResult:
    """Quick sentiment analysis of text."""
    return get_analyzer().analyze(text)


def analyze_texts(texts: list[str]) -> list[FinBERTResult]:
    """Quick batch sentiment analysis."""
    return get_analyzer().analyze_batch(texts)


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinBERT sentiment analysis")
    parser.add_argument("texts", nargs="+", help="Texts to analyze")
    parser.add_argument("--model", default="finbert", help="Model variant")
    
    args = parser.parse_args()
    
    analyzer = FinBERTAnalyzer(model_name=args.model)
    
    for text in args.texts:
        result = analyzer.analyze(text)
        print(f"\nText: {result.text}")
        print(f"Label: {result.label} (confidence: {result.score:.3f})")
        print(f"Sentiment Score: {result.sentiment_score:+.3f}")
        print(f"  Positive: {result.positive_prob:.3f}")
        print(f"  Negative: {result.negative_prob:.3f}")
        print(f"  Neutral: {result.neutral_prob:.3f}")
