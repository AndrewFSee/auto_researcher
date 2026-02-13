#!/usr/bin/env python
"""
Detailed Model & Agent Test Script.

This script runs all models on Mag 7 stocks with verbose logging to show:
1. What each model is doing (via DEBUG logs)
2. Each model's specific signal/recommendation per stock
3. A detailed breakdown of all model suggestions
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

sys.path.insert(0, 'src')

# ============================================================================
# Configure VERBOSE logging to see all model calls
# ============================================================================
logging.basicConfig(
    level=logging.DEBUG,  # Show everything
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# Quiet down noisy libraries
for quiet_logger in ['urllib3', 'httpx', 'httpcore', 'filelock', 'huggingface_hub', 
                      'transformers', 'datasets', 'tqdm', 'fsspec']:
    logging.getLogger(quiet_logger).setLevel(logging.WARNING)

# Enable DEBUG for our models/agents
for verbose_logger in [
    'auto_researcher.models.sector_momentum',
    'auto_researcher.models.topic_sentiment',
    'auto_researcher.models.insider_cluster',
    'auto_researcher.models.pead_enhanced',
    'auto_researcher.models.early_adopter',
    'auto_researcher.models.quality_value',
    'auto_researcher.models.earnings_tech_signal',
    'auto_researcher.agents.sentiment_agent',
    'auto_researcher.agents.thematic_agent',
    'auto_researcher.agents.insider_trading_agent',
    'auto_researcher.agents.fundamental_agent',
    'auto_researcher.agents.earnings_agent',
    '__main__',
]:
    logging.getLogger(verbose_logger).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

MAG7 = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']


@dataclass
class ModelSignal:
    """Detailed signal from a single model."""
    model_name: str
    ticker: str
    score: float  # 0-1 scale
    signal: str   # e.g., "bullish", "neutral", "bearish"
    recommendation: str  # e.g., "BUY", "HOLD", "SELL"
    confidence: float = 0.5
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class StockAnalysis:
    """Complete analysis for a stock."""
    ticker: str
    model_signals: List[ModelSignal] = field(default_factory=list)
    composite_score: float = 0.0
    final_recommendation: str = "HOLD"


def signal_to_recommendation(score: float, signal: str = None) -> str:
    """Convert score to recommendation."""
    if score >= 0.7:
        return "STRONG BUY"
    elif score >= 0.6:
        return "BUY"
    elif score >= 0.4:
        return "HOLD"
    elif score >= 0.3:
        return "SELL"
    else:
        return "STRONG SELL"


def print_model_header(model_num: int, model_name: str, description: str):
    """Print a header for each model test."""
    print("\n" + "=" * 80)
    print(f"  [{model_num}/8] {model_name}")
    print(f"  {description}")
    print("=" * 80)


def print_signal_table(signals: Dict[str, ModelSignal]):
    """Print a table of signals for all tickers."""
    print()
    print(f"  {'Ticker':<8} {'Score':<8} {'Signal':<12} {'Recommendation':<16} {'Details'}")
    print("  " + "-" * 70)
    for ticker in MAG7:
        if ticker in signals:
            s = signals[ticker]
            details_str = s.reasoning[:40] + "..." if len(s.reasoning) > 40 else s.reasoning
            print(f"  {ticker:<8} {s.score:.3f}    {s.signal:<12} {s.recommendation:<16} {details_str}")
        else:
            print(f"  {ticker:<8} N/A")
    print()


# ============================================================================
# MODEL TESTS
# ============================================================================

def test_sector_momentum() -> Dict[str, ModelSignal]:
    """Test SectorMomentumModel with detailed output."""
    print_model_header(1, "SECTOR MOMENTUM MODEL", "Analyzes sector rotation and momentum")
    
    signals = {}
    try:
        from auto_researcher.models.sector_momentum import SectorMomentumModel
        
        logger.info("Initializing SectorMomentumModel...")
        model = SectorMomentumModel()
        
        snapshot = model.get_rotation_snapshot()
        print(f"\n  Market Regime: {snapshot.regime}")
        print(f"  Overweight Sectors: {snapshot.overweight_sectors[:3]}")
        print(f"  Underweight Sectors: {snapshot.underweight_sectors[:2]}")
        
        for ticker in MAG7:
            logger.debug(f"Getting sector signal for {ticker}")
            sig = model.get_stock_signal(ticker)
            
            # Convert boost to 0-1 score
            score = 0.5 + (sig.sector_boost if sig.sector_boost else 0)
            score = max(0, min(1, score))
            
            signal_type = "bullish" if sig.sector_boost and sig.sector_boost > 0 else \
                          "bearish" if sig.sector_boost and sig.sector_boost < 0 else "neutral"
            
            signals[ticker] = ModelSignal(
                model_name="SectorMomentum",
                ticker=ticker,
                score=score,
                signal=signal_type,
                recommendation=signal_to_recommendation(score),
                details={
                    'sector': sig.sector,
                    'direction': sig.direction,
                    'boost': sig.sector_boost,
                },
                reasoning=f"{sig.sector} sector is {sig.direction or 'neutral'} ({sig.sector_boost*100:+.0f}% boost)" if sig.sector_boost else f"{sig.sector} sector neutral"
            )
            logger.info(f"{ticker}: Sector={sig.sector}, Direction={sig.direction}, Boost={sig.sector_boost}")
        
        print_signal_table(signals)
        print("  [OK] SectorMomentumModel complete")
        
    except Exception as e:
        logger.error(f"SectorMomentumModel failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals


def test_topic_sentiment() -> Dict[str, ModelSignal]:
    """Test TopicSentimentModel with detailed output."""
    print_model_header(2, "TOPIC SENTIMENT MODEL", "Analyzes news articles by topic")
    
    signals = {}
    try:
        from auto_researcher.models.topic_sentiment import TopicSentimentModel
        import sqlite3
        import pandas as pd
        
        logger.info("Initializing TopicSentimentModel...")
        model = TopicSentimentModel()
        
        conn = sqlite3.connect('data/news.db')
        
        for ticker in MAG7:
            logger.debug(f"Fetching news for {ticker} from local DB")
            df = pd.read_sql('''
                SELECT title, full_text, published_date FROM articles 
                WHERE ticker = ? AND full_text IS NOT NULL 
                ORDER BY published_date DESC LIMIT 30
            ''', conn, params=(ticker,))
            
            if len(df) > 0:
                articles = [
                    {'title': r['title'], 'text': r['full_text'], 'published_date': r['published_date']} 
                    for _, r in df.iterrows()
                ]
                logger.debug(f"Analyzing {len(articles)} articles for {ticker}")
                result = model.analyze_articles(articles, ticker)
                
                # Convert composite_score (-1 to 1) to (0 to 1)
                score = (result.composite_score + 1) / 2
                
                # Log topic breakdown
                topic_details = {}
                if hasattr(result, 'topic_scores'):
                    for topic, topic_score in result.topic_scores.items():
                        topic_details[topic] = round(topic_score, 3)
                        logger.debug(f"  {ticker} topic '{topic}': {topic_score:.3f}")
                
                signals[ticker] = ModelSignal(
                    model_name="TopicSentiment",
                    ticker=ticker,
                    score=score,
                    signal=result.composite_signal,
                    recommendation=signal_to_recommendation(score),
                    details={
                        'raw_score': result.composite_score,
                        'article_count': len(df),
                        'topics': topic_details,
                    },
                    reasoning=f"{len(df)} articles, overall {result.composite_signal} ({result.composite_score:+.3f})"
                )
                logger.info(f"{ticker}: {len(df)} articles -> {result.composite_signal} ({result.composite_score:+.3f})")
            else:
                signals[ticker] = ModelSignal(
                    model_name="TopicSentiment",
                    ticker=ticker,
                    score=0.5,
                    signal="neutral",
                    recommendation="HOLD",
                    reasoning="No articles found"
                )
                logger.warning(f"{ticker}: No articles in local database")
        
        conn.close()
        print_signal_table(signals)
        print("  [OK] TopicSentimentModel complete")
        
    except Exception as e:
        logger.error(f"TopicSentimentModel failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals


def test_insider_cluster() -> Dict[str, ModelSignal]:
    """Test InsiderClusterModel with detailed output."""
    print_model_header(3, "INSIDER CLUSTER MODEL", "Detects coordinated insider trading")
    
    signals = {}
    try:
        from auto_researcher.models.insider_cluster import InsiderClusterModel
        
        logger.info("Initializing InsiderClusterModel...")
        model = InsiderClusterModel()
        
        for ticker in MAG7:
            logger.debug(f"Fetching insider transactions for {ticker}")
            sig = model.get_signal(ticker)
            
            # Score based on cluster presence and strength
            if sig.has_cluster:
                strength_map = {'strong': 0.9, 'moderate': 0.7, 'weak': 0.6}
                base_score = strength_map.get(sig.strength, 0.5)
                if sig.direction == 'long':
                    score = base_score
                    signal_type = "bullish"
                elif sig.direction == 'short':
                    score = 1 - base_score
                    signal_type = "bearish"
                else:
                    score = 0.5
                    signal_type = "neutral"
            else:
                score = 0.5
                signal_type = "neutral"
            
            logger.info(f"{ticker}: cluster={sig.has_cluster}, direction={sig.direction}, strength={sig.strength}")
            
            signals[ticker] = ModelSignal(
                model_name="InsiderCluster",
                ticker=ticker,
                score=score,
                signal=signal_type,
                recommendation=signal_to_recommendation(score),
                details={
                    'has_cluster': sig.has_cluster,
                    'direction': sig.direction,
                    'strength': sig.strength,
                    'num_insiders': sig.num_insiders if hasattr(sig, 'num_insiders') else None,
                },
                reasoning=f"{'CLUSTER detected' if sig.has_cluster else 'No cluster'}: {sig.direction or 'N/A'} ({sig.strength or 'N/A'})"
            )
        
        print_signal_table(signals)
        print("  [OK] InsiderClusterModel complete")
        
    except Exception as e:
        logger.error(f"InsiderClusterModel failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals


def test_pead_model() -> Dict[str, ModelSignal]:
    """Test EnhancedPEADModel with detailed output."""
    print_model_header(4, "ENHANCED PEAD MODEL", "Post-Earnings Announcement Drift detection")
    
    signals = {}
    try:
        from auto_researcher.models.pead_enhanced import EnhancedPEADModel
        
        logger.info("Initializing EnhancedPEADModel...")
        model = EnhancedPEADModel()
        
        for ticker in MAG7:
            logger.debug(f"Getting PEAD signal for {ticker}")
            sig = model.get_signal(ticker)
            
            # Convert direction to score
            if sig.direction == 'long':
                score = 0.5 + min(0.4, sig.expected_return * 10)  # Scale expected return
            elif sig.direction == 'short':
                score = 0.5 - min(0.4, abs(sig.expected_return) * 10)
            else:
                score = 0.5
            
            signal_type = "bullish" if sig.direction == 'long' else \
                          "bearish" if sig.direction == 'short' else "neutral"
            
            logger.info(f"{ticker}: direction={sig.direction}, SUE={sig.sue:.3f}, expected={sig.expected_return*100:.1f}%")
            
            signals[ticker] = ModelSignal(
                model_name="EnhancedPEAD",
                ticker=ticker,
                score=score,
                signal=signal_type,
                recommendation=signal_to_recommendation(score),
                details={
                    'direction': sig.direction,
                    'sue': sig.sue,
                    'expected_return': sig.expected_return,
                    'is_actionable': sig.is_actionable,
                },
                reasoning=f"{sig.direction.upper() if sig.direction else 'NEUTRAL'}: SUE={sig.sue:.2f}, expected {sig.expected_return*100:.1f}%"
            )
        
        print_signal_table(signals)
        print("  [OK] EnhancedPEADModel complete")
        
    except Exception as e:
        logger.error(f"EnhancedPEADModel failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals


def test_early_adopter() -> Dict[str, ModelSignal]:
    """Test EarlyAdopterModel with detailed output."""
    print_model_header(5, "EARLY ADOPTER MODEL", "Tech Pioneer Detection via earnings transcripts")
    
    signals = {}
    try:
        from auto_researcher.models.early_adopter import EarlyAdopterModel
        
        logger.info("Initializing EarlyAdopterModel...")
        model = EarlyAdopterModel()
        
        for ticker in MAG7:
            logger.debug(f"Analyzing tech adoption for {ticker}")
            sig = model.analyze_company(ticker)
            
            logger.info(f"{ticker}: pioneer_score={sig.pioneer_score:.2f}, signal={sig.signal}, "
                       f"tech_count={sig.total_techs_adopted}, early_adopted={sig.techs_adopted_early}")
            
            signals[ticker] = ModelSignal(
                model_name="EarlyAdopter",
                ticker=ticker,
                score=sig.pioneer_score,
                signal=sig.signal.replace('_', ' '),
                recommendation=signal_to_recommendation(sig.pioneer_score),
                details={
                    'tech_count': sig.total_techs_adopted,
                    'early_adopted': sig.techs_adopted_early,
                    'genai_score': sig.genai_score,
                    'genai_adopted': sig.genai_adopted[:5] if sig.genai_adopted else [],
                },
                reasoning=f"Pioneer={sig.pioneer_score:.2f}, {sig.total_techs_adopted} techs mentioned, {sig.techs_adopted_early} adopted early"
            )
        
        # Cleanup
        model.clear_cache()
        
        print_signal_table(signals)
        print("  [OK] EarlyAdopterModel complete")
        
    except Exception as e:
        logger.error(f"EarlyAdopterModel failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals


def test_quality_value() -> Dict[str, ModelSignal]:
    """Test QualityValueModel with detailed output."""
    print_model_header(6, "QUALITY VALUE MODEL", "Fundamental quality and value analysis")
    
    signals = {}
    try:
        from auto_researcher.models.quality_value import QualityValueModel
        
        logger.info("Initializing QualityValueModel...")
        model = QualityValueModel()
        
        for ticker in MAG7:
            logger.debug(f"Analyzing quality-value for {ticker}")
            sig = model.get_signal(ticker)
            
            # Normalize composite to 0-1
            normalized_score = sig.composite_score / 100.0
            
            signal_type = "bullish" if normalized_score >= 0.6 else \
                          "bearish" if normalized_score <= 0.4 else "neutral"
            
            logger.info(f"{ticker}: quality={sig.quality_score:.0f}, value={sig.value_score:.0f}, "
                       f"composite={sig.composite_score:.0f} (normalized={normalized_score:.2f})")
            
            signals[ticker] = ModelSignal(
                model_name="QualityValue",
                ticker=ticker,
                score=normalized_score,
                signal=signal_type,
                recommendation=signal_to_recommendation(normalized_score),
                details={
                    'quality_score': sig.quality_score,
                    'value_score': sig.value_score,
                    'raw_composite': sig.composite_score,
                },
                reasoning=f"Quality={sig.quality_score:.0f}, Value={sig.value_score:.0f} -> {sig.composite_score:.0f}/100"
            )
        
        print_signal_table(signals)
        print("  [OK] QualityValueModel complete")
        
    except Exception as e:
        logger.error(f"QualityValueModel failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals


def test_sentiment_agent() -> Dict[str, ModelSignal]:
    """Test SentimentAgent (uses FinBERT + TopicSentiment models)."""
    print_model_header(7, "SENTIMENT AGENT", "Combines FinBERT + TopicSentiment for news analysis")
    
    signals = {}
    try:
        from auto_researcher.agents.sentiment_agent import SentimentAgent
        
        logger.info("Initializing SentimentAgent with FinBERT...")
        agent = SentimentAgent(finbert_only=True)
        
        for ticker in MAG7:
            logger.debug(f"Running full sentiment analysis for {ticker}")
            result = agent.analyze_ticker(ticker)
            
            if result:
                score = result.overall_score if hasattr(result, 'overall_score') else 0.5
                signal_type = result.overall_signal if hasattr(result, 'overall_signal') else 'neutral'
                
                logger.info(f"{ticker}: signal={signal_type}, score={score:.3f}")
                
                # Log component scores if available
                if hasattr(result, 'finbert_score') and result.finbert_score:
                    logger.debug(f"  FinBERT: {result.finbert_score:.3f}")
                if hasattr(result, 'topic_score') and result.topic_score:
                    logger.debug(f"  TopicSentiment: {result.topic_score:.3f}")
                
                signals[ticker] = ModelSignal(
                    model_name="SentimentAgent",
                    ticker=ticker,
                    score=score,
                    signal=signal_type,
                    recommendation=signal_to_recommendation(score),
                    details={
                        'finbert_score': getattr(result, 'finbert_score', None),
                        'topic_score': getattr(result, 'topic_score', None),
                        'article_count': getattr(result, 'article_count', None),
                    },
                    reasoning=f"Combined sentiment: {signal_type} ({score:.3f})"
                )
            else:
                signals[ticker] = ModelSignal(
                    model_name="SentimentAgent",
                    ticker=ticker,
                    score=0.5,
                    signal="neutral",
                    recommendation="HOLD",
                    reasoning="No data available"
                )
        
        print_signal_table(signals)
        print("  [OK] SentimentAgent complete")
        
    except Exception as e:
        logger.error(f"SentimentAgent failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals


def test_ml_ranking() -> Dict[str, ModelSignal]:
    """Test ML Ranking Model."""
    print_model_header(8, "ML RANKING MODEL", "XGBoost-based stock ranking")
    
    signals = {}
    try:
        from auto_researcher.models.xgb_ranking_model import XGBRankingModel
        
        logger.info("Initializing XGBRankingModel...")
        model = XGBRankingModel()
        
        if model.model is None:
            logger.warning("Model not trained - using mock scores based on ticker order")
            for i, ticker in enumerate(MAG7):
                score = 0.5 + (0.1 * (len(MAG7) - i - 1) / len(MAG7))
                signals[ticker] = ModelSignal(
                    model_name="XGBRanking",
                    ticker=ticker,
                    score=score,
                    signal="mock",
                    recommendation=signal_to_recommendation(score),
                    reasoning="Model not trained - using mock score"
                )
        else:
            logger.info("Model is trained - generating real predictions")
            # Would need to load features for prediction
            for ticker in MAG7:
                signals[ticker] = ModelSignal(
                    model_name="XGBRanking",
                    ticker=ticker,
                    score=0.5,
                    signal="neutral",
                    recommendation="HOLD",
                    reasoning="Feature data needed for prediction"
                )
        
        print_signal_table(signals)
        print("  [OK] XGBRankingModel complete")
        
    except Exception as e:
        logger.error(f"XGBRankingModel failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals


# ============================================================================
# AGGREGATION & FINAL REPORT
# ============================================================================

def print_detailed_breakdown(all_signals: Dict[str, Dict[str, ModelSignal]]):
    """Print detailed breakdown of all model signals per stock."""
    print("\n" + "=" * 90)
    print("  DETAILED MODEL BREAKDOWN BY STOCK")
    print("=" * 90)
    
    model_names = list(all_signals.keys())
    
    for ticker in MAG7:
        print(f"\n  {ticker}")
        print("  " + "-" * 80)
        print(f"  {'Model':<20} {'Score':<8} {'Signal':<12} {'Rec':<14} {'Details'}")
        print("  " + "-" * 80)
        
        total_score = 0
        count = 0
        
        for model_name, signals in all_signals.items():
            if ticker in signals:
                s = signals[ticker]
                total_score += s.score
                count += 1
                print(f"  {model_name:<20} {s.score:.3f}    {s.signal:<12} {s.recommendation:<14} {s.reasoning[:35]}...")
        
        if count > 0:
            avg_score = total_score / count
            print("  " + "-" * 80)
            print(f"  {'AVERAGE':<20} {avg_score:.3f}    {'---':<12} {signal_to_recommendation(avg_score):<14}")
    
    print()


def print_final_rankings(all_signals: Dict[str, Dict[str, ModelSignal]]):
    """Print final rankings based on averaged scores."""
    print("\n" + "=" * 90)
    print("  FINAL COMPOSITE RANKINGS")
    print("=" * 90)
    
    # Calculate average scores
    rankings = []
    for ticker in MAG7:
        scores = []
        for model_name, signals in all_signals.items():
            if ticker in signals:
                scores.append(signals[ticker].score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.5
        rankings.append((ticker, avg_score, signal_to_recommendation(avg_score)))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print()
    print(f"  {'Rank':<6} {'Ticker':<10} {'Score':<10} {'Recommendation'}")
    print("  " + "-" * 50)
    for i, (ticker, score, rec) in enumerate(rankings, 1):
        print(f"  {i:<6} {ticker:<10} {score:.4f}    {rec}")
    
    print("\n  " + "=" * 50)
    print("  TOP PICKS:")
    print("  " + "=" * 50)
    for i, (ticker, score, rec) in enumerate(rankings[:3], 1):
        print(f"  #{i} {ticker}: {rec} (composite={score:.4f})")
    
    print()


def main():
    """Run detailed model tests."""
    print()
    print("*" * 90)
    print("*" + " " * 88 + "*")
    print("*   DETAILED MODEL & AGENT TEST - MAGNIFICENT 7 STOCKS" + " " * 33 + "*")
    print("*   Shows each model's specific signal and reasoning" + " " * 35 + "*")
    print("*" + " " * 88 + "*")
    print("*" * 90)
    print()
    print("Testing: AAPL, NVDA, MSFT, GOOGL, AMZN, META, TSLA")
    print("Logging level: DEBUG (shows all model calls)")
    print()
    
    start_time = time.time()
    
    # Run all tests
    all_signals = {}
    
    all_signals['SectorMomentum'] = test_sector_momentum()
    all_signals['TopicSentiment'] = test_topic_sentiment()
    all_signals['InsiderCluster'] = test_insider_cluster()
    all_signals['EnhancedPEAD'] = test_pead_model()
    all_signals['EarlyAdopter'] = test_early_adopter()
    all_signals['QualityValue'] = test_quality_value()
    all_signals['SentimentAgent'] = test_sentiment_agent()
    all_signals['XGBRanking'] = test_ml_ranking()
    
    # Print detailed breakdown
    print_detailed_breakdown(all_signals)
    
    # Print final rankings
    print_final_rankings(all_signals)
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f} seconds")
    print()


if __name__ == "__main__":
    main()
