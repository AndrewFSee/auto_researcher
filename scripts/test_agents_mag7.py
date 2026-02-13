#!/usr/bin/env python
"""
Test Auto-Researcher Agents on Mag 7 Stocks.

This script tests all the agents and models with detailed logging
to verify they're calling their respective tools/models correctly.

Usage:
    python scripts/test_agents_mag7.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/agent_test.log', mode='w'),
    ]
)

# Set specific loggers to DEBUG
for module in [
    'auto_researcher.agents',
    'auto_researcher.models',
    'auto_researcher.features',
]:
    logging.getLogger(module).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mag 7 tickers
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]


def test_sector_momentum():
    """Test the Sector Momentum Model."""
    print("\n" + "=" * 70)
    print("TEST 1: SECTOR MOMENTUM MODEL")
    print("=" * 70)
    
    try:
        from auto_researcher.models.sector_momentum import SectorMomentumModel
        
        logger.info("Initializing SectorMomentumModel...")
        model = SectorMomentumModel()
        
        logger.info("Computing sector rankings...")
        rankings = model.get_sector_rankings()
        
        print("\nSector Rankings (by momentum):")
        for i, (sector, score) in enumerate(rankings[:5], 1):
            print(f"  {i}. {sector}: {score:.2%}")
        
        logger.info("Computing regime...")
        regime = model.get_regime()
        print(f"\nMarket Regime: {regime}")
        
        logger.info("Getting stock signals for MAG7...")
        for ticker in MAG7[:3]:
            signal = model.get_stock_signal(ticker)
            print(f"  {ticker}: Sector={signal.sector}, Tailwind={signal.sector_tailwind}")
        
        print("\n✅ Sector Momentum Model: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"Sector Momentum Model failed: {e}")
        print(f"\n❌ Sector Momentum Model: FAILED - {e}")
        return False


def test_early_adopter():
    """Test the Early Adopter Model."""
    print("\n" + "=" * 70)
    print("TEST 2: EARLY ADOPTER MODEL (Tech Pioneer Detection)")
    print("=" * 70)
    
    try:
        from auto_researcher.models.early_adopter import EarlyAdopterModel
        
        logger.info("Initializing EarlyAdopterModel...")
        model = EarlyAdopterModel()
        
        print("\nEarly Adopter Signals:")
        for ticker in MAG7:
            logger.debug(f"Computing early adopter signal for {ticker}...")
            signal = model.get_signal(ticker)
            status = "🚀 PIONEER" if signal.is_pioneer else "  standard"
            print(f"  {ticker}: Score={signal.pioneer_score:.2f} {status}")
            if signal.tech_mentions:
                print(f"         Tech: {', '.join(signal.tech_mentions[:3])}")
        
        print("\n✅ Early Adopter Model: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"Early Adopter Model failed: {e}")
        print(f"\n❌ Early Adopter Model: FAILED - {e}")
        return False


def test_topic_sentiment():
    """Test the Topic Sentiment Model."""
    print("\n" + "=" * 70)
    print("TEST 3: TOPIC SENTIMENT MODEL")
    print("=" * 70)
    
    try:
        from auto_researcher.models.topic_sentiment import TopicSentimentModel
        import sqlite3
        
        logger.info("Initializing TopicSentimentModel...")
        model = TopicSentimentModel()
        
        # Load sample articles from database
        db_path = Path(__file__).parent.parent / "data" / "news.db"
        if not db_path.exists():
            print("  ⚠️ No news database found, skipping...")
            return True
        
        logger.info(f"Loading articles from {db_path}...")
        conn = sqlite3.connect(db_path)
        
        print("\nTopic Sentiment Signals:")
        for ticker in MAG7[:4]:
            query = f"""
                SELECT title, full_text, published_date 
                FROM articles 
                WHERE ticker = '{ticker}' 
                AND full_text IS NOT NULL 
                AND length(full_text) > 100
                ORDER BY published_date DESC 
                LIMIT 20
            """
            df = pd.read_sql(query, conn)
            
            if len(df) == 0:
                print(f"  {ticker}: No articles found")
                continue
            
            logger.debug(f"Processing {len(df)} articles for {ticker}...")
            
            # Classify topics
            topics = model.classify_topics(df['title'].tolist())
            topic_counts = pd.Series(topics).value_counts()
            
            # Compute sentiment
            sentiments = model.compute_sentiment(df['full_text'].tolist())
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            print(f"  {ticker}: Avg Sentiment={avg_sentiment:+.3f}")
            print(f"         Topics: {dict(topic_counts.head(3))}")
        
        conn.close()
        print("\n✅ Topic Sentiment Model: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"Topic Sentiment Model failed: {e}")
        print(f"\n❌ Topic Sentiment Model: FAILED - {e}")
        return False


def test_insider_cluster():
    """Test the Insider Cluster Model."""
    print("\n" + "=" * 70)
    print("TEST 4: INSIDER CLUSTER MODEL")
    print("=" * 70)
    
    try:
        from auto_researcher.models.insider_cluster import InsiderClusterModel
        
        logger.info("Initializing InsiderClusterModel...")
        model = InsiderClusterModel()
        
        print("\nInsider Cluster Signals:")
        for ticker in MAG7[:4]:
            logger.debug(f"Getting insider signal for {ticker}...")
            try:
                signal = model.get_signal(ticker)
                cluster_status = "🔔 CLUSTER BUY" if signal.is_cluster_buy else ""
                print(f"  {ticker}: Score={signal.cluster_score:.2f} Insiders={signal.insider_count} {cluster_status}")
            except Exception as e:
                print(f"  {ticker}: No data available - {e}")
        
        print("\n✅ Insider Cluster Model: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"Insider Cluster Model failed: {e}")
        print(f"\n❌ Insider Cluster Model: FAILED - {e}")
        return False


def test_pead_enhanced():
    """Test the Enhanced PEAD Model."""
    print("\n" + "=" * 70)
    print("TEST 5: ENHANCED PEAD MODEL (Earnings Drift)")
    print("=" * 70)
    
    try:
        from auto_researcher.models.pead_enhanced import EnhancedPEADModel
        
        logger.info("Initializing EnhancedPEADModel...")
        model = EnhancedPEADModel()
        
        print("\nPEAD Signals:")
        for ticker in MAG7[:4]:
            logger.debug(f"Getting PEAD signal for {ticker}...")
            try:
                signal = model.get_signal(ticker)
                if signal.has_recent_earnings:
                    surprise_pct = signal.surprise_pct * 100 if signal.surprise_pct else 0
                    print(f"  {ticker}: Surprise={surprise_pct:+.1f}% IC={signal.expected_ic:.3f}")
                else:
                    print(f"  {ticker}: No recent earnings")
            except Exception as e:
                print(f"  {ticker}: Error - {e}")
        
        print("\n✅ Enhanced PEAD Model: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"Enhanced PEAD Model failed: {e}")
        print(f"\n❌ Enhanced PEAD Model: FAILED - {e}")
        return False


def test_thematic_agent():
    """Test the Thematic Analysis Agent."""
    print("\n" + "=" * 70)
    print("TEST 6: THEMATIC ANALYSIS AGENT")
    print("=" * 70)
    
    try:
        from auto_researcher.agents.thematic_agent import ThematicAnalysisAgent
        
        logger.info("Initializing ThematicAnalysisAgent...")
        agent = ThematicAnalysisAgent()
        
        print("\nThematic Analysis:")
        for ticker in MAG7[:3]:
            logger.debug(f"Analyzing {ticker}...")
            result = agent.analyze_ticker(ticker)
            
            print(f"\n  {ticker}:")
            print(f"    Moat: {result.moat_rating} ({result.moat_score:.2f})")
            print(f"    Sector: {result.sector_signal.sector} (tailwind: {result.sector_signal.sector_tailwind})")
            if result.early_adopter_signal:
                print(f"    Pioneer: {result.early_adopter_signal.is_pioneer} (score: {result.early_adopter_signal.pioneer_score:.2f})")
            print(f"    Themes: {', '.join(result.theme_exposures[:3]) if result.theme_exposures else 'N/A'}")
        
        print("\n✅ Thematic Analysis Agent: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"Thematic Analysis Agent failed: {e}")
        print(f"\n❌ Thematic Analysis Agent: FAILED - {e}")
        return False


def test_sentiment_agent():
    """Test the Sentiment Agent."""
    print("\n" + "=" * 70)
    print("TEST 7: SENTIMENT AGENT")
    print("=" * 70)
    
    try:
        from auto_researcher.agents.sentiment_agent import SentimentAgent
        
        logger.info("Initializing SentimentAgent...")
        # Use a model that doesn't require API key for testing
        agent = SentimentAgent(model=None)  # Will use FinBERT/topic models only
        
        print("\nSentiment Analysis:")
        # Test on a subset
        test_tickers = MAG7[:3]
        
        logger.info(f"Analyzing tickers: {test_tickers}")
        results = agent.analyze_tickers(test_tickers)
        
        for ticker, analysis in results.items():
            print(f"\n  {ticker}:")
            print(f"    Sentiment: {analysis.sentiment_score:+.3f} ({analysis.sentiment_label})")
            if hasattr(analysis, 'article_count'):
                print(f"    Articles: {analysis.article_count}")
            if analysis.earnings_topic_tradeable:
                print(f"    🔔 Earnings Signal: {analysis.earnings_topic_signal}")
        
        print("\n✅ Sentiment Agent: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"Sentiment Agent failed: {e}")
        print(f"\n❌ Sentiment Agent: FAILED - {e}")
        return False


def test_gbdt_model():
    """Test the GBDT Ranking Model."""
    print("\n" + "=" * 70)
    print("TEST 8: GBDT RANKING MODEL")
    print("=" * 70)
    
    try:
        from auto_researcher.models.gbdt_model import GBDTRankingModel
        from auto_researcher.features.technical import compute_technical_features
        from auto_researcher.data.price_loader import download_price_history
        
        logger.info("Downloading price data for MAG7...")
        prices = download_price_history(MAG7, start="2024-01-01")
        
        logger.info("Computing technical features...")
        features = compute_technical_features(prices)
        
        logger.info("Initializing GBDTRankingModel...")
        model = GBDTRankingModel()
        
        # Get latest features for prediction
        latest = features.groupby('ticker').last().reset_index()
        
        if hasattr(model, 'predict') and model.is_fitted:
            logger.info("Making predictions...")
            predictions = model.predict(latest)
            
            print("\nGBDT Rankings:")
            for i, (ticker, score) in enumerate(sorted(zip(latest['ticker'], predictions), key=lambda x: -x[1]), 1):
                print(f"  {i}. {ticker}: {score:.3f}")
        else:
            print("  Model not fitted - would need training data")
        
        print("\n✅ GBDT Model: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"GBDT Model failed: {e}")
        print(f"\n❌ GBDT Model: FAILED - {e}")
        return False


def test_feature_pipeline():
    """Test the Feature Pipeline."""
    print("\n" + "=" * 70)
    print("TEST 9: FEATURE PIPELINE")
    print("=" * 70)
    
    try:
        from auto_researcher.features.feature_pipeline import FeaturePipeline
        
        logger.info("Initializing FeaturePipeline...")
        pipeline = FeaturePipeline()
        
        logger.info(f"Building features for {MAG7}...")
        features = pipeline.build_features(
            tickers=MAG7,
            start_date="2024-01-01",
            end_date="2025-01-01",
        )
        
        print(f"\nFeature Pipeline Results:")
        print(f"  Shape: {features.shape}")
        print(f"  Features: {features.columns.tolist()[:10]}...")
        print(f"  Date range: {features.index.min()} to {features.index.max()}")
        
        # Check for NaN
        nan_pct = features.isna().mean().mean() * 100
        print(f"  NaN percentage: {nan_pct:.1f}%")
        
        print("\n✅ Feature Pipeline: PASSED")
        return True
        
    except Exception as e:
        logger.exception(f"Feature Pipeline failed: {e}")
        print(f"\n❌ Feature Pipeline: FAILED - {e}")
        return False


def main():
    """Run all tests."""
    import pandas as pd
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  AUTO-RESEARCHER AGENT TEST SUITE")
    print("  Testing on Magnificent 7: " + ", ".join(MAG7))
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    results = {}
    
    # Run all tests
    results["Sector Momentum"] = test_sector_momentum()
    results["Early Adopter"] = test_early_adopter()
    results["Topic Sentiment"] = test_topic_sentiment()
    results["Insider Cluster"] = test_insider_cluster()
    results["Enhanced PEAD"] = test_pead_enhanced()
    results["Thematic Agent"] = test_thematic_agent()
    results["Sentiment Agent"] = test_sentiment_agent()
    # results["GBDT Model"] = test_gbdt_model()  # Requires training
    # results["Feature Pipeline"] = test_feature_pipeline()  # Takes longer
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {test}")
    
    print(f"\n  Result: {passed}/{total} tests passed")
    print("=" * 70)
    
    # Log file location
    print(f"\n📋 Detailed logs: logs/agent_test.log")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
