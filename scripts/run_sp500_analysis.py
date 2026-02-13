#!/usr/bin/env python
"""
Full S&P 500 Analysis with Auto-Researcher.

Runs all models on the entire S&P 500 with:
- Progress tracking and ETA
- Memory management (clearing caches between batches)
- Error handling for individual stocks
- Results saved to CSV
- Resume capability if interrupted

Usage:
    python scripts/run_sp500_analysis.py
    python scripts/run_sp500_analysis.py --resume  # Continue from last run
    python scripts/run_sp500_analysis.py --batch-size 25
"""

import logging
import sys
import os
import time
import json
import gc
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

sys.path.insert(0, 'src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# Quiet noisy loggers
for quiet in ['urllib3', 'httpx', 'httpcore', 'yfinance', 'filelock', 
              'huggingface_hub', 'transformers', 'datasets', 'tqdm', 'fsspec']:
    logging.getLogger(quiet).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ==============================================================================
# S&P 500 TICKER LIST (as of 2024)
# ==============================================================================

SP500_TICKERS = [
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "ACN", "CSCO",
    "IBM", "INTC", "QCOM", "TXN", "NOW", "INTU", "AMAT", "ADI", "LRCX", "MU",
    "KLAC", "SNPS", "CDNS", "APH", "MSI", "FTNT", "PANW", "ANSS", "KEYS", "FSLR",
    "MPWR", "MCHP", "ON", "CTSH", "HPQ", "HPE", "DELL", "WDC", "STX", "NTAP",
    "AKAM", "JNPR", "ZBRA", "TDY", "TER", "SWKS", "QRVO", "NXPI", "GLW", "TEL",
    "IT", "GDDY", "EPAM", "CDW", "ENPH", "SMCI", "PLTR", "CRWD",
    
    # Communication Services
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "WBD", "PARA", "LYV", "OMC", "IPG", "MTCH", "FOX", "FOXA",
    "NWS", "NWSA",
    
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ORLY",
    "MAR", "CMG", "GM", "F", "AZO", "ROST", "DHI", "LEN", "YUM", "DARDEN",
    "HLT", "GRMN", "POOL", "BBY", "EBAY", "ULTA", "LVS", "WYNN", "MGM", "RCL",
    "CCL", "NCLH", "DRI", "EXPE", "APTV", "BWA", "LULU", "ETSY", "TPR", "RL",
    "VFC", "PVH", "HAS", "WHR", "MHK", "NVR", "PHM", "TOL", "LKQ", "KMX",
    "GPC", "AAP", "DG", "DLTR", "TSCO", "CASY",
    
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "EL",
    "KMB", "GIS", "K", "HSY", "SJM", "CAG", "CPB", "HRL", "MKC", "CHD",
    "CLX", "KHC", "ADM", "BG", "TSN", "SYY", "KR", "WBA", "TAP", "STZ",
    "BF.B", "MNST", "KVUE",
    
    # Health Care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "VRTX", "REGN", "MDT", "ISRG", "SYK", "BDX", "ZBH", "EW",
    "BSX", "CI", "ELV", "HUM", "CNC", "MCK", "CAH", "ABC", "CVS", "WBA",
    "DXCM", "IDXX", "IQV", "A", "MTD", "WAT", "TECH", "HOLX", "ALGN", "RMD",
    "COO", "BIIB", "MRNA", "CTLT", "CRL", "DGX", "LH", "HSIC", "MOH", "HCA",
    "UHS", "DVA",
    
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
    "C", "AXP", "SCHW", "CB", "MMC", "PGR", "AON", "CME", "ICE", "MCO",
    "PNC", "USB", "TFC", "COF", "AIG", "MET", "PRU", "AFL", "TRV", "ALL",
    "AJG", "MSCI", "FDS", "NDAQ", "CBOE", "TROW", "BEN", "IVZ", "NTRS", "STT",
    "KEY", "CFG", "RF", "HBAN", "FITB", "MTB", "ZION", "CMA", "FHN", "SIVB",
    "FRC", "SBNY", "DFS", "SYF", "ALLY", "AMP", "RJF", "LPLA", "MKTX", "CINF",
    "L", "GL", "RE", "WRB", "AIZ", "BRO",
    
    # Industrials
    "GE", "CAT", "UNP", "HON", "UPS", "RTX", "BA", "DE", "LMT", "GD",
    "NOC", "TDG", "ITW", "EMR", "ETN", "PH", "ROK", "CMI", "PCAR", "CTAS",
    "FDX", "CSX", "NSC", "ODFL", "JBHT", "CHRW", "DAL", "UAL", "LUV", "AAL",
    "WM", "RSG", "WCN", "VRSK", "CPRT", "IR", "XYL", "DOV", "GNRC", "AME",
    "SWK", "FAST", "NDSN", "GWW", "WSO", "AOS", "RRX", "HWM", "TXT", "GPN",
    "PAYC", "PAYX", "LDOS", "BAH", "LHX", "AXON", "BLDR", "HII", "PWR", "J",
    "CARR", "OTIS", "WAB", "HUBB", "ALLE", "MAS", "IEX", "SNA", "TT", "POOL",
    
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "PXD",
    "DVN", "HES", "FANG", "HAL", "BKR", "KMI", "WMB", "OKE", "TRGP", "EQT",
    "CTRA", "MRO", "APA",
    
    # Materials
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "STLD", "VMC", "MLM",
    "DOW", "DD", "PPG", "ALB", "CTVA", "FMC", "IFF", "CE", "EMN", "LYB",
    "CF", "MOS", "BALL", "PKG", "IP", "AVY", "SEE", "WRK", "AMCR",
    
    # Real Estate
    "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "AVB",
    "EQR", "ESS", "MAA", "UDR", "CPT", "VTR", "PEAK", "ARE", "BXP", "SLG",
    "KIM", "REG", "FRT", "VNO", "HST", "CBRE", "JLL", "CSGP", "IRM", "EXR",
    "CUBE", "SBAC", "WY", "RYN",
    
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC",
    "PEG", "ES", "AWK", "DTE", "ETR", "FE", "PPL", "AEE", "CMS", "CNP",
    "EVRG", "LNT", "ATO", "NI", "PNW", "NRG",
]

# Remove duplicates and sort
SP500_TICKERS = sorted(list(set(SP500_TICKERS)))

# Output directory
OUTPUT_DIR = Path("results")
RESULTS_FILE = OUTPUT_DIR / f"sp500_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
PROGRESS_FILE = OUTPUT_DIR / "sp500_progress.json"


@dataclass
class StockAnalysis:
    """Complete analysis results for a single stock."""
    ticker: str
    analysis_date: str = ""
    
    # Model scores (0-1 scale)
    sector_momentum_score: float = 0.5
    sector_direction: str = "neutral"
    sector_boost: float = 0.0
    
    topic_sentiment_score: float = 0.5
    topic_sentiment_signal: str = "neutral"
    
    insider_cluster_score: float = 0.5
    insider_has_cluster: bool = False
    insider_direction: str = ""
    insider_strength: str = ""
    
    pead_score: float = 0.5
    pead_direction: str = "neutral"
    pead_sue: float = 0.0
    pead_expected_return: float = 0.0
    
    early_adopter_score: float = 0.5
    early_adopter_signal: str = "neutral"
    early_adopter_techs: int = 0
    early_adopter_early: int = 0
    
    quality_value_score: float = 0.5
    quality_score: float = 0.5  # Normalized 0-1 (was 0-100)
    value_score: float = 0.5  # Normalized 0-1 (was 0-100)
    
    sentiment_agent_score: float = 0.5
    sentiment_agent_signal: str = "neutral"
    sentiment_news_count: int = 0
    
    # Composite
    composite_score: float = 0.5
    recommendation: str = "HOLD"
    
    # Metadata
    errors: List[str] = field(default_factory=list)
    processing_time_sec: float = 0.0


def score_to_recommendation(score: float) -> str:
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


class SP500Analyzer:
    """Runs comprehensive analysis on S&P 500 stocks."""
    
    def __init__(self, batch_size: int = 20):
        self.batch_size = batch_size
        self.results: Dict[str, StockAnalysis] = {}
        self.completed_tickers: List[str] = []
        self.failed_tickers: List[str] = []
        
        # Models (lazy loaded)
        self._sector_model = None
        self._topic_model = None
        self._insider_model = None
        self._pead_model = None
        self._early_adopter_model = None
        self._quality_value_model = None
        self._sentiment_agent = None
        
        # Database connection for topic sentiment
        self._news_conn = None
        
        # Statistics
        self.start_time = None
        self.stocks_processed = 0
        
    def _init_models(self):
        """Initialize all models (lazy loading)."""
        logger.info("Initializing models...")
        
        try:
            from auto_researcher.models.sector_momentum import SectorMomentumModel
            self._sector_model = SectorMomentumModel()
            logger.info("  [OK] SectorMomentumModel")
        except Exception as e:
            logger.error(f"  [X] SectorMomentumModel: {e}")
        
        try:
            from auto_researcher.models.topic_sentiment import TopicSentimentModel
            self._topic_model = TopicSentimentModel()
            logger.info("  [OK] TopicSentimentModel")
        except Exception as e:
            logger.error(f"  [X] TopicSentimentModel: {e}")
        
        try:
            from auto_researcher.models.insider_cluster import InsiderClusterModel
            self._insider_model = InsiderClusterModel()
            logger.info("  [OK] InsiderClusterModel")
        except Exception as e:
            logger.error(f"  [X] InsiderClusterModel: {e}")
        
        try:
            from auto_researcher.models.pead_enhanced import EnhancedPEADModel
            self._pead_model = EnhancedPEADModel()
            logger.info("  [OK] EnhancedPEADModel")
        except Exception as e:
            logger.error(f"  [X] EnhancedPEADModel: {e}")
        
        try:
            from auto_researcher.models.early_adopter import EarlyAdopterModel
            self._early_adopter_model = EarlyAdopterModel()
            logger.info("  [OK] EarlyAdopterModel")
        except Exception as e:
            logger.error(f"  [X] EarlyAdopterModel: {e}")
        
        try:
            from auto_researcher.models.quality_value import QualityValueModel
            self._quality_value_model = QualityValueModel()
            logger.info("  [OK] QualityValueModel")
        except Exception as e:
            logger.error(f"  [X] QualityValueModel: {e}")
        
        # Open news database
        try:
            import sqlite3
            self._news_conn = sqlite3.connect('data/news.db')
            logger.info("  [OK] News database connected")
        except Exception as e:
            logger.error(f"  [X] News database: {e}")
        
        # Initialize SentimentAgent (uses FinBERT - no LLM calls needed)
        try:
            from auto_researcher.agents.sentiment_agent import SentimentAgent, SentimentAgentConfig
            config = SentimentAgentConfig(
                finbert_only=True,  # Fast, no LLM calls
                use_topic_model=False,  # We run TopicSentiment separately
                use_scraped_db=True,
                scraped_db_lookback_days=30,
            )
            self._sentiment_agent = SentimentAgent(config=config, finbert_only=True)
            logger.info("  [OK] SentimentAgent (FinBERT mode)")
        except Exception as e:
            logger.error(f"  [X] SentimentAgent: {e}")
            self._sentiment_agent = None
        
        logger.info("Model initialization complete")
    
    def _clear_caches(self):
        """Clear model caches to free memory."""
        if self._early_adopter_model:
            self._early_adopter_model.clear_cache()
        
        # Clear torch CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        gc.collect()
        logger.debug("Memory caches cleared")
    
    def _unload_heavy_models(self):
        \"\"\"Unload heavy models (FinBERT) to free memory.\"\"\"
        if self._sentiment_agent and hasattr(self._sentiment_agent, '_finbert'):
            if self._sentiment_agent._finbert and hasattr(self._sentiment_agent._finbert, 'unload'):
                self._sentiment_agent._finbert.unload()
                logger.info("FinBERT model unloaded to free memory")
        gc.collect()
    
    def analyze_stock(self, ticker: str) -> StockAnalysis:
        """Analyze a single stock with all models."""
        start = time.time()
        result = StockAnalysis(
            ticker=ticker,
            analysis_date=datetime.now().isoformat()
        )
        
        # 1. Sector Momentum
        if self._sector_model:
            try:
                sig = self._sector_model.get_stock_signal(ticker)
                result.sector_boost = sig.sector_boost if sig.sector_boost else 0.0
                result.sector_direction = sig.direction or "neutral"
                result.sector_momentum_score = 0.5 + result.sector_boost
            except Exception as e:
                result.errors.append(f"SectorMomentum: {str(e)[:50]}")
        
        # 2. Topic Sentiment (from local news.db)
        if self._topic_model and self._news_conn:
            try:
                import pandas as pd
                df = pd.read_sql('''
                    SELECT title, full_text, published_date FROM articles 
                    WHERE ticker = ? AND full_text IS NOT NULL 
                    ORDER BY published_date DESC LIMIT 30
                ''', self._news_conn, params=(ticker,))
                
                if len(df) > 0:
                    articles = [
                        {'title': r['title'], 'text': r['full_text'], 'published_date': r['published_date']} 
                        for _, r in df.iterrows()
                    ]
                    topic_result = self._topic_model.analyze_articles(articles, ticker)
                    result.topic_sentiment_score = (topic_result.composite_score + 1) / 2
                    result.topic_sentiment_signal = topic_result.composite_signal
            except Exception as e:
                result.errors.append(f"TopicSentiment: {str(e)[:50]}")
        
        # 3. Insider Cluster
        if self._insider_model:
            try:
                sig = self._insider_model.get_signal(ticker)
                result.insider_has_cluster = sig.has_cluster
                result.insider_direction = sig.direction or ""
                result.insider_strength = sig.strength or ""
                
                if sig.has_cluster:
                    strength_map = {'strong': 0.9, 'moderate': 0.7, 'weak': 0.6}
                    base = strength_map.get(sig.strength, 0.5)
                    if sig.direction == 'long':
                        result.insider_cluster_score = base
                    elif sig.direction == 'short':
                        result.insider_cluster_score = 1 - base
                    else:
                        result.insider_cluster_score = 0.5
            except Exception as e:
                result.errors.append(f"InsiderCluster: {str(e)[:50]}")
        
        # 4. PEAD
        if self._pead_model:
            try:
                sig = self._pead_model.get_signal(ticker)
                result.pead_direction = sig.direction or "neutral"
                result.pead_sue = sig.sue
                result.pead_expected_return = sig.expected_return
                
                if sig.direction == 'long':
                    result.pead_score = 0.5 + min(0.4, sig.expected_return * 10)
                elif sig.direction == 'short':
                    result.pead_score = 0.5 - min(0.4, abs(sig.expected_return) * 10)
            except Exception as e:
                result.errors.append(f"PEAD: {str(e)[:50]}")
        
        # 5. Early Adopter
        if self._early_adopter_model:
            try:
                sig = self._early_adopter_model.analyze_company(ticker)
                result.early_adopter_score = sig.pioneer_score
                result.early_adopter_signal = sig.signal
                result.early_adopter_techs = sig.total_techs_adopted
                result.early_adopter_early = sig.techs_adopted_early
            except Exception as e:
                result.errors.append(f"EarlyAdopter: {str(e)[:50]}")
        
        # 6. Quality Value - normalize all to 0-1 scale
        if self._quality_value_model:
            try:
                sig = self._quality_value_model.get_signal(ticker)
                # Quality and Value are 0-100, normalize to 0-1
                result.quality_score = sig.quality_score / 100.0
                result.value_score = sig.value_score / 100.0
                result.quality_value_score = sig.composite_score / 100.0
            except Exception as e:
                result.errors.append(f"QualityValue: {str(e)[:50]}")
        
        # 7. SentimentAgent (FinBERT on scraped news)
        if self._sentiment_agent:
            try:
                news = self._sentiment_agent.fetch_news(ticker, max_items=20)
                if news:
                    # Get FinBERT scores for each news item
                    scores = []
                    for item in news:
                        text = f"{item.title}. {item.snippet or ''}"
                        if self._sentiment_agent._finbert:
                            # FinBERT.analyze() returns FinBERTResult with sentiment_score
                            result_fb = self._sentiment_agent._finbert.analyze(text)
                            scores.append(result_fb.sentiment_score)
                    
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        # FinBERT returns -1 to +1, convert to 0-1
                        result.sentiment_agent_score = (avg_score + 1) / 2
                        result.sentiment_news_count = len(scores)
                        
                        if avg_score > 0.2:
                            result.sentiment_agent_signal = "bullish"
                        elif avg_score < -0.2:
                            result.sentiment_agent_signal = "bearish"
                        else:
                            result.sentiment_agent_signal = "neutral"
            except Exception as e:
                result.errors.append(f"SentimentAgent: {str(e)[:50]}")
        
        # Calculate composite score with weights
        # Weights based on backtest strength:
        # - PEAD: strong alpha signal
        # - EarlyAdopter: forward-looking innovation signal  
        # - QualityValue: fundamental quality
        # - TopicSentiment: news-based
        # - SentimentAgent: FinBERT on fresh news
        # - SectorMomentum: macro regime
        # - InsiderCluster: insider activity
        weights = {
            'sector_momentum': 0.10,
            'topic_sentiment': 0.15,
            'insider_cluster': 0.10,
            'pead': 0.20,
            'early_adopter': 0.15,
            'quality_value': 0.20,
            'sentiment_agent': 0.10,
        }
        
        weighted_sum = (
            result.sector_momentum_score * weights['sector_momentum'] +
            result.topic_sentiment_score * weights['topic_sentiment'] +
            result.insider_cluster_score * weights['insider_cluster'] +
            result.pead_score * weights['pead'] +
            result.early_adopter_score * weights['early_adopter'] +
            result.quality_value_score * weights['quality_value'] +
            result.sentiment_agent_score * weights['sentiment_agent']
        )
        
        result.composite_score = weighted_sum
        result.recommendation = score_to_recommendation(result.composite_score)
        result.processing_time_sec = time.time() - start
        
        return result
    
    def _save_progress(self):
        """Save current progress to allow resuming."""
        progress = {
            'completed': self.completed_tickers,
            'failed': self.failed_tickers,
            'timestamp': datetime.now().isoformat(),
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f)
    
    def _load_progress(self) -> List[str]:
        """Load previously completed tickers."""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                return progress.get('completed', [])
        return []
    
    def _save_results_csv(self):
        """Save all results to CSV."""
        import pandas as pd
        
        rows = []
        for ticker, analysis in self.results.items():
            row = asdict(analysis)
            row['errors'] = '; '.join(row['errors'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('composite_score', ascending=False)
        df.to_csv(RESULTS_FILE, index=False)
        logger.info(f"Results saved to {RESULTS_FILE}")
    
    def _print_progress(self, current: int, total: int, ticker: str, result: StockAnalysis):
        """Print progress with ETA."""
        elapsed = time.time() - self.start_time
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate if rate > 0 else 0
        eta = timedelta(seconds=int(remaining))
        
        rec_emoji = {
            "STRONG BUY": "🟢",
            "BUY": "🟡",
            "HOLD": "⚪",
            "SELL": "🟠",
            "STRONG SELL": "🔴",
        }.get(result.recommendation, "⚪")
        
        errors_str = f" ({len(result.errors)} errors)" if result.errors else ""
        
        print(f"[{current}/{total}] {ticker:<6} | {result.composite_score:.3f} {rec_emoji} {result.recommendation:<12} | "
              f"ETA: {eta}{errors_str}")
    
    def run(self, tickers: List[str] = None, resume: bool = False):
        """Run analysis on all tickers."""
        if tickers is None:
            tickers = SP500_TICKERS
        
        # Create output directory
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Resume from previous run
        if resume:
            previously_completed = self._load_progress()
            tickers = [t for t in tickers if t not in previously_completed]
            self.completed_tickers = previously_completed
            logger.info(f"Resuming: {len(previously_completed)} already completed, {len(tickers)} remaining")
        
        total = len(tickers)
        logger.info(f"Starting S&P 500 analysis: {total} stocks")
        
        # Initialize models
        self._init_models()
        
        self.start_time = time.time()
        
        print("\n" + "=" * 80)
        print("  S&P 500 COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Stocks: {total}")
        print(f"  Batch Size: {self.batch_size}")
        print("=" * 80 + "\n")
        
        for i, ticker in enumerate(tickers, 1):
            try:
                result = self.analyze_stock(ticker)
                self.results[ticker] = result
                self.completed_tickers.append(ticker)
                self._print_progress(len(self.completed_tickers), total + len(self.completed_tickers) - i, ticker, result)
                
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
                self.failed_tickers.append(ticker)
                # Create minimal result
                self.results[ticker] = StockAnalysis(
                    ticker=ticker,
                    analysis_date=datetime.now().isoformat(),
                    errors=[str(e)]
                )
            
            # Memory management: clear caches every batch_size stocks
            if i % self.batch_size == 0:
                self._clear_caches()
                self._save_progress()
                self._save_results_csv()
                logger.info(f"Checkpoint saved ({i}/{total})")
                
                # More aggressive memory cleanup every 100 stocks
                if i % 100 == 0:
                    self._unload_heavy_models()
                    gc.collect()
                    logger.info(f"Heavy cleanup after {i} stocks")
        
        # Final save
        self._save_progress()
        self._save_results_csv()
        
        # Close database
        if self._news_conn:
            self._news_conn.close()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print analysis summary."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("  ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"  Total Time: {timedelta(seconds=int(elapsed))}")
        print(f"  Stocks Analyzed: {len(self.completed_tickers)}")
        print(f"  Failed: {len(self.failed_tickers)}")
        print(f"  Results saved to: {RESULTS_FILE}")
        print("=" * 80)
        
        # Top 10 stocks
        if self.results:
            sorted_results = sorted(
                self.results.values(),
                key=lambda x: x.composite_score,
                reverse=True
            )
            
            print("\n  TOP 25 STOCKS:")
            print("  " + "-" * 100)
            print(f"  {'Rank':<5} {'Ticker':<8} {'Score':<8} {'Rec':<12} {'Quality':<8} {'Value':<8} {'Pioneer':<8} {'PEAD':<8} {'Sent':<8}")
            print("  " + "-" * 100)
            
            for i, r in enumerate(sorted_results[:25], 1):
                print(f"  {i:<5} {r.ticker:<8} {r.composite_score:.4f}   {r.recommendation:<12} "
                      f"{r.quality_score:.2f}     {r.value_score:.2f}     {r.early_adopter_score:.2f}     {r.pead_score:.2f}     {r.sentiment_agent_score:.2f}")
            
            print("\n  BOTTOM 10 STOCKS:")
            print("  " + "-" * 100)
            for i, r in enumerate(sorted_results[-10:], len(sorted_results) - 9):
                print(f"  {i:<5} {r.ticker:<8} {r.composite_score:.4f}   {r.recommendation:<12} "
                      f"{r.quality_score:.2f}     {r.value_score:.2f}     {r.early_adopter_score:.2f}     {r.pead_score:.2f}     {r.sentiment_agent_score:.2f}")
            
            # Recommendation breakdown
            rec_counts = {}
            for r in sorted_results:
                rec_counts[r.recommendation] = rec_counts.get(r.recommendation, 0) + 1
            
            print("\n  RECOMMENDATION BREAKDOWN:")
            print("  " + "-" * 40)
            for rec in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
                count = rec_counts.get(rec, 0)
                pct = count / len(sorted_results) * 100
                bar = "█" * int(pct / 2)
                print(f"  {rec:<14} {count:>4} ({pct:>5.1f}%) {bar}")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Run S&P 500 Analysis')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--batch-size', type=int, default=15, help='Batch size for memory management (smaller = less memory)')
    parser.add_argument('--test', action='store_true', help='Test mode: only analyze first 10 stocks')
    parser.add_argument('--low-memory', action='store_true', help='Low memory mode: aggressive cleanup, smaller batches')
    args = parser.parse_args()
    
    # Low memory mode adjustments
    if args.low_memory:
        args.batch_size = 10
        logger.info("Low memory mode enabled: batch_size=10, aggressive cleanup")
    
    analyzer = SP500Analyzer(batch_size=args.batch_size)
    
    if args.test:
        # Test mode with just a few stocks
        test_tickers = SP500_TICKERS[:10]
        logger.info(f"Test mode: analyzing {len(test_tickers)} stocks")
        analyzer.run(tickers=test_tickers, resume=False)
    else:
        analyzer.run(resume=args.resume)


if __name__ == "__main__":
    main()
