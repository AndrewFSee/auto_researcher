"""
Multi-Stage Stock Ranking System.

This module implements a two-stage ranking approach:
1. ML Model screens the universe (e.g., S&P 500) → top N candidates
2. Agents analyze candidates with specialized expertise → final ranking

The final composite score combines:
- ML quantitative score (momentum, volatility, etc.)
- Agent qualitative scores (sentiment, fundamentals, SEC, earnings, insider, thematic)

Usage:
    from auto_researcher.ranking_system import RankingSystem
    
    ranker = RankingSystem()
    results = ranker.rank(
        universe="sp500",
        ml_top_n=25,      # ML model picks top 25
        final_top_n=10,   # Final ranked list of 10
    )
    
    for r in results.final_rankings:
        print(f"{r.rank}. {r.ticker}: {r.composite_score:.2f}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class AgentScore:
    """Score from a single agent."""
    agent: str
    score: float  # -1 to +1
    confidence: float  # 0 to 1
    signal: str  # strong_buy, buy, hold, sell, strong_sell
    weighted_score: float = 0.0  # score * weight


@dataclass
class RankedStock:
    """A stock with its complete ranking information."""
    rank: int
    ticker: str
    company_name: str
    
    # ML Model Score
    ml_score: float  # Raw score from XGBoost
    ml_percentile: float  # Percentile rank (0-100)
    ml_weighted: float  # Weighted contribution to composite
    
    # Agent Scores
    agent_scores: dict[str, AgentScore] = field(default_factory=dict)
    agent_composite: float = 0.0  # Combined agent score
    
    # Final Composite
    composite_score: float = 0.0  # ML + Agents combined
    composite_signal: str = "hold"
    
    # Metadata
    sector: str = ""
    key_factors: list[str] = field(default_factory=list)


@dataclass
class RankingResult:
    """Complete ranking results."""
    generated_at: datetime
    universe: str
    universe_size: int
    
    # Stage 1: ML Screening
    ml_candidates: list[RankedStock]
    ml_top_n: int
    
    # Stage 2: Final Rankings
    final_rankings: list[RankedStock]
    final_top_n: int
    
    # Configuration
    ml_weight: float
    agent_weights: dict[str, float]
    
    def to_summary(self) -> str:
        """Generate a text summary of rankings."""
        lines = [
            f"{'='*70}",
            f"  STOCK RANKING RESULTS - {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"{'='*70}",
            f"  Universe: {self.universe} ({self.universe_size} stocks)",
            f"  ML Screened: Top {self.ml_top_n} → Agent Analysis → Top {self.final_top_n}",
            f"",
            f"  Weights: ML={self.ml_weight:.0%}, Agents={1-self.ml_weight:.0%}",
            f"  Agent Breakdown: {', '.join(f'{k}={v:.0%}' for k,v in self.agent_weights.items())}",
            f"{'='*70}",
            f"",
            f"  FINAL RANKINGS",
            f"  {'Rank':<6}{'Ticker':<8}{'Composite':>10}{'ML':>8}{'Agents':>8}{'Signal':<14}",
            f"  {'-'*54}",
        ]
        
        for stock in self.final_rankings:
            lines.append(
                f"  {stock.rank:<6}{stock.ticker:<8}"
                f"{stock.composite_score:>10.2f}"
                f"{stock.ml_weighted:>8.2f}"
                f"{stock.agent_composite:>8.2f}"
                f"  {stock.composite_signal}"
            )
        
        lines.append(f"  {'-'*54}")
        lines.append("")
        
        # Detailed breakdown for top 5
        lines.append("  DETAILED AGENT BREAKDOWN (Top 5)")
        lines.append("")
        
        for stock in self.final_rankings[:5]:
            lines.append(f"  {stock.ticker} - {stock.company_name}")
            lines.append(f"    Composite: {stock.composite_score:.2f} | Signal: {stock.composite_signal.upper()}")
            lines.append(f"    ML: {stock.ml_percentile:.0f}th percentile ({stock.ml_weighted:+.2f})")
            
            for agent, score in stock.agent_scores.items():
                emoji = "🟢" if score.score > 0.1 else "🔴" if score.score < -0.1 else "🟡"
                lines.append(f"    {emoji} {agent:<12}: {score.signal:<12} ({score.score:+.2f} × {score.confidence:.0%})")
            
            if stock.key_factors:
                lines.append(f"    Key Factors: {', '.join(stock.key_factors[:3])}")
            lines.append("")
        
        return "\n".join(lines)


# ==============================================================================
# RANKING SYSTEM
# ==============================================================================

class RankingSystem:
    """
    Multi-stage stock ranking system.
    
    Stage 1: ML Model screens universe → top N candidates
    Stage 2: Agents analyze candidates → final composite ranking
    """
    
    # Default weights for combining scores
    DEFAULT_ML_WEIGHT = 0.35  # ML contributes 35% to final score
    
    DEFAULT_AGENT_WEIGHTS = {
        "sentiment": 0.15,
        "fundamental": 0.20,
        "sec": 0.10,
        "earnings": 0.10,
        "insider": 0.15,
        "thematic": 0.30,  # Forward-looking gets higher weight
    }
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        ml_weight: Optional[float] = None,
        agent_weights: Optional[dict[str, float]] = None,
        use_cache: bool = True,
        parallel: bool = True,
    ):
        """
        Initialize the ranking system.
        
        Args:
            model: LLM model for agents.
            ml_weight: Weight for ML score in composite (0-1).
            agent_weights: Dict of agent name -> weight.
            use_cache: Enable caching for agent results.
            parallel: Run agent analyses in parallel.
        """
        self.model = model
        self.ml_weight = ml_weight or self.DEFAULT_ML_WEIGHT
        self.agent_weights = agent_weights or self.DEFAULT_AGENT_WEIGHTS.copy()
        self.use_cache = use_cache
        self.parallel = parallel
        
        # Normalize agent weights to sum to (1 - ml_weight)
        agent_total = sum(self.agent_weights.values())
        if agent_total > 0:
            scale = (1 - self.ml_weight) / agent_total
            self.agent_weights = {k: v * scale for k, v in self.agent_weights.items()}
        
        # Lazy-loaded components
        self._orchestrator = None
        self._ml_model = None
    
    @property
    def orchestrator(self):
        """Lazy-load the research orchestrator."""
        if self._orchestrator is None:
            from auto_researcher.agents.research_orchestrator import ResearchOrchestrator
            self._orchestrator = ResearchOrchestrator(
                model=self.model,
                use_cache=self.use_cache,
                parallel=self.parallel,
            )
        return self._orchestrator
    
    def _get_universe_tickers(self, universe: str) -> list[str]:
        """Get list of tickers for a universe."""
        # Import universes from recommend.py
        universes = {
            "sp500": self._get_sp500(),
            "sp100": self._get_sp100(),
            "large_cap": self._get_large_cap(),
            "core_tech": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                "AMD", "INTC", "CRM", "ORCL", "ADBE", "NFLX", "PYPL",
            ],
        }
        
        if universe in universes:
            return universes[universe]
        elif isinstance(universe, list):
            return universe
        else:
            raise ValueError(f"Unknown universe: {universe}")
    
    def _get_sp500(self) -> list[str]:
        """Get S&P 500 tickers."""
        try:
            import pandas as pd
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            return [t for t in tickers if isinstance(t, str) and len(t) <= 5]
        except Exception as e:
            logger.warning(f"Failed to fetch S&P 500: {e}")
            # Fallback to stored list
            return self._get_sp100()  # Use S&P 100 as fallback
    
    def _get_sp100(self) -> list[str]:
        """Get S&P 100 tickers."""
        return [
            "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMZN", "AVGO",
            "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT", "CHTR",
            "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "DE",
            "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD", "GE",
            "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
            "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ",
            "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
            "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX",
            "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN",
            "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
        ]
    
    def _get_large_cap(self) -> list[str]:
        """Get large cap tech + finance tickers."""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
            "JPM", "V", "MA", "JNJ", "UNH", "HD", "PG", "XOM", "BAC", "CVX",
            "ABBV", "MRK", "COST", "PEP", "KO", "LLY", "TMO", "AVGO", "ORCL",
            "CRM", "MCD", "ADBE", "AMD", "NFLX", "INTC", "CSCO", "QCOM",
        ]
    
    def _run_ml_screening(
        self,
        tickers: list[str],
        top_n: int,
    ) -> list[RankedStock]:
        """
        Stage 1: Run ML model to screen universe.
        
        Returns top N stocks by ML score.
        """
        logger.info(f"Stage 1: ML screening {len(tickers)} stocks → top {top_n}")
        
        # Import the recommender
        import sys
        sys.path.insert(0, str(__file__).replace("src/auto_researcher/ranking_system.py", ""))
        
        try:
            from recommend import generate_recommendations, Recommendation
        except ImportError:
            # Try alternate import path
            import importlib.util
            import os
            spec = importlib.util.spec_from_file_location(
                "recommend", 
                os.path.join(os.path.dirname(__file__), "..", "..", "recommend.py")
            )
            recommend = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(recommend)
            generate_recommendations = recommend.generate_recommendations
        
        # Run ML model
        recommendations, all_scores, prices = generate_recommendations(
            tickers=tickers,
            top_k=top_n,
            explain=False,
        )
        
        # Convert to RankedStock objects
        ml_candidates = []
        for i, rec in enumerate(recommendations):
            stock = RankedStock(
                rank=i + 1,
                ticker=rec.ticker,
                company_name=rec.ticker,  # Will be updated by agents
                ml_score=rec.score,
                ml_percentile=rec.percentile,
                ml_weighted=self.ml_weight * ((rec.percentile - 50) / 50),  # Normalize to -ml_weight to +ml_weight
                key_factors=rec.drivers[:3] if rec.drivers else [],
            )
            ml_candidates.append(stock)
        
        return ml_candidates
    
    def _run_agent_analysis(
        self,
        candidates: list[RankedStock],
    ) -> list[RankedStock]:
        """
        Stage 2: Run all agents on ML candidates.
        
        Adds agent scores to each RankedStock.
        """
        logger.info(f"Stage 2: Agent analysis on {len(candidates)} candidates")
        
        for stock in candidates:
            ticker = stock.ticker
            logger.info(f"  Analyzing {ticker}...")
            
            try:
                # Run orchestrator
                report = self.orchestrator.analyze(
                    ticker=ticker,
                    ml_percentile=stock.ml_percentile,
                )
                
                # Update company name
                stock.company_name = report.company_name
                
                # Extract agent scores
                signal = report.signal
                
                agent_mapping = {
                    "sentiment": signal.sentiment_signal,
                    "fundamental": signal.fundamental_signal,
                    "sec": signal.sec_signal,
                    "earnings": signal.earnings_signal,
                    "insider": signal.insider_signal,
                    "thematic": signal.thematic_signal,
                }
                
                stock.agent_scores = {}
                agent_composite = 0.0
                
                for agent_name, agent_signal in agent_mapping.items():
                    if agent_signal:
                        weight = self.agent_weights.get(agent_name, 0)
                        weighted = agent_signal.score * weight
                        
                        stock.agent_scores[agent_name] = AgentScore(
                            agent=agent_name,
                            score=agent_signal.score,
                            confidence=agent_signal.confidence,
                            signal=agent_signal.signal,
                            weighted_score=weighted,
                        )
                        agent_composite += weighted
                
                stock.agent_composite = agent_composite
                
                # Get sector from thematic analysis
                if report.thematic_analysis:
                    stock.sector = getattr(report.thematic_analysis, 'sector', '')
                
                # Add catalysts to key factors
                if report.key_catalysts:
                    stock.key_factors.extend(report.key_catalysts[:2])
                
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
                stock.agent_composite = 0.0
        
        return candidates
    
    def _compute_composite_scores(
        self,
        candidates: list[RankedStock],
    ) -> list[RankedStock]:
        """Compute final composite scores and rank."""
        for stock in candidates:
            # Composite = ML weighted + Agent composite
            stock.composite_score = stock.ml_weighted + stock.agent_composite
            
            # Determine signal from composite
            if stock.composite_score >= 0.3:
                stock.composite_signal = "strong_buy"
            elif stock.composite_score >= 0.1:
                stock.composite_signal = "buy"
            elif stock.composite_score > -0.1:
                stock.composite_signal = "hold"
            elif stock.composite_score > -0.3:
                stock.composite_signal = "sell"
            else:
                stock.composite_signal = "strong_sell"
        
        # Sort by composite score descending
        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Update ranks
        for i, stock in enumerate(candidates):
            stock.rank = i + 1
        
        return candidates
    
    def rank(
        self,
        universe: str | list[str] = "sp100",
        ml_top_n: int = 25,
        final_top_n: int = 10,
    ) -> RankingResult:
        """
        Run the full ranking pipeline.
        
        Args:
            universe: Universe name or list of tickers.
            ml_top_n: Number of stocks to pass from ML screening.
            final_top_n: Number of stocks in final ranking.
            
        Returns:
            RankingResult with complete rankings.
        """
        start_time = datetime.now()
        
        # Get universe tickers
        if isinstance(universe, list):
            tickers = universe
            universe_name = f"custom ({len(tickers)} stocks)"
        else:
            tickers = self._get_universe_tickers(universe)
            universe_name = universe
        
        logger.info(f"Ranking {len(tickers)} stocks from {universe_name}")
        
        # Stage 1: ML Screening
        ml_candidates = self._run_ml_screening(tickers, ml_top_n)
        
        # Stage 2: Agent Analysis
        analyzed = self._run_agent_analysis(ml_candidates)
        
        # Stage 3: Compute Composite Scores
        ranked = self._compute_composite_scores(analyzed)
        
        # Take final top N
        final_rankings = ranked[:final_top_n]
        
        return RankingResult(
            generated_at=start_time,
            universe=universe_name,
            universe_size=len(tickers),
            ml_candidates=ml_candidates,
            ml_top_n=ml_top_n,
            final_rankings=final_rankings,
            final_top_n=final_top_n,
            ml_weight=self.ml_weight,
            agent_weights=self.agent_weights,
        )


# ==============================================================================
# CLI
# ==============================================================================

def main():
    """Command-line interface for ranking system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-stage stock ranking system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m auto_researcher.ranking_system --universe sp100 --ml-top 25 --final-top 10
  python -m auto_researcher.ranking_system --universe sp500 --ml-top 50 --final-top 20
  python -m auto_researcher.ranking_system --tickers AAPL,MSFT,GOOGL,NVDA --ml-top 4 --final-top 3
        """,
    )
    
    parser.add_argument(
        "--universe",
        choices=["sp500", "sp100", "large_cap", "core_tech"],
        default="sp100",
        help="Stock universe to rank (default: sp100)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers (overrides --universe)",
    )
    parser.add_argument(
        "--ml-top",
        type=int,
        default=25,
        help="Number of stocks to pass from ML screening (default: 25)",
    )
    parser.add_argument(
        "--final-top",
        type=int,
        default=10,
        help="Number of stocks in final ranking (default: 10)",
    )
    parser.add_argument(
        "--ml-weight",
        type=float,
        default=0.35,
        help="Weight for ML score in composite (default: 0.35)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model for agents (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for results (txt or json)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Determine universe
    if args.tickers:
        universe = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        universe = args.universe
    
    # Create ranker
    ranker = RankingSystem(
        model=args.model,
        ml_weight=args.ml_weight,
        use_cache=not args.no_cache,
    )
    
    # Run ranking
    print(f"\n{'='*70}")
    print("  MULTI-STAGE STOCK RANKING SYSTEM")
    print(f"{'='*70}")
    print(f"  Universe: {args.universe if not args.tickers else 'custom'}")
    print(f"  ML Screening: Top {args.ml_top}")
    print(f"  Final Rankings: Top {args.final_top}")
    print(f"  ML Weight: {args.ml_weight:.0%}")
    print(f"{'='*70}\n")
    
    results = ranker.rank(
        universe=universe,
        ml_top_n=args.ml_top,
        final_top_n=args.final_top,
    )
    
    # Print summary
    print(results.to_summary())
    
    # Save output if requested
    if args.output:
        if args.output.endswith(".json"):
            import json
            # Convert to JSON-serializable format
            data = {
                "generated_at": results.generated_at.isoformat(),
                "universe": results.universe,
                "universe_size": results.universe_size,
                "ml_top_n": results.ml_top_n,
                "final_top_n": results.final_top_n,
                "ml_weight": results.ml_weight,
                "agent_weights": results.agent_weights,
                "final_rankings": [
                    {
                        "rank": s.rank,
                        "ticker": s.ticker,
                        "company_name": s.company_name,
                        "composite_score": s.composite_score,
                        "composite_signal": s.composite_signal,
                        "ml_percentile": s.ml_percentile,
                        "ml_weighted": s.ml_weighted,
                        "agent_composite": s.agent_composite,
                        "agent_scores": {
                            k: {"score": v.score, "signal": v.signal, "confidence": v.confidence}
                            for k, v in s.agent_scores.items()
                        },
                        "sector": s.sector,
                        "key_factors": s.key_factors,
                    }
                    for s in results.final_rankings
                ],
            }
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(args.output, "w") as f:
                f.write(results.to_summary())
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
