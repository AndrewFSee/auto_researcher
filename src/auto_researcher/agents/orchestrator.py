"""
Research Orchestrator.

Combines ML ranking, sentiment analysis, and fundamental analysis
into unified research reports with investment recommendations.

Usage:
    from auto_researcher.agents.orchestrator import ResearchOrchestrator
    
    orchestrator = ResearchOrchestrator()
    report = orchestrator.research(tickers=["GOOGL", "NVDA", "MRK"])
    print(report.to_markdown())
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Literal

import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

class SignalStrength(Enum):
    """Signal strength categories."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class StockResearch:
    """Complete research report for a single stock."""
    ticker: str
    
    # ML Ranking
    ml_score: float
    ml_rank: int
    ml_percentile: float
    ml_drivers: list[str]
    
    # Sentiment (optional)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_summary: Optional[str] = None
    sentiment_themes: Optional[list[str]] = None
    
    # Fundamentals (optional)
    fundamental_quality: Optional[float] = None
    fundamental_label: Optional[str] = None
    value_assessment: Optional[str] = None
    growth_outlook: Optional[str] = None
    risk_level: Optional[str] = None
    fundamental_summary: Optional[str] = None
    strengths: Optional[list[str]] = None
    concerns: Optional[list[str]] = None
    
    # Key metrics
    pe_ratio: Optional[float] = None
    roe: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    sector: Optional[str] = None
    
    # Combined signal
    combined_signal: SignalStrength = SignalStrength.HOLD
    confidence: float = 0.5
    
    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    
    def signal_alignment(self) -> str:
        """Check if all signals are aligned."""
        signals = []
        
        # ML signal
        if self.ml_percentile >= 80:
            signals.append("bullish")
        elif self.ml_percentile <= 20:
            signals.append("bearish")
        else:
            signals.append("neutral")
        
        # Sentiment signal
        if self.sentiment_score is not None:
            if self.sentiment_score > 0.2:
                signals.append("bullish")
            elif self.sentiment_score < -0.2:
                signals.append("bearish")
            else:
                signals.append("neutral")
        
        # Fundamental signal
        if self.fundamental_quality is not None:
            if self.fundamental_quality >= 0.6 and self.value_assessment != "overvalued":
                signals.append("bullish")
            elif self.fundamental_quality < 0.4 or self.value_assessment == "overvalued":
                signals.append("bearish")
            else:
                signals.append("neutral")
        
        # Check alignment
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")
        
        if bullish_count == len(signals):
            return "FULLY_ALIGNED_BULLISH"
        elif bearish_count == len(signals):
            return "FULLY_ALIGNED_BEARISH"
        elif bullish_count > bearish_count:
            return "MOSTLY_BULLISH"
        elif bearish_count > bullish_count:
            return "MOSTLY_BEARISH"
        else:
            return "MIXED"


@dataclass
class ResearchReport:
    """Complete research report for multiple stocks."""
    stocks: list[StockResearch]
    universe_size: int
    analysis_date: datetime
    data_date: str
    model_used: str
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        data = []
        for s in self.stocks:
            data.append({
                "ticker": s.ticker,
                "ml_score": s.ml_score,
                "ml_rank": s.ml_rank,
                "ml_percentile": s.ml_percentile,
                "sentiment_score": s.sentiment_score,
                "sentiment_label": s.sentiment_label,
                "fundamental_quality": s.fundamental_quality,
                "value_assessment": s.value_assessment,
                "growth_outlook": s.growth_outlook,
                "risk_level": s.risk_level,
                "combined_signal": s.combined_signal.value,
                "alignment": s.signal_alignment(),
                "confidence": s.confidence,
                "sector": s.sector,
            })
        return pd.DataFrame(data)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []
        
        lines.append(f"# Stock Research Report")
        lines.append(f"**Generated:** {self.analysis_date.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Data through:** {self.data_date}")
        lines.append(f"**Universe:** {self.universe_size} stocks analyzed")
        lines.append(f"**Model:** {self.model_used}")
        lines.append("")
        
        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Rank | Ticker | ML Score | Sentiment | Quality | Signal | Alignment |")
        lines.append("|------|--------|----------|-----------|---------|--------|-----------|")
        
        for s in self.stocks:
            sent = f"{s.sentiment_score:+.2f}" if s.sentiment_score is not None else "N/A"
            qual = f"{s.fundamental_quality:.2f}" if s.fundamental_quality is not None else "N/A"
            signal = s.combined_signal.value.replace("_", " ").title()
            alignment = s.signal_alignment().replace("_", " ").title()
            
            lines.append(f"| {s.ml_rank} | **{s.ticker}** | {s.ml_score:.3f} | {sent} | {qual} | {signal} | {alignment} |")
        
        lines.append("")
        
        # Detailed analysis
        lines.append("## Detailed Analysis")
        lines.append("")
        
        for s in self.stocks:
            lines.append(f"### {s.ticker}")
            lines.append("")
            
            # Key stats
            if s.sector:
                lines.append(f"**Sector:** {s.sector}")
            lines.append(f"**ML Rank:** #{s.ml_rank} (top {100-s.ml_percentile:.0f}%)")
            lines.append("")
            
            # ML Analysis
            lines.append("#### Technical/Quantitative Signals")
            lines.append(f"- **ML Score:** {s.ml_score:.3f}")
            if s.ml_drivers:
                lines.append(f"- **Key Drivers:** {', '.join(s.ml_drivers)}")
            lines.append("")
            
            # Sentiment
            if s.sentiment_score is not None:
                lines.append("#### Sentiment Analysis")
                lines.append(f"- **Score:** {s.sentiment_score:+.2f} ({s.sentiment_label})")
                if s.sentiment_summary:
                    lines.append(f"- **Summary:** {s.sentiment_summary}")
                if s.sentiment_themes:
                    lines.append(f"- **Themes:** {', '.join(s.sentiment_themes)}")
                lines.append("")
            
            # Fundamentals
            if s.fundamental_quality is not None:
                lines.append("#### Fundamental Analysis")
                lines.append(f"- **Quality Score:** {s.fundamental_quality:.2f} ({s.fundamental_label})")
                lines.append(f"- **Valuation:** {s.value_assessment}")
                lines.append(f"- **Growth:** {s.growth_outlook}")
                lines.append(f"- **Risk:** {s.risk_level}")
                if s.fundamental_summary:
                    lines.append(f"- **Analysis:** {s.fundamental_summary}")
                if s.strengths:
                    lines.append(f"- ✅ **Strengths:** {', '.join(s.strengths)}")
                if s.concerns:
                    lines.append(f"- ⚠️ **Concerns:** {', '.join(s.concerns)}")
                lines.append("")
            
            # Key metrics
            metrics = []
            if s.pe_ratio:
                metrics.append(f"P/E: {s.pe_ratio:.1f}")
            if s.roe:
                metrics.append(f"ROE: {s.roe*100:.1f}%")
            if s.profit_margin:
                metrics.append(f"Margin: {s.profit_margin*100:.1f}%")
            if s.revenue_growth:
                metrics.append(f"Rev Growth: {s.revenue_growth*100:.1f}%")
            
            if metrics:
                lines.append("#### Key Metrics")
                lines.append(f"{' | '.join(metrics)}")
                lines.append("")
            
            # Combined view
            lines.append("#### Investment Thesis")
            alignment = s.signal_alignment()
            signal = s.combined_signal.value.replace("_", " ").upper()
            lines.append(f"**Signal:** {signal} | **Alignment:** {alignment.replace('_', ' ')} | **Confidence:** {s.confidence:.0%}")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Disclaimer
        lines.append("## Disclaimer")
        lines.append("")
        lines.append("*This report is for research and educational purposes only. It does not constitute financial advice. Past performance is not indicative of future results. Always do your own due diligence before making investment decisions.*")
        
        return "\n".join(lines)


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for the research orchestrator."""
    llm_model: str = "gpt-4o-mini"
    include_sentiment: bool = True
    include_fundamentals: bool = True
    top_k: int = 5


class ResearchOrchestrator:
    """
    Orchestrates multiple research agents to produce comprehensive stock analysis.
    
    Combines:
    - ML Ranking Model (quantitative)
    - Sentiment Agent (news analysis)
    - Fundamental Agent (financial analysis)
    
    Into unified research reports with combined signals.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the orchestrator."""
        self.config = config or OrchestratorConfig()
        
        # Lazy load agents
        self._sentiment_agent = None
        self._fundamental_agent = None
        
        logger.info(f"Initialized ResearchOrchestrator with model: {self.config.llm_model}")
    
    @property
    def sentiment_agent(self):
        """Lazy load sentiment agent."""
        if self._sentiment_agent is None and self.config.include_sentiment:
            try:
                from auto_researcher.agents.sentiment_agent import SentimentAgent
                self._sentiment_agent = SentimentAgent(model=self.config.llm_model)
            except ImportError as e:
                logger.warning(f"Could not load SentimentAgent: {e}")
        return self._sentiment_agent
    
    @property
    def fundamental_agent(self):
        """Lazy load fundamental agent."""
        if self._fundamental_agent is None and self.config.include_fundamentals:
            try:
                from auto_researcher.agents.fundamental_agent import FundamentalAgent
                self._fundamental_agent = FundamentalAgent(model=self.config.llm_model)
            except ImportError as e:
                logger.warning(f"Could not load FundamentalAgent: {e}")
        return self._fundamental_agent
    
    def _compute_combined_signal(self, research: StockResearch) -> tuple[SignalStrength, float]:
        """
        Compute combined signal from all sources.
        
        Returns (signal, confidence).
        """
        scores = []
        weights = []
        
        # ML Score (highest weight)
        if research.ml_percentile is not None:
            ml_normalized = (research.ml_percentile - 50) / 50  # -1 to +1
            scores.append(ml_normalized)
            weights.append(0.5)
        
        # Sentiment
        if research.sentiment_score is not None:
            scores.append(research.sentiment_score)  # Already -1 to +1
            weights.append(0.25)
        
        # Fundamental Quality
        if research.fundamental_quality is not None:
            fund_normalized = (research.fundamental_quality - 0.5) * 2  # -1 to +1
            
            # Adjust for valuation
            if research.value_assessment == "overvalued":
                fund_normalized -= 0.3
            elif research.value_assessment == "undervalued":
                fund_normalized += 0.2
            
            # Adjust for risk
            if research.risk_level == "high":
                fund_normalized -= 0.2
            
            scores.append(fund_normalized)
            weights.append(0.25)
        
        if not scores:
            return SignalStrength.HOLD, 0.5
        
        # Weighted average
        total_weight = sum(weights)
        combined_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        # Convert to signal
        if combined_score > 0.5:
            signal = SignalStrength.STRONG_BUY
        elif combined_score > 0.2:
            signal = SignalStrength.BUY
        elif combined_score > -0.2:
            signal = SignalStrength.HOLD
        elif combined_score > -0.5:
            signal = SignalStrength.SELL
        else:
            signal = SignalStrength.STRONG_SELL
        
        # Confidence based on agreement
        alignment = research.signal_alignment()
        if "FULLY_ALIGNED" in alignment:
            confidence = 0.9
        elif "MOSTLY" in alignment:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return signal, confidence
    
    def research(
        self,
        tickers: list[str] = None,
        universe: str = "large_cap",
        top_k: int = None,
    ) -> ResearchReport:
        """
        Run complete research pipeline.
        
        Args:
            tickers: Specific tickers to research (overrides universe).
            universe: Stock universe if tickers not specified.
            top_k: Number of top stocks to include.
            
        Returns:
            ResearchReport with comprehensive analysis.
        """
        import sys
        sys.path.insert(0, str(__file__).rsplit("src", 1)[0])
        
        top_k = top_k or self.config.top_k
        
        # Get ML recommendations
        print("\n" + "=" * 70)
        print("  RESEARCH ORCHESTRATOR")
        print("=" * 70)
        print(f"  Model: {self.config.llm_model}")
        print(f"  Sentiment: {'✓' if self.config.include_sentiment else '✗'}")
        print(f"  Fundamentals: {'✓' if self.config.include_fundamentals else '✗'}")
        print("")
        
        # Import here to avoid circular imports
        try:
            from recommend import generate_recommendations, UNIVERSES
        except ImportError:
            # Try alternate import path
            import importlib.util
            import os
            spec = importlib.util.spec_from_file_location(
                "recommend",
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "recommend.py")
            )
            recommend_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(recommend_module)
            generate_recommendations = recommend_module.generate_recommendations
            UNIVERSES = recommend_module.UNIVERSES
        
        # Get ticker list
        if tickers is None:
            tickers = UNIVERSES.get(universe, UNIVERSES["large_cap"])
        
        # Run ML model
        print("[1/3] Running ML ranking model...")
        recommendations, all_scores, prices = generate_recommendations(
            tickers=tickers,
            top_k=top_k,
            explain=True,
        )
        
        top_tickers = [r.ticker for r in recommendations]
        data_date = prices.index[-1].strftime("%Y-%m-%d")
        
        # Initialize research objects
        stock_research = {}
        for r in recommendations:
            stock_research[r.ticker] = StockResearch(
                ticker=r.ticker,
                ml_score=r.score,
                ml_rank=r.rank,
                ml_percentile=r.percentile,
                ml_drivers=r.drivers,
            )
        
        # Run sentiment analysis
        if self.config.include_sentiment and self.sentiment_agent:
            print("[2/3] Running sentiment analysis...")
            try:
                sentiment_results = self.sentiment_agent.analyze_tickers(top_tickers)
                for ticker, result in sentiment_results.items():
                    if ticker in stock_research:
                        sr = stock_research[ticker]
                        sr.sentiment_score = result.sentiment_score
                        sr.sentiment_label = result.sentiment_label
                        sr.sentiment_summary = result.summary
                        sr.sentiment_themes = getattr(result, 'themes', getattr(result, 'key_themes', []))
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        else:
            print("[2/3] Skipping sentiment analysis")
        
        # Run fundamental analysis
        if self.config.include_fundamentals and self.fundamental_agent:
            print("[3/3] Running fundamental analysis...")
            try:
                fundamental_results = self.fundamental_agent.analyze_tickers(top_tickers)
                for ticker, result in fundamental_results.items():
                    if ticker in stock_research:
                        sr = stock_research[ticker]
                        sr.fundamental_quality = result.quality_score
                        sr.fundamental_label = result.quality_label
                        sr.value_assessment = result.value_assessment
                        sr.growth_outlook = result.growth_outlook
                        sr.risk_level = result.risk_level
                        sr.fundamental_summary = result.summary
                        sr.strengths = result.strengths
                        sr.concerns = result.concerns
                        
                        # Key metrics
                        m = result.metrics
                        sr.pe_ratio = m.pe_ratio
                        sr.roe = m.roe
                        sr.profit_margin = m.profit_margin
                        sr.revenue_growth = m.revenue_growth
                        sr.sector = m.sector
            except Exception as e:
                logger.warning(f"Fundamental analysis failed: {e}")
        else:
            print("[3/3] Skipping fundamental analysis")
        
        # Compute combined signals
        for sr in stock_research.values():
            sr.combined_signal, sr.confidence = self._compute_combined_signal(sr)
        
        # Build report
        stocks = sorted(stock_research.values(), key=lambda x: x.ml_rank)
        
        report = ResearchReport(
            stocks=stocks,
            universe_size=len(all_scores),
            analysis_date=datetime.now(),
            data_date=data_date,
            model_used=self.config.llm_model,
        )
        
        print("\n✓ Research complete!")
        return report
    
    def print_summary(self, report: ResearchReport) -> None:
        """Print a summary of the research report."""
        print("\n" + "=" * 70)
        print("  RESEARCH SUMMARY")
        print("=" * 70)
        print(f"  Generated: {report.analysis_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Data through: {report.data_date}")
        print("")
        
        print(f"{'Rank':<6} {'Ticker':<8} {'ML':>6} {'Sent':>6} {'Qual':>6} {'Signal':<12} {'Align'}")
        print("-" * 70)
        
        for s in report.stocks:
            sent = f"{s.sentiment_score:+.2f}" if s.sentiment_score is not None else "N/A"
            qual = f"{s.fundamental_quality:.2f}" if s.fundamental_quality is not None else "N/A"
            signal = s.combined_signal.value.replace("_", " ").title()
            alignment = s.signal_alignment().replace("_", " ").title()
            
            # Indicator
            if s.combined_signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
                indicator = "🟢"
            elif s.combined_signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
                indicator = "🔴"
            else:
                indicator = "🟡"
            
            print(f"{indicator} {s.ml_rank:<4} {s.ticker:<8} {s.ml_score:>5.2f} {sent:>6} {qual:>6} {signal:<12} {alignment}")
        
        print("-" * 70)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive stock research")
    parser.add_argument(
        "--universe",
        default="large_cap",
        help="Stock universe to analyze",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top stocks",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use",
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Skip sentiment analysis",
    )
    parser.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="Skip fundamental analysis",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save markdown report to file",
    )
    
    args = parser.parse_args()
    
    config = OrchestratorConfig(
        llm_model=args.model,
        include_sentiment=not args.no_sentiment,
        include_fundamentals=not args.no_fundamentals,
        top_k=args.top_k,
    )
    
    orchestrator = ResearchOrchestrator(config=config)
    report = orchestrator.research(universe=args.universe)
    
    # Print summary
    orchestrator.print_summary(report)
    
    # Save if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        print(f"\nSaved report to: {args.output}")
