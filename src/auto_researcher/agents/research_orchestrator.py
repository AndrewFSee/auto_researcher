"""
Research Orchestrator.

Coordinates all analysis agents and produces a unified research report
with aggregated signals and confidence scores.

Features:
- Parallel agent execution for speed
- Caching to avoid redundant API calls
- Signal aggregation across all sources
- Unified confidence scoring
- PDF/Markdown report generation

Usage:
    from auto_researcher.agents.research_orchestrator import ResearchOrchestrator
    
    orchestrator = ResearchOrchestrator(model="gpt-4o-mini")
    report = orchestrator.analyze("NVDA")
    print(report.to_markdown())
"""

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Literal

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "auto_researcher"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SignalStrength:
    """Signal strength from an analysis source."""
    source: str  # sentiment, fundamental, sec, earnings
    signal: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float  # 0-1
    score: float  # -1 to +1 normalized
    rationale: str = ""


@dataclass  
class AggregatedSignal:
    """Combined signal from all sources."""
    overall_signal: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    overall_confidence: float
    overall_score: float  # -1 to +1
    ml_signal: Optional[SignalStrength] = None
    sentiment_signal: Optional[SignalStrength] = None
    fundamental_signal: Optional[SignalStrength] = None
    sec_signal: Optional[SignalStrength] = None
    earnings_signal: Optional[SignalStrength] = None
    insider_signal: Optional[SignalStrength] = None
    thematic_signal: Optional[SignalStrength] = None
    agreement_ratio: float = 0.0  # How many signals agree


@dataclass
class UnifiedReport:
    """Complete research report for a ticker."""
    ticker: str
    company_name: str
    generated_at: datetime
    
    # Aggregated signal
    signal: AggregatedSignal
    
    # Individual analyses
    ml_analysis: Optional[dict] = None
    sentiment_analysis: Optional[Any] = None
    fundamental_analysis: Optional[Any] = None
    sec_analysis: Optional[Any] = None
    earnings_analysis: Optional[Any] = None
    insider_analysis: Optional[Any] = None
    thematic_analysis: Optional[Any] = None
    
    # Summary
    executive_summary: str = ""
    key_catalysts: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        md = []
        md.append(f"# {self.ticker} - {self.company_name}")
        md.append(f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}*\n")
        
        # Signal summary
        sig = self.signal
        signal_emoji = {
            "strong_buy": "🟢🟢",
            "buy": "🟢",
            "hold": "🟡",
            "sell": "🔴",
            "strong_sell": "🔴🔴",
        }
        md.append(f"## Overall Signal: {signal_emoji.get(sig.overall_signal, '')} {sig.overall_signal.upper()}")
        md.append(f"**Confidence:** {sig.overall_confidence:.0%} | **Score:** {sig.overall_score:+.2f} | **Agreement:** {sig.agreement_ratio:.0%}\n")
        
        # Executive summary
        if self.executive_summary:
            md.append("## Executive Summary")
            md.append(self.executive_summary + "\n")
        
        # Individual signals
        md.append("## Signal Breakdown\n")
        md.append("| Source | Signal | Confidence | Score |")
        md.append("|--------|--------|------------|-------|")
        for name, signal in [
            ("ML Model", sig.ml_signal),
            ("Sentiment", sig.sentiment_signal),
            ("Fundamentals", sig.fundamental_signal),
            ("SEC Filings", sig.sec_signal),
            ("Earnings", sig.earnings_signal),
            ("Insider Trading", sig.insider_signal),
            ("Thematic", sig.thematic_signal),
        ]:
            if signal:
                md.append(f"| {name} | {signal.signal} | {signal.confidence:.0%} | {signal.score:+.2f} |")
        md.append("")
        
        # Catalysts and risks
        if self.key_catalysts:
            md.append("## Key Catalysts 🚀")
            for c in self.key_catalysts:
                md.append(f"- {c}")
            md.append("")
        
        if self.key_risks:
            md.append("## Key Risks ⚠️")
            for r in self.key_risks:
                md.append(f"- {r}")
            md.append("")
        
        # Detailed analyses
        if self.sentiment_analysis:
            md.append("## Sentiment Analysis")
            s = self.sentiment_analysis
            md.append(f"**Score:** {s.sentiment_score:+.2f} ({s.sentiment_label})")
            md.append(f"**Summary:** {s.summary}")
            if hasattr(s, 'key_themes') and s.key_themes:
                md.append(f"**Themes:** {', '.join(s.key_themes)}")
            md.append("")
        
        if self.fundamental_analysis:
            md.append("## Fundamental Analysis")
            f = self.fundamental_analysis
            md.append(f"**Quality:** {f.quality_score:.2f} ({f.quality_label})")
            md.append(f"**Value:** {f.value_assessment} | **Growth:** {f.growth_outlook} | **Risk:** {f.risk_level}")
            md.append(f"**Summary:** {f.summary}")
            md.append("")
        
        if self.sec_analysis:
            md.append("## SEC Filing Analysis")
            sec = self.sec_analysis
            md.append(f"**Risk Score:** {sec.risk_score:.2f} | **Sentiment:** {sec.sentiment_score:+.2f}")
            md.append(f"**Summary:** {sec.summary}")
            if sec.key_risks:
                md.append(f"**Risks:** {', '.join(sec.key_risks[:3])}")
            md.append("")
        
        if self.earnings_analysis:
            md.append("## Earnings Analysis")
            e = self.earnings_analysis
            md.append(f"**Tone Score:** {e.tone_score:+.2f} | **Guidance:** {e.guidance_direction}")
            md.append(f"**Summary:** {e.summary}")
            md.append("")
        
        if self.insider_analysis:
            md.append("## Insider Trading Analysis")
            ins = self.insider_analysis
            md.append(f"**Signal:** {ins.signal} | **Score:** {ins.score:+.2f} | **Confidence:** {ins.confidence:.0%}")
            md.append(f"**Transactions (90d):** {ins.total_transactions} | Buys: {ins.buy_count} | Sells: {ins.sell_count}")
            if ins.has_cluster_buying:
                md.append("🟢 **Cluster buying detected** - Multiple insiders buying together")
            if ins.has_cluster_selling:
                md.append("🔴 **Cluster selling detected** - Multiple insiders selling together")
            md.append(f"**Summary:** {ins.summary}")
            if ins.key_insights:
                for insight in ins.key_insights:
                    md.append(f"- {insight}")
            md.append("")
        
        if self.thematic_analysis:
            md.append("## Thematic / Forward-Looking Analysis")
            th = self.thematic_analysis
            md.append(f"**Sector:** {th.sector} | **Outlook:** {th.sector_outlook}")
            if th.moat:
                md.append(f"**Moat:** {th.moat.moat_rating.upper()} ({th.moat.moat_score:.2f})")
                if th.moat.moat_sources:
                    md.append(f"**Moat Sources:** {', '.join(th.moat.moat_sources[:3])}")
            if th.theme_exposures:
                themes = [f"{e.theme_name} ({e.exposure_score:.0%})" for e in th.theme_exposures[:3]]
                md.append(f"**Theme Exposure:** {', '.join(themes)}")
            md.append(f"**Summary:** {th.summary}")
            md.append("")
        
        return "\n".join(md)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "generated_at": self.generated_at.isoformat(),
            "signal": {
                "overall": self.signal.overall_signal,
                "confidence": self.signal.overall_confidence,
                "score": self.signal.overall_score,
                "agreement_ratio": self.signal.agreement_ratio,
            },
            "executive_summary": self.executive_summary,
            "key_catalysts": self.key_catalysts,
            "key_risks": self.key_risks,
        }


# ==============================================================================
# CACHE MANAGER
# ==============================================================================

class CacheManager:
    """Simple file-based cache for API results."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_key(self, namespace: str, identifier: str) -> str:
        """Generate cache key."""
        raw = f"{namespace}:{identifier}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def _get_path(self, key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{key}.json"
    
    def _serialize(self, obj: Any) -> Any:
        """Recursively serialize object for JSON."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if hasattr(obj, '__dict__'):
            return {"__class__": type(obj).__name__, **{k: self._serialize(v) for k, v in obj.__dict__.items()}}
        # Fallback - try to convert to string
        try:
            return str(obj)
        except:
            return None
    
    def get(self, namespace: str, identifier: str) -> Optional[dict]:
        """Get cached value if valid."""
        key = self._get_key(namespace, identifier)
        path = self._get_path(key)
        
        if not path.exists():
            return None
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Check TTL
            cached_at = datetime.fromisoformat(data["_cached_at"])
            if datetime.now() - cached_at > self.ttl:
                path.unlink()
                return None
            
            return data.get("value")
        except Exception:
            return None
    
    def set(self, namespace: str, identifier: str, value: dict) -> None:
        """Cache a value."""
        key = self._get_key(namespace, identifier)
        path = self._get_path(key)
        
        try:
            # Serialize value to handle datetime and nested objects
            serialized_value = self._serialize(value)
            data = {
                "_cached_at": datetime.now().isoformat(),
                "_namespace": namespace,
                "_identifier": identifier,
                "value": serialized_value,
            }
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache. Returns number of files deleted."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                if namespace:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("_namespace") != namespace:
                        continue
                path.unlink()
                count += 1
            except Exception:
                pass
        return count


# ==============================================================================
# RESEARCH ORCHESTRATOR
# ==============================================================================

class ResearchOrchestrator:
    """
    Orchestrates multiple analysis agents to produce unified research reports.
    
    Features:
    - Parallel execution of agents
    - Result caching to avoid redundant API calls
    - Signal aggregation with confidence weighting
    - Markdown/JSON report generation
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_cache: bool = True,
        cache_ttl_hours: int = 24,
        parallel: bool = True,
        max_workers: int = 4,
        sentiment_mode: str = "hybrid",
    ):
        """
        Initialize the orchestrator.
        
        Args:
            model: LLM model for agents.
            use_cache: Whether to cache results.
            cache_ttl_hours: Cache time-to-live.
            parallel: Run agents in parallel.
            max_workers: Max parallel workers.
            sentiment_mode: "llm", "finbert", or "hybrid".
        """
        self.model = model
        self.use_cache = use_cache
        self.parallel = parallel
        self.max_workers = max_workers
        self.sentiment_mode = sentiment_mode
        
        self.cache = CacheManager(ttl_hours=cache_ttl_hours) if use_cache else None
        
        # Lazy-load agents
        self._sentiment_agent = None
        self._fundamental_agent = None
        self._sec_agent = None
        self._earnings_agent = None
        self._insider_agent = None
        self._thematic_agent = None
    
    @property
    def sentiment_agent(self):
        if self._sentiment_agent is None:
            try:
                from .sentiment_agent import SentimentAgent, SentimentAgentConfig
                config = SentimentAgentConfig(
                    model=self.model,
                    hybrid_mode=(self.sentiment_mode == "hybrid"),
                    finbert_only=(self.sentiment_mode == "finbert"),
                )
                self._sentiment_agent = SentimentAgent(config=config)
            except ImportError as e:
                logger.warning(f"SentimentAgent not available: {e}")
        return self._sentiment_agent
    
    @property
    def fundamental_agent(self):
        if self._fundamental_agent is None:
            try:
                from .fundamental_agent import FundamentalAgent
                self._fundamental_agent = FundamentalAgent(model=self.model)
            except ImportError as e:
                logger.warning(f"FundamentalAgent not available: {e}")
        return self._fundamental_agent
    
    @property
    def sec_agent(self):
        if self._sec_agent is None:
            try:
                from .sec_filing_agent import SECFilingAgent
                self._sec_agent = SECFilingAgent(model=self.model)
            except ImportError as e:
                logger.warning(f"SECFilingAgent not available: {e}")
        return self._sec_agent
    
    @property
    def earnings_agent(self):
        if self._earnings_agent is None:
            try:
                from .earnings_agent import EarningsCallAgent
                self._earnings_agent = EarningsCallAgent(model=self.model)
            except ImportError as e:
                logger.warning(f"EarningsCallAgent not available: {e}")
        return self._earnings_agent
    
    @property
    def insider_agent(self):
        if self._insider_agent is None:
            try:
                from .insider_trading_agent import InsiderTradingAgent
                self._insider_agent = InsiderTradingAgent()
            except ImportError as e:
                logger.warning(f"InsiderTradingAgent not available: {e}")
        return self._insider_agent
    
    @property
    def thematic_agent(self):
        if self._thematic_agent is None:
            try:
                from .thematic_agent import ThematicAnalysisAgent
                self._thematic_agent = ThematicAnalysisAgent(model=self.model)
            except ImportError as e:
                logger.warning(f"ThematicAnalysisAgent not available: {e}")
        return self._thematic_agent
    
    def _score_to_signal(self, score: float, thresholds: tuple = (0.3, 0.1)) -> str:
        """Convert numeric score to signal label."""
        high, low = thresholds
        if score >= high:
            return "strong_buy"
        elif score >= low:
            return "buy"
        elif score > -low:
            return "hold"
        elif score > -high:
            return "sell"
        else:
            return "strong_sell"
    
    def _run_sentiment(self, ticker: str) -> Optional[Any]:
        """Run sentiment analysis with caching."""
        cache_key = f"sentiment:{self.sentiment_mode}:{ticker}"
        
        if self.cache:
            cached = self.cache.get("sentiment", cache_key)
            if cached:
                logger.info(f"Cache hit for sentiment:{ticker}")
                return cached
        
        if not self.sentiment_agent:
            return None
        
        try:
            result = self.sentiment_agent.analyze_ticker(ticker)
            if self.cache and result:
                # Convert to dict for caching
                self.cache.set("sentiment", cache_key, result.__dict__)
            return result
        except Exception as e:
            import traceback
            logger.error(f"Sentiment analysis failed for {ticker}: {e}\n{traceback.format_exc()}")
            return None
    
    def _run_fundamental(self, ticker: str) -> Optional[Any]:
        """Run fundamental analysis with caching."""
        if self.cache:
            cached = self.cache.get("fundamental", ticker)
            if cached:
                logger.info(f"Cache hit for fundamental:{ticker}")
                return cached
        
        if not self.fundamental_agent:
            return None
        
        try:
            result = self.fundamental_agent.analyze_ticker(ticker)
            if self.cache and result:
                self.cache.set("fundamental", ticker, result.__dict__)
            return result
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {ticker}: {e}")
            return None
    
    def _run_sec(self, ticker: str) -> Optional[Any]:
        """Run SEC filing analysis with caching."""
        if self.cache:
            cached = self.cache.get("sec", ticker)
            if cached:
                logger.info(f"Cache hit for sec:{ticker}")
                return cached
        
        if not self.sec_agent:
            return None
        
        try:
            result = self.sec_agent.analyze_ticker(ticker)
            if self.cache and result:
                self.cache.set("sec", ticker, result.__dict__)
            return result
        except Exception as e:
            logger.error(f"SEC analysis failed for {ticker}: {e}")
            return None
    
    def _run_earnings(self, ticker: str) -> Optional[Any]:
        """Run earnings analysis with caching."""
        if self.cache:
            cached = self.cache.get("earnings", ticker)
            if cached:
                logger.info(f"Cache hit for earnings:{ticker}")
                return cached
        
        if not self.earnings_agent:
            return None
        
        try:
            result = self.earnings_agent.analyze_ticker(ticker)
            if self.cache and result:
                self.cache.set("earnings", ticker, result.__dict__)
            return result
        except Exception as e:
            logger.error(f"Earnings analysis failed for {ticker}: {e}")
            return None
    
    def _run_insider(self, ticker: str) -> Optional[Any]:
        """Run insider trading analysis with caching."""
        if self.cache:
            cached = self.cache.get("insider", ticker)
            if cached:
                logger.info(f"Cache hit for insider:{ticker}")
                return cached
        
        if not self.insider_agent:
            return None
        
        try:
            result = self.insider_agent.analyze_ticker(ticker)
            if self.cache and result:
                self.cache.set("insider", ticker, result.__dict__)
            return result
        except Exception as e:
            logger.error(f"Insider analysis failed for {ticker}: {e}")
            return None
    
    def _run_thematic(self, ticker: str) -> Optional[Any]:
        """Run thematic/forward-looking analysis with caching."""
        if self.cache:
            cached = self.cache.get("thematic", ticker)
            if cached:
                logger.info(f"Cache hit for thematic:{ticker}")
                return cached
        
        if not self.thematic_agent:
            return None
        
        try:
            result = self.thematic_agent.analyze_ticker(ticker)
            if self.cache and result:
                self.cache.set("thematic", ticker, result.__dict__)
            return result
        except Exception as e:
            logger.error(f"Thematic analysis failed for {ticker}: {e}")
            return None
    
    def _aggregate_signals(
        self,
        ml_score: Optional[float],
        sentiment: Optional[Any],
        fundamental: Optional[Any],
        sec: Optional[Any],
        earnings: Optional[Any],
        insider: Optional[Any] = None,
        thematic: Optional[Any] = None,
    ) -> AggregatedSignal:
        """Aggregate signals from all sources."""
        signals = []
        
        # ML Signal
        ml_signal = None
        if ml_score is not None:
            # Normalize percentile to -1 to +1
            normalized = (ml_score - 50) / 50
            ml_signal = SignalStrength(
                source="ml",
                signal=self._score_to_signal(normalized),
                confidence=min(abs(normalized) + 0.5, 1.0),
                score=normalized,
                rationale="Based on quantitative factor model",
            )
            signals.append(ml_signal)
        
        # Sentiment Signal
        sentiment_signal = None
        if sentiment:
            score = getattr(sentiment, 'sentiment_score', 0)
            sentiment_signal = SignalStrength(
                source="sentiment",
                signal=self._score_to_signal(score),
                confidence=getattr(sentiment, 'confidence', 0.5),
                score=score,
                rationale=getattr(sentiment, 'summary', ''),
            )
            signals.append(sentiment_signal)
        
        # Fundamental Signal
        fundamental_signal = None
        if fundamental:
            # Convert quality score (0-1) to -1 to +1
            quality = getattr(fundamental, 'quality_score', 0.5)
            score = (quality - 0.5) * 2
            fundamental_signal = SignalStrength(
                source="fundamental",
                signal=self._score_to_signal(score),
                confidence=min(quality + 0.3, 1.0),
                score=score,
                rationale=getattr(fundamental, 'summary', ''),
            )
            signals.append(fundamental_signal)
        
        # SEC Signal - derive score from risk level
        sec_signal = None
        if sec:
            risk = getattr(sec, 'risk_score', 0.5)
            # Convert risk to sentiment: low risk = positive, high risk = negative
            # risk 0 -> score +0.5, risk 0.5 -> score 0, risk 1 -> score -0.5
            score = (0.5 - risk)
            sec_signal = SignalStrength(
                source="sec",
                signal=self._score_to_signal(score),
                confidence=max(0.3, 1.0 - risk),
                score=score,
                rationale=getattr(sec, 'summary', ''),
            )
            signals.append(sec_signal)
        
        # Earnings Signal - convert tone and guidance to score
        earnings_signal = None
        if earnings:
            # Convert management_tone to score
            tone = getattr(earnings, 'management_tone', 'neutral')
            tone_map = {
                'very_positive': 0.5, 'positive': 0.3, 'neutral': 0.0,
                'cautious': -0.2, 'negative': -0.4, 'very_negative': -0.5
            }
            tone_score = tone_map.get(tone.lower(), 0.0)
            
            # Boost/reduce based on guidance
            guidance = getattr(earnings, 'guidance_direction', 'maintained')
            guidance_adj = {'raised': 0.2, 'above': 0.15, 'maintained': 0.0, 
                           'lowered': -0.2, 'below': -0.15}.get(guidance.lower(), 0.0)
            
            score = tone_score + guidance_adj
            confidence = getattr(earnings, 'confidence_level', 0.5)
            
            earnings_signal = SignalStrength(
                source="earnings",
                signal=self._score_to_signal(score),
                confidence=confidence,
                score=score,
                rationale=getattr(earnings, 'summary', ''),
            )
            signals.append(earnings_signal)
        
        # Insider Signal
        insider_signal = None
        if insider:
            score = getattr(insider, 'score', 0)
            insider_signal = SignalStrength(
                source="insider",
                signal=self._score_to_signal(score),
                confidence=getattr(insider, 'confidence', 0.3),
                score=score,
                rationale=getattr(insider, 'summary', ''),
            )
            signals.append(insider_signal)
        
        # Thematic/Forward-Looking Signal
        thematic_signal = None
        if thematic:
            score = getattr(thematic, 'forward_score', 0)
            thematic_signal = SignalStrength(
                source="thematic",
                signal=self._score_to_signal(score),
                confidence=getattr(thematic, 'confidence', 0.4),
                score=score,
                rationale=getattr(thematic, 'summary', ''),
            )
            signals.append(thematic_signal)
        
        # Aggregate
        if not signals:
            return AggregatedSignal(
                overall_signal="hold",
                overall_confidence=0.0,
                overall_score=0.0,
            )
        
        # Weighted average by confidence
        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            total_weight = len(signals)
        
        weighted_score = sum(s.score * s.confidence for s in signals) / total_weight
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # Check agreement
        signal_directions = [1 if s.score > 0.1 else -1 if s.score < -0.1 else 0 for s in signals]
        non_neutral = [d for d in signal_directions if d != 0]
        if non_neutral:
            agreement = sum(1 for d in non_neutral if d == max(set(non_neutral), key=non_neutral.count)) / len(non_neutral)
        else:
            agreement = 1.0
        
        return AggregatedSignal(
            overall_signal=self._score_to_signal(weighted_score),
            overall_confidence=avg_confidence,
            overall_score=weighted_score,
            ml_signal=ml_signal,
            sentiment_signal=sentiment_signal,
            fundamental_signal=fundamental_signal,
            sec_signal=sec_signal,
            earnings_signal=earnings_signal,
            insider_signal=insider_signal,
            thematic_signal=thematic_signal,
            agreement_ratio=agreement,
        )
    
    def _generate_summary(
        self,
        ticker: str,
        signal: AggregatedSignal,
        sentiment: Optional[Any],
        fundamental: Optional[Any],
        sec: Optional[Any],
        earnings: Optional[Any],
    ) -> tuple[str, list[str], list[str]]:
        """Generate executive summary and extract catalysts/risks."""
        
        # Collect catalysts
        catalysts = []
        if sentiment and hasattr(sentiment, 'key_themes'):
            for theme in sentiment.key_themes[:2]:
                if any(word in theme.lower() for word in ['growth', 'beat', 'strong', 'positive', 'launch', 'expand']):
                    catalysts.append(theme)
        if fundamental:
            for strength in getattr(fundamental, 'strengths', [])[:2]:
                catalysts.append(strength)
        if earnings:
            guidance = getattr(earnings, 'guidance_direction', '')
            if guidance.lower() in ['raised', 'above']:
                catalysts.append(f"Raised guidance: {getattr(earnings, 'summary', '')[:50]}")
        
        # Collect risks
        risks = []
        if sec:
            for risk in getattr(sec, 'key_risks', [])[:2]:
                risks.append(risk)
        if fundamental:
            for concern in getattr(fundamental, 'concerns', [])[:2]:
                risks.append(concern)
        if earnings:
            for risk in getattr(earnings, 'key_risks', [])[:2]:
                risks.append(risk)
        
        # Build summary
        parts = []
        parts.append(f"{ticker} receives an overall **{signal.overall_signal.upper()}** rating with {signal.overall_confidence:.0%} confidence.")
        
        if signal.agreement_ratio >= 0.8:
            parts.append("All analysis sources are in strong agreement.")
        elif signal.agreement_ratio >= 0.5:
            parts.append("Most analysis sources agree on direction.")
        else:
            parts.append("Analysis sources show mixed signals - exercise caution.")
        
        if sentiment:
            parts.append(f"Sentiment is {getattr(sentiment, 'sentiment_label', 'neutral')} based on recent news.")
        if fundamental:
            parts.append(f"Fundamentals rated {getattr(fundamental, 'quality_label', 'average')} with {getattr(fundamental, 'value_assessment', 'fair')} valuation.")
        
        summary = " ".join(parts)
        
        return summary, catalysts[:5], risks[:5]
    
    def analyze(
        self,
        ticker: str,
        ml_percentile: Optional[float] = None,
        include_sentiment: bool = True,
        include_fundamentals: bool = True,
        include_sec: bool = True,
        include_earnings: bool = True,
        include_insider: bool = True,
        include_thematic: bool = True,
    ) -> UnifiedReport:
        """
        Run all analyses and produce unified report.
        
        Args:
            ticker: Stock ticker symbol.
            ml_percentile: Optional ML model percentile (0-100).
            include_*: Which analyses to include.
            
        Returns:
            UnifiedReport with all analyses and aggregated signal.
        """
        logger.info(f"Starting unified analysis for {ticker}")
        start_time = time.time()
        
        # Pre-initialize agents that need heavy loading (FinBERT) before parallel execution
        # This avoids race conditions during model loading
        if include_sentiment and self.sentiment_agent is None:
            pass  # Property access triggers lazy loading
        _ = self.sentiment_agent  # Force initialization
        
        # Run analyses
        sentiment = None
        fundamental = None
        sec = None
        earnings = None
        insider = None
        thematic = None
        
        if self.parallel:
            # Parallel execution - higher timeout for FinBERT model loading
            tasks = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                if include_sentiment:
                    tasks["sentiment"] = executor.submit(self._run_sentiment, ticker)
                if include_fundamentals:
                    tasks["fundamental"] = executor.submit(self._run_fundamental, ticker)
                if include_sec:
                    tasks["sec"] = executor.submit(self._run_sec, ticker)
                if include_earnings:
                    tasks["earnings"] = executor.submit(self._run_earnings, ticker)
                if include_insider:
                    tasks["insider"] = executor.submit(self._run_insider, ticker)
                if include_thematic:
                    tasks["thematic"] = executor.submit(self._run_thematic, ticker)
                
                for name, future in tasks.items():
                    try:
                        # 120s timeout to allow for FinBERT model loading
                        result = future.result(timeout=120)
                        if name == "sentiment":
                            sentiment = result
                        elif name == "fundamental":
                            fundamental = result
                        elif name == "sec":
                            sec = result
                        elif name == "earnings":
                            earnings = result
                        elif name == "insider":
                            insider = result
                        elif name == "thematic":
                            thematic = result
                    except Exception as e:
                        logger.error(f"{name} analysis failed: {e}")
        else:
            # Sequential execution
            if include_sentiment:
                sentiment = self._run_sentiment(ticker)
            if include_fundamentals:
                fundamental = self._run_fundamental(ticker)
            if include_sec:
                sec = self._run_sec(ticker)
            if include_earnings:
                earnings = self._run_earnings(ticker)
            if include_insider:
                insider = self._run_insider(ticker)
            if include_thematic:
                thematic = self._run_thematic(ticker)
        
        # Aggregate signals
        signal = self._aggregate_signals(
            ml_score=ml_percentile,
            sentiment=sentiment,
            fundamental=fundamental,
            sec=sec,
            earnings=earnings,
            insider=insider,
            thematic=thematic,
        )
        
        # Generate summary
        summary, catalysts, risks = self._generate_summary(
            ticker, signal, sentiment, fundamental, sec, earnings
        )
        
        # Get company name
        company_name = ticker
        if fundamental and hasattr(fundamental, 'metrics'):
            company_name = getattr(fundamental.metrics, 'company_name', ticker)
        
        elapsed = time.time() - start_time
        logger.info(f"Completed analysis for {ticker} in {elapsed:.1f}s")
        
        return UnifiedReport(
            ticker=ticker,
            company_name=company_name,
            generated_at=datetime.now(),
            signal=signal,
            ml_analysis={"percentile": ml_percentile} if ml_percentile else None,
            sentiment_analysis=sentiment,
            fundamental_analysis=fundamental,
            sec_analysis=sec,
            earnings_analysis=earnings,
            insider_analysis=insider,
            thematic_analysis=thematic,
            executive_summary=summary,
            key_catalysts=catalysts,
            key_risks=risks,
        )
    
    def analyze_batch(
        self,
        tickers: list[str],
        ml_percentiles: Optional[dict[str, float]] = None,
        clear_memory_between: bool = True,
        **kwargs,
    ) -> dict[str, UnifiedReport]:
        """
        Analyze multiple tickers with memory management.
        
        Args:
            tickers: List of ticker symbols.
            ml_percentiles: Optional dict of ticker -> ML percentile.
            clear_memory_between: If True, clear caches between tickers to manage memory.
            **kwargs: Passed to analyze().
            
        Returns:
            Dict mapping ticker to UnifiedReport.
        """
        import gc
        
        results = {}
        ml_percentiles = ml_percentiles or {}
        
        for i, ticker in enumerate(tickers):
            try:
                report = self.analyze(
                    ticker,
                    ml_percentile=ml_percentiles.get(ticker),
                    **kwargs,
                )
                results[ticker] = report
                
                # Memory management: clear caches periodically to prevent OOM
                if clear_memory_between and (i + 1) % 3 == 0:
                    self._clear_agent_caches()
                    gc.collect()
                    logger.info(f"Memory cleanup after {i + 1} tickers")
                    
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
        
        # Final cleanup
        if clear_memory_between:
            self._clear_agent_caches()
            gc.collect()
        
        return results
    
    def _clear_agent_caches(self):
        """Clear caches from memory-heavy agents."""
        try:
            if self.thematic_agent and hasattr(self.thematic_agent, '_early_adopter_model'):
                ea_model = self.thematic_agent._early_adopter_model
                if ea_model and hasattr(ea_model, 'clear_cache'):
                    ea_model.clear_cache()
        except Exception as e:
            logger.debug(f"Cache clear error: {e}")
    
    def clear_cache(self, namespace: Optional[str] = None) -> int:
        """Clear cached results."""
        if self.cache:
            return self.cache.clear(namespace)
        return 0


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Orchestrator")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--sentiment-mode", choices=["llm", "finbert", "hybrid"], default="hybrid")
    parser.add_argument("--output", "-o", help="Output file (md or json)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = ResearchOrchestrator(
        model=args.model,
        use_cache=not args.no_cache,
        sentiment_mode=args.sentiment_mode,
    )
    
    for ticker in args.tickers:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}...")
        print("="*60)
        
        report = orchestrator.analyze(ticker)
        
        if args.output:
            if args.output.endswith(".json"):
                with open(args.output, "w") as f:
                    json.dump(report.to_dict(), f, indent=2)
            else:
                with open(args.output, "w") as f:
                    f.write(report.to_markdown())
            print(f"Saved to {args.output}")
        else:
            print(report.to_markdown())
