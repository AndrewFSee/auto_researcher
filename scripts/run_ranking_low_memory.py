"""
Memory-Optimized Stock Ranking Pipeline.

This script runs the ML ranking followed by agent analysis in a memory-efficient way:
1. Stage 1: ML Screening (loads XGBoost, computes features, unloads)
2. Stage 2: Agent Analysis (processes one stock at a time, clears memory between)
3. Stage 3: Final Ranking

Usage:
    python scripts/run_ranking_low_memory.py --universe sp100 --ml-top 25 --final-top 10
    python scripts/run_ranking_low_memory.py --universe sp500 --ml-top 50 --final-top 25 --skip-ml
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Results directory (relative to CWD)
RESULTS_DIR = Path("data/ranking_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Report directory (repo root)
REPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "ranking_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _score_percentile(values: list[float], value: float) -> float:
    """Compute percentile rank using midpoint for ties (0-100)."""
    if not values:
        return 50.0
    n = len(values)
    less = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return 100.0 * (less + 0.5 * equal) / n


def _band_from_percentile(pct: float) -> str:
    if pct >= 90:
        return "very_high"
    if pct >= 80:
        return "high"
    if pct <= 10:
        return "very_low"
    if pct <= 20:
        return "low"
    return "neutral"


def add_context_bands(rankings: list["StockRanking"]) -> None:
    """Add percentile/z-score context bands to agent rationales."""
    if not rankings:
        return

    score_map = {
        "sentiment": "sentiment_score",
        "fundamental": "fundamental_score",
        "earnings": "earnings_score",
        "insider": "insider_score",
        "thematic": "thematic_score",
    }

    # Precompute distributions
    distributions: dict[str, dict[str, float | list[float]]] = {}
    for agent, attr in score_map.items():
        values = [getattr(s, attr, 0.0) for s in rankings]
        mean = sum(values) / len(values) if values else 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
        std = var ** 0.5
        distributions[agent] = {"values": values, "mean": mean, "std": std}

    for stock in rankings:
        if stock.agent_rationales is None:
            stock.agent_rationales = {}
        for agent, attr in score_map.items():
            value = getattr(stock, attr, 0.0)
            dist = distributions[agent]
            values = dist["values"]
            pct = _score_percentile(values, value)
            std = dist["std"]
            z = (value - dist["mean"]) / std if std else 0.0
            band = _band_from_percentile(pct)

            agent_entry = stock.agent_rationales.get(agent, {})
            agent_entry["context"] = {
                "percentile": round(pct, 1),
                "zscore": round(z, 2),
                "band": band,
            }
            stock.agent_rationales[agent] = agent_entry


def add_conflict_resolution(rankings: list["StockRanking"]) -> None:
    """Add a short conflict-resolution line when agents disagree."""
    if not rankings:
        return

    score_map = {
        "sentiment": "sentiment_score",
        "fundamental": "fundamental_score",
        "earnings": "earnings_score",
        "insider": "insider_score",
        "thematic": "thematic_score",
    }
    neutral_band = 0.1

    for stock in rankings:
        if stock.agent_rationales is None:
            stock.agent_rationales = {}

        positives: list[str] = []
        negatives: list[str] = []
        active_scores: dict[str, float] = {}

        for agent, attr in score_map.items():
            value = getattr(stock, attr, 0.0)
            if abs(value) < neutral_band:
                continue
            active_scores[agent] = value
            if value > 0:
                positives.append(agent)
            else:
                negatives.append(agent)

        if positives and negatives:
            winner = max(active_scores.items(), key=lambda kv: abs(kv[1]))
            winner_name, winner_val = winner
            winner_dir = "bullish" if winner_val > 0 else "bearish"
            summary = (
                f"Disagreement: bullish from {', '.join(positives)}; "
                f"bearish from {', '.join(negatives)}. "
                f"Net tilt: {winner_dir} led by {winner_name}."
            )
        else:
            summary = "No material disagreement among agent signals."

        stock.agent_rationales["conflict_resolution"] = {
            "summary": summary,
            "bullish_agents": positives,
            "bearish_agents": negatives,
        }


def add_evidence_budget(rankings: list["StockRanking"]) -> None:
    """Ensure each agent emits 2–3 concrete facts supporting the score."""
    if not rankings:
        return

    for stock in rankings:
        if stock.agent_rationales is None:
            stock.agent_rationales = {}

        # Fundamental
        fundamental = stock.agent_rationales.get("fundamental", {})
        evidence: list[str] = []
        strengths = fundamental.get("strengths", []) or []
        concerns = fundamental.get("concerns", []) or []
        for item in strengths[:2]:
            evidence.append(item)
        if len(evidence) < 2:
            for item in concerns[:2]:
                evidence.append(item)
                if len(evidence) >= 2:
                    break
        if fundamental.get("quality_score") is not None:
            evidence.append(f"Quality score: {fundamental.get('quality_score'):.1f}")
        if fundamental.get("value_score") is not None and len(evidence) < 3:
            evidence.append(f"Value score: {fundamental.get('value_score'):.1f}")
        fundamental["evidence"] = evidence[:3]
        stock.agent_rationales["fundamental"] = fundamental

        # Insider
        insider = stock.agent_rationales.get("insider", {})
        evidence = []
        if insider.get("num_insiders") is not None:
            evidence.append(f"Insiders in cluster: {insider.get('num_insiders')}")
        if insider.get("total_value") is not None:
            evidence.append(f"Total value: ${insider.get('total_value'):,.0f}")
        if insider.get("days_since_cluster") is not None:
            evidence.append(f"Days since cluster: {insider.get('days_since_cluster')}")
        insider["evidence"] = evidence[:3]
        stock.agent_rationales["insider"] = insider

        # Earnings
        earnings = stock.agent_rationales.get("earnings", {})
        evidence = []
        for reason in (earnings.get("reasons", []) or [])[:2]:
            evidence.append(reason)
        if earnings.get("days_since_earnings") is not None:
            evidence.append(f"Days since earnings: {earnings.get('days_since_earnings')}")
        earnings["evidence"] = evidence[:3]
        stock.agent_rationales["earnings"] = earnings

        # Thematic
        thematic = stock.agent_rationales.get("thematic", {})
        evidence = []
        if thematic.get("pioneer_score") is not None:
            evidence.append(f"Pioneer score: {thematic.get('pioneer_score'):.2f}")
        if thematic.get("signal_strength") is not None:
            evidence.append(f"Signal strength: {thematic.get('signal_strength'):+.2f}")
        adopted = thematic.get("genai_adopted", []) or []
        if adopted:
            evidence.append(f"GenAI adopted: {', '.join(adopted[:3])}")
        thematic["evidence"] = evidence[:3]
        stock.agent_rationales["thematic"] = thematic

        # Sentiment
        sentiment = stock.agent_rationales.get("sentiment", {})
        evidence = []
        if sentiment.get("news_count") is not None:
            evidence.append(f"News count: {sentiment.get('news_count')}")
        if sentiment.get("sentiment_label"):
            evidence.append(f"Sentiment label: {sentiment.get('sentiment_label')}")
        if sentiment.get("confidence") is not None:
            evidence.append(f"Confidence: {sentiment.get('confidence'):.2f}")
        sentiment["evidence"] = evidence[:3]
        stock.agent_rationales["sentiment"] = sentiment


def add_consistency_checks(rankings: list["StockRanking"]) -> None:
    """Flag cases where summary keywords conflict with score direction."""
    if not rankings:
        return

    keyword_map = {
        "bullish": 1,
        "positive": 1,
        "strong": 1,
        "moderate": 1,
        "buy": 1,
        "long": 1,
        "bearish": -1,
        "negative": -1,
        "weak": -1,
        "sell": -1,
        "short": -1,
        "miss": -1,
        "beat": 1,
    }

    score_map = {
        "sentiment": "sentiment_score",
        "fundamental": "fundamental_score",
        "earnings": "earnings_score",
        "insider": "insider_score",
        "thematic": "thematic_score",
    }

    for stock in rankings:
        if stock.agent_rationales is None:
            stock.agent_rationales = {}

        for agent, attr in score_map.items():
            entry = stock.agent_rationales.get(agent, {})
            summary = str(entry.get("summary", "")).lower()
            if not summary:
                continue
            score = getattr(stock, attr, 0.0)
            score_dir = 0
            if score > 0.1:
                score_dir = 1
            elif score < -0.1:
                score_dir = -1

            keyword_dir = 0
            for word, direction in keyword_map.items():
                if word in summary:
                    keyword_dir += direction

            if keyword_dir > 0:
                keyword_dir = 1
            elif keyword_dir < 0:
                keyword_dir = -1

            conflict = score_dir != 0 and keyword_dir != 0 and score_dir != keyword_dir
            entry["consistency"] = {
                "summary_direction": keyword_dir,
                "score_direction": score_dir,
                "conflict": conflict,
            }
            stock.agent_rationales[agent] = entry


def add_lightweight_cross_validation(rankings: list["StockRanking"]) -> None:
    """Cross-validate agent signals against recent price returns."""
    if not rankings:
        return

    try:
        import pandas as pd
        from auto_researcher.data.price_loader import download_price_history, get_adjusted_close
    except Exception:
        return

    tickers = [s.ticker for s in rankings]
    if not tickers:
        return

    end = datetime.now().date()
    start = (datetime.now() - pd.Timedelta(days=120)).date()

    try:
        prices = download_price_history(tickers, start=str(start), end=str(end))
        adj = get_adjusted_close(prices)
    except Exception as e:
        for stock in rankings:
            stock.errors.append(f"CrossVal: {str(e)[:50]}")
        return

    ret_5d = adj.pct_change(5)
    ret_20d = adj.pct_change(20)

    score_map = {
        "sentiment": "sentiment_score",
        "fundamental": "fundamental_score",
        "earnings": "earnings_score",
        "insider": "insider_score",
        "thematic": "thematic_score",
    }

    for stock in rankings:
        if stock.agent_rationales is None:
            stock.agent_rationales = {}
        ticker = stock.ticker
        try:
            r5 = float(ret_5d[ticker].dropna().iloc[-1])
        except Exception:
            r5 = None
        try:
            r20 = float(ret_20d[ticker].dropna().iloc[-1])
        except Exception:
            r20 = None

        for agent, attr in score_map.items():
            entry = stock.agent_rationales.get(agent, {})
            score = getattr(stock, attr, 0.0)
            score_dir = 1 if score > 0.1 else -1 if score < -0.1 else 0

            def _agreement(ret: float | None) -> str:
                if ret is None or score_dir == 0:
                    return "neutral"
                return "supports" if (ret > 0 and score_dir > 0) or (ret < 0 and score_dir < 0) else "conflicts"

            entry["cross_validation"] = {
                "ret_5d": round(r5, 4) if r5 is not None else None,
                "ret_20d": round(r20, 4) if r20 is not None else None,
                "agreement_5d": _agreement(r5),
                "agreement_20d": _agreement(r20),
            }
            stock.agent_rationales[agent] = entry


def add_risk_qualifiers(rankings: list["StockRanking"]) -> None:
    """Append risk flags for stale data, earnings age, drawdown, and low news."""
    if not rankings:
        return

    try:
        import pandas as pd
        from auto_researcher.data.price_loader import download_price_history, get_adjusted_close
    except Exception:
        return

    tickers = [s.ticker for s in rankings]
    end = datetime.now().date()
    start = (datetime.now() - pd.Timedelta(days=365)).date()

    max_dd_by_ticker: dict[str, float] = {}
    try:
        prices = download_price_history(tickers, start=str(start), end=str(end))
        adj = get_adjusted_close(prices)
        for t in tickers:
            if t not in adj.columns:
                continue
            series = adj[t].dropna()
            if series.empty:
                continue
            running_max = series.cummax()
            drawdown = (series / running_max) - 1.0
            max_dd_by_ticker[t] = float(drawdown.min())
    except Exception as e:
        for stock in rankings:
            stock.errors.append(f"RiskQual: {str(e)[:50]}")

    for stock in rankings:
        if stock.agent_rationales is None:
            stock.agent_rationales = {}

        flags: list[str] = []

        # Earnings age / staleness
        earnings = stock.agent_rationales.get("earnings", {})
        freshness = earnings.get("data_freshness")
        days_since = earnings.get("days_since_earnings")
        if freshness in {"stale", "expired"}:
            flags.append(f"earnings_{freshness}")
        if isinstance(days_since, (int, float)) and days_since > 60:
            flags.append("earnings_old")

        # Low news volume
        sentiment = stock.agent_rationales.get("sentiment", {})
        news_count = sentiment.get("news_count")
        if isinstance(news_count, (int, float)) and news_count < 5:
            flags.append("low_news_volume")

        # High drawdown
        max_dd = max_dd_by_ticker.get(stock.ticker)
        if max_dd is not None and max_dd <= -0.4:
            flags.append("high_drawdown")

        stock.agent_rationales["risk_qualifiers"] = {
            "flags": flags,
            "max_drawdown_1y": round(max_dd, 4) if max_dd is not None else None,
            "news_count": news_count,
            "days_since_earnings": days_since,
            "earnings_freshness": freshness,
        }


def apply_freshness_gates(rankings: list["StockRanking"]) -> None:
    """Auto-downgrade confidence when data is stale and explain why."""
    if not rankings:
        return

    for stock in rankings:
        if stock.agent_rationales is None:
            stock.agent_rationales = {}

        gating = {
            "applied": False,
            "reasons": [],
            "confidence_before": None,
            "confidence_after": None,
        }

        # Earnings freshness gate
        earnings = stock.agent_rationales.get("earnings", {})
        freshness = earnings.get("data_freshness")
        if freshness in {"stale", "expired"}:
            conf = earnings.get("confidence", 1.0)
            if conf is None:
                conf = 1.0
            new_conf = round(conf * (0.6 if freshness == "stale" else 0.3), 3)
            earnings["confidence"] = new_conf
            gating["applied"] = True
            gating["reasons"].append(f"earnings_data_{freshness}")
            gating["confidence_before"] = conf
            gating["confidence_after"] = new_conf
            stock.agent_rationales["earnings"] = earnings

        # Sentiment freshness gate (low news volume proxy)
        sentiment = stock.agent_rationales.get("sentiment", {})
        news_count = sentiment.get("news_count")
        if isinstance(news_count, (int, float)) and news_count < 5:
            conf = sentiment.get("confidence", 1.0)
            if conf is None:
                conf = 1.0
            new_conf = round(conf * 0.7, 3)
            sentiment["confidence"] = new_conf
            gating["applied"] = True
            gating["reasons"].append("low_news_volume")
            gating["confidence_before"] = conf if gating["confidence_before"] is None else gating["confidence_before"]
            gating["confidence_after"] = new_conf
            stock.agent_rationales["sentiment"] = sentiment

        stock.agent_rationales["freshness_gate"] = gating


@dataclass
class StockRanking:
    """Ranking data for a single stock."""
    ticker: str
    ml_score: float = 0.0
    ml_percentile: float = 50.0
    ml_rank: int = 0
    ml_ic: float = 0.15  # Historical IC of ML model for weighting
    
    # Agent scores (-1 to +1)
    sentiment_score: float = 0.0
    fundamental_score: float = 0.0
    earnings_score: float = 0.0
    insider_score: float = 0.0
    thematic_score: float = 0.0
    momentum_score: float = 0.0
    filing_tone_score: float = 0.0
    earnings_call_qual_score: float = 0.0
    
    # Composite
    composite_score: float = 0.0
    composite_confidence: float = 0.5
    final_rank: int = 0
    signal: str = "hold"
    
    # Metadata
    company_name: str = ""
    sector: str = ""
    key_factors: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    agent_rationales: dict = field(default_factory=dict)


def run_ml_screening(
    universe: str = "sp100",
    top_k: int = 25,
) -> tuple[list[StockRanking], float]:
    """
    Stage 1: Run ML model to screen universe.
    
    Loads the XGBoost model, generates predictions, then unloads.
    Returns rankings and model IC for weighting.
    """
    logger.info(f"=" * 60)
    logger.info(f"STAGE 1: ML SCREENING (6-month rolling window)")
    logger.info(f"=" * 60)
    
    # Import here to avoid loading until needed
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from recommend import generate_recommendations, UNIVERSES
    
    # Get tickers
    if universe in UNIVERSES:
        tickers = UNIVERSES[universe]()
    else:
        raise ValueError(f"Unknown universe: {universe}")
    
    logger.info(f"Universe: {universe} ({len(tickers)} stocks)")
    logger.info(f"Selecting top {top_k} by ML score...")
    
    try:
        recommendations, all_scores, prices = generate_recommendations(
            tickers=tickers,
            top_k=top_k,
            explain=False,
        )
        
        # Get the ML model's historical IC from the recommendations metadata
        # The generate_recommendations function now returns IC in recommendations[0] metadata
        ml_ic = 0.15  # Default
        if recommendations and hasattr(recommendations[0], 'predicted_return'):
            # We store IC in a module-level variable as a workaround
            import recommend
            if hasattr(recommend, '_last_model_ic'):
                ml_ic = recommend._last_model_ic
        
        logger.info(f"ML model historical IC: {ml_ic:.3f}")
        
        # Convert to StockRanking objects
        rankings = []
        for rec in recommendations:
            rankings.append(StockRanking(
                ticker=rec.ticker,
                ml_score=rec.score,
                ml_percentile=rec.percentile,
                ml_rank=rec.rank,
                ml_ic=ml_ic,
                key_factors=rec.drivers[:3] if rec.drivers else [],
            ))
        
        logger.info(f"ML screening complete: {len(rankings)} candidates")
        
        # Clear memory
        del recommendations, all_scores, prices
        clear_memory()
        
        return rankings, ml_ic
        
    except Exception as e:
        logger.error(f"ML screening failed: {e}")
        raise


def load_ml_results(filepath: Path) -> list[StockRanking]:
    """Load ML screening results from file."""
    with open(filepath) as f:
        data = json.load(f)
    
    rankings = []
    for item in data:
        rankings.append(StockRanking(**item))
    
    return rankings


def save_results(rankings: list[StockRanking], filepath: Path):
    """Save rankings to JSON file."""
    data = [asdict(r) for r in rankings]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Results saved to {filepath}")


def analyze_single_stock_lightweight(
    ticker: str,
    ml_percentile: float = 50.0,
    verbose: bool = False,
) -> dict:
    """
    Analyze a single stock with minimal memory usage.
    
    Loads only the models needed, processes, then clears.
    """
    results = {
        "sentiment_score": 0.0,
        "fundamental_score": 0.0,
        "earnings_score": 0.0,
        "insider_score": 0.0,
        "thematic_score": 0.0,
        "momentum_score": 0.0,
        "filing_tone_score": 0.0,
        "earnings_call_qual_score": 0.0,
        "company_name": ticker,
        "sector": "",
        "errors": [],
        "agent_rationales": {},
    }
    
    if verbose:
        logger.info(f"      [AGENTS] Starting analysis for {ticker}")
    
    # 1. Fundamental Analysis (lightweight - just API calls)
    try:
        from auto_researcher.models.quality_value import QualityValueModel
        if verbose:
            logger.info(f"      [FUNDAMENTAL] Loading QualityValueModel...")
        model = QualityValueModel()
        signal = model.get_signal(ticker)
        if signal:
            # Normalize score to -1 to +1
            quality_norm = (signal.quality_score / 100) if signal.quality_score else 0.5
            value_norm = (signal.value_score / 100) if signal.value_score else 0.5
            combined = (quality_norm + value_norm) / 2
            results["fundamental_score"] = (combined - 0.5) * 2  # Map 0-1 to -1 to +1
            # Extract company name from yfinance data
            if getattr(signal, 'company_name', '') and signal.company_name != ticker:
                results["company_name"] = signal.company_name
            # Extract sector from fundamental signal as well
            if getattr(signal, 'sector', '') and not results["sector"]:
                results["sector"] = signal.sector
            results["agent_rationales"]["fundamental"] = {
                "summary": getattr(signal, "summary", ""),
                "strengths": getattr(signal, "strengths", []),
                "concerns": getattr(signal, "concerns", []),
                "quality_score": getattr(signal, "quality_score", None),
                "value_score": getattr(signal, "value_score", None),
            }
            if verbose:
                logger.info(f"      [FUNDAMENTAL] quality={signal.quality_score:.1f}, value={signal.value_score:.1f} -> score={results['fundamental_score']:.3f}")
        else:
            results["agent_rationales"]["fundamental"] = {
                "missing": True,
                "reason": "no_signal",
            }
            if verbose:
                logger.info(f"      [FUNDAMENTAL] No signal returned -> score=0.0")
        del model
        clear_memory()
    except Exception as e:
        results["errors"].append(f"Fundamental: {str(e)[:50]}")
        results["agent_rationales"]["fundamental"] = {
            "missing": True,
            "reason": "error",
        }
        if verbose:
            logger.warning(f"      [FUNDAMENTAL] ERROR: {str(e)[:80]}")
    
    # 2. Insider Analysis (lightweight)
    try:
        from auto_researcher.models.insider_cluster import InsiderClusterModel
        if verbose:
            logger.info(f"      [INSIDER] Loading InsiderClusterModel...")
        model = InsiderClusterModel()
        signal = model.get_signal(ticker)
        if signal and signal.direction:
            # Map direction + strength to score
            direction_mult = 1.0 if signal.direction == "long" else -1.0
            strength_map = {"strong": 0.8, "moderate": 0.5, "weak": 0.2, "none": 0.0}
            strength_val = strength_map.get(signal.strength, 0.0)
            results["insider_score"] = direction_mult * strength_val
            results["agent_rationales"]["insider"] = {
                "summary": getattr(signal, "summary", ""),
                "direction": getattr(signal, "direction", None),
                "strength": getattr(signal, "strength", None),
                "num_insiders": getattr(signal, "num_insiders", None),
                "total_value": getattr(signal, "total_value", None),
                "days_since_cluster": getattr(signal, "days_since_cluster", None),
            }
            if verbose:
                logger.info(f"      [INSIDER] direction={signal.direction}, strength={signal.strength} -> score={results['insider_score']:.3f}")
        else:
            results["agent_rationales"]["insider"] = {
                "missing": True,
                "reason": "no_signal",
            }
            if verbose:
                logger.info(f"      [INSIDER] No actionable signal -> score=0.0")
        del model
        clear_memory()
    except Exception as e:
        results["errors"].append(f"Insider: {str(e)[:50]}")
        results["agent_rationales"]["insider"] = {
            "missing": True,
            "reason": "error",
        }
        if verbose:
            logger.warning(f"      [INSIDER] ERROR: {str(e)[:80]}")
    
    # 3. Earnings/PEAD (lightweight)
    try:
        from auto_researcher.models.pead_enhanced import EnhancedPEADModel
        if verbose:
            logger.info(f"      [EARNINGS] Loading EnhancedPEADModel...")
        model = EnhancedPEADModel()
        signal = model.get_signal(ticker)
        if signal and signal.is_actionable:
            # Convert PEAD signal to score: direction * strength mapping
            direction_mult = {"long": 1.0, "short": -1.0, "neutral": 0.0}.get(signal.direction, 0.0)
            strength_mult = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}.get(signal.strength, 0.0)
            # Also factor in data freshness
            freshness_mult = {"fresh": 1.0, "recent": 0.8, "stale": 0.5, "expired": 0.0}.get(signal.data_freshness, 0.0)
            results["earnings_score"] = direction_mult * strength_mult * freshness_mult
            results["agent_rationales"]["earnings"] = {
                "summary": getattr(signal, "summary", ""),
                "reasons": getattr(signal, "reasons", []),
                "direction": getattr(signal, "direction", None),
                "strength": getattr(signal, "strength", None),
                "data_freshness": getattr(signal, "data_freshness", None),
                "days_since_earnings": getattr(signal, "days_since_earnings", None),
                "revenue_sue": getattr(signal, "revenue_sue", None),
                "has_revenue_surprise": getattr(signal, "has_revenue_surprise", False),
                "dual_beat": getattr(signal, "dual_beat", False),
                "dual_miss": getattr(signal, "dual_miss", False),
                "composite_sue": getattr(signal, "composite_sue", None),
            }
            if verbose:
                logger.info(f"      [EARNINGS] direction={signal.direction}, strength={signal.strength}, freshness={signal.data_freshness} -> score={results['earnings_score']:.3f}")
        else:
            if signal:
                results["agent_rationales"]["earnings"] = {
                    "missing": False,
                    "reason": "not_actionable",
                    "data_freshness": getattr(signal, "data_freshness", None),
                    "days_since_earnings": getattr(signal, "days_since_earnings", None),
                }
                if verbose:
                    logger.info(f"      [EARNINGS] Not actionable (freshness={signal.data_freshness}, days_since={signal.days_since_earnings}) -> score=0.0")
            else:
                results["agent_rationales"]["earnings"] = {
                    "missing": True,
                    "reason": "no_signal",
                }
                if verbose:
                    logger.info(f"      [EARNINGS] No signal returned -> score=0.0")
        del model
        clear_memory()
    except Exception as e:
        results["errors"].append(f"Earnings: {str(e)[:50]}")
        results["agent_rationales"]["earnings"] = {
            "missing": True,
            "reason": "error",
        }
        if verbose:
            logger.warning(f"      [EARNINGS] ERROR: {str(e)[:80]}")
    
    # 4. Thematic/Early Adopter (fast mode - uses precomputed baselines)
    try:
        from auto_researcher.models.early_adopter import EarlyAdopterModel
        if verbose:
            logger.info(f"      [THEMATIC] Loading EarlyAdopterModel (fast mode)...")
        model = EarlyAdopterModel(lookback_years=3, focus_recent=True)
        signal = model.analyze_company(ticker, fast_mode=True)  # Fast mode: only load target ticker
        if signal:
            # pioneer_score is 0-1, signal_strength is -0.3 to 1.0
            results["thematic_score"] = signal.signal_strength if hasattr(signal, 'signal_strength') else 0.0
            # Use detected sector from EarlyAdopterModel (sector-aware)
            if hasattr(signal, 'detected_sector') and signal.detected_sector:
                results["sector"] = signal.detected_sector
            results["agent_rationales"]["thematic"] = {
                "signal": getattr(signal, "signal", None),
                "signal_strength": getattr(signal, "signal_strength", None),
                "pioneer_score": getattr(signal, "pioneer_score", None),
                "genai_adopted": getattr(signal, "genai_adopted", []),
                "detected_sector": getattr(signal, "detected_sector", ""),
                "sector_innovation_score": getattr(signal, "sector_innovation_score", 0.0),
                "sector_relevant_techs": getattr(signal, "sector_relevant_techs", []),
            }
            if verbose:
                genai_str = ", ".join(signal.genai_adopted[:3]) if signal.genai_adopted else "none"
                sector_str = signal.detected_sector if hasattr(signal, 'detected_sector') and signal.detected_sector else "unknown"
                sec_innov = signal.sector_innovation_score if hasattr(signal, 'sector_innovation_score') else 0.0
                logger.info(f"      [THEMATIC] pioneer={signal.pioneer_score:.3f}, sector={sector_str}, sector_innov={sec_innov:.3f}, genai=[{genai_str}], signal={signal.signal} -> score={results['thematic_score']:.3f}")
        else:
            results["agent_rationales"]["thematic"] = {
                "missing": True,
                "reason": "no_signal",
            }
            if verbose:
                logger.info(f"      [THEMATIC] No signal returned -> score=0.0")
        model.clear_cache()
        del model
        clear_memory()
    except Exception as e:
        results["errors"].append(f"Thematic: {str(e)[:50]}")
        results["agent_rationales"]["thematic"] = {
            "missing": True,
            "reason": "error",
        }
        if verbose:
            logger.warning(f"      [THEMATIC] ERROR: {str(e)[:80]}")
    
    # 5. Sector Momentum (lightweight - ETF relative strength)
    try:
        from auto_researcher.models.sector_momentum import SectorMomentumModel
        if verbose:
            logger.info(f"      [MOMENTUM] Loading SectorMomentumModel...")
        model = SectorMomentumModel()
        signal = model.get_stock_signal(ticker)
        if signal and signal.is_actionable:
            # sector_boost ranges from -0.25 to +0.30, normalize to -1/+1
            results["momentum_score"] = max(-1.0, min(1.0, signal.sector_boost * 3.33))
            if signal.sector:
                results["sector"] = signal.sector
            results["agent_rationales"]["momentum"] = {
                "summary": getattr(signal, "summary", ""),
                "rationale": getattr(signal, "rationale", ""),
                "direction": getattr(signal, "direction", None),
                "strength": getattr(signal, "strength", None),
                "sector": getattr(signal, "sector", None),
                "sector_boost": getattr(signal, "sector_boost", None),
                "expected_alpha": getattr(signal, "expected_alpha", None),
            }
            if verbose:
                logger.info(f"      [MOMENTUM] sector={signal.sector}, direction={signal.direction}, strength={signal.strength}, boost={signal.sector_boost:+.3f} -> score={results['momentum_score']:.3f}")
        else:
            results["agent_rationales"]["momentum"] = {
                "missing": False if signal else True,
                "reason": "not_actionable" if signal else "no_signal",
                "sector": getattr(signal, "sector", None) if signal else None,
            }
            if verbose:
                logger.info(f"      [MOMENTUM] {'Not actionable' if signal else 'No signal'} -> score=0.0")
        del model
        clear_memory()
    except Exception as e:
        results["errors"].append(f"Momentum: {str(e)[:50]}")
        results["agent_rationales"]["momentum"] = {
            "missing": True,
            "reason": "error",
        }
        if verbose:
            logger.warning(f"      [MOMENTUM] ERROR: {str(e)[:80]}")

    # 6. Sentiment (HEAVY - Hybrid FinBERT + LLM with RAG)
    # Using SentimentAgent in hybrid mode for RAG-grounded analysis
    try:
        from auto_researcher.agents.sentiment_agent import SentimentAgent
        # Try hybrid mode (FinBERT + LLM with RAG context)
        # Falls back to finbert_only if LLM/litellm not available
        try:
            import litellm
            use_hybrid = True
        except ImportError:
            use_hybrid = False
        
        mode_label = "Hybrid+RAG" if use_hybrid else "FinBERT"
        if verbose:
            logger.info(f"      [SENTIMENT] Loading SentimentAgent ({mode_label} mode)...")
        agent = SentimentAgent(hybrid_mode=use_hybrid, finbert_only=not use_hybrid)
        result = agent.analyze_ticker(ticker)
        if result:
            # sentiment_score is typically -1 to +1
            results["sentiment_score"] = result.sentiment_score if hasattr(result, 'sentiment_score') else 0.0
            results["agent_rationales"]["sentiment"] = {
                "summary": getattr(result, "summary", ""),
                "sentiment_label": getattr(result, "sentiment_label", None),
                "confidence": getattr(result, "confidence", None),
                "key_themes": getattr(result, "key_themes", []),
                "news_count": getattr(result, "news_count", None),
                "earnings_topic_signal": getattr(result, "earnings_topic_signal", None),
                "earnings_expected_alpha": getattr(result, "earnings_expected_alpha", None),
                "topic_composite_score": getattr(result, "topic_composite_score", None),
                "earnings_topic_score": getattr(result, "earnings_topic_score", None),
                "topic_ic_score": getattr(result, "topic_ic_score", None),
                "topic_ic_confidence": getattr(result, "topic_ic_confidence", None),
                "topic_ic_alpha": getattr(result, "topic_ic_alpha", None),
                "sentiment_score_base": getattr(result, "sentiment_score_base", None),
                "method": getattr(result, "method", None),
                "llm_agrees_with_finbert": getattr(result, "llm_agrees_with_finbert", None),
            }
            if verbose:
                method = getattr(result, 'method', 'unknown')
                logger.info(f"      [SENTIMENT] score={results['sentiment_score']:.3f}, label={result.sentiment_label}, method={method}, news_count={result.news_count}")
        else:
            results["agent_rationales"]["sentiment"] = {
                "missing": True,
                "reason": "no_signal",
            }
            if verbose:
                logger.info(f"      [SENTIMENT] No result returned -> score=0.0")
        # Cleanup FinBERT model and vector store
        if hasattr(agent, 'finbert_analyzer') and agent.finbert_analyzer:
            agent.finbert_analyzer.unload()
        if hasattr(agent, '_vectorstore') and agent._vectorstore:
            agent._vectorstore = None
        del agent
        clear_memory()
    except Exception as e:
        results["errors"].append(f"Sentiment: {str(e)[:50]}")
        results["agent_rationales"]["sentiment"] = {
            "missing": True,
            "reason": "error",
        }
        if verbose:
            logger.warning(f"      [SENTIMENT] ERROR: {str(e)[:80]}")
    
    # 7. Filing Tone (seasonal - 10-K YoY tone change via Loughran-McDonald)
    try:
        from auto_researcher.models.filing_tone import FilingToneModel
        if verbose:
            logger.info(f"      [FILING_TONE] Loading FilingToneModel...")
        model = FilingToneModel()
        signal = model.get_signal(ticker)
        if signal and signal.is_actionable:
            # net_tone_change is typically -0.15 to +0.15, normalize to -1/+1
            # Scale by ~6.67 so strong_change (0.06) maps to ~0.4, strong (0.15) → ~1.0
            raw = signal.net_tone_change * 6.67
            # Apply signal_decay so stale filings fade
            raw *= signal.signal_decay
            results["filing_tone_score"] = max(-1.0, min(1.0, raw))
            results["agent_rationales"]["filing_tone"] = {
                "summary": getattr(signal, "summary", ""),
                "rationale": getattr(signal, "rationale", ""),
                "direction": getattr(signal, "direction", None),
                "strength": getattr(signal, "strength", None),
                "net_tone_change": getattr(signal, "net_tone_change", None),
                "expected_alpha": getattr(signal, "expected_alpha", None),
                "days_since_filing": getattr(signal, "days_since_filing", None),
                "signal_decay": getattr(signal, "signal_decay", None),
            }
            if verbose:
                logger.info(f"      [FILING_TONE] direction={signal.direction}, strength={signal.strength}, tone_change={signal.net_tone_change:+.4f}, decay={signal.signal_decay:.2f} -> score={results['filing_tone_score']:.3f}")
        else:
            results["agent_rationales"]["filing_tone"] = {
                "missing": False if signal else True,
                "reason": "not_actionable" if signal else "no_signal",
                "summary": getattr(signal, "summary", "") if signal else "",
            }
            if verbose:
                logger.info(f"      [FILING_TONE] {'Not actionable' if signal else 'No signal'} -> score=0.0")
        del model
        clear_memory()
    except Exception as e:
        results["errors"].append(f"FilingTone: {str(e)[:50]}")
        results["agent_rationales"]["filing_tone"] = {
            "missing": True,
            "reason": "error",
        }
        if verbose:
            logger.warning(f"      [FILING_TONE] ERROR: {str(e)[:80]}")

    # 8. Earnings Call Qualitative (FinBERT tone + hedging + guidance + peer comparison on transcript text)
    try:
        from auto_researcher.models.earnings_call_qual import EarningsCallQualModel
        if verbose:
            logger.info(f"      [CALL_QUAL] Loading EarningsCallQualModel...")
        # Load transcript vectorstore for RAG-based peer comparison (if available)
        transcript_vs = None
        try:
            from auto_researcher.data.transcript_vectorstore import TranscriptVectorStore, DEFAULT_CHROMA_PATH
            if DEFAULT_CHROMA_PATH.exists():
                transcript_vs = TranscriptVectorStore()
                if transcript_vs.get_index_count() > 0:
                    if verbose:
                        logger.info(f"      [CALL_QUAL] Transcript vectorstore loaded ({transcript_vs.get_index_count():,} chunks)")
                else:
                    transcript_vs = None
                    if verbose:
                        logger.info(f"      [CALL_QUAL] Transcript vectorstore empty, peer comparison disabled")
        except Exception as vs_err:
            if verbose:
                logger.info(f"      [CALL_QUAL] No transcript vectorstore: {vs_err}")
        model = EarningsCallQualModel(use_finbert=True, transcript_vectorstore=transcript_vs)
        signal = model.get_signal(ticker)
        if signal and signal.is_actionable:
            results["earnings_call_qual_score"] = max(-1.0, min(1.0, signal.composite_score))
            results["agent_rationales"]["earnings_call_qual"] = {
                "summary": signal.summary,
                "rationale": signal.rationale,
                "signal": signal.signal,
                "management_tone": signal.management_tone,
                "tone_gap_score": signal.tone_gap_score,
                "hedging_score": signal.hedging_score,
                "guidance_specificity_score": signal.guidance_specificity_score,
                "tone_change_score": signal.tone_change_score,
                "peer_tone_delta_score": signal.peer_tone_delta_score,
                "composite_score": signal.composite_score,
                "days_since_call": signal.days_since_call,
                "signal_decay": signal.signal_decay,
                "confidence": signal.confidence,
            }
            if verbose:
                peer_str = f", peer_delta={signal.peer_tone_delta_score:+.2f}" if transcript_vs else ""
                logger.info(f"      [CALL_QUAL] signal={signal.signal}, mgmt_tone={signal.management_tone:+.2f}, hedging={signal.hedging_score:+.2f}, guidance={signal.guidance_specificity_score:+.2f}, change={signal.tone_change_score:+.2f}{peer_str} -> score={results['earnings_call_qual_score']:.3f}")
        else:
            results["agent_rationales"]["earnings_call_qual"] = {
                "missing": False if signal else True,
                "reason": "not_actionable" if signal else "no_signal",
                "summary": signal.summary if signal else "",
            }
            if verbose:
                logger.info(f"      [CALL_QUAL] {'Not actionable' if signal else 'No signal'} -> score=0.0")
        model.unload()
        del model, transcript_vs
        clear_memory()
    except Exception as e:
        results["errors"].append(f"EarningsCallQual: {str(e)[:50]}")
        results["agent_rationales"]["earnings_call_qual"] = {
            "missing": True,
            "reason": "error",
        }
        if verbose:
            logger.warning(f"      [CALL_QUAL] ERROR: {str(e)[:80]}")

    if verbose:
        logger.info(f"      [AGENTS] Final scores: sent={results['sentiment_score']:.3f}, fund={results['fundamental_score']:.3f}, earn={results['earnings_score']:.3f}, ins={results['insider_score']:.3f}, them={results['thematic_score']:.3f}, mom={results['momentum_score']:.3f}, tone={results['filing_tone_score']:.3f}, call_qual={results['earnings_call_qual_score']:.3f}")
    
    return results


def run_agent_analysis(
    rankings: list[StockRanking],
    batch_size: int = 5,
    verbose: bool = False,
) -> list[StockRanking]:
    """
    Stage 2: Run agent analysis on ML candidates.
    
    Processes stocks one at a time to minimize memory usage.
    """
    logger.info(f"=" * 60)
    logger.info(f"STAGE 2: AGENT ANALYSIS")
    logger.info(f"=" * 60)
    logger.info(f"Analyzing {len(rankings)} stocks...")
    logger.info(f"Agents to run: QualityValueModel, InsiderClusterModel, EnhancedPEADModel, EarlyAdopterModel, SectorMomentumModel, TopicSentimentModel, FilingToneModel, EarningsCallQualModel")
    if verbose:
        logger.info(f"[VERBOSE MODE] Detailed agent logging enabled")
    
    # Pre-index transcript vectorstore for all pipeline tickers
    try:
        from auto_researcher.data.transcript_vectorstore import TranscriptVectorStore
        all_tickers = [s.ticker for s in rankings]
        tvs = TranscriptVectorStore()
        n_new = tvs.ensure_tickers_indexed(all_tickers, min_year=2023)
        if n_new > 0:
            logger.info(f"  [TRANSCRIPT_VS] Indexed {n_new:,} new chunks for {len(all_tickers)} tickers")
        del tvs
    except Exception as e:
        logger.info(f"  [TRANSCRIPT_VS] Skipping pre-index: {e}")

    total = len(rankings)
    start_time = time.time()
    
    for i, stock in enumerate(rankings, 1):
        ticker = stock.ticker
        logger.info(f"  [{i}/{total}] Analyzing {ticker}...")
        
        try:
            results = analyze_single_stock_lightweight(
                ticker=ticker,
                ml_percentile=stock.ml_percentile,
                verbose=verbose,
            )
            
            stock.sentiment_score = results["sentiment_score"]
            stock.fundamental_score = results["fundamental_score"]
            stock.earnings_score = results["earnings_score"]
            stock.insider_score = results["insider_score"]
            stock.thematic_score = results["thematic_score"]
            stock.momentum_score = results["momentum_score"]
            stock.filing_tone_score = results["filing_tone_score"]
            stock.earnings_call_qual_score = results["earnings_call_qual_score"]
            stock.company_name = results["company_name"]
            stock.sector = results["sector"]
            stock.errors = results["errors"]
            stock.agent_rationales = results.get("agent_rationales", {})
            
        except Exception as e:
            logger.error(f"    Failed: {e}")
            stock.errors.append(str(e))
        
        # Clear memory every batch_size stocks
        if i % batch_size == 0:
            clear_memory()
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            logger.info(f"    Progress: {i}/{total} ({i/total*100:.1f}%), ETA: {eta/60:.1f} min")
    
    logger.info(f"Agent analysis complete")
    return rankings


def compute_composite_scores(
    rankings: list[StockRanking],
    ml_weight: float = 0.35,
    verbose: bool = False,
    ml_ic: float = 0.15,
    enable_factor_rotation: bool = True,
) -> list[StockRanking]:
    """
    Stage 3: Compute final composite scores using IC-weighted model combination.
    
    Models are weighted proportionally to their historical IC (predictive power).
    This gives more influence to models that have been more accurate.
    """
    logger.info(f"=" * 60)
    logger.info(f"STAGE 3: IC-WEIGHTED COMPOSITE SCORING")
    logger.info(f"=" * 60)
    
    # Default ICs (fallback if calibration file missing)
    default_ics = {
        "ml": 0.15,
        "sentiment": 0.08,
        "fundamental": 0.10,
        "earnings": 0.12,
        "insider": 0.06,
        "thematic": 0.05,
        "momentum": 0.07,
        "filing_tone": 0.04,
        "earnings_call_qual": 0.05,
    }
    
    # Load empirically calibrated ICs from data/agent_ic.json if available
    agent_ic_file = RESULTS_DIR.parent / "agent_ic.json"
    model_ics = dict(default_ics)  # start with defaults
    if agent_ic_file.exists():
        try:
            with open(agent_ic_file) as f:
                calibrated = json.load(f)
            ic_source = "calibrated"
            for agent in ["sentiment", "fundamental", "earnings", "insider", "thematic", "momentum", "filing_tone", "earnings_call_qual"]:
                if agent in calibrated and "mean_ic" in calibrated[agent]:
                    # Use absolute IC (direction doesn't matter for weighting)
                    # Apply floor of 0.02 so no agent gets zero weight
                    empirical_ic = abs(calibrated[agent]["mean_ic"])
                    model_ics[agent] = max(empirical_ic, 0.02)
            logger.info(f"Loaded calibrated ICs from {agent_ic_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load calibrated ICs: {e}, using defaults")
            ic_source = "defaults"
    else:
        ic_source = "defaults"
        logger.info(f"No calibration file found, using default ICs")
    
    # ML IC comes from runtime OOS calculation
    model_ics["ml"] = max(ml_ic, 0.05)
    
    # ── Factor Rotation: Regime-Aware IC Adjustment ──────────────────
    # Dynamically adjust factor weights based on leading indicators
    # (VIX term structure, credit spreads, factor momentum, etc.)
    regime_state = None
    if enable_factor_rotation:
        try:
            from auto_researcher.models.factor_rotation import FactorRotationModel
            rotation_model = FactorRotationModel(
                tilt_strength=0.5,
                use_factor_momentum=True,
                verbose=verbose,
            )
            regime_state = rotation_model.detect_regime(
                rankings=[asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in rankings[:50]]
            )
            model_ics = rotation_model.adjust_ic_weights(model_ics, regime_state)
            logger.info(f"Factor rotation: {regime_state.regime.value} "
                        f"(conf={regime_state.regime_confidence:.0%}, "
                        f"score={regime_state.regime_score:+.3f})")
            if regime_state.transition_probability > 0.1:
                logger.info(f"  ⚠️ Transition alert: {regime_state.transition_probability:.0%} "
                            f"probability → {regime_state.transition_direction}")
            del rotation_model
        except ImportError:
            logger.debug("Factor rotation model not available, using static ICs")
        except Exception as e:
            logger.warning(f"Factor rotation failed, using static ICs: {e}")
    
    # Calculate IC-proportional weights
    total_ic = sum(model_ics.values())
    ic_weights = {k: v / total_ic for k, v in model_ics.items()}
    
    # Allow user to override ML weight if desired
    if abs(ml_weight - 0.35) > 0.01:  # User specified different weight
        # Rescale: keep ML at user-specified weight, redistribute rest
        user_ml_weight = ml_weight
        other_weight = 1 - user_ml_weight
        other_total_ic = total_ic - model_ics["ml"]
        for k in ic_weights:
            if k == "ml":
                ic_weights[k] = user_ml_weight
            else:
                ic_weights[k] = (model_ics[k] / other_total_ic) * other_weight
    
    logger.info(f"Model ICs ({ic_source}):")
    for name, ic in model_ics.items():
        default_ic = default_ics.get(name, 0)
        delta = ic - default_ic
        delta_str = f" ({delta:+.3f} vs default)" if ic_source == "calibrated" and abs(delta) > 0.001 else ""
        logger.info(f"  {name}: IC = {ic:.3f}{delta_str}")
    logger.info(f"")
    logger.info(f"IC-Weighted Model Contributions:")
    for name, weight in ic_weights.items():
        logger.info(f"  {name}: {weight:.1%}")
    logger.info(f"")
    logger.info(f"Signal assignment: percentile-based (top 10%=strong_buy, top 25%=buy, bottom 10%=strong_sell, bottom 25%=sell)")
    logger.info(f"")
    
    for stock in rankings:
        # ML contribution: normalize percentile to -1 to +1
        ml_contrib = ic_weights["ml"] * ((stock.ml_percentile - 50) / 50)
        
        # Agent contributions (scores are already -1 to +1)
        sent_contrib = ic_weights["sentiment"] * stock.sentiment_score
        fund_contrib = ic_weights["fundamental"] * stock.fundamental_score
        earn_contrib = ic_weights["earnings"] * stock.earnings_score
        ins_contrib = ic_weights["insider"] * stock.insider_score
        them_contrib = ic_weights["thematic"] * stock.thematic_score
        mom_contrib = ic_weights["momentum"] * stock.momentum_score
        tone_contrib = ic_weights["filing_tone"] * stock.filing_tone_score
        call_qual_contrib = ic_weights["earnings_call_qual"] * stock.earnings_call_qual_score
        agent_contrib = sent_contrib + fund_contrib + earn_contrib + ins_contrib + them_contrib + mom_contrib + tone_contrib + call_qual_contrib
        
        stock.composite_score = ml_contrib + agent_contrib

        # Consensus confidence: agreement among non-neutral agent signals
        agent_scores = [
            stock.sentiment_score,
            stock.fundamental_score,
            stock.earnings_score,
            stock.insider_score,
            stock.thematic_score,
            stock.momentum_score,
            stock.filing_tone_score,
            stock.earnings_call_qual_score,
        ]
        dirs = [1 if s > 0.1 else -1 if s < -0.1 else 0 for s in agent_scores]
        non_neutral = [d for d in dirs if d != 0]
        if non_neutral:
            pos = sum(1 for d in non_neutral if d > 0)
            neg = sum(1 for d in non_neutral if d < 0)
            consensus = max(pos, neg) / len(non_neutral)
        else:
            consensus = 0.0
        # Blend with base 0.5 to avoid overconfidence
        stock.composite_confidence = round(0.5 + 0.5 * consensus, 3)

        # Missing-data penalties
        if stock.agent_rationales is None:
            stock.agent_rationales = {}
        missing_agents: set[str] = set()
        error_map = {
            "Fundamental": "fundamental",
            "Insider": "insider",
            "Earnings": "earnings",
            "Thematic": "thematic",
            "Momentum": "momentum",
            "Sentiment": "sentiment",
            "FilingTone": "filing_tone",
            "EarningsCallQual": "earnings_call_qual",
        }
        for err in stock.errors or []:
            for prefix, agent in error_map.items():
                if str(err).startswith(prefix):
                    missing_agents.add(agent)

        for agent in ["sentiment", "fundamental", "earnings", "insider", "thematic", "momentum", "filing_tone", "earnings_call_qual"]:
            entry = (stock.agent_rationales or {}).get(agent)
            if entry is None:
                missing_agents.add(agent)
                continue
            if isinstance(entry, dict) and entry.get("missing") is True:
                missing_agents.add(agent)

        missing_count = len(missing_agents)
        penalty_score = round(min(0.2, 0.05 * missing_count), 3)
        penalty_confidence = round(min(0.5, 0.1 * missing_count), 3)
        if penalty_score:
            stock.composite_score -= penalty_score
        if penalty_confidence:
            stock.composite_confidence = round(
                max(0.0, stock.composite_confidence - penalty_confidence),
                3,
            )

        stock.agent_rationales["missing_data"] = {
            "missing_agents": sorted(missing_agents),
            "missing_count": missing_count,
            "penalty_score": penalty_score,
            "penalty_confidence": penalty_confidence,
        }

    # ── Sector Rotation Overlay ──────────────────────────────────────
    # Apply sector tilt multipliers based on breadth-price divergence.
    # This adjusts composite scores *before* signal thresholds so that
    # stocks in sectors with leading fundamental improvement get a boost.
    try:
        from auto_researcher.models.sector_rotation_overlay import (
            SectorRotationOverlay,
        )

        overlay = SectorRotationOverlay()
        sector_tilts = overlay.compute_tilts(rankings, verbose=verbose)

        for stock in rankings:
            sector = stock.sector or ""
            tilt = sector_tilts.get(sector, 1.0)
            if tilt != 1.0:
                pre_tilt = stock.composite_score
                stock.composite_score *= tilt
                if stock.agent_rationales is None:
                    stock.agent_rationales = {}
                stock.agent_rationales["sector_overlay"] = {
                    "sector": sector,
                    "tilt": round(tilt, 4),
                    "pre_tilt_score": round(pre_tilt, 4),
                    "post_tilt_score": round(stock.composite_score, 4),
                }
    except Exception as e:
        logger.warning(f"Sector rotation overlay skipped: {e}")

    # ── Signal Assignment (Percentile-Based) ───────────────────────
    # Absolute thresholds don't work because the IC-weighted composite
    # naturally compresses scores (9 agents each weighted ~5-20%).
    # Percentile-based signals are standard in quantitative finance —
    # they reflect relative ranking within the universe.
    #
    # Percentile cuts (from top):
    #   Top  10% → strong_buy  (if composite > 0)
    #   Top  25% → buy         (if composite > 0)
    #   Bot  10% → strong_sell (if composite < 0)
    #   Bot  25% → sell        (if composite < 0)
    #   Rest     → hold
    sorted_for_signals = sorted(rankings, key=lambda x: x.composite_score, reverse=True)
    n = len(sorted_for_signals)
    for i, stock in enumerate(sorted_for_signals):
        pctl = i / max(n, 1)  # 0.0 = highest score, 1.0 = lowest
        if pctl < 0.10 and stock.composite_score > 0:
            stock.signal = "strong_buy"
        elif pctl < 0.25 and stock.composite_score > 0:
            stock.signal = "buy"
        elif pctl >= 0.90 and stock.composite_score < 0:
            stock.signal = "strong_sell"
        elif pctl >= 0.75 and stock.composite_score < 0:
            stock.signal = "sell"
        else:
            stock.signal = "hold"
    
    if verbose:
        for stock in rankings:
            logger.info(f"  {stock.ticker}: composite={stock.composite_score:+.3f} -> {stock.signal} (conf={stock.composite_confidence:.2f})")
    
    # Sort by composite score
    rankings.sort(key=lambda x: x.composite_score, reverse=True)
    
    # Update final ranks
    for i, stock in enumerate(rankings, 1):
        stock.final_rank = i
        # Attach factor rotation regime to each stock for dashboard visibility
        if regime_state is not None:
            if stock.agent_rationales is None:
                stock.agent_rationales = {}
            stock.agent_rationales["factor_regime"] = {
                "regime": regime_state.regime.value,
                "regime_score": regime_state.regime_score,
                "confidence": regime_state.regime_confidence,
                "transition_probability": regime_state.transition_probability,
                "transition_direction": regime_state.transition_direction,
                "signals": {
                    "vix_term_structure": regime_state.vix_signal,
                    "credit_spreads": regime_state.credit_signal,
                    "dispersion": regime_state.dispersion_signal,
                    "breadth": regime_state.breadth_signal,
                    "factor_momentum": regime_state.factor_momentum_signal,
                },
                "rationale": regime_state.rationale,
            }
    
    return rankings


def print_results(rankings: list[StockRanking], top_n: int = 25):
    """Print final results."""
    print("\n" + "=" * 100)
    print("  FINAL STOCK RANKINGS")
    print("=" * 100)
    print(f"  {'Rank':<5} {'Ticker':<7} {'Score':>7} {'Signal':<10} {'ML%':>5} {'Fund':>5} {'Earn':>5} {'Sent':>5} {'Ins':>5} {'Them':>5} {'Mom':>5} {'Tone':>5} {'CaQ':>5}")
    print("  " + "-" * 108)
    
    for stock in rankings[:top_n]:
        print(f"  {stock.final_rank:<5} {stock.ticker:<7} {stock.composite_score:>7.3f} "
              f"{stock.signal:<10} {stock.ml_percentile:>5.1f} "
              f"{stock.fundamental_score:>5.2f} {stock.earnings_score:>5.2f} "
              f"{stock.sentiment_score:>5.2f} {stock.insider_score:>5.2f} {stock.thematic_score:>5.2f} "
              f"{stock.momentum_score:>5.2f} {stock.filing_tone_score:>5.2f} {stock.earnings_call_qual_score:>5.2f}")
    
    print("=" * 108)
    
    # Signal breakdown
    signal_counts = {}
    for stock in rankings[:top_n]:
        signal_counts[stock.signal] = signal_counts.get(stock.signal, 0) + 1
    
    print("\n  SIGNAL BREAKDOWN (Top {}):".format(top_n))
    for signal in ["strong_buy", "buy", "hold", "sell", "strong_sell"]:
        count = signal_counts.get(signal, 0)
        bar = "█" * count
        print(f"  {signal:<12} {count:>3} {bar}")
    
    print()


def generate_report(
    universe: str,
    ml_results_file: Path,
    final_results_file: Path,
    output_path: Path,
    review_summary=None,
) -> None:
    """Generate a markdown report from JSON outputs."""
    if not ml_results_file.exists() or not final_results_file.exists():
        logger.warning("Report not generated: missing input files")
        return

    with ml_results_file.open() as f:
        ml = json.load(f)
    with final_results_file.open() as f:
        final = json.load(f)

    ml_ic = ml[0].get("ml_ic") if ml else None
    ml_sorted = sorted(ml, key=lambda x: x.get("ml_rank", 0))
    final_sorted = sorted(final, key=lambda x: x.get("final_rank", 0))

    date_tag = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# {universe.upper()} Ranking Report ({date_tag})")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Universe: {universe.upper()}")
    lines.append(f"- ML top count: {len(ml_sorted)}")
    lines.append(f"- Final top count: {len(final_sorted)}")
    if ml_ic is not None:
        lines.append(f"- ML OOS IC: {ml_ic:.4f} (from {ml_results_file.as_posix()})")
    else:
        lines.append(f"- ML OOS IC: n/a (from {ml_results_file.as_posix()})")
    lines.append("")
    lines.append("## 1) Initial ML Top Rankings")
    lines.append(f"Source: [{ml_results_file.as_posix()}]({ml_results_file.as_posix()})")
    lines.append("")
    lines.append("| ML Rank | Ticker | ML Score | ML Percentile |")
    lines.append("|---:|---|---:|---:|")
    for row in ml_sorted:
        lines.append(
            f"| {row['ml_rank']} | {row['ticker']} | {row['ml_score']:.4f} | {row['ml_percentile']:.1f} |"
        )

    lines.append("")
    lines.append("## 2) Additional Model Outputs")
    lines.append(f"Source: [{final_results_file.as_posix()}]({final_results_file.as_posix()})")
    lines.append("")
    lines.append("| Ticker | Sentiment | Fundamental | Earnings | Insider | Thematic | Momentum | FilingTone | CallQual |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in final_sorted:
        lines.append(
            f"| {row['ticker']} | {row['sentiment_score']:.3f} | {row['fundamental_score']:.3f} | {row['earnings_score']:.3f} | {row['insider_score']:.3f} | {row['thematic_score']:.3f} | {row.get('momentum_score', 0.0):.3f} | {row.get('filing_tone_score', 0.0):.3f} | {row.get('earnings_call_qual_score', 0.0):.3f} |"
        )

    lines.append("")
    lines.append("## 3) Final Rankings (IC-Weighted Composite)")
    lines.append(f"Source: [{final_results_file.as_posix()}]({final_results_file.as_posix()})")
    lines.append("")
    lines.append("| Final Rank | Ticker | Composite Score | Signal |")
    lines.append("|---:|---|---:|---|")
    for row in final_sorted:
        lines.append(
            f"| {row['final_rank']} | {row['ticker']} | {row['composite_score']:.3f} | {row['signal']} |"
        )

    lines.append("")
    lines.append("## 4) Context Bands (Top 10)")
    lines.append(f"Source: [{final_results_file.as_posix()}]({final_results_file.as_posix()})")
    lines.append("")
    lines.append("Bands computed across the ML-top universe (n=10). Values shown as band (percentile).")
    lines.append("")
    lines.append("| Ticker | Sentiment | Fundamental | Earnings | Insider | Thematic |")
    lines.append("|---|---|---|---|---|---|")
    for row in final_sorted:
        r = row.get("agent_rationales", {})
        def band(agent: str) -> str:
            ctx = (r.get(agent) or {}).get("context") or {}
            if not ctx:
                return "n/a"
            return f"{ctx.get('band')} ({ctx.get('percentile')})"
        lines.append(
            f"| {row['ticker']} | {band('sentiment')} | {band('fundamental')} | {band('earnings')} | {band('insider')} | {band('thematic')} |"
        )

    lines.append("")
    lines.append("## 5) Conflict Resolution (Top 10)")
    lines.append(f"Source: [{final_results_file.as_posix()}]({final_results_file.as_posix()})")
    lines.append("")
    for row in final_sorted:
        cr = (row.get("agent_rationales") or {}).get("conflict_resolution") or {}
        summary = cr.get("summary") or "No material disagreement among agent signals."
        lines.append(f"- {row['ticker']}: {summary}")

    # LLM Red-Team Review section
    if review_summary is not None:
        lines.append("")
        lines.append(review_summary.to_markdown())

    output_path.write_text("\n".join(lines))
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Memory-optimized stock ranking")
    parser.add_argument("--universe", default="sp100", choices=["sp500", "sp100", "large_cap", "core_tech"])
    parser.add_argument("--ml-top", type=int, default=25, help="Stocks to pass from ML screening")
    parser.add_argument("--final-top", type=int, default=10, help="Final top N to display")
    parser.add_argument("--ml-weight", type=float, default=0.35, help="ML score weight in composite")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for memory clearing")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML, load from previous results")
    parser.add_argument("--skip-agents", action="store_true", help="Skip agents, load from previous results")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose agent logging")
    parser.add_argument("--llm-review", action="store_true", help="Enable LLM red-team review of top/bottom stocks")
    parser.add_argument("--review-model", type=str, default="o3", help="LLM model for review (default: o3). Options: o3, o4-mini, gpt-5.2, gpt-4.1, claude-opus-4, claude-sonnet-4, gemini-2.5-pro")
    parser.add_argument("--review-top-n", type=int, default=5, help="Review top-N buys + sells (default: 5)")
    parser.add_argument("--reflexion-rounds", type=int, default=1, help="Reflexion self-critique rounds (default: 1)")
    parser.add_argument("--deep-research", action="store_true", help="Run deep web research on top-ranked stocks via GPT-Researcher")
    parser.add_argument("--deep-research-tickers", type=str, default="", help="Comma-separated tickers to deep-research (default: auto top-N)")
    parser.add_argument("--deep-research-top-n", type=int, default=3, help="Auto-research top-N ranked stocks (default: 3)")
    parser.add_argument("--deep-research-type", type=str, default="research_report", choices=["research_report", "deep"], help="Research mode: research_report (fast) or deep (recursive, ~5min)")
    
    args = parser.parse_args()
    
    # File paths for intermediate results
    ml_results_file = RESULTS_DIR / f"ml_screening_{args.universe}_{args.ml_top}.json"
    agent_results_file = RESULTS_DIR / f"agent_analysis_{args.universe}_{args.ml_top}.json"
    final_results_file = args.output or RESULTS_DIR / f"final_ranking_{args.universe}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    print("\n" + "=" * 80)
    print("  MEMORY-OPTIMIZED STOCK RANKING PIPELINE")
    print("=" * 80)
    print(f"  Universe: {args.universe}")
    print(f"  ML Top: {args.ml_top}")
    print(f"  Final Top: {args.final_top}")
    print(f"  ML Weight: {args.ml_weight}")
    print(f"  Verbose: {args.verbose}")
    print("=" * 80 + "\n")
    
    # Stage 1: ML Screening
    ml_ic = 0.15  # Default
    if args.skip_ml and ml_results_file.exists():
        logger.info(f"Loading ML results from {ml_results_file}")
        rankings = load_ml_results(ml_results_file)
        # Try to get IC from first ranking
        if rankings and hasattr(rankings[0], 'ml_ic'):
            ml_ic = rankings[0].ml_ic
    else:
        rankings, ml_ic = run_ml_screening(universe=args.universe, top_k=args.ml_top)
        save_results(rankings, ml_results_file)
    
    clear_memory()
    
    # Stage 2: Agent Analysis
    if args.skip_agents and agent_results_file.exists():
        logger.info(f"Loading agent results from {agent_results_file}")
        rankings = load_ml_results(agent_results_file)
    else:
        rankings = run_agent_analysis(rankings, batch_size=args.batch_size, verbose=args.verbose)
        add_context_bands(rankings)
        add_conflict_resolution(rankings)
        add_evidence_budget(rankings)
        add_consistency_checks(rankings)
        add_lightweight_cross_validation(rankings)
        add_risk_qualifiers(rankings)
        apply_freshness_gates(rankings)
        save_results(rankings, agent_results_file)

    if args.skip_agents and agent_results_file.exists():
        add_context_bands(rankings)
        add_conflict_resolution(rankings)
        add_evidence_budget(rankings)
        add_consistency_checks(rankings)
        add_lightweight_cross_validation(rankings)
        add_risk_qualifiers(rankings)
        apply_freshness_gates(rankings)
    
    clear_memory()
    
    # Stage 3: Composite Scoring (IC-weighted)
    rankings = compute_composite_scores(rankings, ml_weight=args.ml_weight, verbose=args.verbose, ml_ic=ml_ic)
    
    # Save final results
    save_results(rankings, Path(final_results_file))

    # Stage 4 (optional): LLM Red-Team Review
    review_summary = None
    if args.llm_review:
        try:
            from auto_researcher.agents.llm_review_agent import (
                LLMReviewAgent,
                LLMReviewConfig,
            )
            print("\n" + "=" * 80)
            print("  STAGE 4: LLM RED-TEAM REVIEW")
            print("=" * 80 + "\n")
            review_config = LLMReviewConfig(
                model=args.review_model,
                top_n_review=args.review_top_n,
                reflexion_rounds=args.reflexion_rounds,
            )
            review_agent = LLMReviewAgent(config=review_config)
            review_summary = review_agent.review_rankings(
                rankings, verbose=args.verbose
            )
            # Inject reviews into agent_rationales (does NOT change scores)
            review_agent.inject_reviews_into_rankings(rankings, review_summary)
            # Re-save with reviews included
            save_results(rankings, Path(final_results_file))
            logger.info(
                "LLM review complete: %d stocks reviewed, %d calls",
                len(review_summary.reviews),
                review_summary.total_llm_calls,
            )
            del review_agent
            gc.collect()
        except ImportError:
            logger.warning("litellm not installed — skipping LLM review")
        except Exception as e:
            logger.error("LLM review failed: %s", e)

    # Stage 5 (optional): Deep Research via GPT-Researcher
    if args.deep_research:
        try:
            import asyncio
            from auto_researcher.agents.deep_research_agent import (
                DeepResearchAgent,
                DeepResearchConfig,
            )
            print("\n" + "=" * 80)
            print("  STAGE 5: DEEP WEB RESEARCH (GPT-Researcher)")
            print("=" * 80 + "\n")

            dr_config = DeepResearchConfig(
                report_type=args.deep_research_type,
                verbose=args.verbose,
            )
            dr_agent = DeepResearchAgent(config=dr_config)

            # Determine target tickers
            if args.deep_research_tickers:
                target_tickers = [t.strip().upper() for t in args.deep_research_tickers.split(",") if t.strip()]
            else:
                sorted_rankings = sorted(rankings, key=lambda r: r.composite_score, reverse=True)
                target_tickers = [r.ticker for r in sorted_rankings[:args.deep_research_top_n]]

            from dataclasses import asdict
            research_stocks = []
            for t in target_tickers:
                stock = next((asdict(r) for r in rankings if r.ticker == t), {"ticker": t})
                research_stocks.append(stock)

            print(f"  Researching: {', '.join(target_tickers)}")
            dr_summary = asyncio.run(dr_agent.research_batch(research_stocks, max_concurrent=2))

            # Save reports
            for result in dr_summary.results:
                if result.report and not result.error:
                    report_path = REPORT_DIR / f"deep_research_{result.ticker}_{datetime.now().strftime('%Y%m%d')}.md"
                    report_path.write_text(result.report, encoding="utf-8")
                    print(f"  ✅ {result.ticker}: {len(result.sources)} sources, ${result.costs:.2f}")
                elif result.error:
                    print(f"  ❌ {result.ticker}: {result.error}")

            logger.info(
                "Deep research complete: %d stocks, cost=$%.2f",
                len(dr_summary.results),
                dr_summary.total_costs,
            )
            del dr_agent
            gc.collect()
        except ImportError:
            logger.warning("gpt-researcher not installed — skipping deep research")
        except Exception as e:
            logger.error("Deep research failed: %s", e)

    # Generate markdown report in repo root
    report_file = REPORT_DIR / f"report_{args.universe}_{datetime.now().strftime('%Y%m%d')}.md"
    generate_report(
        universe=args.universe,
        ml_results_file=Path(ml_results_file),
        final_results_file=Path(final_results_file),
        output_path=report_file,
        review_summary=review_summary,
    )
    
    # Print results
    print_results(rankings, top_n=args.final_top)
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
