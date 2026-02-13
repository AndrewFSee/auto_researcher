"""
LLM Review & Red-Team Agent.

Post-ranking review layer that uses a frontier LLM to:
1. **Conflict Narrator** — For stocks where agents strongly disagree,
   explain which signal is likely more informative in context.
2. **Red-Teamer** — Challenge the top BUY and SELL theses with
   adversarial critique using a reflexion loop.
3. **Risk Flagger** — Surface hidden risks not captured by quantitative agents.

Design Principles:
- Scores are NEVER changed — this is a narrative/review layer only.
- Adds ``llm_review`` key to ``agent_rationales`` with structured output.
- Uses litellm for model routing (same pattern as SentimentAgent).
- Fully optional — pipeline works identically without it.
- Costs are controlled: only reviews top-N BUY + SELL stocks.

Usage (standalone):
    from auto_researcher.agents.llm_review_agent import LLMReviewAgent
    agent = LLMReviewAgent(model="gpt-4o")
    reviews = agent.review_rankings(rankings)

Usage (pipeline):
    python scripts/run_ranking_low_memory.py --universe sp100 --llm-review

References:
    - Shinn et al. 2023 — Reflexion: Language Agents with Verbal Reinforcement Learning
    - Madaan et al. 2023 — Self-Refine: Iterative Refinement with Self-Feedback
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy import litellm
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    litellm = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMReviewConfig:
    """Configuration for the LLM review agent."""

    # Model settings
    model: str = "o3"  # Frontier reasoning model for adversarial review
    temperature: float = 0.3  # Moderate creativity for adversarial critique
    max_tokens: int = 1500  # Per-stock review budget
    timeout: int = 60  # API timeout seconds

    # Review scope
    top_n_review: int = 5  # Review top-N buys + top-N sells
    conflict_threshold: float = 0.4  # Min score spread to flag as conflict

    # Reflexion settings
    reflexion_rounds: int = 1  # Additional self-critique rounds (0 = none)

    # Cost controls
    max_total_calls: int = 15  # Hard cap on LLM calls per run
    skip_hold_signals: bool = True  # Don't review holds


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an adversarial equity research analyst conducting a red-team \
review of algorithmically-generated stock rankings. Your job is to CHALLENGE each thesis, \
not confirm it.

Rules:
1. Be specific — cite the actual scores and data provided, not generalities.
2. Identify the #1 risk the algorithm may have missed.
3. If agents disagree, explain which signal is likely more informative and why.
4. Do NOT produce a new score or signal — you are reviewing, not ranking.
5. Keep each review to 3-5 concise bullet points.
6. End with a confidence assessment: HIGH / MEDIUM / LOW confidence in the signal."""

STOCK_REVIEW_TEMPLATE = """Review this {signal} signal for {ticker} ({company}):

COMPOSITE SCORE: {composite:.3f} (rank #{rank})
SECTOR: {sector}

AGENT SCORES (-1 bearish … +1 bullish):
  ML Model:      {ml_pct:.0f}th percentile (normalized: {ml_norm:+.3f})
  Sentiment:     {sentiment:+.3f}
  Fundamental:   {fundamental:+.3f}
  Earnings/PEAD: {earnings:+.3f}
  Insider:       {insider:+.3f}
  Thematic:      {thematic:+.3f}
  Momentum:      {momentum:+.3f}
  Filing Tone:   {filing_tone:+.3f}
  Call Quality:  {call_qual:+.3f}

CONFLICT SUMMARY: {conflict_summary}

KEY FACTORS: {key_factors}

EVIDENCE:
{evidence_block}

Challenge this {signal} thesis. What could go wrong? Which agent signal is \
most/least reliable here? What risk is being underweighted?"""

REFLEXION_TEMPLATE = """You previously wrote this review of {ticker} ({signal}):

---
{previous_review}
---

Now critique your OWN review:
1. Did you identify a genuinely novel risk, or just restate what the scores already show?
2. Is your confidence assessment well-calibrated given the evidence?
3. Sharpen or correct your review. Output the FINAL improved version only (3-5 bullets + confidence)."""


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class StockReview:
    """LLM review output for a single stock."""

    ticker: str
    signal: str
    composite_score: float
    review_text: str
    confidence_in_signal: str  # HIGH / MEDIUM / LOW
    conflict_agents: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    reflexion_applied: bool = False
    model_used: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "composite_score": self.composite_score,
            "review_text": self.review_text,
            "confidence_in_signal": self.confidence_in_signal,
            "conflict_agents": self.conflict_agents,
            "risk_flags": self.risk_flags,
            "reflexion_applied": self.reflexion_applied,
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ReviewSummary:
    """Aggregate review output for the full ranking set."""

    reviews: list[StockReview] = field(default_factory=list)
    total_llm_calls: int = 0
    total_latency_ms: int = 0
    model_used: str = ""
    skipped_tickers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviews": [r.to_dict() for r in self.reviews],
            "total_llm_calls": self.total_llm_calls,
            "total_latency_ms": self.total_latency_ms,
            "model_used": self.model_used,
            "skipped_tickers": self.skipped_tickers,
        }

    def to_markdown(self) -> str:
        """Generate a markdown section for the report."""
        lines = [
            "## 6) LLM Red-Team Review",
            "",
            f"Model: `{self.model_used}` | "
            f"Stocks reviewed: {len(self.reviews)} | "
            f"LLM calls: {self.total_llm_calls} | "
            f"Latency: {self.total_latency_ms / 1000:.1f}s",
            "",
        ]

        # Group by signal type
        buys = [r for r in self.reviews if r.signal in ("strong_buy", "buy")]
        sells = [r for r in self.reviews if r.signal in ("strong_sell", "sell")]
        holds = [r for r in self.reviews if r.signal == "hold"]

        for group_label, group in [
            ("Bullish Signals", buys),
            ("Bearish Signals", sells),
            ("Holds (conflict review)", holds),
        ]:
            if not group:
                continue
            lines.append(f"### {group_label}")
            lines.append("")
            for review in group:
                conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(
                    review.confidence_in_signal, "⚪"
                )
                lines.append(
                    f"**{review.ticker}** ({review.signal}, "
                    f"composite={review.composite_score:+.3f}) "
                    f"— Signal confidence: {conf_emoji} {review.confidence_in_signal}"
                )
                if review.reflexion_applied:
                    lines.append("*Reflexion-refined*")
                lines.append("")
                lines.append(review.review_text)
                lines.append("")
                if review.risk_flags:
                    lines.append(
                        f"**Risk flags:** {', '.join(review.risk_flags)}"
                    )
                    lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LLMReviewAgent:
    """
    Post-ranking LLM review agent with adversarial reflexion.

    Consumes scored ``StockRanking`` objects (or dicts) and produces
    narrative reviews that challenge each thesis without altering scores.
    """

    def __init__(
        self,
        model: str | None = None,
        config: LLMReviewConfig | None = None,
    ) -> None:
        self.config = config or LLMReviewConfig()
        if model is not None:
            self.config.model = model

        if not HAS_LITELLM:
            raise ImportError(
                "litellm is required for LLMReviewAgent. "
                "Install with: pip install litellm"
            )

        self._call_count = 0
        logger.info(
            "LLMReviewAgent initialized (model=%s, reflexion_rounds=%d, top_n=%d)",
            self.config.model,
            self.config.reflexion_rounds,
            self.config.top_n_review,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review_rankings(
        self,
        rankings: list,
        verbose: bool = False,
    ) -> ReviewSummary:
        """
        Review a list of ranked stocks.

        Parameters
        ----------
        rankings : list
            List of StockRanking objects or dicts with the standard fields.
        verbose : bool
            If True, log each review as it completes.

        Returns
        -------
        ReviewSummary
            Aggregate review with per-stock narratives.
        """
        self._call_count = 0
        summary = ReviewSummary(model_used=self.config.model)

        stocks = self._normalize_to_dicts(rankings)
        to_review = self._select_stocks_for_review(stocks)

        logger.info(
            "LLM review: %d stocks selected for review out of %d total",
            len(to_review),
            len(stocks),
        )

        reviewed_tickers = set()
        for stock in to_review:
            if self._call_count >= self.config.max_total_calls:
                logger.warning(
                    "LLM call cap reached (%d). Stopping review.",
                    self.config.max_total_calls,
                )
                break

            ticker = stock["ticker"]
            reviewed_tickers.add(ticker)

            try:
                review = self._review_single_stock(stock, verbose=verbose)
                summary.reviews.append(review)
                summary.total_latency_ms += review.latency_ms
            except Exception as e:
                logger.error("LLM review failed for %s: %s", ticker, e)
                summary.skipped_tickers.append(ticker)

        # Track skipped tickers
        all_selected = {s["ticker"] for s in to_review}
        summary.skipped_tickers.extend(
            sorted(all_selected - reviewed_tickers - set(summary.skipped_tickers))
        )
        summary.total_llm_calls = self._call_count

        logger.info(
            "LLM review complete: %d reviews, %d calls, %.1fs total latency",
            len(summary.reviews),
            summary.total_llm_calls,
            summary.total_latency_ms / 1000,
        )

        return summary

    def inject_reviews_into_rankings(
        self,
        rankings: list,
        summary: ReviewSummary,
    ) -> None:
        """
        Inject LLM review data into agent_rationales of each ranking.

        Mutates rankings in-place. Does NOT change scores or signals.
        """
        review_map = {r.ticker: r for r in summary.reviews}

        for stock in rankings:
            ticker = self._get_ticker(stock)
            if ticker not in review_map:
                continue

            review = review_map[ticker]
            rationales = self._get_rationales(stock)
            rationales["llm_review"] = review.to_dict()

    # ------------------------------------------------------------------
    # Stock selection
    # ------------------------------------------------------------------

    def _select_stocks_for_review(
        self,
        stocks: list[dict],
    ) -> list[dict]:
        """Select stocks that warrant LLM review."""
        n = self.config.top_n_review

        # Sort by composite
        sorted_stocks = sorted(
            stocks,
            key=lambda s: s.get("composite_score", 0),
            reverse=True,
        )

        selected: list[dict] = []
        seen: set[str] = set()

        # Top-N buys
        for stock in sorted_stocks:
            if len(selected) >= n:
                break
            sig = stock.get("signal", "hold")
            if sig in ("strong_buy", "buy"):
                selected.append(stock)
                seen.add(stock["ticker"])

        # Top-N sells (from bottom)
        for stock in reversed(sorted_stocks):
            if sum(1 for s in selected if s.get("signal") in ("strong_sell", "sell")) >= n:
                break
            sig = stock.get("signal", "hold")
            if sig in ("strong_sell", "sell") and stock["ticker"] not in seen:
                selected.append(stock)
                seen.add(stock["ticker"])

        # High-conflict stocks (regardless of signal)
        for stock in sorted_stocks:
            if stock["ticker"] in seen:
                continue
            if self._is_high_conflict(stock):
                selected.append(stock)
                seen.add(stock["ticker"])
                if len(selected) >= n * 3:  # Hard cap
                    break

        # Fallback: if we haven't selected enough stocks (e.g. all HOLD),
        # fill remaining slots with top-N by composite score
        if len(selected) < n:
            for stock in sorted_stocks:
                if stock["ticker"] in seen:
                    continue
                selected.append(stock)
                seen.add(stock["ticker"])
                if len(selected) >= n:
                    break

        return selected

    def _is_high_conflict(self, stock: dict) -> bool:
        """Check if agents strongly disagree on this stock."""
        score_fields = [
            "sentiment_score",
            "fundamental_score",
            "earnings_score",
            "insider_score",
            "thematic_score",
            "momentum_score",
            "filing_tone_score",
            "earnings_call_qual_score",
        ]

        scores = []
        for f in score_fields:
            val = stock.get(f, 0.0)
            if val is not None and abs(val) > 0.05:  # Non-trivial
                scores.append(val)

        if len(scores) < 2:
            return False

        has_positive = any(s > self.config.conflict_threshold for s in scores)
        has_negative = any(s < -self.config.conflict_threshold for s in scores)
        return has_positive and has_negative

    # ------------------------------------------------------------------
    # Single-stock review
    # ------------------------------------------------------------------

    def _review_single_stock(
        self,
        stock: dict,
        verbose: bool = False,
    ) -> StockReview:
        """Generate an LLM review for one stock."""
        ticker = stock["ticker"]
        signal = stock.get("signal", "hold")

        # Build the review prompt
        prompt = self._build_review_prompt(stock)

        # Initial review
        t0 = time.time()
        review_text = self._call_llm(prompt)
        latency = int((time.time() - t0) * 1000)

        # Reflexion loop
        reflexion_applied = False
        for _round in range(self.config.reflexion_rounds):
            if self._call_count >= self.config.max_total_calls:
                break
            reflexion_prompt = REFLEXION_TEMPLATE.format(
                ticker=ticker,
                signal=signal,
                previous_review=review_text,
            )
            t1 = time.time()
            review_text = self._call_llm(reflexion_prompt)
            latency += int((time.time() - t1) * 1000)
            reflexion_applied = True

        # Parse confidence and risk flags
        confidence = self._extract_confidence(review_text)
        risk_flags = self._extract_risk_flags(stock, review_text, confidence)
        conflict_agents = self._get_conflict_agents(stock)

        review = StockReview(
            ticker=ticker,
            signal=signal,
            composite_score=stock.get("composite_score", 0.0),
            review_text=review_text.strip(),
            confidence_in_signal=confidence,
            conflict_agents=conflict_agents,
            risk_flags=risk_flags,
            reflexion_applied=reflexion_applied,
            model_used=self.config.model,
            latency_ms=latency,
        )

        if verbose:
            logger.info(
                "Reviewed %s: signal=%s confidence=%s (%dms)",
                ticker,
                signal,
                confidence,
                latency,
            )

        return review

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_review_prompt(self, stock: dict) -> str:
        """Build the full review prompt for a stock."""
        rationales = stock.get("agent_rationales", {})

        # Gather evidence from all agents
        evidence_lines = []
        for agent in [
            "fundamental",
            "earnings",
            "insider",
            "thematic",
            "sentiment",
        ]:
            agent_data = rationales.get(agent, {})
            evidence = agent_data.get("evidence", [])
            if evidence:
                evidence_lines.append(f"  [{agent.upper()}]")
                for item in evidence:
                    evidence_lines.append(f"    - {item}")

        # Filing tone rationale
        filing_data = rationales.get("filing_tone", {})
        if filing_data:
            evidence_lines.append("  [FILING TONE]")
            for key in ("tone_label", "lm_positive_pct", "uncertainty_pct"):
                if key in filing_data:
                    evidence_lines.append(f"    - {key}: {filing_data[key]}")

        # Earnings call qualitative rationale
        call_data = rationales.get("earnings_call_qual", {})
        if call_data:
            evidence_lines.append("  [EARNINGS CALL QUALITY]")
            for key in (
                "mgmt_confidence",
                "analyst_probing",
                "guidance_clarity",
                "peer_tone_delta",
            ):
                if key in call_data:
                    evidence_lines.append(f"    - {key}: {call_data[key]}")

        evidence_block = (
            "\n".join(evidence_lines) if evidence_lines else "  No detailed evidence available."
        )

        # Conflict summary
        conflict = rationales.get("conflict_resolution", {})
        conflict_summary = conflict.get(
            "summary", "No material disagreement among agent signals."
        )

        # Key factors
        key_factors = stock.get("key_factors", [])
        key_factors_str = (
            "; ".join(key_factors[:5]) if key_factors else "None recorded"
        )

        ml_norm = (stock.get("ml_percentile", 50) - 50) / 50

        return STOCK_REVIEW_TEMPLATE.format(
            ticker=stock["ticker"],
            company=stock.get("company_name", "Unknown"),
            signal=stock.get("signal", "hold").upper(),
            composite=stock.get("composite_score", 0.0),
            rank=stock.get("final_rank", 0),
            sector=stock.get("sector", "Unknown"),
            ml_pct=stock.get("ml_percentile", 50),
            ml_norm=ml_norm,
            sentiment=stock.get("sentiment_score", 0.0),
            fundamental=stock.get("fundamental_score", 0.0),
            earnings=stock.get("earnings_score", 0.0),
            insider=stock.get("insider_score", 0.0),
            thematic=stock.get("thematic_score", 0.0),
            momentum=stock.get("momentum_score", 0.0),
            filing_tone=stock.get("filing_tone_score", 0.0),
            call_qual=stock.get("earnings_call_qual_score", 0.0),
            conflict_summary=conflict_summary,
            key_factors=key_factors_str,
            evidence_block=evidence_block,
        )

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Make a single LLM call via litellm."""
        self._call_count += 1

        try:
            response = litellm.completion(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Post-processing / extraction
    # ------------------------------------------------------------------

    def _extract_confidence(self, review_text: str) -> str:
        """Extract confidence level from review text."""
        text_upper = review_text.upper()

        # Look for explicit confidence statements
        for level in ("HIGH", "MEDIUM", "LOW"):
            # Match patterns like "confidence: HIGH", "HIGH confidence"
            if f"CONFIDENCE: {level}" in text_upper or f"CONFIDENCE IN THE SIGNAL: {level}" in text_upper:
                return level
            if f"{level} CONFIDENCE" in text_upper:
                return level

        # Fallback: if "low" appears near "confidence", treat as LOW
        if "LOW" in text_upper and "CONFIDEN" in text_upper:
            return "LOW"
        if "HIGH" in text_upper and "CONFIDEN" in text_upper:
            return "HIGH"

        return "MEDIUM"  # Default

    def _extract_risk_flags(
        self,
        stock: dict,
        review_text: str,
        confidence: str,
    ) -> list[str]:
        """Generate risk flags based on scores and review."""
        flags = []

        # Low confidence from LLM = flag
        if confidence == "LOW":
            flags.append("llm_low_confidence")

        # High conflict
        if self._is_high_conflict(stock):
            flags.append("agent_disagreement")

        # Extreme signal with sparse data
        signal = stock.get("signal", "hold")
        missing_count = sum(
            1
            for f in [
                "sentiment_score",
                "fundamental_score",
                "earnings_score",
                "insider_score",
                "thematic_score",
                "momentum_score",
            ]
            if abs(stock.get(f, 0.0)) < 0.01
        )
        if signal in ("strong_buy", "strong_sell") and missing_count >= 3:
            flags.append("thin_evidence_for_strong_signal")

        # Sentiment-driven (sentiment dominates composite)
        if abs(stock.get("sentiment_score", 0.0)) > 0.7 and abs(
            stock.get("fundamental_score", 0.0)
        ) < 0.2:
            flags.append("sentiment_dominated")

        return flags

    def _get_conflict_agents(self, stock: dict) -> list[str]:
        """Return list of agents in conflict."""
        rationales = stock.get("agent_rationales", {})
        conflict = rationales.get("conflict_resolution", {})
        bullish = conflict.get("bullish_agents", [])
        bearish = conflict.get("bearish_agents", [])
        return bullish + bearish if (bullish and bearish) else []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_to_dicts(self, rankings: list) -> list[dict]:
        """Convert StockRanking objects or dicts to dicts."""
        result = []
        for item in rankings:
            if isinstance(item, dict):
                result.append(item)
            elif hasattr(item, "__dataclass_fields__"):
                # Dataclass → dict
                from dataclasses import asdict

                result.append(asdict(item))
            else:
                # Try attribute access
                result.append(
                    {
                        "ticker": getattr(item, "ticker", "???"),
                        "composite_score": getattr(item, "composite_score", 0.0),
                        "signal": getattr(item, "signal", "hold"),
                        "ml_percentile": getattr(item, "ml_percentile", 50.0),
                        "sentiment_score": getattr(item, "sentiment_score", 0.0),
                        "fundamental_score": getattr(
                            item, "fundamental_score", 0.0
                        ),
                        "earnings_score": getattr(item, "earnings_score", 0.0),
                        "insider_score": getattr(item, "insider_score", 0.0),
                        "thematic_score": getattr(item, "thematic_score", 0.0),
                        "momentum_score": getattr(item, "momentum_score", 0.0),
                        "filing_tone_score": getattr(
                            item, "filing_tone_score", 0.0
                        ),
                        "earnings_call_qual_score": getattr(
                            item, "earnings_call_qual_score", 0.0
                        ),
                        "company_name": getattr(item, "company_name", ""),
                        "sector": getattr(item, "sector", ""),
                        "key_factors": getattr(item, "key_factors", []),
                        "agent_rationales": getattr(
                            item, "agent_rationales", {}
                        ),
                    }
                )
        return result

    @staticmethod
    def _get_ticker(stock) -> str:
        if isinstance(stock, dict):
            return stock.get("ticker", "")
        return getattr(stock, "ticker", "")

    @staticmethod
    def _get_rationales(stock) -> dict:
        if isinstance(stock, dict):
            if "agent_rationales" not in stock:
                stock["agent_rationales"] = {}
            return stock["agent_rationales"]
        if not hasattr(stock, "agent_rationales") or stock.agent_rationales is None:
            stock.agent_rationales = {}
        return stock.agent_rationales


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    """Standalone CLI for testing the review agent."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="LLM Review Agent (standalone)")
    parser.add_argument(
        "input_json",
        help="Path to final_ranking JSON file",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model (default: gpt-4o)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top-N buys + sells to review",
    )
    parser.add_argument(
        "--reflexion-rounds",
        type=int,
        default=1,
        help="Reflexion self-critique rounds (0=none)",
    )
    parser.add_argument(
        "--output",
        help="Output markdown file (default: stdout)",
    )
    args = parser.parse_args()

    from pathlib import Path

    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    with input_path.open() as f:
        rankings = json.load(f)

    config = LLMReviewConfig(
        model=args.model,
        top_n_review=args.top_n,
        reflexion_rounds=args.reflexion_rounds,
    )
    agent = LLMReviewAgent(config=config)
    summary = agent.review_rankings(rankings, verbose=True)

    md = summary.to_markdown()
    if args.output:
        Path(args.output).write_text(md)
        print(f"Review saved to {args.output}")
    else:
        print(md)

    print(f"\nStats: {summary.total_llm_calls} calls, "
          f"{summary.total_latency_ms / 1000:.1f}s latency")


if __name__ == "__main__":
    _cli()
