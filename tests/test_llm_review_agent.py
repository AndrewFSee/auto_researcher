"""
Tests for LLM Review Agent.

Tests cover:
- Stock selection logic (top buys, top sells, high-conflict)
- Conflict detection
- Prompt building
- Confidence extraction from review text
- Risk flag generation
- Review injection into rankings
- ReviewSummary markdown output
- Normalization of StockRanking objects to dicts
- CLI argument support in pipeline

All tests run WITHOUT LLM calls (mocked).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_researcher.agents.llm_review_agent import (
    LLMReviewAgent,
    LLMReviewConfig,
    ReviewSummary,
    StockReview,
    SYSTEM_PROMPT,
    STOCK_REVIEW_TEMPLATE,
    REFLEXION_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stock(
    ticker: str,
    composite: float = 0.0,
    signal: str = "hold",
    sentiment: float = 0.0,
    fundamental: float = 0.0,
    earnings: float = 0.0,
    insider: float = 0.0,
    thematic: float = 0.0,
    momentum: float = 0.0,
    filing_tone: float = 0.0,
    call_qual: float = 0.0,
    ml_percentile: float = 50.0,
    sector: str = "Technology",
    rationales: dict | None = None,
) -> dict:
    """Create a stock dict for testing."""
    return {
        "ticker": ticker,
        "composite_score": composite,
        "signal": signal,
        "ml_percentile": ml_percentile,
        "ml_score": 0.5,
        "sentiment_score": sentiment,
        "fundamental_score": fundamental,
        "earnings_score": earnings,
        "insider_score": insider,
        "thematic_score": thematic,
        "momentum_score": momentum,
        "filing_tone_score": filing_tone,
        "earnings_call_qual_score": call_qual,
        "company_name": f"{ticker} Inc",
        "sector": sector,
        "key_factors": ["factor1", "factor2"],
        "agent_rationales": rationales or {
            "conflict_resolution": {
                "summary": "No material disagreement.",
                "bullish_agents": [],
                "bearish_agents": [],
            }
        },
        "final_rank": 1,
    }


@pytest.fixture
def sample_rankings():
    """A set of rankings with mixed signals and conflicts."""
    return [
        _make_stock("AAPL", composite=0.8, signal="strong_buy", sentiment=0.9, fundamental=0.7, earnings=0.6),
        _make_stock("NVDA", composite=0.5, signal="buy", sentiment=0.6, fundamental=0.4, insider=-0.5,
                     rationales={"conflict_resolution": {
                         "summary": "Disagreement: bullish from sentiment; bearish from insider.",
                         "bullish_agents": ["sentiment"],
                         "bearish_agents": ["insider"],
                     }}),
        _make_stock("MSFT", composite=0.2, signal="hold", sentiment=0.1, fundamental=0.2),
        _make_stock("GOOG", composite=0.1, signal="hold", sentiment=-0.1, fundamental=0.3),
        _make_stock("META", composite=-0.1, signal="hold", sentiment=0.0, fundamental=-0.1),
        _make_stock("TSLA", composite=-0.5, signal="sell", sentiment=-0.7, fundamental=-0.3),
        _make_stock("GME", composite=-0.8, signal="strong_sell", sentiment=-0.9, fundamental=-0.6, insider=0.7,
                     rationales={"conflict_resolution": {
                         "summary": "Disagreement: bullish from insider; bearish from sentiment.",
                         "bullish_agents": ["insider"],
                         "bearish_agents": ["sentiment"],
                     }}),
    ]


@pytest.fixture
def mock_litellm():
    """Patch litellm.completion to return a canned review."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        "- The bullish thesis relies heavily on sentiment, which is noisy short-term.\n"
        "- Fundamental score is solid but doesn't account for rising rate environment.\n"
        "- Insider activity is neutral — no conviction signal from management.\n"
        "- Key risk: Multiple compression if earnings growth decelerates.\n"
        "- Signal confidence: HIGH confidence in the signal."
    )
    with patch("auto_researcher.agents.llm_review_agent.litellm") as mock_ll:
        mock_ll.completion.return_value = mock_response
        yield mock_ll


# ---------------------------------------------------------------------------
# Tests: Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config(self):
        cfg = LLMReviewConfig()
        assert cfg.model == "o3"
        assert cfg.temperature == 0.3
        assert cfg.top_n_review == 5
        assert cfg.reflexion_rounds == 1
        assert cfg.max_total_calls == 15

    def test_custom_config(self):
        cfg = LLMReviewConfig(
            model="claude-3-5-sonnet-20241022",
            top_n_review=3,
            reflexion_rounds=2,
            conflict_threshold=0.3,
        )
        assert cfg.model == "claude-3-5-sonnet-20241022"
        assert cfg.top_n_review == 3
        assert cfg.reflexion_rounds == 2
        assert cfg.conflict_threshold == 0.3


# ---------------------------------------------------------------------------
# Tests: Stock Selection
# ---------------------------------------------------------------------------


class TestStockSelection:
    def test_selects_top_buys_and_sells(self, sample_rankings, mock_litellm):
        config = LLMReviewConfig(top_n_review=2, reflexion_rounds=0)
        agent = LLMReviewAgent(config=config)
        selected = agent._select_stocks_for_review(sample_rankings)

        tickers = {s["ticker"] for s in selected}
        # Should include top 2 buys and top 2 sells
        assert "AAPL" in tickers  # strong_buy
        assert "NVDA" in tickers  # buy
        assert "GME" in tickers   # strong_sell
        assert "TSLA" in tickers  # sell

    def test_includes_high_conflict_holds(self, mock_litellm):
        # Hold signal but agents strongly disagree
        stocks = [
            _make_stock("CONFLICT", composite=0.0, signal="hold",
                        sentiment=0.8, fundamental=-0.6, insider=0.5),
        ]
        config = LLMReviewConfig(top_n_review=5, reflexion_rounds=0, conflict_threshold=0.4)
        agent = LLMReviewAgent(config=config)
        selected = agent._select_stocks_for_review(stocks)
        assert any(s["ticker"] == "CONFLICT" for s in selected)

    def test_skips_non_conflict_holds(self, mock_litellm):
        stocks = [
            _make_stock("BORING", composite=0.0, signal="hold",
                        sentiment=0.1, fundamental=0.1),
        ]
        config = LLMReviewConfig(top_n_review=5, reflexion_rounds=0)
        agent = LLMReviewAgent(config=config)
        selected = agent._select_stocks_for_review(stocks)
        # BORING has no buy/sell signal and no conflict
        assert not any(s["ticker"] == "BORING" for s in selected)


# ---------------------------------------------------------------------------
# Tests: Conflict Detection
# ---------------------------------------------------------------------------


class TestConflictDetection:
    def test_detects_conflict(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock("X", sentiment=0.8, fundamental=-0.6)
        assert agent._is_high_conflict(stock) is True

    def test_no_conflict_when_aligned(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock("X", sentiment=0.8, fundamental=0.6)
        assert agent._is_high_conflict(stock) is False

    def test_no_conflict_when_neutral(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock("X", sentiment=0.03, fundamental=-0.02)
        assert agent._is_high_conflict(stock) is False

    def test_conflict_threshold_respected(self, mock_litellm):
        config = LLMReviewConfig(conflict_threshold=0.6, reflexion_rounds=0)
        agent = LLMReviewAgent(config=config)
        # 0.5 is above default 0.4 but below 0.6
        stock = _make_stock("X", sentiment=0.5, fundamental=-0.5)
        assert agent._is_high_conflict(stock) is False

        stock2 = _make_stock("X", sentiment=0.7, fundamental=-0.7)
        assert agent._is_high_conflict(stock2) is True


# ---------------------------------------------------------------------------
# Tests: Confidence Extraction
# ---------------------------------------------------------------------------


class TestConfidenceExtraction:
    def test_extracts_high(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        text = "Overall, HIGH confidence in the signal."
        assert agent._extract_confidence(text) == "HIGH"

    def test_extracts_low(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        text = "Given thin evidence, confidence: LOW"
        assert agent._extract_confidence(text) == "LOW"

    def test_extracts_medium(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        text = "MEDIUM confidence in the buy signal."
        assert agent._extract_confidence(text) == "MEDIUM"

    def test_defaults_to_medium(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        text = "The stock looks interesting but uncertain."
        assert agent._extract_confidence(text) == "MEDIUM"


# ---------------------------------------------------------------------------
# Tests: Risk Flags
# ---------------------------------------------------------------------------


class TestRiskFlags:
    def test_low_confidence_flag(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock("X", signal="buy")
        flags = agent._extract_risk_flags(stock, "", "LOW")
        assert "llm_low_confidence" in flags

    def test_agent_disagreement_flag(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock("X", sentiment=0.8, fundamental=-0.6)
        flags = agent._extract_risk_flags(stock, "", "HIGH")
        assert "agent_disagreement" in flags

    def test_thin_evidence_flag(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        # strong_buy with most scores at 0.0
        stock = _make_stock("X", signal="strong_buy", sentiment=0.9)
        flags = agent._extract_risk_flags(stock, "", "HIGH")
        assert "thin_evidence_for_strong_signal" in flags

    def test_sentiment_dominated_flag(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock("X", sentiment=0.8, fundamental=0.1)
        flags = agent._extract_risk_flags(stock, "", "HIGH")
        assert "sentiment_dominated" in flags


# ---------------------------------------------------------------------------
# Tests: LLM Call & Review
# ---------------------------------------------------------------------------


class TestReviewExecution:
    def test_single_stock_review(self, mock_litellm):
        config = LLMReviewConfig(reflexion_rounds=0)
        agent = LLMReviewAgent(config=config)
        stock = _make_stock("AAPL", composite=0.8, signal="buy", sentiment=0.7, fundamental=0.6)
        review = agent._review_single_stock(stock)

        assert review.ticker == "AAPL"
        assert review.signal == "buy"
        assert review.confidence_in_signal == "HIGH"
        assert "bullish thesis" in review.review_text.lower() or "sentiment" in review.review_text.lower()
        assert review.reflexion_applied is False
        assert review.model_used == "o3"
        mock_litellm.completion.assert_called_once()

    def test_reflexion_makes_extra_call(self, mock_litellm):
        config = LLMReviewConfig(reflexion_rounds=2)
        agent = LLMReviewAgent(config=config)
        stock = _make_stock("AAPL", composite=0.8, signal="buy", sentiment=0.7)
        review = agent._review_single_stock(stock)

        assert review.reflexion_applied is True
        # 1 initial + 2 reflexion rounds = 3 calls
        assert mock_litellm.completion.call_count == 3

    def test_call_cap_stops_reflexion(self, mock_litellm):
        config = LLMReviewConfig(reflexion_rounds=5, max_total_calls=2)
        agent = LLMReviewAgent(config=config)
        stock = _make_stock("AAPL", composite=0.8, signal="buy")
        review = agent._review_single_stock(stock)

        # Should have made at most 2 calls (cap hit during reflexion)
        assert mock_litellm.completion.call_count <= 2


# ---------------------------------------------------------------------------
# Tests: Full Review Flow
# ---------------------------------------------------------------------------


class TestFullReview:
    def test_review_rankings_end_to_end(self, sample_rankings, mock_litellm):
        config = LLMReviewConfig(top_n_review=2, reflexion_rounds=0)
        agent = LLMReviewAgent(config=config)
        summary = agent.review_rankings(sample_rankings, verbose=True)

        assert isinstance(summary, ReviewSummary)
        assert len(summary.reviews) > 0
        assert summary.model_used == "o3"
        assert summary.total_llm_calls > 0

        # Each review should have required fields
        for review in summary.reviews:
            assert review.ticker
            assert review.signal
            assert review.review_text
            assert review.confidence_in_signal in ("HIGH", "MEDIUM", "LOW")

    def test_inject_reviews_into_rankings(self, mock_litellm):
        config = LLMReviewConfig(reflexion_rounds=0)
        agent = LLMReviewAgent(config=config)

        stock = _make_stock("AAPL", composite=0.8, signal="buy")
        rankings = [stock]

        review = StockReview(
            ticker="AAPL",
            signal="buy",
            composite_score=0.8,
            review_text="Test review",
            confidence_in_signal="HIGH",
            model_used="gpt-4o",
        )
        summary = ReviewSummary(reviews=[review], model_used="gpt-4o")

        agent.inject_reviews_into_rankings(rankings, summary)

        assert "llm_review" in rankings[0]["agent_rationales"]
        assert rankings[0]["agent_rationales"]["llm_review"]["ticker"] == "AAPL"
        assert rankings[0]["agent_rationales"]["llm_review"]["review_text"] == "Test review"

    def test_call_cap_limits_total_reviews(self, sample_rankings, mock_litellm):
        config = LLMReviewConfig(
            top_n_review=10,
            reflexion_rounds=0,
            max_total_calls=3,
        )
        agent = LLMReviewAgent(config=config)
        summary = agent.review_rankings(sample_rankings)

        assert mock_litellm.completion.call_count <= 3


# ---------------------------------------------------------------------------
# Tests: Prompt Building
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    def test_builds_valid_prompt(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock(
            "AAPL",
            composite=0.8,
            signal="buy",
            sentiment=0.7,
            fundamental=0.5,
            ml_percentile=75.0,
            sector="Technology",
        )
        stock["agent_rationales"]["fundamental"] = {
            "evidence": ["Quality score: 8.5", "Low debt-to-equity"],
        }
        prompt = agent._build_review_prompt(stock)

        assert "AAPL" in prompt
        assert "BUY" in prompt
        assert "0.800" in prompt
        assert "Technology" in prompt
        assert "Quality score: 8.5" in prompt
        assert "Low debt-to-equity" in prompt

    def test_handles_missing_rationales(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock("XYZ", composite=0.3, signal="buy")
        stock["agent_rationales"] = {}
        prompt = agent._build_review_prompt(stock)

        assert "XYZ" in prompt
        assert "No detailed evidence" in prompt


# ---------------------------------------------------------------------------
# Tests: ReviewSummary Output
# ---------------------------------------------------------------------------


class TestReviewSummary:
    def test_to_dict(self):
        review = StockReview(
            ticker="AAPL",
            signal="buy",
            composite_score=0.8,
            review_text="Good stock",
            confidence_in_signal="HIGH",
            model_used="gpt-4o",
        )
        summary = ReviewSummary(
            reviews=[review],
            total_llm_calls=1,
            model_used="gpt-4o",
        )
        d = summary.to_dict()
        assert d["total_llm_calls"] == 1
        assert len(d["reviews"]) == 1
        assert d["reviews"][0]["ticker"] == "AAPL"

    def test_to_markdown(self):
        review = StockReview(
            ticker="AAPL",
            signal="buy",
            composite_score=0.8,
            review_text="- Risk of multiple compression\n- Sentiment-driven",
            confidence_in_signal="HIGH",
            risk_flags=["sentiment_dominated"],
            model_used="gpt-4o",
        )
        summary = ReviewSummary(
            reviews=[review],
            total_llm_calls=2,
            total_latency_ms=3500,
            model_used="gpt-4o",
        )
        md = summary.to_markdown()

        assert "## 6) LLM Red-Team Review" in md
        assert "AAPL" in md
        assert "buy" in md
        assert "HIGH" in md
        assert "sentiment_dominated" in md
        assert "2" in md  # llm calls
        assert "3.5s" in md  # latency

    def test_groups_by_signal(self):
        buys = StockReview(ticker="A", signal="buy", composite_score=0.5,
                           review_text="ok", confidence_in_signal="HIGH", model_used="gpt-4o")
        sells = StockReview(ticker="B", signal="sell", composite_score=-0.5,
                            review_text="bad", confidence_in_signal="LOW", model_used="gpt-4o")
        summary = ReviewSummary(reviews=[buys, sells], model_used="gpt-4o")
        md = summary.to_markdown()

        assert "Bullish Signals" in md
        assert "Bearish Signals" in md


# ---------------------------------------------------------------------------
# Tests: Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalizes_dicts(self, mock_litellm):
        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        stock = _make_stock("AAPL")
        result = agent._normalize_to_dicts([stock])
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_normalizes_dataclass(self, mock_litellm):
        @dataclass
        class FakeRanking:
            ticker: str = "NVDA"
            composite_score: float = 0.5
            signal: str = "buy"
            ml_percentile: float = 70.0
            sentiment_score: float = 0.3
            fundamental_score: float = 0.4
            earnings_score: float = 0.0
            insider_score: float = 0.0
            thematic_score: float = 0.0
            momentum_score: float = 0.0
            filing_tone_score: float = 0.0
            earnings_call_qual_score: float = 0.0
            company_name: str = "NVIDIA"
            sector: str = "Technology"
            key_factors: list = field(default_factory=list)
            agent_rationales: dict = field(default_factory=dict)

        agent = LLMReviewAgent(config=LLMReviewConfig(reflexion_rounds=0))
        result = agent._normalize_to_dicts([FakeRanking()])
        assert result[0]["ticker"] == "NVDA"
        assert result[0]["composite_score"] == 0.5


# ---------------------------------------------------------------------------
# Tests: Template Constants
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_system_prompt_exists(self):
        assert "adversarial" in SYSTEM_PROMPT.lower()
        assert "red-team" in SYSTEM_PROMPT.lower()
        assert "score" in SYSTEM_PROMPT.lower()

    def test_stock_template_has_placeholders(self):
        assert "{ticker}" in STOCK_REVIEW_TEMPLATE
        assert "{signal}" in STOCK_REVIEW_TEMPLATE
        assert "{composite:" in STOCK_REVIEW_TEMPLATE
        assert "{evidence_block}" in STOCK_REVIEW_TEMPLATE

    def test_reflexion_template_has_placeholders(self):
        assert "{ticker}" in REFLEXION_TEMPLATE
        assert "{previous_review}" in REFLEXION_TEMPLATE


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_handles_llm_failure_gracefully(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception("API timeout")
        config = LLMReviewConfig(top_n_review=1, reflexion_rounds=0)
        agent = LLMReviewAgent(config=config)

        stocks = [_make_stock("FAIL", composite=0.8, signal="buy")]
        summary = agent.review_rankings(stocks)

        assert len(summary.reviews) == 0
        assert "FAIL" in summary.skipped_tickers

    def test_no_litellm_raises_import_error(self):
        with patch.dict("sys.modules", {"litellm": None}):
            with patch("auto_researcher.agents.llm_review_agent.HAS_LITELLM", False):
                with pytest.raises(ImportError, match="litellm"):
                    LLMReviewAgent()
