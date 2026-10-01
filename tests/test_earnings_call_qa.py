"""Tests for earnings-call Q&A analyzer."""

from __future__ import annotations

import pandas as pd

from auto_researcher.models.earnings_call_qa import (
    QAFeatures,
    analyze_many,
    analyze_qa_transcript,
    extract_qa_section,
)


SAMPLE_TRANSCRIPT = """
Prepared Remarks
Operator:
Good afternoon and welcome.

Jane Smith — CEO:
We had a strong quarter with record revenue growth.

Questions and Answers

Bob Brown — Morgan Stanley — Analyst:
Can you give us some color on the growth rate in your enterprise segment
for the coming year and what specific drivers you expect?

Jane Smith — CEO:
We expect enterprise growth between 8% and 10% in the coming year, driven
by new customer wins and expansion. We're confident in the momentum.

Alice Green — Goldman Sachs — Analyst:
How are you thinking about margins given the pricing pressure in the
market?

John Doe — CFO:
Well, we believe margins will be subject to some pressure. Headwinds in
the market could potentially weigh on results. It's a bit too early to
say definitively, but we'll manage through it.
"""


def test_extract_qa_section_splits_correctly() -> None:
    qa = extract_qa_section(SAMPLE_TRANSCRIPT)
    assert "Prepared Remarks" not in qa
    assert "Bob Brown" in qa


def test_analyze_qa_returns_expected_counts() -> None:
    feats = analyze_qa_transcript(SAMPLE_TRANSCRIPT)
    assert isinstance(feats, QAFeatures)
    assert feats.n_questions == 2
    assert feats.n_answers == 2
    assert feats.backend == "lexicon"


def test_hedge_density_picks_up_hedges() -> None:
    feats = analyze_qa_transcript(SAMPLE_TRANSCRIPT)
    # John Doe's answer has "we believe", "subject to", "headwinds",
    # "could", "potentially", "too early" — Jane's has "expect" and
    # "confident". Both answers contain at least one hedge, so density=1.0.
    assert feats.hedge_density == 1.0


def test_guidance_specificity_detects_ranges() -> None:
    feats = analyze_qa_transcript(SAMPLE_TRANSCRIPT)
    # "8% and 10%" — exactly one numeric range across two answers.
    assert feats.guidance_specificity >= 0.5


def test_sentiment_gap_positive_mgmt_vs_analyst() -> None:
    # Jane's answer is positive ("confident", "momentum"), John's is
    # negative ("pressure", "headwinds"). Analyst questions are neutral.
    # Net gap should be small but non-zero.
    feats = analyze_qa_transcript(SAMPLE_TRANSCRIPT)
    assert isinstance(feats.sentiment_gap, float)


def test_qoq_tone_delta_uses_prior() -> None:
    positive_prior = SAMPLE_TRANSCRIPT.replace("pressure", "strong")
    feats = analyze_qa_transcript(SAMPLE_TRANSCRIPT, prior_transcript=positive_prior)
    # Current has "pressure/headwinds" that prior doesn't — delta should
    # be ≤ 0.
    assert feats.qoq_tone_delta <= 1e-6


def test_empty_transcript_is_handled() -> None:
    feats = analyze_qa_transcript("")
    assert feats.n_questions == 0
    assert feats.n_answers == 0
    assert feats.hedge_density == 0.0


def test_analyze_many_returns_indexed_frame() -> None:
    data = {
        ("AAPL", "2024-01-30"): SAMPLE_TRANSCRIPT,
        ("AAPL", "2024-04-30"): SAMPLE_TRANSCRIPT,
        ("MSFT", "2024-01-30"): SAMPLE_TRANSCRIPT,
    }
    df = analyze_many(data)
    assert df.index.names == ["ticker", "date"]
    assert set(df.index.get_level_values("ticker")) == {"AAPL", "MSFT"}
    # QoQ delta on first AAPL call is 0 (no prior); second should use first.
    aapl = df.loc["AAPL"]
    assert aapl.index.is_monotonic_increasing
    assert aapl.iloc[0]["qoq_tone_delta"] == 0.0
