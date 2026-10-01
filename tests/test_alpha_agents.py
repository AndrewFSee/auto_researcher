"""Tests for the AlphaAgents debate-and-consensus module.

All LLM interaction is stubbed via ``_StubClient`` so the suite runs
without network access or API credentials. Each test exercises a
specific contract the module is promising to uphold:

* masking actually strips ticker/company mentions before any text
  reaches the client,
* the short-circuit path avoids debate rounds when agents agree,
* the debate path runs up to ``max_debate_rounds`` when they disagree,
* caching is deterministic on the masked input,
* the parser / consensus-agent fallback produce sensible numbers even
  when responses are malformed,
* the risk-mode selector flips mode at the configured threshold.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest


@pytest.fixture
def repo_tmp_path():
    """Repo-local temp dir — the OS ``%TEMP%`` can be permissioned-out on
    this machine, which breaks pytest's ``tmp_path`` fixture. This creates
    a sibling ``.pytest_tmp`` under the repo so tests can still read/write."""
    base = Path(__file__).parent.parent / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="alpha_agents_", dir=str(base)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

from auto_researcher.agents.alpha_agents import (
    AlphaAgentsDebate,
    AgentOpinion,
    ConsensusAgent,
    DebateCache,
    DebateConfig,
    LLMClient,
    PromptMasker,
    StockContext,
    _heuristic_consensus,
    _parse_agent_response,
    select_risk_mode,
)


# ---------------------------------------------------------------------------
# Stub LLM client
# ---------------------------------------------------------------------------

@dataclass
class _StubClient:
    """Records every (system, user) call and returns scripted responses.

    ``responses`` is a dict from ``persona`` → list of JSON-encoded strings
    that will be popped one at a time. ``calls`` records everything sent
    to the LLM so tests can assert on masking behavior.
    """

    responses: dict[str, list[str]]
    calls: list[dict] = field(default_factory=list)
    default: str = '{"direction":"HOLD","score":0.0,"confidence":0.5,"rationale":"default"}'

    def complete(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        persona = self._detect_persona(system)
        self.calls.append({
            "persona": persona, "system": system, "user": user,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        bucket = self.responses.get(persona, [])
        if bucket:
            return bucket.pop(0)
        return self.default

    @staticmethod
    def _detect_persona(system: str) -> str:
        lowered = system.lower()
        # Order matters — the consensus system prompt mentions
        # "fundamental, sentiment, and valuation analysts", so check for it first.
        if "consensus moderator" in lowered:
            return "consensus"
        if "fundamental analyst" in lowered:
            return "fundamental"
        if "sentiment" in lowered and "flow analyst" in lowered:
            return "sentiment"
        if "valuation analyst" in lowered:
            return "valuation"
        return "unknown"


def _ctx() -> StockContext:
    return StockContext(
        ticker="AAPL", date="2026-02-01",
        sector="Technology", size_bucket="Large",
        company_name="Apple Inc.",
        fundamental_signals={"roe": 1.2, "margin": 0.3},
        sentiment_signals={"finbert_tone": 0.4},
        valuation_signals={"pe_vs_sector": -0.5},
        evidence_by_agent={
            "fundamental": ["Apple Inc. had a 30% margin", "AAPL grew revenue"],
            "sentiment": ["positive coverage of Apple this quarter"],
        },
    )


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def test_masker_scrubs_ticker_and_company_case_insensitive():
    m = PromptMasker("AAPL", "Apple Inc.", "Technology", "Large")
    text = "aapl is up 3%, Apple Inc. reported earnings, and apple's CEO spoke."
    out = m.scrub(text)
    # AAPL and Apple Inc. must be gone; "apple's" stays (not a word boundary match).
    assert "AAPL" not in out.upper().replace("REDACTED", "")
    assert "Apple Inc." not in out
    assert "REDACTED" in out


def test_masker_leaves_sector_anchor_intact():
    m = PromptMasker("NVDA", "NVIDIA", "Technology", "Large")
    assert m.placeholder_label() == "{LARGE-TECHNOLOGY-STOCK}"


def test_masker_scrub_evidence_preserves_structure():
    m = PromptMasker("AAPL", "Apple", "Technology", "Large")
    ev = {"a": ["AAPL rallied", "Apple reported"], "b": ["no mention"]}
    scrubbed = m.scrub_evidence(ev)
    assert set(scrubbed) == {"a", "b"}
    assert all("AAPL" not in s for s in scrubbed["a"])
    assert scrubbed["b"] == ["no mention"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_parse_accepts_plain_json():
    raw = '{"direction":"BUY","score":0.6,"confidence":0.8,"rationale":"strong"}'
    d, s, c, r = _parse_agent_response(raw)
    assert (d, s, c, r) == ("BUY", 0.6, 0.8, "strong")


def test_parse_strips_markdown_fences():
    raw = '```json\n{"direction":"SELL","score":-0.4,"confidence":0.7,"rationale":"x"}\n```'
    d, s, c, r = _parse_agent_response(raw)
    assert d == "SELL" and s == -0.4


def test_parse_enforces_direction_score_consistency():
    # BUY but negative score → parser flips to positive.
    raw = '{"direction":"BUY","score":-0.3,"confidence":0.6,"rationale":"x"}'
    d, s, _, _ = _parse_agent_response(raw)
    assert d == "BUY" and s > 0
    # HOLD clamps into [-0.2, 0.2].
    raw = '{"direction":"HOLD","score":0.9,"confidence":0.6,"rationale":"x"}'
    d, s, _, _ = _parse_agent_response(raw)
    assert d == "HOLD" and abs(s) <= 0.2


def test_parse_raises_on_no_json_object():
    with pytest.raises(ValueError):
        _parse_agent_response("no json here")


# ---------------------------------------------------------------------------
# Cache key and persistence
# ---------------------------------------------------------------------------

def test_cache_roundtrip_memory_only():
    c = DebateCache(cache_dir=None)
    k = DebateCache.key("fundamental", 0, "user", "system", "risk_neutral")
    assert c.get(k) is None
    c.put(k, "abc")
    assert c.get(k) == "abc"


def test_cache_roundtrip_on_disk(repo_tmp_path: Path):
    tmp_path = repo_tmp_path
    c = DebateCache(cache_dir=tmp_path)
    k = DebateCache.key("sentiment", 1, "user", "sys", "risk_averse")
    c.put(k, '{"direction":"HOLD"}')
    # New cache instance pointing at the same dir sees the prior write.
    c2 = DebateCache(cache_dir=tmp_path)
    assert c2.get(k) == '{"direction":"HOLD"}'


def test_cache_key_differs_on_persona_and_risk_mode():
    a = DebateCache.key("fundamental", 0, "user", "sys", "risk_neutral")
    b = DebateCache.key("sentiment",   0, "user", "sys", "risk_neutral")
    c = DebateCache.key("fundamental", 0, "user", "sys", "risk_averse")
    assert len({a, b, c}) == 3


# ---------------------------------------------------------------------------
# Heuristic consensus + risk selector
# ---------------------------------------------------------------------------

def test_heuristic_consensus_unanimous_bull():
    ops = [
        AgentOpinion("fundamental", "BUY", 0.6, 0.9, "x"),
        AgentOpinion("sentiment",   "BUY", 0.5, 0.8, "y"),
        AgentOpinion("valuation",   "BUY", 0.4, 0.7, "z"),
    ]
    d, score, dissent = _heuristic_consensus(ops)
    assert d == "BUY" and score > 0.3 and dissent == pytest.approx(0.2, abs=1e-9)


def test_heuristic_consensus_penalizes_dissent():
    ops = [
        AgentOpinion("fundamental", "BUY",  0.7, 0.9, ""),
        AgentOpinion("sentiment",   "BUY",  0.6, 0.8, ""),
        AgentOpinion("valuation",   "SELL", -0.8, 0.9, ""),
    ]
    _, score, dissent = _heuristic_consensus(ops)
    # Dissent is 1.5 → penalty = max(0, 1 - 0.75) = 0.25 → small consensus.
    assert dissent == pytest.approx(1.5, abs=1e-9)
    assert abs(score) < 0.2


def test_select_risk_mode_threshold():
    assert select_risk_mode(0.0) == "risk_neutral"
    assert select_risk_mode(-0.02) == "risk_neutral"
    assert select_risk_mode(-0.05) == "risk_averse"
    assert select_risk_mode(-0.12) == "risk_averse"


# ---------------------------------------------------------------------------
# Full debate flow
# ---------------------------------------------------------------------------

def _bull_response(score: float = 0.5) -> str:
    return json.dumps({
        "direction": "BUY", "score": score, "confidence": 0.8,
        "rationale": "positive numbers",
    })


def _bear_response(score: float = -0.6) -> str:
    return json.dumps({
        "direction": "SELL", "score": score, "confidence": 0.85,
        "rationale": "negative numbers",
    })


def test_debate_short_circuits_when_agents_agree():
    responses = {
        "fundamental": [_bull_response(0.5)],
        "sentiment":   [_bull_response(0.4)],
        "valuation":   [_bull_response(0.45)],
        "consensus":   [json.dumps({
            "consensus_direction": "BUY", "consensus_score": 0.45,
            "dissent_margin": 0.1, "summary": "all agree",
        })],
    }
    client = _StubClient(responses=responses)
    debate = AlphaAgentsDebate(
        config=DebateConfig(cache_dir=None, disagreement_threshold=0.4),
        client=client, cache=DebateCache(cache_dir=None),
    )
    verdict = debate.analyze(_ctx(), risk_mode="risk_neutral")
    assert verdict.n_debate_rounds == 0
    assert len(verdict.opinions_by_round) == 1
    assert verdict.consensus_direction == "BUY"
    assert verdict.consensus_score > 0.3


def test_debate_runs_rounds_on_disagreement():
    # Round 0: disagree hard → triggers debate.
    # Round 1: still disagree.
    # Round 2: converge — but debate only runs max_debate_rounds=2 here.
    responses = {
        "fundamental": [_bull_response(0.8), _bull_response(0.7), _bull_response(0.4)],
        "sentiment":   [_bull_response(0.6), _bull_response(0.5), _bull_response(0.3)],
        "valuation":   [_bear_response(-0.8), _bear_response(-0.5), _bull_response(0.2)],
        "consensus":   [json.dumps({
            "consensus_direction": "BUY", "consensus_score": 0.3,
            "dissent_margin": 0.2, "summary": "moved after debate",
        })],
    }
    client = _StubClient(responses=responses)
    debate = AlphaAgentsDebate(
        config=DebateConfig(
            cache_dir=None, disagreement_threshold=0.4, max_debate_rounds=2,
        ),
        client=client, cache=DebateCache(cache_dir=None),
    )
    verdict = debate.analyze(_ctx(), risk_mode="risk_neutral")
    assert verdict.n_debate_rounds >= 1
    assert len(verdict.opinions_by_round) >= 2


def test_debate_masks_ticker_in_all_user_prompts():
    responses = {
        "fundamental": [_bull_response(0.3)],
        "sentiment":   [_bull_response(0.3)],
        "valuation":   [_bull_response(0.3)],
        "consensus":   [json.dumps({
            "consensus_direction": "BUY", "consensus_score": 0.3,
            "dissent_margin": 0.0, "summary": "",
        })],
    }
    client = _StubClient(responses=responses)
    debate = AlphaAgentsDebate(
        config=DebateConfig(cache_dir=None, disagreement_threshold=0.4),
        client=client, cache=DebateCache(cache_dir=None),
    )
    debate.analyze(_ctx(), risk_mode="risk_neutral")

    # Every user prompt the client received must not mention AAPL or Apple Inc.
    # as bare words.
    aapl = re.compile(r"\bAAPL\b", re.IGNORECASE)
    apple_inc = re.compile(r"\bApple Inc\.", re.IGNORECASE)
    for call in client.calls:
        assert not aapl.search(call["user"]), f"leaked ticker in: {call['user']!r}"
        assert not apple_inc.search(call["user"]), f"leaked company in: {call['user']!r}"


def test_debate_honors_risk_mode_in_system_prompt():
    responses = {
        "fundamental": [_bull_response(0.3)],
        "sentiment":   [_bull_response(0.3)],
        "valuation":   [_bull_response(0.3)],
        "consensus":   [json.dumps({
            "consensus_direction": "HOLD", "consensus_score": 0.1,
            "dissent_margin": 0.0, "summary": "",
        })],
    }
    client = _StubClient(responses=responses)
    debate = AlphaAgentsDebate(
        config=DebateConfig(cache_dir=None, disagreement_threshold=0.4),
        client=client, cache=DebateCache(cache_dir=None),
    )
    debate.analyze(_ctx(), risk_mode="risk_averse")

    # Every persona system prompt must carry the risk-averse addendum.
    for call in client.calls:
        if call["persona"] in ("fundamental", "sentiment", "valuation"):
            assert "RISK-AVERSE" in call["system"].upper(), (
                f"risk mode missing from {call['persona']} system prompt"
            )


def test_cache_hits_avoid_second_llm_call(repo_tmp_path: Path):
    tmp_path = repo_tmp_path
    responses = {
        "fundamental": [_bull_response(0.3)],
        "sentiment":   [_bull_response(0.3)],
        "valuation":   [_bull_response(0.3)],
        "consensus":   [json.dumps({
            "consensus_direction": "BUY", "consensus_score": 0.3,
            "dissent_margin": 0.0, "summary": "",
        })],
    }
    client = _StubClient(responses=responses)
    cache = DebateCache(cache_dir=tmp_path)
    debate = AlphaAgentsDebate(
        config=DebateConfig(cache_dir=tmp_path, disagreement_threshold=0.4),
        client=client, cache=cache,
    )
    debate.analyze(_ctx(), risk_mode="risk_neutral")
    first_call_count = len(client.calls)

    # Second run with a fresh client but same cache dir → zero new calls.
    client2 = _StubClient(responses={})
    debate2 = AlphaAgentsDebate(
        config=DebateConfig(cache_dir=tmp_path, disagreement_threshold=0.4),
        client=client2, cache=DebateCache(cache_dir=tmp_path),
    )
    verdict = debate2.analyze(_ctx(), risk_mode="risk_neutral")
    assert len(client2.calls) == 0
    assert verdict.consensus_direction == "BUY"
    assert first_call_count > 0


# ---------------------------------------------------------------------------
# ConsensusAgent fallback behavior
# ---------------------------------------------------------------------------

def test_consensus_agent_falls_back_on_malformed_response():
    client = _StubClient(responses={"consensus": ["not json"]})
    consensus = ConsensusAgent(client=client, cache=DebateCache(cache_dir=None))
    ops = [
        AgentOpinion("fundamental", "BUY", 0.6, 0.9, ""),
        AgentOpinion("sentiment",   "BUY", 0.5, 0.8, ""),
        AgentOpinion("valuation",   "BUY", 0.4, 0.7, ""),
    ]
    d, score, dissent, summary = consensus.synthesize(
        _ctx(), PromptMasker("AAPL", "Apple", "Technology", "Large"),
        "risk_neutral", ops,
    )
    assert d == "BUY" and score > 0.3
    assert summary == ""  # malformed → no summary
