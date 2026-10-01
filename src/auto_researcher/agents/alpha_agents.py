"""
AlphaAgents-style debate-and-consensus LLM agents.

What this module is
-------------------
A small, self-contained implementation of the AlphaAgents
(arxiv:2508.11152) pattern for equity research: three LLM personas
reason over the *same* stock context, disagree, and either short-circuit
to consensus or run a bounded debate loop before a consensus agent
synthesizes the final verdict.

Why three personas
------------------
A single LLM prompt tends to collapse into "analyze this stock" —
whatever the model's priors are become the answer. Forcing three
distinct personas with disjoint concerns and then making them debate
surfaces genuinely different objections:

* **Fundamental** — bottom-up on earnings, margins, balance-sheet health.
  Style: patient, numbers-driven, skeptical of narrative.
* **Sentiment**   — crowd positioning, news flow, thematic momentum.
  Style: fast-moving, attentive to flows and narrative shifts.
* **Valuation**   — relative multiples, peer comparisons, mean reversion.
  Style: contrarian, allergic to "expensive-for-a-reason" arguments.

A consensus agent then reads the transcript and returns a bounded score
plus a *dissent margin* — how much the agents disagreed — so downstream
code can penalize low-conviction signals.

Leakage controls
----------------
Ticker and company name are masked before any text reaches an LLM
(replaced with ``REDACTED`` / ``{SECTOR}`` / ``{SIZE_BUCKET}``). The
paper's core concern (and ours) is that frontier LLMs have memorized
"AAPL is a quality compounder" / "TSLA is volatile" priors from their
training corpus, and letting the ticker leak turns the debate into a
retrieval exercise rather than a reasoning one.

Risk-tolerance mode
-------------------
Each agent's system prompt is conditioned on a ``risk_mode`` of
``risk_neutral`` or ``risk_averse``. The portfolio builder picks the
mode from the current drawdown state (see :func:`select_risk_mode`):
inside a drawdown we want agents that weight downside risk more
heavily and raise the bar for BUY recommendations.

Caching
-------
Every LLM call is cached on a key that includes the masked context
hash, the analysis date, the persona name, the risk mode, and a
prompt version stamp. Backtests are therefore deterministic *and* cheap
to rerun — the cache lives in ``cache/alpha_agents/``.

LLM backend
-----------
Uses ``litellm`` (same as the rest of the repo). Tests inject a stub
client via the ``LLMClient`` protocol so the test suite runs without
API credentials or network access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# Bump this when you change any of the prompt templates — it invalidates
# every cached response so stale narratives don't haunt new backtests.
PROMPT_VERSION = "alpha_agents-v1"

RiskMode = Literal["risk_neutral", "risk_averse"]


# ---------------------------------------------------------------------------
# LLM client protocol + default litellm implementation
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Minimal client interface so tests can inject a stub."""

    def complete(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str: ...


class LiteLLMClient:
    """Thin litellm wrapper — imported lazily so tests don't need the package."""

    def __init__(self, model: str, timeout: int = 60):
        import litellm  # noqa: F401 — imported here to fail loudly if missing
        self.model = model
        self.timeout = timeout

    def complete(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        import litellm

        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

@dataclass
class StockContext:
    """Everything an agent needs to reason about one stock on one date.

    The ``ticker`` / ``company_name`` fields are the *raw* identifiers —
    they're scrubbed before any text reaches the LLM. Agent-relevant
    quantitative context lives in the ``*_signals`` dicts; narrative
    context (already-computed evidence bullets from other agents) goes
    in ``evidence_by_agent``.
    """

    ticker: str
    date: str                                          # "YYYY-MM-DD"
    sector: str = "Unknown"
    size_bucket: str = "Unknown"                       # "Large" / "Mid" / "Small"
    company_name: str = ""
    fundamental_signals: dict[str, float] = field(default_factory=dict)
    sentiment_signals: dict[str, float] = field(default_factory=dict)
    valuation_signals: dict[str, float] = field(default_factory=dict)
    evidence_by_agent: dict[str, list[str]] = field(default_factory=dict)


class PromptMasker:
    """Replace ticker/company mentions with sector/size placeholders.

    Word-boundary regex, case-insensitive. Sector and size bucket are
    kept because they're useful reasoning anchors — you can't mask away
    "this is a tech company" without destroying the signal.
    """

    def __init__(self, ticker: str, company: str, sector: str, size_bucket: str):
        self.ticker = ticker
        self.company = company
        self.sector = sector or "Unknown"
        self.size_bucket = size_bucket or "Unknown"
        # Lookbehind/lookahead (rather than \b) so needles ending in
        # punctuation — e.g. "Apple Inc." — still match. \b only triggers at
        # word/non-word transitions, which fails when the trailing char is
        # already non-word.
        self._patterns: list[re.Pattern] = []
        for needle in (ticker, company):
            if needle and isinstance(needle, str) and needle.strip():
                self._patterns.append(
                    re.compile(
                        rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])",
                        flags=re.IGNORECASE,
                    )
                )

    def scrub(self, text: str) -> str:
        if not text:
            return text
        for pat in self._patterns:
            text = pat.sub("REDACTED", text)
        return text

    def placeholder_label(self) -> str:
        """Stable identity string the agents can reference in debate."""
        return f"{{{self.size_bucket.upper()}-{self.sector.upper()}-STOCK}}"

    def scrub_evidence(
        self, evidence_by_agent: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        return {
            agent: [self.scrub(str(item)) for item in items]
            for agent, items in evidence_by_agent.items()
        }


# ---------------------------------------------------------------------------
# Agent opinion types
# ---------------------------------------------------------------------------

@dataclass
class AgentOpinion:
    """One persona's view of a stock — output of ``propose`` and ``revise``."""

    persona: str
    direction: Literal["BUY", "HOLD", "SELL"]
    score: float                                       # -1 .. +1
    confidence: float                                  # 0 .. 1
    rationale: str                                     # bullet list text
    round_index: int = 0                               # 0 = initial, 1+ = debate
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "direction": self.direction,
            "score": self.score,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "round_index": self.round_index,
        }


@dataclass
class ConsensusVerdict:
    """Final output of a debate — one row per stock."""

    ticker: str
    date: str
    consensus_score: float                             # -1 .. +1, bounded
    consensus_direction: Literal["BUY", "HOLD", "SELL"]
    dissent_margin: float                              # 0 = unanimous, 2 = max
    n_debate_rounds: int
    opinions_by_round: list[list[AgentOpinion]]
    risk_mode: RiskMode
    model_used: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "consensus_score": self.consensus_score,
            "consensus_direction": self.consensus_direction,
            "dissent_margin": self.dissent_margin,
            "n_debate_rounds": self.n_debate_rounds,
            "opinions_by_round": [
                [op.to_dict() for op in rnd] for rnd in self.opinions_by_round
            ],
            "risk_mode": self.risk_mode,
            "model_used": self.model_used,
        }


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class DebateCache:
    """File-backed cache keyed by (masked-context hash, date, prompt-version).

    Backtests rerun often. Without a cache, every rerun re-bills the LLM
    for identical prompts — and worse, introduces nondeterminism from
    temperature > 0. The cache key intentionally hashes the *masked*
    context so two stocks with the same features + sector / size share
    cache entries (which is the whole point of masking).
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self._mem: dict[str, str] = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(
        persona: str,
        round_index: int,
        masked_user_prompt: str,
        system_prompt: str,
        risk_mode: str,
    ) -> str:
        h = hashlib.sha256()
        h.update(PROMPT_VERSION.encode())
        h.update(b"|")
        h.update(persona.encode())
        h.update(b"|")
        h.update(str(round_index).encode())
        h.update(b"|")
        h.update(risk_mode.encode())
        h.update(b"|")
        h.update(system_prompt.encode())
        h.update(b"|")
        h.update(masked_user_prompt.encode())
        return h.hexdigest()[:32]

    def get(self, key: str) -> Optional[str]:
        if key in self._mem:
            return self._mem[key]
        if self.cache_dir is None:
            return None
        f = self.cache_dir / f"{key}.json"
        if not f.exists():
            return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            self._mem[key] = data["response"]
            return data["response"]
        except Exception as e:
            logger.warning("DebateCache read error for %s: %s", key, e)
            return None

    def put(self, key: str, response: str) -> None:
        self._mem[key] = response
        if self.cache_dir is None:
            return
        f = self.cache_dir / f"{key}.json"
        try:
            f.write_text(
                json.dumps({"response": response, "ts": time.time()}),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("DebateCache write error for %s: %s", key, e)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RISK_ADDENDUM = {
    "risk_neutral": (
        "Risk posture: RISK-NEUTRAL. Size conviction by expected edge — "
        "do not penalize moderate downside if the upside case is strong."
    ),
    "risk_averse": (
        "Risk posture: RISK-AVERSE. The portfolio is in drawdown. "
        "Raise your bar for BUY: require confirming signals AND a plausible "
        "downside case bounded below a 15% loss. Prefer HOLD over BUY when "
        "unsure. Weight tail-risk evidence (guidance cuts, insider selling, "
        "filing uncertainty language) more heavily than usual."
    ),
}


_FUNDAMENTAL_SYSTEM = """You are a bottom-up FUNDAMENTAL analyst in a structured debate.
You care about earnings quality, margin trends, cash conversion, balance-sheet \
health, and return on invested capital. You are SKEPTICAL of narrative and \
momentum — show me the numbers.

Rules of the debate:
1. Reference the quantitative fundamental signals provided. Name which ones support \
your view and which push against it.
2. Explicitly note where you DISAGREE with the sentiment or valuation view if relevant.
3. Output a JSON object and NOTHING else, with keys:
   {"direction": "BUY"|"HOLD"|"SELL",
    "score": <float in [-1, 1]>,
    "confidence": <float in [0, 1]>,
    "rationale": "<3-5 short bullet points, plain text with \\n separators>"}
4. ``direction`` must be consistent with the sign of ``score`` (SELL→negative, BUY→positive).
5. You may NOT invent numbers not present in the provided context.
"""

_SENTIMENT_SYSTEM = """You are a SENTIMENT / flow analyst in a structured debate.
You care about news tone, analyst revisions, crowd positioning, thematic momentum, \
and narrative shifts. You are SUSPICIOUS of multiples in a vacuum — price discovery \
is a flow phenomenon.

Rules of the debate:
1. Reference the sentiment signals provided (FinBERT tone, news counts, topic \
weighted sentiment, earnings-call tone).
2. Explicitly note where you DISAGREE with the fundamental or valuation view if relevant.
3. Output a JSON object and NOTHING else, with keys:
   {"direction": "BUY"|"HOLD"|"SELL",
    "score": <float in [-1, 1]>,
    "confidence": <float in [0, 1]>,
    "rationale": "<3-5 short bullet points, plain text with \\n separators>"}
4. ``direction`` must be consistent with the sign of ``score``.
5. Do NOT invent numbers not present in the provided context.
"""

_VALUATION_SYSTEM = """You are a relative-VALUATION analyst in a structured debate.
You care about P/E, PEG, EV/EBITDA, P/B versus sector peers and historical bands. \
You are CONTRARIAN — "expensive for a reason" is usually a rationalization.

Rules of the debate:
1. Reference the valuation signals provided (multiples, peer spreads, historical \
percentiles).
2. Explicitly note where you DISAGREE with the fundamental or sentiment view if relevant.
3. Output a JSON object and NOTHING else, with keys:
   {"direction": "BUY"|"HOLD"|"SELL",
    "score": <float in [-1, 1]>,
    "confidence": <float in [0, 1]>,
    "rationale": "<3-5 short bullet points, plain text with \\n separators>"}
4. ``direction`` must be consistent with the sign of ``score``.
5. Do NOT invent numbers not present in the provided context.
"""

_CONSENSUS_SYSTEM = """You are the CONSENSUS moderator of a three-analyst debate.
You have read the fundamental, sentiment, and valuation analysts' final opinions. \
Your job is to produce ONE bounded score that reflects their agreement AND \
penalize low agreement.

Rules:
1. If all three agree (same direction, similar scores), output a confident score.
2. If one disagrees strongly, the consensus score must move toward the center \
and dissent must be acknowledged.
3. Output a JSON object and NOTHING else, with keys:
   {"consensus_direction": "BUY"|"HOLD"|"SELL",
    "consensus_score": <float in [-1, 1]>,
    "dissent_margin": <float in [0, 2], max score - min score>,
    "summary": "<1-3 sentence plain-text synthesis citing which agent dissented if any>"}
4. ``consensus_direction`` sign must match the sign of ``consensus_score``.
"""


def _format_context_block(
    ctx: StockContext, masker: PromptMasker, include_agents: tuple[str, ...]
) -> str:
    lines: list[str] = [
        f"Subject: {masker.placeholder_label()}",
        f"Sector: {ctx.sector}  |  Size: {ctx.size_bucket}  |  Date: {ctx.date}",
    ]

    def _dict_block(title: str, d: Mapping[str, float]) -> None:
        if not d:
            return
        lines.append(f"\n{title}:")
        for k, v in d.items():
            try:
                lines.append(f"  {k}: {float(v):+.3f}")
            except (TypeError, ValueError):
                lines.append(f"  {k}: {v}")

    _dict_block("Fundamental signals", ctx.fundamental_signals)
    _dict_block("Sentiment signals", ctx.sentiment_signals)
    _dict_block("Valuation signals", ctx.valuation_signals)

    if ctx.evidence_by_agent:
        scrubbed = masker.scrub_evidence(ctx.evidence_by_agent)
        included = [a for a in include_agents if scrubbed.get(a)]
        if included:
            lines.append("\nEvidence from other agents:")
            for agent in included:
                lines.append(f"  [{agent.upper()}]")
                for item in scrubbed[agent]:
                    lines.append(f"    - {item}")

    return "\n".join(lines)


def _build_debate_followup(
    ctx: StockContext,
    masker: PromptMasker,
    others: Sequence[AgentOpinion],
    include_agents: tuple[str, ...],
) -> str:
    base = _format_context_block(ctx, masker, include_agents)
    lines = [base, "\nOther analysts' last round:"]
    for op in others:
        lines.append(f"  - [{op.persona.upper()}] direction={op.direction} "
                     f"score={op.score:+.2f} confidence={op.confidence:.2f}")
        for bullet in (op.rationale or "").splitlines():
            if bullet.strip():
                lines.append(f"      {bullet.strip()}")
    lines.append(
        "\nYou have now seen the other analysts' views. Either (a) revise your "
        "opinion citing which of their points moved you, or (b) defend your "
        "prior opinion and say explicitly which of their claims you reject "
        "and why. Output the same JSON schema as before."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_DIRECTION_ALIASES = {
    "BUY": "BUY", "LONG": "BUY", "OVERWEIGHT": "BUY",
    "HOLD": "HOLD", "NEUTRAL": "HOLD",
    "SELL": "SELL", "SHORT": "SELL", "UNDERWEIGHT": "SELL",
}


def _parse_agent_response(text: str) -> tuple[str, float, float, str]:
    """Extract (direction, score, confidence, rationale) from an LLM response."""
    # Strip markdown fences if present.
    cleaned = text.strip()
    for fence in ("```json", "```JSON", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()
    # Grab the first JSON object from the response.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in agent response: {text!r}")
    obj = json.loads(cleaned[start:end + 1])

    raw_dir = str(obj.get("direction", "HOLD")).upper().strip()
    direction = _DIRECTION_ALIASES.get(raw_dir, "HOLD")
    score = float(obj.get("score", 0.0))
    score = max(-1.0, min(1.0, score))
    conf = float(obj.get("confidence", 0.5))
    conf = max(0.0, min(1.0, conf))
    rationale = str(obj.get("rationale", "")).strip()

    # Enforce direction/score sign consistency.
    if direction == "BUY" and score < 0:
        score = abs(score)
    elif direction == "SELL" and score > 0:
        score = -abs(score)
    elif direction == "HOLD":
        score = max(min(score, 0.2), -0.2)
    return direction, score, conf, rationale


# ---------------------------------------------------------------------------
# Persona agents
# ---------------------------------------------------------------------------

_INCLUDE_AGENTS = {
    "fundamental": ("fundamental", "earnings", "filing_tone"),
    "sentiment":   ("sentiment", "thematic", "earnings_call_qual"),
    "valuation":   ("fundamental", "earnings"),
}


def _persona_system(persona: str, risk_mode: RiskMode) -> str:
    base = {
        "fundamental": _FUNDAMENTAL_SYSTEM,
        "sentiment":   _SENTIMENT_SYSTEM,
        "valuation":   _VALUATION_SYSTEM,
    }[persona]
    return base + "\n\n" + _RISK_ADDENDUM[risk_mode]


class PersonaAgent:
    """One analyst persona. Stateless across calls — all state lives in opinions."""

    def __init__(
        self,
        persona: Literal["fundamental", "sentiment", "valuation"],
        client: LLMClient,
        cache: DebateCache,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ):
        self.persona = persona
        self.client = client
        self.cache = cache
        self.temperature = temperature
        self.max_tokens = max_tokens

    def propose(
        self, ctx: StockContext, masker: PromptMasker, risk_mode: RiskMode
    ) -> AgentOpinion:
        user = _format_context_block(ctx, masker, _INCLUDE_AGENTS[self.persona])
        user += (
            "\n\nGive your INITIAL opinion. Remember to output ONLY the JSON "
            "object, no preamble, no explanation outside ``rationale``."
        )
        return self._call_and_parse(user, risk_mode, round_index=0)

    def revise(
        self,
        ctx: StockContext,
        masker: PromptMasker,
        risk_mode: RiskMode,
        others: Sequence[AgentOpinion],
        round_index: int,
    ) -> AgentOpinion:
        user = _build_debate_followup(ctx, masker, others, _INCLUDE_AGENTS[self.persona])
        return self._call_and_parse(user, risk_mode, round_index=round_index)

    def _call_and_parse(
        self, user: str, risk_mode: RiskMode, round_index: int
    ) -> AgentOpinion:
        system = _persona_system(self.persona, risk_mode)
        key = DebateCache.key(
            persona=self.persona, round_index=round_index,
            masked_user_prompt=user, system_prompt=system, risk_mode=risk_mode,
        )
        cached = self.cache.get(key)
        if cached is not None:
            raw = cached
        else:
            raw = self.client.complete(
                system=system, user=user,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            self.cache.put(key, raw)

        try:
            direction, score, conf, rationale = _parse_agent_response(raw)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(
                "AlphaAgents parse error for persona=%s round=%d: %s — defaulting to HOLD/0",
                self.persona, round_index, e,
            )
            direction, score, conf, rationale = "HOLD", 0.0, 0.1, ""

        return AgentOpinion(
            persona=self.persona,
            direction=direction,
            score=score,
            confidence=conf,
            rationale=rationale,
            round_index=round_index,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

def _dissent_margin(opinions: Sequence[AgentOpinion]) -> float:
    """Max minus min score across final opinions. Zero = unanimous."""
    if not opinions:
        return 0.0
    scores = [op.score for op in opinions]
    return float(max(scores) - min(scores))


def _direction_from_score(score: float) -> str:
    if score > 0.15:
        return "BUY"
    if score < -0.15:
        return "SELL"
    return "HOLD"


def _heuristic_consensus(
    opinions: Sequence[AgentOpinion],
) -> tuple[str, float, float]:
    """Confidence-weighted mean score, with dissent penalty.

    Why this exists
    ---------------
    Even with an LLM consensus agent, we want a deterministic fallback —
    both for the "disagreement below threshold" short-circuit and as a
    sanity anchor when the consensus LLM produces a weird direction that
    doesn't match its score. The dissent penalty (``1 - dissent/2``) is
    the paper's insight: low-agreement signals should be smaller.
    """
    if not opinions:
        return "HOLD", 0.0, 0.0
    weights = [max(op.confidence, 0.05) for op in opinions]
    total_w = sum(weights)
    weighted = sum(op.score * w for op, w in zip(opinions, weights)) / total_w
    dissent = _dissent_margin(opinions)
    penalty = max(0.0, 1.0 - (dissent / 2.0))
    consensus_score = weighted * penalty
    consensus_score = max(-1.0, min(1.0, consensus_score))
    return _direction_from_score(consensus_score), consensus_score, dissent


class ConsensusAgent:
    """Summarizes final opinions into a bounded score + dissent margin.

    Falls back to :func:`_heuristic_consensus` if the LLM response is
    unparseable or disagrees with the weighted-mean sign (a hedge
    against consensus LLMs that occasionally produce text at odds with
    the math).
    """

    def __init__(
        self,
        client: LLMClient,
        cache: DebateCache,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ):
        self.client = client
        self.cache = cache
        self.temperature = temperature
        self.max_tokens = max_tokens

    def synthesize(
        self,
        ctx: StockContext,
        masker: PromptMasker,
        risk_mode: RiskMode,
        final_opinions: Sequence[AgentOpinion],
    ) -> tuple[str, float, float, str]:
        heur_dir, heur_score, dissent = _heuristic_consensus(final_opinions)

        lines = [
            f"Subject: {masker.placeholder_label()}  Date: {ctx.date}",
            f"Risk posture: {risk_mode}",
            "\nFinal opinions:",
        ]
        for op in final_opinions:
            lines.append(
                f"  - [{op.persona.upper()}] direction={op.direction} "
                f"score={op.score:+.2f} confidence={op.confidence:.2f}"
            )
            for bullet in (op.rationale or "").splitlines():
                if bullet.strip():
                    lines.append(f"      {bullet.strip()}")
        lines.append(
            f"\nHeuristic weighted-mean score = {heur_score:+.3f}, "
            f"dissent margin = {dissent:.3f}. Produce the consensus JSON."
        )
        user = "\n".join(lines)

        key = DebateCache.key(
            persona="consensus", round_index=99,
            masked_user_prompt=user, system_prompt=_CONSENSUS_SYSTEM, risk_mode=risk_mode,
        )
        cached = self.cache.get(key)
        if cached is not None:
            raw = cached
        else:
            raw = self.client.complete(
                system=_CONSENSUS_SYSTEM, user=user,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            self.cache.put(key, raw)

        try:
            cleaned = raw.strip().lstrip("`")
            start = cleaned.find("{"); end = cleaned.rfind("}")
            obj = json.loads(cleaned[start:end + 1])
            direction = _DIRECTION_ALIASES.get(
                str(obj.get("consensus_direction", "HOLD")).upper(), "HOLD"
            )
            score = max(-1.0, min(1.0, float(obj.get("consensus_score", heur_score))))
            llm_dissent = float(obj.get("dissent_margin", dissent))
            summary = str(obj.get("summary", "")).strip()
        except Exception as e:
            logger.warning("ConsensusAgent parse error: %s — falling back to heuristic", e)
            return heur_dir, heur_score, dissent, ""

        # Sanity-check: if the LLM's direction disagrees with the sign of its own
        # score OR with the heuristic, trust the math.
        if direction == "BUY" and score < 0:
            score = abs(score)
        elif direction == "SELL" and score > 0:
            score = -abs(score)
        if (score > 0) != (heur_score > 0) and abs(heur_score) > 0.05:
            logger.info(
                "ConsensusAgent disagreed with heuristic (llm=%+.2f heur=%+.2f) — "
                "using heuristic for deterministic backtesting",
                score, heur_score,
            )
            return heur_dir, heur_score, dissent, summary

        # If LLM dissent disagrees with our computation, trust ours —
        # LLMs sometimes eyeball a magnitude wrong.
        return _direction_from_score(score), score, dissent, summary


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DebateConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 500
    max_debate_rounds: int = 2
    disagreement_threshold: float = 0.4   # dissent above this → trigger debate
    cache_dir: Optional[Path] = None
    timeout_s: int = 60


class AlphaAgentsDebate:
    """End-to-end debate for one stock on one date.

    Wire-up::

        debate = AlphaAgentsDebate(DebateConfig(model="gpt-4o-mini"))
        verdict = debate.analyze(stock_context, risk_mode="risk_averse")

    The ``verdict.consensus_score`` plugs into the composite ranking
    with its own weight; ``verdict.dissent_margin`` is surfaced to the
    downstream review layer so low-agreement signals can be discounted.
    """

    def __init__(
        self,
        config: Optional[DebateConfig] = None,
        client: Optional[LLMClient] = None,
        cache: Optional[DebateCache] = None,
    ):
        self.config = config or DebateConfig()
        self.client = client or LiteLLMClient(
            model=self.config.model, timeout=self.config.timeout_s
        )
        self.cache = cache or DebateCache(cache_dir=self.config.cache_dir)
        self._personas = {
            name: PersonaAgent(
                name, self.client, self.cache,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            for name in ("fundamental", "sentiment", "valuation")
        }
        self._consensus = ConsensusAgent(self.client, self.cache)

    def analyze(
        self, ctx: StockContext, risk_mode: RiskMode = "risk_neutral"
    ) -> ConsensusVerdict:
        masker = PromptMasker(
            ticker=ctx.ticker, company=ctx.company_name,
            sector=ctx.sector, size_bucket=ctx.size_bucket,
        )

        # Round 0: initial proposals.
        round0 = [
            agent.propose(ctx, masker, risk_mode)
            for agent in self._personas.values()
        ]
        opinions_by_round: list[list[AgentOpinion]] = [round0]

        final = round0
        dissent = _dissent_margin(final)
        rounds_run = 0

        # Debate only if disagreement is material.
        if dissent > self.config.disagreement_threshold:
            for r in range(1, self.config.max_debate_rounds + 1):
                next_round: list[AgentOpinion] = []
                for name, agent in self._personas.items():
                    others = [op for op in final if op.persona != name]
                    next_round.append(agent.revise(ctx, masker, risk_mode, others, r))
                opinions_by_round.append(next_round)
                final = next_round
                rounds_run = r
                new_dissent = _dissent_margin(final)
                if new_dissent <= self.config.disagreement_threshold:
                    break
                dissent = new_dissent

        # Consensus synthesis.
        direction, score, final_dissent, _summary = self._consensus.synthesize(
            ctx, masker, risk_mode, final
        )

        return ConsensusVerdict(
            ticker=ctx.ticker,
            date=ctx.date,
            consensus_score=score,
            consensus_direction=direction,
            dissent_margin=final_dissent,
            n_debate_rounds=rounds_run,
            opinions_by_round=opinions_by_round,
            risk_mode=risk_mode,
            model_used=self.config.model,
        )


# ---------------------------------------------------------------------------
# Risk-mode selector
# ---------------------------------------------------------------------------

def select_risk_mode(
    drawdown_pct: float,
    risk_averse_threshold: float = -0.05,
) -> RiskMode:
    """Pick agent risk posture from current drawdown.

    Why this and not a fancy regime model
    -------------------------------------
    Inside a meaningful drawdown we want agents to raise their bar for
    BUY. A single threshold beats a more elaborate regime detector here
    because the effect we want (more skepticism when bleeding) is
    asymmetric and cheap to justify.

    Args:
        drawdown_pct: Signed drawdown as a fraction (``-0.08`` = down 8%).
        risk_averse_threshold: Switch to risk-averse once drawdown ≤ this
            (default -5%).
    """
    if drawdown_pct <= risk_averse_threshold:
        return "risk_averse"
    return "risk_neutral"
