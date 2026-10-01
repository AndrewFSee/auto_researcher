"""
Composite-score weights from per-agent IC evidence.

Each agent's weight is proportional to its *shrunk, signed* information
coefficient::

    w_a ∝ max((n_a · IC_a + n0 · prior) / (n_a + n0), 0)

* **Signed.** An agent whose scores are anti-correlated with forward returns
  gets zero weight. The previous pipeline used ``abs(IC)``, which added
  anti-predictive signals to the composite with a *positive* weight.
* **Shrunk toward the prior.** An IC measured over few independent periods
  stays close to the prior every unmeasured agent gets (``n0 = 36`` periods,
  about three years of monthly observations, is the prior's weight). A small
  positive measurement therefore never ranks *below* an agent with no
  evidence, and a well-measured zero or negative IC drives the weight to zero.
* **Evidence, not guesses.** Only entries with a measured per-period IC and a
  period count (``n_periods``) count as evidence. Literature priors, event-
  level counts, runtime placeholders and unmeasured agents all receive the
  same small prior, so they still contribute but less than anything with
  real out-of-sample support.
* **Known-bad evidence is ignored.** Calibrations built on data with
  look-ahead are listed in ``INVALID_SOURCES`` and fall back to the prior.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

AGENTS: tuple[str, ...] = (
    "ml",
    "sentiment",
    "fundamental",
    "earnings",
    "insider",
    "thematic",
    "momentum",
    "filing_tone",
    "earnings_call_qual",
)

PRIOR_IC = 0.02
SHRINK_PERIODS = 36.0

INVALID_SOURCES: dict[str, str] = {
    "pead_backtest_results.parquet": (
        "event dates are fiscal quarter-ends, so the forward-return window "
        "contains the earnings announcement itself (look-ahead)"
    ),
    "news_combined_results.parquet": (
        "after-close news leaked into its return windows and its signal weights "
        "were chosen on the full sample (see docs/results/sentiment_audit.md)"
    ),
    "combined_fundamentals_results.parquet": (
        "built from a single current snapshot of estimate revisions and from "
        "earnings surprises dated at fiscal period ends (look-ahead; see "
        "models/fundamentals_alpha.py). Point-in-time fundamental factors are "
        "evaluated in docs/results/fundamental_factors.md"
    ),
}


@dataclass(frozen=True)
class AgentEvidence:
    """A measured, per-period IC for one agent."""

    ic: float
    n_periods: float
    source: str

    def shrunk_ic(self, prior_ic: float = PRIOR_IC) -> float:
        """Precision-weighted average of the measured IC and the prior."""
        n = self.n_periods
        return (n * self.ic + SHRINK_PERIODS * prior_ic) / (n + SHRINK_PERIODS)


def parse_agent_evidence(calibration: dict) -> dict[str, AgentEvidence]:
    """
    Extract usable evidence from an ``agent_ic.json``-style mapping.

    Entries are skipped (and the agent falls back to the prior) when they are
    marked ``"status": "invalid"``, come from a source in ``INVALID_SOURCES``,
    lack a finite ``mean_ic``, or lack a positive ``n_periods``.
    """
    evidence: dict[str, AgentEvidence] = {}
    for agent, entry in calibration.items():
        if agent.startswith("_") or not isinstance(entry, dict):
            continue
        source = str(entry.get("source", "unknown"))
        if entry.get("status") == "invalid" or source in INVALID_SOURCES:
            reason = INVALID_SOURCES.get(source, entry.get("reason", "marked invalid"))
            logger.warning("Ignoring %s IC calibration from %s: %s", agent, source, reason)
            continue
        ic = entry.get("mean_ic")
        n = entry.get("n_periods")
        if ic is None or n is None:
            continue
        try:
            ic, n = float(ic), float(n)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ic) or not math.isfinite(n) or n <= 0:
            continue
        evidence[agent] = AgentEvidence(ic=ic, n_periods=n, source=source)
    return evidence


def load_agent_evidence(path: Path) -> dict[str, AgentEvidence]:
    """Read and parse an ``agent_ic.json`` file; missing or unreadable files yield ``{}``."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            return parse_agent_evidence(json.load(fh))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read IC calibration %s: %s", path, exc)
        return {}


def composite_weights(
    evidence: dict[str, AgentEvidence],
    agents: tuple[str, ...] = AGENTS,
    prior_ic: float = PRIOR_IC,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """
    Turn evidence into normalized weights.

    Returns:
        ``(weights, effective_ics, provenance)`` keyed by agent. ``provenance``
        says whether each weight came from measured evidence or the prior.
        If no agent has a positive effective IC the weights are uniform.
    """
    effective: dict[str, float] = {}
    provenance: dict[str, str] = {}
    for agent in agents:
        ev = evidence.get(agent)
        if ev is None:
            effective[agent] = prior_ic
            provenance[agent] = "prior"
        else:
            effective[agent] = max(ev.shrunk_ic(prior_ic), 0.0)
            provenance[agent] = (
                f"measured IC {ev.ic:+.4f} over {ev.n_periods:.0f} periods ({ev.source})"
            )

    total = sum(effective.values())
    if total <= 0:
        return {a: 1.0 / len(agents) for a in agents}, effective, provenance
    return {a: v / total for a, v in effective.items()}, effective, provenance
