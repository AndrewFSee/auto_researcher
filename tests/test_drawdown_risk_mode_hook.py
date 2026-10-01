"""
Phase 4 wire-up: enhanced_portfolio.pick_agent_risk_mode.

The portfolio layer needs a single-line way to pick the AlphaAgents risk
posture from the current drawdown state. These tests pin the contract:

* Float input is forwarded to ``select_risk_mode`` verbatim.
* ``DrawdownState`` input pulls ``drawdown_pct`` off the state and
  delegates (so the two import paths stay consistent with exactly one
  threshold constant).
* The default threshold matches ``select_risk_mode``'s default (-5%),
  which is where the debate was calibrated.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass

from auto_researcher.backtest.enhanced_portfolio import pick_agent_risk_mode


@dataclass
class _FakeDrawdownState:
    """Minimal stand-in for risk.drawdown_control.DrawdownState.

    We don't import the real class — tests shouldn't take a hard
    dependency on the risk package to verify this thin delegation.
    """
    drawdown_pct: float
    current_value: float = 1_000_000.0
    peak_value: float = 1_100_000.0
    days_in_drawdown: int = 0


class TestPickAgentRiskMode:
    def test_shallow_drawdown_returns_risk_neutral(self) -> None:
        assert pick_agent_risk_mode(0.0) == "risk_neutral"
        assert pick_agent_risk_mode(-0.01) == "risk_neutral"
        assert pick_agent_risk_mode(-0.049) == "risk_neutral"

    def test_deep_drawdown_returns_risk_averse(self) -> None:
        assert pick_agent_risk_mode(-0.05) == "risk_averse"
        assert pick_agent_risk_mode(-0.08) == "risk_averse"
        assert pick_agent_risk_mode(-0.25) == "risk_averse"

    def test_accepts_drawdown_state_dataclass(self) -> None:
        state = _FakeDrawdownState(drawdown_pct=-0.12)
        assert pick_agent_risk_mode(state) == "risk_averse"

        state_shallow = _FakeDrawdownState(drawdown_pct=-0.02)
        assert pick_agent_risk_mode(state_shallow) == "risk_neutral"

    def test_custom_threshold(self) -> None:
        # At -3% drawdown with a -2% threshold, we should be risk-averse.
        assert (
            pick_agent_risk_mode(-0.03, risk_averse_threshold=-0.02)
            == "risk_averse"
        )
        # Same drawdown with a -10% threshold stays risk-neutral.
        assert (
            pick_agent_risk_mode(-0.03, risk_averse_threshold=-0.10)
            == "risk_neutral"
        )

    def test_delegates_to_alpha_agents_select_risk_mode(self) -> None:
        """Decisions must match ``select_risk_mode`` — one threshold, one truth."""
        from auto_researcher.agents.alpha_agents import select_risk_mode

        for dd in (-0.10, -0.05, -0.04, 0.0, +0.02):
            assert pick_agent_risk_mode(dd) == select_risk_mode(dd)

    def test_interop_with_real_drawdown_state_if_available(self) -> None:
        """Real DrawdownState should Just Work without importing its module
        into enhanced_portfolio (duck typing on `.drawdown_pct`)."""
        try:
            from auto_researcher.risk.drawdown_control import DrawdownState
        except ImportError:  # pragma: no cover — module should exist
            import pytest
            pytest.skip("drawdown_control module unavailable in this env")

        state = DrawdownState(
            current_value=900_000.0,
            peak_value=1_000_000.0,
            drawdown_pct=-0.10,
        )
        assert pick_agent_risk_mode(state) == "risk_averse"
