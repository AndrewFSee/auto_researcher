"""Tests for evidence-based composite weights and the event-date guard."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from auto_researcher.composite import (
    AGENTS,
    PRIOR_IC,
    SHRINK_PERIODS,
    AgentEvidence,
    composite_weights,
    load_agent_evidence,
    parse_agent_evidence,
)
from auto_researcher.validation.event_dates import (
    PeriodEndDateError,
    assert_announcement_dates,
    month_end_fraction,
)

# ---------------------------------------------------------------------------
# Composite weights
# ---------------------------------------------------------------------------


def test_weights_sum_to_one_and_cover_all_agents():
    weights, effective, provenance = composite_weights({})
    assert set(weights) == set(AGENTS) == set(effective) == set(provenance)
    assert sum(weights.values()) == pytest.approx(1.0)
    # No evidence at all: every agent gets the same prior weight.
    assert len(set(np.round(list(weights.values()), 12))) == 1
    assert all(v == "prior" for v in provenance.values())


def test_negative_ic_gets_zero_weight_not_abs():
    evidence = parse_agent_evidence({"momentum": {"mean_ic": -0.03, "n_periods": 100, "source": "x"}})
    weights, effective, _ = composite_weights(evidence)
    assert effective["momentum"] == 0.0
    assert weights["momentum"] == 0.0


def test_shrinkage_toward_prior_by_number_of_periods():
    ev = AgentEvidence(ic=0.10, n_periods=36, source="x")
    assert ev.shrunk_ic() == pytest.approx((36 * 0.10 + SHRINK_PERIODS * PRIOR_IC) / (36 + SHRINK_PERIODS))
    few = AgentEvidence(ic=0.10, n_periods=5, source="x")
    many = AgentEvidence(ic=0.10, n_periods=500, source="x")
    assert PRIOR_IC < few.shrunk_ic() < many.shrunk_ic() < 0.10


def test_small_positive_measurement_never_ranks_below_the_prior():
    # A weak but positive measurement (9 quarters) must not lose to "no evidence".
    evidence = parse_agent_evidence({"earnings": {"mean_ic": 0.09, "n_periods": 9, "source": "study"}})
    weights, _, _ = composite_weights(evidence)
    assert weights["earnings"] > weights["insider"]


def test_well_measured_zero_ic_gets_little_weight():
    evidence = parse_agent_evidence({"ml": {"mean_ic": 0.0, "n_periods": 400, "source": "wf"}})
    _, effective, _ = composite_weights(evidence)
    assert effective["ml"] < 0.2 * PRIOR_IC


def test_measured_positive_ic_beats_prior():
    evidence = parse_agent_evidence({"earnings": {"mean_ic": 0.08, "n_periods": 120, "source": "study"}})
    weights, _, provenance = composite_weights(evidence)
    assert weights["earnings"] > weights["insider"]
    assert provenance["earnings"].startswith("measured IC +0.0800 over 120 periods")


def test_entries_without_period_counts_are_priors():
    calibration = {
        "insider": {"mean_ic": 0.06, "source": "academic_literature"},
        "thematic": {"mean_ic": 0.0998, "n_obs": 31, "source": "tech_backtest_results.csv"},
        "ml": {"mean_ic": 0.15, "source": "computed_at_runtime"},
    }
    assert parse_agent_evidence(calibration) == {}


def test_known_contaminated_calibration_is_ignored():
    calibration = {
        "earnings": {"mean_ic": 0.1067, "n_periods": 80, "source": "pead_backtest_results.parquet"},
        "fundamental": {"mean_ic": 0.0196, "n_periods": 90, "source": "news_combined_results.parquet"},
        "sentiment": {"mean_ic": 0.02, "n_periods": 90, "status": "invalid"},
    }
    assert parse_agent_evidence(calibration) == {}


def test_malformed_and_metadata_entries_are_skipped():
    calibration = {
        "_ic_weights": {"ml": 1.0},
        "_calibrated_at": "2026-01-01",
        "fundamental": {"mean_ic": "n/a", "n_periods": 5},
        "sentiment": {"mean_ic": float("nan"), "n_periods": 5},
        "momentum": {"mean_ic": 0.01, "n_periods": 0},
    }
    assert parse_agent_evidence(calibration) == {}


def test_all_non_positive_evidence_falls_back_to_uniform():
    evidence = {a: AgentEvidence(ic=-0.01, n_periods=50, source="x") for a in AGENTS}
    weights, _, _ = composite_weights(evidence)
    assert all(w == pytest.approx(1 / len(AGENTS)) for w in weights.values())


def test_prior_value_is_small():
    _, effective, _ = composite_weights({})
    assert all(v == PRIOR_IC for v in effective.values())


def test_load_agent_evidence_from_file(repo_tmp_path):
    path = repo_tmp_path / "agent_ic.json"
    path.write_text(json.dumps({"ml": {"mean_ic": -0.025, "n_periods": 105, "source": "walkforward.json"}}))
    evidence = load_agent_evidence(path)
    assert evidence["ml"].ic == pytest.approx(-0.025)
    assert load_agent_evidence(path.with_name("missing.json")) == {}
    path.write_text("{not json")
    assert load_agent_evidence(path) == {}


# ---------------------------------------------------------------------------
# Event-date guard
# ---------------------------------------------------------------------------


def test_fiscal_quarter_ends_are_rejected():
    quarter_ends = pd.date_range("2018-03-31", periods=24, freq="QE")
    assert month_end_fraction(quarter_ends) == 1.0
    with pytest.raises(PeriodEndDateError, match="fiscal period ends"):
        assert_announcement_dates(quarter_ends)


def test_announcement_like_dates_pass():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", "2025-12-31")
    sample = pd.Series(rng.choice(dates, size=500))
    assert month_end_fraction(sample) < 0.15
    assert_announcement_dates(sample)  # does not raise


def test_empty_dates_are_not_flagged():
    assert month_end_fraction(pd.Series([], dtype="datetime64[ns]")) == 0.0
    assert_announcement_dates(pd.Series([None, None]))
