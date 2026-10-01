"""Tests for the deflated Sharpe ratio."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from auto_researcher.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
)


# ---------------------------------------------------------------------------
# CPCV
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------


def test_expected_max_sharpe_grows_with_trials() -> None:
    e1 = expected_max_sharpe_under_null(1)
    e10 = expected_max_sharpe_under_null(10)
    e100 = expected_max_sharpe_under_null(100)
    assert e1 == 0.0  # No search → no penalty
    assert e10 > 0
    assert e100 > e10


def test_deflated_sharpe_penalizes_selection_bias() -> None:
    # Same raw Sharpe, but one was the best of many trials.
    dsr_single = deflated_sharpe_ratio(
        sharpe=1.0, n_obs=120, n_trials=1, sharpe_std=1.0
    )
    dsr_many = deflated_sharpe_ratio(
        sharpe=1.0, n_obs=120, n_trials=200, sharpe_std=1.0
    )

    assert 0.0 <= dsr_many <= dsr_single <= 1.0
    # The multi-trial version should clearly penalize.
    assert dsr_single - dsr_many > 0.05


def test_deflated_sharpe_probability_bounds() -> None:
    # Obscene Sharpe with a single trial → deflated ≈ 1 (virtually certain).
    p_high = deflated_sharpe_ratio(
        sharpe=5.0, n_obs=250, n_trials=1, sharpe_std=1.0
    )
    # Zero Sharpe → deflated is 0.5 (no evidence either way) when n_trials=1.
    p_zero = deflated_sharpe_ratio(
        sharpe=0.0, n_obs=250, n_trials=1, sharpe_std=1.0
    )

    assert p_high > 0.95
    assert abs(p_zero - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Regression tests for the 2026-09 audit fixes
# ---------------------------------------------------------------------------






def test_deflated_sharpe_accepts_annualized_inputs() -> None:
    """An annual SR of 1.0 over 3 years of daily data is ~96% likely positive, not ~100%."""
    p = deflated_sharpe_ratio(sharpe=1.0, n_obs=756, n_trials=1, periods_per_year=252)
    assert 0.93 < p < 0.98
    # Treating the annual number as per-period (the old usage) wildly overstates it.
    assert deflated_sharpe_ratio(sharpe=1.0, n_obs=756, n_trials=1) > 0.9999


def test_deflated_sharpe_default_null_dispersion_is_sampling_sd() -> None:
    """With no cross-trial spread given, the null uses 1/sqrt(n-1) per period."""
    n, trials = 250, 20
    expected_max = expected_max_sharpe_under_null(trials, 1.0 / math.sqrt(n - 1))
    # A per-period Sharpe exactly at the expected null maximum is a coin flip.
    p = deflated_sharpe_ratio(sharpe=expected_max, n_obs=n, n_trials=trials)
    assert abs(p - 0.5) < 0.01
