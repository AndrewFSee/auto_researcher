"""
Regression tests for the Phase 0 data-leakage fixes.

These are deliberately surgical: each test isolates one leakage bug that was
present before the Phase 0 audit and asserts the fix stays in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 0.1 Technical features shift
# ---------------------------------------------------------------------------


def test_compute_momentum_simple_excludes_day_t() -> None:
    """``compute_momentum_simple`` must not include day-t return in the signal at t."""
    from auto_researcher.features.technical import compute_momentum_simple

    # Flat history for 20 days, then a huge jump on day t.
    n = 25
    returns = pd.DataFrame(
        np.zeros((n, 1)),
        index=pd.date_range("2023-01-02", periods=n, freq="B"),
        columns=["AAA"],
    )
    day_t = returns.index[20]
    returns.loc[day_t, "AAA"] = 0.25  # Huge shock on day t

    mom = compute_momentum_simple(returns, window=20)

    # At day t the window ends at t-1 (due to shift), so the shock has NOT
    # yet entered the signal — it should still be ~0.
    assert abs(mom.loc[day_t, "AAA"]) < 1e-9, (
        f"compute_momentum_simple leaked day-t shock into signal: "
        f"{mom.loc[day_t, 'AAA']}"
    )

    # One day later the shock is inside the rolling window.
    next_day = returns.index[21]
    assert abs(mom.loc[next_day, "AAA"] - 0.25) < 1e-9


def test_compute_volatility_excludes_day_t() -> None:
    from auto_researcher.features.technical import compute_volatility

    n = 40
    returns = pd.DataFrame(
        np.zeros((n, 1)),
        index=pd.date_range("2023-01-02", periods=n, freq="B"),
        columns=["AAA"],
    )
    day_t = returns.index[30]
    returns.loc[day_t, "AAA"] = 0.2  # Vol shock

    vol = compute_volatility(returns, window=20, annualize=False)
    assert abs(vol.loc[day_t, "AAA"]) < 1e-9, (
        "compute_volatility leaked day-t return into the realized vol at t"
    )


# ---------------------------------------------------------------------------
# 0.4 IC-weight calibration lag
# ---------------------------------------------------------------------------


def test_compute_ic_weights_lags_by_horizon() -> None:
    """``compute_ic_weights`` must lag ICs by the label horizon before rolling."""
    from auto_researcher.backtest.metrics import (
        compute_ic_weights,
        ICWeightConfig,
    )

    # Ramp IC series: mom IC increases with t, qual IC decreases. With a
    # horizon shift the weight at date D is computed from IC values whose
    # index is ≤ D - horizon.
    idx = pd.date_range("2022-01-03", periods=24, freq="MS")
    ic_mom = pd.Series(np.linspace(-0.1, 0.2, len(idx)), index=idx)
    ic_qual = pd.Series(np.linspace(0.2, -0.1, len(idx)), index=idx)

    cfg_lag0 = ICWeightConfig(
        window_mom=3, window_qual=3, label_horizon_periods=0
    )
    cfg_lag3 = ICWeightConfig(
        window_mom=3, window_qual=3, label_horizon_periods=3
    )
    w_lag0 = compute_ic_weights(ic_mom, ic_qual, cfg_lag0)
    w_lag3 = compute_ic_weights(ic_mom, ic_qual, cfg_lag3)

    # If horizon lag works, w_lag3 at date D should equal w_lag0 at D - 3.
    probe_date = idx[10]
    prior_date = idx[7]
    assert abs(w_lag3.loc[probe_date, "w_mom"] - w_lag0.loc[prior_date, "w_mom"]) < 1e-9, (
        "compute_ic_weights is not shifting IC by label_horizon_periods — "
        "weight at D still sees IC[D], which peeks into the realized future."
    )


# ---------------------------------------------------------------------------
# 0.5 Regime assignment is causal
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 0.6 Purged splits inside the hyperparam tuner
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# 0.3 Fundamentals filing-lag enforcement
# ---------------------------------------------------------------------------


