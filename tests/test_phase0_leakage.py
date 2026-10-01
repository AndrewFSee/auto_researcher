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


def test_causal_regime_assigner_only_uses_past() -> None:
    from auto_researcher.models.regimes import CausalRegimeAssigner

    # Build two prefix-matching benchmark series that diverge after day 600.
    idx = pd.date_range("2018-01-02", periods=1000, freq="B")
    base = pd.Series(np.cumprod(1.0 + 0.0003 * np.ones(1000)), index=idx)

    mirror = base.copy()
    mirror.iloc[600:] *= np.linspace(1.0, 0.5, 400)  # crash in the tail

    probe_date = idx[500]  # well before the divergence

    assigner_base = CausalRegimeAssigner(base)
    assigner_mirror = CausalRegimeAssigner(mirror)

    # Because both series are identical through probe_date, the regime label
    # there must be identical — the future crash must not bleed back.
    assert assigner_base.assign(probe_date) == assigner_mirror.assign(probe_date)


# ---------------------------------------------------------------------------
# 0.6 Purged splits inside the hyperparam tuner
# ---------------------------------------------------------------------------


def test_purged_time_series_splits_respects_purge_gap() -> None:
    from auto_researcher.models.hyperparam_tuner import _purged_time_series_splits

    dates = pd.DatetimeIndex(pd.date_range("2023-01-02", periods=500, freq="B"))

    splits = _purged_time_series_splits(
        dates=dates,
        n_samples=len(dates),
        n_splits=3,
        purge_days=21,
        embargo_days=0,
    )

    assert splits, "expected at least one inner split"
    for train_idx, val_idx in splits:
        train_max_date = dates[train_idx.max()]
        val_min_date = dates[val_idx.min()]
        gap = (val_min_date - train_max_date).days
        assert gap >= 21, (
            f"purged split leaked: train ends {train_max_date.date()}, "
            f"val starts {val_min_date.date()} — gap {gap}d < 21d purge"
        )


def test_groups_from_dates_are_per_date() -> None:
    from auto_researcher.models.hyperparam_tuner import _groups_from_dates

    dates = pd.DatetimeIndex(
        ["2023-01-02", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-04", "2023-01-04"]
    )
    indices = np.arange(len(dates))
    groups = _groups_from_dates(dates, indices)
    assert list(groups) == [2, 1, 3], (
        "Expected per-date group sizes [2, 1, 3], got " + repr(groups)
    )


# ---------------------------------------------------------------------------
# 0.3 Fundamentals filing-lag enforcement
# ---------------------------------------------------------------------------


def test_fundamentals_alignment_enforces_filing_lag() -> None:
    from auto_researcher.features.feature_pipeline import (
        _align_fundamentals_to_prices,
    )

    price_dates = pd.date_range("2023-01-03", "2023-09-29", freq="B")
    prices = pd.DataFrame(
        np.ones((len(price_dates), 1)),
        index=price_dates,
        columns=["AAA"],
    )

    # Fundamentals stamped on fiscal-quarter-end, shape (date, ticker).
    fund_long = pd.DataFrame(
        {"pe_ratio": [10.0, 11.0, 12.0]},
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-03-31"), "AAA"),
                (pd.Timestamp("2023-06-30"), "AAA"),
                (pd.Timestamp("2023-09-30"), "AAA"),
            ],
            names=["date", "ticker"],
        ),
    )

    aligned = _align_fundamentals_to_prices(fund_long, prices, filing_lag_days=45)

    # Convert to a ticker-level slice we can probe: expected columns are
    # MultiIndex (ticker, factor) or similar — just find the matching column.
    aaa_cols = [c for c in aligned.columns if "AAA" in str(c)]
    assert aaa_cols, f"Expected an AAA column after alignment, got {aligned.columns}"
    series = aligned[aaa_cols[0]]

    # On 2023-04-03 (Mon), only 3 calendar days after fiscal end — filing lag
    # should still mask the value.
    early = pd.Timestamp("2023-04-03")
    if early in series.index:
        assert pd.isna(series.loc[early]) or series.loc[early] != 10.0, (
            f"Fundamentals leaked through filing lag: got {series.loc[early]} "
            f"on {early.date()} from fiscal-end 2023-03-31."
        )

    # 60 days later (well past the 45-day lag), the value should be visible.
    late = pd.Timestamp("2023-06-01")
    if late in series.index:
        assert series.loc[late] == 10.0
