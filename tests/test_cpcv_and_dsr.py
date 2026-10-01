"""Tests for Combinatorial Purged Cross-Validation + Deflated Sharpe."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from auto_researcher.validation.cpcv import (
    CPCVSplit,
    combinatorial_purged_splits,
    n_cpcv_paths,
)
from auto_researcher.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
)


# ---------------------------------------------------------------------------
# CPCV
# ---------------------------------------------------------------------------


def test_cpcv_emits_all_combinations() -> None:
    dates = pd.date_range("2020-01-02", periods=600, freq="B")
    splits = list(
        combinatorial_purged_splits(
            dates=dates,
            n_splits=6,
            n_test_splits=2,
            horizon_days=21,
            embargo_pct=0.01,
        )
    )
    # Should produce C(6,2) = 15 combos.
    assert len(splits) == 15

    # Every distinct pair of groups appears exactly once.
    combos = {s.test_group_ids for s in splits}
    assert combos == set(combinations(range(6), 2))


def test_cpcv_respects_purge_and_embargo() -> None:
    dates = pd.date_range("2020-01-02", periods=600, freq="B")
    n_splits = 6
    horizon = 21
    embargo_pct = 0.01
    total_days = (dates[-1] - dates[0]).days
    embargo_days = int(round(embargo_pct * total_days))
    purge_td = pd.Timedelta(days=horizon)
    embargo_td = pd.Timedelta(days=embargo_days)

    # Reconstruct the group boundaries the implementation uses so we can
    # check per-group purge windows.
    edges = np.linspace(0, len(dates), n_splits + 1, dtype=int)
    group_windows = [
        (dates[edges[i]], dates[edges[i + 1] - 1]) for i in range(n_splits)
    ]

    for split in combinatorial_purged_splits(
        dates=dates,
        n_splits=n_splits,
        n_test_splits=2,
        horizon_days=horizon,
        embargo_pct=embargo_pct,
    ):
        train_dates = dates[split.train_idx]

        # No training row may fall inside any individual test group's
        # [group_start - horizon, group_end + embargo] window.
        for g in split.test_group_ids:
            gs, ge = group_windows[g]
            in_window = (train_dates >= gs - purge_td) & (
                train_dates <= ge + embargo_td
            )
            assert not in_window.any(), (
                f"training row leaked into purge window of group {g}"
            )

        # No overlap between train and test indices.
        assert not np.any(np.isin(split.train_idx, split.test_idx))


def test_cpcv_paths_formula() -> None:
    # López de Prado: C(N, K) * K / N distinct paths.
    assert n_cpcv_paths(6, 2) == 15 * 2 // 6 == 5


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


def test_cpcv_purges_trading_days_on_both_sides_of_each_test_group() -> None:
    """Horizon is in trading days and applies before AND after each test group."""
    calendar = pd.bdate_range("2020-01-02", periods=300)
    # Panel layout: 4 tickers per date, rows sorted by date.
    dates = pd.DatetimeIndex(np.repeat(calendar, 4))
    horizon = 21
    pos = {d: i for i, d in enumerate(calendar)}

    for split in combinatorial_purged_splits(
        dates=dates, n_splits=6, n_test_splits=2, horizon_days=horizon, embargo_pct=0.0
    ):
        train_pos = np.array([pos[d] for d in dates[split.train_idx]])
        test_pos = np.array([pos[d] for d in dates[split.test_idx]])
        # Split the test rows back into their contiguous groups.
        breaks = np.flatnonzero(np.diff(test_pos) > 1)
        for grp in np.split(test_pos, breaks + 1):
            lo, hi = grp.min(), grp.max()
            too_close = (train_pos >= lo - horizon) & (train_pos <= hi + horizon)
            assert not too_close.any(), "training label window overlaps a test label window"


def test_cpcv_old_calendar_purge_would_have_leaked() -> None:
    """21 calendar days ~ 15 trading days: the old purge left rows this new one removes."""
    calendar = pd.bdate_range("2020-01-02", periods=300)
    splits = list(
        combinatorial_purged_splits(
            dates=calendar, n_splits=6, n_test_splits=1, horizon_days=21, embargo_pct=0.0
        )
    )
    middle = splits[2]  # a test group with training data on both sides
    lo, hi = middle.test_idx.min(), middle.test_idx.max()
    train = set(middle.train_idx)
    # Positions 16-21 trading days before the group, and 1-21 after it, must be purged.
    assert not any(p in train for p in range(lo - 21, lo))
    assert not any(p in train for p in range(hi + 1, hi + 22))


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
