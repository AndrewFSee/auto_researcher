"""
Phase 0.6 audit fix — verify the inner-stop slice is disjoint from inner-val
and properly purged from inner-fit.

Without these guarantees, Optuna trials get to peek at the scoring fold via
``early_stopping_rounds`` and the reported best-IC is an inflated max-over-
trials on a single realization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_researcher.models.hyperparam_tuner import (
    TunerConfig,
    _carve_inner_stop_slice,
    _purged_time_series_splits,
)


def _build_panel(n_dates: int = 200, n_per_date: int = 10) -> pd.DatetimeIndex:
    """Date axis for a small (date, ticker) panel."""
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    return pd.DatetimeIndex(np.repeat(dates.values, n_per_date))


def test_stop_slice_is_disjoint_from_val():
    dates = _build_panel()
    n = len(dates)
    splits = _purged_time_series_splits(
        dates=dates, n_samples=n, n_splits=3, purge_days=21, embargo_days=0,
    )
    assert splits, "Expected at least one purged split"

    cfg = TunerConfig()
    for train_idx, val_idx in splits:
        fit_idx, stop_idx = _carve_inner_stop_slice(
            train_idx, dates, stop_frac=cfg.stop_slice_frac, purge_days=cfg.purge_days,
        )
        # 1. stop_idx is non-empty (panels of this size should support it)
        assert len(stop_idx) > 0, "stop slice empty — early stopping would silently fall back"

        # 2. stop and val are disjoint by date — the leak we are fixing
        stop_dates = set(pd.DatetimeIndex(dates.take(stop_idx)).unique())
        val_dates = set(pd.DatetimeIndex(dates.take(val_idx)).unique())
        assert stop_dates.isdisjoint(val_dates), (
            "stop slice overlaps val by date — Optuna would still peek"
        )

        # 3. fit and stop are purged: max(fit_date) + purge_days <= min(stop_date)
        fit_max = pd.DatetimeIndex(dates.take(fit_idx)).max()
        stop_min = pd.DatetimeIndex(dates.take(stop_idx)).min()
        assert (stop_min - fit_max).days >= cfg.purge_days, (
            f"fit→stop gap {(stop_min - fit_max).days}d < purge {cfg.purge_days}d"
        )

        # 4. fit ∩ stop is empty by row index too
        assert set(fit_idx).isdisjoint(set(stop_idx))


def test_stop_slice_no_dates_falls_back_to_row_count():
    """When dates can't be inferred, we still carve a stop slice by row count."""
    train_idx = np.arange(0, 100)
    fit_idx, stop_idx = _carve_inner_stop_slice(
        train_idx, dates=None, stop_frac=0.2, purge_days=21,
    )
    assert len(stop_idx) == 20
    assert len(fit_idx) == 80
    assert set(fit_idx).isdisjoint(set(stop_idx))
    # Stop slice is the *tail* — it should be the most recent rows
    assert stop_idx[0] == 80
    assert stop_idx[-1] == 99


def test_stop_slice_too_small_returns_empty():
    """Pathologically small train slices disable early stopping rather than leak."""
    short_dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=3))
    train_idx = np.arange(len(short_dates))
    fit_idx, stop_idx = _carve_inner_stop_slice(
        train_idx, short_dates, stop_frac=0.15, purge_days=21,
    )
    # Caller checks len(stop_idx) > 0 before enabling early stopping
    assert len(stop_idx) == 0
    assert np.array_equal(fit_idx, train_idx)


@pytest.mark.skipif(
    not pytest.importorskip("xgboost", reason="xgboost not installed"),
    reason="xgboost required",
)
def test_tuner_runs_end_to_end_without_leak():
    """Smoke test — tuner completes with the new inner-stop carving in place."""
    pytest.importorskip("optuna")
    from auto_researcher.models.hyperparam_tuner import tune_xgb_hyperparams

    rng = np.random.default_rng(0)
    n_dates, n_per = 150, 8
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, range(n_per)], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((n_dates * n_per, 5)), index=idx,
                     columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.standard_normal(n_dates * n_per) * 0.01, index=idx)

    cfg = TunerConfig(n_trials=3, n_cv_splits=2, timeout_seconds=30)
    best = tune_xgb_hyperparams(X, y, config=cfg)
    assert "n_estimators" in best
    assert "learning_rate" in best

    # Phase 1: deflation diagnostics should have been recorded.
    from auto_researcher.models.hyperparam_tuner import get_last_tuning_diagnostics
    diag = get_last_tuning_diagnostics()
    assert diag is not None
    assert diag["n_trials"] >= 1
    # With 3 trials and random features, expected_max_under_null is nonzero.
    assert np.isfinite(diag["expected_max_under_null"])
    # deflated_best_value = best - E[max|null] is defined.
    assert "deflated_best_value" in diag
