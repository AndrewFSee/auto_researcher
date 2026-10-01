"""Tests for the stage-1 ML screen (``auto_researcher.screening``)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from auto_researcher.screening import (
    UNIVERSES,
    ScreeningModel,
    ScreeningModelConfig,
    build_feature_panel,
    estimate_holdout_ic,
    per_date_spearman_ic,
)


def _panel(n_dates: int = 260, n_names: int = 30, beta: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-01", periods=n_dates), [f"T{i:02d}" for i in range(n_names)]],
        names=["date", "ticker"],
    )
    X = pd.DataFrame(rng.normal(size=(len(idx), 6)), index=idx,
                     columns=["signal", "n1", "n2", "n3", "n4", "n5"])
    y = beta * X["signal"] + pd.Series(rng.normal(size=len(idx)), index=idx)
    return X, y


def test_per_date_spearman_matches_scipy():
    X, y = _panel(n_dates=40)
    expected = np.mean([spearmanr(X["signal"].xs(d), y.xs(d)).statistic
                        for d in X.index.get_level_values(0).unique()])
    assert per_date_spearman_ic(X["signal"], y) == pytest.approx(expected, abs=1e-12)


def test_per_date_spearman_ignores_tiny_cross_sections():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2021-01-01", periods=2), ["A", "B"]])
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    assert np.isnan(per_date_spearman_ic(s, s))


def test_model_prunes_to_informative_features():
    X, y = _panel()
    model = ScreeningModel(ScreeningModelConfig(min_features=1)).fit(X, y)
    assert "signal" in model.features_
    assert model.feature_ic_["signal"] == pytest.approx(model.feature_ic_.max())


def test_model_keeps_minimum_feature_count():
    X, y = _panel()
    model = ScreeningModel(ScreeningModelConfig(min_features=4)).fit(X, y)
    assert len(model.features_) >= 4


def test_contributions_add_up_to_predictions():
    X, y = _panel()
    model = ScreeningModel().fit(X, y)
    sample = X.iloc[:20]
    contrib = model.feature_contributions(sample)
    assert list(contrib.columns) == model.features_
    bias = model.predict(sample) - contrib.sum(axis=1).to_numpy()
    assert np.allclose(bias, bias[0], atol=1e-4)  # constant base value


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        ScreeningModel().predict(_panel(n_dates=5)[0])


def test_holdout_ic_detects_signal_and_needs_enough_history():
    X, y = _panel()
    assert estimate_holdout_ic(X, y, horizon_days=21) > 0.1
    short_X, short_y = _panel(n_dates=100)
    assert np.isnan(estimate_holdout_ic(short_X, short_y, horizon_days=21))


def test_holdout_ic_is_near_zero_without_signal():
    X, y = _panel(beta=0.0, seed=4)
    assert abs(estimate_holdout_ic(X, y, horizon_days=21)) < 0.1


def test_feature_panel_shape_and_clipping():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2019-01-01", periods=400)
    cols = [f"T{i}" for i in range(10)] + ["SPY"]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (400, 11)), axis=0)),
                          index=dates, columns=cols)
    panel = build_feature_panel(prices, benchmark="SPY")
    assert panel.index.names == ["date", "ticker"]
    assert "SPY" not in panel.index.get_level_values("ticker")
    assert panel.abs().max().max() <= 3.0
    assert "tech_resid_mom_252" in panel.columns


def test_universes_are_callables_returning_tickers():
    for name in ("sp100", "large_cap", "core_tech"):
        tickers = UNIVERSES[name]()
        assert tickers and all(isinstance(t, str) for t in tickers)


def test_repository_root_recommend_module_is_gone():
    # The package must not depend on a module that only exists at the repo root.
    assert importlib.util.find_spec("recommend") is None


def test_feature_panel_requires_benchmark():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2019-01-01", periods=300)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (300, 5)), axis=0)),
                          index=dates, columns=[f"T{i}" for i in range(5)])
    with pytest.raises(ValueError, match="benchmark"):
        build_feature_panel(prices, benchmark="SPY")
