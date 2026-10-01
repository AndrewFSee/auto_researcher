"""Tests for fundamental factor panels and the purged growth forecast (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_researcher.features.fundamental_factors import (
    GROWTH_FEATURES,
    composite_score,
    growth_labels,
    intrinsic_ev,
)
from auto_researcher.features.valuation import dcf_value
from auto_researcher.models.growth_forecast import baseline_forecasts, walk_forward_forecasts


def test_intrinsic_ev_reduces_to_constant_growth_dcf():
    # Growth already at the terminal rate and an unchanged margin: a plain DCF.
    got = intrinsic_ev(np.array([1000.0]), np.array([0.1]), np.array([0.025]), np.array([0.1]))
    assert got[0] == pytest.approx(dcf_value(100.0, 0.025, 0.09, 10, 0.025))


def test_intrinsic_ev_rises_with_growth_and_margin():
    base = intrinsic_ev(np.array([1000.0]), np.array([0.1]), np.array([0.05]), np.array([0.1]))
    assert intrinsic_ev(np.array([1000.0]), np.array([0.1]), np.array([0.10]), np.array([0.1])) > base
    assert intrinsic_ev(np.array([1000.0]), np.array([0.1]), np.array([0.05]), np.array([0.15])) > base


def test_composite_respects_signs_and_requires_all_factors():
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-31")], ["A", "B", "C"]],
                                     names=["date", "symbol"])
    panel = pd.DataFrame({"good": [3.0, 2.0, 1.0], "bad": [1.0, 2.0, np.nan]}, index=idx)
    score = composite_score(panel, {"good": 1, "bad": -1})
    assert score["2020-01-31", "A"] > score["2020-01-31", "B"]
    assert np.isnan(score["2020-01-31", "C"])


def _frame(dates, n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n)]], names=["date", "symbol"])
    X = pd.DataFrame(rng.normal(size=(len(idx), len(GROWTH_FEATURES))), index=idx, columns=GROWTH_FEATURES)
    X["sector"] = pd.Categorical(rng.choice(["a", "b"], len(idx)), categories=["a", "b"])
    return X


def test_forecasts_train_only_on_labels_known_before_refit():
    dates = pd.date_range("2010-03-31", "2020-12-31", freq="QE")
    train = _frame(dates)
    train["y_growth"] = 0.05
    train["y_margin"] = 0.1
    train["label_date"] = train.index.get_level_values("date") + pd.Timedelta(days=1096)
    # Poison labels that are not yet known at the 2017 refit: if they leaked, the
    # 2017 predictions would move towards 0.9.
    late = train["label_date"] >= pd.Timestamp("2017-01-01")
    train.loc[late, ["y_growth", "y_margin"]] = 0.9
    predict = _frame(pd.date_range("2017-01-31", "2017-12-31", freq="ME"), seed=1)
    got = walk_forward_forecasts(train, predict, [pd.Timestamp("2017-01-01")], min_rows=100)
    assert len(got) == len(predict)
    assert got["growth_ml"].max() < 0.1 and got["margin_ml"].max() < 0.2
    assert (got["model_date"] == pd.Timestamp("2017-01-01")).all()


def test_forecasts_skip_refits_without_enough_history():
    dates = pd.date_range("2015-03-31", "2015-12-31", freq="QE")
    train = _frame(dates, n=10)
    train[["y_growth", "y_margin"]] = 0.0
    train["label_date"] = train.index.get_level_values("date") + pd.Timedelta(days=1096)
    got = walk_forward_forecasts(train, _frame([pd.Timestamp("2016-06-30")]), [pd.Timestamp("2016-01-01")])
    assert got.empty


def test_baselines_shrink_towards_sector():
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2020-06-30")], ["A", "B"]], names=["date", "symbol"])
    X = pd.DataFrame({"rev_cagr_3y": [0.30, np.nan], "sector_median_growth": [0.10, 0.10],
                      "fcff_margin": [0.20, 0.10],
                      "sector": pd.Categorical(["x", "x"])}, index=idx)
    b = baseline_forecasts(X)
    assert b.loc[("2020-06-30", "A"), "growth_b2"] == pytest.approx(0.20)
    assert b.loc[("2020-06-30", "B"), "growth_b2"] == pytest.approx(0.10)
    assert b.loc[("2020-06-30", "A"), "margin_m2"] == pytest.approx(0.175)


def test_labels_unknown_after_data_end():
    table = pd.DataFrame({
        "symbol": "A", "item": "revenue",
        "period_end": pd.to_datetime(["2015-12-31", "2018-12-31", "2021-12-31"]),
        "available_date": pd.to_datetime(["2016-02-15", "2019-02-15", "2022-02-15"]),
        "value": [100.0, 133.1, 177.0],
    })
    lab = growth_labels(table, pd.DatetimeIndex(["2016-06-30", "2019-06-30"]), ["A"],
                        data_end=pd.Timestamp("2022-01-01"))
    assert lab.loc[("2016-06-30", "A"), "y_growth"] == pytest.approx(0.10, abs=1e-3)
    assert np.isnan(lab.loc[("2019-06-30", "A"), "y_growth"])  # label date 2022-07 > data end
