"""Formula checks for the published price/volume factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_researcher.backtest.walk_forward import make_training_target
from auto_researcher.features.alpha_factors import (
    _seasonal,
    compute_alpha_factors,
    cross_sectional_rank,
)


def _panel(n_days: int = 1400, n: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rets = rng.normal(0.0004, 0.015, size=(n_days, n + 1))
    cols = [f"S{i}" for i in range(n)] + ["SPY"]
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols)


def test_ranks_are_centered_percentiles():
    frame = pd.DataFrame([[1.0, 2.0, 3.0, np.nan]], columns=list("abcd"))
    r = cross_sectional_rank(frame).iloc[0]
    assert r[["a", "b", "c"]].tolist() == pytest.approx([1 / 3 - 0.5, 2 / 3 - 0.5, 0.5])
    assert np.isnan(r["d"])


def test_momentum_skips_the_last_month():
    prices = _panel()
    stocks = prices.drop(columns="SPY")
    t = prices.index[-1]
    panel = compute_alpha_factors(prices).xs(t, level="date")
    expected_raw = stocks.shift(21).loc[t] / stocks.shift(252).loc[t] - 1
    assert panel["mom_12_1"].rank().tolist() == expected_raw.rank().tolist()


def test_high_52w_orders_by_distance_from_high():
    prices = _panel()
    stocks = prices.drop(columns="SPY")
    t = prices.index[-1]
    raw = stocks.loc[t] / stocks.iloc[-252:].max()
    got = compute_alpha_factors(prices).xs(t, level="date")["high_52w"]
    assert got.rank().tolist() == raw.rank().tolist()


def test_seasonal_uses_the_same_window_in_prior_years():
    dates = pd.bdate_range("2015-01-01", periods=800)
    close = pd.DataFrame({"A": np.arange(1, 801, dtype=float)}, index=dates)
    s = _seasonal(close, years=2, min_years=2)["A"]
    t = 700
    k1 = close["A"].iloc[t - 252 + 21] / close["A"].iloc[t - 252] - 1
    k2 = close["A"].iloc[t - 504 + 21] / close["A"].iloc[t - 504] - 1
    assert s.iloc[t] == pytest.approx((k1 + k2) / 2)
    assert np.isnan(s.iloc[400])  # only one prior year available


def test_optional_inputs_add_their_factors():
    prices = _panel()
    base = compute_alpha_factors(prices)
    assert "abn_volume" not in base and "sector_mom_6_1" not in base
    vol = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
    sectors = pd.Series({c: "X" for c in prices.columns})
    full = compute_alpha_factors(prices, vol, sectors=sectors)
    assert {"abn_volume", "dollar_volume_trend", "sector_mom_6_1", "ret_21_in_sector"} <= set(full)


def test_requires_benchmark():
    with pytest.raises(ValueError, match="benchmark"):
        compute_alpha_factors(_panel().drop(columns="SPY"))


def test_group_rank_target_removes_group_level():
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-01")], ["A", "B", "C", "D"]],
                                     names=["date", "ticker"])
    # Group X beat group Y by 25%; within each group the spread is the same.
    # (Binary-exact values so the within-group excesses tie exactly.)
    fwd = pd.Series([0.375, 0.125, 0.125, -0.125], index=idx)
    groups = pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"})
    target = make_training_target(fwd, "group_rank", groups=groups)
    # Within-group winners (A, C) tie at the top; losers (B, D) at the bottom.
    assert target["2020-01-01"].loc[["A", "C"]].tolist() == pytest.approx([0.375, 0.375])
    assert target["2020-01-01"].loc[["B", "D"]].tolist() == pytest.approx([-0.125, -0.125])
    with pytest.raises(ValueError):
        make_training_target(fwd, "group_rank")
