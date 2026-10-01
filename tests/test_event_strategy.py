"""Tests for the event-driven strategy backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_researcher.backtest.event_strategy import (
    EventStrategyConfig,
    classify_events,
    return_stats,
    simulate_event_strategy,
)


def _events(n=300, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"ticker": "A", "usable_date": days, "value": rng.normal(size=n)})


def test_percentiles_use_only_earlier_events():
    cfg = EventStrategyConfig(min_history=50, lookback_days=365)
    ev = _events()
    base = classify_events(ev, cfg)
    # Changing a late event's value must not change any earlier classification.
    ev2 = ev.copy()
    ev2.loc[280, "value"] = 1e6
    moved = classify_events(ev2, cfg)
    pd.testing.assert_series_equal(base["pct"].iloc[:280], moved["pct"].iloc[:280])
    assert base["pct"].iloc[:50].isna().all()  # not enough history yet
    assert set(base["side"].unique()) <= {-1, 0, 1}


def test_percentile_thresholds_assign_sides():
    cfg = EventStrategyConfig(min_history=10, lookback_days=1000)
    days = pd.bdate_range("2020-01-01", periods=13)
    ev = pd.DataFrame({"ticker": "A", "usable_date": days,
                       "value": list(range(10)) + [100.0, -100.0, 4.5]})
    out = classify_events(ev, cfg).set_index("value")
    assert out.loc[100.0, "side"] == 1
    assert out.loc[-100.0, "side"] == -1
    assert out.loc[4.5, "side"] == 0


def test_returns_start_after_entry_and_stop_after_hold():
    cal = pd.bdate_range("2021-01-04", periods=30)
    price = pd.Series(100.0, index=cal)
    price.iloc[5:] = 110.0     # jump on day 5 (before entry): must not be earned
    price.iloc[12:] = 121.0    # +10% on day 12: inside the holding window
    price.iloc[20:] = 133.1    # +10% on day 20: after exit
    close = pd.DataFrame({"A": price, "B": 100.0})
    ev = pd.DataFrame({"ticker": ["A"], "usable_date": [cal[5]], "side": [1]})
    cfg = EventStrategyConfig(hold_days=10, entry_lag=1, cost_bps=10)
    out = simulate_event_strategy(ev, close, cfg)
    # Entry at close of day 6; returns earned on days 7..16.
    assert (out["n_long"].iloc[7:17] == 1).all() and out["n_long"].iloc[:7].sum() == 0
    assert out["long"].iloc[12] == pytest.approx(0.10)
    assert out["long"].iloc[5] == 0 and out["long"].iloc[20] == 0
    # One unit bought on day 7 and sold on day 17: 10 bps each time.
    assert out["long_cost"].iloc[7] == pytest.approx(0.001)
    assert out["long_cost"].iloc[17] == pytest.approx(0.001)
    assert out["long_short_net"].sum() == pytest.approx(0.10 - 0.002)


def test_return_stats():
    rng = np.random.default_rng(1)
    s = return_stats(pd.Series(rng.normal(0.003, 0.01, 1000)), n_trials=3)
    assert s["t_stat"] > 2 and 0 < s["deflated_prob"] <= 1 and s["max_drawdown"] <= 0
    assert np.isnan(return_stats(pd.Series([0.0] * 5))["sharpe"])
