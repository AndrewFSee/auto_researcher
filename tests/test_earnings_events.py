"""Tests for point-in-time earnings events and features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_researcher.features.earnings_events import (
    EARNINGS_FEATURES,
    announcement_dates,
    earnings_feature_panel,
    time_series_sue,
)


def _filings(rows):
    return pd.DataFrame(rows, columns=["symbol", "form_type", "filing_date", "report_date"])


def test_picks_release_8k_not_earlier_unrelated_filing():
    filings = _filings([
        ("AAA", "8-K", "2020-04-10", "2020-04-08"),   # unrelated, event two days earlier
        ("AAA", "8-K", "2020-04-28", "2020-04-28"),   # earnings release (same-day event)
        ("AAA", "10-Q", "2020-05-01", "2020-03-31"),
        ("AAA", "8-K", "2020-05-20", "2020-05-20"),   # after the 10-Q: outside the window
        ("BBB", "10-K", "2020-02-20", "2019-12-31"),  # no 8-K: fall back to the 10-K date
    ])
    ann = announcement_dates(filings).set_index("symbol")
    assert ann.loc["AAA", "announce_date"] == pd.Timestamp("2020-04-28")
    assert ann.loc["AAA", "source"] == "8-K"
    assert ann.loc["BBB", "announce_date"] == pd.Timestamp("2020-02-20")
    assert ann.loc["BBB", "source"] == "10-Q/10-K"


def test_ignores_8k_right_after_period_end():
    filings = _filings([
        ("AAA", "8-K", "2020-04-02", "2020-04-02"),   # within a week of period end
        ("AAA", "10-Q", "2020-05-01", "2020-03-31"),
    ])
    ann = announcement_dates(filings)
    assert ann["announce_date"].iloc[0] == pd.Timestamp("2020-05-01")


def test_time_series_sue_formula():
    periods = pd.date_range("2015-03-31", periods=16, freq="QE")
    eps = pd.DataFrame({"symbol": "AAA", "period_end": periods, "eps": np.arange(16, dtype=float) ** 1.5})
    out = time_series_sue(eps, window=8, min_obs=4).set_index("period_end")
    e = eps.set_index("period_end")["eps"]
    d4 = e - e.shift(4)
    q = periods[13]
    expected = d4.loc[q] / d4.shift(1).rolling(8, min_periods=4).std().loc[q]
    assert out.loc[q, "sue"] == pytest.approx(expected)
    assert out.loc[q, "beat"] == 1.0
    assert out["sue"].iloc[:8].isna().all()  # needs 4 YoY changes before it


def _setup():
    cal = pd.bdate_range("2021-01-04", periods=60)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 4)), axis=0)),
                         index=cal, columns=["A", "B", "C", "SPY"])
    events = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "announce_date": [cal[10], cal[10], cal[12]],
        "sue": [2.0, -1.0, 0.5],
        "beat": [1.0, 0.0, 1.0],
    })
    return cal, close, events


def test_features_appear_two_trading_days_after_filing():
    cal, close, events = _setup()
    panel = earnings_feature_panel(events, close, delay=2)
    sue = panel["sue"].unstack()
    assert (sue.loc[: cal[11]] == 0).all().all()          # nothing public yet
    assert sue.loc[cal[12], "A"] > sue.loc[cal[12], "B"]   # A and B public at D + 2
    assert sue.loc[cal[13], "C"] == 0                     # C filed later
    assert set(panel.columns) == set(EARNINGS_FEATURES)


def test_announcement_return_uses_close_before_filing_to_availability():
    cal, close, events = _setup()
    panel = earnings_feature_panel(events[events.symbol != "C"], close, delay=2)
    ear = panel["earnings_ann_return"].unstack().loc[cal[12]]
    a = (close["A"].iloc[12] / close["A"].iloc[9] - 1) - (close["SPY"].iloc[12] / close["SPY"].iloc[9] - 1)
    b = (close["B"].iloc[12] / close["B"].iloc[9] - 1) - (close["SPY"].iloc[12] / close["SPY"].iloc[9] - 1)
    assert (ear["A"] > ear["B"]) == (a > b)


def test_earnings_panel_is_causal():
    cal, close, events = _setup()
    base = earnings_feature_panel(events, close)
    moved = close.copy()
    moved.iloc[40:] *= 1.7
    later = pd.concat([events, pd.DataFrame({"symbol": ["A"], "announce_date": [cal[45]],
                                             "sue": [-3.0], "beat": [0.0]})])
    pert = earnings_feature_panel(later, moved)
    cut = cal[39]
    pd.testing.assert_frame_equal(base.loc[:cut], pert.loc[:cut])
