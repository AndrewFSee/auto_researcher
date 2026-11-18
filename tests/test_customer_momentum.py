"""Tests for the customer momentum test's timing, signal and portfolio rules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from customer_momentum import fama_macbeth, links_at, monthly, own_slopes, portfolio, residual_signal, signals  # noqa: E402


def _filing(cik, filed, customers, ticker="SUP"):
    return {"cik": cik, "supplier": ticker, "price_ticker": ticker, "accession": f"{cik}-{filed}", "filed": filed,
            "customers": customers, "source": {t: "text" for t in customers}}


def test_monthly_returns_start_after_listing_and_stop_at_delisting():
    ends = pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29"])
    tri = pd.Series([1.0, 1.1, 1.21, 1.0], index=pd.to_datetime(["2020-02-10", "2020-02-28", "2020-03-31", "2020-04-15"]))
    r = monthly(tri, ends)
    assert np.isnan(r.iloc[1])                       # listed mid-February: no full month
    assert abs(r.iloc[2] - 0.10) < 1e-12
    assert abs(r.iloc[3] - (1.0 / 1.21 - 1)) < 1e-12  # delisted mid-April: return to the last close
    assert np.isnan(r.iloc[4])                       # then cash


def test_links_start_after_filing_are_replaced_by_the_next_10k_and_expire():
    ends = pd.DatetimeIndex(pd.date_range("2020-01-31", "2021-12-31", freq="ME"))
    filings = [_filing(1, "2020-02-29", {"WMT": 20.0}), _filing(1, "2021-02-15", {}),
               _filing(2, "2020-03-10", {"AAPL": 10.0}, ticker="OTH")]
    live = links_at(filings, ends)
    assert 1 not in live[pd.Timestamp("2020-01-31")]
    assert 1 not in live[pd.Timestamp("2020-02-29")]   # filed on the month-end: counts from the next one
    assert live[pd.Timestamp("2020-03-31")][1]["customers"] == {"WMT": 20.0}
    assert 1 not in live[pd.Timestamp("2021-02-28")]   # the next 10-K has no links: the old ones end
    assert 2 in live[pd.Timestamp("2021-05-31")]       # within 15 months of 2020-03-10 (to 2021-06-10) ...
    assert 2 not in live[pd.Timestamp("2021-06-30")]   # ... and no longer


def test_signal_is_share_weighted_and_portfolio_takes_thirds_with_costs():
    ends = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31"])
    n = 30
    sup = [f"S{i:02d}" for i in range(n)]
    rets = pd.DataFrame(0.0, index=ends, columns=sup + ["A", "B"])
    rets.loc[ends[1], "A"], rets.loc[ends[1], "B"] = 0.10, -0.20
    filings = [_filing(i, "2020-01-15", {"A": 30.0, "B": 10.0} if i < 10 else {"B": 10.0} if i < 20 else {"A": 10.0},
                       ticker=s) for i, s in enumerate(sup)]
    for i, s in enumerate(sup):
        rets.loc[ends[2], s] = 0.01 * i
    live = links_at(filings, ends)
    sig = signals(live, rets, ends)
    feb = sig[sig["month"] == ends[1]].set_index("supplier")
    assert abs(feb.loc["S00", "cm"] - (30 * 0.10 + 10 * -0.20) / 40) < 1e-12
    port = portfolio(sig[sig["month"] == ends[1]], 3)
    row = port.iloc[0]
    assert row["k"] == 10
    # Long the A-only suppliers (S20-S29), short the B-only ones (S10-S19).
    assert abs(row["gross"] - (np.mean([0.01 * i for i in range(20, 30)]) - np.mean([0.01 * i for i in range(10, 20)]))) < 1e-12
    assert abs(row["turnover"] - 2.0) < 1e-12 and abs(row["net"] - (row["gross"] - 0.002)) < 1e-12
    assert portfolio(sig[(sig["month"] == ends[1]) & (sig["supplier"] < "S29")], 3).empty  # under 30 suppliers


def test_residual_sort_removes_the_own_return_and_the_regression_recovers_known_slopes():
    rng = np.random.default_rng(0)
    rows = []
    for m in pd.date_range("2020-01-31", periods=6, freq="ME"):
        own = rng.normal(0, 0.05, 40)
        cm = 0.5 * own + rng.normal(0, 0.03, 40)
        fwd = 0.2 * cm - 0.1 * own + 0.01
        rows += [{"month": m, "hold": m + pd.offsets.MonthEnd(1), "supplier": f"S{i:02d}", "cm": c, "own": o,
                  "fwd": f, "customers": "A"} for i, (c, o, f) in enumerate(zip(cm, own, fwd))]
    sig = pd.DataFrame(rows)
    res = residual_signal(sig)
    for _, g in res.groupby("month"):
        assert abs(g["cm"].mean()) < 1e-12 and abs(np.corrcoef(g["cm"], g["own"])[0, 1]) < 1e-9
    b, b0 = fama_macbeth(sig)
    assert np.allclose(b, 0.2) and np.allclose(own_slopes(sig), -0.1)
    assert (b0 < 0.2).all()  # without the control the reversal pulls the slope down
    assert len(portfolio(res, 3)) == 6
