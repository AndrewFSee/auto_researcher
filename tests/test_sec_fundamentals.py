"""Tests for point-in-time SEC fundamentals and valuation metrics (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_researcher.data.sec_fundamentals import (
    build_pit_table,
    download_company_facts,
    extract_facts,
    fundamentals_asof,
    select_reported,
)
from auto_researcher.features.valuation import (
    dcf_value,
    implied_growth,
    split_factor_after,
    valuation_metrics,
)


def fact(concept, start, end, val, filed, form="10-Q", unit="USD", taxonomy="us-gaap"):
    return taxonomy, concept, unit, {"start": start, "end": end, "val": val, "filed": filed,
                                     "form": form, "accn": f"a-{filed}"}


def payload(*entries):
    facts: dict = {}
    for taxonomy, concept, unit, e in entries:
        e = {k: v for k, v in e.items() if v is not None}
        facts.setdefault(taxonomy, {}).setdefault(concept, {"units": {}})["units"].setdefault(unit, []).append(e)
    return {"facts": facts}


CFO = "NetCashProvidedByUsedInOperatingActivities"


def calendar_year_cfo():
    """Cash flow reported year-to-date only, as in real 10-Qs."""
    return [
        fact(CFO, "2022-01-01", "2022-06-30", 190.0, "2022-08-01"),
        fact(CFO, "2022-01-01", "2022-12-31", 400.0, "2023-02-01", form="10-K"),
        fact(CFO, "2023-01-01", "2023-03-31", 120.0, "2023-05-01"),
        fact(CFO, "2023-01-01", "2023-06-30", 210.0, "2023-08-01"),
        fact(CFO, "2022-01-01", "2022-03-31", 100.0, "2022-05-01"),
    ]


def test_extract_keeps_periodic_reports_and_known_concepts():
    p = payload(*calendar_year_cfo(),
                fact(CFO, "2023-01-01", "2023-06-30", 999.0, "2023-08-02", form="8-K"),
                fact("SomethingElse", "2023-01-01", "2023-06-30", 1.0, "2023-08-01"))
    df = extract_facts(p, "AAA")
    assert set(df["item"]) == {"operating_cash_flow"} and len(df) == 5
    assert df["filed"].dtype == "datetime64[ns]"


def test_ttm_from_year_to_date_figures():
    table = build_pit_table(extract_facts(payload(*calendar_year_cfo()), "AAA"))
    ttm = table.set_index("period_end")["value"]
    assert ttm[pd.Timestamp("2022-12-31")] == 400.0
    # H1 2023: 210 + 400 - 190; Q1 2023: 120 + 400 - 100
    assert ttm[pd.Timestamp("2023-06-30")] == pytest.approx(420.0)
    assert ttm[pd.Timestamp("2023-03-31")] == pytest.approx(420.0)
    row = table.set_index("period_end").loc[pd.Timestamp("2023-06-30")]
    assert row["available_date"] == pd.Timestamp("2023-08-01")


def test_ttm_with_non_calendar_fiscal_year():
    rev = "Revenues"
    p = payload(
        fact(rev, "2021-10-01", "2022-03-31", 50.0, "2022-05-01"),
        fact(rev, "2021-10-01", "2022-09-30", 110.0, "2022-11-15", form="10-K"),
        fact(rev, "2022-10-01", "2023-03-31", 70.0, "2023-05-01"),
    )
    table = build_pit_table(extract_facts(p, "AAA"))
    assert table.set_index("period_end")["value"][pd.Timestamp("2023-03-31")] == pytest.approx(130.0)


def test_first_reported_ignores_later_restatement():
    p = payload(
        fact("Assets", None, "2022-12-31", 1000.0, "2023-02-01", form="10-K"),
        fact("Assets", None, "2022-12-31", 900.0, "2024-02-01", form="10-K"),  # restated comparative
    )
    facts = extract_facts(p, "AAA")
    assert select_reported(facts, "first")["value"].tolist() == [1000.0]
    assert select_reported(facts, "last")["value"].tolist() == [900.0]
    table = build_pit_table(facts)
    assert table["available_date"].tolist() == [pd.Timestamp("2023-02-01")]


def test_same_filing_prefers_higher_priority_concept():
    p = payload(
        fact("RevenueFromContractWithCustomerExcludingAssessedTax", "2022-01-01", "2022-12-31",
             95.0, "2023-02-01", form="10-K"),
        fact("Revenues", "2022-01-01", "2022-12-31", 100.0, "2023-02-01", form="10-K"),
    )
    assert build_pit_table(extract_facts(p, "AAA"))["value"].tolist() == [100.0]


def test_asof_is_strictly_after_filing_and_latest_period_wins():
    table = pd.DataFrame({
        "symbol": "AAA", "item": "total_assets",
        "period_end": pd.to_datetime(["2023-03-31", "2023-06-30", "2022-12-31"]),
        "available_date": pd.to_datetime(["2023-05-01", "2023-08-01", "2023-09-01"]),
        "value": [1.0, 2.0, 99.0],  # the last row is an old period first tagged late
    })
    dates = pd.to_datetime(["2023-05-01", "2023-05-02", "2023-08-01", "2023-08-02", "2023-09-05"])
    got = fundamentals_asof(table, dates)["total_assets"].to_numpy()
    np.testing.assert_array_equal(got, [np.nan, 1.0, 1.0, 2.0, 2.0])
    stale = fundamentals_asof(table, pd.to_datetime(["2024-12-31"]), max_age_days=400)
    assert stale["total_assets"].isna().all()


def test_download_caches_and_records_unknown_tickers(repo_tmp_path):
    import json

    class Resp:
        def __init__(self, status, body):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

        def raise_for_status(self):
            pass

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, headers, timeout):
            assert "User-Agent" in headers
            self.urls.append(url)
            if url.endswith("company_tickers.json"):
                return Resp(200, {"0": {"ticker": "AAA", "cik_str": 1}, "1": {"ticker": "BBB", "cik_str": 2}})
            if "CIK0000000002" in url:
                return Resp(404, {})
            return Resp(200, payload(*calendar_year_cfo()))

    s = download_company_facts(["AAA", "BBB", "ZZZ"], repo_tmp_path, "test agent", Session(), pause=0)
    assert s.downloaded == ["AAA"] and sorted(s.unavailable) == ["BBB", "ZZZ"]
    assert json.loads((repo_tmp_path / "_unavailable.json").read_text()) == ["BBB", "ZZZ"]
    again = Session()
    download_company_facts(["AAA", "BBB", "ZZZ"], repo_tmp_path, "test agent", again, pause=0)
    assert not any("companyfacts" in u for u in again.urls)


def test_split_factor_and_market_cap():
    days = pd.bdate_range("2020-06-01", "2020-12-31")
    splits = pd.DataFrame(0.0, index=days, columns=["AAA"])
    splits.loc[pd.Timestamp("2020-08-31"), "AAA"] = 4.0
    f = split_factor_after(splits, pd.Series(["AAA", "AAA", "BBB"]),
                           pd.Series(pd.to_datetime(["2020-07-01", "2020-09-01", "2020-07-01"])))
    np.testing.assert_array_equal(f, [4.0, 1.0, 1.0])

    # 100 shares reported before a 4-for-1 split; split-adjusted price 25 throughout
    # (raw price 100 before, 25 after): true market cap is 10,000 on both dates.
    close = pd.DataFrame(25.0, index=days, columns=["AAA"])
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-07-15", "2020-12-15"]), ["AAA"]],
                                     names=["date", "symbol"])
    fund = pd.DataFrame({
        "shares_cover": 100.0, "shares_cover_available": pd.Timestamp("2020-07-01"),
        "shares_diluted": np.nan, "shares_diluted_available": pd.NaT,
        "operating_cash_flow": 500.0, "capex": 100.0, "interest_expense": 0.0,
        "gross_profit": 300.0, "revenue": 1000.0, "total_assets": 2000.0,
        "operating_income": 250.0, "net_income": 200.0, "equity": 1500.0,
        "debt_noncurrent": 1000.0, "debt_current": 0.0, "cash": 500.0,
    }, index=idx)
    m = valuation_metrics(fund, close, splits)
    np.testing.assert_allclose(m["market_cap"], [10_000.0, 10_000.0])
    np.testing.assert_allclose(m["enterprise_value"], [10_500.0, 10_500.0])
    np.testing.assert_allclose(m["fcf_yield"], [0.04, 0.04])
    np.testing.assert_allclose(m["accruals"], [-0.15, -0.15])


def test_cover_shares_fall_back_to_diluted_when_stale():
    days = pd.bdate_range("2021-01-01", "2021-12-31")
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2021-12-15"), "AAA")], names=["date", "symbol"])
    fund = pd.DataFrame({
        "shares_cover": 100.0, "shares_cover_available": pd.Timestamp("2021-01-15"),
        "shares_diluted": 120.0, "shares_diluted_available": pd.Timestamp("2021-11-01"),
        "operating_cash_flow": 1.0, "capex": 0.0, "gross_profit": 1.0, "revenue": 1.0,
        "total_assets": 1.0, "operating_income": 1.0, "net_income": 1.0,
    }, index=idx)
    m = valuation_metrics(fund, pd.DataFrame(10.0, index=days, columns=["AAA"]),
                          pd.DataFrame(index=days))
    assert m["market_cap"].iloc[0] == 1200.0


def test_reverse_dcf_round_trip_and_edge_cases():
    value = dcf_value(100.0, 0.07, 0.09)
    assert implied_growth(value, 100.0, 0.09) == pytest.approx(0.07, abs=1e-5)
    # Zero growth with no terminal growth is a perpetuity: value = fcf / r.
    assert dcf_value(100.0, 0.0, 0.10, terminal_growth=0.0) == pytest.approx(1000.0)
    assert np.isnan(implied_growth(value, -5.0, 0.09))
    assert np.isnan(implied_growth(1e15, 100.0, 0.09))  # beyond the 100% bound
    with pytest.raises(ValueError):
        dcf_value(100.0, 0.05, 0.02)


def test_pipeline_is_causal_in_filing_dates():
    """Changing anything filed on or after T leaves every value known before T unchanged."""
    rng = np.random.default_rng(0)
    entries = []
    for year in range(2015, 2024):
        filed_fy = f"{year + 1}-02-15"
        entries.append(fact(CFO, f"{year}-01-01", f"{year}-12-31", float(rng.uniform(300, 500)),
                            filed_fy, form="10-K"))
        entries.append(fact("Assets", None, f"{year}-12-31", float(rng.uniform(900, 1100)),
                            filed_fy, form="10-K"))
        for q, end in ((1, "03-31"), (2, "06-30"), (3, "09-30")):
            filed = f"{year}-{3 * q + 2:02d}-01"
            entries.append(fact(CFO, f"{year}-01-01", f"{year}-{end}", float(rng.uniform(80, 120) * q),
                                filed))
            entries.append(fact("Assets", None, f"{year}-{end}", float(rng.uniform(900, 1100)), filed))
    facts = extract_facts(payload(*entries), "AAA")
    cutoff = pd.Timestamp("2020-01-01")
    later = facts["filed"] >= cutoff
    perturbed = facts.copy()
    perturbed.loc[later, "value"] *= rng.uniform(0.5, 1.5, later.sum())
    # Restatements filed after the cutoff for periods before it must not leak back either.
    restated = facts[~later].assign(filed=pd.Timestamp("2021-06-01"), value=lambda d: d["value"] * 3,
                                    accn="restatement")
    perturbed = pd.concat([perturbed, restated], ignore_index=True)

    dates = pd.bdate_range("2016-01-01", cutoff)
    a = fundamentals_asof(build_pit_table(facts), dates)
    b = fundamentals_asof(build_pit_table(perturbed), dates)
    pd.testing.assert_frame_equal(a, b)
    assert a["operating_cash_flow"].notna().mean() > 0.9


def test_predecessor_registrant_history_is_merged(repo_tmp_path, monkeypatch):
    import auto_researcher.data.sec_fundamentals as sf

    monkeypatch.setitem(sf.PREDECESSOR_CIKS, "NEWCO", (7,))

    class Resp:
        def __init__(self, body):
            self.status_code, self._body = 200, body

        def json(self):
            return self._body

        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, headers, timeout):
            if url.endswith("company_tickers.json"):
                return Resp({"0": {"ticker": "NEWCO", "cik_str": 9}})
            if "CIK0000000007" in url:  # old registrant: the history
                return Resp(payload(*calendar_year_cfo()[:2]))
            return Resp(payload(*calendar_year_cfo()[2:]))

    s = download_company_facts(["NEWCO"], repo_tmp_path, "test agent", Session(), pause=0)
    assert s.downloaded == ["NEWCO"]
    table = build_pit_table(pd.read_parquet(repo_tmp_path / "NEWCO.parquet"))
    # The 2023 TTM needs the 2022 figures that only the old registrant filed.
    assert table.set_index("period_end")["value"][pd.Timestamp("2023-06-30")] == pytest.approx(420.0)


def test_debt_total_used_only_when_no_split_is_tagged():
    days = pd.bdate_range("2021-01-01", "2021-12-31")
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2021-12-15")], ["SPLIT", "TOTAL", "NONE"]],
                                     names=["date", "symbol"])
    fund = pd.DataFrame({
        "shares_cover": 10.0, "shares_cover_available": pd.Timestamp("2021-11-01"),
        "shares_diluted": np.nan, "shares_diluted_available": pd.NaT,
        "operating_cash_flow": 1.0, "capex": 0.0, "gross_profit": 1.0, "revenue": 1.0,
        "total_assets": 1.0, "operating_income": 1.0, "net_income": 1.0,
        "debt_noncurrent": [30.0, np.nan, np.nan], "debt_current": [5.0, np.nan, np.nan],
        "debt_total": [999.0, 50.0, np.nan],
    }, index=idx)
    close = pd.DataFrame(10.0, index=days, columns=["SPLIT", "TOTAL", "NONE"])
    m = valuation_metrics(fund, close, pd.DataFrame(index=days)).droplevel("date")
    assert m.loc["SPLIT", "debt"] == 35.0 and m.loc["TOTAL", "debt"] == 50.0
    assert m.loc["NONE", "debt"] == 0.0
    assert m["debt_known"].tolist() == [True, True, False]
