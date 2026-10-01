"""Tests for the FMP earnings downloader (no network: a fake session is used)."""

from __future__ import annotations

import json

import pytest

from auto_researcher.data.fmp_earnings import (
    consensus_surprises,
    download_fmp_earnings,
    load_fmp_earnings,
    parse_fmp_earnings,
)

ROWS = [
    {"symbol": "AAA", "date": "2024-01-25", "epsActual": 1.10, "epsEstimated": 1.00,
     "revenueActual": 5.0e9, "revenueEstimated": 4.9e9, "lastUpdated": "2024-02-01"},
    {"symbol": "AAA", "date": "2011-10-20", "epsActual": 0.68, "epsEstimated": 0.68,
     "revenueActual": 1.7e10, "revenueEstimated": 1.7e10, "lastUpdated": "2025-04-24"},
    {"symbol": "AAA", "date": "2026-10-28", "epsActual": None, "epsEstimated": 1.2,
     "revenueActual": None, "revenueEstimated": 6e9, "lastUpdated": "2026-10-01"},
]


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeSession:
    """Serves scripted responses per symbol and counts calls."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def get(self, url, params, timeout):
        self.calls.append(params["symbol"])
        return self.script(params["symbol"], len(self.calls))


def test_parse_flags_backfilled_rows():
    df = parse_fmp_earnings(ROWS, "AAA")
    assert list(df["date"].dt.year) == [2011, 2024, 2026]
    assert df.set_index(df["date"].dt.year).loc[2011, "suspect_backfill"]
    assert not df.set_index(df["date"].dt.year).loc[2024, "suspect_backfill"]


def test_surprises_skip_backfilled_and_unreported_quarters():
    s = consensus_surprises(parse_fmp_earnings(ROWS, "AAA"))
    assert len(s) == 1
    assert s["sue"].iloc[0] == pytest.approx(0.10)


def test_download_caches_resumes_and_stops_at_quota(repo_tmp_path):
    def script(sym, n):
        if n > 2:
            return FakeResponse(429, {"Error Message": "Limit Reach . Please upgrade your plan"})
        return FakeResponse(200, [dict(ROWS[0], symbol=sym)])

    s1 = download_fmp_earnings(["A", "B", "C", "D"], "k", repo_tmp_path, pause=0, session=FakeSession(script))
    assert s1.downloaded == ["A", "B"] and s1.quota_hit and s1.remaining == ["C", "D"]

    ok = FakeSession(lambda sym, n: FakeResponse(200, [dict(ROWS[0], symbol=sym)]))
    s2 = download_fmp_earnings(["A", "B", "C", "D"], "k", repo_tmp_path, pause=0, session=ok)
    assert ok.calls == ["C", "D"] and s2.already_cached == 2
    assert set(load_fmp_earnings(repo_tmp_path)["symbol"]) == {"A", "B", "C", "D"}


def test_unavailable_symbols_are_remembered(repo_tmp_path):
    def script(sym, n):
        if sym == "X":
            return FakeResponse(402, {"Error Message": "Premium Query Parameter: not on your plan"})
        return FakeResponse(200, [])

    download_fmp_earnings(["X", "Y"], "k", repo_tmp_path, pause=0, session=FakeSession(script))
    assert json.loads((repo_tmp_path / "_unavailable.json").read_text()) == ["X"]
    again = FakeSession(script)
    download_fmp_earnings(["X", "Y"], "k", repo_tmp_path, pause=0, session=again)
    assert again.calls == []


def test_max_requests_budget(repo_tmp_path):
    sess = FakeSession(lambda sym, n: FakeResponse(200, []))
    s = download_fmp_earnings(list("ABCDE"), "k", repo_tmp_path, max_requests=3, pause=0, session=sess)
    assert len(sess.calls) == 3 and s.remaining == ["D", "E"]
