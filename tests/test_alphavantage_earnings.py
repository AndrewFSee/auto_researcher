"""Tests for the Alpha Vantage earnings downloader (no network: a fake session is used)."""

from __future__ import annotations

import json

import pytest

from auto_researcher.data.alphavantage_earnings import download_av_earnings, parse_av_earnings
from auto_researcher.data.fmp_earnings import consensus_surprises, load_fmp_earnings

QUARTERS = [
    {"fiscalDateEnding": "2024-03-31", "reportedDate": "2024-05-02", "reportedEPS": "1.53",
     "estimatedEPS": "1.5", "surprise": "0.03", "surprisePercentage": "2", "reportTime": "post-market"},
    {"fiscalDateEnding": "2005-03-31", "reportedDate": "2005-04-13", "reportedEPS": "0.05",
     "estimatedEPS": "0.05", "surprise": "0", "surprisePercentage": "0"},
    {"fiscalDateEnding": "2015-03-31", "reportedDate": "2015-04-27", "reportedEPS": "0.58",
     "estimatedEPS": "0.58", "surprise": "0", "surprisePercentage": "0"},
    {"fiscalDateEnding": "1996-03-31", "reportedDate": "1996-04-17", "reportedEPS": "-0.03",
     "estimatedEPS": "None", "surprise": "None", "surprisePercentage": "None"},
]
PAYLOAD = {"symbol": "AAA", "quarterlyEarnings": QUARTERS}
THROTTLE = {"Information": "Thank you for using Alpha Vantage! ... 25 requests per day ..."}


class FakeResponse:
    def __init__(self, body):
        self.status_code, self._body = 200, body

    def json(self):
        return self._body


class FakeSession:
    """Serves scripted responses per symbol and counts calls."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def get(self, url, params, timeout):
        assert params["function"] == "EARNINGS"
        self.calls.append(params["symbol"])
        return FakeResponse(self.script(params["symbol"], len(self.calls)))


def test_parse_normalizes_and_flags_early_exact_matches():
    df = parse_av_earnings(PAYLOAD, "AAA")
    assert list(df["date"].dt.year) == [1996, 2005, 2015, 2024]
    flags = dict(zip(df["date"].dt.year, df["suspect_backfill"]))
    # Exact matches are suspect only before 2010; a missing estimate is not a match.
    assert flags == {1996: False, 2005: True, 2015: False, 2024: False}
    assert df["eps_estimated"].isna().sum() == 1


def test_surprises_use_shared_definition():
    s = consensus_surprises(parse_av_earnings(PAYLOAD, "AAA"))
    assert len(s) == 2  # 2015 (zero surprise) and 2024; 1996 lacks an estimate, 2005 is flagged
    assert s["sue"].iloc[-1] == pytest.approx(0.02)


def test_throttle_retry_then_quota_stop_and_resume(repo_tmp_path, monkeypatch):
    monkeypatch.setattr("auto_researcher.data.alphavantage_earnings.time.sleep", lambda s: None)

    # Call 2 is a per-second throttle (retried successfully); calls 4+ are the daily limit.
    def script(sym, n):
        return THROTTLE if n == 2 or n >= 4 else PAYLOAD

    sess = FakeSession(script)
    s1 = download_av_earnings(list("ABCD"), "k", repo_tmp_path, pause=0, session=sess)
    assert sess.calls == ["A", "B", "B", "C", "C"]
    assert s1.downloaded == ["A", "B"] and s1.quota_hit and s1.remaining == ["C", "D"]
    assert s1.unavailable == []

    ok = FakeSession(lambda sym, n: PAYLOAD)
    s2 = download_av_earnings(list("ABCD"), "k", repo_tmp_path, pause=0, session=ok)
    assert ok.calls == ["C", "D"] and s2.already_cached == 2
    assert set(load_fmp_earnings(repo_tmp_path)["symbol"]) == set("ABCD")


def test_unknown_symbols_are_remembered(repo_tmp_path, monkeypatch):
    monkeypatch.setattr("auto_researcher.data.alphavantage_earnings.time.sleep", lambda s: None)

    def script(sym, n):
        return {} if sym == "X" else PAYLOAD

    download_av_earnings(["X", "Y"], "k", repo_tmp_path, pause=0, session=FakeSession(script))
    assert json.loads((repo_tmp_path / "_unavailable.json").read_text()) == ["X"]
    again = FakeSession(script)
    download_av_earnings(["X", "Y"], "k", repo_tmp_path, pause=0, session=again)
    assert again.calls == []


def test_max_requests_budget(repo_tmp_path, monkeypatch):
    monkeypatch.setattr("auto_researcher.data.alphavantage_earnings.time.sleep", lambda s: None)
    sess = FakeSession(lambda sym, n: PAYLOAD)
    s = download_av_earnings(list("ABCDE"), "k", repo_tmp_path, max_requests=3, pause=0, session=sess)
    assert len(sess.calls) == 3 and s.remaining == ["D", "E"]


def test_spare_budget_refreshes_oldest_stale_files(repo_tmp_path, monkeypatch):
    import os

    monkeypatch.setattr("auto_researcher.data.alphavantage_earnings.time.sleep", lambda s: None)
    download_av_earnings(list("ABC"), "k", repo_tmp_path, pause=0,
                         session=FakeSession(lambda sym, n: PAYLOAD))
    day = 86400
    for sym, age in (("A", 40), ("B", 60), ("C", 5)):
        path = repo_tmp_path / f"{sym}.parquet"
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime - age * day))

    sess = FakeSession(lambda sym, n: PAYLOAD)
    s = download_av_earnings(list("ABCD"), "k", repo_tmp_path, max_requests=2, pause=0,
                             session=sess, refresh_after_days=30)
    # New symbol first, then the oldest stale file; C is fresh, A is over budget.
    assert sess.calls == ["D", "B"]
    assert s.remaining == [] and s.downloaded == ["D", "B"]
