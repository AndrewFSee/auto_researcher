"""Tests for the analyst rating downloaders and their timing rules (no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from auto_researcher.data.analyst_ratings import (
    download_fmp_grades,
    download_yahoo_ratings,
    first_reacting_session,
    load_ratings,
    normalize_firm,
    parse_fmp_grades,
    parse_yahoo_ratings,
    session_bucket,
)


def yahoo_frame(times_utc: list[str], **cols) -> pd.DataFrame:
    """A frame shaped like yfinance's ``upgrades_downgrades`` (naive UTC index)."""
    n = len(times_utc)
    data = {"Firm": ["Morgan Stanley"] * n, "ToGrade": ["Buy"] * n, "FromGrade": ["Hold"] * n,
            "Action": ["up"] * n, "priceTargetAction": ["Raises"] * n,
            "currentPriceTarget": [200.0] * n, "priorPriceTarget": [180.0] * n}
    data.update(cols)
    return pd.DataFrame(data, index=pd.DatetimeIndex(pd.to_datetime(times_utc), name="GradeDate"))


def test_yahoo_times_become_new_york_time_and_dates():
    raw = yahoo_frame(["2024-03-01 03:00:00", "2024-07-15 12:00:00"],
                      currentPriceTarget=[0.0, 250.0])
    df = parse_yahoo_ratings(raw, "AAA")
    # 03:00 UTC on Mar 1 is 22:00 the evening before in New York (EST, UTC-5)
    assert df["timestamp_et"].iloc[0] == pd.Timestamp("2024-02-29 22:00")
    assert df["date"].iloc[0] == pd.Timestamp("2024-02-29")
    # summer: EDT, UTC-4
    assert df["timestamp_et"].iloc[1] == pd.Timestamp("2024-07-15 08:00")
    assert pd.isna(df["price_target"].iloc[0])  # Yahoo's 0 means "no target"
    assert df["price_target"].iloc[1] == 250.0
    assert list(df["source"].unique()) == ["yahoo"]


def test_session_buckets_follow_market_hours():
    ts = pd.Series(pd.to_datetime(["2024-07-15 09:29", "2024-07-15 09:30", "2024-07-15 15:59",
                                   "2024-07-15 16:00", None]))
    assert list(session_bucket(ts)) == ["pre-market", "session", "session", "after close", "date only"]


def test_first_reacting_session():
    cal = pd.DatetimeIndex(["2024-07-11", "2024-07-12", "2024-07-15", "2024-07-16"])  # Thu Fri Mon Tue
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-07-12", "2024-07-12", "2024-07-13", "2024-07-15", "2024-07-15"]),
        "timestamp_et": pd.to_datetime(["2024-07-12 07:00", "2024-07-12 17:30", "2024-07-13 10:00",
                                        "2024-07-15 11:00", None]),
    })
    got = [cal[i] for i in first_reacting_session(df, cal)]
    assert got == [pd.Timestamp("2024-07-12"),   # Friday pre-market -> Friday
                   pd.Timestamp("2024-07-15"),   # Friday after the close -> Monday
                   pd.Timestamp("2024-07-15"),   # Saturday -> Monday
                   pd.Timestamp("2024-07-15"),   # Monday during the session -> Monday
                   pd.Timestamp("2024-07-15")]   # date only (FMP) -> that day


def test_fmp_actions_are_normalized():
    df = parse_fmp_grades([
        {"symbol": "AAPL", "date": "2024-05-03", "gradingCompany": "Wedbush",
         "previousGrade": "Neutral", "newGrade": "Outperform", "action": "upgrade"},
        {"symbol": "AAPL", "date": "2024-05-02", "gradingCompany": "Citi",
         "previousGrade": "Buy", "newGrade": "Buy", "action": "maintain"},
    ], "AAPL")
    assert list(df["action"]) == ["main", "up"]  # sorted by date
    assert df["timestamp_et"].isna().all() and list(df["source"].unique()) == ["fmp"]


def test_normalize_firm_matches_broker_names_across_sources():
    assert normalize_firm("Morgan Stanley & Co. LLC") == normalize_firm("Morgan Stanley") == "morgan stanley"
    assert normalize_firm("Goldman Sachs Group") == "goldman sachs"


class RateLimited(Exception):
    pass


RateLimited.__name__ = "YFRateLimitError"


def test_yahoo_download_resumes_after_rate_limit_and_remembers_empty(repo_tmp_path):
    calls = []

    def fetch(sym):
        calls.append(sym)
        if sym == "C" and calls.count("C") == 1:
            raise RateLimited("Too Many Requests")
        return pd.DataFrame() if sym == "B" else yahoo_frame(["2024-07-15 12:00:00"])

    s1 = download_yahoo_ratings(list("ABCD"), repo_tmp_path, pause=0, fetch=fetch)
    assert s1.downloaded == ["A"] and s1.unavailable == ["B"] and s1.quota_hit and s1.remaining == ["C", "D"]
    s2 = download_yahoo_ratings(list("ABCD"), repo_tmp_path, pause=0, fetch=fetch)
    assert s2.downloaded == ["C", "D"] and calls[-2:] == ["C", "D"]  # B not asked again
    assert json.loads((repo_tmp_path / "_empty.json").read_text()) == ["B"]
    download_yahoo_ratings(list("ABCD"), repo_tmp_path, pause=0, fetch=fetch, retry_empty=True)
    assert calls[-1] == "B"
    assert sorted(load_ratings(repo_tmp_path)["symbol"].unique()) == ["A", "C", "D"]


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return self._body


class FakeSession:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def get(self, url, params, timeout):
        self.calls.append(params["symbol"])
        return self.replies[params["symbol"]]


def test_fmp_download_records_refusals_and_stops_at_the_limit(repo_tmp_path):
    rows = [{"symbol": "AAPL", "date": "2024-05-03", "gradingCompany": "Wedbush",
             "previousGrade": "Neutral", "newGrade": "Outperform", "action": "upgrade"}]
    session = FakeSession({
        "AAPL": FakeResponse(200, rows),
        "SYK": FakeResponse(402, "Premium Query Parameter: 'Special Endpoint ..."),
        "MSFT": FakeResponse(429, "Limit Reach"),
    })
    s = download_fmp_grades(["AAPL", "SYK", "MSFT"], "k", repo_tmp_path, pause=0, session=session)
    assert s.downloaded == ["AAPL"] and s.unavailable == ["SYK"] and s.quota_hit and s.remaining == ["MSFT"]
    again = FakeSession(session.replies)
    download_fmp_grades(["AAPL", "SYK", "MSFT"], "k", repo_tmp_path, pause=0, session=again)
    assert again.calls == ["MSFT"]  # cached and refused symbols are not asked again


@pytest.mark.parametrize("utc,bucket", [("2024-01-10 14:00:00", "pre-market"),   # 09:00 EST
                                        ("2024-01-10 15:00:00", "session"),      # 10:00 EST
                                        ("2024-01-10 21:30:00", "after close")])  # 16:30 EST
def test_parsed_yahoo_times_feed_the_buckets(utc, bucket):
    df = parse_yahoo_ratings(yahoo_frame([utc]), "AAA")
    assert session_bucket(df["timestamp_et"]).iloc[0] == bucket
