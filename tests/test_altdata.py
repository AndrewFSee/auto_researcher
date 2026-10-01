"""Tests for the alt-data adapter package.

All upstream HTTP / third-party API calls are stubbed. The point is to
verify the adapters' shape, normalization, caching, and graceful-failure
behavior — not to smoke-test the public APIs themselves.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from auto_researcher.data.altdata import (
    AltDataCache,
    SEC8KEventAdapter,
    WikipediaPageviewsAdapter,
    zscore_panel,
)
from auto_researcher.data.altdata.base import FetchWindow, empty_altdata_series


# ---------------------------------------------------------------------------
# repo-local tmp fixture (work around broken %TEMP% on this machine)
# ---------------------------------------------------------------------------

@pytest.fixture
def repo_tmp_path():
    base = Path(__file__).parent.parent / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="altdata_", dir=str(base)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stub HTTP session — records GET calls, returns scripted responses
# ---------------------------------------------------------------------------

@dataclass
class _StubResponse:
    status_code: int = 200
    _json: Any = None
    url: str = ""

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


@dataclass
class _StubSession:
    """Map from URL-substring → _StubResponse."""

    rules: list[tuple[str, _StubResponse]] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def get(self, url: str, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        for needle, resp in self.rules:
            if needle in url:
                resp.url = url
                return resp
        # Default: 404
        return _StubResponse(status_code=404, _json={}, url=url)


# ---------------------------------------------------------------------------
# base.py — cache + normalization
# ---------------------------------------------------------------------------

def test_fetchwindow_validates_order():
    with pytest.raises(ValueError):
        FetchWindow.from_inputs("2024-05-01", "2024-04-01")


def test_cache_roundtrip(repo_tmp_path):
    cache = AltDataCache(adapter="test", cache_dir=repo_tmp_path)
    assert cache.get("AAPL", "2024-01-01", "2024-02-01") is None
    cache.put("AAPL", "2024-01-01", "2024-02-01", {"x": 1})
    cached = AltDataCache.unwrap(cache.get("AAPL", "2024-01-01", "2024-02-01"))
    assert cached == {"x": 1}


def test_cache_key_sensitive_to_extra(repo_tmp_path):
    cache = AltDataCache(adapter="t", cache_dir=repo_tmp_path)
    cache.put("AAPL", "a", "b", {"v": 1}, extra="kw1")
    cache.put("AAPL", "a", "b", {"v": 2}, extra="kw2")
    assert AltDataCache.unwrap(cache.get("AAPL", "a", "b", extra="kw1")) == {"v": 1}
    assert AltDataCache.unwrap(cache.get("AAPL", "a", "b", extra="kw2")) == {"v": 2}


def test_zscore_xsec_zeroes_single_value_dates():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"),
         (pd.Timestamp("2024-01-03"), "A"),
         (pd.Timestamp("2024-01-03"), "B")],
        names=["date", "ticker"],
    )
    s = pd.Series([5.0, 1.0, 3.0], index=idx)
    z = zscore_panel(s, mode="zscore_xsec")
    # Single-ticker day → std is 0 → zeroed out by design.
    assert z.loc[(pd.Timestamp("2024-01-02"), "A")] == 0.0
    # Two-ticker day: (1 - 2) / 1.414 and (3 - 2) / 1.414.
    val_a = z.loc[(pd.Timestamp("2024-01-03"), "A")]
    val_b = z.loc[(pd.Timestamp("2024-01-03"), "B")]
    assert val_a == pytest.approx(-val_b)


def test_zscore_panel_raw_noop():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-02"), "A")],
                                    names=["date", "ticker"])
    s = pd.Series([42.0], index=idx)
    out = zscore_panel(s, mode="raw")
    assert out.iloc[0] == 42.0


def test_empty_altdata_series_has_multiindex():
    s = empty_altdata_series("foo")
    assert isinstance(s.index, pd.MultiIndex)
    assert s.index.names == ["date", "ticker"]
    assert len(s) == 0


# ---------------------------------------------------------------------------
# Wikipedia pageviews
# ---------------------------------------------------------------------------

def _wiki_payload(dates_views: list[tuple[str, int]]) -> dict:
    return {
        "items": [
            {"timestamp": ts + "00", "views": v} for ts, v in dates_views
        ],
    }


def test_wikipedia_fetch_parses_and_applies_log1p(repo_tmp_path):
    session = _StubSession(rules=[
        ("Apple_Inc.", _StubResponse(200, _wiki_payload([
            ("20240102", 1000), ("20240103", 2000),
        ]))),
    ])
    adapter = WikipediaPageviewsAdapter(
        cache_dir=repo_tmp_path, session=session,
    )
    out = adapter.fetch(["AAPL"], "2024-01-02", "2024-01-10")
    assert isinstance(out.index, pd.MultiIndex)
    assert len(out) == 2
    assert out.iloc[0] == pytest.approx(np.log1p(1000))
    assert out.iloc[1] == pytest.approx(np.log1p(2000))
    # User-Agent was set on the request.
    assert any("auto-researcher-altdata" in (c["headers"] or {}).get("User-Agent", "")
               for c in session.calls)


def test_wikipedia_unknown_ticker_skipped():
    session = _StubSession(rules=[])
    adapter = WikipediaPageviewsAdapter(cache_dir=None, session=session)
    out = adapter.fetch(["NOT_A_REAL_TICKER"], "2024-01-02", "2024-01-10")
    assert out.empty
    # Should not have hit the network at all.
    assert len(session.calls) == 0


def test_wikipedia_404_returns_empty(repo_tmp_path):
    session = _StubSession(rules=[
        ("Apple_Inc.", _StubResponse(404, {"detail": "not found"})),
    ])
    adapter = WikipediaPageviewsAdapter(cache_dir=repo_tmp_path, session=session)
    out = adapter.fetch(["AAPL"], "2024-01-02", "2024-01-10")
    assert out.empty


def test_wikipedia_cache_avoids_second_network_call(repo_tmp_path):
    session = _StubSession(rules=[
        ("Apple_Inc.", _StubResponse(200, _wiki_payload([("20240102", 500)]))),
    ])
    adapter = WikipediaPageviewsAdapter(cache_dir=repo_tmp_path, session=session)
    adapter.fetch(["AAPL"], "2024-01-02", "2024-01-10")
    first_calls = len(session.calls)

    # Fresh adapter pointing at the same cache dir — zero new calls.
    session2 = _StubSession(rules=[])
    adapter2 = WikipediaPageviewsAdapter(cache_dir=repo_tmp_path, session=session2)
    out = adapter2.fetch(["AAPL"], "2024-01-02", "2024-01-10")
    assert len(session2.calls) == 0
    assert len(out) == 1
    assert first_calls >= 1


def test_wikipedia_custom_ticker_article_map(repo_tmp_path):
    session = _StubSession(rules=[
        ("Beyond_Meat", _StubResponse(200, _wiki_payload([("20240102", 123)]))),
    ])
    adapter = WikipediaPageviewsAdapter(
        ticker_article={"BYND": "Beyond_Meat"},
        cache_dir=repo_tmp_path, session=session,
    )
    out = adapter.fetch(["BYND"], "2024-01-02", "2024-01-10")
    assert len(out) == 1
    assert out.index.get_level_values("ticker")[0] == "BYND"


# ---------------------------------------------------------------------------
# SEC 8-K events
# ---------------------------------------------------------------------------

def _tickers_json(pairs: list[tuple[str, int]]) -> dict:
    return {
        str(i): {"ticker": t, "cik_str": c, "title": f"{t} Corp"}
        for i, (t, c) in enumerate(pairs)
    }


def _submissions_payload(forms_dates_items: list[tuple[str, str, str]]) -> dict:
    return {
        "filings": {
            "recent": {
                "form": [row[0] for row in forms_dates_items],
                "filingDate": [row[1] for row in forms_dates_items],
                "items": [row[2] for row in forms_dates_items],
            }
        }
    }


def test_sec_extracts_8k_with_item_weights(repo_tmp_path):
    session = _StubSession(rules=[
        ("company_tickers.json", _StubResponse(200, _tickers_json([("AAPL", 320193)]))),
        ("CIK0000320193.json", _StubResponse(200, _submissions_payload([
            ("8-K", "2024-02-01", "2.02, 7.01"),
            ("8-K", "2024-02-05", "5.02"),
            ("10-K", "2024-02-10", ""),                    # filtered out (not 8-K)
            ("8-K", "2023-12-31", "2.02"),                 # filtered out (before window)
        ]))),
    ])
    adapter = SEC8KEventAdapter(
        cache_dir=repo_tmp_path, session=session, rate_limit_sleep=0.0,
    )
    out = adapter.fetch(["AAPL"], "2024-01-01", "2024-03-01")
    assert len(out) == 2
    # 2.02 (0.0) + 7.01 (0.1) = 0.1
    assert out.loc[(pd.Timestamp("2024-02-01"), "AAPL")] == pytest.approx(0.1)
    # 5.02 = -0.2
    assert out.loc[(pd.Timestamp("2024-02-05"), "AAPL")] == pytest.approx(-0.2)


def test_sec_skips_tickers_without_cik(repo_tmp_path):
    session = _StubSession(rules=[
        ("company_tickers.json", _StubResponse(200, _tickers_json([("AAPL", 320193)]))),
    ])
    adapter = SEC8KEventAdapter(
        cache_dir=repo_tmp_path, session=session, rate_limit_sleep=0.0,
    )
    out = adapter.fetch(["NOTREAL"], "2024-01-01", "2024-03-01")
    assert out.empty


def test_sec_respects_ticker_cik_override(repo_tmp_path):
    # Pre-seeding the map means we never hit company_tickers.json.
    session = _StubSession(rules=[
        ("CIK0000320193.json", _StubResponse(200, _submissions_payload([
            ("8-K", "2024-02-01", "2.02"),
        ]))),
    ])
    adapter = SEC8KEventAdapter(
        cache_dir=repo_tmp_path, session=session, rate_limit_sleep=0.0,
        ticker_cik_map={"AAPL": "320193"},
    )
    out = adapter.fetch(["AAPL"], "2024-01-01", "2024-03-01")
    assert len(out) == 1
    # No tickers.json request was made.
    assert not any("company_tickers.json" in c["url"] for c in session.calls)


def test_sec_custom_item_weights_override_defaults(repo_tmp_path):
    session = _StubSession(rules=[
        ("company_tickers.json", _StubResponse(200, _tickers_json([("AAPL", 320193)]))),
        ("CIK0000320193.json", _StubResponse(200, _submissions_payload([
            ("8-K", "2024-02-01", "2.02"),
        ]))),
    ])
    adapter = SEC8KEventAdapter(
        cache_dir=repo_tmp_path, session=session, rate_limit_sleep=0.0,
        item_weights={"2.02": 0.5},
    )
    out = adapter.fetch(["AAPL"], "2024-01-01", "2024-03-01")
    assert out.iloc[0] == pytest.approx(0.5)


def test_sec_network_failure_returns_empty(repo_tmp_path):
    # tickers.json comes back but CIK JSON is 500.
    session = _StubSession(rules=[
        ("company_tickers.json", _StubResponse(200, _tickers_json([("AAPL", 320193)]))),
        ("CIK0000320193.json", _StubResponse(500, {})),
    ])
    adapter = SEC8KEventAdapter(
        cache_dir=repo_tmp_path, session=session, rate_limit_sleep=0.0,
    )
    out = adapter.fetch(["AAPL"], "2024-01-01", "2024-03-01")
    assert out.empty


# ---------------------------------------------------------------------------
# Google Trends — only run if pytrends is importable
# ---------------------------------------------------------------------------

def test_google_trends_optional_import_contract():
    """If pytrends isn't installed, GoogleTrendsAdapter must not be exported."""
    from auto_researcher.data.altdata import HAS_GOOGLE_TRENDS

    if HAS_GOOGLE_TRENDS:
        from auto_researcher.data.altdata import GoogleTrendsAdapter
        assert GoogleTrendsAdapter is not None
    else:
        with pytest.raises(ImportError):
            from auto_researcher.data.altdata.google_trends import GoogleTrendsAdapter
            GoogleTrendsAdapter()


# ---------------------------------------------------------------------------
# Reddit — only run if praw is importable
# ---------------------------------------------------------------------------

def test_reddit_optional_import_contract():
    from auto_researcher.data.altdata import HAS_REDDIT

    if HAS_REDDIT:
        from auto_researcher.data.altdata import RedditMentionsAdapter
        assert RedditMentionsAdapter is not None
    else:
        with pytest.raises(ImportError):
            from auto_researcher.data.altdata.reddit import RedditMentionsAdapter
            RedditMentionsAdapter()


def test_reddit_adapter_with_injected_client_needs_no_praw(repo_tmp_path):
    """The adapter accepts a pre-built client, so tests can exercise it
    without praw installed."""
    # Build a fake post + subreddit + reddit client.
    fake_post = MagicMock()
    fake_post.id = "xyz"
    fake_post.created_utc = pd.Timestamp("2024-02-01").timestamp()
    fake_post.score = 99
    fake_post.title = "AAPL to the moon"
    fake_post.selftext = ""

    fake_sub = MagicMock()
    fake_sub.search.return_value = iter([fake_post])

    fake_reddit = MagicMock()
    fake_reddit.subreddit.return_value = fake_sub

    # Importing via the module path — only available if praw installs OK.
    try:
        from auto_researcher.data.altdata.reddit import RedditMentionsAdapter
    except ImportError:
        pytest.skip("praw not installed")

    adapter = RedditMentionsAdapter(
        cache_dir=repo_tmp_path, reddit_client=fake_reddit,
        subreddits=["wallstreetbets"],
        use_finbert=False, sleep_between_tickers=0.0,
    )
    out = adapter.fetch(["AAPL"], "2024-01-01", "2024-03-01")
    assert len(out) == 1
    # With use_finbert=False, tone is 0 → weighted = log1p(99) * 0 → we emit
    # log1p(score) unchanged as a fallback.
    assert out.iloc[0] == pytest.approx(np.log1p(99))
