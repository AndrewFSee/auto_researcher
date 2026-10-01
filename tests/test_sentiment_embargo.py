"""
Phase 2.3a — sentiment/news embargo gate.

These tests pin the invariant that, when a backtest caller supplies
``as_of_date``, no article published on or after ``as_of_date - embargo_days``
can reach the downstream sentiment analyzer. They exercise the pure
filtering helper directly so they don't depend on network or scraper DB.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from auto_researcher.models.earnings_topic_model import (
    _filter_embargoed,
    _parse_article_date,
)


def _mk(pub: str | None, **kwargs) -> dict:
    out = {"title": "t"}
    if pub is not None:
        out["published_date"] = pub
    out.update(kwargs)
    return out


class TestParseArticleDate:
    def test_parses_iso_utc_with_z(self) -> None:
        dt = _parse_article_date("2024-06-15T10:30:00Z")
        assert dt is not None
        assert dt.year == 2024 and dt.month == 6 and dt.day == 15

    def test_parses_bare_date_string(self) -> None:
        dt = _parse_article_date("2024-06-15")
        assert dt is not None
        assert dt.year == 2024

    def test_returns_none_for_unparseable(self) -> None:
        assert _parse_article_date("not a date") is None
        assert _parse_article_date("") is None
        assert _parse_article_date(None) is None


class TestFilterEmbargoed:
    """The cut is strict (< cutoff); an article published at the cut is dropped."""

    def test_drops_article_on_or_after_cut(self) -> None:
        as_of = datetime(2024, 6, 15)
        articles = [
            _mk("2024-06-10T12:00:00"),    # well before cut
            _mk("2024-06-13T23:59:59"),    # just before cut (as_of - 1d)
            _mk("2024-06-14T00:00:00"),    # at the cut
            _mk("2024-06-14T10:00:00"),    # after the cut
            _mk("2024-06-15T00:00:00"),    # equal to as_of
        ]
        kept = _filter_embargoed(
            articles,
            as_of_date=as_of,
            embargo_days=1,
            date_field="published_date",
        )
        kept_dates = [a["published_date"] for a in kept]
        assert kept_dates == [
            "2024-06-10T12:00:00",
            "2024-06-13T23:59:59",
        ]

    def test_respects_custom_embargo_days(self) -> None:
        as_of = datetime(2024, 6, 15)
        articles = [
            _mk("2024-06-10T12:00:00"),
            _mk("2024-06-11T12:00:00"),
        ]
        # embargo=7 → cut is 2024-06-08; nothing passes
        kept = _filter_embargoed(
            articles, as_of_date=as_of, embargo_days=7, date_field="published_date"
        )
        assert kept == []

    def test_drops_undated_articles(self) -> None:
        as_of = datetime(2024, 6, 15)
        articles = [
            _mk(None),                                   # no date → drop
            _mk("garbage"),                              # unparseable → drop
            _mk("2024-06-10T12:00:00"),                  # keep
        ]
        kept = _filter_embargoed(
            articles, as_of_date=as_of, embargo_days=1, date_field="published_date"
        )
        assert [a["published_date"] for a in kept if "published_date" in a] == [
            "2024-06-10T12:00:00"
        ]

    def test_handles_tz_aware_published_with_naive_cut(self) -> None:
        """Aware article + naive cut: compare by wall-clock time (aware side is stripped)."""
        as_of = datetime(2024, 6, 15)
        articles = [
            {"published_date": datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)},
            {"published_date": datetime(2024, 6, 14, 12, 0, tzinfo=timezone.utc)},
        ]
        kept = _filter_embargoed(
            articles, as_of_date=as_of, embargo_days=1, date_field="published_date"
        )
        assert len(kept) == 1
        assert kept[0]["published_date"].day == 10

    def test_alternate_date_field(self) -> None:
        as_of = datetime(2024, 6, 15)
        articles = [
            {"pubDate": "2024-06-10T12:00:00"},
            {"pubDate": "2024-06-14T12:00:00"},
        ]
        kept = _filter_embargoed(
            articles, as_of_date=as_of, embargo_days=1, date_field="pubDate"
        )
        assert [a["pubDate"] for a in kept] == ["2024-06-10T12:00:00"]


class TestEarningsTopicModelIntegration:
    """analyze_news should route through _filter_embargoed when as_of_date is set."""

    def test_analyze_news_drops_post_embargo_articles(self) -> None:
        # Lightweight smoke test — we want to confirm the wiring, not the
        # underlying topic classifier. A caller that supplies articles all
        # dated on or after the cut should get n_earnings == 0.
        try:
            from auto_researcher.models.earnings_topic_model import EarningsTopicModel
        except Exception:
            import pytest
            pytest.skip("earnings_topic_model dependencies not available")

        model = EarningsTopicModel()
        as_of = datetime(2024, 6, 15)
        articles = [
            {"title": "Apple beats earnings expectations on strong iPhone sales",
             "published_date": "2024-06-15T12:00:00"},  # at cut — should drop
            {"title": "Apple earnings: revenue up on services growth",
             "published_date": "2024-06-16T12:00:00"},  # post — should drop
        ]
        signal = model.analyze_news(
            articles,
            ticker="AAPL",
            as_of_date=as_of,
            embargo_days=1,
        )
        assert signal.earnings_articles == 0
        assert signal.total_articles == 0

    def test_analyze_news_keeps_pre_embargo_articles(self) -> None:
        try:
            from auto_researcher.models.earnings_topic_model import EarningsTopicModel
        except Exception:
            import pytest
            pytest.skip("earnings_topic_model dependencies not available")

        model = EarningsTopicModel()
        as_of = datetime(2024, 6, 15)
        articles = [
            {"title": "Apple beats earnings expectations on strong iPhone sales",
             "published_date": "2024-06-10T12:00:00"},  # well before cut
        ]
        signal = model.analyze_news(
            articles,
            ticker="AAPL",
            as_of_date=as_of,
            embargo_days=1,
        )
        # Should have picked up ≥ 1 earnings-relevant article.
        assert signal.total_articles >= 1
