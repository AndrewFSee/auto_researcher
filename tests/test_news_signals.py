"""Tests for point-in-time news-sentiment signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_researcher.features.news_signals import NEWS_SIGNALS, news_signal_panel

CAL = pd.bdate_range("2024-01-01", "2024-03-29")


def _articles(rows):
    return pd.DataFrame(rows, columns=["ticker", "published", "score"]).assign(
        published=lambda d: pd.to_datetime(d["published"], format="mixed"))


def test_article_is_usable_only_from_the_next_trading_day():
    arts = _articles([("A", "2024-02-07 21:30", 0.8)])  # Wednesday, after the close
    panel = news_signal_panel(arts, CAL)
    s = panel.xs("A", level="ticker")["sent_7d"]
    assert np.isnan(s.loc["2024-02-07"])            # not on its own day
    assert s.loc["2024-02-08"] == pytest.approx(0.8)  # next trading day
    assert np.isnan(s.loc["2024-02-15"])            # out of the 7-day window


def test_weekend_article_counts_from_monday():
    arts = _articles([("A", "2024-02-10 12:00", -0.5)])  # Saturday
    s = news_signal_panel(arts, CAL).xs("A", level="ticker")["sent_7d"]
    assert pd.isna(s.get(pd.Timestamp("2024-02-09")))  # no row before the first article
    assert s.loc["2024-02-12"] == pytest.approx(-0.5)


def test_window_means_change_and_surge():
    arts = _articles([
        ("A", "2024-01-05", 0.2), ("A", "2024-01-25", -0.4),
        ("A", "2024-02-01", 0.6), ("A", "2024-02-02", 0.0),
    ])
    row = news_signal_panel(arts, CAL).xs(("2024-02-05", "A"))
    assert row["sent_7d"] == pytest.approx(0.3)                  # Feb 1-2 (window Jan 29 - Feb 4)
    assert row["sent_30d"] == pytest.approx((-0.4 + 0.6 + 0.0) / 3)
    assert row["sent_change"] == pytest.approx(row["sent_7d"] - row["sent_30d"])
    assert row["news_surge"] > 0                                 # 2 articles vs a quiet history
    assert list(news_signal_panel(arts, CAL).columns) == list(NEWS_SIGNALS)


def test_future_articles_do_not_change_past_signals():
    rng = np.random.default_rng(0)
    days = rng.choice(pd.date_range("2024-01-01", "2024-03-28"), 300)
    arts = _articles([(t, d, rng.uniform(-1, 1)) for t, d in zip(rng.choice(["A", "B"], 300), days)])
    cut = pd.Timestamp("2024-02-15")
    base = news_signal_panel(arts, CAL)
    extra = _articles([("A", "2024-02-15 08:00", 1.0), ("B", "2024-03-01", -1.0)])
    moved = news_signal_panel(pd.concat([arts, extra]), CAL)
    pd.testing.assert_frame_equal(base.loc[:cut], moved.loc[:cut])
