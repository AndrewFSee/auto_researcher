"""Tests for topic-weighted sentiment feature wiring.

Verifies the full pipe: article-level loader → RollingBERTopic →
TopicICCalibrator → daily per-ticker series, with the critical
guarantee that articles ON OR AFTER the calibration cutoff do not
influence the per-topic weights.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

st = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers required for topic-weighted sentiment",
)

from auto_researcher.features.sentiment import (
    _load_article_level_sentiment,
    compute_topic_weighted_sentiment,
)


@pytest.fixture
def scratch_dir():
    """Project-local scratch dir — sidesteps pytest's %TEMP% permission
    issues on this Windows box."""
    scratch = Path(__file__).parent / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic news.db
# ---------------------------------------------------------------------------

EARNINGS_HEADLINES = [
    "AAPL reports record earnings beat in Q{q}",
    "AAPL revenue surges on strong guidance Q{q}",
    "AAPL Q{q} earnings exceed analyst expectations",
    "AAPL profit climbs on product growth Q{q}",
]

LAWSUIT_HEADLINES = [
    "AAPL faces class-action lawsuit alleging antitrust Q{q}",
    "AAPL sued over App Store practices in Q{q}",
    "AAPL investigated by DOJ for monopoly behavior Q{q}",
    "AAPL settles privacy lawsuit with state AGs Q{q}",
]


def _make_news_db(scratch_dir: Path) -> Path:
    """Build a tiny news.db with two clear topic clusters."""
    db = scratch_dir / "news.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            article_hash TEXT,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            published_date TIMESTAMP,
            source TEXT DEFAULT 'Business Insider',
            snippet TEXT,
            full_text TEXT,
            scraped_at TIMESTAMP,
            sentiment_score REAL,
            sentiment_label TEXT,
            fulltext_sentiment_score REAL,
            fulltext_sentiment_label TEXT
        )
        """
    )
    rows = []
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2022-01-03")
    aid = 1
    for q in range(1, 13):  # 3 years of quarterly news
        quarter_date = start + pd.Timedelta(days=90 * (q - 1))
        for i, template in enumerate(EARNINGS_HEADLINES):
            rows.append(
                (
                    aid,
                    f"h{aid}",
                    "AAPL",
                    template.format(q=q),
                    f"http://example.com/{aid}",
                    (quarter_date + pd.Timedelta(days=i)).isoformat(),
                    "test",
                    "",
                    "",
                    pd.Timestamp.utcnow().isoformat(),
                    float(rng.uniform(0.3, 0.9)),
                    "positive",
                    None,
                    None,
                )
            )
            aid += 1
        for i, template in enumerate(LAWSUIT_HEADLINES):
            rows.append(
                (
                    aid,
                    f"h{aid}",
                    "AAPL",
                    template.format(q=q),
                    f"http://example.com/{aid}",
                    (quarter_date + pd.Timedelta(days=30 + i)).isoformat(),
                    "test",
                    "",
                    "",
                    pd.Timestamp.utcnow().isoformat(),
                    float(rng.uniform(-0.9, -0.3)),
                    "negative",
                    None,
                    None,
                )
            )
            aid += 1
    conn.executemany(
        "INSERT INTO articles VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def _make_forward_returns(dates: pd.DatetimeIndex, rng_seed: int = 0) -> pd.DataFrame:
    """Forward returns that are positively correlated with our 'earnings'
    cluster and negatively with the 'lawsuit' cluster, so calibration has
    something real to find."""
    # Crude: use a deterministic pattern keyed on the day offset. The
    # synthetic news.db publishes earnings articles on days 0-3 of each
    # quarter and lawsuit articles on days 30-33 — so we alternate
    # positive / negative returns at those offsets.
    rng = np.random.default_rng(rng_seed)
    rows = []
    start = pd.Timestamp("2022-01-03")
    for q in range(1, 13):
        qd = start + pd.Timedelta(days=90 * (q - 1))
        for i in range(4):
            rows.append((qd + pd.Timedelta(days=i), float(rng.uniform(0.02, 0.08))))
        for i in range(4):
            rows.append(
                (qd + pd.Timedelta(days=30 + i), float(rng.uniform(-0.08, -0.02)))
            )
    df = pd.DataFrame(rows, columns=["date", "AAPL"]).set_index("date")
    return df.reindex(dates).ffill()


def test_article_level_loader_returns_rows(scratch_dir: Path) -> None:
    db = _make_news_db(scratch_dir)
    df = _load_article_level_sentiment(
        ["AAPL"], "2022-01-01", "2025-01-01", db_path=db
    )
    assert len(df) == 12 * 8  # 12 quarters × 8 articles each
    assert set(df.columns) == {"ticker", "title", "published_date", "sentiment_score"}


def test_topic_weighted_sentiment_basic_flow(scratch_dir: Path) -> None:
    db = _make_news_db(scratch_dir)
    dates = pd.date_range("2022-01-01", "2025-01-01", freq="B")
    fwd = _make_forward_returns(dates)

    out = compute_topic_weighted_sentiment(
        tickers=["AAPL"],
        start_date="2022-01-01",
        end_date="2025-01-01",
        forward_returns=fwd,
        calibration_cutoff="2024-01-01",
        db_path=db,
    )

    assert not out.empty
    assert ("AAPL", "sent_topic_weighted") in out.columns
    # Should produce SOME non-NaN days.
    series = out[("AAPL", "sent_topic_weighted")]
    assert series.notna().sum() > 50


def test_cutoff_truly_excludes_future_articles(scratch_dir: Path) -> None:
    """Changing article sentiment AFTER the cutoff must not move the
    per-topic weights — otherwise the calibrator is leaking."""
    db = _make_news_db(scratch_dir)
    dates = pd.date_range("2022-01-01", "2025-01-01", freq="B")
    fwd = _make_forward_returns(dates)

    # Run 1 — original db.
    out1 = compute_topic_weighted_sentiment(
        tickers=["AAPL"],
        start_date="2022-01-01",
        end_date="2025-01-01",
        forward_returns=fwd,
        calibration_cutoff="2024-01-01",
        db_path=db,
    )

    # Run 2 — mutate post-cutoff sentiment to complete garbage.
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE articles SET sentiment_score = 0.0 "
        "WHERE DATE(published_date) >= '2024-01-01'"
    )
    conn.commit()
    conn.close()

    out2 = compute_topic_weighted_sentiment(
        tickers=["AAPL"],
        start_date="2022-01-01",
        end_date="2025-01-01",
        forward_returns=fwd,
        calibration_cutoff="2024-01-01",
        db_path=db,
    )

    # Pre-cutoff rows must be identical — if they change, calibration
    # leaked future info into past weights.
    pre1 = out1.loc[out1.index < "2024-01-01", ("AAPL", "sent_topic_weighted")]
    pre2 = out2.loc[out2.index < "2024-01-01", ("AAPL", "sent_topic_weighted")]
    pd.testing.assert_series_equal(pre1, pre2, check_names=False)


def test_missing_returns_skips_gracefully(scratch_dir: Path) -> None:
    from auto_researcher.features.sentiment import compute_all_sentiment_features

    db = _make_news_db(scratch_dir)
    # use_topic_weighted=True but no returns supplied — should warn &
    # still return the other sentiment features (if any) without crashing.
    out = compute_all_sentiment_features(
        tickers=["AAPL"],
        start_date="2022-01-01",
        end_date="2025-01-01",
        news_db_path=db,
        use_topic_weighted=True,
        topic_weighted_forward_returns=None,
        topic_weighted_cutoff=None,
    )
    # Empty is fine — the point is no crash.
    assert isinstance(out, pd.DataFrame)
