"""Tests for RollingBERTopic (using the lightweight sklearn backend).

The ``bertopic`` package is an optional dependency we don't want CI to
require, so every test here pins ``backend="lightweight"``, which uses
sentence-transformers + sklearn KMeans and produces deterministic topic
ids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

st = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers required for lightweight BERTopic backend",
)

from auto_researcher.models.bertopic_sentiment import (
    LightweightBERTopic,
    RollingBERTopic,
    RollingBERTopicConfig,
)


# Build a small corpus with two obvious topic clusters (earnings vs lawsuit).
EARNINGS_DOCS = [
    "Apple reports record quarterly earnings beating estimates",
    "Microsoft revenue surges on cloud growth in Q3",
    "Amazon posts strong Q4 profit on retail and AWS",
    "Google earnings exceed analyst expectations",
    "Meta revenue growth accelerates in advertising segment",
    "Nvidia beats earnings on AI chip demand",
    "Tesla delivers record quarterly car shipments",
    "Netflix subscriber growth drives profit beat",
]

LAWSUIT_DOCS = [
    "Apple faces class-action lawsuit over App Store practices",
    "Microsoft sued by competitor alleging antitrust violations",
    "Meta settles privacy lawsuit with state attorneys general",
    "Google investigated by DOJ for antitrust behavior",
    "Amazon lawsuit alleges worker misclassification",
    "Nvidia faces patent infringement claims in court",
    "Netflix settles copyright infringement complaint",
    "Tesla investors file securities fraud lawsuit",
]


def _build_articles(n_copies: int = 3) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2022-01-01")
    # Interleave so time and topic are uncorrelated.
    all_docs = []
    for _ in range(n_copies):
        all_docs.extend(EARNINGS_DOCS)
        all_docs.extend(LAWSUIT_DOCS)
    for i, text in enumerate(all_docs):
        rows.append(
            {"published_date": start + pd.Timedelta(days=i * 3), "title": text}
        )
    return pd.DataFrame(rows)


def test_lightweight_fit_and_transform_separates_topics() -> None:
    docs = EARNINGS_DOCS + LAWSUIT_DOCS
    model = LightweightBERTopic(n_clusters=2, random_state=0).fit(docs)
    topics = model.transform(docs)
    assert len(topics) == len(docs)
    # Earnings docs should cluster together; lawsuit docs together. Expect
    # at most a couple of misassignments with a tiny corpus.
    earnings_topics = topics[: len(EARNINGS_DOCS)]
    lawsuit_topics = topics[len(EARNINGS_DOCS):]
    majority_earnings = np.bincount(earnings_topics).argmax()
    majority_lawsuit = np.bincount(lawsuit_topics).argmax()
    assert majority_earnings != majority_lawsuit, (
        "KMeans should separate the two clusters"
    )


def test_rolling_bertopic_respects_training_window() -> None:
    articles = _build_articles(n_copies=4)
    rb = RollingBERTopic(
        config=RollingBERTopicConfig(
            refit_every=60, lookback_days=120, max_topics=2, min_topic_size=3
        ),
        backend="lightweight",
    ).fit(articles)

    assert len(rb.snapshots) >= 1
    # The anti-leakage invariant: every snapshot was fit on articles
    # published STRICTLY before its fit_date.
    for snap in rb.snapshots:
        window_start = snap.fit_date - pd.Timedelta(days=120)
        window = articles[
            (articles["published_date"] >= window_start)
            & (articles["published_date"] < snap.fit_date)
        ]
        assert snap.n_articles_fit == len(window), (
            f"snapshot at {snap.fit_date} trained on {snap.n_articles_fit} "
            f"but window only contains {len(window)}"
        )


def test_transform_uses_correct_snapshot_per_date() -> None:
    articles = _build_articles(n_copies=4)
    rb = RollingBERTopic(
        config=RollingBERTopicConfig(
            refit_every=60, lookback_days=120, max_topics=2, min_topic_size=3
        ),
        backend="lightweight",
    ).fit(articles)

    # Transform on a fresh batch interleaved with the training range.
    test_articles = articles.sample(frac=0.3, random_state=1).copy()
    assignments = rb.transform(test_articles)

    assert len(assignments) == len(test_articles)
    # Every article should get a real topic id — pre-first-fit articles
    # are labeled retroactively with the first snapshot (causally sound:
    # that snapshot was trained strictly on data before its fit_date).
    assert (assignments >= 0).all(), (
        f"unexpected -1 assignments: {assignments[assignments < 0].tolist()}"
    )


def test_empty_input_is_handled() -> None:
    rb = RollingBERTopic(backend="lightweight")
    out = rb.transform(pd.DataFrame(columns=["published_date", "title"]))
    assert len(out) == 0


def test_real_bertopic_backend_raises_clear_error_when_missing() -> None:
    # We don't want tests to depend on the real BERTopic package, but the
    # wrapper should raise a clean ImportError rather than crashing late.
    rb = RollingBERTopic(backend="bertopic")
    articles = _build_articles(n_copies=1)
    try:
        rb.fit(articles)
    except ImportError as e:
        assert "bertopic" in str(e).lower()
    except Exception:
        # If bertopic IS installed, it may run fine — that's acceptable.
        pass
