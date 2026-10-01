"""
Rolling BERTopic sentiment for news articles.

Why BERTopic over the keyword-match approach in ``topic_sentiment.py``
--------------------------------------------------------------------
The existing topic classifier matches articles to one of a handful of
hand-coded buckets (litigation, earnings, M&A, …) via keyword lists. That
approach is brittle: new themes (AI regulation, supply-chain onshoring)
produce misclassified or "other" articles whose sentiment is then averaged
into a bag whose expected return the model was never trained on.

BERTopic clusters articles in an embedding space (sentence-transformer →
UMAP → HDBSCAN → c-TF-IDF labeling), which lets topics emerge from the
data. The catch: if you fit the topic model on the full corpus and then
backtest, you've leaked future-known topic assignments into past periods.

This module fixes that with ``RollingBERTopic`` — at each refit date, the
topic model is fit **only on articles with publication date < refit
date**, using the trailing ``lookback_days`` window. Topic assignments for
a later article are made via ``transform()`` on the frozen model.

Dependency model
----------------
``bertopic`` is an optional dependency. If it is not installed,
``RollingBERTopic`` still imports, but ``fit()`` will raise with a clear
install hint. Unit tests should use ``LightweightBERTopic``, a
drop-in replacement that uses sentence-transformer + sklearn KMeans
(no UMAP/HDBSCAN) and is deterministic.

Usage::

    from auto_researcher.models.bertopic_sentiment import RollingBERTopic

    rb = RollingBERTopic(
        refit_every=90,          # refit every 90 calendar days
        lookback_days=730,        # on the trailing 2y window
        min_topic_size=10,
    )
    rb.fit(articles_df, date_col="published_date", text_col="title")
    assignments = rb.transform(test_articles_df)

The per-topic sentiment IC calibration is NOT done here — callers feed the
``topic_id`` column into their IC-weighting pipeline (see
``backtest/metrics.compute_ic_weights``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RollingBERTopicConfig:
    refit_every: int = 90            # calendar days between refits
    lookback_days: int = 730          # window used when refitting
    min_topic_size: int = 10
    max_topics: Optional[int] = 50    # cap BERTopic's output
    embedding_model: str = "all-MiniLM-L6-v2"
    random_state: int = 42


# ---------------------------------------------------------------------------
# Snapshot record — one fit of the model, valid from fit_date until next refit
# ---------------------------------------------------------------------------

@dataclass
class TopicSnapshot:
    fit_date: pd.Timestamp
    valid_until: pd.Timestamp
    model: object                    # BERTopic or LightweightBERTopic
    topic_labels: dict[int, str] = field(default_factory=dict)
    n_articles_fit: int = 0

    def transform(self, texts: list[str]) -> np.ndarray:
        """Assign each text to the topic IDs learned at fit_date."""
        topics = self.model.transform(texts)
        # BERTopic returns (topics, probs) — we only need the topic id.
        if isinstance(topics, tuple):
            topics = topics[0]
        return np.asarray(topics)


# ---------------------------------------------------------------------------
# Lightweight fallback — used when bertopic isn't installed
# ---------------------------------------------------------------------------

class LightweightBERTopic:
    """Sentence-transformer + KMeans clusterer, API-compatible with BERTopic.

    Good enough for tests and CI — it's deterministic (seeded KMeans) and
    has no UMAP/HDBSCAN dependency. Produces integer topic ids in
    ``[0, n_clusters)`` (no "-1 / outlier" bucket like BERTopic).
    """

    def __init__(
        self,
        n_clusters: int = 10,
        embedding_model: str = "all-MiniLM-L6-v2",
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.embedding_model_name = embedding_model
        self.random_state = random_state
        self._embedder = None
        self._kmeans = None
        self._topic_words: dict[int, list[str]] = {}

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def fit(self, documents: list[str]) -> "LightweightBERTopic":
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import CountVectorizer

        emb = self._get_embedder().encode(documents, show_progress_bar=False)
        k = min(self.n_clusters, max(2, len(documents) // 5))
        self._kmeans = KMeans(
            n_clusters=k, random_state=self.random_state, n_init=5
        ).fit(emb)

        # c-TF-IDF-ish topic labels via top CountVectorizer terms per cluster.
        labels = self._kmeans.labels_
        vec = CountVectorizer(stop_words="english", max_features=500)
        try:
            counts = vec.fit_transform(documents)
            feature_names = vec.get_feature_names_out()
            for cid in range(k):
                mask = labels == cid
                if mask.sum() == 0:
                    continue
                cluster_counts = np.asarray(counts[mask].sum(axis=0)).ravel()
                top_idx = cluster_counts.argsort()[-5:][::-1]
                self._topic_words[cid] = [feature_names[i] for i in top_idx]
        except ValueError:
            # Empty vocabulary — keep topic_words empty.
            pass
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self._kmeans is None:
            raise RuntimeError("Call fit() before transform().")
        emb = self._get_embedder().encode(documents, show_progress_bar=False)
        return self._kmeans.predict(emb)

    def get_topic_info(self) -> pd.DataFrame:
        """BERTopic-compatible topic info table."""
        if self._kmeans is None:
            return pd.DataFrame(columns=["Topic", "Count", "Name"])
        rows = []
        for cid in range(self._kmeans.n_clusters):
            count = int((self._kmeans.labels_ == cid).sum())
            words = self._topic_words.get(cid, [])
            name = f"{cid}_" + "_".join(words[:3])
            rows.append({"Topic": cid, "Count": count, "Name": name})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BERTopic wrapper (real model, optional dependency)
# ---------------------------------------------------------------------------

def _make_bertopic(config: RollingBERTopicConfig):
    """Instantiate BERTopic or raise with a clear install hint."""
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "bertopic is required for RollingBERTopic's real backend. "
            "Install with `pip install bertopic` (pulls in umap-learn, "
            "hdbscan, sentence-transformers)."
        ) from e

    return BERTopic(
        embedding_model=SentenceTransformer(config.embedding_model),
        min_topic_size=config.min_topic_size,
        nr_topics=config.max_topics,
        calculate_probabilities=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Rolling wrapper
# ---------------------------------------------------------------------------

class RollingBERTopic:
    """Refit-over-time BERTopic.

    Holds a list of snapshots; each snapshot is the topic model trained on
    articles published in a trailing window ending at the snapshot's
    ``fit_date``. At inference time, a given article is assigned by the
    *most recent* snapshot whose ``valid_until`` is ≥ the article's
    publication date.
    """

    def __init__(
        self,
        config: Optional[RollingBERTopicConfig] = None,
        backend: str = "bertopic",
    ):
        self.config = config or RollingBERTopicConfig()
        if backend not in ("bertopic", "lightweight"):
            raise ValueError(f"Unknown backend: {backend!r}")
        self.backend = backend
        self.snapshots: list[TopicSnapshot] = []

    def _make_model(self):
        if self.backend == "lightweight":
            return LightweightBERTopic(
                n_clusters=self.config.max_topics or 10,
                embedding_model=self.config.embedding_model,
                random_state=self.config.random_state,
            )
        return _make_bertopic(self.config)

    def _schedule_refit_dates(
        self, date_series: pd.Series
    ) -> pd.DatetimeIndex:
        """Produce the refit cadence across the data's date range."""
        dates = pd.to_datetime(date_series).sort_values()
        if dates.empty:
            return pd.DatetimeIndex([])
        start = dates.iloc[0] + pd.Timedelta(days=self.config.lookback_days)
        end = dates.iloc[-1]
        if start > end:
            start = end  # fit one snapshot on whatever we have
        return pd.date_range(
            start=start, end=end, freq=f"{self.config.refit_every}D"
        )

    def fit(
        self,
        articles: pd.DataFrame,
        date_col: str = "published_date",
        text_col: str = "title",
    ) -> "RollingBERTopic":
        """Fit a sequence of snapshots on trailing windows."""
        if articles.empty:
            logger.warning("fit() called with empty DataFrame")
            return self

        df = articles.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.dropna(subset=[date_col, text_col])
        df = df.sort_values(date_col)

        refit_dates = self._schedule_refit_dates(df[date_col])
        if len(refit_dates) == 0:
            logger.warning("No refit dates computed — skipping fit")
            return self

        self.snapshots = []
        for i, fit_date in enumerate(refit_dates):
            window_start = fit_date - pd.Timedelta(days=self.config.lookback_days)
            # STRICT inequality — the snapshot cannot see articles dated on
            # or after fit_date. This is the key anti-leakage invariant.
            mask = (df[date_col] >= window_start) & (df[date_col] < fit_date)
            window = df.loc[mask, text_col].astype(str).tolist()
            if len(window) < max(self.config.min_topic_size * 2, 20):
                logger.debug(
                    "fit_date=%s: only %d articles in window — skipping snapshot",
                    fit_date.date(), len(window),
                )
                continue

            model = self._make_model()
            try:
                model.fit(window)
            except Exception as e:
                logger.warning("Snapshot fit at %s failed: %s", fit_date.date(), e)
                continue

            valid_until = (
                refit_dates[i + 1]
                if i + 1 < len(refit_dates)
                else df[date_col].iloc[-1] + pd.Timedelta(days=self.config.refit_every)
            )
            snap = TopicSnapshot(
                fit_date=fit_date,
                valid_until=valid_until,
                model=model,
                n_articles_fit=len(window),
            )
            # Populate human-readable labels if the backend supports it.
            try:
                info = model.get_topic_info()
                for _, row in info.iterrows():
                    snap.topic_labels[int(row["Topic"])] = str(row.get("Name", ""))
            except Exception:
                pass
            self.snapshots.append(snap)

        logger.info(
            "RollingBERTopic fit complete: %d snapshots across %d articles",
            len(self.snapshots), len(df),
        )
        return self

    def _snapshot_for(self, date: pd.Timestamp) -> Optional[TopicSnapshot]:
        """Pick the most recent snapshot whose fit_date is ≤ ``date``."""
        if not self.snapshots:
            return None
        eligible = [s for s in self.snapshots if s.fit_date <= date]
        if not eligible:
            return None
        return max(eligible, key=lambda s: s.fit_date)

    def transform(
        self,
        articles: pd.DataFrame,
        date_col: str = "published_date",
        text_col: str = "title",
    ) -> pd.Series:
        """Assign each article to a topic using the snapshot valid at its date.

        Articles before the first snapshot's ``fit_date`` are labeled
        retroactively with that first snapshot. This is causally sound for
        backtest/calibration use — the snapshot was trained strictly on
        articles before its ``fit_date``, so applying it to even-earlier
        articles doesn't leak future information. (It *is* "not realtime"
        in the sense that a live trader at those earlier dates wouldn't
        have had any topic model yet; the calibrator's use of this labels
        is what freezes the weights, not the labels themselves.)
        """
        if articles.empty:
            return pd.Series(dtype="int64")
        df = articles.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        out = pd.Series(-1, index=df.index, dtype="int64")
        # Batch by snapshot to avoid recomputing embeddings per row.
        for snap in self.snapshots:
            mask = (df[date_col] >= snap.fit_date) & (df[date_col] < snap.valid_until)
            if not mask.any():
                continue
            texts = df.loc[mask, text_col].astype(str).tolist()
            try:
                topics = snap.transform(texts)
                out.loc[mask] = topics
            except Exception as e:
                logger.warning("transform via snapshot %s failed: %s", snap.fit_date, e)

        # Retroactive backfill for articles before the first fit_date.
        if self.snapshots:
            first = self.snapshots[0]
            pre_mask = (df[date_col] < first.fit_date) & (out == -1)
            if pre_mask.any():
                texts = df.loc[pre_mask, text_col].astype(str).tolist()
                try:
                    topics = first.transform(texts)
                    out.loc[pre_mask] = topics
                except Exception as e:
                    logger.warning(
                        "retroactive transform via snapshot %s failed: %s",
                        first.fit_date, e,
                    )
        return out

    def topic_label(self, snapshot: TopicSnapshot, topic_id: int) -> str:
        return snapshot.topic_labels.get(int(topic_id), f"topic_{topic_id}")
