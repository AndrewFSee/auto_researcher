"""
Per-topic IC calibration with signed weights.

Problem
-------
A clustered topic model (``RollingBERTopic``) gives us topic ids per
article, but treating every topic the same dilutes signal. Prior
backtesting found that some topics carry strong forward-return
predictive power (earnings beats, guidance raises), some carry strong
*inverse* predictive power (litigation headlines, management departures,
broad-macro scare pieces), and most are pure noise that should be
excluded from the composite altogether.

What this module does
---------------------
For each topic ``t`` produced by the clusterer, measure its Information
Coefficient (signed Spearman correlation) between ``sentiment_score × 1[topic == t]``
and forward return on **the training fold only**. Keep the full signed IC as
the weight — so a consistently-contrarian topic survives as a negative
coefficient rather than being tossed.

At inference time, the per-article contribution becomes::

    contribution = sentiment_score * topic_ic_weight[topic_id]

Articles in dropped topics (|IC| below the significance floor) contribute
zero — equivalent to the old "retain topics whose IC > threshold" rule,
but two-sided and sign-preserving.

Coherence floor
---------------
``coherence_floor`` is optional; when supplied, topics with poor
semantic coherence (a proxy for "the cluster isn't really about one
thing") are dropped regardless of IC. The default 0.0 skips the check.
This mirrors the BERTopic literature — a topic that ICs well but has no
semantic backbone is almost always a data artifact (e.g., a single
outlier event repeated across similar headlines).

Usage::

    from auto_researcher.models.topic_ic_calibrator import (
        TopicICCalibrator, TopicICConfig,
    )

    cal = TopicICCalibrator(TopicICConfig(min_obs=30, ic_floor=0.02))
    cal.fit(
        topic_ids=train_topic_ids,       # int per article
        sentiment=train_sentiment,        # FinBERT score in [-1, 1]
        forward_return=train_fwd_return,  # target, aligned by article
    )

    weights = cal.signed_weights()
    # {topic_id: signed IC}; missing topics → drop (treat as 0)

    contribution = cal.apply(test_topic_ids, test_sentiment)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TopicICConfig:
    """Calibration thresholds.

    Attributes:
        min_obs: Skip any topic with fewer than this many observations
            in the training window — you can't estimate an IC from 3
            articles. 30 is a reasonable minimum for a Spearman on
            weekly frequency data; raise it for noisier regimes.
        ic_floor: Absolute IC floor. Topics with ``|IC| < ic_floor``
            are treated as noise and dropped (weight = 0). Default 0.02
            is chosen to clear a rough ~1.5σ threshold at n_obs ≈ 100.
        coherence_floor: Optional coherence threshold (0..1). When the
            calibrator is supplied a ``topic_coherence`` map, any topic
            below this floor is dropped regardless of IC. Default 0.0
            disables the check.
        winsorize_pct: Tail-clip returns before correlation, so a single
            blown-up day doesn't dominate the IC. 0.01 → clip 1% tails.
            Set to 0 to disable.
    """

    min_obs: int = 30
    ic_floor: float = 0.02
    coherence_floor: float = 0.0
    winsorize_pct: float = 0.01


@dataclass
class TopicDiagnostics:
    """Per-topic inspection record — what survived and what didn't."""

    topic_id: int
    n_obs: int
    ic: float
    coherence: float
    weight: float              # Signed weight applied at inference
    dropped_reason: str = ""   # "" if kept, else human-readable reason


class TopicICCalibrator:
    """Fit signed per-topic IC weights on a training fold only.

    The signed weight is just the measured IC — this preserves direction
    (contrarian topics stay negative) and magnitude (high-IC topics
    contribute more than marginal ones) in a single coefficient.
    """

    def __init__(self, config: Optional[TopicICConfig] = None):
        self.config = config or TopicICConfig()
        self.diagnostics: list[TopicDiagnostics] = []
        self._weights: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        topic_ids: np.ndarray | pd.Series,
        sentiment: np.ndarray | pd.Series,
        forward_return: np.ndarray | pd.Series,
        topic_coherence: Optional[Mapping[int, float]] = None,
    ) -> "TopicICCalibrator":
        """Measure per-topic IC and populate signed weights.

        Args:
            topic_ids: Integer topic id per training-fold article.
            sentiment: Sentiment score aligned to ``topic_ids``.
            forward_return: Realized forward return aligned to the same rows.
            topic_coherence: Optional {topic_id: coherence in [0, 1]};
                supplied when the clusterer can produce coherence scores
                (BERTopic's built-in metric, or a manual c_v computation).
        """
        df = pd.DataFrame(
            {
                "topic": np.asarray(topic_ids, dtype=int),
                "sent": np.asarray(sentiment, dtype=float),
                "ret": np.asarray(forward_return, dtype=float),
            }
        ).dropna()

        if df.empty:
            logger.warning("TopicICCalibrator.fit() received no usable rows")
            return self

        if self.config.winsorize_pct > 0:
            lo, hi = df["ret"].quantile(
                [self.config.winsorize_pct, 1 - self.config.winsorize_pct]
            )
            df["ret"] = df["ret"].clip(lo, hi)

        self.diagnostics = []
        self._weights = {}

        for topic_id, group in df.groupby("topic"):
            n = len(group)
            coherence = float(
                topic_coherence.get(int(topic_id), 1.0) if topic_coherence else 1.0
            )

            drop_reason = ""
            if n < self.config.min_obs:
                ic = float("nan")
                drop_reason = f"n_obs={n} < min_obs={self.config.min_obs}"
            else:
                # Spearman rank correlation — robust to non-Gaussian returns.
                ic = float(
                    group["sent"].rank().corr(group["ret"].rank())
                )
                if not np.isfinite(ic):
                    drop_reason = "IC is NaN (zero-variance sentiment?)"
                elif abs(ic) < self.config.ic_floor:
                    drop_reason = (
                        f"|IC|={abs(ic):.3f} < floor={self.config.ic_floor}"
                    )
                elif coherence < self.config.coherence_floor:
                    drop_reason = (
                        f"coherence={coherence:.2f} < floor="
                        f"{self.config.coherence_floor}"
                    )

            weight = 0.0 if drop_reason else ic

            self.diagnostics.append(
                TopicDiagnostics(
                    topic_id=int(topic_id),
                    n_obs=n,
                    ic=ic,
                    coherence=coherence,
                    weight=weight,
                    dropped_reason=drop_reason,
                )
            )
            if weight != 0.0:
                self._weights[int(topic_id)] = weight

        kept = [d for d in self.diagnostics if not d.dropped_reason]
        contrarian = [d for d in kept if d.weight < 0]
        logger.info(
            "TopicICCalibrator: kept %d/%d topics (%d contrarian, "
            "min IC=%.3f max IC=%.3f)",
            len(kept),
            len(self.diagnostics),
            len(contrarian),
            min((d.weight for d in kept), default=0.0),
            max((d.weight for d in kept), default=0.0),
        )
        return self

    # ------------------------------------------------------------------
    # Inspection / application
    # ------------------------------------------------------------------

    def signed_weights(self) -> dict[int, float]:
        """Return the {topic_id: signed weight} map."""
        return dict(self._weights)

    def diagnostics_frame(self) -> pd.DataFrame:
        """Human-readable per-topic breakdown."""
        if not self.diagnostics:
            return pd.DataFrame(
                columns=["topic_id", "n_obs", "ic", "coherence",
                         "weight", "dropped_reason"]
            )
        return pd.DataFrame([d.__dict__ for d in self.diagnostics]).sort_values(
            "ic", ascending=False, na_position="last"
        )

    def apply(
        self,
        topic_ids: np.ndarray | pd.Series,
        sentiment: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """Per-article ``sentiment × signed_weight[topic]`` at inference time.

        Articles whose topic was dropped contribute 0 — equivalent to
        excluding them from the composite. Articles with unknown topic
        ids (e.g., a new topic id from a fresh snapshot that wasn't in
        training) also contribute 0, which is the safe default.
        """
        tids = np.asarray(topic_ids, dtype=int)
        sents = np.asarray(sentiment, dtype=float)
        out = np.zeros(len(tids), dtype=float)
        for i, (t, s) in enumerate(zip(tids, sents)):
            out[i] = s * self._weights.get(int(t), 0.0)
        return out
