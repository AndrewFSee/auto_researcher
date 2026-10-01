"""Tests for TopicICCalibrator — signed per-topic IC, contrarian handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from auto_researcher.models.topic_ic_calibrator import (
    TopicICCalibrator,
    TopicICConfig,
)


def _make_fixture(
    n_per_topic: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixture with three clearly-separated topics:

    * topic 0: positive IC — sentiment correlates with forward return.
    * topic 1: contrarian — sentiment is *negatively* correlated.
    * topic 2: noise — sentiment and return are independent.
    """
    rng = np.random.default_rng(seed)

    def block(topic_id: int, ic: float):
        sent = rng.normal(size=n_per_topic)
        # Induce a controlled IC by mixing sentiment into the return.
        noise = rng.normal(size=n_per_topic)
        ret = ic * sent + np.sqrt(max(1 - ic**2, 0.01)) * noise
        return (
            np.full(n_per_topic, topic_id, dtype=int),
            sent,
            ret,
        )

    t0, s0, r0 = block(0, ic=0.40)   # positive
    t1, s1, r1 = block(1, ic=-0.35)  # contrarian
    t2, s2, r2 = block(2, ic=0.0)    # noise

    topics = np.concatenate([t0, t1, t2])
    sentiment = np.concatenate([s0, s1, s2])
    returns = np.concatenate([r0, r1, r2])

    # Shuffle so topic order isn't time-order.
    idx = rng.permutation(len(topics))
    return topics[idx], sentiment[idx], returns[idx]


def test_positive_topic_retained_with_positive_weight() -> None:
    topics, sent, ret = _make_fixture()
    cal = TopicICCalibrator(TopicICConfig(min_obs=30, ic_floor=0.02))
    cal.fit(topics, sent, ret)

    weights = cal.signed_weights()
    assert 0 in weights
    assert weights[0] > 0.2, (
        f"expected strong positive IC for topic 0, got {weights[0]:.3f}"
    )


def test_contrarian_topic_kept_with_negative_weight() -> None:
    topics, sent, ret = _make_fixture()
    cal = TopicICCalibrator(TopicICConfig(min_obs=30, ic_floor=0.02))
    cal.fit(topics, sent, ret)

    weights = cal.signed_weights()
    # The whole point of this module: topic 1 should NOT be dropped —
    # its negative IC should be preserved so its contribution at
    # inference time flips sign appropriately.
    assert 1 in weights, "contrarian topic was dropped instead of kept as negative"
    assert weights[1] < -0.2, (
        f"expected strongly negative IC for topic 1, got {weights[1]:.3f}"
    )


def test_noise_topic_dropped() -> None:
    topics, sent, ret = _make_fixture()
    cal = TopicICCalibrator(TopicICConfig(min_obs=30, ic_floor=0.05))
    cal.fit(topics, sent, ret)

    weights = cal.signed_weights()
    # topic 2 was generated with IC=0 — should be filtered as noise.
    assert 2 not in weights
    # And the diagnostic record should explain why.
    diag = cal.diagnostics_frame()
    noise_row = diag[diag["topic_id"] == 2].iloc[0]
    assert "floor" in noise_row["dropped_reason"].lower()


def test_apply_flips_sign_for_contrarian() -> None:
    topics, sent, ret = _make_fixture()
    cal = TopicICCalibrator(TopicICConfig(min_obs=30, ic_floor=0.02))
    cal.fit(topics, sent, ret)

    # Two test articles: one in topic 0 (positive), one in topic 1 (contrarian),
    # both with identical +1.0 sentiment.
    contrib = cal.apply([0, 1], [1.0, 1.0])
    assert contrib[0] > 0, "positive-weight topic should keep sign"
    assert contrib[1] < 0, "contrarian topic should flip sign"


def test_min_obs_drops_underpowered_topics() -> None:
    # Tiny sample for an otherwise-strong topic — should be dropped.
    topics = np.array([5, 5, 5, 5])   # only 4 obs
    sent = np.array([1.0, 0.5, -0.5, -1.0])
    ret = np.array([1.0, 0.5, -0.5, -1.0])
    cal = TopicICCalibrator(TopicICConfig(min_obs=30, ic_floor=0.02))
    cal.fit(topics, sent, ret)

    assert cal.signed_weights() == {}
    assert "min_obs" in cal.diagnostics[0].dropped_reason


def test_coherence_floor_filters_low_coherence() -> None:
    topics, sent, ret = _make_fixture()
    cal = TopicICCalibrator(
        TopicICConfig(min_obs=30, ic_floor=0.02, coherence_floor=0.5)
    )
    # Topic 0 has good IC but poor coherence → should be dropped.
    cal.fit(topics, sent, ret, topic_coherence={0: 0.1, 1: 0.8, 2: 0.8})
    weights = cal.signed_weights()
    assert 0 not in weights
    diag = cal.diagnostics_frame()
    reason = diag[diag["topic_id"] == 0].iloc[0]["dropped_reason"]
    assert "coherence" in reason


def test_unknown_topic_at_inference_contributes_zero() -> None:
    topics, sent, ret = _make_fixture()
    cal = TopicICCalibrator().fit(topics, sent, ret)

    # Topic 99 was never in training — safe default: zero contribution.
    contrib = cal.apply([99], [1.0])
    assert contrib[0] == 0.0
