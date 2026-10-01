"""
Combinatorial Purged Cross-Validation (CPCV).

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
    Chapter 12: "Cross-Validation in Finance".

Standard k-fold CV leaks information when labels overlap — in a panel dataset
the forward return at date D depends on prices through D+horizon, so the
label at D is correlated with labels at D+1 … D+horizon. A naive train/test
split where those dates fall on opposite sides silently trains on information
that bleeds into the test set.

CPCV fixes this in two ways:

1. **Purge** — rows in the training set whose forward-return window overlaps
   any test-set date are dropped.
2. **Embargo** — a small buffer after the test window is also excluded from
   training, to protect against serial correlation in residuals.
3. **Combinatorial** — instead of k sequential folds, we split into N groups
   and take all C(N, K) ways of choosing K groups as the test set. The result
   is a *distribution* of OOS metrics (typically 15 test paths for N=6, K=2),
   not a single point estimate, which lets us compute deflated Sharpe /
   bootstrap intervals instead of eyeballing one path.

Usage::

    from auto_researcher.validation.cpcv import combinatorial_purged_splits

    for train_idx, test_idx in combinatorial_purged_splits(
        dates=price_dates,
        n_splits=6,
        n_test_splits=2,
        horizon_days=21,
        embargo_pct=0.01,
    ):
        fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = predict(X.iloc[test_idx])
        # aggregate preds across test paths → OOS metric distribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CPCVSplit:
    """One (train, test) index pair produced by CPCV."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_group_ids: tuple[int, ...]

    def __iter__(self):
        yield self.train_idx
        yield self.test_idx


def _group_boundaries(n_samples: int, n_splits: int) -> list[tuple[int, int]]:
    """Divide ``[0, n_samples)`` into ``n_splits`` contiguous groups."""
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n_samples < n_splits:
        raise ValueError(
            f"n_samples={n_samples} < n_splits={n_splits}; "
            "cannot split into groups"
        )
    edges = np.linspace(0, n_samples, n_splits + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_splits)]


def combinatorial_purged_splits(
    dates: pd.DatetimeIndex | pd.Series,
    n_splits: int = 6,
    n_test_splits: int = 2,
    horizon_days: int = 21,
    embargo_pct: float = 0.01,
) -> Iterator[CPCVSplit]:
    """
    Yield all C(n_splits, n_test_splits) purged-and-embargoed (train, test)
    index pairs.

    Args:
        dates: Per-row date labels. Can be a DatetimeIndex (sample index) or
            a Series where the value at each position is the row's date.
            Order matters — rows are grouped by their position in ``dates``.
        n_splits: Total number of contiguous date groups (N in C(N, K)).
        n_test_splits: Number of groups held out for testing at a time (K).
        horizon_days: Label horizon in **trading days** (positions in the
            sorted unique dates, matching ``TargetConfig.horizon_days``).
            A training row is purged when its label window overlaps the label
            window of any test row, i.e. when its date lies within
            ``horizon_days`` trading days before *or after* a test group.
        embargo_pct: Additional buffer after each test group, as a fraction
            of the number of unique dates. 0.01 is a reasonable default.

    Yields:
        ``CPCVSplit`` records, one per combination.

    Notes:
        With n_splits=6, n_test_splits=2 you get C(6,2) = 15 paths — the
        usual setting in López de Prado's examples.

        Earlier versions measured the purge in calendar days and only purged
        rows *before* each test group. With a 21-trading-day label that left
        about six trading days of overlap before each group and roughly
        ``horizon`` days after it, so the reported CPCV ICs were mildly
        optimistic.
    """
    if isinstance(dates, pd.Series):
        date_values = pd.to_datetime(dates.to_numpy())
    else:
        date_values = pd.to_datetime(np.asarray(dates))

    n_samples = len(date_values)
    if n_samples == 0:
        return

    # Position of each row's date in the trading calendar. Purge windows are
    # expressed in these positions so a "21-day" horizon means 21 trading days.
    _, row_pos = np.unique(date_values.to_numpy(), return_inverse=True)
    n_dates = int(row_pos.max()) + 1
    groups = _group_boundaries(n_samples, n_splits)
    horizon = max(int(horizon_days), 0)
    embargo = max(int(round(embargo_pct * n_dates)), 0)

    group_pos: list[tuple[int, int]] = []
    for start, end in groups:
        if end <= start:
            raise ValueError("empty CPCV group")
        group_pos.append((int(row_pos[start]), int(row_pos[end - 1])))

    for test_combo in combinations(range(n_splits), n_test_splits):
        test_rows = np.concatenate(
            [np.arange(groups[g][0], groups[g][1]) for g in test_combo]
        )

        keep_mask = np.ones(n_samples, dtype=bool)
        keep_mask[test_rows] = False
        # A training row at position p has label window (p, p + h]; a test
        # group spanning [lo, hi] has label windows ending at hi + h. They
        # overlap iff lo - h <= p <= hi + h. The embargo extends the
        # post-test side further.
        for g in test_combo:
            lo, hi = group_pos[g]
            forbidden = (row_pos >= lo - horizon) & (row_pos <= hi + horizon + embargo)
            keep_mask &= ~forbidden

        train_rows = np.flatnonzero(keep_mask)
        if len(train_rows) == 0 or len(test_rows) == 0:
            logger.debug(
                "Skipping CPCV combo %s — empty train/test after purge",
                test_combo,
            )
            continue

        yield CPCVSplit(
            train_idx=train_rows,
            test_idx=test_rows,
            test_group_ids=test_combo,
        )


def n_cpcv_paths(n_splits: int, n_test_splits: int) -> int:
    """
    How many CPCV *paths* you can reconstruct from the (N, K) partition.

    Each group appears in C(N-1, K-1) / C(N, K) fraction of test sets; stitching
    together groups from different splits into non-overlapping time sequences
    gives C(N, K) * K / N distinct full-history paths. This matches López de
    Prado's formula and is useful for sizing the deflated Sharpe calculation.
    """
    from math import comb

    return comb(n_splits, n_test_splits) * n_test_splits // n_splits
