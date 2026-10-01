"""
Leakage-safe train/test splits for panel (date x ticker) data.

Every horizon in this module is measured in **trading days**: positions in
the sorted array of unique panel dates, not calendar days. A 21-day forward
return label spans 21 *rows* of the trading calendar (~30 calendar days), so
purging "21 days" of calendar time leaves roughly a week of label overlap in
the training set.

Label timing convention (shared with ``backtest.walk_forward``)::

    decision at close of t  ->  enter at close of t + lag  ->  exit at close of t + lag + horizon

The label for decision date ``t`` is therefore only *observable* at the close
of ``t + lag + horizon``. A model that predicts at test date ``T`` may only be
trained on rows with ``pos(t) + lag + horizon + embargo <= pos(T)``.

The legacy walk-forward scripts trained on every row with ``t < T``. With
overlapping 21-day labels that hands the model ~20 rows whose labels are
nearly identical to the test label, which is how the repository reported a
walk-forward IC of +0.145 (t = 7.6) for a purely price-based model.
``purge=False`` reproduces that behavior so the leak can be measured, never
to produce results.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """One walk-forward step: fit on ``train_dates``, then predict ``test_date``."""

    test_date: pd.Timestamp
    train_dates: pd.DatetimeIndex


def last_trainable_position(
    test_pos: int,
    horizon: int,
    execution_lag: int = 0,
    embargo: int = 0,
) -> int:
    """
    Largest date position whose label is fully realized at ``test_pos``.

    A label for decision date ``t`` spans prices from ``t + lag`` to
    ``t + lag + horizon``, so it is known at the close of test date ``T`` iff
    ``pos(t) + lag + horizon <= pos(T)``. ``embargo`` removes further rows to
    guard against serial correlation that outlives the label window.
    """
    if horizon < 0 or execution_lag < 0 or embargo < 0:
        raise ValueError("horizon, execution_lag and embargo must be non-negative")
    return test_pos - horizon - execution_lag - embargo


def purged_walk_forward_splits(
    dates: Sequence | pd.DatetimeIndex,
    test_dates: Sequence | pd.DatetimeIndex,
    horizon: int,
    execution_lag: int = 0,
    embargo: int = 0,
    train_window: int | None = None,
    min_train_dates: int = 1,
    purge: bool = True,
) -> Iterator[WalkForwardSplit]:
    """
    Yield walk-forward splits whose training labels never overlap the test date.

    Args:
        dates: The trading calendar (unique dates; sorted internally).
        test_dates: Dates to predict. Each must be present in ``dates``.
        horizon: Label horizon in trading days.
        execution_lag: Trading days between the decision close and the entry close.
        embargo: Extra trading days removed between the train and test sets.
        train_window: If set, keep only the most recent ``train_window`` trainable
            dates (rolling window). ``None`` means an expanding window.
        min_train_dates: Skip test dates with fewer trainable dates than this
            (capped at ``train_window`` when a rolling window is used).
        purge: When ``False``, train on every date strictly before the test date.
            That is the legacy, leaky behavior; it exists only so the size of the
            leak can be measured.

    Yields:
        ``WalkForwardSplit`` records in chronological order.
    """
    calendar = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(dates))).sort_values()
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    if train_window is not None:
        if train_window < 1:
            raise ValueError("train_window must be >= 1")
        min_train_dates = min(min_train_dates, train_window)

    for test_date in pd.DatetimeIndex(test_dates).sort_values():
        if test_date not in positions.index:
            raise KeyError(f"test date {test_date.date()} is not in the trading calendar")
        test_pos = int(positions[test_date])

        if purge:
            hi = last_trainable_position(test_pos, horizon, execution_lag, embargo)
        else:
            hi = test_pos - 1
        if hi < 0:
            continue

        lo = 0 if train_window is None else max(0, hi - train_window + 1)
        if hi - lo + 1 < min_train_dates:
            continue

        yield WalkForwardSplit(
            test_date=test_date,
            train_dates=calendar[lo : hi + 1],
        )


def label_overlap_days(
    train_dates: pd.DatetimeIndex,
    test_date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    horizon: int,
    execution_lag: int = 0,
) -> int:
    """
    Number of trading days by which the latest training label overlaps the
    test label window. Zero means the split is leak-free. Used by tests and
    as a runtime assertion in the walk-forward harness.
    """
    if len(train_dates) == 0:
        return 0
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    last_train_end = int(positions[train_dates.max()]) + execution_lag + horizon
    return max(0, last_train_end - int(positions[test_date]))
