"""
Sanity checks for event-study date columns.

Earnings datasets often carry the *fiscal period end* (e.g. 2024-03-31) next
to, or instead of, the *announcement date* (~3-6 weeks later). Measuring
forward returns from the period end puts the announcement-day reaction inside
the window, so any surprise signal looks strongly predictive. This is exactly
how the repository once reported a PEAD IC of +0.22.
"""

from __future__ import annotations

import pandas as pd


class PeriodEndDateError(ValueError):
    """Raised when event dates look like fiscal period ends rather than announcements."""


def month_end_fraction(dates: pd.Series | pd.DatetimeIndex) -> float:
    """Fraction of (non-missing) dates that fall on a calendar month end."""
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna()))
    if len(d) == 0:
        return 0.0
    return float(d.is_month_end.mean())


def assert_announcement_dates(
    dates: pd.Series | pd.DatetimeIndex,
    threshold: float = 0.9,
    name: str = "event dates",
) -> None:
    """
    Raise ``PeriodEndDateError`` if ``dates`` are almost all month ends.

    Real announcement dates spread across the month (well under 10% land on a
    month end); fiscal period ends are ~100% month ends.
    """
    frac = month_end_fraction(dates)
    if frac >= threshold:
        raise PeriodEndDateError(
            f"{frac:.0%} of {name} fall on a calendar month end, so they look like "
            "fiscal period ends, not announcement dates. Forward returns measured "
            "from them include the announcement reaction (look-ahead). Re-key the "
            "events to report dates first (see scripts/pead_event_study.py)."
        )
