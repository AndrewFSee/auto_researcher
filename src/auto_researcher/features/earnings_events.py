"""
Point-in-time earnings events and features from SEC filings and EPS history.

Analyst estimate histories are not available point-in-time, so surprises use
the time-series definition of Bernard & Thomas (1989): the year-over-year
change in quarterly EPS, scaled by the volatility of past changes.

Announcement dates come from SEC filings. Large companies furnish an 8-K
(Item 2.02, "Results of Operations") on the day they release earnings, then
file the 10-Q/10-K days or weeks later. The filing index has no item numbers,
so for each periodic report the announcement is taken to be the *last* 8-K
filed between a week after the fiscal period end and the 10-Q/10-K filing,
preferring 8-Ks whose event date equals their filing date; without one, the
10-Q/10-K filing date is used. Choosing the last candidate errs late (safe)
rather than early (a leak). Against independent report dates for 500 large
caps (2021-2025) this rule is exact 71% of the time, a day early 15%, two or
more days early 2.9% (the reference itself is that early about 1.3% of the
time) and late 11% (median 8 days); the first-candidate rule was two or more
days early 31% of the time.

Timing: an announcement filed on day D is treated as public at the close of
the second trading day after D (``delay=2``), so a rare early pick still
cannot place a position before the market has seen the release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARNINGS_FEATURES = ("sue", "earnings_ann_return", "eps_beat_streak", "days_since_ann")


def announcement_dates(filings: pd.DataFrame, min_lag_days: int = 7) -> pd.DataFrame:
    """
    Infer one earnings announcement per periodic report.

    Args:
        filings: Columns ``symbol, form_type, filing_date, report_date``
            (SEC filing index, e.g. DefeatBeta ``stock_sec_filing``).
        min_lag_days: Ignore 8-Ks filed sooner than this after the period end.

    Returns:
        DataFrame with ``symbol, period_end, announce_date, source`` where
        ``source`` is ``"8-K"`` or ``"10-Q/10-K"`` (fallback).
    """
    f = filings[["symbol", "form_type", "filing_date", "report_date"]].copy()
    f["filing_date"] = pd.to_datetime(f["filing_date"], errors="coerce")
    f["report_date"] = pd.to_datetime(f["report_date"], errors="coerce")
    f = f.dropna(subset=["symbol", "filing_date"])

    periodic = f[f["form_type"].isin(["10-Q", "10-K"])].dropna(subset=["report_date"])
    periodic = periodic.rename(columns={"report_date": "period_end", "filing_date": "periodic_filed"})
    periodic = periodic.sort_values("periodic_filed").drop_duplicates(["symbol", "period_end"])
    eightk = f[f["form_type"] == "8-K"][["symbol", "filing_date", "report_date"]]
    eightk = eightk.assign(same_day=eightk["report_date"] == eightk["filing_date"])

    merged = periodic[["symbol", "period_end", "periodic_filed"]].merge(eightk, on="symbol", how="left")
    in_window = (merged["filing_date"] >= merged["period_end"] + pd.Timedelta(days=min_lag_days)) & (
        merged["filing_date"] <= merged["periodic_filed"]
    )
    # Same-day 8-Ks first, then the latest filing date within each group.
    cand = merged[in_window].sort_values(["symbol", "period_end", "same_day", "filing_date"],
                                         ascending=[True, True, False, False])
    chosen = cand.drop_duplicates(["symbol", "period_end"])[["symbol", "period_end", "filing_date"]]

    out = periodic[["symbol", "period_end", "periodic_filed"]].merge(
        chosen, on=["symbol", "period_end"], how="left"
    )
    out["source"] = np.where(out["filing_date"].notna(), "8-K", "10-Q/10-K")
    out["announce_date"] = out["filing_date"].fillna(out["periodic_filed"])
    return out[["symbol", "period_end", "announce_date", "source"]].sort_values(
        ["symbol", "announce_date"]
    ).reset_index(drop=True)


def time_series_sue(eps: pd.DataFrame, window: int = 8, min_obs: int = 4) -> pd.DataFrame:
    """
    Seasonal-random-walk surprise per fiscal quarter.

    ``SUE_q = (EPS_q - EPS_{q-4}) / std(EPS_j - EPS_{j-4}, j = q-window..q-1)``

    Args:
        eps: Columns ``symbol, period_end, eps`` (quarterly EPS by fiscal period end).

    Returns:
        Columns ``symbol, period_end, eps, sue, beat`` (``beat``: EPS above the
        same quarter a year earlier).
    """
    e = eps[["symbol", "period_end", "eps"]].copy()
    e["period_end"] = pd.to_datetime(e["period_end"], errors="coerce")
    e["eps"] = pd.to_numeric(e["eps"], errors="coerce")
    e = e.dropna().sort_values(["symbol", "period_end"]).drop_duplicates(["symbol", "period_end"])
    g = e.groupby("symbol")["eps"]
    e["d4"] = e["eps"] - g.shift(4)
    prior_sd = e.groupby("symbol")["d4"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_obs).std()
    )
    e["sue"] = e["d4"] / prior_sd.where(prior_sd > 0)
    e["beat"] = (e["d4"] > 0).astype(float).where(e["d4"].notna())
    return e[["symbol", "period_end", "eps", "sue", "beat"]]


def earnings_feature_panel(
    events: pd.DataFrame,
    close: pd.DataFrame,
    benchmark: str = "SPY",
    stale_after: int = 126,
    delay: int = 2,
) -> pd.DataFrame:
    """
    Daily point-in-time earnings features for each (date, ticker).

    Args:
        events: Columns ``symbol, announce_date, sue, beat`` (one row per quarter).
        close: Adjusted closes (date x ticker) including ``benchmark``.
        stale_after: Trading days after which the last announcement is ignored.
        delay: An announcement filed on day D becomes public at the close of the
            ``delay``-th trading day after D.

    Returns:
        Long frame indexed by ``(date, ticker)`` with ``EARNINGS_FEATURES``,
        each a per-date rank in [-0.5, 0.5]; missing values are 0 (neutral).
        ``earnings_ann_return`` is the stock's return minus the benchmark's
        from the close before the filing day to the close when it becomes public.
    """
    cal = close.index
    bench = close[benchmark].to_numpy()
    tickers = [t for t in close.columns if t != benchmark]
    shape = (len(cal), len(tickers))
    sue = np.full(shape, np.nan)
    ear = np.full(shape, np.nan)
    streak = np.full(shape, np.nan)
    since = np.full(shape, np.nan)

    by_symbol = {s: g.sort_values("announce_date") for s, g in events.groupby("symbol")}
    for j, ticker in enumerate(tickers):
        g = by_symbol.get(ticker)
        if g is None:
            continue
        px = close[ticker].to_numpy()
        beats = g["beat"].to_numpy()
        for k, row in enumerate(g.itertuples(index=False)):
            ann = pd.Timestamp(row.announce_date)
            d = cal.searchsorted(ann)  # first trading day on or after the filing day
            on_trading_day = d < len(cal) and cal[d] == ann
            avail = d + delay if on_trading_day else d + delay - 1
            if avail >= len(cal) or d < 1:
                continue
            # Later announcements overwrite these values from their own availability date.
            end = min(avail + stale_after, len(cal))
            reaction = np.nan
            if np.isfinite(px[d - 1]) and np.isfinite(px[avail]):
                reaction = (px[avail] / px[d - 1] - 1) - (bench[avail] / bench[d - 1] - 1)
            recent = beats[max(0, k - 3): k + 1]
            sue[avail:end, j] = row.sue
            ear[avail:end, j] = reaction
            streak[avail:end, j] = np.nansum(recent) if np.isfinite(recent).any() else np.nan
            since[avail:end, j] = np.arange(end - avail)

    def ranked(values: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(values, index=cal, columns=tickers)
        return (frame.rank(axis=1, pct=True) - 0.5).fillna(0.0)

    panel = pd.concat(
        {
            "sue": ranked(sue).stack(),
            "earnings_ann_return": ranked(ear).stack(),
            "eps_beat_streak": ranked(streak).stack(),
            "days_since_ann": ranked(since).stack(),
        },
        axis=1,
    )
    panel.index = panel.index.set_names(["date", "ticker"])
    return panel


def attach_surprises(
    announcements: pd.DataFrame,
    surprises: pd.DataFrame,
    tolerance_days: int = 10,
) -> pd.DataFrame:
    """
    Join announcement dates to per-quarter surprises by symbol and fiscal period end.

    Period ends are matched to the nearest date within ``tolerance_days``,
    because 52/53-week fiscal calendars put the 10-Q's period end a few days
    away from the calendar quarter end used in EPS histories.

    Returns:
        Columns ``symbol, announce_date, period_end, sue, beat``.
    """
    left = announcements.dropna(subset=["period_end"]).copy()
    left["period_end"] = pd.to_datetime(left["period_end"]).astype("datetime64[ns]")
    left = left.sort_values("period_end")
    right = surprises.rename(columns={"period_end": "eps_period_end"}).copy()
    right["eps_period_end"] = pd.to_datetime(right["eps_period_end"]).astype("datetime64[ns]")
    right = right.sort_values("eps_period_end")
    merged = pd.merge_asof(
        left,
        right[["symbol", "eps_period_end", "sue", "beat"]],
        left_on="period_end",
        right_on="eps_period_end",
        by="symbol",
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    return merged[["symbol", "announce_date", "period_end", "sue", "beat"]].sort_values(
        ["symbol", "announce_date"]
    ).reset_index(drop=True)
