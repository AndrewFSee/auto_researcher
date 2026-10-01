"""
Point-in-time news-sentiment signals from scored articles (``data/news.db``).

Timing: article timestamps carry no timezone, so an article dated calendar day
``d`` is treated as usable only from the close of the first trading day
*after* ``d``, safe whether timestamps are UTC or US Eastern and whether the
article appeared before or after the close. Windows are measured in calendar
days back from each trading day's usable cut-off, so a stock with no recent
news has a missing value rather than a stale one.

Signals (per ticker and trading day):

* ``sent_7d``  mean FinBERT score of articles in the last 7 calendar days
  (the live sentiment agent's news window)
* ``sent_30d`` mean over the last 30 days (the agent's scraped-news window)
* ``sent_change`` ``sent_7d - sent_30d``
* ``news_surge`` log(article count in the last 7 days / weekly average of the
  last 90 days): attention
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEWS_SIGNALS = ("sent_7d", "sent_30d", "sent_change", "news_surge")


def load_articles(db_path: str, start: str | None = None) -> pd.DataFrame:
    """Scored articles as ``ticker, published, score`` (FinBERT positive minus negative)."""
    import sqlite3

    query = (
        "SELECT ticker, published_date, sentiment_score FROM articles "
        "WHERE sentiment_score IS NOT NULL AND published_date IS NOT NULL"
    )
    params: list[str] = []
    if start:
        query += " AND published_date >= ?"
        params.append(start)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, params=params)
    df["published"] = pd.to_datetime(df["published_date"], errors="coerce")
    return df.dropna(subset=["published"]).rename(columns={"sentiment_score": "score"})[
        ["ticker", "published", "score"]
    ]


def news_signal_panel(
    articles: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    usable_same_day: bool = False,
) -> pd.DataFrame:
    """
    Daily point-in-time news signals indexed by ``(date, ticker)``.

    Args:
        articles: Columns ``ticker, published, score``.
        calendar: Trading days on which to evaluate the signals.
        usable_same_day: Use articles dated ``d`` at the close of ``d``, the
            legacy alignment, which leaks after-close news. For measuring the
            leak only.

    Returns:
        DataFrame with ``NEWS_SIGNALS`` columns (raw values, NaN when the
        window has no articles).
    """
    calendar = pd.DatetimeIndex(calendar).sort_values()
    a = articles[["ticker", "score"]].copy()
    a["day"] = articles["published"].dt.normalize()
    a = a[a["day"] < calendar.max()]

    # Daily (calendar-day) sums per ticker, then calendar-time rolling windows.
    daily = a.groupby(["ticker", "day"]).agg(total=("score", "sum"), n=("score", "size"))
    out = []
    for ticker, g in daily.groupby(level="ticker"):
        g = g.droplevel("ticker")
        full = g.reindex(pd.date_range(g.index.min(), calendar.max(), freq="D"), fill_value=0)
        # Days before a ticker's first article simply have no articles, so
        # partial windows are valid (min_periods=1).
        s7, n7 = full["total"].rolling(7, min_periods=1).sum(), full["n"].rolling(7, min_periods=1).sum()
        s30 = full["total"].rolling(30, min_periods=1).sum()
        n30 = full["n"].rolling(30, min_periods=1).sum()
        n90 = full["n"].rolling(90, min_periods=1).sum()
        days90 = pd.Series(1.0, index=full.index).rolling(90, min_periods=1).sum()
        frame = pd.DataFrame({
            "sent_7d": s7 / n7.where(n7 > 0),
            "sent_30d": s30 / n30.where(n30 > 0),
            "news_surge": np.log((n7 + 1) / (n90 / (days90 / 7) + 1)),
        })
        frame["sent_change"] = frame["sent_7d"] - frame["sent_30d"]
        # A trading day t may use calendar days up to (but excluding) t itself:
        # the window ending on the previous calendar day.
        lagged = frame if usable_same_day else frame.shift(1)
        lagged = lagged.reindex(calendar[calendar >= g.index.min()])
        lagged["ticker"] = ticker
        out.append(lagged)

    if not out:
        return pd.DataFrame(columns=list(NEWS_SIGNALS),
                            index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
    panel = pd.concat(out).rename_axis("date").set_index("ticker", append=True)
    return panel[list(NEWS_SIGNALS)].sort_index()
