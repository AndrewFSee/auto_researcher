"""
Sentiment feature engineering.

This module computes sentiment features from the news.db SQLite database
(FinBERT scores) and sentiment_500.csv (earnings call sentiment). Features
include rolling averages, sentiment momentum, dispersion, and topic-weighted
composites.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_NEWS_DB = _PROJECT_ROOT / "data" / "news.db"
_DEFAULT_SENTIMENT_CSV = _PROJECT_ROOT / "data" / "sentiment_500.csv"
_DEFAULT_TOPIC_IC = _PROJECT_ROOT / "data" / "topic_ic.json"


_DEFAULT_SENTIMENT_EMBARGO_DAYS = 1
_DEFAULT_EARNINGS_FFILL_LIMIT = 30


def _load_daily_sentiment_from_db(
    tickers: list[str],
    start_date: str,
    end_date: str,
    db_path: Path | str = _DEFAULT_NEWS_DB,
    embargo_days: int = _DEFAULT_SENTIMENT_EMBARGO_DAYS,
) -> pd.DataFrame:
    """
    Load per-ticker daily aggregated sentiment from news.db.

    Aggregates article-level FinBERT scores into daily statistics:
    mean sentiment, article count, and positive/negative ratio.

    Causality
    ---------
    Articles published on day D are treated as known only from day D+embargo
    onward. The aggregated daily series is shifted by ``embargo_days`` before
    being forward-filled, so the value at feature-date t never references an
    article that was published on/after t - embargo.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        db_path: Path to news.db SQLite file.
        embargo_days: Bars to lag every sentiment series. Default 1 — a
            Friday-afternoon article lands in Monday's feature, not Friday's.

    Returns:
        DataFrame with MultiIndex columns (ticker, feature).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning(f"News database not found at {db_path}")
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    try:
        placeholders = ",".join("?" for _ in tickers)
        query = f"""
            SELECT
                ticker,
                DATE(published_date) as date,
                AVG(sentiment_score) as mean_sentiment,
                COUNT(*) as article_count,
                AVG(CASE WHEN sentiment_label = 'positive' THEN 1.0
                         WHEN sentiment_label = 'negative' THEN -1.0
                         ELSE 0.0 END) as label_score,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 1.0
                    / COUNT(*) as pos_ratio
            FROM articles
            WHERE ticker IN ({placeholders})
              AND DATE(published_date) >= ?
              AND DATE(published_date) <= ?
              AND sentiment_score IS NOT NULL
            GROUP BY ticker, DATE(published_date)
        """
        params = list(tickers) + [start_date, end_date]
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        logger.warning("No sentiment data found in news.db for requested tickers/dates")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])

    # Pivot to wide format: one column per (ticker, metric)
    features = {}
    trading_dates = pd.date_range(start_date, end_date, freq="B")

    shift_bars = max(int(embargo_days), 0)

    for ticker in tickers:
        ticker_data = df[df["ticker"] == ticker].set_index("date")
        if ticker_data.empty:
            continue

        # Reindex to trading days. Shift by the embargo BEFORE forward-fill so
        # that an article published on D only becomes observable on D+embargo.
        ticker_data = ticker_data.reindex(trading_dates)
        if shift_bars:
            ticker_data = ticker_data.shift(shift_bars)
        # Article count should be 0 on days with no news, not forward-filled.
        article_count = ticker_data["article_count"].fillna(0)
        # Sentiment scores forward-fill (last known sentiment persists).
        mean_sent = ticker_data["mean_sentiment"].ffill()
        label_score = ticker_data["label_score"].ffill()
        pos_ratio = ticker_data["pos_ratio"].ffill()

        features[(ticker, "sent_raw")] = mean_sent
        features[(ticker, "sent_label")] = label_score
        features[(ticker, "sent_pos_ratio")] = pos_ratio
        features[(ticker, "sent_article_count")] = article_count

    if not features:
        return pd.DataFrame()

    result = pd.DataFrame(features, index=trading_dates)
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=["ticker", "feature"]
    )
    return result


def _compute_rolling_sentiment_features(
    daily_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rolling windows over daily sentiment: 5d, 10d, 20d averages,
    sentiment momentum (5d - 20d), and dispersion (rolling std).

    Args:
        daily_sentiment: Output of _load_daily_sentiment_from_db.

    Returns:
        DataFrame with additional rolling features appended.
    """
    if daily_sentiment.empty:
        return daily_sentiment

    new_features = {}
    tickers = daily_sentiment.columns.get_level_values("ticker").unique()

    for ticker in tickers:
        raw = daily_sentiment.get((ticker, "sent_raw"))
        if raw is None:
            continue

        # Rolling averages
        for window in [5, 10, 20]:
            rolled = raw.rolling(window=window, min_periods=max(1, window // 2)).mean()
            new_features[(ticker, f"sent_ma{window}")] = rolled

        # Sentiment momentum: short-term vs long-term
        ma5 = raw.rolling(window=5, min_periods=2).mean()
        ma20 = raw.rolling(window=20, min_periods=5).mean()
        new_features[(ticker, "sent_momentum")] = ma5 - ma20

        # Sentiment dispersion (rolling std of raw scores)
        new_features[(ticker, "sent_dispersion")] = raw.rolling(
            window=20, min_periods=5
        ).std()

        # Article count momentum (news intensity change)
        count = daily_sentiment.get((ticker, "sent_article_count"))
        if count is not None:
            count_ma5 = count.rolling(window=5, min_periods=1).mean()
            count_ma20 = count.rolling(window=20, min_periods=5).mean()
            new_features[(ticker, "sent_news_intensity")] = count_ma5 / count_ma20.replace(0, np.nan)

    if not new_features:
        return daily_sentiment

    new_df = pd.DataFrame(new_features, index=daily_sentiment.index)
    new_df.columns = pd.MultiIndex.from_tuples(
        new_df.columns, names=["ticker", "feature"]
    )
    return pd.concat([daily_sentiment, new_df], axis=1)


def _load_earnings_sentiment(
    tickers: list[str],
    start_date: str,
    end_date: str,
    csv_path: Path | str = _DEFAULT_SENTIMENT_CSV,
    embargo_days: int = _DEFAULT_SENTIMENT_EMBARGO_DAYS,
    ffill_limit: int = _DEFAULT_EARNINGS_FFILL_LIMIT,
) -> pd.DataFrame:
    """
    Load earnings call FinBERT sentiment from sentiment_500.csv.

    This is a quarterly signal that gets forward-filled to daily frequency,
    then capped so a stale earnings tone doesn't persist for months after the
    report. The report date itself is also embargoed by ``embargo_days`` so a
    call released after market close leaks into the NEXT trading day.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        csv_path: Path to sentiment CSV.
        embargo_days: Bars to shift the report-date series forward.
        ffill_limit: Max days a single report's tone is allowed to persist.

    Returns:
        DataFrame with MultiIndex columns (ticker, feature).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning(f"Earnings sentiment CSV not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()

    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df[(df["report_date"] >= start_date) & (df["report_date"] <= end_date)]

    trading_dates = pd.date_range(start_date, end_date, freq="B")
    features = {}

    shift_bars = max(int(embargo_days), 0)
    ffill_cap = max(int(ffill_limit), 1)

    for ticker in tickers:
        ticker_data = df[df["symbol"] == ticker].set_index("report_date")
        if ticker_data.empty:
            continue

        # Reindex to trading days, embargo the report date by `shift_bars`,
        # then cap the forward-fill so a single call's tone only persists
        # `ffill_cap` trading days (prevents a stale quarterly signal from
        # silently bleeding across the whole quarter).
        earnings_sent = ticker_data["finbert_sentiment"].reindex(trading_dates)
        if shift_bars:
            earnings_sent = earnings_sent.shift(shift_bars)
        earnings_sent = earnings_sent.ffill(limit=ffill_cap)
        features[(ticker, "earnings_finbert")] = earnings_sent

    if not features:
        return pd.DataFrame()

    result = pd.DataFrame(features, index=trading_dates)
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=["ticker", "feature"]
    )
    return result


def compute_news_sentiment(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Compute news sentiment scores for tickers from news.db.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date for sentiment data.
        end_date: End date for sentiment data.

    Returns:
        Sentiment scores DataFrame with rolling features.
    """
    daily = _load_daily_sentiment_from_db(tickers, start_date, end_date)
    if daily.empty:
        return daily
    return _compute_rolling_sentiment_features(daily)


def _load_article_level_sentiment(
    tickers: list[str],
    start_date: str,
    end_date: str,
    db_path: Path | str = _DEFAULT_NEWS_DB,
) -> pd.DataFrame:
    """Pull article-level rows for topic-weighted sentiment.

    Returns a DataFrame with columns
    ``ticker, title, published_date, sentiment_score``. Empty if the
    db file is missing.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(
            columns=["ticker", "title", "published_date", "sentiment_score"]
        )

    conn = sqlite3.connect(str(db_path))
    try:
        placeholders = ",".join("?" for _ in tickers)
        query = f"""
            SELECT ticker, title, published_date, sentiment_score
            FROM articles
            WHERE ticker IN ({placeholders})
              AND DATE(published_date) >= ?
              AND DATE(published_date) <= ?
              AND sentiment_score IS NOT NULL
              AND title IS NOT NULL
        """
        params = list(tickers) + [start_date, end_date]
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if not df.empty:
        df["published_date"] = pd.to_datetime(df["published_date"])
    return df


def compute_topic_weighted_sentiment(
    tickers: list[str],
    start_date: str,
    end_date: str,
    forward_returns: pd.DataFrame,
    calibration_cutoff: pd.Timestamp | str,
    db_path: Path | str = _DEFAULT_NEWS_DB,
    embargo_days: int = _DEFAULT_SENTIMENT_EMBARGO_DAYS,
    bertopic_config: "RollingBERTopicConfig | None" = None,  # noqa: F821
    ic_config: "TopicICConfig | None" = None,  # noqa: F821
) -> pd.DataFrame:
    """Per-ticker daily topic-weighted sentiment.

    For each article published strictly before ``calibration_cutoff``, we
    (a) assign a topic via ``RollingBERTopic`` fit on the trailing window
    that ends just before the article's date, and (b) record
    ``sentiment × topic_weight`` where the topic weight is the signed IC
    of that topic measured on articles before ``calibration_cutoff``.

    Articles published on or after ``calibration_cutoff`` apply the
    **frozen** calibrator — no future data leaks into the weights.

    Causality invariants
    --------------------
    * Topic assignments use ``RollingBERTopic``, which fits each snapshot
      on articles strictly before its ``fit_date``.
    * Per-topic IC weights are computed from articles whose
      ``published_date < calibration_cutoff`` using their realized
      ``forward_returns``. Rows after the cutoff re-use those frozen
      weights.
    * The resulting daily series is shifted by ``embargo_days`` the same
      way ``_load_daily_sentiment_from_db`` shifts ``sent_raw`` — so a
      post-close article lands in the next-day feature, not today's.

    Args:
        tickers: List of ticker symbols.
        start_date, end_date: Date range (YYYY-MM-DD).
        forward_returns: DataFrame indexed by date with one column per
            ticker, containing the realized forward return used for IC
            calibration. Only rows before ``calibration_cutoff`` are
            consulted.
        calibration_cutoff: Timestamp separating calibration from
            inference. Must be ≤ ``end_date``.
        db_path: Path to news.db.
        embargo_days: Bars to lag the emitted daily series.
        bertopic_config: Override for topic-model refit cadence.
        ic_config: Override for IC floor / min obs.

    Returns:
        DataFrame with MultiIndex columns ``(ticker, feature)`` containing
        the single column ``sent_topic_weighted`` per ticker.  Empty if
        news.db has no rows in the window.
    """
    from auto_researcher.models.bertopic_sentiment import (
        RollingBERTopic,
        RollingBERTopicConfig,
    )
    from auto_researcher.models.topic_ic_calibrator import (
        TopicICCalibrator,
        TopicICConfig,
    )

    cutoff = pd.to_datetime(calibration_cutoff)
    articles = _load_article_level_sentiment(tickers, start_date, end_date, db_path)
    if articles.empty:
        logger.info("No article-level rows for topic-weighted sentiment; skipping")
        return pd.DataFrame()

    # Step 1 — rolling topic assignment on ALL articles. The model itself
    # is causal (fit windows are strictly pre-fit-date), so passing the
    # full range is fine; the calibrator handles the train/inference
    # separation.
    rb_cfg = bertopic_config or RollingBERTopicConfig(
        refit_every=90, lookback_days=365, max_topics=8, min_topic_size=8
    )
    # Use the lightweight backend — the real BERTopic install is optional,
    # and callers who want the full model can pass a pre-fit RollingBERTopic
    # via a future override.
    rb = RollingBERTopic(config=rb_cfg, backend="lightweight")
    rb.fit(articles, date_col="published_date", text_col="title")
    topic_ids = rb.transform(articles, date_col="published_date", text_col="title")
    articles = articles.assign(topic=topic_ids.to_numpy())

    # Step 2 — build calibration set: each pre-cutoff article paired with
    # the realized forward return for its ticker on the publication date.
    cal_mask = articles["published_date"] < cutoff
    cal_articles = articles.loc[cal_mask & (articles["topic"] >= 0)]
    if cal_articles.empty:
        logger.warning(
            "No pre-cutoff articles with assigned topics for calibration; "
            "topic-weighted sentiment will be all zeros"
        )
        return pd.DataFrame()

    fwd = forward_returns.copy()
    fwd.index = pd.to_datetime(fwd.index)
    dates = cal_articles["published_date"].dt.normalize()
    fwd_per_article = np.array(
        [
            fwd.loc[d, t] if (d in fwd.index and t in fwd.columns) else np.nan
            for d, t in zip(dates, cal_articles["ticker"])
        ]
    )
    usable = np.isfinite(fwd_per_article)
    if usable.sum() < 50:
        logger.warning(
            "Only %d usable (article, return) calibration pairs; weights "
            "will be unreliable", int(usable.sum()),
        )

    cal = TopicICCalibrator(ic_config or TopicICConfig())
    cal.fit(
        topic_ids=cal_articles["topic"].to_numpy()[usable],
        sentiment=cal_articles["sentiment_score"].to_numpy()[usable],
        forward_return=fwd_per_article[usable],
    )

    # Step 3 — apply the frozen weights to EVERY article (pre + post
    # cutoff), then aggregate to daily per ticker.
    contrib = cal.apply(
        topic_ids=articles["topic"].to_numpy(),
        sentiment=articles["sentiment_score"].to_numpy(),
    )
    articles = articles.assign(weighted_contrib=contrib)

    articles["date"] = articles["published_date"].dt.normalize()
    daily = (
        articles.groupby(["ticker", "date"])["weighted_contrib"]
        .mean()
        .unstack("ticker")
    )

    trading_dates = pd.date_range(start_date, end_date, freq="B")
    daily = daily.reindex(trading_dates)
    shift_bars = max(int(embargo_days), 0)
    if shift_bars:
        daily = daily.shift(shift_bars)
    daily = daily.ffill()

    # Emit as MultiIndex (ticker, feature) to match the rest of the module.
    out = pd.DataFrame(
        {(tkr, "sent_topic_weighted"): daily[tkr] for tkr in daily.columns if tkr in daily},
        index=trading_dates,
    )
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["ticker", "feature"])
    logger.info(
        "Topic-weighted sentiment: %d tickers, %d topics kept (%d contrarian)",
        out.columns.get_level_values("ticker").nunique(),
        len(cal.signed_weights()),
        sum(1 for w in cal.signed_weights().values() if w < 0),
    )
    return out


def compute_social_sentiment(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Compute social media sentiment scores.

    Not yet implemented -- requires social media data source.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date.
        end_date: End date.

    Returns:
        Empty DataFrame (no social data source available).
    """
    logger.debug("Social sentiment not available - no data source configured")
    return pd.DataFrame()


def compute_earnings_call_sentiment(
    tickers: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute sentiment from earnings call transcripts.

    Uses sentiment_500.csv FinBERT scores, forward-filled quarterly.

    Args:
        tickers: List of ticker symbols.
        dates: Dates for which to compute sentiment.

    Returns:
        Earnings call sentiment DataFrame.
    """
    if len(dates) < 2:
        return pd.DataFrame(index=dates)
    start = dates.min().strftime("%Y-%m-%d")
    end = dates.max().strftime("%Y-%m-%d")
    return _load_earnings_sentiment(tickers, start, end)


def compute_all_sentiment_features(
    tickers: list[str],
    start_date: str,
    end_date: str,
    news_db_path: Path | str = _DEFAULT_NEWS_DB,
    sentiment_csv_path: Path | str = _DEFAULT_SENTIMENT_CSV,
    use_topic_weighted: bool = False,
    topic_weighted_forward_returns: pd.DataFrame | None = None,
    topic_weighted_cutoff: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Compute all sentiment features for the ML feature pipeline.

    Combines:
    - News sentiment from news.db (daily FinBERT scores + rolling features)
    - Earnings call sentiment from sentiment_500.csv (quarterly FinBERT)

    Features produced per ticker:
    - sent_raw: Daily mean FinBERT sentiment score
    - sent_label: Daily mean sentiment label score (-1/0/+1)
    - sent_pos_ratio: Fraction of positive articles
    - sent_article_count: Daily article count
    - sent_ma5/10/20: Rolling sentiment averages
    - sent_momentum: Short-term minus long-term sentiment
    - sent_dispersion: Rolling sentiment std (disagreement)
    - sent_news_intensity: Article count momentum
    - earnings_finbert: Quarterly earnings call sentiment

    Args:
        tickers: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        news_db_path: Path to news.db SQLite file.
        sentiment_csv_path: Path to sentiment_500.csv.

    Returns:
        DataFrame with MultiIndex columns (ticker, feature).
    """
    parts = []

    # News sentiment features
    daily = _load_daily_sentiment_from_db(tickers, start_date, end_date, news_db_path)
    if not daily.empty:
        news_features = _compute_rolling_sentiment_features(daily)
        parts.append(news_features)
        n_tickers = news_features.columns.get_level_values("ticker").nunique()
        logger.info(
            f"Loaded news sentiment for {n_tickers} tickers "
            f"({len(news_features.columns)} features)"
        )

    # Earnings call sentiment
    earnings = _load_earnings_sentiment(
        tickers, start_date, end_date, sentiment_csv_path
    )
    if not earnings.empty:
        parts.append(earnings)
        logger.info(
            f"Loaded earnings sentiment for "
            f"{earnings.columns.get_level_values('ticker').nunique()} tickers"
        )

    # Topic-weighted sentiment — only when the caller supplies both the
    # realized-return series and the calibration cutoff. These are
    # required because the calibrator needs a strictly-past training
    # sample; we refuse to fabricate them.
    if use_topic_weighted:
        if topic_weighted_forward_returns is None or topic_weighted_cutoff is None:
            logger.warning(
                "use_topic_weighted=True but forward_returns / cutoff missing; "
                "skipping topic-weighted sentiment feature"
            )
        else:
            topic_weighted = compute_topic_weighted_sentiment(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                forward_returns=topic_weighted_forward_returns,
                calibration_cutoff=topic_weighted_cutoff,
                db_path=news_db_path,
            )
            if not topic_weighted.empty:
                parts.append(topic_weighted)

    if not parts:
        logger.warning("No sentiment data available for requested tickers/dates")
        return pd.DataFrame()

    if len(parts) == 1:
        return parts[0]

    # Align on common dates
    result = parts[0]
    for other in parts[1:]:
        common = result.index.intersection(other.index)
        result = pd.concat([result.loc[common], other.loc[common]], axis=1)

    return result
