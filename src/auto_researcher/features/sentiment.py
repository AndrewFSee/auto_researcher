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


def _load_daily_sentiment_from_db(
    tickers: list[str],
    start_date: str,
    end_date: str,
    db_path: Path | str = _DEFAULT_NEWS_DB,
) -> pd.DataFrame:
    """
    Load per-ticker daily aggregated sentiment from news.db.

    Aggregates article-level FinBERT scores into daily statistics:
    mean sentiment, article count, and positive/negative ratio.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        db_path: Path to news.db SQLite file.

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

    for ticker in tickers:
        ticker_data = df[df["ticker"] == ticker].set_index("date")
        if ticker_data.empty:
            continue

        # Reindex to trading days, forward-fill sentiment (decays over weekends)
        ticker_data = ticker_data.reindex(trading_dates)
        # Article count should be 0 on days with no news, not forward-filled
        article_count = ticker_data["article_count"].fillna(0)
        # Sentiment scores forward-fill (last known sentiment persists)
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
) -> pd.DataFrame:
    """
    Load earnings call FinBERT sentiment from sentiment_500.csv.

    This is a quarterly signal that gets forward-filled to daily frequency.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        csv_path: Path to sentiment CSV.

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

    for ticker in tickers:
        ticker_data = df[df["symbol"] == ticker].set_index("report_date")
        if ticker_data.empty:
            continue

        # Reindex to trading days and forward-fill (quarterly signal)
        earnings_sent = ticker_data["finbert_sentiment"].reindex(trading_dates).ffill()
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
