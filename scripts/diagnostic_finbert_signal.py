"""
Replicate the Johnsen/Shasharina (Apr 2026) FinBERT anti-signal diagnostic.

Question: does FinBERT-positive-labeled article-days predict positive next-day
returns, or is the feature an anti-signal as they reported?

Their numbers (for reference):
    - Pearson corr(mean_sentiment, lag+1 return):       -0.045
    - Mean lag+1 return on positive-labeled days:       -0.38%
    - Cross-sector pos-minus-neg return spread at lag+1: +0.28pp
      (FinBERT alone: +0.154pp, hybrid: +0.754pp, pure-Claude: +1.757pp in P3)
    - Signal decay: lag+1 ~ real, lag+2 ~ zero

We use the project's existing news.db (article-level FinBERT scores) and the
price cache (for returns). No re-scoring; this is a pure measurement on what
the pipeline already wrote down.

Outputs: prints a markdown-style report to stdout, writes JSON to
results/diagnostic_finbert_signal.json for record-keeping.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NEWS_DB = _PROJECT_ROOT / "data" / "news.db"
_PRICES = _PROJECT_ROOT / "data" / "price_cache" / "prices_2021-01-01_2026-01-15.parquet"
_RESULTS = _PROJECT_ROOT / "results" / "diagnostic_finbert_signal.json"


def load_article_days(
    tickers: list[str],
    db_path: Path,
    title_like: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (ticker, date): article counts by label + mean score.

    ``title_like`` optionally restricts to articles whose title matches ANY of
    the supplied SQL LIKE patterns (e.g. ``["%earnings call transcript%"]``).
    """
    placeholders = ",".join("?" for _ in tickers)
    title_clause = ""
    title_params: list[str] = []
    if title_like:
        ors = " OR ".join("LOWER(title) LIKE ?" for _ in title_like)
        title_clause = f" AND ({ors})"
        title_params = [p.lower() for p in title_like]
    query = f"""
        SELECT
            ticker,
            DATE(published_date) AS date,
            COUNT(*) AS n_articles,
            AVG(sentiment_score) AS mean_score,
            SUM(CASE WHEN sentiment_label='positive' THEN 1 ELSE 0 END) AS n_pos,
            SUM(CASE WHEN sentiment_label='negative' THEN 1 ELSE 0 END) AS n_neg,
            SUM(CASE WHEN sentiment_label='neutral'  THEN 1 ELSE 0 END) AS n_neu
        FROM articles
        WHERE ticker IN ({placeholders})
          AND sentiment_score IS NOT NULL
          AND published_date IS NOT NULL
          {title_clause}
        GROUP BY ticker, DATE(published_date)
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=list(tickers) + title_params)
    df["date"] = pd.to_datetime(df["date"])
    # Dominant label: whichever count is strictly largest. Ties → neutral.
    pos, neg, neu = df["n_pos"].values, df["n_neg"].values, df["n_neu"].values
    is_pos = (pos > neg) & (pos > neu)
    is_neg = (neg > pos) & (neg > neu)
    df["day_label"] = np.where(is_pos, "positive",
                       np.where(is_neg, "negative", "neutral"))
    return df


def load_returns(prices_path: Path) -> pd.DataFrame:
    """Returns at close-to-close. Wide frame: index=date, cols=ticker."""
    px = pd.read_parquet(prices_path)
    close = px["Adj Close"].copy() if "Adj Close" in px.columns.get_level_values(0) else px["Close"].copy()
    close.index = pd.to_datetime(close.index)
    return close.pct_change()


def align_and_score(
    articles: pd.DataFrame,
    returns: pd.DataFrame,
    lag: int,
) -> pd.DataFrame:
    """For each (ticker, date), attach return(date + lag trading days)."""
    # Shift returns by -lag so that returns_shifted.loc[D] = realized return over (D, D+lag].
    # lag=1 → next-day close-to-close return realized at D+1.
    r_shift = returns.shift(-lag)
    rows = []
    trading_dates = returns.index
    # Map article dates to the *next* trading day if pub date is weekend/holiday.
    # We want "return available after publishing", aligned to the trading calendar.
    trading_index = trading_dates.searchsorted(articles["date"].values, side="left")
    mask = trading_index < len(trading_dates)
    articles = articles.loc[mask].copy()
    articles["trading_date"] = trading_dates[trading_index[mask]]
    for (tkr, dt), grp in articles.groupby(["ticker", "trading_date"]):
        if tkr not in r_shift.columns:
            continue
        r = r_shift.at[dt, tkr]
        if pd.isna(r):
            continue
        rows.append({
            "ticker": tkr,
            "trading_date": dt,
            "mean_score": grp["mean_score"].iloc[0],
            "day_label":  grp["day_label"].iloc[0],
            "n_articles": int(grp["n_articles"].iloc[0]),
            f"ret_lag{lag}": float(r),
        })
    return pd.DataFrame(rows)


def diagnostic(
    tickers: list[str],
    start: str,
    end: str,
    db_path: Path,
    prices_path: Path,
    title_like: list[str] | None = None,
    label: str = "ALL",
) -> dict:
    print(f"\n=== Slice: {label} ===")
    print(f"Loading article-days for {len(tickers)} tickers from {db_path.name}...")
    articles = load_article_days(tickers, db_path, title_like=title_like)
    articles = articles[(articles["date"] >= start) & (articles["date"] <= end)]
    print(f"  {len(articles):,} article-days, "
          f"{articles['date'].min().date()} -> {articles['date'].max().date()}")

    print(f"Loading returns from {prices_path.name}...")
    returns = load_returns(prices_path)

    summary = {
        "label": label,
        "n_article_days": int(len(articles)),
        "tickers": len(tickers),
        "title_like": title_like,
        "window": [start, end],
    }
    if articles.empty:
        print("  no rows after filtering — skipping slice")
        return summary

    for lag in (0, 1, 2, 5):
        df = align_and_score(articles, returns, lag=lag)
        if df.empty:
            continue
        col = f"ret_lag{lag}"

        # Pearson on mean_score.
        pearson = df[["mean_score", col]].corr().iloc[0, 1]

        # Mean return by dominant day label.
        by_label = df.groupby("day_label")[col].agg(["count", "mean"]).to_dict("index")

        # Pos-minus-neg spread (the Substack's headline metric).
        spread_bps = 1e4 * (by_label.get("positive", {}).get("mean", np.nan)
                            - by_label.get("negative", {}).get("mean", np.nan))

        print(f"\n--- lag+{lag} ---")
        print(f"  Pearson(mean_score, ret_lag{lag}): {pearson:+.4f}")
        print(f"  By dominant label:")
        for lab in ("positive", "neutral", "negative"):
            rec = by_label.get(lab, {})
            if rec:
                print(f"    {lab:8s}  n={rec['count']:>6d}   "
                      f"mean_ret={rec['mean']*100:+.3f}%")
        if not np.isnan(spread_bps):
            print(f"  pos - neg spread: {spread_bps:+.1f} bps  ({spread_bps/100:+.3f}pp)")

        summary[f"lag{lag}"] = {
            "pearson": float(pearson),
            "by_label": {k: {"count": int(v["count"]), "mean_ret": float(v["mean"])}
                         for k, v in by_label.items()},
            "spread_bps": float(spread_bps) if not np.isnan(spread_bps) else None,
        }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=_NEWS_DB)
    parser.add_argument("--prices", type=Path, default=_PRICES)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default="2026-01-15")
    parser.add_argument("--tickers", nargs="*",
                        help="Override universe; default uses parquet's tickers.")
    parser.add_argument("--title-like", nargs="*", default=None,
                        help="Restrict to articles whose title matches ANY of these "
                             "SQL LIKE patterns (case-insensitive). "
                             "Example: '%%earnings call transcript%%' '%%beats%%'")
    parser.add_argument("--label", default="ALL",
                        help="Slice label for printing/JSON output.")
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        px = pd.read_parquet(args.prices)
        tickers = sorted(px.columns.get_level_values(1).unique())
    print(f"Universe: {len(tickers)} tickers | window {args.start} -> {args.end}")

    summary = diagnostic(
        tickers, args.start, args.end, args.db, args.prices,
        title_like=args.title_like, label=args.label,
    )

    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS if args.label == "ALL" else _RESULTS.with_name(
        f"diagnostic_finbert_signal_{args.label}.json"
    )
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")

    # Interpretation pointer.
    lag1 = summary.get("lag1", {})
    pos_ret = lag1.get("by_label", {}).get("positive", {}).get("mean_ret")
    spread = lag1.get("spread_bps")
    pearson = lag1.get("pearson")
    if pos_ret is not None:
        print()
        print("Reference (Johnsen/Shasharina, Apr 2026):")
        print(f"    Pearson lag+1 corr:         -0.045    | ours: {pearson:+.4f}")
        print(f"    Mean lag+1 ret on pos days: -0.38%    | ours: {pos_ret*100:+.3f}%")
        print(f"    pos-neg spread at lag+1:    +28 bps   | ours: {spread:+.1f} bps")


if __name__ == "__main__":
    main()
