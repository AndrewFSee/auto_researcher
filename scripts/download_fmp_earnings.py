"""
Download FMP earnings history (actual vs. consensus EPS) and validate it.

The free FMP plan allows about 250 requests a day, so the S&P 500 takes two
days. Each run downloads up to ``--max-requests`` uncached symbols, then
rewrites ``docs/results/fmp_earnings_validation.md``:

* coverage by year and the share of rows that look backfilled (estimate equal
  to the actual, down to the dollar of revenue);
* FMP report dates vs. the announcement dates inferred from SEC filings;
* FMP consensus vs. the Yahoo consensus in ``data/pead_backtest_results.parquet``
  for overlapping quarters (2023+). Close agreement suggests FMP's estimates
  are the pre-announcement consensus rather than later revisions.

Example::

    python scripts/download_fmp_earnings.py            # resumes where it stopped
    python scripts/download_fmp_earnings.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from earnings_feature_research import defeatbeta_table  # noqa: E402
from feature_research import sp500_constituents  # noqa: E402

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.data.fmp_earnings import (  # noqa: E402
    download_fmp_earnings,
    load_fmp_earnings,
)
from auto_researcher.features.earnings_events import announcement_dates  # noqa: E402

logger = logging.getLogger("download_fmp_earnings")


def universe() -> list[str]:
    names = list(sp500_constituents()["ticker"])
    cached = ROOT / "data" / "price_cache" / "prices_2015-01-01_2020-01-01.parquet"
    if cached.exists():
        names += [t for t in pd.read_parquet(cached).columns.get_level_values(1) if t != "SPY"]
    return sorted(dict.fromkeys(names))


def date_agreement(fmp: pd.DataFrame, research_cache: Path) -> dict[str, float]:
    """How FMP report dates compare with announcement dates inferred from SEC filings."""
    filings = defeatbeta_table("stock_sec_filing", research_cache,
                               columns=["symbol", "form_type", "filing_date", "report_date"],
                               filters=[("form_type", "in", ["8-K", "10-Q", "10-K"])])
    sec = announcement_dates(filings)[["symbol", "announce_date"]]
    sec["announce_date"] = sec["announce_date"].astype("datetime64[ns]")
    left = fmp[fmp["eps_actual"].notna()][["symbol", "date"]].copy()
    left["date"] = left["date"].astype("datetime64[ns]")
    m = pd.merge_asof(left.sort_values("date"), sec.sort_values("announce_date"),
                      left_on="date", right_on="announce_date", by="symbol",
                      direction="nearest", tolerance=pd.Timedelta(days=10))
    diff = (m["date"] - m["announce_date"]).dt.days.dropna()
    return {"matched": float(len(diff)) / max(len(m), 1), "exact": float((diff == 0).mean()),
            "within_1d": float((diff.abs() <= 1).mean()), "fmp_earlier_2d+": float((diff <= -2).mean()),
            "n": float(len(diff))}


def yahoo_date_agreement(fmp: pd.DataFrame) -> dict[str, float]:
    """FMP report dates vs. the independent report dates in data/sentiment_500.csv."""
    path = ROOT / "data" / "sentiment_500.csv"
    if not path.exists():
        return {"n": 0.0}
    rep = pd.read_csv(path, parse_dates=["report_date"]).rename(columns={"symbol": "sym"})
    rep["report_date"] = rep["report_date"].astype("datetime64[ns]")
    left = fmp[fmp["eps_actual"].notna()][["symbol", "date"]].copy()
    left["date"] = left["date"].astype("datetime64[ns]")
    m = pd.merge_asof(left.sort_values("date"), rep.sort_values("report_date"),
                      left_on="date", right_on="report_date", left_by="symbol", right_by="sym",
                      direction="nearest", tolerance=pd.Timedelta(days=10))
    diff = (m["date"] - m["report_date"]).dt.days.dropna()
    return {"n": float(len(diff)), "exact": float((diff == 0).mean()),
            "within_1d": float((diff.abs() <= 1).mean()), "early_2d+": float((diff <= -2).mean())}


def consensus_agreement(fmp: pd.DataFrame) -> dict[str, float]:
    """FMP vs. Yahoo consensus EPS for the same quarters."""
    path = ROOT / "data" / "pead_backtest_results.parquet"
    if not path.exists():
        return {}
    y = pd.read_parquet(path)[["ticker", "quarter_date", "eps_estimate", "eps_actual"]].dropna()
    y = y.rename(columns={"ticker": "symbol"})
    y["quarter_date"] = pd.to_datetime(y["quarter_date"]).astype("datetime64[ns]")
    f = fmp[fmp["eps_actual"].notna() & fmp["eps_estimated"].notna()].copy()
    f["date"] = f["date"].astype("datetime64[ns]")
    # A report covers the latest fiscal quarter that ended at least 10 days earlier.
    f["latest_end"] = f["date"] - pd.Timedelta(days=10)
    m = pd.merge_asof(f.sort_values("latest_end"), y.sort_values("quarter_date"),
                      left_on="latest_end", right_on="quarter_date", by="symbol",
                      direction="backward", tolerance=pd.Timedelta(days=110))
    m = m.dropna(subset=["eps_estimate"])
    m = m[(m["date"] - m["quarter_date"]).dt.days.between(10, 120)]
    if m.empty:
        return {"n": 0.0}
    est_gap = (m["eps_estimated"] - m["eps_estimate"]).abs()
    s_fmp = (m["eps_actual_x"] - m["eps_estimated"]) / m["eps_estimated"].abs().clip(lower=0.01)
    s_y = (m["eps_actual_y"] - m["eps_estimate"]) / m["eps_estimate"].abs().clip(lower=0.01)
    return {"n": float(len(m)),
            "estimate_within_1c": float((est_gap <= 0.01 + 1e-9).mean()),
            "estimate_within_5pct": float((est_gap <= 0.05 * m["eps_estimate"].abs()).mean()),
            "actual_within_1c": float(((m["eps_actual_x"] - m["eps_actual_y"]).abs() <= 0.01 + 1e-9).mean()),
            "surprise_rank_corr": float(s_fmp.rank().corr(s_y.rank()))}


def write_report(fmp: pd.DataFrame, total: int, research_cache: Path, out: Path,
                 unavailable: int = 0) -> None:
    reported = fmp[fmp["eps_actual"].notna() & fmp["eps_estimated"].notna()].copy()
    reported["year"] = pd.to_datetime(reported["date"]).dt.year
    by_year = reported.groupby("year").agg(rows=("symbol", "size"), symbols=("symbol", "nunique"),
                                           backfilled=("suspect_backfill", "mean"))
    dates = date_agreement(fmp, research_cache)
    ydates = yahoo_date_agreement(fmp)
    cons = consensus_agreement(fmp)
    lines = [
        "# FMP earnings data validation", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/download_fmp_earnings.py`. "
        f"Symbols downloaded: {fmp['symbol'].nunique()} of {total}; refused by the current "
        f"FMP plan (HTTP 402): {unavailable}. The free plan serves only a fixed list of "
        "popular stocks, too few for the event strategy, which needs at least 100.", "",
        "## Report dates vs. SEC-inferred announcement dates", "",
        f"{dates['n']:,.0f} reported quarters matched within 10 days ({dates['matched']:.0%}). "
        f"Same day: {dates['exact']:.0%}; within one day: {dates['within_1d']:.0%}; FMP two or "
        f"more days earlier: {dates['fmp_earlier_2d+']:.1%}. (The SEC rule deliberately errs "
        "late, so FMP being earlier is expected.)", "",
        f"Against independent report dates (`data/sentiment_500.csv`, {ydates['n']:,.0f} "
        f"quarters): same day {ydates.get('exact', float('nan')):.0%}, within one day "
        f"{ydates.get('within_1d', float('nan')):.0%}, FMP two or more days earlier "
        f"{ydates.get('early_2d+', float('nan')):.1%}.", "",
        "## Consensus vs. Yahoo (overlapping quarters, 2023+)", "",
    ]
    if cons.get("n"):
        lines += [
            f"{cons['n']:,.0f} quarters. Estimates within 1 cent: {cons['estimate_within_1c']:.0%}; "
            f"within 5%: {cons['estimate_within_5pct']:.0%}. Actuals within 1 cent: "
            f"{cons['actual_within_1c']:.0%} (the rest are mostly GAAP vs. adjusted EPS on "
            f"quarters with one-off items). Rank correlation of surprises: "
            f"{cons['surprise_rank_corr']:+.2f}.", "",
        ]
    else:
        lines += ["No overlapping quarters yet.", ""]
    lines += ["## Coverage by year", "",
              "Backfilled = estimate equals actual down to the dollar of revenue (excluded from "
              "surprise events).", "",
              "| Year | Reported quarters | Symbols | Backfilled |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {y} | {int(r.rows):,} | {int(r.symbols)} | {r.backfilled:.0%} |"
              for y, r in by_year.iterrows()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "research_cache" / "fmp_earnings"))
    ap.add_argument("--research-cache", default=str(ROOT / "data" / "research_cache"),
                    help="Where the SEC filing index is cached (for date validation)")
    ap.add_argument("--max-requests", type=int, default=240)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--retry-unavailable", action="store_true",
                    help="Retry symbols the plan refused before (e.g. after upgrading)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "fmp_earnings_validation.md"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    symbols = universe()
    if not args.validate_only:
        from dotenv import dotenv_values

        key = dotenv_values(ROOT / ".env").get("FMP_API_KEY")
        if not key:
            raise SystemExit("FMP_API_KEY is not set in .env")
        s = download_fmp_earnings(symbols, key.strip(), Path(args.cache_dir), args.max_requests,
                                  retry_unavailable=args.retry_unavailable)
        logger.info("Downloaded %d, already cached %d, unavailable %d, remaining %d%s",
                    len(s.downloaded), s.already_cached, len(s.unavailable), len(s.remaining),
                    " (daily limit reached)" if s.quota_hit else "")

    fmp = load_fmp_earnings(Path(args.cache_dir))
    if fmp.empty:
        raise SystemExit("Nothing downloaded yet")
    marker = Path(args.cache_dir) / "_unavailable.json"
    refused = len(json.loads(marker.read_text())) if marker.exists() else 0
    write_report(fmp, len(symbols), Path(args.research_cache), Path(args.out), refused)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
