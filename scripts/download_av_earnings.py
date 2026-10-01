"""
Download Alpha Vantage earnings history (actual vs. consensus EPS) and validate it.

The free key allows 25 requests a day, so the S&P 500 takes about three weeks;
the script is meant to run daily (see ``scripts/schedule_av_download.ps1``).
Each run downloads uncached symbols until the quota is used (and, once all
are cached, refreshes files older than ``--refresh-after-days``), then rewrites
``docs/results/av_earnings_validation.md``:

* coverage by year and the share of quarters whose estimate equals the actual;
* report dates vs. SEC-inferred and Yahoo report dates;
* consensus vs. Yahoo (2023+) and vs. FMP for the stocks both cover. FMP flags
  backfilled rows using revenue, which Alpha Vantage lacks, so the FMP overlap
  shows whether Alpha Vantage's early exact matches are copied actuals.

Stocks FMP already covers are fetched first so the cross-check works early.

Example::

    python scripts/download_av_earnings.py            # resumes where it stopped
    python scripts/download_av_earnings.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_fmp_earnings import (  # noqa: E402
    consensus_agreement,
    date_agreement,
    universe,
    yahoo_date_agreement,
)

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.data.alphavantage_earnings import (  # noqa: E402
    BACKFILL_CUTOFF,
    download_av_earnings,
)
from auto_researcher.data.fmp_earnings import load_fmp_earnings  # noqa: E402

logger = logging.getLogger("download_av_earnings")
CACHE = ROOT / "data" / "research_cache"


def fmp_crosscheck(av: pd.DataFrame, fmp: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha Vantage vs. FMP for the same reports, split at the backfill cutoff.

    The key column is how often an Alpha Vantage exact match (estimate == actual)
    is a quarter FMP shows as a real surprise or flags as backfilled.
    """
    cols = ["symbol", "date", "eps_actual", "eps_estimated", "suspect_backfill"]
    a = av.loc[av["eps_actual"].notna() & av["eps_estimated"].notna(), cols[:4]].copy()
    f = fmp.loc[fmp["eps_actual"].notna() & fmp["eps_estimated"].notna(), cols].copy()
    for d in (a, f):
        d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
    m = pd.merge_asof(a.sort_values("date"), f.sort_values("date").rename(columns={"date": "fmp_date"}),
                      left_on="date", right_on="fmp_date", by="symbol", direction="nearest",
                      tolerance=pd.Timedelta(days=5), suffixes=("_av", "_fmp")).dropna(subset=["fmp_date"])
    if m.empty:
        return pd.DataFrame()
    m["era"] = (m["date"] < BACKFILL_CUTOFF).map({True: f"before {BACKFILL_CUTOFF.year}",
                                                  False: f"{BACKFILL_CUTOFF.year} on"})
    m["same_date"] = m["date"] == m["fmp_date"]
    m["est_1c"] = (m["eps_estimated_av"] - m["eps_estimated_fmp"]).abs() <= 0.01 + 1e-9
    m["av_exact"] = m["eps_actual_av"] == m["eps_estimated_av"]
    exact = m[m["av_exact"]]
    m["exact_fmp_flagged"] = exact["suspect_backfill"].astype(float)
    m["exact_fmp_real_surprise"] = (
        (exact["eps_actual_fmp"] - exact["eps_estimated_fmp"]).abs() > 0.01 + 1e-9).astype(float)
    return m.groupby("era").agg(
        quarters=("symbol", "size"), symbols=("symbol", "nunique"), same_date=("same_date", "mean"),
        estimate_within_1c=("est_1c", "mean"), av_exact=("av_exact", "mean"),
        exact_fmp_flagged=("exact_fmp_flagged", "mean"),
        exact_fmp_real_surprise=("exact_fmp_real_surprise", "mean"))


def write_report(av: pd.DataFrame, total: int, unavailable: int, out: Path) -> None:
    reported = av[av["eps_actual"].notna()].copy()
    reported["year"] = pd.to_datetime(reported["date"]).dt.year
    reported["has_est"] = reported["eps_estimated"].notna()
    reported["exact"] = reported["eps_actual"] == reported["eps_estimated"]
    by_year = reported.groupby("year").agg(rows=("symbol", "size"), symbols=("symbol", "nunique"),
                                           with_estimate=("has_est", "mean"), exact=("exact", "mean"),
                                           flagged=("suspect_backfill", "mean"))
    dates = date_agreement(av, CACHE)
    ydates = yahoo_date_agreement(av)
    cons = consensus_agreement(av)
    fmp_dir = CACHE / "fmp_earnings"
    cross = fmp_crosscheck(av, load_fmp_earnings(fmp_dir)) if fmp_dir.exists() else pd.DataFrame()

    lines = [
        "# Alpha Vantage earnings data validation", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/download_av_earnings.py`. "
        f"Symbols downloaded: {av['symbol'].nunique()} of {total} (unknown to Alpha Vantage: "
        f"{unavailable}). Earliest report: {pd.to_datetime(av['date']).min():%Y-%m-%d}.", "",
        "## Report dates", "",
        f"Against SEC-inferred announcement dates ({dates['n']:,.0f} quarters): same day "
        f"{dates['exact']:.0%}, within one day {dates['within_1d']:.0%}, Alpha Vantage two or more "
        f"days earlier {dates['fmp_earlier_2d+']:.1%}.", "",
        f"Against Yahoo report dates (`data/sentiment_500.csv`, {ydates['n']:,.0f} quarters): "
        f"same day {ydates.get('exact', float('nan')):.0%}, within one day "
        f"{ydates.get('within_1d', float('nan')):.0%}.", "",
        "## Consensus vs. Yahoo (2023+)", "",
    ]
    if cons.get("n"):
        lines += [f"{cons['n']:,.0f} quarters. Estimates within 1 cent: {cons['estimate_within_1c']:.0%}; "
                  f"within 5%: {cons['estimate_within_5pct']:.0%}. Rank correlation of surprises: "
                  f"{cons['surprise_rank_corr']:+.2f}.", ""]
    else:
        lines += ["No overlapping quarters yet.", ""]
    lines += ["## Cross-check against FMP", ""]
    if cross.empty:
        lines += ["No overlapping stocks yet.", ""]
    else:
        lines += [
            "Same report matched within 5 days. *AV exact* is the share of quarters where Alpha "
            "Vantage's estimate equals the actual; the last two columns describe those quarters: "
            "the share FMP flags as backfilled (revenue estimate equals actual too), and the share "
            "where FMP shows a real surprise (so Alpha Vantage's estimate was overwritten).", "",
            "| Period | Quarters | Stocks | Same date | Estimate within 1c | AV exact | "
            "of which FMP-flagged | of which FMP real surprise |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines += [f"| {era} | {int(r.quarters):,} | {int(r.symbols)} | {r.same_date:.0%} | "
                  f"{r.estimate_within_1c:.0%} | {r.av_exact:.0%} | {r.exact_fmp_flagged:.0%} | "
                  f"{r.exact_fmp_real_surprise:.0%} |" for era, r in cross.iterrows()]
        lines.append("")
    lines += [
        "## Coverage by year", "",
        f"Flagged = estimate equals actual before {BACKFILL_CUTOFF.year}; excluded from surprise "
        "events as probable backfill.", "",
        "| Year | Reported quarters | Symbols | With estimate | Estimate = actual | Flagged |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [f"| {y} | {int(r.rows):,} | {int(r.symbols)} | {r.with_estimate:.0%} | {r.exact:.0%} | "
              f"{r.flagged:.0%} |" for y, r in by_year.iterrows()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(CACHE / "av_earnings"))
    ap.add_argument("--max-requests", type=int, default=25)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--retry-unavailable", action="store_true")
    ap.add_argument("--refresh-after-days", type=float, default=30,
                    help="Once every symbol is cached, re-download files older than this")
    ap.add_argument("--log-file", help="Also append the log to this file (for scheduled runs)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "av_earnings_validation.md"))
    args = ap.parse_args(argv)
    # pythonw (used by the scheduled task) has no console streams.
    handlers: list[logging.Handler] = [logging.StreamHandler()] if sys.stderr else []
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)

    cache_dir = Path(args.cache_dir)
    symbols = universe()
    fmp_dir = CACHE / "fmp_earnings"
    if fmp_dir.exists():
        fmp_syms = sorted(p.stem for p in fmp_dir.glob("*.parquet"))
        symbols = list(dict.fromkeys([s for s in fmp_syms if s in symbols] + symbols))

    if not args.validate_only:
        from dotenv import dotenv_values

        key = dotenv_values(ROOT / ".env").get("ALPHAVANTAGE_API_KEY")
        if not key:
            raise SystemExit("ALPHAVANTAGE_API_KEY is not set in .env")
        s = download_av_earnings(symbols, key.strip(), cache_dir, args.max_requests,
                                 retry_unavailable=args.retry_unavailable,
                                 refresh_after_days=args.refresh_after_days)
        logger.info("Downloaded %d, already cached %d, unavailable %d, remaining %d%s",
                    len(s.downloaded), s.already_cached, len(s.unavailable), len(s.remaining),
                    " (daily limit reached)" if s.quota_hit else "")

    av = load_fmp_earnings(cache_dir) if cache_dir.exists() else pd.DataFrame()
    if av.empty:
        logger.info("Nothing downloaded yet; no report written")
        return
    marker = cache_dir / "_unavailable.json"
    unavailable = len(json.loads(marker.read_text())) if marker.exists() else 0
    write_report(av, len(symbols), unavailable, Path(args.out))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Scheduled runs have no console; make failures visible in the log file.
        logger.exception("download_av_earnings failed")
        raise
