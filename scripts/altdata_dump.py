"""
Materialize an alt-data adapter's output to a long-format parquet.

Phase 5 workflow — the plan calls for validating each alt-data source
via Phase 1 CPCV before blending it into the composite. The existing
``scripts/validate_signal_ic.py`` cross-sectional mode already consumes
long-format ``(date, ticker, score)`` parquets, so the cleanest path is:

1. Run this script once to materialize an adapter's output to disk.
2. Hand the parquet to ``validate_signal_ic.py --signal`` for the
   walk-forward IC + Newey-West + Deflated Sharpe report.
3. Decide whether the adapter earns its keep in
   ``FeaturePipelineConfig.altdata_adapters``.

Caching lives inside each adapter, so rerunning is cheap.

Usage::

    python scripts/altdata_dump.py \\
        --adapter wikipedia \\
        --universe sp100 \\
        --start 2022-01-01 --end 2025-12-31 \\
        --out results/altdata/wikipedia_sp100.parquet

    python scripts/validate_signal_ic.py \\
        --signal results/altdata/wikipedia_sp100.parquet \\
        --prices data/sp100_prices.csv \\
        --horizon 21 \\
        --name wikipedia_pageviews
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from auto_researcher.data.altdata import (
    HAS_GOOGLE_TRENDS,
    HAS_REDDIT,
    SEC8KEventAdapter,
    WikipediaPageviewsAdapter,
)
from auto_researcher.screening import UNIVERSES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
)
logger = logging.getLogger(__name__)


def _make_adapter(name: str, cache_dir: Path):
    """Instantiate an adapter by short name."""
    name = name.lower()
    if name in ("wikipedia", "wikipedia_pageviews", "wiki"):
        return WikipediaPageviewsAdapter(cache_dir=cache_dir)
    if name in ("sec", "sec_8k", "8k", "edgar"):
        return SEC8KEventAdapter(cache_dir=cache_dir)
    if name in ("google_trends", "trends", "gtrends"):
        if not HAS_GOOGLE_TRENDS:
            raise ImportError(
                "google_trends adapter requires pytrends — "
                "`pip install pytrends`"
            )
        from auto_researcher.data.altdata import GoogleTrendsAdapter
        return GoogleTrendsAdapter(cache_dir=cache_dir)
    if name in ("reddit", "wsb"):
        if not HAS_REDDIT:
            raise ImportError(
                "reddit adapter requires praw — `pip install praw`"
            )
        from auto_researcher.data.altdata import RedditMentionsAdapter
        return RedditMentionsAdapter(cache_dir=cache_dir)
    raise ValueError(f"unknown adapter: {name!r}")


def dump_adapter_to_parquet(
    adapter_name: str,
    tickers: list[str],
    start: str,
    end: str,
    out: Path,
    cache_dir: Path,
) -> Path:
    """Fetch adapter scores and persist in the format validate_signal_ic.py
    expects: long-format with ``date``, ``ticker``, ``score`` columns."""
    adapter = _make_adapter(adapter_name, cache_dir=cache_dir)

    logger.info(
        "Fetching %s for %d tickers from %s to %s", adapter.name, len(tickers), start, end
    )
    raw = adapter.fetch(tickers, start, end)

    if raw.empty:
        logger.warning("adapter %s returned empty Series — writing empty parquet",
                       adapter.name)
        long_df = pd.DataFrame(
            {"date": pd.Series(dtype="datetime64[ns]"),
             "ticker": pd.Series(dtype=str),
             "score": pd.Series(dtype=float)}
        )
    else:
        long_df = raw.rename("score").reset_index()
        # Guarantee canonical column order for downstream tools.
        long_df = long_df[["date", "ticker", "score"]]
        long_df["date"] = pd.to_datetime(long_df["date"])

    out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out, index=False)
    logger.info("Wrote %d rows to %s", len(long_df), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", required=True,
        choices=["wikipedia", "sec_8k", "google_trends", "reddit"],
        help="Which adapter to dump.",
    )
    parser.add_argument(
        "--universe", default=None,
        help=f"Ticker universe (keys of recommend.UNIVERSES: "
             f"{sorted(UNIVERSES.keys())}). Mutually exclusive with --tickers.",
    )
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated tickers. Mutually exclusive with --universe.",
    )
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output parquet path (.parquet).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "cache" / "altdata",
        help="Adapter cache directory.",
    )
    args = parser.parse_args()

    if (args.universe is None) == (args.tickers is None):
        parser.error("exactly one of --universe or --tickers must be supplied")

    if args.universe:
        tickers = UNIVERSES[args.universe]()
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    dump_adapter_to_parquet(
        adapter_name=args.adapter,
        tickers=tickers,
        start=args.start,
        end=args.end,
        out=args.out,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
