"""
Build the point-in-time fundamentals table from SEC XBRL filings and validate it.

Downloads company facts for the S&P 500 (about 500 requests; SEC allows 10 a
second, so a few minutes), builds ``data/research_cache/sec_fundamentals_pit.parquet``
and writes ``docs/results/sec_fundamentals_validation.md``:

* coverage by year and item;
* filing lags (period end to availability) and a check that nothing is
  available before its period ends;
* how often first-reported figures were later restated;
* agreement with Yahoo's statements (DefeatBeta) on fiscal-year figures;
* a share-count check against Yahoo: companies whose SEC share count does not
  match their traded share class (e.g. Berkshire reports Class A equivalents)
  are listed in ``sec_share_mismatch.json`` and left out of market-cap ratios.

Example::

    python scripts/build_fundamentals.py              # downloads what is missing
    python scripts/build_fundamentals.py --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_fmp_earnings import universe  # noqa: E402
from earnings_feature_research import defeatbeta_table  # noqa: E402

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.data.sec_fundamentals import (  # noqa: E402
    build_pit_table,
    download_company_facts,
    fundamentals_asof,
    load_company_facts,
)
from auto_researcher.features.valuation import adjusted_shares  # noqa: E402

logger = logging.getLogger("build_fundamentals")
CACHE = ROOT / "data" / "research_cache"
PIT_PATH = CACHE / "sec_fundamentals_pit.parquet"
MISMATCH_PATH = CACHE / "sec_share_mismatch.json"
PRICE_START = "2009-01-01"
SHARE_BAND = (0.8, 1.25)


def load_close_and_splits(tickers: list[str], cache_dir: Path = CACHE, start: str = PRICE_START,
                          max_age_days: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split-adjusted (not dividend-adjusted) close and split ratios from yfinance,
    cached for ``max_age_days``.
    """
    import yfinance as yf

    key = hashlib.sha1(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    path = cache_dir / f"close_splits_{key}_{start}.parquet"
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_days * 86400:
        frames = []
        for i in range(0, len(tickers), 100):
            batch = tickers[i: i + 100]
            for attempt in range(3):
                try:
                    data = yf.download(batch, start=start, auto_adjust=False, actions=True,
                                       progress=False, threads=True)
                    frames.append(data[["Close", "Stock Splits"]])
                    break
                except Exception as exc:  # rate limits, transient errors
                    logger.warning("price batch %d attempt %d failed: %s", i // 100, attempt + 1, exc)
                    time.sleep(5 * (attempt + 1))
        raw = pd.concat(frames, axis=1)
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(path)
    raw = pd.read_parquet(path)
    close = raw["Close"].dropna(axis=1, how="all").sort_index()
    splits = raw["Stock Splits"].reindex(columns=close.columns).fillna(0.0).sort_index()
    return close, splits


def share_check(table: pd.DataFrame, close: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Our split-adjusted share count on the last price date vs. Yahoo's latest count."""
    date = close.index[-1]
    fund = fundamentals_asof(table, [date], items=["shares_cover", "shares_diluted"],
                             with_available=("shares_cover", "shares_diluted"))
    ours = adjusted_shares(fund, splits).droplevel("date")
    yahoo = defeatbeta_table("stock_shares_outstanding", CACHE)
    yahoo = yahoo.sort_values("report_date").groupby("symbol")["shares_outstanding"].last()
    df = pd.DataFrame({"sec": ours, "yahoo": yahoo.reindex(ours.index)})
    df["ratio"] = df["sec"] / df["yahoo"]
    df["ok"] = df["ratio"].between(*SHARE_BAND)
    return df


def yahoo_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """Fiscal-year figures vs. Yahoo's annual statements (both as currently reported)."""
    pairs = {"revenue": "total_revenue", "net_income": "net_income_common_stockholders",
             "operating_cash_flow": "operating_cash_flow", "total_assets": "total_assets"}
    y = defeatbeta_table("stock_statement", CACHE,
                         filters=[("period_type", "==", "annual"),
                                  ("item_name", "in", list(pairs.values()))])
    y = y[y["report_date"] != "TTM"].dropna(subset=["item_value"])
    y = y.assign(period_end=pd.to_datetime(y["report_date"]).astype("datetime64[ns]"),
                 yahoo=pd.to_numeric(y["item_value"], errors="coerce"),
                 item=y["item_name"].map({v: k for k, v in pairs.items()}))
    rows = []
    for item in pairs:
        ours = table[table["item"] == item][["symbol", "period_end", "value"]].sort_values("period_end")
        theirs = y[y["item"] == item][["symbol", "period_end", "yahoo"]].sort_values("period_end")
        m = pd.merge_asof(theirs, ours, on="period_end", by="symbol", direction="nearest",
                          tolerance=pd.Timedelta(days=7)).dropna(subset=["value", "yahoo"])
        m = m[m["yahoo"].abs() > 0]
        gap = (m["value"] / m["yahoo"] - 1).abs()
        rows.append({"item": item, "n": len(m), "within_1pct": (gap <= 0.01).mean(),
                     "within_5pct": (gap <= 0.05).mean()})
    return pd.DataFrame(rows).set_index("item")


def restatement_rates(first: pd.DataFrame, last: pd.DataFrame) -> pd.DataFrame:
    key = ["symbol", "item", "period_end"]
    m = first.merge(last, on=key, suffixes=("_first", "_last"))
    m = m[m["value_first"].abs() > 0]
    m["change"] = (m["value_last"] / m["value_first"] - 1).abs()
    items = ["revenue", "net_income", "operating_cash_flow", "total_assets", "shares_diluted"]
    g = m[m["item"].isin(items)].groupby("item")
    return pd.DataFrame({"periods": g.size(), "changed_over_1pct": g["change"].apply(lambda c: (c > 0.01).mean()),
                         "changed_over_5pct": g["change"].apply(lambda c: (c > 0.05).mean()),
                         "first_to_last_delay_days": g.apply(
                             lambda d: (d["available_date_last"] - d["available_date_first"]).dt.days.median(),
                             include_groups=False)})


def write_report(facts: pd.DataFrame, table: pd.DataFrame, restated: pd.DataFrame,
                 shares: pd.DataFrame, n_universe: int, out: Path) -> None:
    lag = (table["available_date"] - table["period_end"]).dt.days
    # From 2011 on: earlier periods mostly first appear as comparatives during the
    # XBRL phase-in (2009-2011), so their (correct) availability dates are late.
    flows = table["item"].isin(["revenue", "operating_cash_flow"]) & (table["period_end"].dt.year >= 2011)
    fy_like = table["period_end"].dt.month == 12
    by_year = (table[table["item"].isin(["revenue", "operating_cash_flow", "total_assets", "shares_diluted"])]
               .assign(year=lambda d: d["period_end"].dt.year)
               .groupby(["year", "item"])["symbol"].nunique().unstack())
    latest = fundamentals_asof(table, [pd.Timestamp.today().normalize()])
    cover = latest.notna().mean().sort_values(ascending=False)
    rest = restatement_rates(table, restated)
    agree = yahoo_agreement(restated)
    bad = shares[~shares["ok"]].sort_values("ratio")

    lines = [
        "# SEC point-in-time fundamentals: validation", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/build_fundamentals.py`. Companies with "
        f"facts: {facts['symbol'].nunique()} of {n_universe}; {len(table):,} point-in-time values.", "",
        "## Timing", "",
        "Each value becomes usable the day after the filing that first reported it. Days from "
        "period end to availability, for periods ending 2011 or later. The long tail is "
        "periods first disclosed later (pre-IPO years in a prospectus-era 10-K, spin-offs, "
        "late filers); earlier periods were mostly first tagged as comparatives during the "
        "2009–2011 XBRL phase-in and become usable only then.", "",
        "| Values | 10th pct | Median | 90th pct |", "| --- | ---: | ---: | ---: |",
        f"| TTM revenue / cash flow, December period ends (mostly 10-K) | "
        f"{lag[flows & fy_like].quantile(.1):.0f} | {lag[flows & fy_like].median():.0f} | "
        f"{lag[flows & fy_like].quantile(.9):.0f} |",
        f"| TTM revenue / cash flow, other period ends | {lag[flows & ~fy_like].quantile(.1):.0f} | "
        f"{lag[flows & ~fy_like].median():.0f} | {lag[flows & ~fy_like].quantile(.9):.0f} |", "",
        f"Values available before their period ended: {int((lag < 0).sum())}.", "",
        "## Restatements", "",
        "Share of periods whose figure in the latest filing differs from the first-reported "
        "one. The table keeps first-reported values; using the latest would leak later "
        "corrections into backtests.", "",
        "| Item | Periods | Changed > 1% | Changed > 5% | Median delay to last version (days) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [f"| {i} | {int(r.periods):,} | {r.changed_over_1pct:.1%} | {r.changed_over_5pct:.1%} | "
              f"{r.first_to_last_delay_days:.0f} |" for i, r in rest.iterrows()]
    lines += [
        "", "## Agreement with Yahoo (fiscal-year figures)", "",
        "Latest-reported SEC figures vs. Yahoo's annual statements (DefeatBeta), matched "
        "within 7 days of the fiscal year end. Differences are mostly definitions (e.g. "
        "Yahoo's net income to common holders).", "",
        "| Item | Matched years | Within 1% | Within 5% |", "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {i} | {int(r.n):,} | {r.within_1pct:.0%} | {r.within_5pct:.0%} |" for i, r in agree.iterrows()]
    lines += [
        "", "## Share counts", "",
        f"SEC share count (cover page, else diluted average; split-adjusted) vs. Yahoo's latest, "
        f"{shares['yahoo'].notna().sum()} companies compared. Outside "
        f"{SHARE_BAND[0]:.2f}–{SHARE_BAND[1]:.2f}×: {len(bad)}. These are excluded from "
        "market-cap ratios (`data/research_cache/sec_share_mismatch.json`); most are "
        "multi-class companies.", "",
    ]
    if len(bad):
        lines += ["| Symbol | SEC / Yahoo |", "| --- | ---: |"]
        lines += [f"| {s} | {r.ratio:.3g} |" for s, r in bad.iterrows()]
        lines.append("")
    lines += ["## Coverage", "", "Share of companies with a current value (as of today):", "",
              "| Item | Coverage |", "| --- | ---: |"]
    lines += [f"| {i} | {v:.0%} |" for i, v in cover.items() if not i.endswith("_available")]
    lines += ["", "Companies with a value, by period-end year:", "",
              "| Year | " + " | ".join(by_year.columns) + " |",
              "| --- |" + " ---: |" * len(by_year.columns)]
    lines += [f"| {y} | " + " | ".join(f"{v:.0f}" for v in r.fillna(0)) + " |"
              for y, r in by_year.iterrows() if y >= 2008]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(CACHE / "sec_facts"))
    ap.add_argument("--refresh-after-days", type=float, default=30)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "sec_fundamentals_validation.md"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    symbols = universe()
    cache_dir = Path(args.cache_dir)
    if not args.validate_only:
        from dotenv import dotenv_values

        ua = (dotenv_values(ROOT / ".env").get("SEC_API_USER_AGENT") or "").strip().strip('"')
        if not ua or "example.com" in ua:
            raise SystemExit("Set SEC_API_USER_AGENT in .env to 'Your Name your@email' (SEC requires it)")
        s = download_company_facts(symbols, cache_dir, ua, refresh_after_days=args.refresh_after_days)
        logger.info("Downloaded %d, cached %d, unavailable %d%s", len(s.downloaded), s.already_cached,
                    len(s.unavailable), " (stopped: rate limited)" if s.quota_hit else "")

    facts = load_company_facts(cache_dir)
    if facts.empty:
        raise SystemExit("No company facts cached yet")
    table = build_pit_table(facts)
    table.to_parquet(PIT_PATH)
    logger.info("Wrote %s (%d rows, %d companies)", PIT_PATH, len(table), table["symbol"].nunique())
    restated = build_pit_table(facts, mode="last")

    close, splits = load_close_and_splits(sorted(table["symbol"].unique()))
    shares = share_check(table, close, splits)
    MISMATCH_PATH.write_text(json.dumps(sorted(shares.index[~shares["ok"] & shares["yahoo"].notna()])))
    write_report(facts, table, restated, shares[shares["yahoo"].notna()], len(symbols), Path(args.out))
    logger.info("Wrote %s", args.out)
    if np.any(table["available_date"] < table["period_end"]):
        raise SystemExit("Point-in-time violation: a value is available before its period ends")


if __name__ == "__main__":
    main()
