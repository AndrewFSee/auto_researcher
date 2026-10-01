"""
Reverse DCF and valuation snapshot for S&P 500 companies.

For each company, solve for the 10-year growth in free cash flow to the firm
that today's enterprise value implies (then 2.5% a year forever), at 8%, 9%
and 10% discount rates, and set it beside what the company actually delivered
over the past three years. Inputs are the latest SEC filings known today
(``scripts/build_fundamentals.py``) and today's price.

Read the output as "what the market is assuming", not as a buy/sell signal:
it has not been backtested, and the model is deliberately simple (one growth
rate, then a terminal value that is usually most of the answer).

Excluded: financial companies and industrials with captive finance arms (free
cash flow and debt mix in a lending business), companies whose SEC share count
does not match their traded share class, companies with no tagged debt but
sizeable long-term liabilities (enterprise value would be wrong), and companies
with negative trailing free cash flow (no growth rate can justify a positive
value from a negative base).

Example::

    python scripts/reverse_dcf.py                       # S&P 500 -> docs/results/reverse_dcf.*
    python scripts/reverse_dcf.py --tickers AAPL NVDA KO
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

from build_fundamentals import MISMATCH_PATH, PIT_PATH, load_close_and_splits  # noqa: E402
from feature_research import sp500_constituents  # noqa: E402

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.data.sec_fundamentals import fundamentals_asof  # noqa: E402
from auto_researcher.features.valuation import implied_growth, valuation_metrics  # noqa: E402

logger = logging.getLogger("reverse_dcf")
DISCOUNTS = (0.08, 0.09, 0.10)
PRIMARY = 0.09
# Lending arms make consolidated cash flow and debt mix two businesses.
CAPTIVE_FINANCE = {"F", "GM", "CAT", "DE", "PCAR"}
SHARES = ("shares_cover", "shares_diluted")


def cagr(now: pd.Series, before: pd.Series, years: float) -> pd.Series:
    ok = (now > 0) & (before > 0)
    return ((now / before) ** (1 / years) - 1).where(ok)


def snapshot(table: pd.DataFrame, close: pd.DataFrame, splits: pd.DataFrame, date: pd.Timestamp,
             years: int, terminal_growth: float) -> pd.DataFrame:
    symbols = sorted(set(table["symbol"]) & set(close.columns))

    def asof(when: pd.Timestamp) -> pd.DataFrame:
        f = fundamentals_asof(table, [when], symbols, with_available=SHARES)
        return f.droplevel("date").set_index(pd.MultiIndex.from_product([[date], f.index.get_level_values(
            "symbol")], names=["date", "symbol"]))

    fund = asof(date)
    m = valuation_metrics(fund, close, splits, fund_prior=asof(date - pd.DateOffset(years=1)))
    past = asof(date - pd.DateOffset(years=3))
    past_fcf = past["operating_cash_flow"] - past["capex"].fillna(0)
    m["fcf_cagr_3y"] = cagr(m["fcf"], past_fcf, 3)
    m["revenue_cagr_3y"] = cagr(fund["revenue"], past["revenue"], 3)
    m["revenue"] = fund["revenue"]
    for r in DISCOUNTS:
        m[f"implied_growth_{r:.0%}"] = [
            implied_growth(ev, f, r, years, terminal_growth)
            for ev, f in zip(m["enterprise_value"], m["fcff"])]
    return m.droplevel("date")


def fmt_pct(v: float) -> str:
    return "n/a" if pd.isna(v) else f"{v:+.1%}"


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--terminal-growth", type=float, default=0.025)
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "reverse_dcf"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if not PIT_PATH.exists():
        raise SystemExit("Run scripts/build_fundamentals.py first")
    table = pd.read_parquet(PIT_PATH)
    members = sp500_constituents().set_index("ticker")
    mismatch = set(json.loads(MISMATCH_PATH.read_text())) if MISMATCH_PATH.exists() else set()
    close, splits = load_close_and_splits(sorted(table["symbol"].unique()))
    date = close.index[-1]

    snap = snapshot(table, close, splits, date, args.years, args.terminal_growth)
    snap["sector"] = members["sector"].reindex(snap.index)
    debt_unknown = ~snap["debt_known"] & (snap["noncurrent_liabilities"] > 0.25 * snap["market_cap"])
    snap["excluded"] = np.select(
        [snap["sector"].eq("Financials"), snap.index.isin(list(CAPTIVE_FINANCE)),
         snap.index.isin(list(mismatch)), snap["market_cap"].isna(), debt_unknown, ~(snap["fcff"] > 0)],
        ["financial", "captive finance", "share count mismatch", "no market cap", "debt not tagged",
         "negative FCF"], default="")
    ig = f"implied_growth_{PRIMARY:.0%}"
    usable = snap[snap["excluded"] == ""]

    cols = ["sector", "market_cap", "enterprise_value", "debt", "fcff", "fcff_ev", "fcf_yield",
            *[f"implied_growth_{r:.0%}" for r in DISCOUNTS], "fcf_cagr_3y", "revenue_cagr_3y",
            "ebit_ev", "gross_profitability", "accruals", "net_issuance", "shareholder_yield", "excluded"]
    if args.tickers:
        view = snap.reindex([t.upper() for t in args.tickers])[cols]
        with pd.option_context("display.width", 200, "display.max_columns", 30,
                               "display.float_format", "{:,.3f}".format):
            print(f"As of {date:%Y-%m-%d}; growth implied over {args.years} years, then "
                  f"{args.terminal_growth:.1%} forever")
            print(view.T.to_string())
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    snap[cols].sort_values(ig).to_csv(out.with_suffix(".csv"), float_format="%.6g")

    def row(sym: str, r: pd.Series) -> str:
        return (f"| {sym} | {r['sector']} | {r['market_cap'] / 1e9:,.0f} | {r['fcff_ev']:.1%} | "
                f"{fmt_pct(r[ig])} | {fmt_pct(r['fcf_cagr_3y'])} | {fmt_pct(r['revenue_cagr_3y'])} |")

    header = ["| Company | Sector | Market cap ($bn) | FCFF / EV | Implied growth (9%) | "
              "FCF growth, past 3y | Revenue growth, past 3y |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    ranked = usable.dropna(subset=[ig]).sort_values(ig)
    by_sector = usable.groupby("sector").agg(companies=(ig, "size"), median_implied=(ig, "median"),
                                             median_fcf_yield=("fcff_ev", "median"),
                                             median_past_fcf_growth=("fcf_cagr_3y", "median"))
    counts = snap["excluded"].replace("", "included").value_counts()
    md = [
        "# Reverse DCF: what growth is priced in?", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/reverse_dcf.py`, prices as of "
        f"{date:%Y-%m-%d}. Full table: [reverse_dcf.csv](reverse_dcf.csv).", "",
        f"For each company: the annual growth in free cash flow to the firm (operating cash "
        f"flow − capex + after-tax interest, trailing twelve months, from SEC filings known "
        f"today) over the next {args.years} years that makes a DCF equal today's enterprise "
        f"value, with {args.terminal_growth:.1%} growth afterwards. Main column uses a 9% "
        "discount rate; the CSV also has 8% and 10%. This is a description of market "
        "expectations, not a tested signal.", "",
        "Companies: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".", "",
        "## By sector", "",
        "| Sector | Companies | Median implied growth | Median FCFF / EV | Median past 3y FCF growth |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    md += [f"| {s} | {int(r.companies)} | {fmt_pct(r.median_implied)} | {r.median_fcf_yield:.1%} | "
           f"{fmt_pct(r.median_past_fcf_growth)} |" for s, r in by_sector.sort_values("median_implied").iterrows()]
    md += ["", "## Lowest expectations (market prices in decline or slow growth)", "", *header]
    md += [row(s, r) for s, r in ranked.head(15).iterrows()]
    md += ["", "## Highest expectations", "", *header]
    md += [row(s, r) for s, r in ranked.tail(15).iloc[::-1].iterrows()]
    md += ["", "## Caveats", "",
           "* Terminal value is usually 60–80% of the DCF, so small changes in the discount "
           "rate move the implied growth a lot (compare the 8% and 10% columns).",
           "* Trailing free cash flow can be temporarily depressed (heavy investment) or "
           "inflated (working-capital release); a low or negative base inflates implied growth.",
           "* Debt is the tagged current plus non-current debt (or a reported total); leases and "
           "pensions are left out. Companies with no debt figure but sizeable non-current "
           "liabilities are excluded, as are " + ", ".join(sorted(CAPTIVE_FINANCE)) + " (captive "
           "finance arms).",
           "* FCFF / EV is the yield the growth rate is solved from: free cash flow plus after-tax "
           "interest, over enterprise value (market cap + debt − cash).",
           "* Past growth uses the figures known three years ago, so it matches what investors saw."]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("Wrote %s.md/.csv (%d companies with an implied growth rate)", out, ranked.shape[0])


if __name__ == "__main__":
    main()
