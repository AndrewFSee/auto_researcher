"""
Customer momentum (Cohen & Frazzini 2008) in S&P 500 suppliers, 2018-2026.

Rules pre-registered in ``docs/results/customer_momentum_protocol.md`` (approved
2026-11-17, before any supplier or customer return was computed):

* Links: the frozen customer extraction (52834aa) on 10-Ks filed 2018 to October
  2026 (listed customer, 10%+ of total revenue, for the year the filing reports),
  plus inline-XBRL concentration tags naming a listed company. A filing's links
  count from the first month-end after its filing date until the supplier's next
  10-K is filed, or for 15 months.
* Signal at month-end t: the share-weighted total return of the supplier's
  customers over month t ("10% or more" weighs 10).
* Long the top third, short the bottom third, equal weight, held for month t+1;
  10 bps per unit traded; months with fewer than 30 suppliers skipped.
* Supported if the long-short net monthly return has a Newey-West t (3 lags) of
  2.0 or more over the full sample. No holdout.

Commands::

    python scripts/customer_momentum.py prices   # monthly returns and the price-source check
    python scripts/customer_momentum.py run      # links, portfolios, docs/results/customer_momentum.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import customer_links as cl  # noqa: E402
from validate_signal_ic import deflated_sharpe, newey_west_tstat  # noqa: E402

from auto_researcher.console import use_utf8_output  # noqa: E402

logger = logging.getLogger("customer_momentum")

YEARS = range(2018, 2027)
MODEL = "qwen3:8b"
SENTIQ_PRICES = cl.SENTIQ / "ref" / "prices"            # Tradier
SENTIQ_TIINGO = cl.SENTIQ / "ref" / "prices_tiingo"     # Tiingo, keeps delisted stocks
YF_CACHE = cl.DATA / "prices_yf"
RETURNS = cl.DATA / "monthly_returns.parquet"
SOURCE_CHECK = cl.DATA / "price_source_check.json"
OUT = ROOT / "docs" / "results" / "customer_momentum.md"
OUT_CSV = ROOT / "docs" / "results" / "customer_momentum.csv"

FIRST_SIGNAL = pd.Timestamp("2018-01-01")   # signal months January 2018 ...
LAST_SIGNAL = pd.Timestamp("2026-09-30")    # ... to September 2026; holding months to October 2026
LINK_MONTHS = 15
MIN_SUPPLIERS = 30
COST = 0.001        # 10 bps per unit traded
NW_LAG = 3
THRESHOLD_WEIGHT = 10.0
RENAMED = {"ABC": "COR"}  # lake ticker -> the ticker the price files use
SPLIT = pd.Timestamp("2023-12-31")  # 2018-2023 and 2024-2026 holding months, reported separately
N_VARIANTS = 4


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #

def month_ends() -> pd.DatetimeIndex:
    """The last NYSE session of each month, from SPY's trading days."""
    spy = pd.read_parquet(SENTIQ_PRICES / "SPY.parquet")
    days = pd.DatetimeIndex(pd.to_datetime(spy["date"]))
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby(days.to_period("M")).max().to_numpy())


def _sentiq_tri(ticker: str) -> tuple[pd.Series, str] | None:
    """Daily total-return index from a sentiq price file (closes are split-adjusted; each
    dividend is added back on the first session on or after its ex-date)."""
    for folder, name in ((SENTIQ_PRICES, "tradier"), (SENTIQ_TIINGO, "tiingo")):
        p = folder / f"{ticker}.parquet"
        if not p.exists():
            continue
        px = pd.read_parquet(p)
        close = pd.Series(px["close"].to_numpy(float), index=pd.to_datetime(px["date"])).sort_index()
        close = close[~close.index.duplicated(keep="last")].dropna()
        close = close[close > 0]
        if len(close) < 2:
            continue
        div = pd.Series(0.0, index=close.index)
        dp = folder / f"{ticker}.dividends.parquet"
        if dp.exists():
            for ex, amount in pd.read_parquet(dp)[["ex_date", "amount"]].itertuples(index=False):
                at = close.index.searchsorted(pd.Timestamp(ex))
                if 0 < at < len(close):
                    div.iloc[at] += float(amount)
        ret = (close + div) / close.shift(1) - 1
        return (1 + ret.fillna(0)).cumprod(), name
    return None


def _yf_tri(ticker: str) -> pd.Series | None:
    """Daily adjusted close from yfinance (dividends and splits in the adjustment), cached."""
    YF_CACHE.mkdir(parents=True, exist_ok=True)
    p = YF_CACHE / f"{ticker}.parquet"
    if p.exists():
        s = pd.read_parquet(p)["close"]
    else:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start="2017-11-01", end="2026-11-15", auto_adjust=True, progress=False)
        s = df["Close"].squeeze("columns").dropna() if len(df) else pd.Series(dtype=float)
        s = pd.Series(s.to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(s.index)), name="close")
        if s.index.tz is not None:
            s.index = s.index.tz_convert(None)
        pd.DataFrame({"close": s}).to_parquet(p)
    s = s[s > 0]
    return s / s.iloc[0] if len(s) >= 2 else None


def monthly(tri: pd.Series, ends: pd.DatetimeIndex) -> pd.Series:
    """Month t's return, from the last close on or before month-end t-1 to the last close on or
    before month-end t. Missing before the first close; after the last close, a stock earns
    its return to the last close in that month, then nothing (cash)."""
    first, last = tri.index[0], tri.index[-1]
    level = tri.reindex(tri.index.union(ends)).ffill().reindex(ends)
    out = level / level.shift(1) - 1
    prev = pd.Series(ends, index=ends).shift(1)
    ok = (prev >= first) & (prev < last)
    return out.where(ok)


def build_returns(tickers: list[str], check: bool) -> pd.DataFrame:
    """Monthly total returns for ``tickers`` (sentiq first, yfinance for the rest), and with
    ``check`` the source comparison on tickers both cover."""
    ends = month_ends()
    ends = ends[(ends >= pd.Timestamp("2017-12-01")) & (ends <= pd.Timestamp("2026-10-31"))]
    cols, source, compare = {}, {}, {}
    for n, t in enumerate(sorted(set(tickers)), 1):
        name = RENAMED.get(t, t)
        got = _sentiq_tri(name)
        if got is not None:
            cols[t], source[t] = monthly(got[0], ends), got[1]
            if check:
                yf_tri = _yf_tri(name)
                if yf_tri is not None:
                    both = pd.concat([cols[t], monthly(yf_tri, ends)], axis=1).dropna()
                    if len(both):
                        compare[t] = {"months": len(both),
                                      "off": int(((both.iloc[:, 0] - both.iloc[:, 1]).abs() > 0.005).sum())}
        else:
            yf_tri = _yf_tri(name)
            if yf_tri is not None:
                cols[t], source[t] = monthly(yf_tri, ends), "yfinance"
            else:
                source[t] = "none"
        if n % 25 == 0:
            logger.info("%d / %d tickers", n, len(set(tickers)))
    rets = pd.DataFrame(cols)
    rets.attrs["source"] = source
    if check:
        SOURCE_CHECK.write_text(json.dumps({"source": source, "compare": compare}, indent=1), encoding="utf-8")
    return rets


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #

def link_filings(require_all: bool = True) -> list[dict]:
    """Every 10-K of 2018-2026 with its customer weights: text links (share, "10% or more" as
    10) and XBRL-only pairs (the tagged share); the filer's own ticker never included. Without
    ``require_all``, a year not yet extracted is skipped (for fetching prices early)."""
    index = cl.tickers()
    current = {}
    sec = json.loads(cl.TICKERS.read_text(encoding="utf-8"))
    for r in sec.get("data", sec).values():
        current.setdefault(int(r["cik_str"]), r["ticker"])
    out = []
    for y in YEARS:
        path = cl.run_path(y, MODEL, False)
        if not path.exists() or (y == max(YEARS) and not (cl.DATA / f"report_{y}.md").exists()):
            if require_all:
                raise SystemExit(f"no finished extraction for {y}: run customer_links.py extract, xbrl and report first")
            logger.info("%d: not extracted yet, skipped", y)
            continue
        res = cl.results(cl.load_10ks(y), path)
        rows = cl.link_rows(res, index)
        xb_path = cl.DATA / f"xbrl_{y}.json"
        xbrl = json.loads(xb_path.read_text(encoding="utf-8")) if xb_path.exists() else {}
        pairs = cl.xbrl_pairs(res, xbrl, index)
        by_acc: dict[str, dict] = {r["accession"]: {} for r in res}
        src: dict[str, dict] = {r["accession"]: {} for r in res}
        for r in rows:
            if r["link"]:
                by_acc[r["accession"]][r["match"][0]] = r["percent"] or THRESHOLD_WEIGHT
                src[r["accession"]][r["match"][0]] = "text"
        for (_, t), p in pairs.items():
            if t not in by_acc[p["accession"]]:
                by_acc[p["accession"]][t] = p["percent"]
                src[p["accession"]][t] = "xbrl"
        for r in res:
            own = {r["ticker"], current.get(int(r["cik"]), r["ticker"])}
            out.append({"cik": int(r["cik"]), "supplier": r["ticker"], "price_ticker": current.get(int(r["cik"]), r["ticker"]),
                        "accession": r["accession"], "filed": r["filed"],
                        "customers": {t: w for t, w in by_acc[r["accession"]].items() if t not in own},
                        "source": {t: s for t, s in src[r["accession"]].items() if t not in own}})
        logger.info("%d: %d filings, %d with links", y, len(res), sum(bool(by_acc[r["accession"]]) for r in res))
    return out


def links_at(filings: list[dict], ends: pd.DatetimeIndex) -> dict[pd.Timestamp, dict[int, dict]]:
    """For each month-end, each supplier's links in force: those of its latest 10-K filed before
    the month-end, if filed within ``LINK_MONTHS`` months."""
    by_cik: dict[int, list[dict]] = {}
    for f in filings:
        by_cik.setdefault(f["cik"], []).append(f)
    for fs in by_cik.values():
        fs.sort(key=lambda f: f["filed"])
    out = {}
    for m in ends:
        now = {}
        for cik, fs in by_cik.items():
            prior = [f for f in fs if pd.Timestamp(f["filed"]) < m]
            if prior and m <= pd.Timestamp(prior[-1]["filed"]) + pd.DateOffset(months=LINK_MONTHS) and prior[-1]["customers"]:
                now[cik] = prior[-1]
        out[m] = now
    return out


# --------------------------------------------------------------------------- #
# Portfolios
# --------------------------------------------------------------------------- #

def signals(live: dict, rets: pd.DataFrame, ends: pd.DatetimeIndex, drop: set[str] = frozenset()) -> pd.DataFrame:
    """One row per supplier and signal month: customer momentum (month t), the supplier's own
    month-t return and its month t+1 return; suppliers without a t+1 return are left out."""
    rows = []
    for i, m in enumerate(ends[:-1]):
        nxt = ends[i + 1]
        for cik, f in live[m].items():
            w = {t: v for t, v in f["customers"].items() if t not in drop and t in rets and pd.notna(rets.at[m, t])}
            fwd = rets.at[nxt, f["price_ticker"]] if f["price_ticker"] in rets else np.nan
            if not w or pd.isna(fwd):
                continue
            cm = sum(v * rets.at[m, t] for t, v in w.items()) / sum(w.values())
            own = rets.at[m, f["price_ticker"]]
            rows.append({"month": m, "hold": nxt, "supplier": f["price_ticker"], "cm": cm, "own": own, "fwd": fwd,
                         "customers": ",".join(sorted(w))})
    return pd.DataFrame(rows)


def portfolio(sig: pd.DataFrame, buckets: int) -> pd.DataFrame:
    """Long the top 1/``buckets`` by customer momentum, short the bottom, equal weight; gross and
    net (10 bps on the change in weights) returns per holding month."""
    out, prev = [], pd.Series(dtype=float)
    for m, g in sig.groupby("month"):
        if len(g) < MIN_SUPPLIERS:
            continue
        g = g.sort_values(["cm", "supplier"])
        k = len(g) // buckets
        short, long_ = g.head(k), g.tail(k)
        w = pd.concat([pd.Series(1.0 / k, index=long_["supplier"]), pd.Series(-1.0 / k, index=short["supplier"])])
        w = w.groupby(level=0).sum()
        turnover = w.subtract(prev, fill_value=0).abs().sum()
        gross = long_["fwd"].mean() - short["fwd"].mean()
        out.append({"hold": g["hold"].iloc[0], "n": len(g), "k": k, "long": long_["fwd"].mean(),
                    "short": short["fwd"].mean(), "gross": gross, "turnover": turnover,
                    "net": gross - COST * turnover})
        prev = w
    cols = ["hold", "n", "k", "long", "short", "gross", "turnover", "net"]
    return pd.DataFrame(out, columns=cols).set_index("hold")


def stats(x: pd.Series, n_trials: int = N_VARIANTS) -> dict:
    x = x.dropna()
    t, p = newey_west_tstat(x, NW_LAG)
    sr = x.mean() / x.std(ddof=1) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan
    curve = (1 + x).cumprod()
    return {"months": len(x), "mean": x.mean(), "t": t, "p": p, "ann_mean": 12 * x.mean(),
            "sharpe": sr * np.sqrt(12), "dsr": deflated_sharpe(sr, n_trials, len(x)),
            "max_dd": float((curve / curve.cummax() - 1).min()), "hit": float((x > 0).mean())}


def fama_macbeth(sig: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Monthly OLS of the t+1 return on customer momentum and the supplier's own month-t
    return; the customer-momentum coefficient and, for comparison, without the control."""
    b, b0 = {}, {}
    for m, g in sig.dropna(subset=["cm", "own", "fwd"]).groupby("hold"):
        if len(g) < MIN_SUPPLIERS:
            continue
        X = np.column_stack([np.ones(len(g)), g["cm"], g["own"]])
        b[m] = np.linalg.lstsq(X, g["fwd"].to_numpy(), rcond=None)[0][1]
        b0[m] = np.linalg.lstsq(X[:, :2], g["fwd"].to_numpy(), rcond=None)[0][1]
    return pd.Series(b), pd.Series(b0)


def rank_ic(sig: pd.DataFrame) -> pd.Series:
    from scipy.stats import spearmanr
    ic = {}
    for m, g in sig.groupby("hold"):
        if len(g) >= MIN_SUPPLIERS:
            ic[m] = spearmanr(g["cm"], g["fwd"])[0]
    return pd.Series(ic)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _all_tickers(filings: list[dict]) -> list[str]:
    t = {"SPY"}
    for f in filings:
        if f["customers"]:
            t.add(f["price_ticker"])
            t |= set(f["customers"])
    return sorted(t)


def cmd_prices(args) -> None:
    filings = link_filings(require_all=False)
    rets = build_returns(_all_tickers(filings), check=True)
    rets.to_parquet(RETURNS)
    src = rets.attrs["source"]
    logger.info("wrote %s: %d tickers (%s)", RETURNS, rets.shape[1], dict(Counter(src.values())))


def _fmt(s: dict, pct: bool = True) -> str:
    m = f"{100 * s['mean']:+.2f}%" if pct else f"{s['mean']:+.3f}"
    return f"{m} | {s['t']:.2f} | {s['months']}"


def cmd_run(args) -> None:
    filings = link_filings()
    ends = month_ends()
    ends = ends[(ends >= FIRST_SIGNAL) & (ends <= LAST_SIGNAL + pd.offsets.MonthEnd(1))]
    tickers = _all_tickers(filings)
    check = json.loads(SOURCE_CHECK.read_text(encoding="utf-8")) if SOURCE_CHECK.exists() else {"source": {}, "compare": {}}
    if RETURNS.exists() and set(tickers) <= set(pd.read_parquet(RETURNS).columns) | {t for t, s in check["source"].items() if s == "none"}:
        rets = pd.read_parquet(RETURNS)
    else:
        rets = build_returns(tickers, check=True)
        rets.to_parquet(RETURNS)
        check = json.loads(SOURCE_CHECK.read_text(encoding="utf-8"))
    rets = rets.reindex(ends)
    live = links_at(filings, ends)

    sig = signals(live, rets, ends)
    sig_x = signals(live, rets, ends, drop={"WMT"})
    port3, port5, port_x = portfolio(sig, 3), portfolio(sig, 5), portfolio(sig_x, 3)
    spy = rets["SPY"]
    primary = stats(port3["net"])
    variants = {"Terciles, net (primary)": port3["net"], "Terciles, gross": port3["gross"],
                "Quintiles, net": port5["net"], "Without Walmart's links, net": port_x["net"]}
    fm, fm0 = fama_macbeth(sig)
    ic = rank_ic(sig)
    early, late = port3[port3.index <= SPLIT], port3[port3.index > SPLIT]

    # Counts
    per_month = sig.groupby("month")
    n_sup = per_month.size()
    n_cus = per_month["customers"].apply(lambda s: len({c for x in s for c in x.split(",")}))
    wmt = sig["customers"].str.split(",").apply(lambda c: "WMT" in c).mean()
    linked = {f["price_ticker"] for f in filings if f["customers"]}
    customers = {t for f in filings for t in f["customers"]}
    src = check["source"]
    no_price = sorted(t for t in linked | customers if src.get(t, "none") == "none" or t not in rets)
    off = {t: c for t, c in check["compare"].items() if c["off"] > 0.1 * c["months"]}
    n_links = sum(len(f["customers"]) for f in filings)
    n_xbrl = sum(s == "xbrl" for f in filings for s in f["source"].values())

    verdict = "supported" if primary["t"] >= 2.0 else "not supported"
    lines = [
        "# Customer momentum in S&P 500 suppliers",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/customer_momentum.py`, under the rules pre-registered "
        "in [customer_momentum_protocol.md](customer_momentum_protocol.md). Deviations from the protocol: none.",
        "",
        f"**Result: {verdict}.** The tercile long-short net return averaged {100 * primary['mean']:+.2f}% a month "
        f"(Newey-West t = {primary['t']:.2f}, {primary['months']} months); the decision rule needs t ≥ 2.0.",
        "",
        "## Data",
        "",
        f"* 10-Ks filed 2018 to October 2026: {len(filings)}; with a link: {sum(bool(f['customers']) for f in filings)} "
        f"({n_links} supplier-customer links in all, {n_xbrl} of them from XBRL tags only).",
        f"* Suppliers with a link at some point: {len(linked)}; distinct customers: {len(customers)}.",
        f"* Price sources: " + ", ".join(f"{k} {v}" for k, v in Counter(src.get(t, 'none') for t in linked | customers).most_common())
        + f". Without prices (left out): {', '.join(no_price) or 'none'}.",
        f"* Source check (sentiq vs yfinance, {len(check['compare'])} tickers in both): monthly returns differing by more "
        f"than 0.5 points in {sum(c['off'] for c in check['compare'].values())} of "
        f"{sum(c['months'] for c in check['compare'].values())} ticker-months; tickers over 10% of their months: "
        + (", ".join(f"{t} ({c['off']}/{c['months']})" for t, c in sorted(off.items())) or "none") + " (sentiq used).",
        f"* Suppliers in the sort each month: min {n_sup.min()}, median {int(n_sup.median())}, max {n_sup.max()}; "
        f"distinct customers behind them: median {int(n_cus.median())}. Supplier-months whose signal includes Walmart: "
        f"{wmt:.0%}. Months skipped (fewer than {MIN_SUPPLIERS} suppliers): "
        f"{int((n_sup < MIN_SUPPLIERS).sum())}.",
        f"* Mean monthly turnover of the tercile portfolio: {port3['turnover'].mean():.2f} (sum of absolute weight "
        f"changes; costs {1e4 * COST:.0f} bps per unit, {100 * COST * port3['turnover'].mean():.2f}% a month).",
        "",
        "## Primary test",
        "",
        "| Portfolio | Mean / month | NW t | Months | Ann. mean | Sharpe | Deflated P(SR>0) | Max drawdown | Months > 0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, x in variants.items():
        s = stats(x)
        lines.append(f"| {name} | {100 * s['mean']:+.2f}% | {s['t']:.2f} | {s['months']} | {100 * s['ann_mean']:+.1f}% | "
                     f"{s['sharpe']:.2f} | {s['dsr']:.2f} | {100 * s['max_dd']:.1f}% | {s['hit']:.0%} |")
    legs = pd.DataFrame({"long": port3["long"] - spy.reindex(port3.index), "short": port3["short"] - spy.reindex(port3.index)})
    lines += [
        "",
        "Only the first row decides; the others are reported to explain it.",
        "",
        "## Secondary measures",
        "",
        "| Measure | Mean | NW t | Months |",
        "| --- | ---: | ---: | ---: |",
        f"| Long leg minus SPY (gross) | {_fmt(stats(legs['long']))} |",
        f"| Short leg minus SPY (gross) | {_fmt(stats(legs['short']))} |",
        f"| Fama-MacBeth slope on customer momentum, controlling for own month-t return | {_fmt(stats(fm), pct=False)} |",
        f"| Fama-MacBeth slope on customer momentum, no control | {_fmt(stats(fm0), pct=False)} |",
        f"| Rank correlation of customer momentum with next month's return | {_fmt(stats(ic), pct=False)} |",
        f"| Terciles net, holding months 2018-02 to 2023-12 | {_fmt(stats(early['net']))} |",
        f"| Terciles net, holding months 2024-01 to 2026-10 | {_fmt(stats(late['net']))} |",
        "",
        "The Fama-MacBeth slope is the next-month return per unit of customer return (0.10 means a supplier whose "
        "customers rose 10% earns 1 point more the next month).",
        "",
        "## By year (terciles, net)",
        "",
        "| Year | Months | Mean / month | Suppliers (median) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for y, g in port3.groupby(port3.index.year):
        lines.append(f"| {y} | {len(g)} | {100 * g['net'].mean():+.2f}% | {int(g['n'].median())} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    port3.join(port5[["net"]].rename(columns={"net": "net_quintiles"})).join(
        port_x[["net"]].rename(columns={"net": "net_ex_wmt"})).to_csv(OUT_CSV)
    logger.info("wrote %s and %s: %s (t = %.2f)", OUT, OUT_CSV, verdict, primary["t"])


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prices", help="monthly returns and the price-source check (no signal)")
    sub.add_parser("run", help="links, portfolios and the results")
    args = p.parse_args()
    {"prices": cmd_prices, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
