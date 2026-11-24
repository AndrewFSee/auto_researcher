"""Point-in-time fundamentals and price metrics from TickerLake (read-only).

Fundamentals: TickerLake's SEC company facts (``data/filings_facts/sym_<T>.parquet``),
mapped onto ``data.sec_fundamentals.ITEMS`` and turned into the same point-in-time
table (first-reported values, TTM flows, usable only once filed). A value counts if its
filing was on or before the as-of date. Prices: TickerLake's daily OHLCV, which keeps
delisted names.

Every number here is computed by code; the LLM only chooses which ones to show.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from ...data.sec_fundamentals import FACT_COLUMNS, FORMS, ITEMS, build_pit_table
from ..schemas import Metric, Unit

TICKERLAKE = Path(r"C:\Users\andre\projects\TickerLake\data")
FUNDAMENTAL_MAX_AGE = 200   # days from period end: a 10-K can arrive 90 days after year end
PRICE_MAX_AGE = 5
MAX_PERIOD_AGE = 400        # older periods count as missing (as in fundamentals_asof)

CATALOGUE: dict[str, tuple[Unit, str]] = {
    "revenue_ttm": ("usd", "revenue, trailing twelve months"),
    "revenue_growth": ("pct", "TTM revenue vs the TTM a year earlier"),
    "gross_margin": ("ratio", "gross profit / revenue, TTM"),
    "operating_margin": ("ratio", "operating income / revenue, TTM"),
    "net_margin": ("ratio", "net income / revenue, TTM"),
    "operating_cash_flow_ttm": ("usd", "cash from operations, TTM"),
    "free_cash_flow_ttm": ("usd", "cash from operations minus capex, TTM"),
    "cfo_to_net_income": ("ratio", "cash from operations / net income, TTM (earnings quality)"),
    "accruals_to_assets": ("ratio", "(net income - cash from operations) / total assets"),
    "total_debt": ("usd", "non-current + current debt incl. short-term borrowings (leases excluded)"),
    "cash_and_investments": ("usd", "cash + short-term investments"),
    "net_debt": ("usd", "total debt - cash and investments"),
    "ebitda_ttm": ("usd", "operating income + depreciation and amortization, TTM"),
    "net_debt_to_ebitda": ("x", "net debt / EBITDA"),
    "interest_coverage": ("x", "operating income / interest expense, TTM"),
    "current_ratio": ("x", "current assets / current liabilities"),
    "buybacks_ttm": ("usd", "share repurchases, TTM"),
    "dividends_ttm": ("usd", "dividends paid, TTM"),
    "sbc_to_revenue": ("ratio", "stock compensation / revenue, TTM"),
    "share_count_change_1y": ("pct", "diluted shares vs a year earlier"),
    "price": ("usd", "last close"),
    "market_cap": ("usd", "last close x shares outstanding"),
    "pe_ttm": ("x", "market cap / TTM net income (positive earnings only)"),
    "ev_to_ebitda": ("x", "(market cap + net debt) / EBITDA"),
    "fcf_yield": ("pct", "free cash flow / market cap"),
    "return_1m": ("pct", "total return over 21 trading days"),
    "return_12m": ("pct", "total return over 252 trading days"),
    "return_12m_vs_spy": ("pct", "12-month total return minus SPY's"),
    "volatility_63d": ("pct", "annualized volatility of daily returns, 63 days"),
    "max_drawdown_252d": ("pct", "largest peak-to-trough fall over 252 days"),
    "beta_252d": ("ratio", "beta of daily returns to SPY, 252 days"),
}
PRICE_METRICS = {"price", "return_1m", "return_12m", "return_12m_vs_spy", "volatility_63d", "max_drawdown_252d",
                 "beta_252d"}
MIXED_METRICS = {"market_cap", "pe_ttm", "ev_to_ebitda", "fcf_yield"}   # last close with filed fundamentals
# Given to the drafter with the metric (``Metric.definition``) where the catalogue line
# alone invites a misreading.
READINGS: dict[str, str] = {
    "beta_252d": ("Beta is correlation times the ratio of the stock's volatility to SPY's: a low beta can go with "
                  "high volatility when correlation is low. It is not a measure of total risk."),
    "pe_ttm": "A one-off gain raises net income and so lowers P/E; a one-off charge raises it.",
    "ev_to_ebitda": "A one-off gain in operating income raises EBITDA and so lowers EV/EBITDA.",
    "return_12m": "A price return alone does not show whether the valuation multiple changed.",
}
# For a claim that a multiple or margin has changed, the drafter needs both values: when
# the plan asks for one of these, the data analyst also gets its value computed with only
# the data public a year before the as-of date (name + "_1y_ago"). Not in CATALOGUE, so
# the plan's inputs don't change.
PAST_METRICS = ("gross_margin", "operating_margin", "net_margin", "pe_ttm", "ev_to_ebitda", "fcf_yield")
PAST_DAYS = 365


def facts_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """TickerLake's company facts as ``sec_fundamentals.extract_facts`` returns them."""
    raw = raw[raw["form_type"].isin(FORMS) & raw["value"].notna() & raw["filed_date"].notna()]
    parts = []
    for item, spec in ITEMS.items():
        for priority, name in enumerate(spec.concepts):
            taxonomy, concept = name.split(":")
            g = raw[(raw["taxonomy"] == taxonomy) & (raw["concept"] == concept) & (raw["unit"] == spec.unit)]
            if len(g):
                parts.append(pd.DataFrame({
                    "symbol": symbol, "item": item, "concept": concept, "priority": priority,
                    "start": g["start_date"].to_numpy(), "end": g["end_date"].to_numpy(),
                    "value": g["value"].astype(float).to_numpy(), "filed": g["filed_date"].to_numpy(),
                    "accn": g["accession_number"].to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=FACT_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    df["priority"] = df["priority"].astype("int64")
    return df[FACT_COLUMNS]


@dataclass(frozen=True)
class Point:
    value: float
    period_end: date
    available: date


def latest(table: pd.DataFrame, item: str, as_of: date, end: date | None = None, tol_days: int = 0) -> Point | None:
    """The value public by ``as_of`` for the latest period (or the period ending within
    ``tol_days`` of ``end``); for one period the earliest-filed version."""
    g = table[(table["item"] == item) & (table["available_date"] <= pd.Timestamp(as_of))]
    if end is not None:
        g = g[(g["period_end"] - pd.Timestamp(end)).abs() <= pd.Timedelta(days=tol_days)]
    else:
        g = g[g["period_end"] >= pd.Timestamp(as_of) - pd.Timedelta(days=MAX_PERIOD_AGE)]
    if g.empty:
        return None
    g = g.sort_values(["period_end", "available_date"], ascending=[False, True])
    r = g.iloc[0]
    return Point(float(r["value"]), r["period_end"].date(), r["available_date"].date())


DEBT_NONCURRENT = ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations")
DEBT_CURRENT = ("DebtCurrent",)  # current debt including short-term borrowings
LTD_CURRENT = ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent")
SHORT_TERM = ("ShortTermBorrowings", "CommercialPaper")
LTD_TOTAL = ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities")
DEBT_COMBINED = ("DebtLongtermAndShorttermCombinedAmount",)


def total_debt(raw: pd.DataFrame, as_of: date) -> Point | None:
    """Debt on the latest balance sheet filed by ``as_of`` (first-reported values, leases
    excluded): non-current debt plus current debt, where current debt is ``DebtCurrent`` if
    tagged, else current long-term debt plus short-term borrowings.

    ``sec_fundamentals.ITEMS["debt_current"]`` lists those concepts as alternatives and takes
    the first one tagged, so it misses short-term borrowings whenever current long-term debt
    is also tagged (Walmart, 2026-07-31: $10.5B of short-term borrowings dropped from $50.4B).
    That module is left unchanged because earlier studies depend on it."""
    if raw.empty or "concept" not in raw.columns:
        return None
    r = raw[(raw["unit"] == "USD") & raw["form_type"].isin(FORMS) & raw["value"].notna()
            & raw["start_date"].isna() & (raw["taxonomy"] == "us-gaap")].copy()
    r["filed"], r["end"] = pd.to_datetime(r["filed_date"]), pd.to_datetime(r["end_date"])
    r = r[(r["filed"] <= pd.Timestamp(as_of)) & (r["end"] >= pd.Timestamp(as_of) - pd.Timedelta(days=MAX_PERIOD_AGE))]
    anchors = r[r["concept"].isin(DEBT_NONCURRENT + LTD_TOTAL + DEBT_COMBINED)]
    if anchors.empty:
        return None
    end = anchors["end"].max()
    at = r[r["end"] == end].sort_values("filed").drop_duplicates("concept")
    val, filed = dict(zip(at["concept"], at["value"])), dict(zip(at["concept"], at["filed"]))

    def first(names: tuple[str, ...]) -> str | None:
        return next((n for n in names if n in val), None)

    used: list[str] = []
    if non := first(DEBT_NONCURRENT):
        cur = first(DEBT_CURRENT)
        used = [non, cur] if cur else [non, *(n for n in (first(LTD_CURRENT), first(SHORT_TERM)) if n)]
    elif tot := first(LTD_TOTAL):
        used = [tot, *(n for n in (first(SHORT_TERM),) if n)]
    elif comb := first(DEBT_COMBINED):
        used = [comb]
    return Point(float(sum(val[n] for n in used)), end.date(), max(filed[n] for n in used).date())


class TickerLakeMetrics:
    version = "tickerlake-1"

    def __init__(self, root: Path = TICKERLAKE) -> None:
        self.root = root
        self._tables: dict[str, pd.DataFrame] = {}
        self._raw: dict[str, pd.DataFrame] = {}
        self._splits: dict[str, pd.Series] = {}

    def available(self) -> dict[str, str]:
        return {k: v[1] for k, v in CATALOGUE.items()}

    # -- data ------------------------------------------------------------------------
    def table(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._tables:
            path = self.root / "filings_facts" / f"sym_{symbol}.parquet"
            raw = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["form_type", "value", "filed_date"])
            self._raw[symbol] = raw
            self._tables[symbol] = build_pit_table(facts_frame(raw, symbol)) if len(raw) else pd.DataFrame(
                columns=["symbol", "item", "period_end", "available_date", "value"])
        return self._tables[symbol]

    def prices(self, symbols: list[str], as_of: date, years: int = 2) -> pd.DataFrame:
        """Adjusted closes up to ``as_of`` (dates x symbols)."""
        d = ds.dataset(self.root / "ohlcv", format="parquet", partitioning="hive")
        flt = (ds.field("year") >= as_of.year - years) & (ds.field("year") <= as_of.year) \
            & ds.field("symbol").isin(pa.array(symbols))
        t = d.to_table(filter=flt, columns=["symbol", "date", "close", "adj_close"]).to_pandas()
        if t.empty:
            return pd.DataFrame()
        t["date"] = pd.to_datetime(t["date"])
        t = t[t["date"] <= pd.Timestamp(as_of)]
        return t.pivot_table(index="date", columns="symbol", values=["close", "adj_close"]).sort_index()

    def splits(self, symbol: str) -> pd.Series:
        """Split ratios by date (10.0 for 10-for-1), all years."""
        if symbol not in self._splits:
            d = ds.dataset(self.root / "ohlcv", format="parquet", partitioning="hive")
            if "stock_splits" not in d.schema.names:
                self._splits[symbol] = pd.Series(dtype=float)
            else:
                t = d.to_table(filter=ds.field("symbol") == pa.scalar(symbol),
                               columns=["date", "stock_splits"]).to_pandas()
                t = t[t["stock_splits"].fillna(0) > 0]
                self._splits[symbol] = pd.Series(t["stock_splits"].to_numpy(float),
                                                 index=pd.to_datetime(t["date"])).sort_index()
        return self._splits[symbol]

    def on_price_basis(self, symbol: str, shares: Point) -> Point:
        """A share count on the split basis of the stored closes, which are adjusted for every
        split up to the download. A filing reports counts on the basis of its filing date, so
        the splits after that date apply (NFLX: 425M shares reported in July 2025 are 4.25B
        after the 10-for-1 split of 2025-11-17)."""
        sp = self.splits(symbol)
        factor = float(np.prod(sp[sp.index > pd.Timestamp(shares.available)].to_numpy())) if len(sp) else 1.0
        return Point(shares.value * factor, shares.period_end, shares.available)

    # -- metrics ---------------------------------------------------------------------
    def metrics(self, ticker: str, names: list[str], as_of: date) -> list[Metric]:
        values = self.compute(ticker, as_of)
        out = []
        for i, name in enumerate(names, 1):
            if name in CATALOGUE:
                out.append(self._metric(f"m{i}", name, name, values.get(name), as_of))
        past = [n for n in PAST_METRICS if n in names]
        if past:
            then = self.compute(ticker, as_of - timedelta(days=PAST_DAYS))
            for k, name in enumerate(past, len(names) + 1):
                out.append(self._metric(f"m{k}", f"{name}_1y_ago", name, then.get(name), as_of - timedelta(days=PAST_DAYS),
                                        extra_age=PAST_DAYS))
        return out

    @staticmethod
    def _metric(mid: str, name: str, base: str, v: tuple[float, date, date, str] | None, as_of: date,
                extra_age: int = 0) -> Metric:
        unit, desc = CATALOGUE[base]
        definition = " ".join(filter(None, [desc + ".", READINGS.get(base)]))
        if extra_age:
            definition = f"{definition} Computed with only the data public on {as_of}, a year before the as-of date."
        if v is None:
            return Metric(id=mid, name=name, value=None, unit=unit, as_of=as_of, source="TickerLake",
                          snapshot_id="unavailable", definition=definition)
        value, data_date, available, snap = v
        if base in PRICE_METRICS:
            period, max_age = None, PRICE_MAX_AGE
        elif base in MIXED_METRICS:
            # Dated by the last close, but built on the latest filed period: say both.
            period, max_age = f"price {data_date}, fundamentals to {snap.rsplit('@', 1)[-1]}", PRICE_MAX_AGE
        else:
            period, max_age = f"to {data_date}", FUNDAMENTAL_MAX_AGE
        return Metric(id=mid, name=name, value=value, unit=unit, as_of=data_date, available_date=available,
                      period=period, source="TickerLake " + ("OHLCV" if base in PRICE_METRICS else "SEC company facts"),
                      snapshot_id=snap, definition=definition, max_age_days=max_age + extra_age)

    def compute(self, ticker: str, as_of: date) -> dict[str, tuple[float, date, date, str]]:
        """name -> (value, data date, date it became public, snapshot id)."""
        t = self.table(ticker)
        out: dict[str, tuple[float, date, date, str]] = {}

        def put(name: str, value: float | None, *pts: Point | None) -> None:
            if value is None or not np.isfinite(value) or any(p is None for p in pts):
                return
            ps = [p for p in pts if p is not None]
            end, avail = max(p.period_end for p in ps), max(p.available for p in ps)
            out[name] = (float(value), end, avail, f"facts/{ticker}@{end}")

        def ratio(a: Point | None, b: Point | None) -> float | None:
            return a.value / b.value if a and b and b.value else None

        rev, ni, cfo = (latest(t, k, as_of) for k in ("revenue", "net_income", "operating_cash_flow"))
        gp, oi, capex = (latest(t, k, as_of) for k in ("gross_profit", "operating_income", "capex"))
        dep, intx, sbc = (latest(t, k, as_of) for k in ("depreciation", "interest_expense", "stock_comp"))
        assets, ca, cl = (latest(t, k, as_of) for k in ("total_assets", "current_assets", "current_liabilities"))
        cash = latest(t, "cash", as_of)
        debt = total_debt(self._raw[ticker], as_of)

        if rev:
            put("revenue_ttm", rev.value, rev)
            prev = latest(t, "revenue", as_of, rev.period_end - timedelta(days=365), 20)
            put("revenue_growth", rev.value / prev.value - 1 if prev and prev.value else None, rev, prev)
        put("gross_margin", ratio(gp, rev), gp, rev)
        put("operating_margin", ratio(oi, rev), oi, rev)
        put("net_margin", ratio(ni, rev), ni, rev)
        if cfo:
            put("operating_cash_flow_ttm", cfo.value, cfo)
        fcf = Point(cfo.value - capex.value, max(cfo.period_end, capex.period_end),
                    max(cfo.available, capex.available)) if cfo and capex else None
        if fcf:
            put("free_cash_flow_ttm", fcf.value, fcf)
        put("cfo_to_net_income", ratio(cfo, ni) if ni and ni.value > 0 else None, cfo, ni)
        put("accruals_to_assets", (ni.value - cfo.value) / assets.value if ni and cfo and assets and assets.value
            else None, ni, cfo, assets)
        cash_inv = None
        if cash:
            sti_same = latest(t, "short_term_investments", as_of, cash.period_end, 5)
            cash_inv = Point(cash.value + (sti_same.value if sti_same else 0.0), cash.period_end, cash.available)
            put("cash_and_investments", cash_inv.value, cash_inv)
        if debt:
            put("total_debt", debt.value, debt)
        net_debt = Point(debt.value - cash_inv.value, max(debt.period_end, cash_inv.period_end),
                         max(debt.available, cash_inv.available)) if debt and cash_inv else None
        if net_debt:
            put("net_debt", net_debt.value, net_debt)
        ebitda = Point(oi.value + dep.value, max(oi.period_end, dep.period_end),
                       max(oi.available, dep.available)) if oi and dep else None
        if ebitda:
            put("ebitda_ttm", ebitda.value, ebitda)
            put("net_debt_to_ebitda", ratio(net_debt, ebitda) if ebitda.value > 0 else None, net_debt, ebitda)
        put("interest_coverage", ratio(oi, intx), oi, intx)
        put("current_ratio", ratio(ca, cl), ca, cl)
        for item, name in (("buybacks", "buybacks_ttm"), ("dividends_paid", "dividends_ttm")):
            p = latest(t, item, as_of)
            if p:
                put(name, p.value, p)
        put("sbc_to_revenue", ratio(sbc, rev), sbc, rev)
        sh = latest(t, "shares_diluted", as_of)
        sh = self.on_price_basis(ticker, sh) if sh else None
        if sh:
            prev = latest(t, "shares_diluted", as_of, sh.period_end - timedelta(days=365), 20)
            prev = self.on_price_basis(ticker, prev) if prev else None
            put("share_count_change_1y", sh.value / prev.value - 1 if prev and prev.value else None, sh, prev)

        px = self.prices([ticker, "SPY"], as_of)
        if px.empty or ("adj_close", ticker) not in px.columns:
            return out
        adj = px["adj_close"][ticker].dropna()
        if adj.empty:
            return out
        last = adj.index[-1].date()
        snap = f"ohlcv/{ticker}@{last}"

        def mput(name: str, value: float | None) -> None:
            if value is not None and np.isfinite(value):
                out[name] = (float(value), last, last, snap)

        close = float(px["close"][ticker].dropna().iloc[-1])
        mput("price", close)
        r = adj.pct_change().dropna()
        mput("return_1m", adj.iloc[-1] / adj.iloc[-22] - 1 if len(adj) > 21 else None)
        if len(adj) > 252:
            r12 = adj.iloc[-1] / adj.iloc[-253] - 1
            mput("return_12m", r12)
            spy = px["adj_close"]["SPY"].dropna() if ("adj_close", "SPY") in px.columns else pd.Series(dtype=float)
            if len(spy) > 252:
                mput("return_12m_vs_spy", r12 - (spy.iloc[-1] / spy.iloc[-253] - 1))
            window = adj.iloc[-253:]
            mput("max_drawdown_252d", float((window / window.cummax() - 1).min()))
            if len(spy) > 252:
                both = pd.concat([r, spy.pct_change()], axis=1, keys=["s", "m"]).dropna().iloc[-252:]
                mput("beta_252d", float(both["s"].cov(both["m"]) / both["m"].var()) if len(both) > 200 else None)
        mput("volatility_63d", float(r.iloc[-63:].std() * np.sqrt(252)) if len(r) >= 63 else None)

        cover = latest(t, "shares_cover", as_of)
        shares = self.on_price_basis(ticker, cover) if cover else sh
        if shares:
            mcap = close * shares.value
            out["market_cap"] = (mcap, last, last, f"{snap}+facts/{ticker}@{shares.period_end}")
            if ni and ni.value > 0:
                out["pe_ttm"] = (mcap / ni.value, last, last, f"{snap}+facts/{ticker}@{ni.period_end}")
            if ebitda and ebitda.value > 0 and net_debt:
                out["ev_to_ebitda"] = ((mcap + net_debt.value) / ebitda.value, last, last,
                                       f"{snap}+facts/{ticker}@{ebitda.period_end}")
            if fcf:
                out["fcf_yield"] = (fcf.value / mcap, last, last, f"{snap}+facts/{ticker}@{fcf.period_end}")
        return out
