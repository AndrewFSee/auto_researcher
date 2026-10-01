"""
Point-in-time fundamentals from SEC XBRL company facts.

``data.sec.gov/api/xbrl/companyfacts`` lists every value a company has tagged
in its 10-K and 10-Q filings since about 2009, each with the date it was
filed. That gives two things the usual statement datasets lack:

* **Availability dates.** A value is usable only after the filing that first
  reported it (``available_date = filed``; the as-of lookup excludes the filing
  day itself).
* **First-reported values.** Later filings repeat earlier periods as
  comparatives, sometimes restated. Keeping the earliest filing per period gives
  what investors actually saw, not the hindsight-corrected figure.

Income and cash-flow items are converted to trailing twelve months (TTM).
10-Qs mostly report year-to-date figures (cash flow statements only those), so
TTM = YTD + prior fiscal year − prior-year YTD; fourth quarters come from the
10-K annual figure.

Concept coverage is pragmatic, not complete: each item tries a short list of
us-gaap concepts in priority order, and debt is the sum of the current and
non-current portions where tagged, else a reported total (leases excluded).
Financial companies'
statements do not map well onto these items.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auto_researcher.data.fmp_earnings import DownloadSummary

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT"}
FACT_COLUMNS = ["symbol", "item", "concept", "priority", "start", "end", "value", "filed", "accn"]
TABLE_COLUMNS = ["symbol", "item", "period_end", "available_date", "value"]

# Companies that moved to a new registrant (holding-company reorganization or
# re-domicile): their history is filed under the old CIK. Verified against SEC
# submissions (2026-10). Spin-offs and mergers are deliberately not linked.
PREDECESSOR_CIKS: dict[str, tuple[int, ...]] = {
    "XOM": (34088,),     # Exxon Mobil Corp -> ExxonMobil Holdings Corp (2026)
    "BLK": (1364742,),   # BlackRock Finance, Inc. (old parent) -> BlackRock, Inc. (2024)
    "BG": (1144519,),    # Bunge Ltd -> Bunge Global SA (2023)
    "FERG": (1832433,),  # Ferguson (Jersey) Ltd -> Ferguson Enterprises Inc. (2024)
}


@dataclass(frozen=True)
class Item:
    kind: str  # "flow" (summed over a period), "instant" (balance) or "average" (share counts)
    concepts: tuple[str, ...]  # "taxonomy:Concept", highest priority first
    unit: str = "USD"


def _gaap(*names: str) -> tuple[str, ...]:
    return tuple(f"us-gaap:{n}" for n in names)


ITEMS: dict[str, Item] = {
    "revenue": Item("flow", _gaap(
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
        "SalesRevenueGoodsNet", "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense")),
    "cost_of_revenue": Item("flow", _gaap(
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices")),
    "gross_profit": Item("flow", _gaap("GrossProfit")),
    "operating_income": Item("flow", _gaap("OperatingIncomeLoss")),
    "interest_expense": Item("flow", _gaap(
        "InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt")),
    "net_income": Item("flow", _gaap(
        "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss")),
    "operating_cash_flow": Item("flow", _gaap(
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
    "capex": Item("flow", _gaap(
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements")),
    "stock_comp": Item("flow", _gaap("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense")),
    "depreciation": Item("flow", _gaap(
        "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet", "Depreciation")),
    "dividends_paid": Item("flow", _gaap("PaymentsOfDividends", "PaymentsOfDividendsCommonStock")),
    "buybacks": Item("flow", _gaap("PaymentsForRepurchaseOfCommonStock")),
    "total_assets": Item("instant", _gaap("Assets")),
    "current_assets": Item("instant", _gaap("AssetsCurrent")),
    "current_liabilities": Item("instant", _gaap("LiabilitiesCurrent")),
    "total_liabilities": Item("instant", _gaap("Liabilities")),
    "equity": Item("instant", _gaap(
        "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    "cash": Item("instant", _gaap(
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash")),
    "short_term_investments": Item("instant", _gaap(
        "ShortTermInvestments", "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent")),
    "debt_noncurrent": Item("instant", _gaap(
        "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations")),
    "debt_current": Item("instant", _gaap(
        "DebtCurrent", "LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "ShortTermBorrowings")),
    # Total debt for filers that do not split it (common for REITs and
    # homebuilders); used only when neither portion above is tagged.
    "debt_total": Item("instant", _gaap(
        "LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "DebtLongtermAndShorttermCombinedAmount", "NotesAndLoansPayable",
        "DebtInstrumentCarryingAmount", "NotesPayable", "SeniorNotes")),
    # Cover-page count (as of a date shortly before filing); absent for most
    # multi-class companies, so diluted weighted-average shares are the fallback.
    "shares_cover": Item("instant", ("dei:EntityCommonStockSharesOutstanding",), unit="shares"),
    "shares_diluted": Item("average", _gaap("WeightedAverageNumberOfDilutedSharesOutstanding"),
                           unit="shares"),
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def extract_facts(payload: dict, symbol: str) -> pd.DataFrame:
    """The 10-K/10-Q facts for the concepts in ``ITEMS``, one row per reported value."""
    facts = payload.get("facts", {})
    rows = []
    for item, spec in ITEMS.items():
        for priority, name in enumerate(spec.concepts):
            taxonomy, concept = name.split(":")
            entries = facts.get(taxonomy, {}).get(concept, {}).get("units", {}).get(spec.unit, [])
            for e in entries:
                if e.get("form") not in FORMS or e.get("val") is None or not e.get("filed"):
                    continue
                rows.append((symbol, item, concept, priority, e.get("start"), e["end"],
                             float(e["val"]), e["filed"], e.get("accn")))
    df = pd.DataFrame(rows, columns=FACT_COLUMNS)
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    df["priority"] = df["priority"].astype("int64")
    return df


def ticker_to_cik(cache_dir: Path, user_agent: str, session: Any = None,
                  max_age_days: float = 7) -> dict[str, int]:
    """SEC's ticker→CIK map (cached for a week). Class suffixes use '-' (BRK-B)."""
    import requests

    path = Path(cache_dir) / "company_tickers.json"
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_days * 86400:
        session = session or requests.Session()
        resp = session.get(TICKERS_URL, headers={"User-Agent": user_agent}, timeout=30)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(resp.json()))
    return {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(path.read_text()).values()}


def download_company_facts(
    symbols: list[str],
    cache_dir: Path,
    user_agent: str,
    session: Any = None,
    pause: float = 0.15,
    refresh_after_days: float | None = None,
) -> DownloadSummary:
    """
    Fetch company facts for ``symbols`` and cache the extracted facts per symbol.

    SEC allows 10 requests a second with a descriptive User-Agent. Symbols
    without a CIK or with no facts are recorded in ``_unavailable.json``; a 403
    or 429 (rate limiting) stops the run so it can resume later. Cached files
    older than ``refresh_after_days`` are fetched again.
    """
    import requests

    session = session or requests.Session()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ciks = ticker_to_cik(cache_dir, user_agent, session)
    marker = cache_dir / "_unavailable.json"
    unavailable = set(json.loads(marker.read_text())) if marker.exists() else set()

    summary = DownloadSummary()
    todo = []
    for sym in dict.fromkeys(symbols):
        path = cache_dir / f"{sym}.parquet"
        fresh = path.exists() and (refresh_after_days is None
                                   or time.time() - path.stat().st_mtime <= refresh_after_days * 86400)
        if fresh or sym in unavailable:
            summary.already_cached += 1
        else:
            todo.append(sym)

    for i, sym in enumerate(todo):
        cik = ciks.get(sym.upper())
        if cik is None:
            unavailable.add(sym)
            summary.unavailable.append(sym)
            continue
        frames, statuses = [], []
        for c in (cik, *PREDECESSOR_CIKS.get(sym.upper(), ())):
            resp = session.get(FACTS_URL.format(cik=c), headers={"User-Agent": user_agent}, timeout=60)
            statuses.append(resp.status_code)
            if resp.status_code == 200:
                frames.append(extract_facts(resp.json(), sym))
            if resp.status_code in (403, 429):
                break
            time.sleep(pause)
        if statuses[-1] in (403, 429):
            summary.quota_hit = True
            summary.remaining = todo[i:]
            logger.warning("SEC refused the request (HTTP %d); stopping", statuses[-1])
            break
        facts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if any(code not in (200, 404) for code in statuses):
            logger.warning("%s: HTTP %s, will retry next run", sym, statuses)
            summary.remaining.append(sym)
        elif facts.empty:
            unavailable.add(sym)
            summary.unavailable.append(sym)
        else:
            facts.to_parquet(cache_dir / f"{sym}.parquet")
            summary.downloaded.append(sym)

    marker.write_text(json.dumps(sorted(unavailable)))
    return summary


def load_company_facts(cache_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(Path(cache_dir).glob("*.parquet"))]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FACT_COLUMNS)


# ---------------------------------------------------------------------------
# Point-in-time table
# ---------------------------------------------------------------------------

def select_reported(facts: pd.DataFrame, mode: str = "first") -> pd.DataFrame:
    """
    One value per (symbol, item, period): the earliest filing's (``mode="first"``)
    or the latest filing's (``"last"``, i.e. restated, for comparison only).
    Within one filing the highest-priority concept wins.
    """
    if mode not in ("first", "last"):
        raise ValueError(mode)
    ordered = facts.sort_values(["filed", "priority"], ascending=[mode == "first", True], kind="stable")
    return ordered.drop_duplicates(["symbol", "item", "start", "end"], keep="first")


def _kind(items: pd.Series) -> pd.Series:
    return items.map({k: v.kind for k, v in ITEMS.items()})


def _nearest(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str,
             by: list[str], tol_days: int) -> pd.DataFrame:
    return pd.merge_asof(left.sort_values(left_on), right.sort_values(right_on),
                         left_on=left_on, right_on=right_on, by=by, direction="nearest",
                         tolerance=pd.Timedelta(days=tol_days))


def ttm_flows(selected: pd.DataFrame) -> pd.DataFrame:
    """
    Trailing-twelve-month values for flow items.

    Annual periods are used as reported. A year-to-date (or single-quarter)
    period P adds the previous fiscal year and subtracts the same-length period
    a year earlier. The result is available when the last of its parts was
    filed. Per period end, the earliest-available version wins.
    """
    key = ["symbol", "item"]
    f = selected[_kind(selected["item"]) == "flow"].dropna(subset=["start"]).copy()
    f["dur"] = (f["end"] - f["start"]).dt.days
    f = f[f["dur"].between(80, 380)]
    annual = f[f["dur"] >= 350].copy()
    part = f[f["dur"] < 350].copy()
    if not part.empty:
        a = annual[key + ["end", "value", "filed"]].rename(
            columns={"end": "a_end", "value": "a_val", "filed": "a_filed"})
        part["fy_prev_end"] = part["start"] - pd.Timedelta(days=1)
        part = _nearest(part, a, "fy_prev_end", "a_end", key, 7)
        part["n_q"] = (part["dur"] / 91).round().astype("int64")
        p = part[key + ["n_q", "end", "value", "filed"]].rename(
            columns={"end": "p_end", "value": "p_val", "filed": "p_filed"})
        part["prev_end"] = part["end"] - pd.Timedelta(days=365)
        part = _nearest(part, p, "prev_end", "p_end", key + ["n_q"], 10)
        part["ttm"] = part["value"] + part["a_val"] - part["p_val"]
        part["available"] = part[["filed", "a_filed", "p_filed"]].max(axis=1, skipna=False)
    annual["ttm"] = annual["value"]
    annual["available"] = annual["filed"]
    both = pd.concat([annual, part], ignore_index=True).dropna(subset=["ttm", "available"])
    both = both.sort_values(key + ["end", "available", "dur"], ascending=[True, True, True, True, False])
    out = both.drop_duplicates(key + ["end"], keep="first")
    return pd.DataFrame({"symbol": out["symbol"], "item": out["item"], "period_end": out["end"],
                         "available_date": out["available"], "value": out["ttm"]})


def _instants(selected: pd.DataFrame) -> pd.DataFrame:
    s = selected[_kind(selected["item"]) == "instant"]
    return pd.DataFrame({"symbol": s["symbol"], "item": s["item"], "period_end": s["end"],
                         "available_date": s["filed"], "value": s["value"]})


def _averages(selected: pd.DataFrame) -> pd.DataFrame:
    """Share-count averages: per period end, the period closest to one quarter."""
    s = selected[_kind(selected["item"]) == "average"].dropna(subset=["start"]).copy()
    s["gap"] = ((s["end"] - s["start"]).dt.days - 91).abs()
    s = s.sort_values(["filed", "gap"]).drop_duplicates(["symbol", "item", "end"], keep="first")
    return pd.DataFrame({"symbol": s["symbol"], "item": s["item"], "period_end": s["end"],
                         "available_date": s["filed"], "value": s["value"]})


def build_pit_table(facts: pd.DataFrame, mode: str = "first") -> pd.DataFrame:
    """Long point-in-time table: ``symbol, item, period_end, available_date, value``."""
    if facts.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    sel = select_reported(facts, mode)
    table = pd.concat([ttm_flows(sel), _instants(sel), _averages(sel)], ignore_index=True)
    # A value cannot be known before its period ends; drop any tagging errors.
    table = table[table["available_date"] >= table["period_end"]]
    return table.sort_values(["symbol", "item", "period_end"]).reset_index(drop=True)[TABLE_COLUMNS]


def fundamentals_asof(
    table: pd.DataFrame,
    dates: pd.DatetimeIndex | list,
    symbols: list[str] | None = None,
    max_age_days: int = 400,
    items: list[str] | None = None,
    with_available: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    What was known at each date: per (date, symbol), each item's value for the
    latest period whose filing came strictly before ``date``.

    Values for periods ending more than ``max_age_days`` before the date are
    treated as missing. ``with_available`` adds ``<item>_available`` columns.
    """
    dates = pd.DatetimeIndex(dates).astype("datetime64[ns]")
    symbols = sorted(table["symbol"].unique()) if symbols is None else list(symbols)
    grid = (pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
            .to_frame(index=False).sort_values("date", kind="stable").reset_index(drop=True))
    out = grid.copy()
    wanted = items or sorted(table["item"].unique())
    for item in wanted:
        g = table[table["item"] == item].sort_values(["symbol", "available_date", "period_end"])
        if g.empty:
            out[item] = np.nan
            continue
        # A filing that only repeats an older period must not displace a newer one.
        seen = g.groupby("symbol")["period_end"].cummax().groupby(g["symbol"]).shift()
        g = g[seen.isna() | (g["period_end"] > seen)]
        right = g[["symbol", "available_date", "period_end", "value"]].sort_values(
            ["available_date", "period_end"])
        m = pd.merge_asof(grid, right, left_on="date", right_on="available_date", by="symbol",
                          allow_exact_matches=False)
        ok = (m["date"] - m["period_end"]).dt.days <= max_age_days
        out[item] = m["value"].where(ok).to_numpy()
        if item in with_available:
            out[f"{item}_available"] = m["available_date"].where(ok).to_numpy()
    return out.set_index(["date", "symbol"]).sort_index()
