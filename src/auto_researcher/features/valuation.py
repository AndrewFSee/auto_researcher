"""
Valuation and quality metrics from point-in-time fundamentals, plus a reverse DCF.

Inputs are the as-of panel from ``data.sec_fundamentals.fundamentals_asof``
(indexed by date and symbol) and prices that are split-adjusted but not
dividend-adjusted (yfinance ``Close`` with ``auto_adjust=False``).

Market cap needs care with splits: reported share counts are in the units of
their filing date, while split-adjusted prices are in today's units. Shares
are therefore multiplied by every split after the filing date, which turns
both into today's units without using information beyond the price series
itself (the product equals the true market cap on each date).

Reverse DCF: instead of guessing growth to get a fair value, solve for the
10-year free-cash-flow growth rate that justifies today's enterprise value,
given a discount rate and a terminal growth rate. A tool for reading a single
company, not a tested trading signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TAX_RATE = 0.21
SHARES_MAX_AGE_DAYS = 200


def split_factor_after(splits: pd.DataFrame, symbols: pd.Series, when: pd.Series) -> np.ndarray:
    """
    Product of split ratios strictly after ``when`` for each (symbol, when).

    ``splits`` is a date × symbol frame of split ratios (4.0 for 4-for-1; 0 or
    NaN on days without a split), as yfinance reports them.
    """
    out = np.ones(len(symbols))
    sym = symbols.to_numpy()
    when_v = pd.to_datetime(when).to_numpy()
    for s in np.unique(sym):
        if s not in splits.columns:
            continue
        col = splits[s]
        events = col[(col > 0) & col.notna() & (col != 1)]
        if events.empty:
            continue
        dates = events.index.to_numpy()
        # rev[i] = product of ratios from event i to the end; rev[n] = 1.
        rev = np.append(np.cumprod(events.to_numpy()[::-1])[::-1], 1.0)
        idx = np.flatnonzero(sym == s)
        valid = ~pd.isna(when_v[idx])
        pos = np.searchsorted(dates, when_v[idx][valid], side="right")
        out[idx[valid]] = rev[pos]
    return out


def _choose_shares(fund: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cover-page shares when recent, else diluted weighted-average shares."""
    dates = pd.Series(fund.index.get_level_values("date"), index=fund.index)
    cover_age = (dates - pd.to_datetime(fund["shares_cover_available"])).dt.days
    cover_ok = fund["shares_cover"].notna() & (cover_age <= SHARES_MAX_AGE_DAYS)
    shares = fund["shares_cover"].where(cover_ok, fund["shares_diluted"])
    filed = fund["shares_cover_available"].where(cover_ok, fund["shares_diluted_available"])
    return shares, pd.to_datetime(filed)


def adjusted_shares(fund: pd.DataFrame, splits: pd.DataFrame) -> pd.Series:
    """Share count in the units of the split-adjusted price series."""
    shares, filed = _choose_shares(fund)
    syms = pd.Series(fund.index.get_level_values("symbol"), index=fund.index)
    factor = split_factor_after(splits, syms, filed)
    return shares * factor


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def valuation_metrics(
    fund: pd.DataFrame,
    close: pd.DataFrame,
    splits: pd.DataFrame,
    fund_prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Per (date, symbol): market cap, enterprise value, free cash flow and the
    standard value/quality ratios.

    ``fund`` must include ``shares_cover``/``shares_diluted`` with their
    ``_available`` columns. ``fund_prior`` (same index, values as known one
    year earlier) enables asset growth and net share issuance.
    """
    dates = fund.index.get_level_values("date")
    syms = fund.index.get_level_values("symbol")
    px = close.reindex(index=close.index.union(dates.unique())).sort_index().ffill()
    price = pd.Series(px.stack().reindex(list(zip(dates, syms))).to_numpy(), index=fund.index)

    shares = adjusted_shares(fund, splits)
    mcap = price * shares
    split_debt = _col(fund, "debt_noncurrent").fillna(0) + _col(fund, "debt_current").fillna(0)
    has_split = _col(fund, "debt_noncurrent").notna() | _col(fund, "debt_current").notna()
    debt_total = _col(fund, "debt_total")
    debt = split_debt.where(has_split, debt_total).fillna(0)
    debt_known = has_split | debt_total.notna()
    cash = _col(fund, "cash").fillna(0) + _col(fund, "short_term_investments").fillna(0)
    ev = mcap + debt - cash
    fcf = fund["operating_cash_flow"] - fund["capex"].fillna(0)
    fcff = fcf + _col(fund, "interest_expense").fillna(0) * (1 - TAX_RATE)
    gross = fund["gross_profit"].fillna(fund["revenue"] - _col(fund, "cost_of_revenue"))
    assets = fund["total_assets"]
    pos_ev = ev.where(ev > 0)

    out = pd.DataFrame({
        "price": price,
        "shares": shares,
        "market_cap": mcap,
        "enterprise_value": ev,
        "debt": debt,
        "debt_known": debt_known,
        "noncurrent_liabilities": _col(fund, "total_liabilities") - _col(fund, "current_liabilities"),
        "fcf": fcf,
        "fcff": fcff,
        "fcf_yield": fcf / mcap,
        "fcff_ev": fcff / pos_ev,
        "ebit_ev": fund["operating_income"] / pos_ev,
        "earnings_yield": fund["net_income"] / mcap,
        "book_to_market": _col(fund, "equity") / mcap,
        "gross_profitability": gross / assets,
        "accruals": (fund["net_income"] - fund["operating_cash_flow"]) / assets,
        "shareholder_yield": (_col(fund, "dividends_paid").fillna(0)
                              + _col(fund, "buybacks").fillna(0)) / mcap,
        "sbc_to_fcf": _col(fund, "stock_comp") / fcf.where(fcf > 0),
    }, index=fund.index)
    if fund_prior is not None:
        prior = fund_prior.reindex(fund.index)
        out["asset_growth"] = assets / prior["total_assets"] - 1
        prior_shares = adjusted_shares(prior, splits)
        out["net_issuance"] = np.log(shares / prior_shares)
        out["revenue_growth"] = fund["revenue"] / prior["revenue"].where(prior["revenue"] > 0) - 1
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Reverse DCF
# ---------------------------------------------------------------------------

def dcf_value(fcf0: float, growth: float, discount: float, years: int = 10,
              terminal_growth: float = 0.025) -> float:
    """
    Present value of ``fcf0`` growing at ``growth`` for ``years`` years, then
    at ``terminal_growth`` forever (Gordon terminal value), discounted at
    ``discount``.
    """
    if discount <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    t = np.arange(1, years + 1)
    flows = fcf0 * (1 + growth) ** t
    pv = float(np.sum(flows / (1 + discount) ** t))
    terminal = float(flows[-1]) * (1 + terminal_growth) / (discount - terminal_growth)
    return pv + terminal / (1 + discount) ** years


def implied_growth(value: float, fcf0: float, discount: float, years: int = 10,
                   terminal_growth: float = 0.025, bounds: tuple[float, float] = (-0.5, 1.0)) -> float:
    """
    The growth rate at which ``dcf_value`` equals ``value`` (e.g. enterprise
    value). NaN when free cash flow or value is not positive, or when the
    answer lies outside ``bounds``.
    """
    from scipy.optimize import brentq

    if not (np.isfinite(value) and np.isfinite(fcf0) and value > 0 and fcf0 > 0):
        return float("nan")

    def gap(g: float) -> float:
        return dcf_value(fcf0, g, discount, years, terminal_growth) - value

    lo, hi = bounds
    if gap(lo) > 0 or gap(hi) < 0:
        return float("nan")
    return float(brentq(gap, lo, hi, xtol=1e-6))
