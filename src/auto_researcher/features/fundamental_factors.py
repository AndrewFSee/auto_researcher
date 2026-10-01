"""
Point-in-time fundamental factor panels.

Builds, for any set of dates, the six published factors pre-registered in
``docs/results/fundamental_factors_protocol.md`` plus the inputs of the growth
forecast. Everything is computed from ``data.sec_fundamentals.fundamentals_asof``
(values known strictly before each date) and prices up to that date.

"One year earlier" means the value known 365 days before the date (not the
value of the period one year before, which may only have been filed later).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from auto_researcher.data.sec_fundamentals import ITEMS, fundamentals_asof
from auto_researcher.features.valuation import TAX_RATE, valuation_metrics

FACTOR_SIGNS: dict[str, int] = {
    "gross_profitability": 1,
    "accruals": -1,
    "asset_growth": -1,
    "net_issuance": -1,
    "fcf_yield": 1,
    "ebit_ev": 1,
}
SHARES = ("shares_cover", "shares_diluted")
GROWTH_FEATURES = [
    "rev_growth_1y", "rev_cagr_3y", "fcff_margin", "fcff_margin_chg_3y", "op_margin",
    "gross_margin", "capex_intensity", "capex_to_dep", "sbc_intensity", "log_revenue",
    "asset_turnover", "accruals", "asset_growth", "sector", "sector_median_growth",
]


def asof_shifted(table: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str],
                 days_back: int = 0) -> pd.DataFrame:
    """
    ``fundamentals_asof`` evaluated ``days_back`` days before each date (negative:
    after), relabelled to the original dates. Uses fixed day counts so distinct
    dates stay distinct.
    """
    dates = pd.DatetimeIndex(dates)
    src = dates - pd.Timedelta(days=days_back)
    f = fundamentals_asof(table, src, symbols, with_available=SHARES)
    relabel = dict(zip(src, dates))
    f = f.reset_index()
    f["date"] = f["date"].map(relabel)
    f = f.set_index(["date", "symbol"]).sort_index()
    # Every item as a column, even if no company reports it.
    return f.reindex(columns=list(dict.fromkeys([*ITEMS, *f.columns])))


def _fcff(f: pd.DataFrame) -> pd.Series:
    return (f["operating_cash_flow"] - f["capex"].fillna(0)
            + f["interest_expense"].fillna(0) * (1 - TAX_RATE))


def factor_panel(table: pd.DataFrame, close: pd.DataFrame, splits: pd.DataFrame,
                 dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Six factors plus FCFF, FCFF/EV, EV and market cap per (date, symbol)."""
    fund = asof_shifted(table, dates, symbols)
    prior = asof_shifted(table, dates, symbols, days_back=365)
    m = valuation_metrics(fund, close, splits, fund_prior=prior)
    cols = [*FACTOR_SIGNS, "fcff", "fcff_ev", "enterprise_value", "market_cap"]
    return m[cols]


def composite_score(panel: pd.DataFrame, signs: dict[str, int] = FACTOR_SIGNS) -> pd.Series:
    """Mean of signed cross-sectional percentile ranks; NaN unless every factor exists."""
    parts = []
    for col, sign in signs.items():
        pct = panel[col].groupby(level="date").rank(pct=True)
        parts.append(pct if sign > 0 else 1 - pct)
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1).where(stacked.notna().all(axis=1))


def _cagr(now: pd.Series, before: pd.Series, years: float) -> pd.Series:
    ok = (now > 0) & (before > 0)
    return ((now / before.where(ok)) ** (1 / years) - 1).where(ok)


def growth_features(table: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str],
                    sectors: pd.Series) -> pd.DataFrame:
    """Inputs of the growth forecast, known at each date (``GROWTH_FEATURES``)."""
    now = asof_shifted(table, dates, symbols)
    y1 = asof_shifted(table, dates, symbols, days_back=365)
    y3 = asof_shifted(table, dates, symbols, days_back=3 * 365 + 1)
    rev = now["revenue"].where(now["revenue"] > 0)
    rev3 = y3["revenue"].where(y3["revenue"] > 0)
    gross = now["gross_profit"].fillna(now["revenue"] - now["cost_of_revenue"])
    margin = _fcff(now) / rev
    sector_names = sorted(sectors.dropna().unique())
    sym = now.index.get_level_values("symbol")
    out = pd.DataFrame({
        "rev_growth_1y": rev / y1["revenue"].where(y1["revenue"] > 0) - 1,
        "rev_cagr_3y": _cagr(rev, rev3, 3),
        "fcff_margin": margin,
        "fcff_margin_chg_3y": margin - _fcff(y3) / rev3,
        "op_margin": now["operating_income"] / rev,
        "gross_margin": gross / rev,
        "capex_intensity": now["capex"] / rev,
        "capex_to_dep": now["capex"] / now["depreciation"].where(now["depreciation"] > 0),
        "sbc_intensity": now["stock_comp"] / rev,
        "log_revenue": np.log(rev),
        "asset_turnover": rev / now["total_assets"],
        "accruals": (now["net_income"] - now["operating_cash_flow"]) / now["total_assets"],
        "asset_growth": now["total_assets"] / y1["total_assets"] - 1,
        "sector": pd.Categorical(sectors.reindex(sym).to_numpy(), categories=sector_names),
        "revenue": rev,
    }, index=now.index).replace([np.inf, -np.inf], np.nan)
    out["sector_median_growth"] = out.groupby(
        [out.index.get_level_values("date"), "sector"], observed=True)["rev_cagr_3y"].transform("median")
    return out


def growth_labels(table: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str],
                  data_end: pd.Timestamp, years: int = 3) -> pd.DataFrame:
    """
    Realized ``years``-year revenue CAGR and FCF-to-firm margin, from values
    known ``years`` years after each date. ``label_date`` is when the label
    became known; labels after ``data_end`` are missing.
    """
    days = years * 365 + 1
    now = asof_shifted(table, dates, symbols)
    fut = asof_shifted(table, dates, symbols, days_back=-days)
    label_date = pd.Series(now.index.get_level_values("date") + pd.Timedelta(days=days), index=now.index)
    known = label_date <= data_end
    fut_rev = fut["revenue"].where(fut["revenue"] > 0)
    return pd.DataFrame({
        "y_growth": _cagr(fut_rev, now["revenue"], years).clip(-0.5, 1.0).where(known),
        "y_margin": (_fcff(fut) / fut_rev).clip(-0.5, 0.8).where(known),
        "label_date": label_date,
    })


def intrinsic_ev(revenue: np.ndarray, margin0: np.ndarray, growth: np.ndarray, margin: np.ndarray,
                 discount: float = 0.09, terminal_growth: float = 0.025, years: int = 10,
                 high_growth_years: int = 3) -> np.ndarray:
    """
    DCF of a revenue and FCF-to-firm margin path: revenue grows at ``growth``
    for ``high_growth_years``, then the rate fades linearly to
    ``terminal_growth`` by ``years``; the margin moves linearly from
    ``margin0`` to ``margin`` over the high-growth years, then stays.
    """
    if discount <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    rev = np.asarray(revenue, dtype=float).copy()
    m0 = np.asarray(margin0, dtype=float)
    g = np.asarray(growth, dtype=float)
    m = np.asarray(margin, dtype=float)
    pv = np.zeros_like(rev)
    fcf = np.zeros_like(rev)
    for k in range(1, years + 1):
        if k <= high_growth_years:
            gk = g
        else:
            gk = g + (terminal_growth - g) * (k - high_growth_years) / (years - high_growth_years)
        rev = rev * (1 + gk)
        mk = m0 + (m - m0) * min(k, high_growth_years) / high_growth_years
        fcf = rev * mk
        pv = pv + fcf / (1 + discount) ** k
    terminal = fcf * (1 + terminal_growth) / (discount - terminal_growth)
    return pv + terminal / (1 + discount) ** years
