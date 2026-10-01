"""
Published cross-sectional equity factors computed from daily prices and volume.

Each factor dated ``t`` uses data up to the close of ``t`` only (checked by
``tests/test_feature_causality.py``). Factors are returned as per-date
cross-sectional ranks in ``[-0.5, 0.5]``, so they are robust to outliers and
directly comparable across dates.

========================  =====================================================
Factor                    Definition (reference)
========================  =====================================================
``mom_12_1``              Return from t-252 to t-21, skipping the last month
                          (Jegadeesh & Titman 1993)
``mom_6_1``               Return from t-126 to t-21
``ret_21``                Last month's return; short-term reversal (Jegadeesh 1990)
``high_52w``              Price / 52-week high (George & Hwang 2004)
``beta_252``              One-year beta to the benchmark (Frazzini & Pedersen 2014)
``vol_63``                Three-month daily return volatility (Ang et al. 2006)
``max_ret_21``            Largest daily return in the last month (Bali, Cakici &
                          Whitelaw 2011)
``seasonal``              Mean return over the same 21-day window of the calendar
                          one to five years ago (Heston & Sadka 2008)
``abn_volume``            5-day / 63-day average volume
``dollar_volume_trend``   log(21-day / 252-day average dollar volume)
``sector_mom_6_1``        Sector average of ``mom_6_1`` (Moskowitz & Grinblatt 1999)
``mom_12_1_in_sector``    ``mom_12_1`` minus its sector average
``ret_21_in_sector``      ``ret_21`` minus its sector average (within-industry
                          reversal; Da, Liu & Schaumburg 2014)
========================  =====================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTH = 21

PRICE_FACTORS = (
    "mom_12_1",
    "mom_6_1",
    "ret_21",
    "high_52w",
    "beta_252",
    "vol_63",
    "max_ret_21",
    "seasonal",
)
VOLUME_FACTORS = ("abn_volume", "dollar_volume_trend")
SECTOR_FACTORS = ("sector_mom_6_1", "mom_12_1_in_sector", "ret_21_in_sector")


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-date (row-wise) percentile rank centered on zero; NaNs stay NaN."""
    return frame.rank(axis=1, pct=True) - 0.5


def _sector_mean(frame: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """For each date and ticker, the mean of ``frame`` over the ticker's sector."""
    groups = sectors.reindex(frame.columns)
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for _, members in groups.dropna().groupby(groups.dropna()):
        cols = list(members.index)
        out[cols] = np.repeat(frame[cols].mean(axis=1).to_numpy()[:, None], len(cols), axis=1)
    return out


def _seasonal(close: pd.DataFrame, years: int = 5, min_years: int = 2) -> pd.DataFrame:
    """
    Average return over the window (t - 252k, t - 252k + 21] for k = 1..years:
    "how this stock did over the coming month in previous years". Known at t
    because the latest window ends 231 trading days before t.
    """
    parts = []
    for k in range(1, years + 1):
        start = close.shift(TRADING_DAYS * k)
        end = close.shift(TRADING_DAYS * k - MONTH)
        parts.append((end / start - 1.0).to_numpy())
    stacked = np.stack(parts)
    count = np.sum(~np.isnan(stacked), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nansum(stacked, axis=0) / np.where(count > 0, count, np.nan)
    mean[count < min_years] = np.nan
    return pd.DataFrame(mean, index=close.index, columns=close.columns)


def compute_alpha_factors(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    benchmark: str = "SPY",
    sectors: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute the factor panel.

    Args:
        close: Adjusted closes (date x ticker), including ``benchmark``.
        volume: Share volume with the same shape (optional; volume factors are
            skipped without it).
        benchmark: Column used for beta; it is excluded from the output.
        sectors: Ticker -> sector label (optional; sector factors are skipped
            without it). Tickers without a sector get NaN sector factors.

    Returns:
        Long DataFrame indexed by ``(date, ticker)`` with one column per factor,
        each a per-date rank in [-0.5, 0.5].
    """
    if benchmark not in close.columns:
        raise ValueError(f"benchmark {benchmark!r} is missing from the price panel")
    close = close.sort_index()
    stocks = close.drop(columns=[benchmark])
    rets = stocks.pct_change(fill_method=None)
    mkt = close[benchmark].pct_change(fill_method=None)

    raw: dict[str, pd.DataFrame] = {
        "mom_12_1": stocks.shift(MONTH) / stocks.shift(TRADING_DAYS) - 1.0,
        "mom_6_1": stocks.shift(MONTH) / stocks.shift(126) - 1.0,
        "ret_21": stocks / stocks.shift(MONTH) - 1.0,
        "high_52w": stocks / stocks.rolling(TRADING_DAYS, min_periods=200).max(),
        "vol_63": rets.rolling(63, min_periods=50).std(),
        "max_ret_21": rets.rolling(MONTH, min_periods=MONTH).max(),
        "seasonal": _seasonal(stocks),
    }
    cov = rets.rolling(TRADING_DAYS, min_periods=200).cov(mkt)
    var = mkt.rolling(TRADING_DAYS, min_periods=200).var()
    raw["beta_252"] = cov.div(var.replace(0.0, np.nan), axis=0)

    if volume is not None:
        vol = volume.reindex(index=stocks.index, columns=stocks.columns).astype(float)
        vol = vol.where(vol > 0)
        raw["abn_volume"] = vol.rolling(5, min_periods=5).mean() / vol.rolling(63, min_periods=50).mean()
        dollar = vol * stocks
        raw["dollar_volume_trend"] = np.log(
            dollar.rolling(MONTH, min_periods=MONTH).mean()
            / dollar.rolling(TRADING_DAYS, min_periods=200).mean()
        )

    if sectors is not None:
        sectors = pd.Series(sectors).dropna()
        raw["sector_mom_6_1"] = _sector_mean(raw["mom_6_1"], sectors)
        raw["mom_12_1_in_sector"] = raw["mom_12_1"] - _sector_mean(raw["mom_12_1"], sectors)
        raw["ret_21_in_sector"] = raw["ret_21"] - _sector_mean(raw["ret_21"], sectors)

    ranked = {name: cross_sectional_rank(frame) for name, frame in raw.items()}
    panel = pd.concat({name: frame.stack(future_stack=True) for name, frame in ranked.items()}, axis=1)
    panel.index = panel.index.set_names(["date", "ticker"])
    return panel
