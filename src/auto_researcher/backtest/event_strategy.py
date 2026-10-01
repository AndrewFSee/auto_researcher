"""
Event-driven long/short backtests (e.g. post-earnings-announcement drift).

Each event has a ticker, the trading day its information became usable, and a
scalar signal. Events are classified against the distribution of *earlier*
events only (a trailing percentile, so no future data sets the thresholds),
then held for a fixed number of trading days:

    usable at close of day u  ->  enter at close of u + entry_lag
                              ->  earn returns on the next ``hold_days`` days

Each leg is equally weighted across its active positions every day. Costs are
charged on the daily change in target weights (entries, exits and the
re-weighting they cause); intra-period drift is not traded, so costs are a
slight underestimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from auto_researcher.backtest.metrics import compute_max_drawdown, compute_sharpe_ratio
from auto_researcher.validation.deflated_sharpe import deflated_sharpe_ratio

TRADING_DAYS = 252


@dataclass
class EventStrategyConfig:
    """
    Attributes:
        hold_days: Trading days each position is held.
        entry_lag: Trading days between usability and the entry close.
        long_pct: Events at or above this trailing percentile go long.
        short_pct: Events at or below this trailing percentile go short.
        lookback_days: Calendar window of earlier events used for percentiles.
        min_history: Minimum number of earlier events needed to classify one.
        cost_bps: One-way cost per unit of notional traded.
    """

    hold_days: int = 40
    entry_lag: int = 1
    long_pct: float = 0.8
    short_pct: float = 0.2
    lookback_days: int = 365
    min_history: int = 200
    cost_bps: float = 10.0


def classify_events(events: pd.DataFrame, cfg: EventStrategyConfig) -> pd.DataFrame:
    """
    Add ``pct`` (percentile of ``value`` among events usable strictly earlier
    and within ``lookback_days``) and ``side`` (+1 long, -1 short, 0 none).

    Args:
        events: Columns ``ticker, usable_date, value``.
    """
    ev = events.dropna(subset=["value", "usable_date"]).sort_values("usable_date").reset_index(drop=True)
    dates = ev["usable_date"].to_numpy()
    values = ev["value"].to_numpy(dtype=float)
    window = np.timedelta64(cfg.lookback_days, "D")
    lo = np.searchsorted(dates, dates - window, side="left")
    hi = np.searchsorted(dates, dates, side="left")  # strictly earlier days only
    pct = np.full(len(ev), np.nan)
    for i in range(len(ev)):
        if hi[i] - lo[i] >= cfg.min_history:
            past = values[lo[i]:hi[i]]
            pct[i] = (np.sum(past < values[i]) + 0.5 * np.sum(past == values[i])) / len(past)
    ev["pct"] = pct
    ev["side"] = np.where(pct >= cfg.long_pct, 1, np.where(pct <= cfg.short_pct, -1, 0))
    ev.loc[np.isnan(pct), "side"] = 0
    return ev


def simulate_event_strategy(
    events: pd.DataFrame,
    close: pd.DataFrame,
    cfg: EventStrategyConfig,
) -> pd.DataFrame:
    """
    Daily returns of the long leg, short leg and long-short portfolio.

    Args:
        events: Output of ``classify_events`` (``ticker, usable_date, side``).
        close: Adjusted closes (date x ticker).

    Returns:
        DataFrame indexed by date with ``long``, ``short`` (gross leg returns),
        ``long_net``, ``short_cost``, ``long_short_net`` and position counts
        ``n_long``, ``n_short``. Days without positions in a leg return 0 for it.
    """
    cal = close.index
    rets = close.pct_change(fill_method=None).to_numpy()
    col = {t: j for j, t in enumerate(close.columns)}
    counts = {1: np.zeros(rets.shape), -1: np.zeros(rets.shape)}
    for e in events[events["side"] != 0].itertuples(index=False):
        j = col.get(e.ticker)
        if j is None:
            continue
        u = cal.searchsorted(pd.Timestamp(e.usable_date))
        entry = u + cfg.entry_lag
        if entry >= len(cal) - 1:
            continue
        counts[e.side][entry + 1: entry + 1 + cfg.hold_days, j] += 1

    out = {}
    for side, name in ((1, "long"), (-1, "short")):
        c = counts[side]
        held = c.sum(axis=1)
        w = np.divide(c, held[:, None], out=np.zeros_like(c), where=held[:, None] > 0)
        r = np.nan_to_num(rets, nan=0.0)
        gross = (w * r).sum(axis=1)
        traded = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, w.shape[1])))).sum(axis=1)
        out[name] = gross
        out[f"{name}_cost"] = traded * cfg.cost_bps / 1e4
        out[f"n_{name}"] = held
    df = pd.DataFrame(out, index=cal)
    df["long_net"] = df["long"] - df["long_cost"]
    df["long_short_net"] = df["long"] - df["short"] - df["long_cost"] - df["short_cost"]
    return df


def return_stats(daily: pd.Series, n_trials: int = 1) -> dict[str, float]:
    """Annualized mean, volatility, Sharpe, t-stat, max drawdown and deflated P(SR > 0)."""
    d = pd.Series(daily, dtype=float).dropna()
    n = len(d)
    if n < 20 or d.std() == 0:
        return {k: float("nan") for k in ("ann_return", "ann_vol", "sharpe", "t_stat", "max_drawdown",
                                           "deflated_prob")} | {"n_days": float(n)}
    sharpe = compute_sharpe_ratio(d, periods_per_year=TRADING_DAYS)
    return {
        "ann_return": float(d.mean() * TRADING_DAYS),
        "ann_vol": float(d.std() * math.sqrt(TRADING_DAYS)),
        "sharpe": sharpe,
        "t_stat": float(d.mean() / d.std() * math.sqrt(n)),
        "max_drawdown": compute_max_drawdown(d),
        "deflated_prob": deflated_sharpe_ratio(
            sharpe, n_obs=n, n_trials=max(n_trials, 1), periods_per_year=TRADING_DAYS,
            skew=float(d.skew()), kurt=float(d.kurt() + 3.0),
        ),
        "n_days": float(n),
    }
