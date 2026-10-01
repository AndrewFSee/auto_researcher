"""
Leakage-safe walk-forward evaluation for cross-sectional ranking models.

This is the single harness every model and baseline in the project should be
scored with, so results are comparable and the timing rules live in one place.

Timing convention (all offsets in trading days)::

    features known at close of t
        -> positions entered at close of t + execution_lag
        -> label = return from t + lag to t + lag + horizon

At each rebalance date ``T`` the model is refit on rows whose labels were
fully realized by the close of ``T`` (see ``validation.splits``), then used to
score every name with features on ``T``. The harness reports:

* **Signal quality**: per-date Spearman IC against *raw* forward returns,
  with a Newey-West t-stat whose lag matches the label overlap
  (``ceil(horizon / rebalance_every) - 1``), plus mean forward returns by
  prediction quantile, in return units.
* **Tradeable performance**: a top-k equal-weight portfolio simulated at daily
  resolution (buy-and-hold drift inside each holding period, turnover-based
  costs at each rebalance), next to the benchmark and an equal-weight
  portfolio of the same investable cross-section. Comparing with the
  equal-weight universe isolates stock-selection skill from universe effects
  such as survivorship.
* **A null distribution**: Sharpe ratios of random top-k selections from the
  same cross-sections, so "beats random" is a percentile rather than a
  comparison against one averaged path.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from scipy import stats

from auto_researcher.backtest.metrics import (
    compute_ic_stats,
    compute_max_drawdown,
    compute_sharpe_ratio,
)
from auto_researcher.validation.deflated_sharpe import deflated_sharpe_ratio
from auto_researcher.validation.splits import (
    label_overlap_days,
    purged_walk_forward_splits,
)

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


class RankingModel(Protocol):
    """Anything with scikit-learn style ``fit`` / ``predict``."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> object: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class WalkForwardConfig:
    """
    Settings for ``run_walk_forward``.

    Attributes:
        horizon: Label horizon in trading days.
        rebalance_every: Trading days between rebalances; also the holding period
            of the simulated portfolio, so portfolio returns never overlap.
        execution_lag: Trading days between the signal close and the entry close.
            ``1`` (trade the next close) is the conservative default.
        embargo: Extra trading days purged between training labels and the test date.
        train_window: Rolling training window in trading days (``None`` = expanding).
        min_train_dates: Minimum number of trainable dates before the first test date.
        top_k: Number of names held by the simulated long-only portfolio.
        n_quantiles: Number of prediction buckets for the quantile-return table.
        cost_bps: One-way transaction cost in basis points per unit of notional traded.
        target: Training target built from the forward return. ``"rank"`` (per-date
            percentile rank, centered) is robust to outliers and market moves;
            ``"vol_norm"`` divides by trailing 21-day volatility scaled to the horizon;
            ``"group_rank"`` ranks the return in excess of its group (e.g. sector)
            average, so the model learns stock selection rather than group timing
            (requires ``groups`` in ``run_walk_forward``).
        purge: ``False`` reproduces the legacy leaky split (train on every date
            before the test date). Only for measuring the leak.
        min_names: Minimum cross-section size required to evaluate a date.
        dropna_features: Drop rows with any missing feature before training/scoring.
        start: First allowed test date.
        end: Last allowed test date.
        n_trials: Number of model/configuration variants tried before this one;
            used to deflate the reported Sharpe ratio.
        n_random_paths: Size of the random-selection null distribution (0 disables).
        random_seed: Seed for the null distribution.
    """

    horizon: int = 21
    rebalance_every: int = 21
    execution_lag: int = 1
    embargo: int = 0
    train_window: int | None = 504
    min_train_dates: int = 252
    top_k: int = 10
    n_quantiles: int = 5
    cost_bps: float = 10.0
    target: Literal["rank", "demeaned", "raw", "vol_norm", "group_rank"] = "rank"
    purge: bool = True
    min_names: int = 10
    dropna_features: bool = True
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    n_trials: int = 1
    n_random_paths: int = 500
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.rebalance_every < 1:
            raise ValueError("horizon and rebalance_every must be >= 1")
        if self.execution_lag < 0 or self.embargo < 0:
            raise ValueError("execution_lag and embargo must be >= 0")
        if self.top_k < 1 or self.n_quantiles < 2:
            raise ValueError("top_k must be >= 1 and n_quantiles >= 2")


# =============================================================================
# Labels
# =============================================================================


def forward_returns(
    prices: pd.DataFrame,
    horizon: int,
    execution_lag: int = 0,
) -> pd.DataFrame:
    """
    Simple return from the close of ``t + lag`` to the close of ``t + lag + horizon``.

    The result is indexed by the decision date ``t``; the last
    ``lag + horizon`` rows are NaN because the exit price is not yet known.
    """
    entry = prices.shift(-execution_lag)
    exit_ = prices.shift(-(execution_lag + horizon))
    return exit_ / entry - 1.0


def make_training_target(
    fwd: pd.Series,
    mode: Literal["rank", "demeaned", "raw", "vol_norm", "group_rank"] = "rank",
    trailing_vol: pd.Series | None = None,
    groups: pd.Series | None = None,
) -> pd.Series:
    """
    Turn raw forward returns (indexed by date, ticker) into a training target.

    ``trailing_vol`` (same index, volatility known at t and scaled to the label
    horizon) is required for ``mode="vol_norm"``; ``groups`` (ticker -> group
    label) for ``mode="group_rank"``.
    """
    if mode == "group_rank":
        if groups is None:
            raise ValueError("group_rank target requires groups")
        labels = pd.Series(fwd.index.get_level_values("ticker"), index=fwd.index).map(groups)
        keys = [fwd.index.get_level_values("date"), labels.fillna("__none__").to_numpy()]
        excess = fwd - fwd.groupby(keys).transform("mean")
        return excess.groupby(level="date").rank(pct=True) - 0.5
    by_date = fwd.groupby(level="date")
    if mode == "rank":
        return by_date.rank(pct=True) - 0.5
    if mode == "demeaned":
        return fwd - by_date.transform("mean")
    if mode == "raw":
        return fwd
    if mode == "vol_norm":
        if trailing_vol is None:
            raise ValueError("vol_norm target requires trailing_vol")
        return fwd / trailing_vol.replace(0.0, np.nan)
    raise ValueError(f"unknown target mode: {mode!r}")


# =============================================================================
# Portfolio simulation
# =============================================================================


@dataclass
class _Window:
    """One holding period: rebalance date, entry position, and exit position."""

    rebalance_date: pd.Timestamp
    entry_pos: int
    exit_pos: int


def _growth_matrix(prices: pd.DataFrame, window: _Window, names: list[str]) -> pd.DataFrame:
    """
    Gross growth of each name relative to the entry close, for each day in the
    window. Rows are ``entry_pos + 1 .. exit_pos``. Missing prices inside the
    window are forward-filled (a halted name is held at its last price).
    """
    block = prices.iloc[window.entry_pos : window.exit_pos + 1][names].ffill()
    entry = block.iloc[0]
    return block.iloc[1:] / entry


def _simulate_window(
    growth: pd.DataFrame,
    weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Buy-and-hold a weight vector through one window.

    Returns the daily portfolio returns and the drifted weights at the window's end.
    """
    w = weights.reindex(growth.columns).fillna(0.0)
    value = growth.to_numpy() @ w.to_numpy()
    path = np.concatenate([[w.sum()], value])
    daily = pd.Series(path[1:] / path[:-1] - 1.0, index=growth.index)
    end_value = float(value[-1]) if len(value) else 0.0
    if end_value > 0:
        drifted = growth.iloc[-1].mul(w) / end_value
    else:
        drifted = w * 0.0
    return daily, drifted[drifted.abs() > 0]


def simulate_portfolio(
    prices: pd.DataFrame,
    windows: list[_Window],
    target_weights: dict[pd.Timestamp, pd.Series],
    cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Simulate a periodically rebalanced long-only portfolio at daily resolution.

    Args:
        prices: Wide price panel (date x ticker).
        windows: Contiguous holding windows in chronological order.
        target_weights: Weights to hold from each window's entry close.
        cost_bps: One-way cost per unit of notional traded.

    Returns:
        ``(gross_daily, net_daily, turnover)``: daily returns before and after
        costs, and one-sided turnover at each rebalance date.
    """
    gross_parts: list[pd.Series] = []
    net_parts: list[pd.Series] = []
    turnover: dict[pd.Timestamp, float] = {}
    held = pd.Series(dtype=float)

    for window in windows:
        weights = target_weights.get(window.rebalance_date)
        if weights is None or weights.empty:
            continue
        names = [t for t in weights.index if pd.notna(prices.iloc[window.entry_pos].get(t))]
        if not names:
            continue
        weights = weights.reindex(names)
        weights = weights / weights.sum()

        traded = weights.sub(held, fill_value=0.0).abs().sum()
        turnover[window.rebalance_date] = 0.5 * float(traded)
        cost = float(traded) * cost_bps / 1e4

        growth = _growth_matrix(prices, window, names)
        if growth.empty:
            continue
        daily, held = _simulate_window(growth, weights)

        net = daily.copy()
        net.iloc[0] = (1.0 + net.iloc[0]) * (1.0 - cost) - 1.0
        gross_parts.append(daily)
        net_parts.append(net)

    if not gross_parts:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    gross = pd.concat(gross_parts)
    net = pd.concat(net_parts)
    return gross, net, pd.Series(turnover, name="turnover").sort_index()


def random_selection_null(
    prices: pd.DataFrame,
    windows: list[_Window],
    universes: dict[pd.Timestamp, list[str]],
    k: int,
    n_paths: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """
    Annualized Sharpe ratios of ``n_paths`` strategies that pick ``k`` names
    uniformly at random from each date's investable universe (gross of costs).

    This is the right null for "does the ranking beat chance?": compare the
    model's Sharpe with this distribution's percentiles. Averaging many random
    paths into one return stream (as the legacy validation script did) builds
    a diversified equal-weight portfolio instead, whose Sharpe says nothing
    about selection skill.
    """
    rng = np.random.default_rng(seed)
    path_returns: list[np.ndarray] = []

    for window in windows:
        universe = [
            t for t in universes.get(window.rebalance_date, [])
            if pd.notna(prices.iloc[window.entry_pos].get(t))
        ]
        if len(universe) < k:
            continue
        growth = _growth_matrix(prices, window, universe).to_numpy()
        if growth.size == 0:
            continue
        picks = np.argsort(rng.random((n_paths, len(universe))), axis=1)[:, :k]
        # value[d, p] = mean growth of path p's k names on day d
        value = growth[:, picks].mean(axis=2)
        value = np.vstack([np.ones((1, n_paths)), value])
        path_returns.append(value[1:] / value[:-1] - 1.0)

    if not path_returns:
        return np.array([])
    daily = np.vstack(path_returns)
    mean = daily.mean(axis=0)
    std = daily.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, mean / std * math.sqrt(TRADING_DAYS_PER_YEAR), 0.0)
    return sharpe


# =============================================================================
# Summary statistics
# =============================================================================


def overlap_lag(horizon: int, step: int) -> int:
    """Newey-West lag implied by labels of ``horizon`` days sampled every ``step`` days."""
    return max(math.ceil(horizon / step) - 1, 0)


def _return_stats(daily: pd.Series) -> dict[str, float]:
    daily = daily.dropna()
    n = len(daily)
    if n < 2:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"),
                "max_drawdown": float("nan"), "n_days": float(n)}
    growth = float((1.0 + daily).prod())
    years = n / TRADING_DAYS_PER_YEAR
    return {
        "cagr": growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0,
        "vol": float(daily.std() * math.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe": compute_sharpe_ratio(daily, periods_per_year=TRADING_DAYS_PER_YEAR),
        "max_drawdown": compute_max_drawdown(daily),
        "n_days": float(n),
    }


@dataclass
class WalkForwardResult:
    """Everything produced by one walk-forward run."""

    name: str
    config: WalkForwardConfig
    predictions: pd.Series
    ic: pd.Series
    quantile_returns: pd.DataFrame
    daily_returns: pd.DataFrame
    turnover: pd.Series
    random_null_sharpes: np.ndarray = field(default_factory=lambda: np.array([]))
    n_train_rows: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> dict[str, float]:
        """Flat dictionary of headline statistics."""
        cfg = self.config
        lag = overlap_lag(cfg.horizon, cfg.rebalance_every)
        ic_stats = compute_ic_stats(self.ic.to_numpy(), horizon_days=lag + 1)
        out: dict[str, float] = {
            "n_periods": float(len(self.ic)),
            "ic_mean": ic_stats["mean"],
            "ic_std": ic_stats["std"],
            "icir": ic_stats["mean"] / ic_stats["std"] if ic_stats["std"] else float("nan"),
            "ic_t_iid": ic_stats["t_stat_iid"],
            "ic_t_nw": ic_stats["t_stat_nw"],
            "ic_p_nw": ic_stats["p_value_nw"],
            "ic_hit_rate": ic_stats["pct_positive"],
            "nw_lag": float(lag),
        }

        if not self.quantile_returns.empty:
            q = self.quantile_returns
            spread = (q.iloc[:, -1] - q.iloc[:, 0]).dropna()
            spread_stats = compute_ic_stats(spread.to_numpy(), horizon_days=lag + 1)
            out["spread_mean"] = spread_stats["mean"]
            out["spread_t_nw"] = spread_stats["t_stat_nw"]
            out["spread_annualized"] = spread_stats["mean"] * TRADING_DAYS_PER_YEAR / cfg.horizon
            for col in q.columns:
                out[f"q{int(col) + 1}_mean"] = float(q[col].mean())

        for col in self.daily_returns.columns:
            for key, value in _return_stats(self.daily_returns[col]).items():
                out[f"{col}_{key}"] = value

        # Active (excess) return streams. A long-only portfolio's own Sharpe
        # mostly measures the market; selection skill shows up in these.
        for base in ("benchmark", "equal_weight"):
            if {"net", base} <= set(self.daily_returns.columns):
                active = (self.daily_returns["net"] - self.daily_returns[base]).dropna()
                if len(active) < 3:
                    continue
                ir = compute_sharpe_ratio(active, periods_per_year=TRADING_DAYS_PER_YEAR)
                out[f"net_active_vs_{base}_ann"] = float(active.mean() * TRADING_DAYS_PER_YEAR)
                out[f"net_ir_vs_{base}"] = ir
                out[f"net_ir_vs_{base}_deflated_prob"] = deflated_sharpe_ratio(
                    sharpe=ir,
                    n_obs=len(active),
                    n_trials=max(cfg.n_trials, 1),
                    skew=float(stats.skew(active)),
                    kurt=float(stats.kurtosis(active, fisher=False)),
                    periods_per_year=TRADING_DAYS_PER_YEAR,
                )

        if len(self.turnover):
            out["avg_turnover"] = float(self.turnover.iloc[1:].mean()) if len(self.turnover) > 1 else float("nan")

        null = self.random_null_sharpes[np.isfinite(self.random_null_sharpes)]
        if len(null) and "gross_sharpe" in out:
            out["random_null_sharpe_mean"] = float(null.mean())
            out["random_null_sharpe_p95"] = float(np.percentile(null, 95))
            out["gross_sharpe_percentile_vs_random"] = float((null < out["gross_sharpe"]).mean())
        return out

    def ic_by_year(self) -> pd.DataFrame:
        """Mean IC and period count per calendar year."""
        if self.ic.empty:
            return pd.DataFrame(columns=["mean_ic", "n"])
        grouped = self.ic.groupby(self.ic.index.year)
        return pd.DataFrame({"mean_ic": grouped.mean(), "n": grouped.size()})


# =============================================================================
# Main entry point
# =============================================================================


def _schedule(
    calendar: pd.DatetimeIndex,
    feature_dates: pd.DatetimeIndex,
    cfg: WalkForwardConfig,
) -> list[pd.Timestamp]:
    """Rebalance dates: every ``rebalance_every`` trading days once enough history exists."""
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    first_feature_pos = int(pos[feature_dates.min()])
    first = first_feature_pos + cfg.min_train_dates + cfg.horizon + cfg.execution_lag + cfg.embargo
    last = len(calendar) - 1 - cfg.execution_lag - max(cfg.horizon, cfg.rebalance_every)
    if first > last:
        return []
    available = set(feature_dates)
    start = pd.Timestamp(cfg.start) if cfg.start is not None else None
    end = pd.Timestamp(cfg.end) if cfg.end is not None else None
    dates = []
    for p in range(first, last + 1, cfg.rebalance_every):
        d = calendar[p]
        if d not in available:
            continue
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        dates.append(d)
    return dates


def run_walk_forward(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    model_factory: Callable[[], RankingModel],
    config: WalkForwardConfig | None = None,
    benchmark: str | None = "SPY",
    name: str = "model",
    groups: pd.Series | None = None,
) -> WalkForwardResult:
    """
    Walk-forward evaluate a ranking model with purged training windows.

    Args:
        features: Feature panel indexed by ``(date, ticker)``. Features dated ``t``
            must only use information available at the close of ``t``.
        prices: Wide adjusted-close panel (date x ticker), including ``benchmark``.
        model_factory: Zero-argument callable returning a fresh unfitted model.
        config: Walk-forward settings.
        benchmark: Benchmark column in ``prices`` (excluded from the investable
            universe). ``None`` disables benchmark reporting.
        name: Label used in logs and reports.
        groups: Ticker -> group label (e.g. GICS sector), used by the
            ``"group_rank"`` training target.

    Returns:
        ``WalkForwardResult``.
    """
    cfg = config or WalkForwardConfig()
    if not isinstance(features.index, pd.MultiIndex) or features.index.nlevels != 2:
        raise ValueError("features must be indexed by (date, ticker)")
    features = features.copy()
    features.index = features.index.set_names(["date", "ticker"])

    prices = prices.sort_index()
    calendar = pd.DatetimeIndex(prices.index)
    if not calendar.is_unique:
        raise ValueError("price index must not contain duplicate dates")

    tickers = sorted(
        (set(features.index.get_level_values("ticker")) & set(prices.columns)) - {benchmark}
    )
    if not tickers:
        raise ValueError("no overlap between feature tickers and price columns")
    stock_prices = prices[tickers]

    fwd_wide = forward_returns(stock_prices, cfg.horizon, cfg.execution_lag)
    fwd_long = fwd_wide.stack(future_stack=True)
    fwd_long.index = fwd_long.index.set_names(["date", "ticker"])

    X_all = features[features.index.get_level_values("ticker").isin(tickers)]
    X_all = X_all[X_all.index.get_level_values("date").isin(calendar)].sort_index()
    if cfg.dropna_features:
        X_all = X_all.dropna()
    if X_all.empty:
        raise ValueError("no usable feature rows after alignment")

    fwd_aligned = fwd_long.reindex(X_all.index)
    trailing_vol = None
    if cfg.target == "vol_norm":
        vol_wide = stock_prices.pct_change(fill_method=None).rolling(21, min_periods=10).std()
        trailing_vol = (vol_wide * math.sqrt(cfg.horizon)).stack(future_stack=True)
        trailing_vol.index = trailing_vol.index.set_names(["date", "ticker"])
        trailing_vol = trailing_vol.reindex(X_all.index)
    target = make_training_target(fwd_aligned, cfg.target, trailing_vol, groups)
    has_target = target.notna().to_numpy()

    pos_of = pd.Series(np.arange(len(calendar)), index=calendar)
    row_pos = pos_of.reindex(X_all.index.get_level_values("date")).to_numpy()
    feature_dates = pd.DatetimeIndex(X_all.index.get_level_values("date").unique()).sort_values()

    test_dates = _schedule(calendar, feature_dates, cfg)
    if not test_dates:
        raise ValueError("not enough history for a single walk-forward step")
    logger.info(
        "%s: %d walk-forward steps from %s to %s (purge=%s)",
        name, len(test_dates), test_dates[0].date(), test_dates[-1].date(), cfg.purge,
    )

    predictions: list[pd.Series] = []
    ic_values: dict[pd.Timestamp, float] = {}
    quantile_rows: dict[pd.Timestamp, pd.Series] = {}
    n_train_rows: dict[pd.Timestamp, int] = {}
    top_weights: dict[pd.Timestamp, pd.Series] = {}
    universe_weights: dict[pd.Timestamp, pd.Series] = {}
    universes: dict[pd.Timestamp, list[str]] = {}

    splits = purged_walk_forward_splits(
        dates=calendar,
        test_dates=test_dates,
        horizon=cfg.horizon,
        execution_lag=cfg.execution_lag,
        embargo=cfg.embargo,
        train_window=cfg.train_window,
        min_train_dates=cfg.min_train_dates,
        purge=cfg.purge,
    )
    for split in splits:
        test_date = split.test_date
        if cfg.purge:
            overlap = label_overlap_days(
                split.train_dates, test_date, calendar, cfg.horizon, cfg.execution_lag
            )
            if overlap > 0:  # pragma: no cover - guarded by the splitter
                raise AssertionError(f"label overlap of {overlap} days at {test_date.date()}")

        lo = int(pos_of[split.train_dates[0]])
        hi = int(pos_of[split.train_dates[-1]])
        train_mask = (row_pos >= lo) & (row_pos <= hi) & has_target
        if train_mask.sum() < max(cfg.min_names * 10, 50):
            continue

        test_mask = row_pos == int(pos_of[test_date])
        X_test = X_all[test_mask]
        if len(X_test) < cfg.min_names:
            continue

        model = model_factory()
        model.fit(X_all[train_mask], target[train_mask])
        scores = pd.Series(
            np.asarray(model.predict(X_test), dtype=float),
            index=X_test.index.get_level_values("ticker"),
            name=test_date,
        ).dropna()
        if len(scores) < cfg.min_names:
            continue

        n_train_rows[test_date] = int(train_mask.sum())
        predictions.append(
            pd.Series(scores.to_numpy(), index=pd.MultiIndex.from_product(
                [[test_date], scores.index], names=["date", "ticker"]))
        )
        universes[test_date] = list(scores.index)
        top = scores.nlargest(min(cfg.top_k, len(scores))).index
        top_weights[test_date] = pd.Series(1.0 / len(top), index=top)
        universe_weights[test_date] = pd.Series(1.0 / len(scores), index=scores.index)

        realized = fwd_wide.loc[test_date, scores.index]
        valid = realized.notna()
        if valid.sum() >= cfg.min_names:
            s, r = scores[valid], realized[valid]
            ic = stats.spearmanr(s, r).statistic
            if np.isfinite(ic):
                ic_values[test_date] = float(ic)
            buckets = pd.qcut(s.rank(method="first"), cfg.n_quantiles, labels=False)
            quantile_rows[test_date] = r.groupby(buckets).mean()

    if not predictions:
        raise ValueError("walk-forward produced no predictions; check data coverage")

    windows = []
    for d in top_weights:
        entry = int(pos_of[d]) + cfg.execution_lag
        windows.append(_Window(d, entry, min(entry + cfg.rebalance_every, len(calendar) - 1)))

    gross, net, turnover = simulate_portfolio(stock_prices, windows, top_weights, cfg.cost_bps)
    ew_gross, _, _ = simulate_portfolio(stock_prices, windows, universe_weights, 0.0)
    daily = pd.DataFrame({"gross": gross, "net": net, "equal_weight": ew_gross})
    if benchmark is not None and benchmark in prices.columns:
        bench = prices[benchmark].pct_change()
        daily["benchmark"] = bench.reindex(daily.index)

    null = np.array([])
    if cfg.n_random_paths > 0:
        null = random_selection_null(
            stock_prices, windows, universes, cfg.top_k, cfg.n_random_paths, cfg.random_seed
        )

    return WalkForwardResult(
        name=name,
        config=cfg,
        predictions=pd.concat(predictions),
        ic=pd.Series(ic_values, name="ic").sort_index(),
        quantile_returns=pd.DataFrame(quantile_rows).T.sort_index(),
        daily_returns=daily,
        turnover=turnover,
        random_null_sharpes=null,
        n_train_rows=pd.Series(n_train_rows, name="n_train_rows"),
    )


def summarize_many(results: Iterable[WalkForwardResult]) -> pd.DataFrame:
    """Side-by-side summary table for several runs (one column per run)."""
    return pd.DataFrame({r.name: r.summary() for r in results})
