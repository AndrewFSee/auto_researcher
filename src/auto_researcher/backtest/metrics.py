"""
Backtest performance metrics.

This module provides functions to compute standard portfolio performance metrics
including returns, volatility, Sharpe ratio, and ranking metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns from a series of periodic returns.

    Args:
        returns: Series of returns (e.g., daily or monthly).

    Returns:
        Cumulative return series (1 + cumulative return).

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01])
        >>> cum_ret = compute_cumulative_returns(returns)
        >>> len(cum_ret) == 3
        True
    """
    return (1 + returns).cumprod()


def compute_total_return(returns: pd.Series) -> float:
    """
    Compute total return over the entire period.

    Args:
        returns: Series of returns.

    Returns:
        Total return as a decimal (e.g., 0.25 for 25%).
    """
    return (1 + returns).prod() - 1


def compute_annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized return from a series of returns.

    Args:
        returns: Series of returns.
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly).

    Returns:
        Annualized return as a decimal.
    """
    total_periods = len(returns)
    if total_periods == 0:
        return 0.0
    
    total_return = compute_total_return(returns)
    years = total_periods / periods_per_year
    
    if years == 0:
        return 0.0
    
    # Annualized return: (1 + total_return)^(1/years) - 1
    annualized = (1 + total_return) ** (1 / years) - 1
    
    return annualized


def compute_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized volatility from a series of returns.

    Args:
        returns: Series of returns.
        periods_per_year: Number of periods per year.

    Returns:
        Annualized volatility (standard deviation).
    """
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the annualized Sharpe ratio.

    Uses the standard definition, mean periodic excess return over its
    standard deviation scaled by ``sqrt(periods_per_year)``. This is the
    quantity the deflated-Sharpe and Newey-West machinery assume. (An earlier
    version divided CAGR by volatility, which is not comparable with those
    statistics.)

    Args:
        returns: Series of periodic returns.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Sharpe ratio (0.0 when undefined).
    """
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return 0.0

    excess = r - risk_free_rate / periods_per_year
    std = excess.std()
    if not np.isfinite(std) or std == 0:
        return 0.0

    return float(excess.mean() / std * np.sqrt(periods_per_year))


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Sortino ratio (uses downside deviation).

    Args:
        returns: Series of returns.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Sortino ratio.
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    
    # Downside deviation
    negative_returns = returns[returns < 0]
    if len(negative_returns) < 2:
        return float('inf') if ann_return > risk_free_rate else 0.0
    
    downside_std = negative_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std == 0:
        return float('inf') if ann_return > risk_free_rate else 0.0
    
    return (ann_return - risk_free_rate) / downside_std


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown from a series of returns.

    The running peak starts at the initial capital (1.0), so a loss in the
    first period counts as drawdown. Previously the first period's value was
    used as the starting peak, which silently hid any drawdown that began
    immediately (e.g. returns of -10%, +5% reported a 0% max drawdown).

    Args:
        returns: Series of returns.

    Returns:
        Maximum drawdown as a non-positive decimal (e.g., -0.20 for -20%).
    """
    r = pd.Series(returns, dtype=float).fillna(0.0)
    if len(r) == 0:
        return 0.0

    wealth = np.concatenate([[1.0], (1.0 + r.to_numpy()).cumprod()])
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0

    return float(drawdown.min())


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Calmar ratio (annualized return / |max drawdown|).

    Args:
        returns: Series of returns.
        periods_per_year: Number of periods per year.

    Returns:
        Calmar ratio.
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    max_dd = compute_max_drawdown(returns)
    
    if max_dd == 0:
        return float('inf') if ann_return > 0 else 0.0
    
    return ann_return / abs(max_dd)


def compute_hit_rate(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Compute hit rate (fraction of periods outperforming benchmark).

    Args:
        portfolio_returns: Portfolio return series.
        benchmark_returns: Benchmark return series.

    Returns:
        Hit rate as a fraction (0.0 to 1.0).
    """
    # Align indices
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    port = portfolio_returns.loc[common_idx]
    bench = benchmark_returns.loc[common_idx]
    
    if len(port) == 0:
        return 0.0
    
    outperformance = port > bench
    return outperformance.mean()


def compute_ic(
    predicted_scores: pd.Series,
    realized_returns: pd.Series,
) -> float:
    """
    Compute Information Coefficient (rank correlation between predictions and realized returns).

    Uses Spearman correlation for robustness to outliers.

    Args:
        predicted_scores: Model predictions (higher = better expected return).
        realized_returns: Actual realized returns.

    Returns:
        Spearman correlation coefficient.
    """
    # Align indices
    common_idx = predicted_scores.index.intersection(realized_returns.index)
    pred = predicted_scores.loc[common_idx]
    real = realized_returns.loc[common_idx]
    
    # Drop NaN
    valid = pred.notna() & real.notna()
    pred = pred[valid]
    real = real[valid]
    
    if len(pred) < 3:
        return 0.0
    
    corr, _ = stats.spearmanr(pred, real)
    
    return corr if not np.isnan(corr) else 0.0


def compute_ic_stats(
    ic_values: np.ndarray | pd.Series,
    horizon_days: int = 1,
) -> dict[str, float]:
    """
    Summarize a sequence of cross-sectional IC observations.

    When the labels the ICs were computed against have a forward-return
    horizon greater than one rebalance step, consecutive IC samples share
    overlapping price windows and are autocorrelated. The plain iid t-stat
    then over-rejects the null ``mean(IC) == 0``. We report a Newey-West-
    adjusted t-stat (lag = ``horizon_days - 1``) alongside the iid one so
    the caller can see both.

    Args:
        ic_values: Sequence of per-rebalance Spearman ICs.
        horizon_days: Label horizon (in rebalance steps) used to set the
            Newey-West lag. ``1`` collapses to the iid case.

    Returns:
        ``{mean, std, n, pct_positive, t_stat_iid, p_value_iid,
        t_stat_nw, p_value_nw, ic_stat}`` where ``ic_stat = mean * sqrt(N)
        / std`` is the classical Lopez de Prado IC statistic.
    """
    from math import sqrt

    x = np.asarray(ic_values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = {
        "mean": float("nan"),
        "std": float("nan"),
        "n": float(n),
        "pct_positive": float("nan"),
        "t_stat_iid": float("nan"),
        "p_value_iid": float("nan"),
        "t_stat_nw": float("nan"),
        "p_value_nw": float("nan"),
        "ic_stat": float("nan"),
    }
    if n < 2:
        if n == 1:
            out["mean"] = float(x[0])
            out["pct_positive"] = float(x[0] > 0)
        return out

    mean = float(x.mean())
    std = float(x.std(ddof=1))
    out["mean"] = mean
    out["std"] = std
    out["pct_positive"] = float((x > 0).mean())

    if std > 0:
        t_iid = mean / (std / sqrt(n))
        p_iid = 2.0 * (1.0 - stats.norm.cdf(abs(t_iid)))
        out["t_stat_iid"] = float(t_iid)
        out["p_value_iid"] = float(p_iid)
        out["ic_stat"] = float(mean * sqrt(n) / std)

    # Newey-West long-run variance (Bartlett kernel).
    u = x - mean
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    k_max = max(int(horizon_days) - 1, 0)
    for k in range(1, k_max + 1):
        if k >= n:
            break
        weight = 1.0 - k / (k_max + 1)
        cov = float(np.dot(u[k:], u[:-k]) / n)
        lrv += 2.0 * weight * cov
    lrv = max(lrv, 1e-12)
    t_nw = mean / sqrt(lrv / n)
    p_nw = 2.0 * (1.0 - stats.norm.cdf(abs(t_nw)))
    out["t_stat_nw"] = float(t_nw)
    out["p_value_nw"] = float(p_nw)

    return out


def compute_ir(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Information Ratio (excess return / tracking error).

    Args:
        portfolio_returns: Portfolio return series.
        benchmark_returns: Benchmark return series.
        periods_per_year: Number of periods per year.

    Returns:
        Information Ratio.
    """
    # Align indices
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    port = portfolio_returns.loc[common_idx]
    bench = benchmark_returns.loc[common_idx]
    
    excess_returns = port - bench
    
    ann_excess = compute_annualized_return(excess_returns, periods_per_year)
    tracking_error = compute_annualized_volatility(excess_returns, periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    return ann_excess / tracking_error


def compute_win_loss_ratio(returns: pd.Series) -> float:
    """
    Compute ratio of average winning return to average losing return.

    Args:
        returns: Series of returns.

    Returns:
        Win/loss ratio (absolute value).
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-10
    
    return avg_win / avg_loss


def compute_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Compute all standard performance metrics.

    Args:
        portfolio_returns: Portfolio return series.
        benchmark_returns: Optional benchmark return series.
        periods_per_year: Number of periods per year.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "total_return": compute_total_return(portfolio_returns),
        "annualized_return": compute_annualized_return(portfolio_returns, periods_per_year),
        "annualized_volatility": compute_annualized_volatility(portfolio_returns, periods_per_year),
        "sharpe_ratio": compute_sharpe_ratio(portfolio_returns, risk_free_rate, periods_per_year),
        "sortino_ratio": compute_sortino_ratio(portfolio_returns, risk_free_rate, periods_per_year),
        "max_drawdown": compute_max_drawdown(portfolio_returns),
        "calmar_ratio": compute_calmar_ratio(portfolio_returns, periods_per_year),
        "win_loss_ratio": compute_win_loss_ratio(portfolio_returns),
    }
    
    if benchmark_returns is not None:
        metrics["hit_rate"] = compute_hit_rate(portfolio_returns, benchmark_returns)
        metrics["information_ratio"] = compute_ir(
            portfolio_returns, benchmark_returns, periods_per_year
        )
        
        # Compute benchmark metrics for comparison
        metrics["benchmark_return"] = compute_annualized_return(
            benchmark_returns, periods_per_year
        )
        metrics["excess_return"] = metrics["annualized_return"] - metrics["benchmark_return"]
    
    return metrics


# =============================================================================
# IC-Weighted Ensemble Utilities
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class ICWeightConfig:
    """
    Configuration for IC-weighted ensemble blending.

    Attributes:
        window_mom: Rolling window for momentum IC (in periods, e.g., 6 months).
        window_qual: Rolling window for quality IC (in periods, e.g., 12 months).
        min_weight: Minimum weight for each model (floor, default 0.0).
        fallback_weight_mom: Weight for momentum if both ICs are non-positive.
        label_horizon_periods: Number of rebalance periods in the label horizon.
            IC indexed at rebal-date D is computed from forward returns over
            [D, D+h] and therefore is only KNOWN at D+h. To avoid look-ahead,
            the rolling mean IC used to size weights at date D must exclude
            any IC realized after D — i.e. we shift the IC series forward by
            ``label_horizon_periods`` before rolling. Default 1 means IC[D] is
            used to decide weights for D+1's rebalance, never D's.
    """
    window_mom: int = 6  # 6 months (monthly rebalancing)
    window_qual: int = 12  # 12 months
    min_weight: float = 0.0
    fallback_weight_mom: float = 0.5  # Equal split if no positive IC
    label_horizon_periods: int = 1


def compute_rolling_ic_mean(
    ic_series: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute rolling mean IC with the given window.
    
    Args:
        ic_series: Series of IC values indexed by date.
        window: Rolling window size (in periods).
    
    Returns:
        Series of rolling mean IC values, aligned with input index.
        First (window-1) values will be NaN.
    """
    if len(ic_series) == 0:
        return pd.Series(dtype=float)
    
    # Use min_periods=1 to get values even before full window
    rolling_mean = ic_series.rolling(window=window, min_periods=1).mean()
    
    return rolling_mean


def compute_ic_weights(
    ic_mom: pd.Series,
    ic_qual: pd.Series,
    config: ICWeightConfig | None = None,
) -> pd.DataFrame:
    """
    Compute normalized non-negative weights from IC series.
    
    For each date:
    - Compute rolling mean IC for each model
    - Set raw weights = max(rolling_IC, 0.0)
    - Normalize to sum to 1, with fallback if both are 0
    
    Args:
        ic_mom: IC series for momentum model.
        ic_qual: IC series for quality model.
        config: Weight configuration. Uses defaults if None.
    
    Returns:
        DataFrame with columns ['w_mom', 'w_qual'] indexed by date.
    """
    if config is None:
        config = ICWeightConfig()

    # Align indices
    common_idx = ic_mom.index.intersection(ic_qual.index)
    ic_mom = ic_mom.loc[common_idx].sort_index()
    ic_qual = ic_qual.loc[common_idx].sort_index()

    # Lag the IC series by the label horizon BEFORE rolling. The IC at date D
    # is computed from forward returns [D, D+h] and is therefore only known at
    # D+h — using it to size weights at D is look-ahead. After this shift the
    # weight at D depends only on IC values labeled ≤ D - horizon.
    horizon_shift = max(int(config.label_horizon_periods), 0)
    if horizon_shift:
        ic_mom = ic_mom.shift(horizon_shift)
        ic_qual = ic_qual.shift(horizon_shift)

    # Compute rolling means
    rolling_mom = compute_rolling_ic_mean(ic_mom, config.window_mom)
    rolling_qual = compute_rolling_ic_mean(ic_qual, config.window_qual)
    
    # Raw weights: max(rolling_IC, 0)
    w_mom_raw = rolling_mom.clip(lower=0.0)
    w_qual_raw = rolling_qual.clip(lower=0.0)
    
    # Normalize
    total = w_mom_raw + w_qual_raw
    
    # Handle cases where total is 0 (fallback to config)
    w_mom = pd.Series(index=common_idx, dtype=float)
    w_qual = pd.Series(index=common_idx, dtype=float)
    
    # Where total > 0, normalize
    positive_total = total > 0
    w_mom[positive_total] = w_mom_raw[positive_total] / total[positive_total]
    w_qual[positive_total] = w_qual_raw[positive_total] / total[positive_total]
    
    # Where total == 0, use fallback
    w_mom[~positive_total] = config.fallback_weight_mom
    w_qual[~positive_total] = 1.0 - config.fallback_weight_mom
    
    # Apply minimum weight floor if configured
    if config.min_weight > 0:
        w_mom = w_mom.clip(lower=config.min_weight)
        w_qual = w_qual.clip(lower=config.min_weight)
        # Renormalize
        total = w_mom + w_qual
        w_mom = w_mom / total
        w_qual = w_qual / total
    
    return pd.DataFrame({
        'w_mom': w_mom,
        'w_qual': w_qual,
    }, index=common_idx)


# =============================================================================
# Transaction Cost Modeling
# =============================================================================


@dataclass
class CostModelConfig:
    """
    Configuration for transaction cost modeling.
    
    Attributes:
        bps_per_side: Basis points cost per side (buy or sell). Default 10 bps.
        apply_to_all_strategies: Whether to apply costs to all strategies.
    """
    bps_per_side: float = 10.0  # 10 basis points per side
    apply_to_all_strategies: bool = True


def compute_turnover(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
) -> float:
    """
    Compute portfolio turnover between two weight dictionaries.
    
    Turnover is defined as 0.5 * sum_i |w_t[i] - w_{t-1}[i]|.
    This represents the fraction of the portfolio that is traded.
    
    Args:
        old_weights: Previous period weights.
        new_weights: New period weights.
    
    Returns:
        Turnover as a fraction (0.0 to 1.0).
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    
    total_change = 0.0
    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0.0)
        new_w = new_weights.get(ticker, 0.0)
        total_change += abs(new_w - old_w)
    
    return total_change / 2.0


def compute_turnover_series(
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]],
    include_initial_build: bool = False,
) -> pd.Series:
    """
    Compute turnover series from portfolio weights history.
    
    Args:
        weights_history: List of (date, weights_dict) tuples.
        include_initial_build: If True, first period turnover is sum of initial weights.
            If False, first period is NaN (skipped from averages).
    
    Returns:
        Series of turnover values indexed by rebalance date.
    """
    if len(weights_history) < 1:
        return pd.Series(dtype=float)
    
    turnover_data = []
    prev_weights: dict[str, float] = {}
    
    for i, (date, weights) in enumerate(weights_history):
        if i == 0:
            if include_initial_build:
                # Initial "build" cost is sum of new weights = 1.0
                turnover = sum(max(w, 0.0) for w in weights.values())
                turnover_data.append((date, turnover))
            else:
                # Skip first period (NaN)
                turnover_data.append((date, np.nan))
        else:
            turnover = compute_turnover(prev_weights, weights)
            turnover_data.append((date, turnover))
        
        prev_weights = weights
    
    if not turnover_data:
        return pd.Series(dtype=float)
    
    dates, turnovers = zip(*turnover_data)
    return pd.Series(turnovers, index=pd.DatetimeIndex(dates), name="turnover")


def compute_net_returns(
    gross_returns: pd.Series,
    turnover_series: pd.Series,
    cost_config: CostModelConfig,
) -> pd.Series:
    """
    Compute net returns after deducting transaction costs.
    
    For each period:
        cost_t = turnover_t * (bps_per_side / 10000) * 2
        net_ret_t = gross_ret_t - cost_t
    
    The factor of 2 accounts for both buy and sell sides.
    
    Args:
        gross_returns: Series of gross portfolio returns indexed by date.
        turnover_series: Series of turnover values indexed by date.
        cost_config: Transaction cost configuration.
    
    Returns:
        Series of net returns indexed by date.
    """
    # Align turnover to return dates
    common_dates = gross_returns.index.intersection(turnover_series.index)
    
    gross_aligned = gross_returns.loc[common_dates]
    turnover_aligned = turnover_series.loc[common_dates]
    
    # Convert bps to decimal
    cost_per_side = cost_config.bps_per_side / 10000.0
    
    # Cost is turnover * cost_per_side * 2 (buy + sell)
    # If turnover is NaN, cost is NaN, so net return is NaN
    costs = turnover_aligned * cost_per_side * 2.0
    
    # Fill NaN costs with 0 for periods without turnover data
    costs = costs.fillna(0.0)
    
    net_returns = gross_aligned - costs
    net_returns.name = "net_return"
    
    return net_returns


def compute_net_metrics(
    net_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 12,
) -> dict[str, float]:
    """
    Compute performance metrics using net returns.
    
    Args:
        net_returns: Series of net returns.
        benchmark_returns: Optional benchmark returns.
        periods_per_year: Number of periods per year (12 for monthly).
    
    Returns:
        Dictionary with net metrics (ann_ret, sharpe, vol, max_dd).
    """
    if len(net_returns) == 0:
        return {
            "net_ann_ret": 0.0,
            "net_sharpe": 0.0,
            "net_vol": 0.0,
            "net_max_dd": 0.0,
        }
    
    metrics = {
        "net_ann_ret": compute_annualized_return(net_returns, periods_per_year),
        "net_sharpe": compute_sharpe_ratio(net_returns, 0.0, periods_per_year),
        "net_vol": compute_annualized_volatility(net_returns, periods_per_year),
        "net_max_dd": compute_max_drawdown(net_returns),
    }
    
    if benchmark_returns is not None:
        # Align to common dates
        common = net_returns.index.intersection(benchmark_returns.index)
        if len(common) > 0:
            metrics["net_hit_rate"] = compute_hit_rate(
                net_returns.loc[common], benchmark_returns.loc[common]
            )
            metrics["net_ir"] = compute_ir(
                net_returns.loc[common], benchmark_returns.loc[common], periods_per_year
            )
    
    return metrics
