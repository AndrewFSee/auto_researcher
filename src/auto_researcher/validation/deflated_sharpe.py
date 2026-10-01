"""
Deflated Sharpe Ratio (DSR).

When you try many strategies and report the best one, its Sharpe is biased
upward — simply by chance one of them will look good on the single out-of-
sample path you tested. Bailey and López de Prado (2014) show how to penalize
the observed Sharpe for (a) the number of configurations tried and (b) the
skew/kurtosis of the return stream.

Reference:
    Bailey, D. H., & López de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting and Non-Normality." Journal of Portfolio Management.

Use this *after* a CPCV run to convert a headline Sharpe into a probability
that the strategy is genuinely profitable.
"""

from __future__ import annotations

import math

import numpy as np

EULER_MASCHERONI = 0.5772156649015328606


def expected_max_sharpe_under_null(n_trials: int, sharpe_std: float = 1.0) -> float:
    """
    Expected maximum Sharpe you'd see after ``n_trials`` draws from N(0, 1),
    scaled by ``sharpe_std`` (the observed cross-trial Sharpe stdev).

    Uses the standard extreme-value approximation:
        E[max_N] ≈ (1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N·e))
    """
    from scipy.stats import norm

    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sharpe_std * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sharpe_std: float | None = None,
    periods_per_year: float | None = None,
) -> float:
    """
    Probability that the true Sharpe ratio is strictly positive, after
    deflating for selection bias and non-Normal returns.

    The estimator variance below (Mertens 2002 / Lo 2002) is defined for the
    **per-period** Sharpe ratio: the mean over the standard deviation of the
    ``n_obs`` period returns, not annualized. Passing an annualized Sharpe
    with a daily ``n_obs`` overstates significance by roughly
    ``sqrt(periods_per_year)`` in z-score terms. Supply ``periods_per_year``
    to pass annualized inputs; they are converted before use.

    Args:
        sharpe: Observed Sharpe of the *chosen* strategy (per-period, or
            annualized when ``periods_per_year`` is given).
        n_obs: Number of return observations behind ``sharpe``.
        n_trials: How many configurations were tried while picking this one.
            Use ``n_trials = 1`` when no search took place.
        skew: Sample skewness of the return stream.
        kurt: Sample kurtosis (not excess) of the return stream. Default 3
            corresponds to Normal returns.
        sharpe_std: Standard deviation of the Sharpes across the ``n_trials``
            candidates, in the same units as ``sharpe``. When unknown, the
            sampling standard deviation of a Sharpe estimate under the null,
            ``1 / sqrt(n_obs - 1)`` per period, is used.
        periods_per_year: If given, ``sharpe`` and ``sharpe_std`` are treated
            as annualized and divided by ``sqrt(periods_per_year)``.

    Returns:
        P(SR* > 0), where SR* is the deflated Sharpe. Values near 1 indicate
        a strategy that survives the multiple-testing penalty; values near
        0.5 or below mean the headline Sharpe is likely a fluke.
    """
    from scipy.stats import norm

    if n_obs < 2 or not np.isfinite(sharpe):
        return float("nan")

    scale = math.sqrt(periods_per_year) if periods_per_year else 1.0
    sr = sharpe / scale
    if sharpe_std is None:
        sr_std = 1.0 / math.sqrt(n_obs - 1.0)
    else:
        sr_std = sharpe_std / scale

    expected_max = expected_max_sharpe_under_null(n_trials, sr_std)

    # Variance of the per-period Sharpe estimator:
    #   Var(SR) ≈ [1 - skew·SR + (kurt - 1)/4 · SR²] / (n - 1)
    sharpe_var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n_obs - 1.0)
    sharpe_var = max(sharpe_var, 1e-12)

    z = (sr - expected_max) / math.sqrt(sharpe_var)
    return float(norm.cdf(z))


def deflated_sharpe_from_samples(
    sample_sharpes: np.ndarray,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Convenience wrapper that estimates ``n_trials`` and ``sharpe_std`` from
    the sample of candidate Sharpes. ``sample_sharpes[-1]`` is taken as the
    chosen strategy's Sharpe; the rest of the array is treated as the
    configurations evaluated during selection.
    """
    s = np.asarray(sample_sharpes, dtype=float)
    if len(s) == 0:
        return float("nan")

    chosen = float(np.nanmax(s))
    n_trials = int(np.isfinite(s).sum())
    std = float(np.nanstd(s, ddof=1)) if n_trials > 1 else 1.0
    return deflated_sharpe_ratio(
        sharpe=chosen,
        n_obs=n_obs,
        n_trials=n_trials,
        skew=skew,
        kurt=kurt,
        sharpe_std=std if std > 0 else 1.0,
    )
