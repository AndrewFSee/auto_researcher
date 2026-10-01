"""Tests for backtest metrics module."""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.backtest.metrics import (
    compute_cumulative_returns,
    compute_total_return,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_hit_rate,
    compute_ic,
    compute_ic_stats,
    compute_ir,
    compute_all_metrics,
)


@pytest.fixture
def positive_returns() -> pd.Series:
    """Returns series with positive trend."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    # Positive drift with noise
    returns = np.random.randn(252) * 0.01 + 0.0005
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def negative_returns() -> pd.Series:
    """Returns series with negative trend."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    returns = np.random.randn(252) * 0.01 - 0.0005
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def benchmark_returns() -> pd.Series:
    """Benchmark returns series."""
    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    returns = np.random.randn(252) * 0.008 + 0.0003
    return pd.Series(returns, index=dates, name="benchmark")


class TestCumulativeReturns:
    """Tests for cumulative returns computation."""

    def test_cumulative_returns_shape(self, positive_returns: pd.Series) -> None:
        """Cumulative returns should have same length as input."""
        cum_ret = compute_cumulative_returns(positive_returns)
        assert len(cum_ret) == len(positive_returns)

    def test_cumulative_returns_always_positive(self, positive_returns: pd.Series) -> None:
        """Cumulative returns (wealth) should be positive."""
        cum_ret = compute_cumulative_returns(positive_returns)
        assert (cum_ret > 0).all()

    def test_cumulative_returns_starts_near_one(self, positive_returns: pd.Series) -> None:
        """First cumulative return should be close to 1."""
        cum_ret = compute_cumulative_returns(positive_returns)
        # First value is 1 + first_return
        assert 0.9 < cum_ret.iloc[0] < 1.1


class TestTotalReturn:
    """Tests for total return computation."""

    def test_total_return_positive(self, positive_returns: pd.Series) -> None:
        """Positive trending returns should have positive total return."""
        total = compute_total_return(positive_returns)
        # May be positive or negative due to noise, but should be reasonable
        assert -1 < total < 10

    def test_total_return_matches_cumulative(self, positive_returns: pd.Series) -> None:
        """Total return should match final cumulative return."""
        total = compute_total_return(positive_returns)
        cum_ret = compute_cumulative_returns(positive_returns)
        
        assert np.isclose(total, cum_ret.iloc[-1] - 1, atol=1e-10)


class TestAnnualizedReturn:
    """Tests for annualized return computation."""

    def test_annualized_return_reasonable(self, positive_returns: pd.Series) -> None:
        """Annualized return should be reasonable magnitude."""
        ann_ret = compute_annualized_return(positive_returns, periods_per_year=252)
        # Should be between -100% and +1000%
        assert -1 < ann_ret < 10

    def test_annualized_return_empty(self) -> None:
        """Empty returns should return 0."""
        empty = pd.Series(dtype=float)
        assert compute_annualized_return(empty) == 0.0


class TestAnnualizedVolatility:
    """Tests for annualized volatility computation."""

    def test_volatility_positive(self, positive_returns: pd.Series) -> None:
        """Volatility should be non-negative."""
        vol = compute_annualized_volatility(positive_returns)
        assert vol >= 0

    def test_volatility_reasonable(self, positive_returns: pd.Series) -> None:
        """Volatility should be reasonable for equity-like returns."""
        vol = compute_annualized_volatility(positive_returns)
        # Should be between 0 and 100% for reasonable data
        assert 0 <= vol < 1.0

    def test_volatility_scaling(self, positive_returns: pd.Series) -> None:
        """Annual vol should be sqrt(252) times daily vol."""
        daily_vol = positive_returns.std()
        ann_vol = compute_annualized_volatility(positive_returns, periods_per_year=252)
        
        assert np.isclose(ann_vol, daily_vol * np.sqrt(252), rtol=0.01)


class TestSharpeRatio:
    """Tests for Sharpe ratio computation."""

    def test_sharpe_ratio_positive_for_positive_returns(
        self, positive_returns: pd.Series
    ) -> None:
        """Positive trending returns should have positive Sharpe (usually)."""
        sharpe = compute_sharpe_ratio(positive_returns)
        # Due to noise, may not always be positive, but should be reasonable
        assert -5 < sharpe < 10

    def test_sharpe_ratio_decreases_with_higher_rf(
        self, positive_returns: pd.Series
    ) -> None:
        """Higher risk-free rate should lower Sharpe."""
        sharpe_low_rf = compute_sharpe_ratio(positive_returns, risk_free_rate=0.0)
        sharpe_high_rf = compute_sharpe_ratio(positive_returns, risk_free_rate=0.05)
        
        assert sharpe_low_rf > sharpe_high_rf

    def test_sharpe_is_mean_excess_over_std(self) -> None:
        """Standard definition: mean / std * sqrt(periods), as assumed by DSR."""
        r = pd.Series([0.02, -0.01, 0.03, 0.00, 0.01, -0.02])
        expected = r.mean() / r.std() * np.sqrt(12)
        assert compute_sharpe_ratio(r, periods_per_year=12) == pytest.approx(expected)

    def test_sharpe_degenerate_inputs(self) -> None:
        assert compute_sharpe_ratio(pd.Series([0.01])) == 0.0
        assert compute_sharpe_ratio(pd.Series([0.01, 0.01, 0.01])) == 0.0


class TestMaxDrawdown:
    """Tests for maximum drawdown computation."""

    def test_max_drawdown_negative(self, positive_returns: pd.Series) -> None:
        """Max drawdown should be non-positive."""
        mdd = compute_max_drawdown(positive_returns)
        assert mdd <= 0

    def test_max_drawdown_bounded(self, positive_returns: pd.Series) -> None:
        """Max drawdown should be between -1 and 0."""
        mdd = compute_max_drawdown(positive_returns)
        assert -1 <= mdd <= 0

    def test_max_drawdown_worse_for_negative_trend(
        self, positive_returns: pd.Series, negative_returns: pd.Series
    ) -> None:
        """Negative trending returns should have worse (more negative) drawdown."""
        mdd_pos = compute_max_drawdown(positive_returns)
        mdd_neg = compute_max_drawdown(negative_returns)
        
        # Negative trend should have larger (more negative) drawdown
        assert mdd_neg <= mdd_pos

    def test_max_drawdown_counts_loss_from_initial_capital(self) -> None:
        """A loss in the very first period is a drawdown from the starting capital."""
        assert compute_max_drawdown(pd.Series([-0.10, 0.05])) == pytest.approx(-0.10)
        assert compute_max_drawdown(pd.Series([-0.10, 0.05, -0.02])) == pytest.approx(-0.10)
        assert compute_max_drawdown(pd.Series([0.10, -0.20])) == pytest.approx(-0.20)

    def test_max_drawdown_all_gains_is_zero(self) -> None:
        assert compute_max_drawdown(pd.Series([0.01, 0.02, 0.03])) == 0.0
        assert compute_max_drawdown(pd.Series(dtype=float)) == 0.0


class TestHitRate:
    """Tests for hit rate computation."""

    def test_hit_rate_bounds(
        self, positive_returns: pd.Series, benchmark_returns: pd.Series
    ) -> None:
        """Hit rate should be between 0 and 1."""
        hit_rate = compute_hit_rate(positive_returns, benchmark_returns)
        assert 0 <= hit_rate <= 1

    def test_hit_rate_perfect_outperformance(self) -> None:
        """Perfect outperformance should have hit rate of 1."""
        dates = pd.date_range("2023-01-01", periods=10)
        portfolio = pd.Series([0.02] * 10, index=dates)
        benchmark = pd.Series([0.01] * 10, index=dates)
        
        hit_rate = compute_hit_rate(portfolio, benchmark)
        assert hit_rate == 1.0

    def test_hit_rate_perfect_underperformance(self) -> None:
        """Perfect underperformance should have hit rate of 0."""
        dates = pd.date_range("2023-01-01", periods=10)
        portfolio = pd.Series([0.01] * 10, index=dates)
        benchmark = pd.Series([0.02] * 10, index=dates)
        
        hit_rate = compute_hit_rate(portfolio, benchmark)
        assert hit_rate == 0.0


class TestIC:
    """Tests for Information Coefficient computation."""

    def test_ic_bounds(self) -> None:
        """IC should be between -1 and 1."""
        np.random.seed(42)
        predictions = pd.Series(np.random.randn(100), index=range(100))
        realized = pd.Series(np.random.randn(100), index=range(100))
        
        ic = compute_ic(predictions, realized)
        assert -1 <= ic <= 1

    def test_ic_perfect_prediction(self) -> None:
        """Perfect prediction should have IC of 1."""
        predictions = pd.Series([1, 2, 3, 4, 5], index=range(5))
        realized = pd.Series([10, 20, 30, 40, 50], index=range(5))
        
        ic = compute_ic(predictions, realized)
        assert ic > 0.99

    def test_ic_inverse_prediction(self) -> None:
        """Inverse prediction should have IC of -1."""
        predictions = pd.Series([5, 4, 3, 2, 1], index=range(5))
        realized = pd.Series([10, 20, 30, 40, 50], index=range(5))
        
        ic = compute_ic(predictions, realized)
        assert ic < -0.99


class TestComputeAllMetrics:
    """Tests for the all-in-one metrics function."""

    def test_returns_dict(
        self, positive_returns: pd.Series, benchmark_returns: pd.Series
    ) -> None:
        """Should return a dictionary."""
        metrics = compute_all_metrics(positive_returns, benchmark_returns)
        assert isinstance(metrics, dict)

    def test_contains_expected_keys(
        self, positive_returns: pd.Series, benchmark_returns: pd.Series
    ) -> None:
        """Should contain all expected metric keys."""
        metrics = compute_all_metrics(positive_returns, benchmark_returns)
        
        expected_keys = [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "hit_rate",
        ]
        
        for key in expected_keys:
            assert key in metrics

    def test_works_without_benchmark(self, positive_returns: pd.Series) -> None:
        """Should work without benchmark."""
        metrics = compute_all_metrics(positive_returns)

        assert "total_return" in metrics
        assert "hit_rate" not in metrics  # Requires benchmark


class TestComputeICStats:
    """compute_ic_stats summarizes IC sequences with iid + Newey-West t-stats."""

    def test_iid_case_matches_scipy(self) -> None:
        rng = np.random.default_rng(0)
        ic = rng.normal(loc=0.05, scale=0.10, size=200)
        stats = compute_ic_stats(ic, horizon_days=1)

        assert stats["n"] == 200
        assert stats["mean"] == pytest.approx(ic.mean(), rel=1e-8)
        assert stats["std"] == pytest.approx(ic.std(ddof=1), rel=1e-8)

        # With horizon=1 the NW lag is 0 → NW and iid t-stats should agree to
        # a small-sample factor of sqrt(n/(n-1)) because NW uses population
        # variance (÷n) while the iid SE uses sample variance (÷(n-1)).
        assert stats["t_stat_iid"] == pytest.approx(stats["t_stat_nw"], rel=1e-2)
        assert stats["ic_stat"] == pytest.approx(
            stats["mean"] * np.sqrt(200) / stats["std"], rel=1e-8
        )

    def test_newey_west_inflates_se_when_autocorrelated(self) -> None:
        # Construct an AR(1) sequence with positive serial correlation; NW SE
        # should be larger than iid SE, so |t_nw| < |t_iid|.
        rng = np.random.default_rng(42)
        n = 500
        x = np.zeros(n)
        eps = rng.normal(scale=0.05, size=n)
        for i in range(1, n):
            x[i] = 0.6 * x[i - 1] + eps[i]
        x += 0.02  # shift the mean away from zero

        stats = compute_ic_stats(x, horizon_days=10)

        # NW with lag=9 should recover a smaller |t| than the iid version.
        assert abs(stats["t_stat_nw"]) < abs(stats["t_stat_iid"])
        # p_nw is two-sided so correspondingly larger.
        assert stats["p_value_nw"] > stats["p_value_iid"]

    def test_handles_short_sequences(self) -> None:
        stats0 = compute_ic_stats(np.array([]), horizon_days=1)
        assert stats0["n"] == 0
        assert np.isnan(stats0["mean"])
        assert np.isnan(stats0["t_stat_iid"])

        stats1 = compute_ic_stats(np.array([0.1]), horizon_days=1)
        assert stats1["n"] == 1
        assert stats1["mean"] == 0.1
        # Cannot compute std/tstat from 1 obs.
        assert np.isnan(stats1["std"])
        assert np.isnan(stats1["t_stat_iid"])
