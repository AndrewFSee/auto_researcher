"""Tests for backtest runner module."""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.backtest.runner import (
    run_backtest,
    run_simple_backtest,
    get_rebalance_dates,
    BacktestResult,
)
from auto_researcher.models.gbdt_model import GBDTModel
from auto_researcher.config import (
    PipelineConfig,
    ResearchConfig,
    FeatureConfig,
    ModelConfig,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=500, freq="D")
    
    # Generate random walk prices for multiple stocks + SPY benchmark
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "SPY"]
    n_tickers = len(tickers)
    
    returns = np.random.randn(500, n_tickers) * 0.02
    # Add some structure: positive drift
    returns += 0.0005
    
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def test_config() -> PipelineConfig:
    """Minimal config for fast tests."""
    return PipelineConfig(
        research=ResearchConfig(
            universe=("AAPL", "MSFT", "GOOG", "AMZN", "META", "SPY"),
            horizon_days=21,
            top_k=2,
            lookback_days=63,
            rebalance_frequency="M",
        ),
        features=FeatureConfig(
            momentum_windows=(21,),
            volatility_windows=(21,),
        ),
        model=ModelConfig(
            n_estimators=10,
            early_stopping_rounds=None,
        ),
    )


class TestGetRebalanceDates:
    """Tests for rebalance date generation."""

    def test_monthly_rebalance(self) -> None:
        """Monthly rebalance should have ~12 dates per year."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        rebal = get_rebalance_dates(dates, "M")
        
        # Should have roughly 12 month-ends
        assert 10 <= len(rebal) <= 14

    def test_weekly_rebalance(self) -> None:
        """Weekly rebalance should have ~52 dates per year."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        rebal = get_rebalance_dates(dates, "W")
        
        assert 45 <= len(rebal) <= 55

    def test_rebalance_dates_in_original(self) -> None:
        """All rebalance dates should be in original date range."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        rebal = get_rebalance_dates(dates, "M")
        
        for d in rebal:
            assert d in dates


class TestRunBacktest:
    """Tests for the main backtest runner."""

    def test_returns_backtest_result(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Should return BacktestResult."""
        result = run_backtest(
            prices=sample_prices,
            horizon_days=test_config.research.horizon_days,
            top_k=test_config.research.top_k,
            benchmark="SPY",
            rebalance_frequency="M",
            min_train_periods=63,
            config=test_config,
        )
        
        assert isinstance(result, BacktestResult)

    def test_portfolio_returns_is_series(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Portfolio returns should be a pandas Series."""
        result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=2,
            benchmark="SPY",
            min_train_periods=63,
            config=test_config,
        )
        
        assert isinstance(result.portfolio_returns, pd.Series)

    def test_benchmark_returns_is_series(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Benchmark returns should be a pandas Series."""
        result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=2,
            benchmark="SPY",
            min_train_periods=63,
            config=test_config,
        )
        
        assert isinstance(result.benchmark_returns, pd.Series)

    def test_returns_are_nonempty(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Should produce non-empty returns."""
        result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=2,
            benchmark="SPY",
            min_train_periods=63,
            config=test_config,
        )
        
        assert len(result.portfolio_returns) > 0

    def test_metrics_dict_populated(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Metrics dictionary should be populated."""
        result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=2,
            benchmark="SPY",
            min_train_periods=63,
            config=test_config,
        )
        
        assert isinstance(result.metrics, dict)
        assert "sharpe_ratio" in result.metrics
        assert "annualized_return" in result.metrics

    def test_weights_history_has_entries(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Weights history should have entries."""
        result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=2,
            benchmark="SPY",
            min_train_periods=63,
            config=test_config,
        )
        
        assert len(result.weights_history) > 0

    def test_weights_sum_to_one(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Portfolio weights should sum to 1."""
        result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=2,
            benchmark="SPY",
            min_train_periods=63,
            config=test_config,
        )
        
        for date, weights in result.weights_history:
            total = sum(weights.values())
            assert np.isclose(total, 1.0, atol=0.01)

    def test_benchmark_not_in_portfolio(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Benchmark ticker should not be in portfolio."""
        result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=2,
            benchmark="SPY",
            min_train_periods=63,
            config=test_config,
        )
        
        for date, weights in result.weights_history:
            assert "SPY" not in weights

    def test_raises_on_missing_benchmark(
        self, sample_prices: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Should raise if benchmark not in prices."""
        with pytest.raises(ValueError, match="Benchmark"):
            run_backtest(
                prices=sample_prices,
                horizon_days=21,
                top_k=2,
                benchmark="INVALID",
                min_train_periods=63,
                config=test_config,
            )


class TestRunSimpleBacktest:
    """Tests for simplified backtest runner."""

    def test_runs_with_default_config(self, sample_prices: pd.DataFrame) -> None:
        """Should run with default config."""
        config = PipelineConfig(
            research=ResearchConfig(
                horizon_days=21,
                top_k=2,
                lookback_days=63,
            ),
            features=FeatureConfig(
                momentum_windows=(21,),
                volatility_windows=(21,),
            ),
            model=ModelConfig(
                n_estimators=10,
                early_stopping_rounds=None,
            ),
        )
        
        result = run_simple_backtest(sample_prices, config)
        
        assert isinstance(result, BacktestResult)

    def test_produces_valid_output(self, sample_prices: pd.DataFrame) -> None:
        """Should produce valid backtest output."""
        config = PipelineConfig(
            research=ResearchConfig(
                horizon_days=21,
                top_k=2,
                lookback_days=63,
            ),
            features=FeatureConfig(
                momentum_windows=(21,),
                volatility_windows=(21,),
            ),
            model=ModelConfig(
                n_estimators=10,
                early_stopping_rounds=None,
            ),
        )
        
        result = run_simple_backtest(sample_prices, config)
        
        assert len(result.portfolio_returns) > 0
        assert "sharpe_ratio" in result.metrics
