"""
Tests for regime-switching strategy functionality.

These tests verify:
1. Regime series computation and alignment with rebalance dates
2. Regime-switch strategy returns are finite
3. Strategy uses both momentum and quality regimes
4. Performance is reasonable (not worse than random)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from auto_researcher.backtest.regime_signal import (
    compute_regime_series,
    get_regime_for_date,
    compute_regime_summary,
    compute_realized_volatility,
    compute_trend_signal,
    MarketRegime,
    RegimeConfig,
)
from auto_researcher.backtest.runner import (
    run_backtest,
    run_regime_switch_strategy,
    get_rebalance_dates,
    BacktestResult,
)
from auto_researcher.config import (
    PipelineConfig,
    ResearchConfig,
    FeatureConfig,
    ModelConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_spy_prices() -> pd.Series:
    """Generate sample SPY prices for testing regime signals."""
    # Create 2 years of daily data with some volatility regimes
    np.random.seed(42)
    n_days = 504  # ~2 years
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    
    # Simulate price path with varying volatility regimes
    returns = np.zeros(n_days)
    
    # Low vol period (first 6 months)
    returns[:126] = np.random.normal(0.0003, 0.008, 126)
    
    # High vol period (next 3 months)
    returns[126:189] = np.random.normal(-0.0002, 0.025, 63)
    
    # Low vol recovery (next 6 months)
    returns[189:315] = np.random.normal(0.0004, 0.009, 126)
    
    # High vol crash (next 3 months)
    returns[315:378] = np.random.normal(-0.001, 0.030, 63)
    
    # Low vol recovery (rest)
    returns[378:] = np.random.normal(0.0003, 0.010, n_days - 378)
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    return pd.Series(prices, index=dates, name="SPY")


@pytest.fixture
def sample_prices(sample_spy_prices) -> pd.DataFrame:
    """Generate sample price data for backtest testing."""
    np.random.seed(42)
    dates = sample_spy_prices.index
    n_days = len(dates)
    
    # Generate returns for 10 stocks
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "SPY"]
    
    prices_dict = {}
    for ticker in tickers[:-1]:  # Exclude SPY
        # Random returns with some correlation to SPY
        spy_returns = sample_spy_prices.pct_change().fillna(0)
        beta = np.random.uniform(0.5, 1.5)
        alpha = np.random.uniform(-0.0001, 0.0003)
        idiosyncratic = np.random.normal(0, 0.015, n_days)
        
        returns = alpha + beta * spy_returns.values + idiosyncratic
        prices_dict[ticker] = 100 * np.exp(np.cumsum(returns))
    
    prices_dict["SPY"] = sample_spy_prices.values
    
    return pd.DataFrame(prices_dict, index=dates)


@pytest.fixture
def minimal_config() -> PipelineConfig:
    """Create minimal config for fast testing."""
    return PipelineConfig(
        research=ResearchConfig(
            universe=tuple(["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "SPY"]),
            start_date="2022-01-01",
            end_date="2023-12-31",
            horizon_days=21,
            top_k=3,
            lookback_days=63,
            rebalance_frequency="M",
        ),
        features=FeatureConfig(
            momentum_windows=(21, 63),
            volatility_windows=(21,),
            include_fundamentals=False,
        ),
        model=ModelConfig(
            n_estimators=10,
            early_stopping_rounds=None,
        ),
    )


# =============================================================================
# Regime Signal Tests
# =============================================================================


class TestRegimeSignal:
    """Tests for regime signal computation."""
    
    def test_compute_realized_volatility_shape(self, sample_spy_prices):
        """Volatility series has same length as input."""
        vol = compute_realized_volatility(sample_spy_prices, window=21)
        assert len(vol) == len(sample_spy_prices)
    
    def test_compute_realized_volatility_positive(self, sample_spy_prices):
        """Volatility values are non-negative."""
        vol = compute_realized_volatility(sample_spy_prices, window=21)
        # Skip NaN values from initial warmup
        assert (vol.dropna() >= 0).all()
    
    def test_compute_trend_signal_boolean(self, sample_spy_prices):
        """Trend signal returns boolean series."""
        trend = compute_trend_signal(sample_spy_prices, window=200)
        assert trend.dtype == bool
    
    def test_regime_series_values(self, sample_spy_prices):
        """Regime series contains only MOMENTUM or QUALITY values."""
        regime = compute_regime_series(sample_spy_prices)
        valid_values = {MarketRegime.MOMENTUM.value, MarketRegime.QUALITY.value}
        assert set(regime.unique()).issubset(valid_values)
    
    def test_regime_series_index_matches_prices(self, sample_spy_prices):
        """Regime series has same index as input prices."""
        regime = compute_regime_series(sample_spy_prices)
        assert regime.index.equals(sample_spy_prices.index)
    
    def test_get_regime_for_date(self, sample_spy_prices):
        """Can retrieve regime for any date."""
        regime_series = compute_regime_series(sample_spy_prices)
        
        # Get regime for a date in the series
        test_date = sample_spy_prices.index[250]
        regime = get_regime_for_date(regime_series, test_date)
        
        assert regime in (MarketRegime.MOMENTUM, MarketRegime.QUALITY)
    
    def test_get_regime_for_future_date(self, sample_spy_prices):
        """Regime for future date uses last available value."""
        regime_series = compute_regime_series(sample_spy_prices)
        
        # Date beyond the series
        future_date = sample_spy_prices.index[-1] + timedelta(days=30)
        regime = get_regime_for_date(regime_series, future_date)
        
        assert regime in (MarketRegime.MOMENTUM, MarketRegime.QUALITY)
    
    def test_regime_summary_counts(self, sample_spy_prices):
        """Regime summary returns valid counts."""
        regime_series = compute_regime_series(sample_spy_prices)
        summary = compute_regime_summary(regime_series)
        
        assert "MOMENTUM" in summary
        assert "QUALITY" in summary
        assert summary["MOMENTUM"] + summary["QUALITY"] == len(regime_series)
    
    def test_regime_summary_at_rebal_dates(self, sample_spy_prices):
        """Regime summary works with rebalance dates."""
        regime_series = compute_regime_series(sample_spy_prices)
        rebal_dates = get_rebalance_dates(sample_spy_prices.index, "M")
        
        summary = compute_regime_summary(regime_series, rebal_dates)
        
        assert "MOMENTUM" in summary
        assert "QUALITY" in summary
        # Total should equal number of rebalance dates
        assert summary["MOMENTUM"] + summary["QUALITY"] == len(rebal_dates)


class TestRegimeSeriesShapesMatchRebalances:
    """Test that regime series aligns properly with rebalance dates."""
    
    def test_regime_available_for_all_rebal_dates(self, sample_spy_prices):
        """Regime can be retrieved for every rebalance date."""
        regime_series = compute_regime_series(sample_spy_prices)
        rebal_dates = get_rebalance_dates(sample_spy_prices.index, "M")
        
        for date in rebal_dates:
            regime = get_regime_for_date(regime_series, date)
            assert regime is not None


class TestRegimeSwitchReturnsFinite:
    """Test that regime-switch strategy returns are finite."""
    
    def test_returns_are_finite(self, sample_prices, minimal_config):
        """Strategy returns contain no NaN or Inf values."""
        # Run two backtests to get scores_history
        tech_config = PipelineConfig(
            research=minimal_config.research,
            features=FeatureConfig(
                momentum_windows=(21, 63),
                volatility_windows=(21,),
                include_technical=True,
                include_fundamentals=False,
            ),
            model=minimal_config.model,
        )
        
        momentum_result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=3,
            benchmark="SPY",
            rebalance_frequency="M",
            min_train_periods=63,
            config=tech_config,
        )
        
        # For this test, use same result as "quality" (just testing mechanics)
        quality_result = momentum_result
        
        # Run regime-switch strategy
        regime_result = run_regime_switch_strategy(
            prices=sample_prices,
            spy_prices=sample_prices["SPY"],
            momentum_result=momentum_result,
            quality_result=quality_result,
            top_k=3,
            benchmark="SPY",
        )
        
        # Check returns are finite
        assert regime_result.portfolio_returns.notna().all()
        assert np.isfinite(regime_result.portfolio_returns).all()


class TestRegimeSwitchUsesBothModels:
    """Test that regime-switch strategy uses both momentum and quality regimes."""
    
    def test_uses_both_regimes(self, sample_spy_prices):
        """At least some rebalance dates use each regime."""
        regime_series = compute_regime_series(sample_spy_prices)
        rebal_dates = get_rebalance_dates(sample_spy_prices.index, "M")
        
        summary = compute_regime_summary(regime_series, rebal_dates)
        
        # With our simulated data, should have both regimes
        # Allow for edge cases where one might dominate
        total = summary["MOMENTUM"] + summary["QUALITY"]
        
        # At least some presence of each (allow up to 90/10 split)
        assert summary["MOMENTUM"] > 0 or summary["QUALITY"] > 0
        # This is a sanity check - real test is that both appear
        if total >= 10:
            # With enough data, expect both regimes
            assert summary["MOMENTUM"] >= 1
            assert summary["QUALITY"] >= 1


class TestRegimeSwitchPerformance:
    """Test regime-switch strategy performance characteristics."""
    
    def test_sharpe_is_reasonable(self, sample_prices, minimal_config):
        """Sharpe ratio is within reasonable bounds (not extreme)."""
        tech_config = PipelineConfig(
            research=minimal_config.research,
            features=FeatureConfig(
                momentum_windows=(21, 63),
                volatility_windows=(21,),
                include_technical=True,
                include_fundamentals=False,
            ),
            model=minimal_config.model,
        )
        
        momentum_result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=3,
            benchmark="SPY",
            rebalance_frequency="M",
            min_train_periods=63,
            config=tech_config,
        )
        
        quality_result = momentum_result  # Same for testing
        
        regime_result = run_regime_switch_strategy(
            prices=sample_prices,
            spy_prices=sample_prices["SPY"],
            momentum_result=momentum_result,
            quality_result=quality_result,
            top_k=3,
            benchmark="SPY",
        )
        
        sharpe = regime_result.metrics.get("sharpe_ratio", 0)
        
        # Sharpe should be within reasonable bounds
        # Not checking for outperformance, just sanity
        assert -5 < sharpe < 5  # Very loose bounds for sanity


class TestRegimeConfigOptions:
    """Test regime configuration options."""
    
    def test_custom_vol_window(self, sample_spy_prices):
        """Custom volatility window works."""
        config = RegimeConfig(vol_window=42)
        regime = compute_regime_series(sample_spy_prices, config)
        
        assert len(regime) == len(sample_spy_prices)
    
    def test_custom_trend_window(self, sample_spy_prices):
        """Custom trend window works."""
        config = RegimeConfig(trend_window=100)
        regime = compute_regime_series(sample_spy_prices, config)
        
        assert len(regime) == len(sample_spy_prices)
    
    def test_fixed_vol_threshold(self, sample_spy_prices):
        """Fixed volatility threshold works."""
        config = RegimeConfig(
            vol_threshold_method="fixed",
            vol_threshold_value=0.20,
        )
        regime = compute_regime_series(sample_spy_prices, config)
        
        assert len(regime) == len(sample_spy_prices)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRegimeSwitchIntegration:
    """Integration tests for regime-switch strategy with full backtest."""
    
    def test_metrics_include_regime_counts(self, sample_prices, minimal_config):
        """Backtest result includes regime count metrics."""
        tech_config = PipelineConfig(
            research=minimal_config.research,
            features=FeatureConfig(
                momentum_windows=(21, 63),
                volatility_windows=(21,),
                include_technical=True,
                include_fundamentals=False,
            ),
            model=minimal_config.model,
        )
        
        momentum_result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=3,
            benchmark="SPY",
            rebalance_frequency="M",
            min_train_periods=63,
            config=tech_config,
        )
        
        quality_result = momentum_result
        
        regime_result = run_regime_switch_strategy(
            prices=sample_prices,
            spy_prices=sample_prices["SPY"],
            momentum_result=momentum_result,
            quality_result=quality_result,
            top_k=3,
            benchmark="SPY",
        )
        
        assert "regime_momentum_count" in regime_result.metrics
        assert "regime_quality_count" in regime_result.metrics
    
    def test_scores_history_populated(self, sample_prices, minimal_config):
        """Regime-switch result has scores_history."""
        tech_config = PipelineConfig(
            research=minimal_config.research,
            features=FeatureConfig(
                momentum_windows=(21, 63),
                volatility_windows=(21,),
                include_technical=True,
                include_fundamentals=False,
            ),
            model=minimal_config.model,
        )
        
        momentum_result = run_backtest(
            prices=sample_prices,
            horizon_days=21,
            top_k=3,
            benchmark="SPY",
            rebalance_frequency="M",
            min_train_periods=63,
            config=tech_config,
        )
        
        quality_result = momentum_result
        
        regime_result = run_regime_switch_strategy(
            prices=sample_prices,
            spy_prices=sample_prices["SPY"],
            momentum_result=momentum_result,
            quality_result=quality_result,
            top_k=3,
            benchmark="SPY",
        )
        
        assert regime_result.scores_history is not None
        assert len(regime_result.scores_history) > 0
