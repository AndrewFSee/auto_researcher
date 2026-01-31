"""
Tests for transaction cost modeling and turnover computation.

This module tests:
- Turnover computation between portfolio weights
- Transaction cost application to returns
- Net-of-cost metrics computation
- BacktestResult integration with cost fields
"""

import pytest
import numpy as np
import pandas as pd

from auto_researcher.backtest.metrics import (
    CostModelConfig,
    compute_turnover,
    compute_turnover_series,
    compute_net_returns,
    compute_net_metrics,
)


class TestComputeTurnover:
    """Test turnover computation between weight dictionaries."""
    
    def test_turnover_no_change(self):
        """Turnover is 0 when weights don't change."""
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        turnover = compute_turnover(weights, weights)
        assert turnover == 0.0
    
    def test_turnover_full_rebalance(self):
        """Turnover is 1.0 when portfolio completely changes."""
        old_weights = {"AAPL": 1.0}
        new_weights = {"MSFT": 1.0}
        turnover = compute_turnover(old_weights, new_weights)
        assert turnover == 1.0
    
    def test_turnover_partial_change(self):
        """Turnover is 0.5 when half the portfolio changes."""
        old_weights = {"AAPL": 0.5, "MSFT": 0.5}
        new_weights = {"AAPL": 0.5, "GOOG": 0.5}  # Swap MSFT for GOOG
        turnover = compute_turnover(old_weights, new_weights)
        # |0.5 - 0.5| + |0.5 - 0| + |0 - 0.5| = 0 + 0.5 + 0.5 = 1.0
        # turnover = 1.0 / 2 = 0.5
        assert turnover == 0.5
    
    def test_turnover_basic_example(self):
        """Test basic turnover calculation with known result."""
        old_weights = {"A": 0.4, "B": 0.3, "C": 0.3}
        new_weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        # |0.4 - 0.5| + |0.3 - 0.3| + |0.3 - 0.2| = 0.1 + 0 + 0.1 = 0.2
        # turnover = 0.2 / 2 = 0.1
        turnover = compute_turnover(old_weights, new_weights)
        assert abs(turnover - 0.1) < 1e-9
    
    def test_turnover_empty_old_weights(self):
        """Turnover when starting from empty portfolio."""
        old_weights = {}
        new_weights = {"AAPL": 0.5, "MSFT": 0.5}
        turnover = compute_turnover(old_weights, new_weights)
        # sum(|new|) / 2 = 1.0 / 2 = 0.5
        assert turnover == 0.5


class TestComputeTurnoverSeries:
    """Test turnover series computation from weights history."""
    
    def test_turnover_series_basic(self):
        """Test basic turnover series computation."""
        weights_history = [
            (pd.Timestamp("2020-01-31"), {"A": 0.5, "B": 0.5}),
            (pd.Timestamp("2020-02-29"), {"A": 0.6, "B": 0.4}),
            (pd.Timestamp("2020-03-31"), {"A": 0.4, "B": 0.6}),
        ]
        
        turnover_series = compute_turnover_series(weights_history, include_initial_build=False)
        
        assert len(turnover_series) == 3
        assert pd.isna(turnover_series.iloc[0])  # First is NaN
        assert abs(turnover_series.iloc[1] - 0.1) < 1e-9  # |0.5-0.6| + |0.5-0.4| = 0.2 / 2 = 0.1
        assert abs(turnover_series.iloc[2] - 0.2) < 1e-9  # |0.6-0.4| + |0.4-0.6| = 0.4 / 2 = 0.2
    
    def test_turnover_series_with_initial_build(self):
        """Test turnover series including initial portfolio build."""
        weights_history = [
            (pd.Timestamp("2020-01-31"), {"A": 0.5, "B": 0.5}),
            (pd.Timestamp("2020-02-29"), {"A": 1.0}),
        ]
        
        turnover_series = compute_turnover_series(weights_history, include_initial_build=True)
        
        assert len(turnover_series) == 2
        assert turnover_series.iloc[0] == 1.0  # Initial build = sum of weights = 1.0
        assert turnover_series.iloc[1] == 0.5  # Full swap from {A:0.5, B:0.5} to {A:1.0}
    
    def test_turnover_series_empty_history(self):
        """Test turnover series with empty history."""
        turnover_series = compute_turnover_series([], include_initial_build=False)
        assert len(turnover_series) == 0


class TestCostApplication:
    """Test transaction cost application to returns."""
    
    def test_cost_application_reduces_returns(self):
        """Net returns should be lower than gross returns for positive cost."""
        gross_returns = pd.Series(
            [0.05, 0.03, -0.02, 0.04, 0.01],
            index=pd.date_range("2020-01-31", periods=5, freq="ME"),
        )
        turnover_series = pd.Series(
            [0.3, 0.4, 0.5, 0.2, 0.3],
            index=pd.date_range("2020-01-31", periods=5, freq="ME"),
        )
        cost_config = CostModelConfig(bps_per_side=10.0)
        
        net_returns = compute_net_returns(gross_returns, turnover_series, cost_config)
        
        # All net returns should be lower than gross returns
        for i in range(len(gross_returns)):
            assert net_returns.iloc[i] < gross_returns.iloc[i]
    
    def test_cost_model_config_default_values(self):
        """Test CostModelConfig has correct default values."""
        config = CostModelConfig()
        assert config.bps_per_side == 10.0
        assert config.apply_to_all_strategies is True
    
    def test_cost_calculation_formula(self):
        """Test that cost is calculated as turnover * bps * 2."""
        gross_returns = pd.Series(
            [0.05],
            index=[pd.Timestamp("2020-01-31")],
        )
        turnover_series = pd.Series(
            [0.5],  # 50% turnover
            index=[pd.Timestamp("2020-01-31")],
        )
        cost_config = CostModelConfig(bps_per_side=10.0)
        
        net_returns = compute_net_returns(gross_returns, turnover_series, cost_config)
        
        # Expected cost: 0.5 * (10/10000) * 2 = 0.5 * 0.001 * 2 = 0.001
        expected_cost = 0.5 * 0.001 * 2  # 0.001
        expected_net = 0.05 - expected_cost  # 0.049
        
        assert abs(net_returns.iloc[0] - expected_net) < 1e-9
    
    def test_zero_turnover_means_zero_cost(self):
        """Zero turnover should result in no cost deduction."""
        gross_returns = pd.Series(
            [0.05, 0.03],
            index=pd.date_range("2020-01-31", periods=2, freq="ME"),
        )
        turnover_series = pd.Series(
            [0.0, 0.0],  # No turnover
            index=pd.date_range("2020-01-31", periods=2, freq="ME"),
        )
        cost_config = CostModelConfig(bps_per_side=10.0)
        
        net_returns = compute_net_returns(gross_returns, turnover_series, cost_config)
        
        pd.testing.assert_series_equal(
            net_returns,
            gross_returns.rename("net_return"),
            check_names=True,
        )


class TestNetMetrics:
    """Test net-of-cost metrics computation."""
    
    def test_net_sharpe_lower_than_gross(self):
        """Net Sharpe should be lower than gross Sharpe for positive costs."""
        # Create a positive return series
        gross_returns = pd.Series(
            [0.02] * 12,  # 2% per month for 12 months
            index=pd.date_range("2020-01-31", periods=12, freq="ME"),
        )
        turnover_series = pd.Series(
            [0.4] * 12,  # 40% turnover each month
            index=pd.date_range("2020-01-31", periods=12, freq="ME"),
        )
        cost_config = CostModelConfig(bps_per_side=10.0)
        
        net_returns = compute_net_returns(gross_returns, turnover_series, cost_config)
        net_metrics = compute_net_metrics(net_returns, None, periods_per_year=12)
        
        # Compute gross Sharpe manually
        from auto_researcher.backtest.metrics import compute_sharpe_ratio
        gross_sharpe = compute_sharpe_ratio(gross_returns, 0.0, 12)
        net_sharpe = net_metrics["net_sharpe"]
        
        assert net_sharpe < gross_sharpe
    
    def test_net_metrics_has_expected_keys(self):
        """Net metrics dictionary should have expected keys."""
        returns = pd.Series(
            [0.01, 0.02, -0.01],
            index=pd.date_range("2020-01-31", periods=3, freq="ME"),
        )
        
        net_metrics = compute_net_metrics(returns, None, periods_per_year=12)
        
        assert "net_ann_ret" in net_metrics
        assert "net_sharpe" in net_metrics
        assert "net_vol" in net_metrics
        assert "net_max_dd" in net_metrics


class TestBacktestResultIntegration:
    """Test BacktestResult integration with transaction cost fields."""
    
    def test_backtest_result_has_cost_fields(self):
        """BacktestResult should have optional cost-related fields."""
        from auto_researcher.backtest.runner import BacktestResult
        from auto_researcher.config import get_default_config
        
        # Create minimal BacktestResult
        result = BacktestResult(
            portfolio_returns=pd.Series([0.01, 0.02]),
            benchmark_returns=pd.Series([0.01, 0.01]),
            weights_history=[],
            ic_history=[],
            metrics={"sharpe_ratio": 1.0},
            config=get_default_config(),
        )
        
        # Check that cost fields exist and default to None
        assert result.turnover_series is None
        assert result.avg_turnover is None
        assert result.p95_turnover is None
        assert result.net_returns is None
        assert result.net_metrics is None
    
    def test_backtest_result_with_cost_data(self):
        """BacktestResult should accept cost data."""
        from auto_researcher.backtest.runner import BacktestResult
        from auto_researcher.config import get_default_config
        
        turnover_series = pd.Series([0.3, 0.4], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
        net_returns = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
        
        result = BacktestResult(
            portfolio_returns=pd.Series([0.02, 0.03]),
            benchmark_returns=pd.Series([0.01, 0.01]),
            weights_history=[],
            ic_history=[],
            metrics={"sharpe_ratio": 1.0},
            config=get_default_config(),
            turnover_series=turnover_series,
            avg_turnover=0.35,
            p95_turnover=0.39,
            net_returns=net_returns,
            net_metrics={"net_ann_ret": 0.15, "net_sharpe": 0.9},
        )
        
        assert result.avg_turnover == 0.35
        assert result.p95_turnover == 0.39
        assert len(result.turnover_series) == 2
        assert len(result.net_returns) == 2
        assert result.net_metrics["net_ann_ret"] == 0.15
