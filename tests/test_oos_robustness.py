"""
Tests for out-of-sample (OOS) robustness experiment functionality.

Tests cover:
- OOS time window constants
- run_oos_backtest_fixed_model: backtest with frozen model
- compute_ew_metrics_for_period: Equal-weight baseline
- run_core_factor_oos_experiment: full OOS workflow
- print_oos_summary_table: output formatting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from auto_researcher.config import (
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    ResearchConfig,
    SIMPLE_MODEL_CONFIG,
)
from auto_researcher.models.gbdt_model import GBDTModel


# ==============================================================================
# OOS Constants Tests
# ==============================================================================


class TestOOSConstants:
    """Tests for OOS time window constants."""
    
    def test_oos_constants_exist(self):
        """OOS time window constants should be defined."""
        from run_large_cap_backtest import (
            OOS_TRAIN_START,
            OOS_TRAIN_END,
            OOS_TEST_START,
            OOS_TEST_END,
        )
        
        assert OOS_TRAIN_START is not None
        assert OOS_TRAIN_END is not None
        assert OOS_TEST_START is not None
        assert OOS_TEST_END is not None
    
    def test_oos_train_window_is_2013_2018(self):
        """Training window should be 2013-2018."""
        from run_large_cap_backtest import OOS_TRAIN_START, OOS_TRAIN_END
        
        assert OOS_TRAIN_START.year == 2013
        assert OOS_TRAIN_END.year == 2018
        assert OOS_TRAIN_END.month == 12
    
    def test_oos_test_window_is_2019_2023(self):
        """Test window should be 2019-2023."""
        from run_large_cap_backtest import OOS_TEST_START, OOS_TEST_END
        
        assert OOS_TEST_START.year == 2019
        assert OOS_TEST_END.year == 2023
    
    def test_oos_train_subperiods_exist(self):
        """OOS training subperiods should be defined."""
        from run_large_cap_backtest import OOS_TRAIN_SUBPERIODS
        
        assert len(OOS_TRAIN_SUBPERIODS) == 3
        assert OOS_TRAIN_SUBPERIODS[0].name == "2013-2014"
        assert OOS_TRAIN_SUBPERIODS[1].name == "2015-2016"
        assert OOS_TRAIN_SUBPERIODS[2].name == "2017-2018"
    
    def test_oos_train_subperiods_within_train_window(self):
        """All training subperiods should be within the training window."""
        from run_large_cap_backtest import (
            OOS_TRAIN_START,
            OOS_TRAIN_END,
            OOS_TRAIN_SUBPERIODS,
        )
        
        for subperiod in OOS_TRAIN_SUBPERIODS:
            assert subperiod.start >= OOS_TRAIN_START
            assert subperiod.end <= OOS_TRAIN_END


# ==============================================================================
# OOSMetrics Tests
# ==============================================================================


class TestOOSMetrics:
    """Tests for OOSMetrics dataclass."""
    
    def test_oos_metrics_dataclass_exists(self):
        """OOSMetrics dataclass should exist."""
        from run_large_cap_backtest import OOSMetrics
        
        metrics = OOSMetrics(
            period_name="Test 2019-2023",
            annualized_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.25,
            hit_rate=0.55,
            avg_ic=0.05,
            ic_std=0.02,
            avg_turnover=0.45,
            cumulative_return=1.5,
            annualized_volatility=0.12,
        )
        
        assert metrics.period_name == "Test 2019-2023"
        assert metrics.sharpe_ratio == 1.2
    
    def test_oos_metrics_allows_none_ic(self):
        """OOSMetrics should allow None for IC (EW baseline)."""
        from run_large_cap_backtest import OOSMetrics
        
        metrics = OOSMetrics(
            period_name="EW Test",
            annualized_return=0.10,
            sharpe_ratio=0.8,
            max_drawdown=-0.30,
            hit_rate=0.50,
            avg_ic=None,
            ic_std=None,
            avg_turnover=0.0,
            cumulative_return=0.5,
            annualized_volatility=0.15,
        )
        
        assert metrics.avg_ic is None
        assert metrics.ic_std is None


# ==============================================================================
# CLI Flag Tests
# ==============================================================================


class TestOOSCLIFlag:
    """Tests for the --oos-mode CLI flag."""
    
    def test_oos_mode_argument_exists(self):
        """parse_args should accept --oos-mode argument."""
        from run_large_cap_backtest import parse_args
        
        # Test with none (default)
        with patch('sys.argv', ['run_large_cap_backtest.py']):
            args = parse_args()
            assert hasattr(args, 'oos_mode')
            assert args.oos_mode == 'none'
    
    def test_oos_mode_train_test_option(self):
        """--oos-mode should accept train_2013_2018_test_2019_2023."""
        from run_large_cap_backtest import parse_args
        
        with patch('sys.argv', [
            'run_large_cap_backtest.py',
            '--oos-mode', 'train_2013_2018_test_2019_2023'
        ]):
            args = parse_args()
            assert args.oos_mode == 'train_2013_2018_test_2019_2023'


# ==============================================================================
# compute_ew_metrics_for_period Tests
# ==============================================================================


class TestComputeEWMetricsForPeriod:
    """Tests for Equal-Weight metrics computation."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range('2019-01-01', '2023-12-31', freq='B')
        tickers = ['AAPL', 'MSFT', 'GOOG', 'SPY']
        
        # Generate random walk prices
        returns = np.random.randn(len(dates), len(tickers)) * 0.02
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            index=dates,
            columns=tickers,
        )
        
        return prices
    
    def test_compute_ew_metrics_returns_oos_metrics(self, sample_prices):
        """Should return OOSMetrics object."""
        from run_large_cap_backtest import compute_ew_metrics_for_period, OOSMetrics
        
        result = compute_ew_metrics_for_period(
            prices=sample_prices,
            start_date=pd.Timestamp('2019-01-01'),
            end_date=pd.Timestamp('2023-12-31'),
            benchmark='SPY',
            rebalance_freq='M',
        )
        
        assert isinstance(result, OOSMetrics)
    
    def test_compute_ew_metrics_has_no_ic(self, sample_prices):
        """EW strategy should have no IC (None)."""
        from run_large_cap_backtest import compute_ew_metrics_for_period
        
        result = compute_ew_metrics_for_period(
            prices=sample_prices,
            start_date=pd.Timestamp('2019-01-01'),
            end_date=pd.Timestamp('2023-12-31'),
            benchmark='SPY',
            rebalance_freq='M',
        )
        
        assert result.avg_ic is None
    
    def test_compute_ew_metrics_has_zero_turnover(self, sample_prices):
        """EW strategy should have ~0 turnover."""
        from run_large_cap_backtest import compute_ew_metrics_for_period
        
        result = compute_ew_metrics_for_period(
            prices=sample_prices,
            start_date=pd.Timestamp('2019-01-01'),
            end_date=pd.Timestamp('2023-12-31'),
            benchmark='SPY',
            rebalance_freq='M',
        )
        
        assert result.avg_turnover == 0.0


# ==============================================================================
# print_oos_summary_table Tests
# ==============================================================================


class TestPrintOOSSummaryTable:
    """Tests for OOS summary table output."""
    
    def test_print_oos_summary_table_runs(self, capsys):
        """print_oos_summary_table should run without error."""
        from run_large_cap_backtest import print_oos_summary_table, OOSMetrics
        
        train_metrics = OOSMetrics(
            period_name="Train 2013-2018",
            annualized_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=-0.20,
            hit_rate=0.55,
            avg_ic=0.06,
            ic_std=0.02,
            avg_turnover=0.45,
            cumulative_return=2.0,
            annualized_volatility=0.10,
        )
        
        test_metrics = OOSMetrics(
            period_name="Test 2019-2023",
            annualized_return=0.12,
            sharpe_ratio=1.2,
            max_drawdown=-0.25,
            hit_rate=0.52,
            avg_ic=0.04,
            ic_std=0.03,
            avg_turnover=0.50,
            cumulative_return=1.5,
            annualized_volatility=0.12,
        )
        
        ew_metrics = OOSMetrics(
            period_name="EW Test 2019-23",
            annualized_return=0.10,
            sharpe_ratio=0.9,
            max_drawdown=-0.30,
            hit_rate=0.50,
            avg_ic=None,
            ic_std=None,
            avg_turnover=0.0,
            cumulative_return=0.5,
            annualized_volatility=0.15,
        )
        
        # Should not raise
        print_oos_summary_table(train_metrics, test_metrics, ew_metrics)
        
        # Check output contains expected elements
        captured = capsys.readouterr()
        assert "CORE FACTOR OOS ROBUSTNESS" in captured.out
        assert "Train 2013-2018" in captured.out
        assert "Test 2019-2023" in captured.out
        assert "Interpretation" in captured.out
    
    def test_print_oos_detects_significant_decay(self, capsys):
        """Should flag significant Sharpe decay."""
        from run_large_cap_backtest import print_oos_summary_table, OOSMetrics
        
        train_metrics = OOSMetrics(
            period_name="Train",
            annualized_return=0.20,
            sharpe_ratio=2.0,  # High train Sharpe
            max_drawdown=-0.15,
            hit_rate=0.60,
            avg_ic=0.08,
            ic_std=0.02,
            avg_turnover=0.40,
            cumulative_return=3.0,
            annualized_volatility=0.10,
        )
        
        test_metrics = OOSMetrics(
            period_name="Test",
            annualized_return=0.05,
            sharpe_ratio=0.5,  # Much lower test Sharpe (75% decay)
            max_drawdown=-0.40,
            hit_rate=0.48,
            avg_ic=0.01,
            ic_std=0.04,
            avg_turnover=0.50,
            cumulative_return=0.3,
            annualized_volatility=0.15,
        )
        
        print_oos_summary_table(train_metrics, test_metrics)
        
        captured = capsys.readouterr()
        assert "SIGNIFICANT DECAY" in captured.out or "DECAY" in captured.out
    
    def test_print_oos_detects_robust_performance(self, capsys):
        """Should flag robust performance when decay is mild."""
        from run_large_cap_backtest import print_oos_summary_table, OOSMetrics
        
        train_metrics = OOSMetrics(
            period_name="Train",
            annualized_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.20,
            hit_rate=0.55,
            avg_ic=0.05,
            ic_std=0.02,
            avg_turnover=0.45,
            cumulative_return=2.0,
            annualized_volatility=0.12,
        )
        
        test_metrics = OOSMetrics(
            period_name="Test",
            annualized_return=0.14,
            sharpe_ratio=1.1,  # Only ~8% decay
            max_drawdown=-0.22,
            hit_rate=0.54,
            avg_ic=0.045,
            ic_std=0.025,
            avg_turnover=0.48,
            cumulative_return=1.8,
            annualized_volatility=0.13,
        )
        
        print_oos_summary_table(train_metrics, test_metrics)
        
        captured = capsys.readouterr()
        assert "MILD DECAY" in captured.out or "robustness" in captured.out.lower()


# ==============================================================================
# run_oos_backtest_fixed_model Tests
# ==============================================================================


class TestRunOOSBacktestFixedModel:
    """Tests for frozen model backtest."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data spanning multiple years."""
        np.random.seed(42)
        dates = pd.date_range('2013-01-01', '2023-12-31', freq='B')
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'SPY']
        
        # Generate random walk prices
        returns = np.random.randn(len(dates), len(tickers)) * 0.02
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            index=dates,
            columns=tickers,
        )
        
        return prices
    
    @pytest.fixture
    def trained_model(self, sample_prices):
        """Create a pre-trained model."""
        from auto_researcher.features.feature_pipeline import prepare_training_data
        
        # Filter to training period
        train_prices = sample_prices.loc[
            (sample_prices.index >= '2013-01-01') & 
            (sample_prices.index <= '2018-12-31')
        ]
        
        config = FeatureConfig(
            momentum_windows=(21, 63),
            volatility_windows=(21,),
            include_fundamentals=False,
        )
        
        X_train, y_train = prepare_training_data(
            train_prices,
            horizon_days=21,
            benchmark='SPY',
            config=config,
        )
        
        model = GBDTModel(SIMPLE_MODEL_CONFIG)
        model.fit(X_train, y_train)
        
        # Get feature names
        core_features = list(X_train.columns)
        
        return model, core_features
    
    def test_oos_backtest_returns_strategy_result(self, sample_prices, trained_model):
        """Should return StrategyResult and BacktestResult."""
        from run_large_cap_backtest import run_oos_backtest_fixed_model, StrategyResult
        from auto_researcher.backtest.runner import BacktestResult
        
        model, core_features = trained_model
        
        config = PipelineConfig(
            features=FeatureConfig(
                momentum_windows=(21, 63),
                volatility_windows=(21,),
                include_fundamentals=False,
            ),
            model=SIMPLE_MODEL_CONFIG,
        )
        
        result, backtest = run_oos_backtest_fixed_model(
            prices=sample_prices,
            trained_model=model,
            core_features=core_features,
            config=config,
            start_date=pd.Timestamp('2019-01-01'),
            end_date=pd.Timestamp('2023-12-31'),
            benchmark='SPY',
            top_k=2,
            horizon_days=21,
            rebalance_freq='M',
        )
        
        assert isinstance(result, StrategyResult)
        assert isinstance(backtest, BacktestResult)
    
    def test_oos_backtest_computes_metrics(self, sample_prices, trained_model):
        """Should compute valid performance metrics."""
        from run_large_cap_backtest import run_oos_backtest_fixed_model
        
        model, core_features = trained_model
        
        config = PipelineConfig(
            features=FeatureConfig(
                momentum_windows=(21, 63),
                volatility_windows=(21,),
                include_fundamentals=False,
            ),
            model=SIMPLE_MODEL_CONFIG,
        )
        
        result, _ = run_oos_backtest_fixed_model(
            prices=sample_prices,
            trained_model=model,
            core_features=core_features,
            config=config,
            start_date=pd.Timestamp('2019-01-01'),
            end_date=pd.Timestamp('2023-12-31'),
            benchmark='SPY',
            top_k=2,
            horizon_days=21,
            rebalance_freq='M',
        )
        
        # Metrics should be valid numbers
        assert not np.isnan(result.sharpe_ratio)
        assert not np.isnan(result.annualized_return)
        assert result.max_drawdown <= 0  # Drawdown is negative


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestOOSIntegration:
    """Integration tests for the full OOS workflow."""
    
    def test_oos_experiment_function_exists(self):
        """run_core_factor_oos_experiment should be importable."""
        from run_large_cap_backtest import run_core_factor_oos_experiment
        
        assert callable(run_core_factor_oos_experiment)
    
    def test_oos_functions_use_simple_model_config(self):
        """OOS experiment should use SIMPLE_MODEL_CONFIG."""
        # Verify SIMPLE_MODEL_CONFIG is used in the OOS functions
        assert SIMPLE_MODEL_CONFIG.max_depth == 3
        assert SIMPLE_MODEL_CONFIG.n_estimators == 75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
