"""
Tests for Regularization Sweep functionality.

This module tests:
- RegularizationSweepConfig: configuration for hyperparameter sweeps
- build_model_config_from_params: factory function for model configs
- SweepResult: result dataclass for sweep results
- print_sweep_results_table: output formatting
- CLI flag integration
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ==============================================================================
# RegularizationSweepConfig Tests
# ==============================================================================


class TestRegularizationSweepConfig:
    """Tests for RegularizationSweepConfig dataclass."""
    
    def test_config_exists(self):
        """RegularizationSweepConfig should be importable."""
        from auto_researcher.config import RegularizationSweepConfig
        
        config = RegularizationSweepConfig()
        assert config is not None
        assert config.enable is False  # Default is disabled
    
    def test_default_grid_values(self):
        """Default grids should have reasonable values."""
        from auto_researcher.config import RegularizationSweepConfig
        
        config = RegularizationSweepConfig()
        
        assert len(config.max_depth_grid) >= 2
        assert len(config.n_estimators_grid) >= 2
        assert len(config.reg_lambda_grid) >= 2
        assert len(config.subsample_grid) >= 2
        assert len(config.colsample_grid) >= 2
        
        # Check value ranges
        assert all(d >= 1 for d in config.max_depth_grid)
        assert all(n >= 1 for n in config.n_estimators_grid)
        assert all(r >= 0 for r in config.reg_lambda_grid)
        assert all(0 < s <= 1 for s in config.subsample_grid)
        assert all(0 < c <= 1 for c in config.colsample_grid)
    
    def test_grid_size_calculation(self):
        """grid_size() should return product of all grid lengths."""
        from auto_researcher.config import RegularizationSweepConfig
        
        config = RegularizationSweepConfig(
            max_depth_grid=(2, 3),
            n_estimators_grid=(50, 75),
            reg_lambda_grid=(0.0, 5.0),
            subsample_grid=(0.6, 0.8),
            colsample_grid=(0.5, 0.8),
        )
        
        expected = 2 * 2 * 2 * 2 * 2  # 32
        assert config.grid_size() == expected
    
    def test_default_sweep_config(self):
        """DEFAULT_REG_SWEEP_CONFIG should be properly configured."""
        from auto_researcher.config import DEFAULT_REG_SWEEP_CONFIG
        
        assert DEFAULT_REG_SWEEP_CONFIG.enable is True
        assert DEFAULT_REG_SWEEP_CONFIG.grid_size() == 32


# ==============================================================================
# build_model_config_from_params Tests
# ==============================================================================


class TestBuildModelConfigFromParams:
    """Tests for build_model_config_from_params function."""
    
    def test_function_exists(self):
        """build_model_config_from_params should be importable."""
        from auto_researcher.config import build_model_config_from_params
        
        assert callable(build_model_config_from_params)
    
    def test_default_base_config(self):
        """Should use SIMPLE_MODEL_CONFIG as default base."""
        from auto_researcher.config import (
            build_model_config_from_params,
            SIMPLE_MODEL_CONFIG,
        )
        
        config = build_model_config_from_params()
        
        assert config.n_estimators == SIMPLE_MODEL_CONFIG.n_estimators
        assert config.max_depth == SIMPLE_MODEL_CONFIG.max_depth
    
    def test_override_max_depth(self):
        """Should override max_depth when specified."""
        from auto_researcher.config import build_model_config_from_params
        
        config = build_model_config_from_params(max_depth=2)
        
        assert config.max_depth == 2
    
    def test_override_multiple_params(self):
        """Should override multiple parameters at once."""
        from auto_researcher.config import build_model_config_from_params
        
        config = build_model_config_from_params(
            max_depth=2,
            n_estimators=50,
            reg_lambda=5.0,
            subsample=0.6,
            colsample_bytree=0.5,
        )
        
        assert config.max_depth == 2
        assert config.n_estimators == 50
        assert config.reg_lambda == 5.0
        assert config.subsample == 0.6
        assert config.colsample_bytree == 0.5
    
    def test_custom_base_config(self):
        """Should accept custom base config."""
        from auto_researcher.config import build_model_config_from_params, ModelConfig
        
        custom_base = ModelConfig(
            n_estimators=200,
            max_depth=10,
        )
        
        config = build_model_config_from_params(
            base_config=custom_base,
            reg_lambda=3.0,
        )
        
        assert config.n_estimators == 200  # From custom base
        assert config.max_depth == 10      # From custom base
        assert config.reg_lambda == 3.0    # Overridden
    
    def test_invalid_param_raises_error(self):
        """Should raise error for unknown parameters."""
        from auto_researcher.config import build_model_config_from_params
        
        with pytest.raises(ValueError, match="Unknown ModelConfig parameter"):
            build_model_config_from_params(invalid_param=42)


# ==============================================================================
# ModelConfig with regularization params Tests
# ==============================================================================


class TestModelConfigRegularization:
    """Tests for regularization parameters in ModelConfig."""
    
    def test_model_config_has_reg_lambda(self):
        """ModelConfig should have reg_lambda field."""
        from auto_researcher.config import ModelConfig
        
        config = ModelConfig(reg_lambda=5.0)
        assert config.reg_lambda == 5.0
    
    def test_model_config_has_subsample(self):
        """ModelConfig should have subsample field."""
        from auto_researcher.config import ModelConfig
        
        config = ModelConfig(subsample=0.8)
        assert config.subsample == 0.8
    
    def test_model_config_has_colsample(self):
        """ModelConfig should have colsample_bytree field."""
        from auto_researcher.config import ModelConfig
        
        config = ModelConfig(colsample_bytree=0.5)
        assert config.colsample_bytree == 0.5
    
    def test_default_values(self):
        """Default regularization values should be non-regularizing."""
        from auto_researcher.config import ModelConfig
        
        config = ModelConfig()
        assert config.reg_lambda == 0.0
        assert config.subsample == 1.0
        assert config.colsample_bytree == 1.0


# ==============================================================================
# GBDTModel with regularization Tests
# ==============================================================================


class TestGBDTModelRegularization:
    """Tests for GBDTModel with regularization parameters."""
    
    def test_model_uses_reg_lambda(self):
        """GBDTModel should pass reg_lambda to LightGBM."""
        from auto_researcher.config import ModelConfig
        from auto_researcher.models.gbdt_model import GBDTModel
        
        config = ModelConfig(
            n_estimators=10,
            max_depth=3,
            reg_lambda=5.0,
        )
        
        model = GBDTModel(config)
        lgb_model = model._create_model()
        
        # Check that the parameter was passed
        assert lgb_model.reg_lambda == 5.0
    
    def test_model_uses_colsample(self):
        """GBDTModel should pass colsample_bytree to LightGBM."""
        from auto_researcher.config import ModelConfig
        from auto_researcher.models.gbdt_model import GBDTModel
        
        config = ModelConfig(
            n_estimators=10,
            max_depth=3,
            colsample_bytree=0.5,
        )
        
        model = GBDTModel(config)
        lgb_model = model._create_model()
        
        assert lgb_model.colsample_bytree == 0.5
    
    def test_model_trains_with_regularization(self):
        """GBDTModel should train successfully with regularization."""
        from auto_researcher.config import ModelConfig
        from auto_researcher.models.gbdt_model import GBDTModel
        
        config = ModelConfig(
            n_estimators=10,
            max_depth=2,
            reg_lambda=5.0,
            subsample=0.6,
            colsample_bytree=0.5,
        )
        
        # Create simple training data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })
        y = pd.Series(np.random.randn(100))
        
        model = GBDTModel(config)
        model.fit(X, y)
        
        # Should predict without error
        predictions = model.predict(X)
        assert len(predictions) == 100


# ==============================================================================
# SweepResult Tests
# ==============================================================================


class TestSweepResult:
    """Tests for SweepResult dataclass."""
    
    def test_sweep_result_exists(self):
        """SweepResult should be importable."""
        from run_large_cap_backtest import SweepResult
        
        result = SweepResult(
            max_depth=2,
            n_estimators=50,
            reg_lambda=5.0,
            subsample=0.6,
            colsample=0.5,
            train_sharpe=2.5,
            train_ic=0.15,
            train_ann_ret=0.25,
            train_maxdd=-0.10,
            test_sharpe=1.5,
            test_ic=0.08,
            test_ann_ret=0.18,
            test_maxdd=-0.15,
            test_turnover=0.22,
        )
        
        assert result.max_depth == 2
        assert result.test_sharpe == 1.5
    
    def test_sweep_result_allows_none_ic(self):
        """SweepResult should allow None for IC."""
        from run_large_cap_backtest import SweepResult
        
        result = SweepResult(
            max_depth=2,
            n_estimators=50,
            reg_lambda=5.0,
            subsample=0.6,
            colsample=0.5,
            train_sharpe=2.5,
            train_ic=None,  # Can be None
            train_ann_ret=0.25,
            train_maxdd=-0.10,
            test_sharpe=1.5,
            test_ic=None,   # Can be None
            test_ann_ret=0.18,
            test_maxdd=-0.15,
            test_turnover=0.22,
        )
        
        assert result.train_ic is None
        assert result.test_ic is None


# ==============================================================================
# print_sweep_results_table Tests
# ==============================================================================


class TestPrintSweepResultsTable:
    """Tests for print_sweep_results_table function."""
    
    def test_function_exists(self):
        """print_sweep_results_table should be importable."""
        from run_large_cap_backtest import print_sweep_results_table
        
        assert callable(print_sweep_results_table)
    
    def test_handles_empty_results(self, capsys):
        """Should handle empty results gracefully."""
        from run_large_cap_backtest import print_sweep_results_table
        
        print_sweep_results_table([])
        
        captured = capsys.readouterr()
        assert "No results" in captured.out
    
    def test_sorts_by_test_sharpe(self, capsys):
        """Should sort results by test Sharpe descending."""
        from run_large_cap_backtest import SweepResult, print_sweep_results_table
        
        results = [
            SweepResult(
                max_depth=2, n_estimators=50, reg_lambda=0.0,
                subsample=1.0, colsample=1.0,
                train_sharpe=2.0, train_ic=0.1, train_ann_ret=0.2, train_maxdd=-0.1,
                test_sharpe=1.0, test_ic=0.05, test_ann_ret=0.1, test_maxdd=-0.1, test_turnover=0.2,
            ),
            SweepResult(
                max_depth=3, n_estimators=75, reg_lambda=5.0,
                subsample=0.8, colsample=0.8,
                train_sharpe=2.5, train_ic=0.12, train_ann_ret=0.25, train_maxdd=-0.08,
                test_sharpe=1.5, test_ic=0.08, test_ann_ret=0.15, test_maxdd=-0.12, test_turnover=0.25,
            ),
        ]
        
        print_sweep_results_table(results)
        
        captured = capsys.readouterr()
        # The second result (test_sharpe=1.5) should appear before the first (test_sharpe=1.0)
        pos_1_5 = captured.out.find("1.500")
        pos_1_0 = captured.out.find("1.000")
        assert pos_1_5 < pos_1_0  # 1.5 should appear first
    
    def test_shows_top_3_summary(self, capsys):
        """Should show Top 3 Configs section."""
        from run_large_cap_backtest import SweepResult, print_sweep_results_table
        
        results = [
            SweepResult(
                max_depth=2, n_estimators=50, reg_lambda=5.0,
                subsample=0.6, colsample=0.5,
                train_sharpe=2.0, train_ic=0.1, train_ann_ret=0.2, train_maxdd=-0.1,
                test_sharpe=1.2, test_ic=0.06, test_ann_ret=0.12, test_maxdd=-0.1, test_turnover=0.22,
            ),
        ]
        
        print_sweep_results_table(results)
        
        captured = capsys.readouterr()
        assert "Top 3 Configs" in captured.out


# ==============================================================================
# CLI Flag Tests
# ==============================================================================


class TestCLIFlags:
    """Tests for CLI flag integration."""
    
    def test_reg_sweep_flag_exists(self):
        """--reg-sweep-core-simple flag should be defined."""
        from run_large_cap_backtest import parse_args
        import sys
        
        # Test that the flag is recognized
        original_argv = sys.argv
        try:
            sys.argv = ['test', '--use-fundamentals', '--use-core-factors', '--simple-model', '--reg-sweep-core-simple']
            args = parse_args()
            assert args.reg_sweep_core_simple is True
        finally:
            sys.argv = original_argv
    
    def test_reg_sweep_default_false(self):
        """--reg-sweep-core-simple should default to False."""
        from run_large_cap_backtest import parse_args
        import sys
        
        original_argv = sys.argv
        try:
            sys.argv = ['test']
            args = parse_args()
            assert args.reg_sweep_core_simple is False
        finally:
            sys.argv = original_argv


# ==============================================================================
# run_core_simple_reg_sweep Tests
# ==============================================================================


class TestRunCoreSimpleRegSweep:
    """Tests for run_core_simple_reg_sweep function."""
    
    def test_function_exists(self):
        """run_core_simple_reg_sweep should be importable."""
        from run_large_cap_backtest import run_core_simple_reg_sweep
        
        assert callable(run_core_simple_reg_sweep)
    
    def test_function_signature(self):
        """run_core_simple_reg_sweep should have correct parameters."""
        from run_large_cap_backtest import run_core_simple_reg_sweep
        import inspect
        
        sig = inspect.signature(run_core_simple_reg_sweep)
        params = list(sig.parameters.keys())
        
        assert 'prices' in params
        assert 'config' in params
        assert 'sweep_config' in params
        assert 'args' in params
