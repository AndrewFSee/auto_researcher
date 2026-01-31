"""Tests for multi-split OOS robustness suite functionality."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import from scripts after path setup
from run_large_cap_backtest import (
    OOSRobustnessSplit,
    CoreOOSRobustnessConfig,
    DEFAULT_CORE_OOS_SPLITS,
    BEST_CORE_SIMPLE_PARAMS,
    OOSSplitResult,
)


class TestOOSRobustnessSplit:
    """Tests for OOSRobustnessSplit dataclass."""

    def test_split_has_required_fields(self):
        """OOSRobustnessSplit should have train/test start/end fields."""
        split = OOSRobustnessSplit(
            name="test_split",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
        )
        assert split.name == "test_split"
        assert split.train_start == "2013-01-01"
        assert split.train_end == "2018-12-31"
        assert split.test_start == "2019-01-01"
        assert split.test_end == "2023-12-31"

    def test_split_train_start_timestamp(self):
        """train_start_ts property should return pandas Timestamp."""
        split = OOSRobustnessSplit(
            name="test_split",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
        )
        ts = split.train_start_ts
        assert ts.year == 2013
        assert ts.month == 1
        assert ts.day == 1

    def test_split_train_end_timestamp(self):
        """train_end_ts property should return pandas Timestamp."""
        split = OOSRobustnessSplit(
            name="test_split",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
        )
        ts = split.train_end_ts
        assert ts.year == 2018
        assert ts.month == 12
        assert ts.day == 31

    def test_split_test_start_timestamp(self):
        """test_start_ts property should return pandas Timestamp."""
        split = OOSRobustnessSplit(
            name="test_split",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
        )
        ts = split.test_start_ts
        assert ts.year == 2019
        assert ts.month == 1
        assert ts.day == 1

    def test_split_test_end_timestamp(self):
        """test_end_ts property should return pandas Timestamp."""
        split = OOSRobustnessSplit(
            name="test_split",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
        )
        ts = split.test_end_ts
        assert ts.year == 2023
        assert ts.month == 12
        assert ts.day == 31


class TestCoreOOSRobustnessConfig:
    """Tests for CoreOOSRobustnessConfig dataclass."""

    def test_config_has_enable_flag(self):
        """Config should have enable flag."""
        config = CoreOOSRobustnessConfig(enable=True, splits=[])
        assert config.enable is True

    def test_config_has_splits_list(self):
        """Config should have splits list."""
        split = OOSRobustnessSplit(
            name="test",
            train_start="2013-01-01",
            train_end="2016-12-31",
            test_start="2017-01-01",
            test_end="2019-12-31",
        )
        config = CoreOOSRobustnessConfig(enable=True, splits=[split])
        assert len(config.splits) == 1
        assert config.splits[0].name == "test"

    def test_config_default_splits_empty(self):
        """Default splits should be empty list."""
        config = CoreOOSRobustnessConfig(enable=False)
        assert config.splits == []


class TestDefaultCoreOOSSplits:
    """Tests for DEFAULT_CORE_OOS_SPLITS constant."""

    def test_default_splits_is_list(self):
        """DEFAULT_CORE_OOS_SPLITS should be a list."""
        assert isinstance(DEFAULT_CORE_OOS_SPLITS, list)

    def test_default_splits_has_three_entries(self):
        """DEFAULT_CORE_OOS_SPLITS should have 3 splits."""
        assert len(DEFAULT_CORE_OOS_SPLITS) == 3

    def test_split_1_train_2013_2016_test_2017_2019(self):
        """First split trains 2013-2016, tests 2017-2019."""
        split = DEFAULT_CORE_OOS_SPLITS[0]
        assert "2013" in split.train_start
        assert "2016" in split.train_end
        assert "2017" in split.test_start
        assert "2019" in split.test_end

    def test_split_2_train_2017_2019_test_2020_2023(self):
        """Second split trains 2017-2019, tests 2020-2023."""
        split = DEFAULT_CORE_OOS_SPLITS[1]
        assert "2017" in split.train_start
        assert "2019" in split.train_end
        assert "2020" in split.test_start
        assert "2023" in split.test_end

    def test_split_3_train_2013_2019_test_2020_2023(self):
        """Third split trains 2013-2019, tests 2020-2023."""
        split = DEFAULT_CORE_OOS_SPLITS[2]
        assert "2013" in split.train_start
        assert "2019" in split.train_end
        assert "2020" in split.test_start
        assert "2023" in split.test_end

    def test_all_splits_have_names(self):
        """All splits should have descriptive names."""
        for split in DEFAULT_CORE_OOS_SPLITS:
            assert split.name is not None
            assert len(split.name) > 0


class TestBestCoreSimpleParams:
    """Tests for BEST_CORE_SIMPLE_PARAMS constant."""

    def test_best_params_is_dict(self):
        """BEST_CORE_SIMPLE_PARAMS should be a dict."""
        assert isinstance(BEST_CORE_SIMPLE_PARAMS, dict)

    def test_best_params_has_reg_lambda(self):
        """Best params should include reg_lambda."""
        assert "reg_lambda" in BEST_CORE_SIMPLE_PARAMS

    def test_best_params_has_subsample(self):
        """Best params should include subsample."""
        assert "subsample" in BEST_CORE_SIMPLE_PARAMS

    def test_best_params_has_colsample_bytree(self):
        """Best params should include colsample_bytree."""
        assert "colsample_bytree" in BEST_CORE_SIMPLE_PARAMS

    def test_reg_lambda_non_negative(self):
        """reg_lambda should be non-negative."""
        assert BEST_CORE_SIMPLE_PARAMS["reg_lambda"] >= 0

    def test_subsample_between_0_and_1(self):
        """subsample should be between 0 and 1."""
        val = BEST_CORE_SIMPLE_PARAMS["subsample"]
        assert 0 < val <= 1.0

    def test_colsample_bytree_between_0_and_1(self):
        """colsample_bytree should be between 0 and 1."""
        val = BEST_CORE_SIMPLE_PARAMS["colsample_bytree"]
        assert 0 < val <= 1.0


class TestOOSSplitResult:
    """Tests for OOSSplitResult dataclass."""

    def test_result_has_split_name(self):
        """OOSSplitResult should have split_name field."""
        result = OOSSplitResult(
            split_name="test_split",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
            core_sharpe_train=1.2,
            core_sharpe_test=0.8,
            core_ic_test=0.05,
            core_ann_ret_test=0.12,
            core_maxdd_test=-0.15,
            core_hit_test=0.55,
            core_turnover_test=0.5,
            ew_sharpe_test=0.6,
            ew_ann_ret_test=0.10,
        )
        assert result.split_name == "test_split"

    def test_result_has_train_dates(self):
        """OOSSplitResult should have train date fields."""
        result = OOSSplitResult(
            split_name="test_split",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
            core_sharpe_train=1.2,
            core_sharpe_test=0.8,
            core_ic_test=0.05,
            core_ann_ret_test=0.12,
            core_maxdd_test=-0.15,
            core_hit_test=0.55,
            core_turnover_test=0.5,
            ew_sharpe_test=0.6,
            ew_ann_ret_test=0.10,
        )
        assert result.train_start == "2013-01-01"
        assert result.train_end == "2018-12-31"

    def test_result_has_test_dates(self):
        """OOSSplitResult should have test date fields."""
        result = OOSSplitResult(
            split_name="test_split",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
            core_sharpe_train=1.2,
            core_sharpe_test=0.8,
            core_ic_test=0.05,
            core_ann_ret_test=0.12,
            core_maxdd_test=-0.15,
            core_hit_test=0.55,
            core_turnover_test=0.5,
            ew_sharpe_test=0.6,
            ew_ann_ret_test=0.10,
        )
        assert result.test_start == "2019-01-01"
        assert result.test_end == "2023-12-31"

    def test_result_has_core_metrics(self):
        """OOSSplitResult should have core model metrics."""
        result = OOSSplitResult(
            split_name="test",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
            core_sharpe_train=1.2,
            core_sharpe_test=0.8,
            core_ic_test=0.05,
            core_ann_ret_test=0.12,
            core_maxdd_test=-0.15,
            core_hit_test=0.55,
            core_turnover_test=0.5,
            ew_sharpe_test=0.6,
            ew_ann_ret_test=0.10,
        )
        assert result.core_sharpe_train == 1.2
        assert result.core_sharpe_test == 0.8
        assert result.core_ic_test == 0.05
        assert result.core_ann_ret_test == 0.12
        assert result.core_maxdd_test == -0.15
        assert result.core_hit_test == 0.55
        assert result.core_turnover_test == 0.5

    def test_result_has_ew_metrics(self):
        """OOSSplitResult should have EW baseline metrics."""
        result = OOSSplitResult(
            split_name="test",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
            core_sharpe_train=1.2,
            core_sharpe_test=0.8,
            core_ic_test=0.05,
            core_ann_ret_test=0.12,
            core_maxdd_test=-0.15,
            core_hit_test=0.55,
            core_turnover_test=0.5,
            ew_sharpe_test=0.6,
            ew_ann_ret_test=0.10,
        )
        assert result.ew_sharpe_test == 0.6
        assert result.ew_ann_ret_test == 0.10

    def test_result_allows_none_ic(self):
        """OOSSplitResult should allow None for IC."""
        result = OOSSplitResult(
            split_name="test",
            train_start="2013-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2023-12-31",
            core_sharpe_train=1.2,
            core_sharpe_test=0.8,
            core_ic_test=None,
            core_ann_ret_test=0.12,
            core_maxdd_test=-0.15,
            core_hit_test=0.55,
            core_turnover_test=0.5,
            ew_sharpe_test=0.6,
            ew_ann_ret_test=0.10,
        )
        assert result.core_ic_test is None


class TestCoreOOSSuiteCLIFlag:
    """Tests for --core-oos-suite CLI flag."""

    def test_core_oos_suite_flag_exists(self):
        """--core-oos-suite flag should be recognized in argparser."""
        import argparse
        # Import the module to get the actual parser
        import run_large_cap_backtest as script
        
        # Access the parser via parse_args internals
        with patch('sys.argv', [
            'script.py',
            '--core-oos-suite',
            '--use-fundamentals',
            '--use-core-factors',
            '--simple-model',
            '--fundamentals-csv', 'data/fundamentals.csv',
        ]):
            args = script.parse_args()
            assert args.core_oos_suite is True

    def test_core_oos_suite_default_false(self):
        """--core-oos-suite should default to False."""
        import run_large_cap_backtest as script
        
        with patch('sys.argv', ['script.py']):
            args = script.parse_args()
            assert getattr(args, 'core_oos_suite', False) is False

    def test_core_oos_suite_with_required_flags(self):
        """--core-oos-suite should work with required flags."""
        import run_large_cap_backtest as script
        
        with patch('sys.argv', [
            'script.py',
            '--core-oos-suite',
            '--use-fundamentals',
            '--use-core-factors',
            '--simple-model',
            '--fundamentals-csv', 'data/fundamentals.csv',
        ]):
            args = script.parse_args()
            assert args.core_oos_suite is True
            assert args.use_fundamentals is True
            assert args.use_core_factors is True
            assert args.simple_model is True
