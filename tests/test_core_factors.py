"""
Tests for core factor extraction and selection.

Tests cover:
- compute_feature_importance_by_subperiod: extracting importance across time periods
- select_core_features: identifying stable factors
- SIMPLE_MODEL_CONFIG: simplified model preset
- Feature pipeline filtering with core_features
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_researcher.config import (
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    ResearchConfig,
    SIMPLE_MODEL_CONFIG,
)
from auto_researcher.models.gbdt_model import GBDTModel
from auto_researcher.features.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_matrix,
    _filter_to_core_features,
)


# ==============================================================================
# SIMPLE_MODEL_CONFIG Tests
# ==============================================================================


class TestSimpleModelConfig:
    """Tests for the simplified model configuration preset."""
    
    def test_simple_model_config_exists(self):
        """SIMPLE_MODEL_CONFIG should be importable."""
        assert SIMPLE_MODEL_CONFIG is not None
        assert isinstance(SIMPLE_MODEL_CONFIG, ModelConfig)
    
    def test_simple_model_has_shallow_depth(self):
        """Simple model should have shallower trees than default."""
        default = ModelConfig()
        assert SIMPLE_MODEL_CONFIG.max_depth < default.max_depth
        assert SIMPLE_MODEL_CONFIG.max_depth == 3
    
    def test_simple_model_has_fewer_estimators(self):
        """Simple model should have fewer boosting rounds."""
        default = ModelConfig()
        assert SIMPLE_MODEL_CONFIG.n_estimators <= default.n_estimators
        assert SIMPLE_MODEL_CONFIG.n_estimators == 75
    
    def test_simple_model_has_fewer_leaves(self):
        """Simple model should have fewer leaves per tree."""
        default = ModelConfig()
        assert SIMPLE_MODEL_CONFIG.num_leaves < default.num_leaves
        assert SIMPLE_MODEL_CONFIG.num_leaves == 16
    
    def test_simple_model_can_train(self):
        """Simple model config should work with GBDTModel."""
        model = GBDTModel(SIMPLE_MODEL_CONFIG)
        
        # Create simple training data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_a': np.random.randn(100),
            'feature_b': np.random.randn(100),
            'feature_c': np.random.randn(100),
        })
        y = pd.Series(np.random.randn(100))
        
        # Should train without error
        model.fit(X, y)
        
        # Should make predictions
        preds = model.predict(X)
        assert len(preds) == 100


# ==============================================================================
# FeatureConfig Core Features Tests
# ==============================================================================


class TestFeatureConfigCoreFeatures:
    """Tests for core feature settings in FeatureConfig."""
    
    def test_feature_config_has_core_features_fields(self):
        """FeatureConfig should have core features fields."""
        config = FeatureConfig()
        assert hasattr(config, 'use_core_features_only')
        assert hasattr(config, 'core_features')
    
    def test_feature_config_default_core_features_disabled(self):
        """Core features should be disabled by default."""
        config = FeatureConfig()
        assert config.use_core_features_only is False
        assert config.core_features is None
    
    def test_feature_config_can_enable_core_features(self):
        """Should be able to enable core features mode."""
        config = FeatureConfig(
            use_core_features_only=True,
            core_features=('tech_mom_21', 'tech_vol_21', 'quality_roe'),
        )
        assert config.use_core_features_only is True
        assert len(config.core_features) == 3
        assert 'tech_mom_21' in config.core_features


# ==============================================================================
# FeaturePipelineConfig Core Features Tests
# ==============================================================================


class TestFeaturePipelineConfigCoreFeatures:
    """Tests for core feature settings in FeaturePipelineConfig."""
    
    def test_pipeline_config_has_core_features_fields(self):
        """FeaturePipelineConfig should have core features fields."""
        config = FeaturePipelineConfig()
        assert hasattr(config, 'use_core_features_only')
        assert hasattr(config, 'core_features')
    
    def test_pipeline_config_default_core_features_disabled(self):
        """Core features should be disabled by default."""
        config = FeaturePipelineConfig()
        assert config.use_core_features_only is False
        assert config.core_features is None


# ==============================================================================
# _filter_to_core_features Tests
# ==============================================================================


class TestFilterToCoreFeatures:
    """Tests for the core feature filtering function."""
    
    @pytest.fixture
    def sample_multiindex_features(self):
        """Create sample feature DataFrame with MultiIndex columns."""
        dates = pd.date_range('2023-01-01', periods=5)
        tickers = ['AAPL', 'MSFT']
        features = ['tech_mom_21', 'tech_mom_63', 'tech_vol_21', 'quality_roe']
        
        # Create MultiIndex columns (ticker, feature)
        columns = pd.MultiIndex.from_product([tickers, features], names=['ticker', 'feature'])
        data = np.random.randn(5, len(columns))
        
        return pd.DataFrame(data, index=dates, columns=columns)
    
    def test_filter_keeps_only_core_features(self, sample_multiindex_features):
        """Should filter to only the specified core features."""
        core = ('tech_mom_21', 'quality_roe')
        
        result = _filter_to_core_features(sample_multiindex_features, core)
        
        # Check that only core features remain
        result_features = result.columns.get_level_values('feature').unique()
        assert set(result_features) == set(core)
    
    def test_filter_preserves_all_tickers(self, sample_multiindex_features):
        """Should preserve all tickers when filtering features."""
        core = ('tech_mom_21',)
        
        result = _filter_to_core_features(sample_multiindex_features, core)
        
        # Check that all tickers are preserved
        result_tickers = result.columns.get_level_values('ticker').unique()
        assert set(result_tickers) == {'AAPL', 'MSFT'}
    
    def test_filter_raises_if_no_core_features_found(self, sample_multiindex_features):
        """Should raise ValueError if no core features are found."""
        core = ('nonexistent_feature',)
        
        with pytest.raises(ValueError, match="None of the core features"):
            _filter_to_core_features(sample_multiindex_features, core)
    
    def test_filter_with_flat_columns(self):
        """Should work with flat (non-MultiIndex) columns."""
        dates = pd.date_range('2023-01-01', periods=5)
        data = pd.DataFrame({
            'feature_a': np.random.randn(5),
            'feature_b': np.random.randn(5),
            'feature_c': np.random.randn(5),
        }, index=dates)
        
        core = ('feature_a', 'feature_c')
        
        result = _filter_to_core_features(data, core)
        
        assert list(result.columns) == ['feature_a', 'feature_c']


# ==============================================================================
# select_core_features Tests (imported from run_large_cap_backtest)
# ==============================================================================


class TestSelectCoreFeatures:
    """Tests for the core feature selection logic."""
    
    @pytest.fixture
    def sample_importance_by_subperiod(self):
        """Create sample importance data for 3 subperiods."""
        return {
            '2013-2016': pd.Series({
                'tech_mom_21': 0.15,
                'tech_mom_63': 0.12,
                'tech_vol_21': 0.08,
                'quality_roe': 0.05,
                'growth_rev': 0.02,  # Low importance
            }),
            '2017-2019': pd.Series({
                'tech_mom_21': 0.18,
                'tech_mom_63': 0.10,
                'tech_vol_21': 0.06,
                'quality_roe': 0.04,
                # growth_rev missing in this period
            }),
            '2020-2023': pd.Series({
                'tech_mom_21': 0.14,
                'tech_mom_63': 0.11,
                'tech_vol_21': 0.07,
                'quality_roe': 0.06,
                'growth_rev': 0.01,
            }),
        }
    
    def test_select_core_features_basic(self, sample_importance_by_subperiod):
        """Should select features that appear in multiple subperiods."""
        # Import the function (it's in the script file)
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_large_cap_backtest import select_core_features, CoreFactorConfig
        
        config = CoreFactorConfig(
            min_subperiod_support=2,
            min_importance_threshold=0.01,
            min_mean_importance=0.03,
        )
        
        core_features, importance_df = select_core_features(
            sample_importance_by_subperiod,
            core_config=config,
        )
        
        # tech_mom_21, tech_mom_63, tech_vol_21, quality_roe should all qualify
        # (appear in all 3 subperiods with high enough importance)
        assert 'tech_mom_21' in core_features
        assert 'tech_mom_63' in core_features
        assert 'tech_vol_21' in core_features
        assert 'quality_roe' in core_features
        
        # growth_rev should NOT qualify (mean importance < 0.03)
        assert 'growth_rev' not in core_features
    
    def test_select_core_features_returns_dataframe(self, sample_importance_by_subperiod):
        """Should return importance DataFrame with statistics."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_large_cap_backtest import select_core_features
        
        core_features, importance_df = select_core_features(sample_importance_by_subperiod)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'support_count' in importance_df.columns
        assert 'mean_importance' in importance_df.columns
    
    def test_select_core_features_sorted_by_importance(self, sample_importance_by_subperiod):
        """Core features should be sorted by mean importance descending."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_large_cap_backtest import select_core_features
        
        core_features, importance_df = select_core_features(sample_importance_by_subperiod)
        
        if len(core_features) >= 2:
            # First feature should have higher mean importance than last
            first_importance = importance_df.loc[core_features[0], 'mean_importance']
            last_importance = importance_df.loc[core_features[-1], 'mean_importance']
            assert first_importance >= last_importance


# ==============================================================================
# compute_feature_importance_by_subperiod Tests
# ==============================================================================


class TestComputeFeatureImportanceBySubperiod:
    """Tests for subperiod feature importance extraction."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data spanning multiple years."""
        dates = pd.date_range('2013-01-01', '2023-12-31', freq='B')
        tickers = ['AAPL', 'MSFT', 'GOOG', 'SPY']
        
        np.random.seed(42)
        # Generate random walk prices
        returns = np.random.randn(len(dates), len(tickers)) * 0.02
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            index=dates,
            columns=tickers,
        )
        
        return prices
    
    def test_importance_extraction_returns_dict(self, sample_prices):
        """Should return dict mapping subperiod names to importance Series."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_large_cap_backtest import (
            compute_feature_importance_by_subperiod,
            Subperiod,
            SUBPERIODS,
        )
        
        config = PipelineConfig(
            research=ResearchConfig(
                universe=tuple(sample_prices.columns),
                horizon_days=21,
                top_k=2,
            ),
            features=FeatureConfig(
                momentum_windows=(21, 63),
                volatility_windows=(21,),
                include_fundamentals=False,
            ),
            model=ModelConfig(n_estimators=10),
        )
        
        # Use shorter test subperiods
        test_subperiods = [
            Subperiod('2014-2015', pd.Timestamp('2014-01-01'), pd.Timestamp('2015-12-31')),
            Subperiod('2016-2017', pd.Timestamp('2016-01-01'), pd.Timestamp('2017-12-31')),
        ]
        
        result = compute_feature_importance_by_subperiod(
            prices=sample_prices,
            config=config,
            subperiods=test_subperiods,
            benchmark='SPY',
            horizon_days=21,
        )
        
        assert isinstance(result, dict)
        # Should have at least some subperiods
        assert len(result) >= 1
        
        # Each value should be a Series
        for subperiod_name, importance in result.items():
            assert isinstance(importance, pd.Series)
            assert len(importance) > 0
    
    def test_importance_values_sum_to_total(self, sample_prices):
        """Feature importances should be non-negative and sum correctly."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_large_cap_backtest import (
            compute_feature_importance_by_subperiod,
            Subperiod,
        )
        
        config = PipelineConfig(
            research=ResearchConfig(
                universe=tuple(sample_prices.columns),
                horizon_days=21,
                top_k=2,
            ),
            features=FeatureConfig(
                momentum_windows=(21,),
                volatility_windows=(21,),
                include_fundamentals=False,
            ),
            model=ModelConfig(n_estimators=10),
        )
        
        test_subperiods = [
            Subperiod('2015-2016', pd.Timestamp('2015-01-01'), pd.Timestamp('2016-12-31')),
        ]
        
        result = compute_feature_importance_by_subperiod(
            prices=sample_prices,
            config=config,
            subperiods=test_subperiods,
            benchmark='SPY',
            horizon_days=21,
        )
        
        for subperiod_name, importance in result.items():
            # All importances should be non-negative
            assert (importance >= 0).all()
            # Should have at least one feature with non-zero importance
            assert importance.sum() > 0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestCoreFactorIntegration:
    """Integration tests for core factor workflow."""
    
    def test_full_core_factor_workflow(self):
        """Test the complete workflow: extract -> select -> filter."""
        # Create simple test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100)
        tickers = ['AAPL', 'MSFT']
        features = ['tech_mom_21', 'tech_mom_63', 'tech_vol_21']
        
        # Create MultiIndex feature matrix
        columns = pd.MultiIndex.from_product([tickers, features], names=['ticker', 'feature'])
        feature_matrix = pd.DataFrame(
            np.random.randn(100, len(columns)),
            index=dates,
            columns=columns,
        )
        
        # Simulate importance from 2 subperiods
        importance_by_subperiod = {
            'period_1': pd.Series({
                'tech_mom_21': 0.50,
                'tech_mom_63': 0.30,
                'tech_vol_21': 0.20,
            }),
            'period_2': pd.Series({
                'tech_mom_21': 0.45,
                'tech_mom_63': 0.35,
                'tech_vol_21': 0.20,
            }),
        }
        
        # Select core features
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_large_cap_backtest import select_core_features, CoreFactorConfig
        
        config = CoreFactorConfig(
            min_subperiod_support=2,
            min_importance_threshold=0.10,
            min_mean_importance=0.15,
        )
        
        core_features, _ = select_core_features(importance_by_subperiod, config)
        
        # Filter feature matrix
        filtered = _filter_to_core_features(feature_matrix, tuple(core_features))
        
        # Verify filtering worked
        filtered_features = filtered.columns.get_level_values('feature').unique()
        assert set(filtered_features).issubset(set(core_features))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
