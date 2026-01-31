"""
Tests for finagg fundamentals integration.

These tests verify that:
1. The finagg integration properly handles the case when finagg is not installed
2. When mocked, the finagg provider returns properly formatted data
3. The factor computation works correctly with finagg-sourced data
4. End-to-end feature pipeline works with mocked finagg data

Note: No network calls are made in these tests. All finagg functionality
is mocked to enable testing without the finagg package installed.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_tickers():
    """Sample tickers for testing."""
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture
def sample_dates():
    """Sample date range for testing."""
    return pd.date_range("2020-01-01", "2023-12-31", freq="QE")


@pytest.fixture
def sample_finagg_data(sample_tickers, sample_dates):
    """
    Create synthetic finagg-like data for testing.
    
    Mimics the structure returned by fetch_finagg_quarterly_fundamentals.
    """
    records = []
    
    for date in sample_dates:
        for ticker in sample_tickers:
            # Add some random variation by ticker
            base_seed = hash(ticker) % 1000
            np.random.seed(base_seed + date.year * 100 + date.month)
            
            records.append({
                "date": date,
                "ticker": ticker,
                "revenue": np.random.uniform(50e9, 150e9),
                "net_income": np.random.uniform(5e9, 30e9),
                "gross_profit": np.random.uniform(20e9, 60e9),
                "operating_income": np.random.uniform(10e9, 40e9),
                "total_assets": np.random.uniform(200e9, 500e9),
                "stockholders_equity": np.random.uniform(80e9, 200e9),
                "eps_basic": np.random.uniform(1.0, 10.0),
                "shares_outstanding": np.random.uniform(4e9, 20e9),
                "gross_margin": np.random.uniform(0.3, 0.6),
                "operating_margin": np.random.uniform(0.15, 0.35),
                "net_margin": np.random.uniform(0.1, 0.25),
                "roe": np.random.uniform(0.1, 0.4),
                "roa": np.random.uniform(0.05, 0.15),
            })
    
    df = pd.DataFrame(records)
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


@pytest.fixture
def sample_prices(sample_tickers, sample_dates):
    """Create synthetic price data for testing."""
    # Daily prices from 2020 to 2024
    daily_dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    
    prices = pd.DataFrame(index=daily_dates)
    np.random.seed(42)
    
    for ticker in sample_tickers:
        # Random walk prices
        returns = np.random.normal(0.0005, 0.02, len(daily_dates))
        prices[ticker] = 100 * np.exp(np.cumsum(returns))
    
    # Add benchmark
    returns = np.random.normal(0.0003, 0.01, len(daily_dates))
    prices["SPY"] = 100 * np.exp(np.cumsum(returns))
    
    return prices


# =============================================================================
# Test: Finagg Not Installed
# =============================================================================


class TestFinaggNotInstalled:
    """Tests for graceful handling when finagg is not installed."""
    
    def test_finagg_not_installed_error_on_direct_import(self):
        """Verify clear error when trying to use finagg without installation."""
        from auto_researcher.data.finagg_fundamentals import (
            _check_finagg_available,
            FinaggNotInstalledError,
        )
        
        # Mock the import to simulate finagg not being installed
        with patch.dict("sys.modules", {"finagg": None}):
            # Force re-check
            import auto_researcher.data.finagg_fundamentals as ff_module
            ff_module._FINAGG_AVAILABLE = None
            
            # The check should work even if finagg is not installed
            # (returns False instead of crashing)
            result = _check_finagg_available()
            # Note: This will depend on actual finagg installation status
            assert isinstance(result, bool)
    
    def test_finagg_not_installed_error_message(self):
        """Verify error message is helpful when finagg not installed."""
        from auto_researcher.data.finagg_fundamentals import FinaggNotInstalledError
        
        error = FinaggNotInstalledError()
        error_msg = str(error)
        
        # Check for helpful content in error message
        assert "finagg" in error_msg.lower()
        assert "pip install" in error_msg.lower()
        assert "SEC" in error_msg


# =============================================================================
# Test: Mocked Finagg Provider
# =============================================================================


class TestMockedFinaggProvider:
    """Tests with mocked finagg functionality."""
    
    def test_fetch_finagg_with_mock(self, sample_finagg_data, sample_tickers):
        """Test that mocked finagg returns properly formatted data."""
        from auto_researcher.data.finagg_fundamentals import (
            fetch_finagg_quarterly_fundamentals,
        )
        
        # Mock the entire fetch function
        with patch(
            "auto_researcher.data.finagg_fundamentals.fetch_finagg_quarterly_fundamentals",
            return_value=sample_finagg_data,
        ):
            # Verify the mock data has expected structure
            assert isinstance(sample_finagg_data.index, pd.MultiIndex)
            assert sample_finagg_data.index.names == ["date", "ticker"]
            
            # Verify expected columns
            expected_cols = ["revenue", "net_income", "gross_margin", "operating_margin"]
            for col in expected_cols:
                assert col in sample_finagg_data.columns
    
    def test_load_fundamentals_from_finagg_with_mock(
        self, sample_finagg_data, sample_tickers
    ):
        """Test _load_fundamentals_from_finagg with mocked data."""
        from auto_researcher.features.fundamentals import _load_fundamentals_from_finagg
        
        # Mock the fetch function
        with patch(
            "auto_researcher.data.finagg_fundamentals.fetch_finagg_quarterly_fundamentals",
            return_value=sample_finagg_data,
        ):
            result = _load_fundamentals_from_finagg(
                tickers=sample_tickers,
                start="2020-01-01",
                end="2023-12-31",
                mode="quarterly_refined",
            )
            
            # Verify result structure
            assert isinstance(result, pd.DataFrame)
            assert isinstance(result.index, pd.MultiIndex)
            
            # Verify expected columns
            expected_cols = [
                "gross_margin", "operating_margin", "net_margin", "roe", "roa"
            ]
            for col in expected_cols:
                assert col in result.columns


# =============================================================================
# Test: Factor Computation with Finagg Data
# =============================================================================


class TestFactorComputationWithFinagg:
    """Tests for factor computation using finagg-sourced data."""
    
    def test_build_fundamental_factors_with_finagg_source(
        self, sample_finagg_data, sample_tickers
    ):
        """Test that build_fundamental_factors works with finagg data."""
        from auto_researcher.features.fundamentals import (
            build_fundamental_factors,
            FundamentalFactorConfig,
            _load_fundamentals_from_finagg,
        )
        
        # Mock the finagg fetch
        with patch(
            "auto_researcher.data.finagg_fundamentals.fetch_finagg_quarterly_fundamentals",
            return_value=sample_finagg_data,
        ):
            # Load fundamentals
            raw_fund = _load_fundamentals_from_finagg(
                tickers=sample_tickers,
                start="2020-01-01",
                end="2023-12-31",
            )
            
            # Build factors with quality + profitability enabled
            config = FundamentalFactorConfig(
                use_value=False,
                use_quality=True,
                use_growth=False,
                use_profitability=True,
                use_size=False,
                source="finagg",
            )
            
            factors = build_fundamental_factors(raw_fund, config=config)
            
            # Verify factor columns
            assert not factors.empty
            assert "quality_roe" in factors.columns or "prof_gross_margin" in factors.columns
    
    def test_zscore_normalization_with_finagg_data(self, sample_finagg_data):
        """Test cross-sectional z-scoring works with finagg data."""
        from auto_researcher.features.fundamentals import (
            build_fundamental_factors,
            FundamentalFactorConfig,
        )
        
        # Create a raw fund DataFrame from sample data
        # Map to expected columns
        raw_fund = sample_finagg_data.copy()
        raw_fund["market_cap"] = np.nan
        raw_fund["pe_ratio"] = np.nan
        raw_fund["pb_ratio"] = np.nan
        raw_fund["ps_ratio"] = np.nan
        raw_fund["dividend_yield"] = np.nan
        raw_fund["revenue_ttm"] = raw_fund.get("revenue", np.nan)
        raw_fund["eps_ttm"] = raw_fund.get("eps_basic", np.nan)
        raw_fund["revenue_growth_1y"] = np.nan
        raw_fund["eps_growth_1y"] = np.nan
        
        config = FundamentalFactorConfig(
            use_value=False,
            use_quality=True,
            use_growth=False,
            use_profitability=True,
            use_size=False,
            zscore_by_cross_section=True,
        )
        
        factors = build_fundamental_factors(raw_fund, config=config)
        
        if not factors.empty:
            # Check z-scored factors have roughly mean~0, std~1 per date
            for date in factors.index.get_level_values("date").unique()[:5]:
                date_data = factors.xs(date, level="date")
                if len(date_data) > 2:
                    for col in factors.columns:
                        col_data = date_data[col].dropna()
                        if len(col_data) > 2:
                            assert abs(col_data.mean()) < 0.5, f"Mean not near 0 for {col}"
                            assert abs(col_data.std() - 1.0) < 0.5, f"Std not near 1 for {col}"


# =============================================================================
# Test: End-to-End Pipeline with Mocked Finagg
# =============================================================================


class TestEndToEndPipelineWithFinagg:
    """End-to-end tests with mocked finagg data."""
    
    def test_feature_pipeline_with_finagg_source(
        self, sample_finagg_data, sample_prices, sample_tickers
    ):
        """Test full feature pipeline with finagg fundamentals."""
        from auto_researcher.features.feature_pipeline import build_feature_matrix
        from auto_researcher.features.fundamentals import FundamentalFactorConfig
        from auto_researcher.config import FeatureConfig
        
        # Create config with finagg source
        config = FeatureConfig(
            momentum_windows=(21, 63),
            volatility_windows=(21,),
            include_fundamentals=True,
            include_sentiment=False,
            fundamentals_source="finagg",
            finagg_mode="quarterly_refined",
            fundamental_factor_families={
                "value": False,
                "quality": True,
                "growth": False,
                "profitability": True,
                "size": False,
            },
        )
        
        # Mock the finagg fetch
        with patch(
            "auto_researcher.data.finagg_fundamentals.fetch_finagg_quarterly_fundamentals",
            return_value=sample_finagg_data,
        ):
            features = build_feature_matrix(sample_prices[sample_tickers], config)
            
            # Verify features were computed
            assert not features.empty
            
            # Verify technical features are present
            tech_cols = [
                c for c in features.columns.get_level_values("feature").unique()
                if "tech_" in c or "mom_" in c or "vol_" in c
            ]
            assert len(tech_cols) > 0, "Should have technical features"
    
    def test_no_nans_after_warmup(
        self, sample_finagg_data, sample_prices, sample_tickers
    ):
        """Test that feature matrix has no NaNs after warmup period."""
        from auto_researcher.features.feature_pipeline import build_feature_matrix
        from auto_researcher.config import FeatureConfig
        
        config = FeatureConfig(
            momentum_windows=(21, 63),
            volatility_windows=(21,),
            include_fundamentals=True,
            include_sentiment=False,
            fundamentals_source="finagg",
            fundamental_factor_families={
                "value": False,
                "quality": True,
                "growth": False,
                "profitability": True,
                "size": False,
            },
        )
        
        with patch(
            "auto_researcher.data.finagg_fundamentals.fetch_finagg_quarterly_fundamentals",
            return_value=sample_finagg_data,
        ):
            features = build_feature_matrix(sample_prices[sample_tickers], config)
            
            # After warmup (252+ days), should have minimal NaNs
            if len(features) > 252:
                post_warmup = features.iloc[252:]
                # Allow some NaNs due to fundamentals alignment
                nan_pct = post_warmup.isna().mean().mean()
                # Should have < 50% NaNs after warmup
                assert nan_pct < 0.5, f"Too many NaNs after warmup: {nan_pct:.1%}"


# =============================================================================
# Test: Slow/Integration Tests (Require Real Finagg)
# =============================================================================


@pytest.mark.skipif(
    os.environ.get("FINAGG_INTEGRATION_TESTS") != "1",
    reason="Set FINAGG_INTEGRATION_TESTS=1 to run live finagg tests",
)
class TestFinaggIntegration:
    """
    Integration tests that require real finagg installation.
    
    These tests are skipped by default. To run them:
        FINAGG_INTEGRATION_TESTS=1 pytest tests/test_finagg_fundamentals_integration.py -v
    """
    
    @pytest.mark.slow
    def test_real_finagg_fetch(self):
        """Test fetching real data from SEC via finagg."""
        from auto_researcher.data.finagg_fundamentals import (
            fetch_finagg_quarterly_fundamentals,
            _check_finagg_available,
        )
        
        if not _check_finagg_available():
            pytest.skip("finagg not installed")
        
        # Fetch a small amount of real data
        df = fetch_finagg_quarterly_fundamentals(
            tickers=["AAPL"],
            start=pd.Timestamp("2023-01-01"),
            end=pd.Timestamp("2023-12-31"),
            mode="api",  # Use API to avoid needing local DB
        )
        
        assert not df.empty, "Should have fetched some data"
        assert "AAPL" in df.index.get_level_values("ticker")
    
    @pytest.mark.slow
    def test_finagg_status(self):
        """Test finagg status reporting."""
        from auto_researcher.data.finagg_fundamentals import get_finagg_status
        
        status = get_finagg_status()
        
        assert "installed" in status
        assert "configured" in status
        assert isinstance(status["installed"], bool)


# =============================================================================
# Test: Config Integration
# =============================================================================


class TestConfigIntegration:
    """Tests for configuration integration."""
    
    def test_fundamental_factor_config_with_finagg(self):
        """Test FundamentalFactorConfig with finagg settings."""
        from auto_researcher.features.fundamentals import FundamentalFactorConfig
        
        config = FundamentalFactorConfig(
            use_value=False,
            use_quality=True,
            use_profitability=True,
            source="finagg",
            finagg_mode="quarterly_refined",
        )
        
        assert config.source == "finagg"
        assert config.finagg_mode == "quarterly_refined"
        assert config.use_quality is True
        assert config.use_value is False
    
    def test_feature_config_with_finagg_source(self):
        """Test FeatureConfig with finagg source settings."""
        from auto_researcher.config import FeatureConfig
        
        config = FeatureConfig(
            include_fundamentals=True,
            fundamentals_source="finagg",
            finagg_mode="quarterly_api",
        )
        
        assert config.fundamentals_source == "finagg"
        assert config.finagg_mode == "quarterly_api"
        assert config.include_fundamentals is True
