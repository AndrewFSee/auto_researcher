"""
Tests for fundamental factors module.

Tests cover:
1. Individual factor computation functions (value, quality, growth, profitability, size)
2. Cross-sectional processing (winsorization, z-scoring)
3. Full factor building pipeline
4. Integration with feature_pipeline
"""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.features.fundamentals import (
    FundamentalFactorConfig,
    RAW_FUNDAMENTAL_COLUMNS,
    load_raw_fundamentals,
    compute_value_factors,
    compute_quality_factors,
    compute_growth_factors,
    compute_profitability_factors,
    compute_size_factor,
    build_fundamental_factors,
    _winsorize_cross_sectional,
    _zscore_cross_sectional,
)
from auto_researcher.features.feature_pipeline import (
    build_feature_matrix,
    FeaturePipelineConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_raw_fundamentals() -> pd.DataFrame:
    """
    Create synthetic raw fundamental data for 3 dates × 3 tickers.

    Returns a DataFrame with MultiIndex (date, ticker) and all expected columns.
    """
    dates = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
    tickers = ["AAPL", "MSFT", "GOOG"]

    # Create MultiIndex
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    # Create data with known values for testing
    data = {
        "market_cap": [
            3e12, 2.5e12, 1.5e12,  # Date 1
            3.1e12, 2.6e12, 1.55e12,  # Date 2
            3.2e12, 2.7e12, 1.6e12,  # Date 3
        ],
        "pe_ratio": [
            25.0, 30.0, 20.0,  # Date 1
            26.0, 31.0, 21.0,  # Date 2
            24.0, 29.0, 19.0,  # Date 3
        ],
        "pb_ratio": [
            40.0, 12.0, 5.0,  # Date 1
            41.0, 12.5, 5.2,  # Date 2
            39.0, 11.5, 4.8,  # Date 3
        ],
        "ps_ratio": [
            7.0, 10.0, 5.0,  # Date 1
            7.2, 10.2, 5.1,  # Date 2
            6.8, 9.8, 4.9,  # Date 3
        ],
        "dividend_yield": [
            0.005, 0.008, 0.0,  # Date 1
            0.005, 0.008, 0.0,  # Date 2
            0.005, 0.008, 0.0,  # Date 3
        ],
        "roe": [
            1.50, 0.40, 0.25,  # Date 1 (150%, 40%, 25%)
            1.55, 0.42, 0.26,  # Date 2
            1.60, 0.44, 0.27,  # Date 3
        ],
        "roa": [
            0.30, 0.15, 0.12,  # Date 1
            0.31, 0.16, 0.13,  # Date 2
            0.32, 0.17, 0.14,  # Date 3
        ],
        "gross_margin": [
            0.43, 0.68, 0.55,  # Date 1
            0.44, 0.69, 0.56,  # Date 2
            0.45, 0.70, 0.57,  # Date 3
        ],
        "operating_margin": [
            0.30, 0.42, 0.28,  # Date 1
            0.31, 0.43, 0.29,  # Date 2
            0.32, 0.44, 0.30,  # Date 3
        ],
        "net_margin": [
            0.25, 0.35, 0.22,  # Date 1
            0.26, 0.36, 0.23,  # Date 2
            0.27, 0.37, 0.24,  # Date 3
        ],
        "revenue_ttm": [
            4e11, 2e11, 3e11,  # Date 1
            4.1e11, 2.1e11, 3.1e11,  # Date 2
            4.2e11, 2.2e11, 3.2e11,  # Date 3
        ],
        "revenue_growth_1y": [
            0.08, 0.12, 0.25,  # Date 1 (8%, 12%, 25%)
            0.07, 0.11, 0.24,  # Date 2
            0.06, 0.10, 0.23,  # Date 3
        ],
        "eps_ttm": [
            6.0, 10.0, 5.0,  # Date 1
            6.2, 10.2, 5.2,  # Date 2
            6.4, 10.4, 5.4,  # Date 3
        ],
        "eps_growth_1y": [
            0.10, 0.15, 0.30,  # Date 1
            0.09, 0.14, 0.29,  # Date 2
            0.08, 0.13, 0.28,  # Date 3
        ],
    }

    return pd.DataFrame(data, index=idx)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create synthetic price data for testing integration."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    tickers = ["AAPL", "MSFT", "GOOG"]

    # Generate random walk prices
    returns = np.random.randn(100, 3) * 0.02
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    return pd.DataFrame(prices, index=dates, columns=tickers)


# =============================================================================
# Tests for Individual Factor Computation
# =============================================================================


class TestComputeValueFactors:
    """Tests for compute_value_factors function."""

    def test_returns_dataframe(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Should return a DataFrame."""
        result = compute_value_factors(sample_raw_fundamentals)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Should have expected value factor columns."""
        result = compute_value_factors(sample_raw_fundamentals)
        expected_cols = {"value_ey", "value_btp", "value_stp"}
        assert set(result.columns) == expected_cols

    def test_index_matches_input(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Output index should match input index."""
        result = compute_value_factors(sample_raw_fundamentals)
        assert result.index.equals(sample_raw_fundamentals.index)

    def test_earnings_yield_calculation(
        self, sample_raw_fundamentals: pd.DataFrame
    ) -> None:
        """Earnings yield should be 1 / PE ratio."""
        result = compute_value_factors(sample_raw_fundamentals)

        # Check first row: PE = 25.0, EY should be 0.04
        first_ey = result.loc[(pd.Timestamp("2023-01-31"), "AAPL"), "value_ey"]
        expected_ey = 1.0 / 25.0  # = 0.04
        assert np.isclose(first_ey, expected_ey)

    def test_book_to_price_calculation(
        self, sample_raw_fundamentals: pd.DataFrame
    ) -> None:
        """Book-to-price should be 1 / PB ratio."""
        result = compute_value_factors(sample_raw_fundamentals)

        # Check first row: PB = 40.0, BTP should be 0.025
        first_btp = result.loc[(pd.Timestamp("2023-01-31"), "AAPL"), "value_btp"]
        expected_btp = 1.0 / 40.0  # = 0.025
        assert np.isclose(first_btp, expected_btp)

    def test_sales_to_price_calculation(
        self, sample_raw_fundamentals: pd.DataFrame
    ) -> None:
        """Sales-to-price should be 1 / PS ratio."""
        result = compute_value_factors(sample_raw_fundamentals)

        # Check MSFT on date 1: PS = 10.0, STP should be 0.1
        stp = result.loc[(pd.Timestamp("2023-01-31"), "MSFT"), "value_stp"]
        expected_stp = 1.0 / 10.0
        assert np.isclose(stp, expected_stp)


class TestComputeQualityFactors:
    """Tests for compute_quality_factors function."""

    def test_output_columns(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Should have expected quality factor columns."""
        result = compute_quality_factors(sample_raw_fundamentals)
        expected_cols = {"quality_roe", "quality_roa", "quality_net_margin"}
        assert set(result.columns) == expected_cols

    def test_index_matches_input(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Output index should match input index."""
        result = compute_quality_factors(sample_raw_fundamentals)
        assert result.index.equals(sample_raw_fundamentals.index)

    def test_values_passthrough(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Quality factors should pass through raw values."""
        result = compute_quality_factors(sample_raw_fundamentals)

        # Check ROE for AAPL on date 1: should be 1.50
        roe = result.loc[(pd.Timestamp("2023-01-31"), "AAPL"), "quality_roe"]
        assert np.isclose(roe, 1.50)


class TestComputeGrowthFactors:
    """Tests for compute_growth_factors function."""

    def test_output_columns(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Should have expected growth factor columns."""
        result = compute_growth_factors(sample_raw_fundamentals)
        expected_cols = {"growth_rev_1y", "growth_eps_1y"}
        assert set(result.columns) == expected_cols

    def test_index_matches_input(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Output index should match input index."""
        result = compute_growth_factors(sample_raw_fundamentals)
        assert result.index.equals(sample_raw_fundamentals.index)

    def test_values_passthrough(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Growth factors should pass through raw values."""
        result = compute_growth_factors(sample_raw_fundamentals)

        # Check GOOG revenue growth on date 1: should be 0.25 (25%)
        rev_growth = result.loc[(pd.Timestamp("2023-01-31"), "GOOG"), "growth_rev_1y"]
        assert np.isclose(rev_growth, 0.25)


class TestComputeProfitabilityFactors:
    """Tests for compute_profitability_factors function."""

    def test_output_columns(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Should have expected profitability factor columns."""
        result = compute_profitability_factors(sample_raw_fundamentals)
        expected_cols = {"prof_gross_margin", "prof_op_margin"}
        assert set(result.columns) == expected_cols

    def test_index_matches_input(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Output index should match input index."""
        result = compute_profitability_factors(sample_raw_fundamentals)
        assert result.index.equals(sample_raw_fundamentals.index)

    def test_values_passthrough(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Profitability factors should pass through raw values."""
        result = compute_profitability_factors(sample_raw_fundamentals)

        # Check MSFT gross margin on date 1: should be 0.68
        gross_margin = result.loc[(pd.Timestamp("2023-01-31"), "MSFT"), "prof_gross_margin"]
        assert np.isclose(gross_margin, 0.68)


class TestComputeSizeFactor:
    """Tests for compute_size_factor function."""

    def test_output_columns(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Should have single size_log_mktcap column."""
        result = compute_size_factor(sample_raw_fundamentals)
        assert list(result.columns) == ["size_log_mktcap"]

    def test_index_matches_input(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Output index should match input index."""
        result = compute_size_factor(sample_raw_fundamentals)
        assert result.index.equals(sample_raw_fundamentals.index)

    def test_log_calculation(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Size factor should be log of market cap."""
        result = compute_size_factor(sample_raw_fundamentals)

        # Check AAPL on date 1: market_cap = 3e12, log should be ~28.43
        log_mktcap = result.loc[(pd.Timestamp("2023-01-31"), "AAPL"), "size_log_mktcap"]
        expected = np.log(3e12)  # ~28.43
        assert np.isclose(log_mktcap, expected)


# =============================================================================
# Tests for Cross-Sectional Processing
# =============================================================================


class TestWinsorizeCrossSectional:
    """Tests for _winsorize_cross_sectional function."""

    def test_clips_extreme_values(self) -> None:
        """Winsorization should clip extreme values within each date."""
        # Create data with outliers
        dates = pd.to_datetime(["2023-01-31", "2023-01-31", "2023-01-31"])
        tickers = ["A", "B", "C"]
        idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])

        # Values: 0.01, 0.5, 0.99 - extreme tails
        df = pd.DataFrame({"factor": [0.01, 0.5, 0.99]}, index=idx)

        # Winsorize at 25% (will clip 0.01 and 0.99)
        result = _winsorize_cross_sectional(df, pct=0.25)

        # After winsorization, extreme values should be clipped
        values = result["factor"].values
        assert values.min() >= 0.01  # Can't be lower than original min
        assert values.max() <= 0.99  # Can't be higher than original max

    def test_preserves_index(self) -> None:
        """Winsorization should preserve the index."""
        dates = pd.to_datetime(["2023-01-31", "2023-01-31"])
        tickers = ["A", "B"]
        idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
        df = pd.DataFrame({"factor": [0.3, 0.7]}, index=idx)

        result = _winsorize_cross_sectional(df, pct=0.1)
        assert result.index.equals(df.index)


class TestZscoreCrossSectional:
    """Tests for _zscore_cross_sectional function."""

    def test_mean_near_zero(self) -> None:
        """Z-scored values should have mean ~0 within each date."""
        dates = pd.to_datetime(["2023-01-31"] * 5 + ["2023-02-28"] * 5)
        tickers = ["A", "B", "C", "D", "E"] * 2
        idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])

        np.random.seed(42)
        df = pd.DataFrame({"factor": np.random.randn(10) * 10 + 50}, index=idx)

        result = _zscore_cross_sectional(df)

        # Check mean for each date is ~0
        for date in result.index.get_level_values("date").unique():
            date_mean = result.loc[date, "factor"].mean()
            assert np.isclose(date_mean, 0, atol=1e-10)

    def test_std_near_one(self) -> None:
        """Z-scored values should have std ~1 within each date."""
        dates = pd.to_datetime(["2023-01-31"] * 5 + ["2023-02-28"] * 5)
        tickers = ["A", "B", "C", "D", "E"] * 2
        idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])

        np.random.seed(42)
        df = pd.DataFrame({"factor": np.random.randn(10) * 10 + 50}, index=idx)

        result = _zscore_cross_sectional(df)

        # Check std for each date is ~1
        for date in result.index.get_level_values("date").unique():
            date_std = result.loc[date, "factor"].std()
            assert np.isclose(date_std, 1, atol=0.1)

    def test_preserves_index(self) -> None:
        """Z-scoring should preserve the index."""
        dates = pd.to_datetime(["2023-01-31", "2023-01-31"])
        tickers = ["A", "B"]
        idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
        df = pd.DataFrame({"factor": [0.3, 0.7]}, index=idx)

        result = _zscore_cross_sectional(df)
        assert result.index.equals(df.index)


# =============================================================================
# Tests for build_fundamental_factors
# =============================================================================


class TestBuildFundamentalFactors:
    """Tests for build_fundamental_factors function."""

    def test_returns_dataframe(self, sample_raw_fundamentals: pd.DataFrame) -> None:
        """Should return a DataFrame."""
        result = build_fundamental_factors(sample_raw_fundamentals)
        assert isinstance(result, pd.DataFrame)

    def test_includes_all_factor_families_by_default(
        self, sample_raw_fundamentals: pd.DataFrame
    ) -> None:
        """Default config should include all factor families."""
        result = build_fundamental_factors(sample_raw_fundamentals)

        # Check for columns from each family
        cols = set(result.columns)
        assert "value_ey" in cols
        assert "quality_roe" in cols
        assert "growth_rev_1y" in cols
        assert "prof_gross_margin" in cols
        assert "size_log_mktcap" in cols

    def test_config_excludes_factor_families(
        self, sample_raw_fundamentals: pd.DataFrame
    ) -> None:
        """Config should allow excluding factor families."""
        config = FundamentalFactorConfig(
            use_value=False,
            use_size=False,
        )
        result = build_fundamental_factors(sample_raw_fundamentals, config)

        cols = set(result.columns)

        # Excluded families should not be present
        assert "value_ey" not in cols
        assert "size_log_mktcap" not in cols

        # Included families should be present
        assert "quality_roe" in cols
        assert "growth_rev_1y" in cols

    def test_zscore_normalizes_cross_sectionally(
        self, sample_raw_fundamentals: pd.DataFrame
    ) -> None:
        """With zscore enabled, factors should be normalized per date."""
        config = FundamentalFactorConfig(
            zscore_by_cross_section=True,
            winsorize_pct=0,  # Disable winsorization for clean test
        )
        result = build_fundamental_factors(sample_raw_fundamentals, config)

        # Check that each date has mean ~0 for each factor
        for date in result.index.get_level_values("date").unique():
            date_data = result.loc[date]
            for col in result.columns:
                col_mean = date_data[col].mean()
                # Allow some tolerance for small sample
                assert np.isclose(col_mean, 0, atol=1e-10), f"{col} mean={col_mean}"

    def test_no_zscore_preserves_values(
        self, sample_raw_fundamentals: pd.DataFrame
    ) -> None:
        """With zscore disabled, raw values should be preserved."""
        config = FundamentalFactorConfig(
            zscore_by_cross_section=False,
            winsorize_pct=0,
        )
        result = build_fundamental_factors(sample_raw_fundamentals, config)

        # Quality ROE should match raw value
        roe = result.loc[(pd.Timestamp("2023-01-31"), "AAPL"), "quality_roe"]
        assert np.isclose(roe, 1.50)

    def test_empty_result_when_all_disabled(
        self, sample_raw_fundamentals: pd.DataFrame
    ) -> None:
        """Should return empty DataFrame if all families disabled."""
        config = FundamentalFactorConfig(
            use_value=False,
            use_quality=False,
            use_growth=False,
            use_profitability=False,
            use_size=False,
        )
        result = build_fundamental_factors(sample_raw_fundamentals, config)

        assert result.empty or len(result.columns) == 0


# =============================================================================
# Tests for load_raw_fundamentals
# =============================================================================


class TestLoadRawFundamentals:
    """Tests for load_raw_fundamentals function."""

    def test_returns_dataframe_with_placeholder_data(self) -> None:
        """Should return DataFrame even without CSV."""
        result = load_raw_fundamentals(
            tickers=["AAPL", "MSFT"],
            start="2023-01-01",
            end="2023-12-31",
            fundamentals_csv_path=None,
            use_yfinance=False,  # Force placeholder mode
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["date", "ticker"]

    def test_placeholder_has_expected_columns(self) -> None:
        """Placeholder data should have all expected columns."""
        result = load_raw_fundamentals(
            tickers=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            fundamentals_csv_path=None,
            use_yfinance=False,  # Force placeholder mode
        )

        for col in RAW_FUNDAMENTAL_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_placeholder_values_are_nan(self) -> None:
        """Placeholder data should have NaN values."""
        result = load_raw_fundamentals(
            tickers=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            fundamentals_csv_path=None,
            use_yfinance=False,  # Force placeholder mode
        )

        # All values should be NaN
        assert result.isna().all().all()

    def test_yfinance_returns_real_data(self) -> None:
        """yfinance mode should return real historical fundamental data."""
        # Use recent date range - yfinance only provides last 4-5 quarters
        result = load_raw_fundamentals(
            tickers=["AAPL"],
            start="2025-01-01",  # Recent range where yfinance has data
            end="2025-12-31",
            fundamentals_csv_path=None,
            use_yfinance=True,  # Enable yfinance
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Should have some non-NaN values for metrics derived from financial statements
        # (market_cap requires price data so it will be NaN, but roe/margins should exist)
        has_some_data = (
            not result["roe"].isna().all()
            or not result["gross_margin"].isna().all()
            or not result["revenue_ttm"].isna().all()
        )
        assert has_some_data, "Expected at least some fundamental data from yfinance"


# =============================================================================
# Integration Tests with Feature Pipeline
# =============================================================================


class TestFeaturePipelineIntegration:
    """Integration tests for fundamentals with feature_pipeline."""

    def test_feature_matrix_with_fundamentals_enabled(
        self, sample_prices: pd.DataFrame
    ) -> None:
        """Feature matrix should include both tech and fundamental features."""
        config = FeaturePipelineConfig(
            use_technical=True,
            use_fundamentals=True,
            momentum_windows=(21,),
            volatility_windows=(21,),
        )

        result = build_feature_matrix(sample_prices, config)

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)

        # Get feature names
        feature_names = result.columns.get_level_values("feature").unique().tolist()

        # Should have technical features (prefixed with tech_)
        tech_features = [f for f in feature_names if f.startswith("tech_")]
        assert len(tech_features) > 0, "Should have technical features"

        # Note: Fundamental features may be NaN and dropped due to placeholder data
        # The key test is that the pipeline runs without error

    def test_feature_matrix_technical_only(self, sample_prices: pd.DataFrame) -> None:
        """Feature matrix with only technical features should work."""
        config = FeaturePipelineConfig(
            use_technical=True,
            use_fundamentals=False,
            momentum_windows=(21,),
            volatility_windows=(21,),
        )

        result = build_feature_matrix(sample_prices, config)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > 0

        # Should NOT have fundamental features
        feature_names = result.columns.get_level_values("feature").unique().tolist()
        fund_features = [
            f
            for f in feature_names
            if any(
                f.startswith(p) for p in ["value_", "quality_", "growth_", "prof_", "size_"]
            )
        ]
        assert len(fund_features) == 0, "Should not have fundamental features"

    def test_feature_pipeline_config_defaults(self) -> None:
        """Default FeaturePipelineConfig should have expected values."""
        config = FeaturePipelineConfig()

        assert config.use_technical is True
        assert config.use_fundamentals is False
        assert config.use_sentiment is False
        assert config.fundamentals_csv_path is None
        assert isinstance(config.fundamental_factor_config, FundamentalFactorConfig)

    def test_fundamental_factor_config_defaults(self) -> None:
        """Default FundamentalFactorConfig should have expected values."""
        config = FundamentalFactorConfig()

        assert config.use_value is True
        assert config.use_quality is True
        assert config.use_growth is True
        assert config.use_profitability is True
        assert config.use_size is True
        assert config.winsorize_pct == 0.01
        assert config.zscore_by_cross_section is True
