"""Tests for feature pipeline module."""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.features.feature_pipeline import (
    build_feature_matrix,
    build_training_data,
    reshape_features_for_model,
    reshape_labels_for_model,
    prepare_training_data,
    get_features_for_date,
)
from auto_researcher.config import FeatureConfig


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    
    # Generate random walk prices including SPY as benchmark
    tickers = ["AAPL", "MSFT", "GOOG", "SPY"]
    returns = np.random.randn(300, 4) * 0.02
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def test_config() -> FeatureConfig:
    """Minimal feature config for fast tests."""
    return FeatureConfig(
        momentum_windows=(21, 63),
        volatility_windows=(21,),
        include_fundamentals=False,
        include_sentiment=False,
    )


class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix function."""

    def test_returns_dataframe(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Should return a DataFrame."""
        features = build_feature_matrix(sample_prices, test_config)
        assert isinstance(features, pd.DataFrame)

    def test_index_is_datetime(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Index should be DatetimeIndex."""
        features = build_feature_matrix(sample_prices, test_config)
        assert isinstance(features.index, pd.DatetimeIndex)

    def test_columns_multiindex(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Columns should be MultiIndex (ticker, feature)."""
        features = build_feature_matrix(sample_prices, test_config)
        
        assert isinstance(features.columns, pd.MultiIndex)
        assert "ticker" in features.columns.names
        assert "feature" in features.columns.names

    def test_same_row_count(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Should have same number of rows as input."""
        features = build_feature_matrix(sample_prices, test_config)
        assert len(features) == len(sample_prices)


class TestBuildTrainingData:
    """Tests for build_training_data function."""

    def test_returns_tuple(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Should return tuple of (features, labels)."""
        result = build_training_data(sample_prices, horizon_days=21, config=test_config)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_labels_exclude_benchmark(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Labels should not include benchmark ticker."""
        features, labels = build_training_data(
            sample_prices, horizon_days=21, benchmark="SPY", config=test_config
        )
        
        assert "SPY" not in labels.columns

    def test_features_and_labels_aligned(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Features and labels should have same index."""
        features, labels = build_training_data(
            sample_prices, horizon_days=21, config=test_config
        )
        
        assert (features.index == labels.index).all()

    def test_no_lookahead_bias(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Labels should be forward-looking relative to features."""
        horizon = 21
        features, labels = build_training_data(
            sample_prices, horizon_days=horizon, config=test_config
        )
        
        # The last date with valid labels should be horizon days before the end
        last_valid_date = labels.index[-1]
        max_possible_date = sample_prices.index[-horizon - 1]
        
        assert last_valid_date <= max_possible_date

    def test_raises_on_missing_benchmark(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Should raise ValueError if benchmark not in prices."""
        with pytest.raises(ValueError, match="Benchmark"):
            build_training_data(
                sample_prices, horizon_days=21, benchmark="INVALID", config=test_config
            )

    def test_no_benchmark(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Should work without benchmark (raw returns)."""
        features, labels = build_training_data(
            sample_prices, horizon_days=21, benchmark=None, config=test_config
        )
        
        # All tickers should be in labels
        assert len(labels.columns) == len(sample_prices.columns)


class TestReshapeForModel:
    """Tests for reshaping functions."""

    def test_reshape_features(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Reshaped features should have (date, ticker) MultiIndex."""
        features = build_feature_matrix(sample_prices, test_config)
        stacked = reshape_features_for_model(features)
        
        assert isinstance(stacked.index, pd.MultiIndex)
        assert stacked.index.names == ["date", "ticker"]

    def test_reshape_labels(self, sample_prices: pd.DataFrame) -> None:
        """Reshaped labels should be Series with MultiIndex."""
        labels = sample_prices.drop(columns=["SPY"]).iloc[:100]
        stacked = reshape_labels_for_model(labels)
        
        assert isinstance(stacked, pd.Series)
        assert stacked.index.names == ["date", "ticker"]


class TestPrepareTrainingData:
    """Tests for complete training data preparation."""

    def test_returns_x_y(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Should return (X, y) tuple."""
        X, y = prepare_training_data(
            sample_prices, horizon_days=21, config=test_config
        )
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_x_y_aligned(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """X and y should have matching indices."""
        X, y = prepare_training_data(
            sample_prices, horizon_days=21, config=test_config
        )
        
        assert len(X) == len(y)
        assert (X.index == y.index).all()

    def test_no_nans_in_output(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Output should not contain NaN values."""
        X, y = prepare_training_data(
            sample_prices, horizon_days=21, config=test_config
        )
        
        assert not X.isna().any().any()
        assert not y.isna().any()


class TestGetFeaturesForDate:
    """Tests for extracting features for a specific date."""

    def test_extracts_single_date(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Should extract features for a single date."""
        features = build_feature_matrix(sample_prices, test_config)
        
        # Get a valid date (after warmup)
        valid_date = features.index[100]
        date_features = get_features_for_date(features, valid_date)
        
        # Should have tickers as rows
        assert len(date_features) == len(sample_prices.columns)

    def test_raises_on_invalid_date(
        self, sample_prices: pd.DataFrame, test_config: FeatureConfig
    ) -> None:
        """Should raise KeyError for invalid date."""
        features = build_feature_matrix(sample_prices, test_config)
        
        with pytest.raises(KeyError):
            get_features_for_date(features, pd.Timestamp("1900-01-01"))
