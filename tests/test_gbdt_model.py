"""Tests for GBDT model module."""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.models.gbdt_model import GBDTModel
from auto_researcher.config import ModelConfig


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic training data."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    
    # Generate target with some signal
    noise = np.random.randn(n_samples) * 0.5
    signal = X["feature_0"] * 0.3 + X["feature_1"] * 0.2 - X["feature_2"] * 0.15
    y = pd.Series(signal + noise, name="target")
    
    return X, y


@pytest.fixture
def fast_config() -> ModelConfig:
    """Fast model config for testing."""
    return ModelConfig(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=8,
        min_child_samples=5,
        random_state=42,
        early_stopping_rounds=None,
    )


class TestGBDTModelFit:
    """Tests for model fitting."""

    def test_fit_runs_without_error(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Model should fit without errors."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        
        # Should not raise
        model.fit(X, y)
        
        assert model.model is not None

    def test_fit_stores_feature_names(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Model should store feature names after fitting."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        model.fit(X, y)
        
        assert model.feature_names == list(X.columns)

    def test_fit_returns_self(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Fit should return self for method chaining."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        
        result = model.fit(X, y)
        
        assert result is model


class TestGBDTModelPredict:
    """Tests for model prediction."""

    def test_predict_returns_array(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Predict should return numpy array."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert isinstance(predictions, np.ndarray)

    def test_predict_correct_length(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Predictions should have same length as input."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)

    def test_predict_without_fit_raises(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Predict should raise if model not fitted."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(X)

    def test_predict_on_new_data(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Should predict on held-out data."""
        X, y = sample_data
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        y_train = y.iloc[:400]
        
        model = GBDTModel(fast_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)


class TestGBDTModelRankStocks:
    """Tests for stock ranking."""

    def test_rank_stocks_returns_series(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """rank_stocks should return Series."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        model.fit(X, y)
        
        # Use subset as "stocks to rank"
        X_stocks = X.iloc[:20].copy()
        X_stocks.index = [f"STOCK_{i}" for i in range(20)]
        
        ranked = model.rank_stocks(X_stocks)
        
        assert isinstance(ranked, pd.Series)

    def test_rank_stocks_sorted_descending(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Ranked stocks should be sorted by score descending."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        model.fit(X, y)
        
        X_stocks = X.iloc[:20].copy()
        X_stocks.index = [f"STOCK_{i}" for i in range(20)]
        
        ranked = model.rank_stocks(X_stocks)
        
        # Check sorted descending
        assert (ranked.values == sorted(ranked.values, reverse=True)).all()

    def test_rank_stocks_preserves_index(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Stock identifiers should be in ranked output."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        model.fit(X, y)
        
        X_stocks = X.iloc[:20].copy()
        stock_names = [f"STOCK_{i}" for i in range(20)]
        X_stocks.index = stock_names
        
        ranked = model.rank_stocks(X_stocks)
        
        assert set(ranked.index) == set(stock_names)


class TestGBDTModelFeatureImportance:
    """Tests for feature importance."""

    def test_feature_importance_returns_series(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Feature importance should return Series."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.Series)

    def test_feature_importance_all_features(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Feature importance should include all features."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == len(X.columns)
        assert set(importance.index) == set(X.columns)


class TestGBDTModelCrossValidate:
    """Tests for cross-validation."""

    def test_cross_validate_returns_scores(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """Cross-validation should return train and test scores."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        
        cv_results = model.cross_validate(X, y, n_splits=3)
        
        assert "train_scores" in cv_results
        assert "test_scores" in cv_results
        assert len(cv_results["train_scores"]) == 3
        assert len(cv_results["test_scores"]) == 3

    def test_cross_validate_scores_are_floats(
        self, sample_data: tuple[pd.DataFrame, pd.Series], fast_config: ModelConfig
    ) -> None:
        """CV scores should be valid floats."""
        X, y = sample_data
        model = GBDTModel(fast_config)
        
        cv_results = model.cross_validate(X, y, n_splits=3)
        
        for score in cv_results["test_scores"]:
            assert isinstance(score, float)
            assert not np.isnan(score)
