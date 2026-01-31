"""
GBDT model for cross-sectional stock ranking.

This module provides a LightGBM-based model for predicting forward returns
and ranking stocks within a cross-section.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from auto_researcher.config import ModelConfig, get_default_config

logger = logging.getLogger(__name__)


class GBDTModel:
    """
    Gradient Boosted Decision Tree model for stock ranking.

    Uses LightGBM to predict forward returns. Designed for cross-sectional
    ranking rather than absolute return prediction.

    Attributes:
        config: Model configuration.
        model: Underlying LightGBM model (after fitting).
        feature_names: Names of features used in training.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """
        Initialize the GBDT model.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        if config is None:
            config = get_default_config().model
        
        self.config = config
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names: list[str] | None = None

    def _create_model(self) -> lgb.LGBMRegressor:
        """Create the underlying LightGBM model."""
        # Build params dict, handling subsample (requires bagging settings)
        params = {
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "num_leaves": self.config.num_leaves,
            "min_child_samples": self.config.min_child_samples,
            "random_state": self.config.random_state,
            "reg_lambda": self.config.reg_lambda,
            "colsample_bytree": self.config.colsample_bytree,
            "verbose": -1,
            "n_jobs": -1,
        }
        
        # LightGBM requires bagging_freq > 0 for subsample to take effect
        if self.config.subsample < 1.0:
            params["subsample"] = self.config.subsample
            params["subsample_freq"] = 1  # Enable bagging every iteration
        
        return lgb.LGBMRegressor(**params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "GBDTModel":
        """
        Train the model on feature matrix X and labels y.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target values (forward returns).
            eval_set: Optional (X_val, y_val) for early stopping.

        Returns:
            Self for method chaining.
        """
        self.feature_names = list(X.columns)
        self.model = self._create_model()

        fit_params: dict[str, Any] = {}
        
        if eval_set is not None and self.config.early_stopping_rounds is not None:
            X_val, y_val = eval_set
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)
            ]

        self.model.fit(X, y, **fit_params)
        
        logger.info(f"Model trained with {len(X)} samples, {len(self.feature_names)} features")
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict forward returns for feature matrix X.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Array of predicted returns.

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.model.predict(X)

    def rank_stocks(self, X: pd.DataFrame) -> pd.Series:
        """
        Rank stocks by predicted return (descending).

        Args:
            X: Feature matrix where index represents stock identifiers.

        Returns:
            Series with stock identifiers as index and predicted scores as values,
            sorted in descending order (best stocks first).

        Raises:
            ValueError: If model has not been fitted.
        """
        predictions = self.predict(X)
        scores = pd.Series(predictions, index=X.index, name="score")
        
        return scores.sort_values(ascending=False)

    def get_feature_importance(self, as_dataframe: bool = False) -> pd.Series | pd.DataFrame:
        """
        Get feature importance from the trained model.

        Args:
            as_dataframe: If True, return DataFrame with columns (feature_name, importance).
                         If False, return Series with feature names as index.

        Returns:
            Series with feature names as index and importance as values (default).
            Or DataFrame with columns ['feature_name', 'importance'] if as_dataframe=True.

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        importance = self.model.feature_importances_
        series = pd.Series(importance, index=self.feature_names, name="importance").sort_values(
            ascending=False
        )
        
        if as_dataframe:
            return pd.DataFrame({
                "feature_name": series.index,
                "importance": series.values,
            }).reset_index(drop=True)
        
        return series

    def get_feature_importance_by_family(self) -> pd.DataFrame:
        """
        Get feature importance aggregated by factor family.

        Factor families are determined by feature name prefixes:
        - tech_: Technical features (momentum, volatility, etc.)
        - value_: Value factors
        - quality_: Quality factors
        - growth_: Growth factors
        - prof_: Profitability factors
        - size_: Size factor

        Features without recognized prefixes are grouped under 'other'.

        Returns:
            DataFrame with columns ['family', 'importance', 'pct'] sorted by importance.

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Define family prefixes
        family_prefixes = {
            "tech": ["tech_", "mom_", "vol_", "rsi_", "ma_"],
            "value": ["value_"],
            "quality": ["quality_"],
            "growth": ["growth_"],
            "profitability": ["prof_"],
            "size": ["size_"],
        }

        importance = self.model.feature_importances_
        family_importance: dict[str, float] = {family: 0.0 for family in family_prefixes}
        family_importance["other"] = 0.0

        for feat_name, imp in zip(self.feature_names, importance):
            assigned = False
            for family, prefixes in family_prefixes.items():
                if any(feat_name.startswith(p) for p in prefixes):
                    family_importance[family] += imp
                    assigned = True
                    break
            if not assigned:
                family_importance["other"] += imp

        # Remove families with zero importance
        family_importance = {k: v for k, v in family_importance.items() if v > 0}

        # Normalize to sum to 1.0
        total = sum(family_importance.values())
        if total > 0:
            normalized = {k: v / total for k, v in family_importance.items()}
        else:
            normalized = family_importance

        # Convert to DataFrame
        df = pd.DataFrame([
            {"family": k, "importance": family_importance[k], "pct": normalized.get(k, 0)}
            for k in family_importance
        ])
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, list[float]]:
        """
        Perform time-series cross-validation.

        Uses expanding window: train on all data up to split, test on next fold.

        Args:
            X: Feature matrix.
            y: Labels.
            n_splits: Number of CV splits.

        Returns:
            Dictionary with 'train_scores' and 'test_scores'.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        train_scores = []
        test_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create fresh model for each fold
            model = self._create_model()
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Use correlation as score (for ranking, correlation > MSE)
            train_corr = np.corrcoef(y_train, train_pred)[0, 1]
            test_corr = np.corrcoef(y_test, test_pred)[0, 1]
            
            train_scores.append(train_corr)
            test_scores.append(test_corr)
        
        return {
            "train_scores": train_scores,
            "test_scores": test_scores,
        }

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model.

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        self.model.booster_.save_model(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "GBDTModel":
        """
        Load a model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Self for method chaining.
        """
        self.model = lgb.LGBMRegressor()
        self.model.booster_ = lgb.Booster(model_file=path)
        logger.info(f"Model loaded from {path}")
        
        return self
