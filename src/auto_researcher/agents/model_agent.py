"""
Model Agent.

This agent is responsible for training, evaluating, and using
the GBDT model for stock ranking.
"""

import logging
from typing import Any

import pandas as pd
import numpy as np

from auto_researcher.agents.fundamental_agent import BaseAgent
from auto_researcher.models.gbdt_model import GBDTModel
from auto_researcher.config import ModelConfig, get_default_config

logger = logging.getLogger(__name__)


class ModelAgent(BaseAgent):
    """
    Agent for managing the GBDT model.

    Responsibilities:
    - Train model on historical data
    - Generate stock rankings
    - Track model performance and feature importance
    """

    @property
    def name(self) -> str:
        return "ModelAgent"

    def __init__(self, config: ModelConfig | None = None) -> None:
        """
        Initialize the Model Agent.

        Args:
            config: Model configuration.
        """
        if config is None:
            config = get_default_config().model
        self.config = config
        self.model = GBDTModel(config)
        self._training_history: list[dict[str, Any]] = []
        logger.info(f"Initializing {self.name}")

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame | None = None,
        y_eval: pd.Series | None = None,
    ) -> GBDTModel:
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_eval: Optional validation features.
            y_eval: Optional validation labels.

        Returns:
            Trained model.
        """
        logger.info(f"{self.name} training model on {len(X_train)} samples")
        
        eval_set = None
        if X_eval is not None and y_eval is not None:
            eval_set = (X_eval, y_eval)
        
        self.model.fit(X_train, y_train, eval_set=eval_set)
        
        # Track training
        self._training_history.append({
            "n_samples": len(X_train),
            "n_features": len(X_train.columns),
        })
        
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        return self.model.predict(X)

    def rank_stocks(self, X: pd.DataFrame) -> pd.Series:
        """
        Rank stocks by predicted return.

        Args:
            X: Feature matrix with stock identifiers as index.

        Returns:
            Series of scores sorted descending.
        """
        return self.model.rank_stocks(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        predictions = self.model.predict(X)
        
        # Correlation (rank-based)
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(predictions, y)
        
        # MSE
        mse = np.mean((predictions - y) ** 2)
        
        return {
            "spearman_correlation": spearman_corr,
            "mse": mse,
            "n_samples": len(X),
        }

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, list[float]]:
        """
        Perform time-series cross-validation.

        Args:
            X: Feature matrix.
            y: Labels.
            n_splits: Number of CV splits.

        Returns:
            Dictionary with train and test scores.
        """
        return self.model.cross_validate(X, y, n_splits)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from trained model.

        Returns:
            Series of feature importances sorted descending.
        """
        return self.model.get_feature_importance()

    def get_training_summary(self) -> dict[str, Any]:
        """
        Get summary of training history.

        Returns:
            Dictionary with training summary.
        """
        if not self._training_history:
            return {"n_trainings": 0}
        
        return {
            "n_trainings": len(self._training_history),
            "last_training": self._training_history[-1],
        }
