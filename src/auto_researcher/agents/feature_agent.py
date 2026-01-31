"""
Feature Agent.

This agent orchestrates feature engineering by combining outputs from
technical, fundamental, and sentiment feature pipelines.
"""

import logging
from typing import Any

import pandas as pd

from auto_researcher.agents.fundamental_agent import BaseAgent
from auto_researcher.features.feature_pipeline import (
    build_feature_matrix,
    prepare_training_data,
)
from auto_researcher.config import FeatureConfig, get_default_config

logger = logging.getLogger(__name__)


class FeatureAgent(BaseAgent):
    """
    Agent for orchestrating feature engineering.

    Responsibilities:
    - Coordinate technical, fundamental, and sentiment feature computation
    - Build unified feature matrix
    - Handle feature alignment and missing data
    """

    @property
    def name(self) -> str:
        return "FeatureAgent"

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """
        Initialize the Feature Agent.

        Args:
            config: Feature configuration.
        """
        if config is None:
            config = get_default_config().features
        self.config = config
        logger.info(f"Initializing {self.name}")

    def run(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Build complete feature matrix from price data.

        Args:
            prices: Adjusted close prices with tickers as columns.
            **kwargs: Additional arguments passed to build_feature_matrix.

        Returns:
            Feature matrix with MultiIndex columns (ticker, feature).
        """
        logger.info(f"{self.name} building feature matrix")
        return build_feature_matrix(prices, self.config)

    def prepare_training_data(
        self,
        prices: pd.DataFrame,
        horizon_days: int,
        benchmark: str | None = "SPY",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare aligned features and labels for model training.

        Args:
            prices: Adjusted close prices.
            horizon_days: Forward return horizon.
            benchmark: Benchmark for excess returns.

        Returns:
            Tuple of (X, y) ready for model training.
        """
        logger.info(f"{self.name} preparing training data")
        return prepare_training_data(prices, horizon_days, benchmark, self.config)

    def get_feature_summary(self, feature_matrix: pd.DataFrame) -> dict[str, Any]:
        """
        Get summary statistics for feature matrix.

        Args:
            feature_matrix: Feature matrix from run().

        Returns:
            Dictionary with summary statistics.
        """
        return {
            "n_dates": len(feature_matrix),
            "n_features": len(feature_matrix.columns.get_level_values("feature").unique()),
            "n_tickers": len(feature_matrix.columns.get_level_values("ticker").unique()),
            "date_range": (
                feature_matrix.index.min().strftime("%Y-%m-%d"),
                feature_matrix.index.max().strftime("%Y-%m-%d"),
            ),
            "missing_pct": feature_matrix.isna().mean().mean() * 100,
        }
