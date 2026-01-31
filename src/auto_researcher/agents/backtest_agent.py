"""
Backtest Agent.

This agent is responsible for running backtests and managing
portfolio construction.
"""

import logging
from typing import Any

import pandas as pd

from auto_researcher.agents.fundamental_agent import BaseAgent
from auto_researcher.backtest.runner import (
    run_backtest,
    run_simple_backtest,
    BacktestResult,
)
from auto_researcher.backtest.metrics import compute_all_metrics
from auto_researcher.config import PipelineConfig, get_default_config

logger = logging.getLogger(__name__)


class BacktestAgent(BaseAgent):
    """
    Agent for running backtests.

    Responsibilities:
    - Execute walk-forward backtests
    - Track portfolio construction
    - Compute and report performance metrics
    """

    @property
    def name(self) -> str:
        return "BacktestAgent"

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """
        Initialize the Backtest Agent.

        Args:
            config: Pipeline configuration.
        """
        if config is None:
            config = get_default_config()
        self.config = config
        self._results_history: list[BacktestResult] = []
        logger.info(f"Initializing {self.name}")

    def run(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> BacktestResult:
        """
        Run a backtest.

        Args:
            prices: Adjusted close prices.
            **kwargs: Additional arguments passed to run_backtest.

        Returns:
            BacktestResult with performance metrics.
        """
        logger.info(f"{self.name} running backtest")
        
        result = run_backtest(
            prices=prices,
            horizon_days=kwargs.get("horizon_days", self.config.research.horizon_days),
            top_k=kwargs.get("top_k", self.config.research.top_k),
            benchmark=kwargs.get("benchmark", self.config.research.benchmark),
            rebalance_frequency=kwargs.get(
                "rebalance_frequency", self.config.research.rebalance_frequency
            ),
            min_train_periods=kwargs.get(
                "min_train_periods", self.config.research.lookback_days
            ),
            config=self.config,
        )
        
        self._results_history.append(result)
        
        return result

    def run_simple(self, prices: pd.DataFrame) -> BacktestResult:
        """
        Run a simple backtest with default settings.

        Args:
            prices: Adjusted close prices.

        Returns:
            BacktestResult.
        """
        result = run_simple_backtest(prices, self.config)
        self._results_history.append(result)
        return result

    def get_last_result(self) -> BacktestResult | None:
        """
        Get the most recent backtest result.

        Returns:
            Last BacktestResult or None if no backtests run.
        """
        return self._results_history[-1] if self._results_history else None

    def compare_results(self) -> pd.DataFrame:
        """
        Compare metrics across all backtest runs.

        Returns:
            DataFrame with metrics for each backtest.
        """
        if not self._results_history:
            return pd.DataFrame()
        
        records = []
        for i, result in enumerate(self._results_history):
            record = {"backtest_id": i, **result.metrics}
            records.append(record)
        
        return pd.DataFrame(records)

    def generate_report(self, result: BacktestResult | None = None) -> str:
        """
        Generate a text report for a backtest.

        Args:
            result: BacktestResult to report. Uses last result if None.

        Returns:
            Formatted report string.
        """
        if result is None:
            result = self.get_last_result()
        
        if result is None:
            return "No backtest results available."
        
        lines = [
            "=" * 50,
            "BACKTEST REPORT",
            "=" * 50,
            "",
            f"Portfolio Returns: {len(result.portfolio_returns)} periods",
            f"Date Range: {result.portfolio_returns.index.min()} to {result.portfolio_returns.index.max()}",
            "",
            "PERFORMANCE METRICS",
            "-" * 30,
        ]
        
        for key, value in result.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        
        lines.extend([
            "",
            f"Number of Rebalances: {len(result.weights_history)}",
            f"Average IC: {result.metrics.get('average_ic', 'N/A')}",
            "=" * 50,
        ])
        
        return "\n".join(lines)

    def plot_results(self, result: BacktestResult | None = None) -> None:
        """
        Plot backtest results.

        TODO: Implement with matplotlib.

        Args:
            result: BacktestResult to plot.
        """
        logger.warning("plot_results not implemented - requires matplotlib")
        # TODO: Implement cumulative return chart, drawdown chart, etc.
