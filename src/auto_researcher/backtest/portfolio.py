"""
Portfolio construction utilities.

This module provides functions for constructing portfolios from model predictions,
including top-k selection and weighting schemes.
"""

import logging
from typing import Literal

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def build_equal_weight_portfolio(
    scores: pd.Series,
    top_k: int,
) -> dict[str, float]:
    """
    Build an equal-weight portfolio from the top-k scoring stocks.

    Args:
        scores: Series with stock identifiers as index and scores as values.
                Higher scores indicate more attractive stocks.
        top_k: Number of stocks to include in the portfolio.

    Returns:
        Dictionary mapping ticker to weight (weights sum to 1.0).

    Examples:
        >>> scores = pd.Series({'AAPL': 0.5, 'MSFT': 0.3, 'GOOG': 0.2})
        >>> portfolio = build_equal_weight_portfolio(scores, top_k=2)
        >>> sum(portfolio.values())
        1.0
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    if len(scores) == 0:
        return {}
    
    # Select top-k stocks
    top_stocks = scores.nlargest(min(top_k, len(scores)))
    
    # Equal weight
    n_stocks = len(top_stocks)
    weight = 1.0 / n_stocks
    
    portfolio = {ticker: weight for ticker in top_stocks.index}
    
    logger.debug(f"Built equal-weight portfolio with {n_stocks} stocks")
    
    return portfolio


def build_score_weighted_portfolio(
    scores: pd.Series,
    top_k: int,
    normalize: bool = True,
) -> dict[str, float]:
    """
    Build a score-weighted portfolio from the top-k scoring stocks.

    Stocks are weighted proportionally to their scores.

    Args:
        scores: Series with stock identifiers as index and scores as values.
        top_k: Number of stocks to include.
        normalize: If True, weights sum to 1.0.

    Returns:
        Dictionary mapping ticker to weight.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    if len(scores) == 0:
        return {}
    
    # Select top-k stocks
    top_stocks = scores.nlargest(min(top_k, len(scores)))
    
    # Handle negative scores by shifting
    min_score = top_stocks.min()
    if min_score < 0:
        shifted_scores = top_stocks - min_score + 1e-6
    else:
        shifted_scores = top_stocks + 1e-6  # Avoid zero weights
    
    # Compute weights
    if normalize:
        total = shifted_scores.sum()
        weights = shifted_scores / total
    else:
        weights = shifted_scores
    
    portfolio = {ticker: weight for ticker, weight in weights.items()}
    
    return portfolio


def compute_portfolio_return(
    weights: dict[str, float],
    returns: pd.Series,
) -> float:
    """
    Compute portfolio return given weights and asset returns.

    Args:
        weights: Dictionary mapping ticker to weight.
        returns: Series with tickers as index and returns as values.

    Returns:
        Portfolio return (weighted sum of asset returns).
    """
    portfolio_return = 0.0
    
    for ticker, weight in weights.items():
        if ticker in returns.index:
            asset_return = returns[ticker]
            if pd.notna(asset_return):
                portfolio_return += weight * asset_return
    
    return portfolio_return


def compute_period_returns(
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]],
    prices: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> pd.Series:
    """
    Compute portfolio returns over multiple holding periods.

    Args:
        weights_history: List of (date, weights_dict) tuples.
        prices: Price DataFrame with tickers as columns.
        rebalance_dates: Dates when portfolio is rebalanced.

    Returns:
        Series of portfolio returns indexed by date.
    """
    portfolio_returns = []
    
    for i, (rebal_date, weights) in enumerate(weights_history):
        # Determine end of holding period
        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[i + 1]
        else:
            next_rebal = prices.index[-1]
        
        # Get prices at start and end
        if rebal_date in prices.index and next_rebal in prices.index:
            start_prices = prices.loc[rebal_date]
            end_prices = prices.loc[next_rebal]
            
            # Compute holding period returns
            period_returns = (end_prices / start_prices) - 1
            
            # Compute portfolio return
            port_return = compute_portfolio_return(weights, period_returns)
            portfolio_returns.append((next_rebal, port_return))
    
    if not portfolio_returns:
        return pd.Series(dtype=float)
    
    dates, returns = zip(*portfolio_returns)
    return pd.Series(returns, index=pd.DatetimeIndex(dates), name="portfolio_return")


def apply_constraints(
    weights: dict[str, float],
    max_weight: float = 0.20,
    min_weight: float = 0.0,
) -> dict[str, float]:
    """
    Apply position constraints to portfolio weights.

    Args:
        weights: Initial portfolio weights.
        max_weight: Maximum weight per position.
        min_weight: Minimum weight per position.

    Returns:
        Constrained weights (re-normalized to sum to 1.0).
    """
    # Clip weights
    constrained = {
        ticker: np.clip(w, min_weight, max_weight)
        for ticker, w in weights.items()
    }
    
    # Re-normalize
    total = sum(constrained.values())
    if total > 0:
        constrained = {t: w / total for t, w in constrained.items()}
    
    return constrained


def get_turnover(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
) -> float:
    """
    Compute portfolio turnover between two weight dictionaries.

    Turnover is defined as sum of absolute weight changes / 2.

    Args:
        old_weights: Previous period weights.
        new_weights: New period weights.

    Returns:
        Turnover as a fraction (0.0 to 1.0).
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    
    total_change = 0.0
    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0.0)
        new_w = new_weights.get(ticker, 0.0)
        total_change += abs(new_w - old_w)
    
    return total_change / 2.0
