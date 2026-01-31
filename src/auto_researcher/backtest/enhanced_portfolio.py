"""
Enhanced portfolio construction with rank-based weighting and risk neutralization.

This module extends the basic portfolio construction with:
- Rank-based weighting (exponential decay)
- Volatility-scaled weights
- Beta neutralization
- Sector-neutral constraints
- Sector-neutral ranking (rank within sectors, then select top across sectors)
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# SECTOR UTILITIES
# =============================================================================

def get_ticker_sectors(
    tickers: list[str],
    sector_mapping: dict[str, str] | None = None,
) -> pd.Series:
    """
    Get sector assignments for a list of tickers.
    
    Args:
        tickers: List of ticker symbols.
        sector_mapping: Optional ticker -> sector mapping. 
                       If None, uses DEFAULT_SECTOR_MAPPING from enhanced features.
    
    Returns:
        Series with ticker as index and sector as values.
    """
    from auto_researcher.features.enhanced import (
        get_sector_for_ticker,
        DEFAULT_SECTOR_MAPPING,
    )
    
    if sector_mapping is None:
        sector_mapping = DEFAULT_SECTOR_MAPPING
    
    sectors = {
        ticker: get_sector_for_ticker(ticker, sector_mapping)
        for ticker in tickers
    }
    return pd.Series(sectors)


@dataclass
class EnhancedPortfolioConfig:
    """Configuration for enhanced portfolio construction.
    
    Attributes:
        top_k: Number of stocks to include in the portfolio.
        weighting_scheme: How to weight stocks:
            - "equal": Equal weight all positions
            - "rank": Exponential decay based on rank
            - "score": Weight proportional to prediction score
        rank_tau: Temperature parameter for rank weighting.
            Lower tau = more concentrated in top picks.
            w_i = exp(-rank_i / tau), then normalized.
        neutralization: Risk neutralization method:
            - "none": No neutralization
            - "vol": Scale weights by inverse volatility
            - "beta": Target zero portfolio beta
            - "vol_beta": Apply both vol scaling and beta neutralization
        target_vol: Target portfolio volatility (if neutralization involves vol).
        max_position_weight: Maximum weight for any single position.
        min_positions: Minimum number of positions to hold.
        apply_sector_neutral: If True, balance weights across sectors.
        sector_max_weight: Maximum weight per sector.
        sector_neutral_ranking: If True, rank within sectors first, then pick
            top stocks from each sector proportionally. This prevents the model
            from concentrating in one sector (e.g., all tech).
        min_stocks_per_sector: Minimum stocks to pick from any represented sector
            when using sector_neutral_ranking.
    """
    
    top_k: int = 25
    weighting_scheme: Literal["equal", "rank", "score"] = "rank"
    rank_tau: float = 5.0
    neutralization: Literal["none", "vol", "beta", "vol_beta"] = "none"
    target_vol: float = 0.15  # 15% annualized
    max_position_weight: float = 0.10  # 10% max
    min_positions: int = 10
    apply_sector_neutral: bool = False
    sector_max_weight: float = 0.30  # 30% max per sector
    sector_neutral_ranking: bool = False  # NEW: rank within sectors first
    min_stocks_per_sector: int = 1  # NEW: minimum picks per sector


def build_rank_weighted_portfolio(
    scores: pd.Series,
    config: Optional[EnhancedPortfolioConfig] = None,
    volatilities: Optional[pd.Series] = None,
    betas: Optional[pd.Series] = None,
    sectors: Optional[pd.Series] = None,
) -> dict[str, float]:
    """
    Build a portfolio with rank-based weighting and optional risk adjustments.
    
    Args:
        scores: Series with tickers as index and prediction scores as values.
            Higher scores indicate more attractive stocks.
        config: Portfolio configuration. Uses defaults if not provided.
        volatilities: Optional series of annualized volatilities per ticker.
            Required if neutralization is "vol" or "vol_beta".
        betas: Optional series of market betas per ticker.
            Required if neutralization is "beta" or "vol_beta".
        sectors: Optional series of sector labels per ticker.
            Required if apply_sector_neutral is True.
    
    Returns:
        Dictionary mapping ticker to portfolio weight (sums to 1.0).
    
    Example:
        >>> config = EnhancedPortfolioConfig(top_k=25, weighting_scheme="rank", rank_tau=5.0)
        >>> scores = pd.Series({'AAPL': 2.1, 'MSFT': 1.8, 'GOOG': 1.5, ...})
        >>> weights = build_rank_weighted_portfolio(scores, config)
    """
    config = config or EnhancedPortfolioConfig()
    
    if len(scores) == 0:
        logger.warning("Empty scores provided")
        return {}
    
    # Select top-k stocks
    effective_k = min(config.top_k, len(scores))
    top_stocks = scores.nlargest(effective_k)
    selected_tickers = top_stocks.index.tolist()
    
    logger.debug(f"Selected top {len(selected_tickers)} stocks from {len(scores)}")
    
    # Step 1: Compute initial weights based on weighting scheme
    if config.weighting_scheme == "equal":
        raw_weights = pd.Series(
            1.0 / len(selected_tickers),
            index=selected_tickers,
        )
    
    elif config.weighting_scheme == "rank":
        # Assign ranks (1 = best)
        ranks = pd.Series(
            range(1, len(selected_tickers) + 1),
            index=selected_tickers,
        )
        # Exponential decay: w_i = exp(-rank_i / tau)
        raw_weights = np.exp(-ranks / config.rank_tau)
        # Normalize
        raw_weights = raw_weights / raw_weights.sum()
    
    elif config.weighting_scheme == "score":
        # Shift scores to be positive
        min_score = top_stocks.min()
        if min_score <= 0:
            shifted = top_stocks - min_score + 1e-6
        else:
            shifted = top_stocks
        raw_weights = shifted / shifted.sum()
    
    else:
        raise ValueError(f"Unknown weighting_scheme: {config.weighting_scheme}")
    
    weights = raw_weights.copy()
    
    # Step 2: Apply volatility scaling
    if config.neutralization in ("vol", "vol_beta"):
        if volatilities is None:
            logger.warning("Volatility neutralization requested but no volatilities provided")
        else:
            weights = _apply_vol_scaling(weights, volatilities, config.target_vol)
    
    # Step 3: Apply beta neutralization
    if config.neutralization in ("beta", "vol_beta"):
        if betas is None:
            logger.warning("Beta neutralization requested but no betas provided")
        else:
            weights = _apply_beta_neutralization(weights, betas)
    
    # Step 4: Apply sector constraints
    if config.apply_sector_neutral:
        if sectors is None:
            logger.warning("Sector neutralization requested but no sectors provided")
        else:
            weights = _apply_sector_constraints(weights, sectors, config.sector_max_weight)
    
    # Step 5: Apply position constraints
    weights = _apply_position_constraints(
        weights,
        max_weight=config.max_position_weight,
    )
    
    # Final normalization
    total = weights.sum()
    if total > 0:
        weights = weights / total
    
    portfolio = {ticker: float(w) for ticker, w in weights.items() if w > 0}
    
    logger.debug(
        f"Built portfolio: {len(portfolio)} positions, "
        f"max weight: {max(portfolio.values()):.2%}, "
        f"top3 conc: {sum(sorted(portfolio.values(), reverse=True)[:3]):.2%}"
    )
    
    return portfolio


# =============================================================================
# SECTOR-NEUTRAL RANKING PORTFOLIO
# =============================================================================

def build_sector_neutral_portfolio(
    scores: pd.Series,
    top_k: int,
    sectors: Optional[pd.Series] = None,
    weighting_scheme: Literal["equal", "rank", "score"] = "equal",
    rank_tau: float = 5.0,
    min_stocks_per_sector: int = 1,
    sector_mapping: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Build a sector-neutral portfolio by ranking within sectors first.
    
    This prevents the model from concentrating in one sector. The algorithm:
    1. Assign each stock to a sector
    2. Within each sector, rank stocks by score
    3. Compute within-sector z-scores (to normalize across sectors)
    4. Select top_k stocks based on z-scores, ensuring sector diversity
    5. Apply weighting scheme to selected stocks
    
    Args:
        scores: Series with tickers as index and prediction scores as values.
        top_k: Total number of stocks to select.
        sectors: Optional pre-computed sector series. If None, computed from mapping.
        weighting_scheme: How to weight selected stocks ("equal", "rank", "score").
        rank_tau: Temperature for rank weighting.
        min_stocks_per_sector: Minimum stocks from any represented sector.
        sector_mapping: Optional ticker -> sector mapping.
    
    Returns:
        Dictionary mapping ticker to portfolio weight (sums to 1.0).
    
    Example:
        >>> scores = pd.Series({'AAPL': 2.1, 'MSFT': 1.8, 'JPM': 1.5, 'BAC': 1.2})
        >>> weights = build_sector_neutral_portfolio(scores, top_k=2)
        >>> # Returns one tech, one financial instead of both tech
    """
    if len(scores) == 0:
        return {}
    
    if len(scores) <= top_k:
        # If fewer stocks than top_k, just equal weight all
        weight = 1.0 / len(scores)
        return {ticker: weight for ticker in scores.index}
    
    # Get sectors if not provided
    if sectors is None:
        sectors = get_ticker_sectors(scores.index.tolist(), sector_mapping)
    else:
        sectors = sectors.reindex(scores.index).fillna("Unknown")
    
    # Combine scores and sectors
    df = pd.DataFrame({
        "score": scores,
        "sector": sectors,
    })
    
    # Compute within-sector rank and z-score
    def sector_transform(group):
        """Compute rank and z-score within sector."""
        g = group.copy()
        g["sector_rank"] = g["score"].rank(ascending=False)
        
        # Z-score within sector (normalized score)
        mean = g["score"].mean()
        std = g["score"].std()
        if std > 0:
            g["sector_zscore"] = (g["score"] - mean) / std
        else:
            g["sector_zscore"] = 0.0
        
        return g
    
    df = df.groupby("sector", group_keys=False).apply(sector_transform)
    
    # Count stocks per sector
    sector_counts = df.groupby("sector").size()
    n_sectors = len(sector_counts)
    
    # Allocate slots per sector proportionally
    # Each sector gets at least min_stocks_per_sector if it has enough stocks
    slots_per_sector = {}
    remaining_slots = top_k
    
    # First pass: give each sector minimum
    for sector in sector_counts.index:
        n_in_sector = sector_counts[sector]
        slots = min(min_stocks_per_sector, n_in_sector)
        slots_per_sector[sector] = slots
        remaining_slots -= slots
    
    # Second pass: allocate remaining slots by sector size (proportional)
    if remaining_slots > 0 and n_sectors > 0:
        # Weight by number of stocks in sector
        sector_weights = sector_counts / sector_counts.sum()
        
        for sector in sector_counts.index:
            # How many more slots should this sector get?
            additional = int(remaining_slots * sector_weights[sector])
            # But cap at sector size minus what we already allocated
            max_additional = sector_counts[sector] - slots_per_sector[sector]
            additional = min(additional, max_additional)
            slots_per_sector[sector] += additional
    
    # Recompute remaining (may have rounding leftovers)
    total_allocated = sum(slots_per_sector.values())
    remaining_slots = top_k - total_allocated
    
    # Third pass: give remaining to sectors with best top z-scores
    if remaining_slots > 0:
        # Get best z-score per sector for stocks not yet selected
        sector_top_zscores = []
        for sector, row in df.groupby("sector"):
            n_allocated = slots_per_sector[sector]
            n_in_sector = len(row)
            if n_allocated < n_in_sector:
                # Get the (n_allocated + 1)th best z-score
                sorted_zscores = row["sector_zscore"].sort_values(ascending=False)
                next_zscore = sorted_zscores.iloc[n_allocated] if n_allocated < len(sorted_zscores) else -np.inf
                sector_top_zscores.append((sector, next_zscore))
        
        # Sort by z-score and allocate remaining slots
        sector_top_zscores.sort(key=lambda x: x[1], reverse=True)
        for sector, _ in sector_top_zscores:
            if remaining_slots <= 0:
                break
            # Can we add one more from this sector?
            if slots_per_sector[sector] < sector_counts[sector]:
                slots_per_sector[sector] += 1
                remaining_slots -= 1
    
    # Select top stocks from each sector
    selected_tickers = []
    for sector, n_slots in slots_per_sector.items():
        if n_slots <= 0:
            continue
        sector_df = df[df["sector"] == sector]
        # Sort by sector z-score (or raw score) and take top n_slots
        top_in_sector = sector_df.nlargest(n_slots, "sector_zscore").index.tolist()
        selected_tickers.extend(top_in_sector)
    
    # Build final scores for selected tickers
    selected_scores = scores.loc[selected_tickers]
    
    # Apply weighting scheme
    if weighting_scheme == "equal":
        weight = 1.0 / len(selected_tickers)
        portfolio = {ticker: weight for ticker in selected_tickers}
    
    elif weighting_scheme == "rank":
        # Rank across all selected (not within sector)
        ranks = selected_scores.rank(ascending=False)
        raw_weights = np.exp(-ranks / rank_tau)
        raw_weights = raw_weights / raw_weights.sum()
        portfolio = {ticker: float(w) for ticker, w in raw_weights.items()}
    
    elif weighting_scheme == "score":
        # Shift scores to be positive
        min_score = selected_scores.min()
        if min_score <= 0:
            shifted = selected_scores - min_score + 1e-6
        else:
            shifted = selected_scores
        normalized = shifted / shifted.sum()
        portfolio = {ticker: float(w) for ticker, w in normalized.items()}
    
    else:
        raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")
    
    # Log sector distribution
    sector_dist = {}
    for ticker in selected_tickers:
        s = sectors.get(ticker, "Unknown")
        sector_dist[s] = sector_dist.get(s, 0) + 1
    
    logger.info(
        f"Sector-neutral portfolio: {len(portfolio)} positions from {len(sector_dist)} sectors"
    )
    logger.debug(f"Sector distribution: {sector_dist}")
    
    return portfolio


def _apply_vol_scaling(
    weights: pd.Series,
    volatilities: pd.Series,
    target_vol: float,
) -> pd.Series:
    """Scale weights by inverse volatility to target a portfolio volatility.
    
    Simple approach: w_scaled_i = w_i / vol_i, then scale to target overall vol.
    This is a heuristic approximation (ignores correlations).
    """
    # Get volatilities for our stocks
    vols = volatilities.reindex(weights.index)
    
    # Handle missing volatilities
    if vols.isna().any():
        median_vol = vols.median()
        if pd.isna(median_vol):
            median_vol = 0.25  # Assume 25% vol
        vols = vols.fillna(median_vol)
    
    # Inverse vol weighting
    inv_vols = 1.0 / vols.clip(lower=0.01)  # Floor vol at 1%
    scaled = weights * inv_vols
    
    # Normalize
    scaled = scaled / scaled.sum()
    
    logger.debug(f"Applied vol scaling: target_vol={target_vol:.2%}")
    
    return scaled


def _apply_beta_neutralization(
    weights: pd.Series,
    betas: pd.Series,
) -> pd.Series:
    """Adjust weights to target zero portfolio beta.
    
    Simple approach: Tilt away from high-beta stocks.
    More sophisticated: Use optimization to target beta=0.
    
    Here we use a simple tilt: w_adjusted_i = w_i * (2 - beta_i) for beta < 2.
    This reduces weight of high-beta stocks.
    """
    # Get betas for our stocks
    stock_betas = betas.reindex(weights.index)
    
    # Handle missing betas
    if stock_betas.isna().any():
        stock_betas = stock_betas.fillna(1.0)  # Assume market beta
    
    # Compute current portfolio beta
    port_beta = (weights * stock_betas).sum()
    
    # Simple tilt: reduce high-beta exposure
    # Factor to apply: want to reduce average beta towards 1.0
    if port_beta > 1.0:
        # Tilt towards low beta stocks
        beta_adjustment = (2.0 - stock_betas).clip(lower=0.5, upper=1.5)
        adjusted = weights * beta_adjustment
    elif port_beta < 0.8:
        # Tilt towards higher beta (unlikely for long-only)
        beta_adjustment = stock_betas.clip(lower=0.5, upper=1.5)
        adjusted = weights * beta_adjustment
    else:
        adjusted = weights
    
    # Normalize
    adjusted = adjusted / adjusted.sum()
    
    new_port_beta = (adjusted * stock_betas).sum()
    logger.debug(f"Beta neutralization: {port_beta:.2f} -> {new_port_beta:.2f}")
    
    return adjusted


def _apply_sector_constraints(
    weights: pd.Series,
    sectors: pd.Series,
    max_sector_weight: float,
) -> pd.Series:
    """Apply sector concentration constraints.
    
    If a sector exceeds max_sector_weight, proportionally reduce its positions
    and redistribute to other sectors.
    """
    stock_sectors = sectors.reindex(weights.index)
    
    # Handle missing sectors
    if stock_sectors.isna().any():
        stock_sectors = stock_sectors.fillna("Unknown")
    
    adjusted = weights.copy()
    
    # Check each sector
    sector_weights = adjusted.groupby(stock_sectors).sum()
    
    for sector, sector_wt in sector_weights.items():
        if sector_wt > max_sector_weight:
            # Scale down this sector's positions
            scale_factor = max_sector_weight / sector_wt
            sector_mask = stock_sectors == sector
            adjusted[sector_mask] *= scale_factor
            
            logger.debug(f"Sector {sector}: {sector_wt:.2%} -> {max_sector_weight:.2%}")
    
    # Normalize
    adjusted = adjusted / adjusted.sum()
    
    return adjusted


def _apply_position_constraints(
    weights: pd.Series,
    max_weight: float,
) -> pd.Series:
    """Apply maximum position weight constraint with iterative clipping."""
    adjusted = weights.copy()
    
    # Iterative clipping (handle cascading effects)
    for _ in range(10):  # Max iterations
        excess = adjusted - max_weight
        if (excess > 0).any():
            # Clip excessive weights
            clipped = adjusted.clip(upper=max_weight)
            # Redistribute excess proportionally to non-maxed positions
            total_excess = excess[excess > 0].sum()
            non_maxed = adjusted < max_weight
            if non_maxed.any():
                redistribution = total_excess * (adjusted[non_maxed] / adjusted[non_maxed].sum())
                adjusted = clipped
                adjusted[non_maxed] += redistribution
            else:
                adjusted = clipped
                break
        else:
            break
    
    return adjusted


def compute_portfolio_risk_stats(
    weights: dict[str, float],
    volatilities: Optional[pd.Series] = None,
    betas: Optional[pd.Series] = None,
    sectors: Optional[pd.Series] = None,
) -> dict[str, float]:
    """Compute risk statistics for a portfolio.
    
    Args:
        weights: Portfolio weights.
        volatilities: Annualized volatilities per ticker.
        betas: Market betas per ticker.
        sectors: Sector labels per ticker.
    
    Returns:
        Dictionary with risk metrics.
    """
    if not weights:
        return {}
    
    weight_series = pd.Series(weights)
    tickers = weight_series.index.tolist()
    
    stats: dict[str, float] = {}
    
    # Concentration stats
    sorted_weights = sorted(weights.values(), reverse=True)
    stats["n_positions"] = len(weights)
    stats["max_weight"] = sorted_weights[0]
    stats["top3_concentration"] = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
    stats["top10_concentration"] = sum(sorted_weights[:10]) if len(sorted_weights) >= 10 else sum(sorted_weights)
    stats["hhi"] = sum(w**2 for w in weights.values())  # Herfindahl index
    
    # Volatility stats
    if volatilities is not None:
        vols = volatilities.reindex(tickers).fillna(volatilities.median())
        # Simple portfolio vol estimate (assuming zero correlation - lower bound)
        # True vol requires correlation matrix
        stats["avg_stock_vol"] = (weight_series * vols).sum()
        stats["vol_weighted_avg"] = (vols * weight_series).sum()
    
    # Beta stats
    if betas is not None:
        beta_values = betas.reindex(tickers).fillna(1.0)
        stats["portfolio_beta"] = (weight_series * beta_values).sum()
        stats["avg_stock_beta"] = (weight_series * beta_values).sum()
    
    # Sector stats
    if sectors is not None:
        sector_values = sectors.reindex(tickers).fillna("Unknown")
        sector_weights = weight_series.groupby(sector_values).sum()
        stats["n_sectors"] = len(sector_weights)
        stats["max_sector_weight"] = sector_weights.max()
        stats["largest_sector"] = sector_weights.idxmax()
    
    return stats


def build_long_short_portfolio(
    scores: pd.Series,
    long_k: int = 25,
    short_k: int = 25,
    config: Optional[EnhancedPortfolioConfig] = None,
    volatilities: Optional[pd.Series] = None,
    betas: Optional[pd.Series] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Build a long-short portfolio.
    
    Args:
        scores: Prediction scores.
        long_k: Number of long positions.
        short_k: Number of short positions.
        config: Portfolio configuration for each leg.
        volatilities: Volatilities for risk adjustment.
        betas: Betas for risk adjustment.
    
    Returns:
        Tuple of (long_weights, short_weights).
        Long weights are positive, short weights are negative.
    """
    config = config or EnhancedPortfolioConfig()
    
    # Build long portfolio from top scores
    long_config = EnhancedPortfolioConfig(
        top_k=long_k,
        weighting_scheme=config.weighting_scheme,
        rank_tau=config.rank_tau,
        neutralization=config.neutralization,
        max_position_weight=config.max_position_weight,
    )
    long_weights = build_rank_weighted_portfolio(
        scores,
        config=long_config,
        volatilities=volatilities,
        betas=betas,
    )
    
    # Build short portfolio from bottom scores (invert scores)
    short_config = EnhancedPortfolioConfig(
        top_k=short_k,
        weighting_scheme=config.weighting_scheme,
        rank_tau=config.rank_tau,
        neutralization=config.neutralization,
        max_position_weight=config.max_position_weight,
    )
    # Invert scores so worst become best
    inverted_scores = -scores
    short_weights_raw = build_rank_weighted_portfolio(
        inverted_scores,
        config=short_config,
        volatilities=volatilities,
        betas=betas,
    )
    
    # Make short weights negative
    short_weights = {t: -w for t, w in short_weights_raw.items()}
    
    return long_weights, short_weights


def compute_portfolio_expected_return(
    weights: dict[str, float],
    expected_returns: pd.Series,
) -> float:
    """Compute expected portfolio return from individual expected returns.
    
    Args:
        weights: Portfolio weights.
        expected_returns: Expected returns per ticker.
    
    Returns:
        Expected portfolio return.
    """
    total = 0.0
    for ticker, weight in weights.items():
        if ticker in expected_returns.index:
            er = expected_returns[ticker]
            if pd.notna(er):
                total += weight * er
    return total
