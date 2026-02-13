"""
Regime-aware ML training and inference.

This module provides:
1. Calendar-based regime labelling (assign_regime)
2. RegimeMode enum for controlling training/inference behavior
3. Helper functions for regime-aware model training and selection

Regime Modes:
- "none":    Current behavior, no regime awareness
- "feature": Single model with regime_id as a categorical feature
- "split":   Separate models per regime, selected at inference time

The calendar-based regime assignment can be swapped for more sophisticated
methods (HMM, Wasserstein distance, etc.) by replacing assign_regime().
"""

import logging
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# REGIME DEFINITIONS
# ==============================================================================

# Calendar-based regime boundaries
# These align with the subperiod analysis in run_large_cap_backtest.py
REGIME_BOUNDARIES = [
    ("2013-2016", None, pd.Timestamp("2017-01-01")),          # Before 2017
    ("2017-2019", pd.Timestamp("2017-01-01"), pd.Timestamp("2020-01-01")),
    ("2020-2023", pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01")),
    ("2024-2026", pd.Timestamp("2024-01-01"), None),          # 2024 onwards
]

# All regime labels
REGIME_LABELS = [r[0] for r in REGIME_BOUNDARIES]

# Default regime (fallback for out-of-sample dates)
DEFAULT_REGIME = "2020-2023"


class RegimeMode(str, Enum):
    """
    Regime-aware ML training/inference mode.
    
    - NONE:    Current behavior, no regime awareness
    - FEATURE: Single model with regime_id as a categorical feature
    - SPLIT:   Separate models per regime, selected at inference time
    """
    NONE = "none"
    FEATURE = "feature"
    SPLIT = "split"


# ==============================================================================
# REGIME ASSIGNMENT
# ==============================================================================

def assign_regime(date: pd.Timestamp) -> str:
    """
    Assign a regime label to a date based on calendar rules.
    
    Current rules (calendar-based):
    - "2013-2016": dates before 2017-01-01
    - "2017-2019": 2017-01-01 <= date < 2020-01-01
    - "2020-2023": dates >= 2020-01-01
    
    This function is pure and can be swapped for HMM/Wasserstein/etc.
    
    Args:
        date: The date to classify.
    
    Returns:
        Regime label string.
    """
    for label, start, end in REGIME_BOUNDARIES:
        in_range = True
        if start is not None and date < start:
            in_range = False
        if end is not None and date >= end:
            in_range = False
        if in_range:
            return label
    
    # Fallback for dates outside defined ranges
    logger.warning(f"Date {date} outside regime boundaries, using default: {DEFAULT_REGIME}")
    return DEFAULT_REGIME


def add_regime_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a regime_id column to a DataFrame based on its index (dates).
    
    For tree-based models, we encode regime as an integer category.
    
    Args:
        df: DataFrame with DatetimeIndex (or MultiIndex with dates at level 0).
    
    Returns:
        DataFrame with 'regime_id' column added.
    """
    df = df.copy()
    
    # Handle MultiIndex (date, ticker) or simple DatetimeIndex
    if isinstance(df.index, pd.MultiIndex):
        # Assume dates are at level 0
        dates = df.index.get_level_values(0)
    else:
        dates = df.index
    
    # Assign regime labels
    regime_labels = [assign_regime(d) for d in dates]
    
    # Encode as integers for tree models (more efficient than one-hot)
    regime_to_int = {label: i for i, label in enumerate(REGIME_LABELS)}
    regime_ids = [regime_to_int.get(r, 0) for r in regime_labels]
    
    df["regime_id"] = regime_ids
    
    return df


# ==============================================================================
# REGIME-AWARE TRAINING HELPERS
# ==============================================================================

def split_data_by_regime(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Split training data into subsets by regime.
    
    Args:
        X: Feature matrix with DatetimeIndex or MultiIndex (date at level 0).
        y: Target series aligned with X.
    
    Returns:
        Dict mapping regime_label -> (X_subset, y_subset).
    """
    result = {}
    
    # Get dates from index
    if isinstance(X.index, pd.MultiIndex):
        dates = X.index.get_level_values(0)
    else:
        dates = X.index
    
    # Assign regimes
    regimes = pd.Series([assign_regime(d) for d in dates], index=X.index)
    
    for regime_label in REGIME_LABELS:
        mask = regimes == regime_label
        if mask.sum() > 0:
            result[regime_label] = (X[mask], y[mask])
            logger.info(f"Regime '{regime_label}': {mask.sum()} samples")
        else:
            # Debug level since this is expected early in the backtest
            logger.debug(f"Regime '{regime_label}': no samples available")
    
    return result


def select_model_for_regime(
    models: dict[str, "GBDTModel"],
    date: pd.Timestamp,
    fallback_model: "GBDTModel | None" = None,
) -> "GBDTModel | None":
    """
    Select the appropriate model for a given date based on its regime.
    
    Fallback logic (if the target regime model is missing):
    1. Try the fallback_model (global model trained on all data)
    2. Try the nearest available regime model (prefer earlier regimes)
    3. Return None if no model available
    
    Args:
        models: Dict mapping regime_label -> trained model.
        date: Current date for regime detection.
        fallback_model: Optional global model to use as fallback.
    
    Returns:
        Selected model, or None if no suitable model found.
    """
    from auto_researcher.models.gbdt_model import GBDTModel  # Avoid circular import
    
    current_regime = assign_regime(date)
    
    # First choice: model for current regime
    if current_regime in models and models[current_regime] is not None:
        return models[current_regime]
    
    # No model for this regime - use fallback (expected early in each regime)
    # Use debug level since this is normal walk-forward behavior
    logger.debug(f"No model for regime '{current_regime}' on {date}, using fallback")
    
    # Second choice: fallback to global model
    if fallback_model is not None:
        return fallback_model
    
    # Third choice: try nearest regime (prefer earlier)
    regime_order = REGIME_LABELS.copy()
    try:
        current_idx = regime_order.index(current_regime)
    except ValueError:
        current_idx = len(regime_order)
    
    # Check earlier regimes first, then later
    for offset in range(1, len(regime_order)):
        for delta in [-offset, offset]:
            idx = current_idx + delta
            if 0 <= idx < len(regime_order):
                alt_regime = regime_order[idx]
                if alt_regime in models and models[alt_regime] is not None:
                    logger.debug(f"Using model from regime '{alt_regime}' as fallback for {date}")
                    return models[alt_regime]
    
    logger.error(f"No model available for {date}")
    return None


def get_regime_aware_features(
    features: pd.DataFrame,
    date: pd.Timestamp,
    regime_mode: RegimeMode | str,
) -> pd.DataFrame:
    """
    Prepare features for prediction based on regime mode.
    
    For FEATURE mode: adds regime_id column.
    For NONE/SPLIT: returns features unchanged.
    
    Args:
        features: Feature matrix for a single date (tickers as index).
        date: Current date for regime assignment.
        regime_mode: The regime mode being used.
    
    Returns:
        Features with regime_id if needed.
    """
    if isinstance(regime_mode, str):
        regime_mode = RegimeMode(regime_mode)
    
    if regime_mode == RegimeMode.FEATURE:
        features = features.copy()
        regime = assign_regime(date)
        regime_to_int = {label: i for i, label in enumerate(REGIME_LABELS)}
        features["regime_id"] = regime_to_int.get(regime, 0)
    
    return features


# ==============================================================================
# STRATEGY NAMING
# ==============================================================================

def get_regime_strategy_suffix(regime_mode: RegimeMode | str) -> str:
    """
    Get the strategy name suffix for a given regime mode.
    
    Args:
        regime_mode: The regime mode.
    
    Returns:
        Suffix string (e.g., " [RegFeature]" or " [RegSplit]").
    """
    if isinstance(regime_mode, str):
        regime_mode = RegimeMode(regime_mode)
    
    if regime_mode == RegimeMode.NONE:
        return ""
    elif regime_mode == RegimeMode.FEATURE:
        return " [RegFeature]"
    elif regime_mode == RegimeMode.SPLIT:
        return " [RegSplit]"
    else:
        return ""
