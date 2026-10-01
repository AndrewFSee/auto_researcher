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
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from auto_researcher.models.gbdt_model import GBDTModel

RegimeAssigner = Callable[[pd.Timestamp], str]

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

# Online regime labels (causal, computed from trailing market features only).
# Axis 1: trailing 12M SPY return sign (bull / bear)
# Axis 2: trailing 60d realized vol vs 5y quantile (calm / turbulent)
ONLINE_REGIME_LABELS = [
    "bull_calm",
    "bull_turbulent",
    "bear_calm",
    "bear_turbulent",
]


def assign_regime(date: pd.Timestamp) -> str:
    """
    Legacy calendar-based regime assignment — kept for backward compatibility.

    .. deprecated::
        Calendar boundaries were drawn with hindsight (e.g. the "2024-2026"
        bucket was named knowing what happened in 2024). Prefer
        :class:`CausalRegimeAssigner`, which only uses trailing market data.

    Current rules:
    - "2013-2016": dates before 2017-01-01
    - "2017-2019": 2017-01-01 <= date < 2020-01-01
    - "2020-2023": 2020-01-01 <= date < 2024-01-01
    - "2024-2026": dates >= 2024-01-01

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


class CausalRegimeAssigner:
    """
    Online (causal) regime classifier.

    Uses only trailing market data to label each date, so the regime at date t
    depends only on prices strictly before t. Two axes:

    * **Trend** — sign of the trailing 12M total return on the benchmark.
        Positive → "bull", non-positive → "bear".
    * **Volatility** — trailing 60-day realized vol vs an expanding-window
        quantile of the same series. Above the 60th percentile → "turbulent",
        else "calm".

    The cross-product gives four labels (see ``ONLINE_REGIME_LABELS``).

    The assigner is built once per training/backtest run; each ``assign(date)``
    call is O(1) and uses pre-computed labels that themselves only saw data
    up to ``date - 1``.

    Attributes:
        labels: Series of regime labels indexed by trading date. Each entry
            was computed using only prior prices.
    """

    def __init__(
        self,
        benchmark_prices: pd.Series,
        trend_window: int = 252,
        vol_window: int = 60,
        turbulence_quantile: float = 0.60,
        min_history: int = 252,
    ) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._turbulence_quantile = turbulence_quantile
        self._min_history = min_history
        self.labels: pd.Series = self._fit(benchmark_prices)

    def _fit(self, benchmark_prices: pd.Series) -> pd.Series:
        prices = benchmark_prices.dropna().sort_index()
        if len(prices) == 0:
            return pd.Series(dtype=object)

        # Trailing 12M total return — shifted by one bar so the value at t
        # only sees returns up to t-1.
        cumret = prices / prices.shift(self._trend_window) - 1.0
        cumret = cumret.shift(1)

        # Trailing realized vol (std of daily log-returns, annualized).
        log_rets = np.log(prices / prices.shift(1))
        vol = log_rets.rolling(self._vol_window).std() * np.sqrt(252)
        vol = vol.shift(1)

        # Expanding-quantile threshold for turbulence — at each t the cutoff
        # is the q-th percentile of vol values realized through t-1.
        threshold = vol.expanding(min_periods=self._min_history).quantile(
            self._turbulence_quantile
        )

        is_bull = cumret > 0
        is_turbulent = vol > threshold

        labels = pd.Series(index=prices.index, dtype=object)
        labels[is_bull & ~is_turbulent] = "bull_calm"
        labels[is_bull & is_turbulent] = "bull_turbulent"
        labels[~is_bull & ~is_turbulent] = "bear_calm"
        labels[~is_bull & is_turbulent] = "bear_turbulent"

        # Before we have enough history we simply don't classify.
        warmup_mask = (cumret.isna()) | (vol.isna()) | (threshold.isna())
        labels[warmup_mask] = pd.NA

        return labels

    def assign(self, date: pd.Timestamp) -> str:
        """
        Return the regime label for ``date`` using only data strictly before it.

        If ``date`` falls in the warm-up window (insufficient trailing history)
        the default regime label is returned so downstream code can still fall
        back on a global model.
        """
        if self.labels.empty:
            return DEFAULT_REGIME

        # Use asof to find the most recent label strictly before `date`.
        idx = self.labels.index
        pos = idx.searchsorted(pd.Timestamp(date), side="left") - 1
        if pos < 0:
            return DEFAULT_REGIME

        value = self.labels.iloc[pos]
        if pd.isna(value):
            return DEFAULT_REGIME
        return str(value)

    def __call__(self, date: pd.Timestamp) -> str:
        return self.assign(date)


def _resolve_assigner(assigner: RegimeAssigner | None) -> tuple[RegimeAssigner, list[str]]:
    """
    Return the (callable, label_list) pair to use for encoding regimes.

    When ``assigner`` is ``None`` we fall back to the legacy calendar assignment
    (``assign_regime`` + ``REGIME_LABELS``) for backwards compatibility. Passing
    a :class:`CausalRegimeAssigner` switches to causal ``ONLINE_REGIME_LABELS``.
    """
    if assigner is None:
        return assign_regime, REGIME_LABELS
    if isinstance(assigner, CausalRegimeAssigner):
        return assigner, ONLINE_REGIME_LABELS
    # Generic callable — assume it emits ONLINE labels unless caller tells us
    # otherwise via the ``labels`` attribute.
    labels = getattr(assigner, "labels_list", ONLINE_REGIME_LABELS)
    return assigner, list(labels)


def add_regime_feature(
    df: pd.DataFrame,
    assigner: RegimeAssigner | None = None,
) -> pd.DataFrame:
    """
    Add a regime_id column to a DataFrame based on its index (dates).

    For tree-based models, we encode regime as an integer category.

    Args:
        df: DataFrame with DatetimeIndex (or MultiIndex with dates at level 0).
        assigner: Optional causal regime assigner. When ``None`` the legacy
            calendar assignment is used (deprecated).

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

    assign_fn, labels = _resolve_assigner(assigner)
    regime_labels = [assign_fn(d) for d in dates]

    # Encode as integers for tree models (more efficient than one-hot)
    regime_to_int = {label: i for i, label in enumerate(labels)}
    regime_ids = [regime_to_int.get(r, 0) for r in regime_labels]

    df["regime_id"] = regime_ids

    return df


# ==============================================================================
# REGIME-AWARE TRAINING HELPERS
# ==============================================================================

def split_data_by_regime(
    X: pd.DataFrame,
    y: pd.Series,
    assigner: RegimeAssigner | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Split training data into subsets by regime.

    Args:
        X: Feature matrix with DatetimeIndex or MultiIndex (date at level 0).
        y: Target series aligned with X.
        assigner: Optional causal regime assigner. When ``None`` the legacy
            calendar assignment is used.

    Returns:
        Dict mapping regime_label -> (X_subset, y_subset).
    """
    result = {}

    # Get dates from index
    if isinstance(X.index, pd.MultiIndex):
        dates = X.index.get_level_values(0)
    else:
        dates = X.index

    assign_fn, labels = _resolve_assigner(assigner)
    regimes = pd.Series([assign_fn(d) for d in dates], index=X.index)

    for regime_label in labels:
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
    assigner: RegimeAssigner | None = None,
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
        assigner: Optional causal regime assigner.

    Returns:
        Selected model, or None if no suitable model found.
    """
    from auto_researcher.models.gbdt_model import GBDTModel  # Avoid circular import

    assign_fn, labels = _resolve_assigner(assigner)
    current_regime = assign_fn(date)

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
    regime_order = list(labels)
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
    assigner: RegimeAssigner | None = None,
) -> pd.DataFrame:
    """
    Prepare features for prediction based on regime mode.

    For FEATURE mode: adds regime_id column.
    For NONE/SPLIT: returns features unchanged.

    Args:
        features: Feature matrix for a single date (tickers as index).
        date: Current date for regime assignment.
        regime_mode: The regime mode being used.
        assigner: Optional causal regime assigner.

    Returns:
        Features with regime_id if needed.
    """
    if isinstance(regime_mode, str):
        regime_mode = RegimeMode(regime_mode)

    if regime_mode == RegimeMode.FEATURE:
        features = features.copy()
        assign_fn, labels = _resolve_assigner(assigner)
        regime = assign_fn(date)
        regime_to_int = {label: i for i, label in enumerate(labels)}
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
