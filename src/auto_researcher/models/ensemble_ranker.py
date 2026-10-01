"""
IC-weighted ensemble of rankers.

Why this and not a plain average
--------------------------------
Different rankers have different out-of-sample IC in different regimes.
A fixed equal-weight blend throws away that information — a calibration
fold often tells you (e.g.) the transformer is carrying twice the signal
of the GNN, and you should lean accordingly. This module takes model
predictions on a calibration fold, measures each model's per-date
Spearman IC against realized returns, and uses those ICs as the blend
weights. Models with non-positive IC on the calibration fold get
dropped (treated as noise).

This is the Phase 3.3 "Stage-2 composite" from the plan: XGBoost +
Transformer + GNN with IC-calibrated weights from the CPCV fold.

Usage::

    ens = ICWeightedEnsemble(models={
        "xgb":   xgb_model,
        "tx":    transformer_model,
        "gnn":   gnn_model,
    })
    ens.fit_weights(X_cal, y_cal)
    preds = ens.predict_with_index(X_oos)

All child models must already be trained — this class only blends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _RankerLike(Protocol):
    """Duck-typed protocol for any fit-once-then-predict ranker."""

    def predict_with_index(self, X: pd.DataFrame) -> pd.Series: ...


@dataclass
class EnsembleWeights:
    """Per-model blend coefficients and the IC that produced them."""

    weights: dict[str, float]
    per_model_ic: dict[str, float]
    dropped: dict[str, str] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for name, ic in self.per_model_ic.items():
            rows.append({
                "model": name,
                "mean_ic": ic,
                "weight": self.weights.get(name, 0.0),
                "dropped_reason": self.dropped.get(name, ""),
            })
        return pd.DataFrame(rows).sort_values("weight", ascending=False)


class ICWeightedEnsemble:
    """Blend already-trained rankers by calibration-fold IC.

    Model predictions are standardized (z-score within each date) before
    blending so raw score scales don't dominate the blend — only the
    cross-sectional ordering matters.
    """

    def __init__(
        self,
        models: Mapping[str, _RankerLike],
        min_ic: float = 0.0,
    ):
        """
        Args:
            models: ``{name: ranker}`` — each ranker must already be fit
                and support ``predict_with_index(X)``.
            min_ic: Drop any model whose calibration IC is below this
                floor. Default 0.0 — keep only strictly positive signals.
        """
        if not models:
            raise ValueError("At least one model required")
        self.models = dict(models)
        self.min_ic = float(min_ic)
        self._calibrated: EnsembleWeights | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit_weights(
        self, X_cal: pd.DataFrame, y_cal: pd.Series
    ) -> EnsembleWeights:
        """Compute per-model calibration IC → normalized blend weights.

        A model's weight = max(IC, 0) / Σ max(IC, 0). Models with IC ≤
        ``min_ic`` are dropped with a human-readable reason.
        """
        if not isinstance(X_cal.index, pd.MultiIndex):
            raise ValueError("X_cal must have MultiIndex (date, ticker)")
        y_cal = y_cal.reindex(X_cal.index)

        per_model_ic: dict[str, float] = {}
        dropped: dict[str, str] = {}

        for name, model in self.models.items():
            preds = model.predict_with_index(X_cal)
            ic = _mean_cross_sectional_ic(preds, y_cal)
            per_model_ic[name] = ic
            if not np.isfinite(ic):
                dropped[name] = "IC is NaN"
            elif ic <= self.min_ic:
                dropped[name] = f"IC={ic:.3f} ≤ min_ic={self.min_ic}"

        kept_ics = {
            n: ic for n, ic in per_model_ic.items() if n not in dropped
        }
        total = sum(kept_ics.values())
        if total <= 0:
            logger.warning(
                "ICWeightedEnsemble: all models dropped — falling back to "
                "equal weight across ALL input models (blend may be noise)"
            )
            n = len(self.models)
            weights = {name: 1.0 / n for name in self.models}
            dropped = {}
        else:
            weights = {n: ic / total for n, ic in kept_ics.items()}

        self._calibrated = EnsembleWeights(
            weights=weights,
            per_model_ic=per_model_ic,
            dropped=dropped,
        )
        logger.info(
            "ICWeightedEnsemble weights: %s",
            {k: round(v, 3) for k, v in weights.items()},
        )
        return self._calibrated

    def weights(self) -> dict[str, float]:
        if self._calibrated is None:
            raise ValueError("call fit_weights() first")
        return dict(self._calibrated.weights)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_with_index(self, X: pd.DataFrame) -> pd.Series:
        """Blend per-date z-scored predictions with the fitted weights."""
        if self._calibrated is None:
            raise ValueError("call fit_weights() first")

        out = pd.Series(0.0, index=X.index, name="prediction")
        active = self._calibrated.weights
        for name, w in active.items():
            if w == 0.0:
                continue
            preds = self.models[name].predict_with_index(X)
            z = _zscore_per_date(preds)
            out = out.add(z.mul(w), fill_value=0.0)
        return out

    def rank_cross_sectionally(self, X: pd.DataFrame) -> pd.Series:
        preds = self.predict_with_index(X)
        return preds.groupby(level=0).rank(ascending=False, method="first").rename("rank")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_cross_sectional_ic(preds: pd.Series, targets: pd.Series) -> float:
    """Average Spearman IC across dates."""
    df = pd.concat([preds.rename("p"), targets.rename("y")], axis=1).dropna()
    if df.empty:
        return float("nan")
    ics = []
    for _, sub in df.groupby(level=0):
        if len(sub) < 3:
            continue
        ics.append(sub["p"].rank().corr(sub["y"].rank()))
    if not ics:
        return float("nan")
    return float(np.nanmean(ics))


def _zscore_per_date(preds: pd.Series) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        std = s.std()
        if not np.isfinite(std) or std < 1e-12:
            return s * 0.0
        return (s - s.mean()) / std
    return preds.groupby(level=0).transform(_z)
