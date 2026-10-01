"""
Baseline "models" for the walk-forward harness.

Every learned model should be reported next to these. If an ML ranker cannot
beat a single well-known factor, or the equal-weight universe, under the same
purged protocol and costs, its complexity is not paying for itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureScoreModel:
    """
    Score stocks by a single feature column; ``fit`` is a no-op.

    Example:
        ``FeatureScoreModel("tech_resid_mom_252")`` is residual momentum,
        ``FeatureScoreModel("tech_mom_5d", sign=-1)`` is short-term reversal.
    """

    def __init__(self, feature: str, sign: float = 1.0) -> None:
        if sign == 0:
            raise ValueError("sign must be non-zero")
        self.feature = feature
        self.sign = float(np.sign(sign))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureScoreModel:
        if self.feature not in X.columns:
            raise KeyError(f"feature {self.feature!r} not in training columns")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.sign * X[self.feature].to_numpy(dtype=float), dtype=float)


class LinearICModel:
    """
    Transparent linear baseline: each feature is weighted by the sign and size
    of its mean per-date Spearman IC on the training window, then the weighted
    (cross-sectionally ranked) features are summed.

    It has no tunable hyperparameters, so it is a fair "is the tree ensemble
    adding anything?" comparison for the gradient-boosted rankers.
    """

    def __init__(self, min_abs_ic: float = 0.0) -> None:
        self.min_abs_ic = min_abs_ic
        self.weights_: pd.Series | None = None

    @staticmethod
    def _rank_by_date(X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X.index, pd.MultiIndex):
            return X.groupby(level=0).rank(pct=True) - 0.5
        return X.rank(pct=True) - 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearICModel:
        ranked = self._rank_by_date(X)
        y_ranked = (
            y.groupby(level=0).rank(pct=True) - 0.5
            if isinstance(y.index, pd.MultiIndex)
            else y.rank(pct=True) - 0.5
        )
        # Mean of per-date rank correlations == mean Spearman IC per feature.
        frame = ranked.assign(__y=y_ranked)
        if isinstance(X.index, pd.MultiIndex):
            per_date = frame.groupby(level=0).corr()["__y"].unstack().drop(columns="__y")
            ic = per_date.mean()
        else:
            ic = frame.corr()["__y"].drop("__y")
        ic = ic.fillna(0.0)
        ic[ic.abs() < self.min_abs_ic] = 0.0
        self.weights_ = ic
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("model has not been fitted")
        ranked = self._rank_by_date(X[self.weights_.index]).fillna(0.0)
        return np.asarray(ranked.to_numpy() @ self.weights_.to_numpy(), dtype=float)
