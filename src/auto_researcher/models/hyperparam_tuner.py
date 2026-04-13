"""
Walk-Forward Hyperparameter Optimization.

Tunes XGBoost/LightGBM hyperparameters per walk-forward fold using
time-series-safe cross-validation. Uses Optuna TPE sampler for
Bayesian optimization within each fold's training window.

Usage:
    from auto_researcher.models.hyperparam_tuner import tune_xgb_hyperparams

    best_params = tune_xgb_hyperparams(
        X_train, y_train,
        n_trials=20,
        model_type="regression",
    )
    # best_params is a dict ready to pass to XGBRegressionConfig or XGBRankingConfig
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@dataclass
class TunerConfig:
    """Configuration for walk-forward hyperparameter tuning.

    Attributes:
        n_trials: Number of Optuna trials per fold.
        n_cv_splits: Number of time-series CV splits within the fold.
        metric: Optimization metric ("ic" for Spearman IC, "mse" for MSE).
        model_type: Model type to tune.
        timeout_seconds: Max seconds per tuning round (None = no limit).
    """
    n_trials: int = 20
    n_cv_splits: int = 3
    metric: Literal["ic", "mse"] = "ic"
    model_type: Literal["regression", "rank_pairwise"] = "regression"
    timeout_seconds: int | None = 120


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation (Information Coefficient)."""
    from scipy.stats import spearmanr
    if len(y_true) < 5:
        return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return ic if np.isfinite(ic) else 0.0


def _time_series_splits(
    n_samples: int, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding-window time-series splits.

    Each split uses the first k/(n_splits+1) of data for training
    and the next chunk for validation.
    """
    splits = []
    fold_size = n_samples // (n_splits + 1)
    if fold_size < 20:
        # Not enough data for multiple splits - use single 80/20 split
        split_point = int(n_samples * 0.8)
        train_idx = np.arange(0, split_point)
        val_idx = np.arange(split_point, n_samples)
        return [(train_idx, val_idx)]

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_start = train_end
        val_end = min(val_start + fold_size, n_samples)
        if val_end <= val_start:
            continue
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))

    return splits


def tune_xgb_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TunerConfig | None = None,
    base_params: dict | None = None,
) -> dict:
    """
    Tune XGBoost hyperparameters using Optuna with time-series CV.

    Args:
        X_train: Training features (already cleaned, no NaN).
        y_train: Training targets.
        config: Tuner configuration.
        base_params: Starting parameter values (used as search center).

    Returns:
        Dictionary of best hyperparameters:
        {max_depth, learning_rate, n_estimators, subsample,
         colsample_bytree, reg_lambda, reg_alpha}
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna not installed, returning default hyperparameters")
        return _default_params(base_params)

    config = config or TunerConfig()
    n_samples = len(X_train)

    if n_samples < 100:
        logger.warning(f"Only {n_samples} training samples, skipping tuning")
        return _default_params(base_params)

    # Generate time-series CV splits
    splits = _time_series_splits(n_samples, config.n_cv_splits)

    X_vals = X_train.values
    y_vals = y_train.values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        }

        import xgboost as xgb

        scores = []
        for train_idx, val_idx in splits:
            X_tr, X_val = X_vals[train_idx], X_vals[val_idx]
            y_tr, y_val = y_vals[train_idx], y_vals[val_idx]

            if config.model_type == "regression":
                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=42,
                    early_stopping_rounds=20,
                    **params,
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                preds = model.predict(X_val)
            else:
                # Ranking model: create group structure
                model = xgb.XGBRanker(
                    objective="rank:pairwise",
                    n_jobs=-1,
                    random_state=42,
                    early_stopping_rounds=20,
                    **params,
                )
                # Single group for the whole fold
                model.fit(
                    X_tr, y_tr,
                    group=[len(y_tr)],
                    eval_set=[(X_val, y_val)],
                    eval_group=[[len(y_val)]],
                    verbose=False,
                )
                preds = model.predict(X_val)

            if config.metric == "ic":
                scores.append(_spearman_ic(y_val, preds))
            else:
                scores.append(-np.mean((y_val - preds) ** 2))  # Negative MSE

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Enqueue default params as first trial for warm start
    defaults = _default_params(base_params)
    study.enqueue_trial(defaults)

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout_seconds,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info(
        f"Tuning complete: {len(study.trials)} trials, "
        f"best {config.metric}={study.best_value:.4f}, "
        f"best params: depth={best['max_depth']}, lr={best['learning_rate']:.4f}, "
        f"n_est={best['n_estimators']}"
    )

    return best


def _default_params(base_params: dict | None = None) -> dict:
    """Return default hyperparameters."""
    defaults = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
    }
    if base_params:
        defaults.update(base_params)
    return defaults
