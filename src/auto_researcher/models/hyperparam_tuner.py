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
        n_cv_splits: Number of *inner* purged time-series splits used for tuning.
        metric: Optimization metric ("ic" for Spearman IC, "mse" for MSE).
        model_type: Model type to tune.
        timeout_seconds: Max seconds per tuning round (None = no limit).
        purge_days: Number of trading days to drop between inner-train and
            inner-val. Set to the label horizon so training labels don't
            overlap into the validation window.
        embargo_days: Extra buffer after the inner-val window before it ends.
            Usually 0 inside the tuner (caller's outer walk-forward already
            embargoes the outer test).
        use_early_stopping: Whether to let XGBoost early-stop. When True, a
            third *inner-stop* slice is carved from the tail of inner-train
            (see ``stop_slice_frac``) and used as ``eval_set`` — never the
            inner-val slice that Optuna scores on. Without that disjointness
            early stopping picks ``n_estimators`` to nail the val fold and
            Optuna then picks the trial whose params nailed the same fold,
            inflating the reported score.
        stop_slice_frac: Fraction of inner-train (by unique-date count) to
            reserve as the early-stopping slice. Taken from the most recent
            tail of inner-train and purged from the remainder. Ignored when
            ``use_early_stopping=False``.
    """
    n_trials: int = 20
    n_cv_splits: int = 3
    metric: Literal["ic", "mse"] = "ic"
    model_type: Literal["regression", "rank_pairwise"] = "regression"
    timeout_seconds: int | None = 120
    purge_days: int = 21
    embargo_days: int = 0
    use_early_stopping: bool = True
    stop_slice_frac: float = 0.15


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation (Information Coefficient)."""
    from scipy.stats import spearmanr
    if len(y_true) < 5:
        return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return ic if np.isfinite(ic) else 0.0


def _extract_dates(X: pd.DataFrame) -> pd.DatetimeIndex | None:
    """
    Pull the date axis out of a feature matrix. Returns ``None`` when we cannot
    find one — caller falls back to row-index splits.
    """
    if isinstance(X.index, pd.MultiIndex):
        # Date is assumed to be at level 0 of the MultiIndex.
        level = X.index.get_level_values(0)
        if isinstance(level, pd.DatetimeIndex):
            return level
    if isinstance(X.index, pd.DatetimeIndex):
        return X.index
    return None


def _purged_time_series_splits(
    dates: pd.DatetimeIndex | None,
    n_samples: int,
    n_splits: int,
    purge_days: int,
    embargo_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding-window time-series splits with a purge gap.

    When ``dates`` is available the purge/embargo is measured in calendar days,
    which is the right unit for panel data where a single row-index step can
    cover many stocks on the same date. Otherwise we approximate with row
    counts.

    Each split:
      * inner_train = rows whose date < cutoff_train
      * (purge gap of ``purge_days``)
      * inner_val   = rows whose date in [cutoff_val_start, cutoff_val_end]
    Training labels that overlap into the validation window are dropped by
    the purge, so rolling-horizon leakage does not inflate the tuning score.
    """
    if n_samples == 0:
        return []

    if dates is None:
        # Fall back to row-count splits (still purged by n-rows approximation).
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        fold_size = n_samples // (n_splits + 1)
        if fold_size < 20:
            split_point = int(n_samples * 0.8)
            purge = min(purge_days, max(split_point - 1, 0))
            train_idx = np.arange(0, split_point - purge)
            val_idx = np.arange(split_point, n_samples)
            if len(train_idx) > 0 and len(val_idx) > 0:
                return [(train_idx, val_idx)]
            return []

        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)
            if val_end <= val_start:
                continue
            purge = min(purge_days, max(train_end - 1, 0))
            train_idx = np.arange(0, train_end - purge)
            val_idx = np.arange(val_start, val_end)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            splits.append((train_idx, val_idx))
        return splits

    unique_dates = pd.DatetimeIndex(dates).unique().sort_values()
    n_unique = len(unique_dates)
    if n_unique < 4:
        # Not enough distinct dates to form a meaningful split.
        split_point = int(n_samples * 0.8)
        purge = min(purge_days, max(split_point - 1, 0))
        train_idx = np.arange(0, split_point - purge)
        val_idx = np.arange(split_point, n_samples)
        if len(train_idx) > 0 and len(val_idx) > 0:
            return [(train_idx, val_idx)]
        return []

    fold_size = max(n_unique // (n_splits + 1), 1)
    row_dates = pd.DatetimeIndex(dates)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    purge_td = pd.Timedelta(days=int(purge_days))
    embargo_td = pd.Timedelta(days=int(embargo_days))

    for i in range(n_splits):
        train_end_idx = fold_size * (i + 1)
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + fold_size, n_unique)
        if val_end_idx <= val_start_idx:
            continue

        train_cutoff = unique_dates[train_end_idx - 1]
        val_start_date = unique_dates[val_start_idx]
        val_end_date = unique_dates[val_end_idx - 1]

        train_mask = row_dates <= (train_cutoff - purge_td)
        val_mask = (row_dates >= val_start_date + embargo_td) & (
            row_dates <= val_end_date
        )

        train_idx = np.flatnonzero(np.asarray(train_mask))
        val_idx = np.flatnonzero(np.asarray(val_mask))
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))

    return splits


def _carve_inner_stop_slice(
    train_idx: np.ndarray,
    dates: pd.DatetimeIndex | None,
    stop_frac: float,
    purge_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a contiguous, date-sorted ``train_idx`` into ``(fit_idx, stop_idx)``.

    ``stop_idx`` is the most recent ``stop_frac`` of unique dates inside
    ``train_idx``; ``fit_idx`` is everything before, with a ``purge_days``
    calendar-day gap dropped between the two so labels with overlapping
    forward-return windows don't leak from fit into stop.

    Returns empty arrays for ``stop_idx`` when ``train_idx`` is too small to
    carve a meaningful stop slice — the caller should then disable early
    stopping for this fold rather than scoring against the val slice.
    """
    if len(train_idx) == 0 or stop_frac <= 0.0:
        return train_idx, np.empty(0, dtype=train_idx.dtype)

    if dates is None:
        # Row-count fallback: take the last ``stop_frac`` of rows, no purge.
        stop_n = max(int(round(len(train_idx) * stop_frac)), 1)
        if stop_n >= len(train_idx):
            return train_idx, np.empty(0, dtype=train_idx.dtype)
        return train_idx[:-stop_n], train_idx[-stop_n:]

    train_dates = pd.DatetimeIndex(dates.take(train_idx))
    unique = train_dates.unique().sort_values()
    if len(unique) < 4:
        return train_idx, np.empty(0, dtype=train_idx.dtype)

    n_stop_dates = max(int(round(len(unique) * stop_frac)), 1)
    if n_stop_dates >= len(unique):
        return train_idx, np.empty(0, dtype=train_idx.dtype)

    stop_start_date = unique[-n_stop_dates]
    fit_cutoff_date = stop_start_date - pd.Timedelta(days=int(purge_days))

    fit_mask = train_dates <= fit_cutoff_date
    stop_mask = train_dates >= stop_start_date

    fit_idx = train_idx[np.asarray(fit_mask)]
    stop_idx = train_idx[np.asarray(stop_mask)]
    return fit_idx, stop_idx


def _groups_from_dates(dates: pd.DatetimeIndex, indices: np.ndarray) -> np.ndarray:
    """
    Build the per-date ``group`` array that XGBRanker expects.

    Each group is the count of rows on the same date, in the order those rows
    appear in ``indices``. This is critical for rank:pairwise / lambdarank —
    using a single group ``[len(y)]`` treats the entire fold as one list,
    which silently breaks the cross-sectional ranking objective.
    """
    selected = pd.DatetimeIndex(dates.take(indices))
    # Group sizes must preserve the order rows were passed to fit, so we do
    # not sort — we count consecutive runs of identical dates.
    changes = np.concatenate(
        ([True], selected[1:].to_numpy() != selected[:-1].to_numpy())
    )
    start_positions = np.flatnonzero(changes)
    end_positions = np.append(start_positions[1:], len(selected))
    return (end_positions - start_positions).astype(np.int64)


def _ensure_date_sorted(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex | None]:
    """Return X, y reordered so rows on the same date are contiguous."""
    dates = _extract_dates(X)
    if dates is None:
        return X, y, None

    order = np.argsort(dates.to_numpy(), kind="mergesort")
    if np.all(order == np.arange(len(order))):
        return X, y, dates

    X_sorted = X.iloc[order]
    y_sorted = y.iloc[order]
    dates_sorted = _extract_dates(X_sorted)
    return X_sorted, y_sorted, dates_sorted


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

    # Sort rows by date so groups (and purged splits) are well-defined even
    # when the caller passed rows in shuffled order.
    X_train, y_train, dates = _ensure_date_sorted(X_train, y_train)

    # Generate purged time-series CV splits. ``purge_days`` should be at least
    # the label horizon: the caller's outer walk-forward handles outer-test
    # embargo, but inside the tuner we still need to drop training labels
    # whose forward return peeks into the inner-val window.
    splits = _purged_time_series_splits(
        dates,
        n_samples=n_samples,
        n_splits=config.n_cv_splits,
        purge_days=config.purge_days,
        embargo_days=config.embargo_days,
    )
    if not splits:
        logger.warning("Purged splits came back empty, returning default params")
        return _default_params(base_params)

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
            # Carve a third inner-stop slice for early stopping, disjoint from
            # both inner-train (purged) and inner-val (separate slice). Without
            # this, early-stopping picks n_estimators to fit val_idx and Optuna
            # then scores trials on the same val_idx, double-counting fit and
            # inflating the reported score.
            if config.use_early_stopping:
                fit_idx, stop_idx = _carve_inner_stop_slice(
                    train_idx,
                    dates,
                    stop_frac=config.stop_slice_frac,
                    purge_days=config.purge_days,
                )
                trial_use_es = len(stop_idx) > 0 and len(fit_idx) > 0
            else:
                fit_idx, stop_idx = train_idx, np.empty(0, dtype=train_idx.dtype)
                trial_use_es = False

            X_tr, X_val = X_vals[fit_idx], X_vals[val_idx]
            y_tr, y_val = y_vals[fit_idx], y_vals[val_idx]
            if trial_use_es:
                X_stop, y_stop = X_vals[stop_idx], y_vals[stop_idx]

            if config.model_type == "regression":
                kwargs: dict = dict(
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=42,
                    **params,
                )
                if trial_use_es:
                    kwargs["early_stopping_rounds"] = 20
                model = xgb.XGBRegressor(**kwargs)
                fit_kwargs = {"verbose": False}
                if trial_use_es:
                    fit_kwargs["eval_set"] = [(X_stop, y_stop)]
                model.fit(X_tr, y_tr, **fit_kwargs)
                preds = model.predict(X_val)
            else:
                # Ranking model: per-date groups so rank:pairwise / lambdarank
                # compare stocks cross-sectionally within the same date.
                if dates is None:
                    # Can't reconstruct groups without dates; treat the whole
                    # fold as one group and accept the known limitation.
                    group_tr = np.array([len(y_tr)], dtype=np.int64)
                    group_val = np.array([len(y_val)], dtype=np.int64)
                    group_stop = np.array([len(y_stop)], dtype=np.int64) if trial_use_es else None
                else:
                    group_tr = _groups_from_dates(dates, fit_idx)
                    group_val = _groups_from_dates(dates, val_idx)
                    group_stop = _groups_from_dates(dates, stop_idx) if trial_use_es else None

                kwargs = dict(
                    objective="rank:pairwise",
                    n_jobs=-1,
                    random_state=42,
                    **params,
                )
                if trial_use_es:
                    kwargs["early_stopping_rounds"] = 20
                model = xgb.XGBRanker(**kwargs)
                fit_kwargs = {"verbose": False, "group": group_tr}
                if trial_use_es:
                    fit_kwargs["eval_set"] = [(X_stop, y_stop)]
                    fit_kwargs["eval_group"] = [group_stop]
                model.fit(X_tr, y_tr, **fit_kwargs)
                preds = model.predict(X_val)

            if config.metric == "ic":
                scores.append(_spearman_ic(y_val, preds))
            else:
                scores.append(-np.mean((y_val - preds) ** 2))  # Negative MSE

        return float(np.mean(scores))

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

    # Multiple-testing deflation of the best trial value (Lopez de Prado/Bailey).
    # With an IC objective, the expected-max-IC-under-null scales with
    # sqrt(2 log N_trials) times the cross-trial stdev. If the reported best
    # value is close to that null expectation, it's likely a fluke.
    global _LAST_TUNING_DIAGNOSTICS
    try:
        trial_values = np.asarray(
            [t.value for t in study.trials if t.value is not None], dtype=float
        )
        n_trials_finite = int(np.isfinite(trial_values).sum())
        trial_std = (
            float(np.nanstd(trial_values, ddof=1))
            if n_trials_finite > 1 else 1.0
        )
        if trial_std <= 0:
            trial_std = 1.0
        # expected_max_sharpe_under_null gives E[max of N draws from N(0, s)].
        from auto_researcher.validation.deflated_sharpe import (
            expected_max_sharpe_under_null,
        )
        exp_max_null = expected_max_sharpe_under_null(
            n_trials=n_trials_finite, sharpe_std=trial_std
        )
        deflated = float(study.best_value) - exp_max_null
        _LAST_TUNING_DIAGNOSTICS = {
            "n_trials": n_trials_finite,
            "best_value": float(study.best_value),
            "trial_std": trial_std,
            "expected_max_under_null": float(exp_max_null),
            "deflated_best_value": deflated,
            "metric": config.metric,
        }
        if deflated <= 0:
            logger.warning(
                f"Deflated {config.metric}={deflated:.4f} (best={study.best_value:.4f}, "
                f"E[max|null]={exp_max_null:.4f} across {n_trials_finite} trials). "
                "Tuning result indistinguishable from selection bias; "
                "consider the fold's default params."
            )
        else:
            logger.info(
                f"Deflated {config.metric}={deflated:.4f} "
                f"(best={study.best_value:.4f}, "
                f"E[max|null]={exp_max_null:.4f}, N={n_trials_finite})"
            )
    except Exception as exc:  # pragma: no cover - diagnostics never block tuning
        logger.debug("Could not compute tuning deflation: %s", exc)
        _LAST_TUNING_DIAGNOSTICS = None

    return best


_LAST_TUNING_DIAGNOSTICS: dict | None = None


def get_last_tuning_diagnostics() -> dict | None:
    """Return the deflation diagnostics from the most recent tuning call."""
    return _LAST_TUNING_DIAGNOSTICS


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
