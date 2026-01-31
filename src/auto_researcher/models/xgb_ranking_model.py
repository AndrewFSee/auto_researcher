"""
XGBoost ranking and regression models for stock prediction.

This module implements:
- XGBRankingModel: Uses rank:pairwise objective for learning-to-rank
- XGBRegressionModel: Uses regression for continuous target prediction (e.g., vol-normalized returns)
"""

from dataclasses import dataclass, field
import logging
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

logger = logging.getLogger(__name__)


@dataclass
class XGBRankingConfig:
    """Configuration for XGBoost ranking model.
    
    Attributes:
        objective: Ranking objective to use:
            - "rank:pairwise": Pairwise ranking (RankNet-style)
            - "rank:ndcg": NDCG optimization
            - "rank:map": MAP optimization
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage (eta).
        reg_lambda: L2 regularization term (lambda).
        reg_alpha: L1 regularization term (alpha).
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns for each tree.
        min_child_weight: Minimum sum of instance weight in a child.
        gamma: Minimum loss reduction for further partition.
        early_stopping_rounds: Stop if no improvement for this many rounds.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel threads (-1 for all cores).
        verbose: Verbosity level for training.
    """
    
    objective: Literal["rank:pairwise", "rank:ndcg", "rank:map"] = "rank:pairwise"
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    reg_lambda: float = 2.0
    reg_alpha: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    gamma: float = 0.0
    early_stopping_rounds: Optional[int] = 50
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0


class XGBRankingModel:
    """XGBoost model with learning-to-rank objective.
    
    This model is designed for cross-sectional stock ranking. It uses XGBoost's
    ranking objectives which learn to correctly order items within groups (dates).
    
    The key difference from regression models:
    - Ranking models optimize the relative ordering, not absolute values
    - They use query/group information to define ranking contexts
    - The pairwise objective compares all pairs within a group
    
    Example:
        >>> config = XGBRankingConfig(objective="rank:pairwise")
        >>> model = XGBRankingModel(config)
        >>> 
        >>> # X has MultiIndex (date, ticker), groups define stocks per date
        >>> groups = X.groupby(level=0).size().values  # stocks per date
        >>> model.fit(X, y, groups=groups)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, config: Optional[XGBRankingConfig] = None):
        """Initialize the XGBoost ranking model.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )
        
        self.config = config or XGBRankingConfig()
        self.model: Optional[xgb.XGBRanker] = None
        self.feature_names: Optional[list[str]] = None
        self._eval_results: dict = {}
    
    def _create_model(self) -> xgb.XGBRanker:
        """Create a fresh XGBRanker instance with configured parameters."""
        return xgb.XGBRanker(
            objective=self.config.objective,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            reg_lambda=self.config.reg_lambda,
            reg_alpha=self.config.reg_alpha,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=self.config.verbose,
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        groups_val: Optional[np.ndarray] = None,
    ) -> "XGBRankingModel":
        """Fit the ranking model.
        
        Args:
            X: Feature matrix with shape (n_samples, n_features).
                Should have MultiIndex (date, ticker) for proper group extraction.
            y: Target values (returns or ranks).
            groups: Array specifying number of items in each query group.
                If None and X has MultiIndex, will compute from date level.
            X_val: Optional validation feature matrix.
            y_val: Optional validation targets.
            groups_val: Optional validation groups.
        
        Returns:
            Self for method chaining.
        """
        self.feature_names = list(X.columns)
        
        # Auto-compute groups from MultiIndex if not provided
        if groups is None and isinstance(X.index, pd.MultiIndex):
            groups = X.groupby(level=0).size().values
            logger.info(f"Auto-computed {len(groups)} query groups from MultiIndex")
        
        if groups is None:
            raise ValueError(
                "groups must be provided or X must have MultiIndex (date, ticker)"
            )
        
        # Log training info
        n_groups = len(groups)
        avg_group_size = np.mean(groups)
        logger.info(
            f"Training XGBRanker with {len(X)} samples, {len(self.feature_names)} features, "
            f"{n_groups} groups (avg size: {avg_group_size:.1f})"
        )
        
        self.model = self._create_model()
        
        # Prepare fit kwargs
        fit_kwargs = {
            "X": X.values,
            "y": y.values,
            "group": groups,
        }
        
        # Add validation set if provided
        if X_val is not None and y_val is not None:
            if groups_val is None and isinstance(X_val.index, pd.MultiIndex):
                groups_val = X_val.groupby(level=0).size().values
            
            fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]
            fit_kwargs["eval_group"] = [groups_val]
            
            if self.config.early_stopping_rounds:
                fit_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds
        
        # Fit the model
        self.model.fit(**fit_kwargs)
        
        # Log training completion
        if hasattr(self.model, "best_iteration"):
            logger.info(f"Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ranking predictions.
        
        Args:
            X: Feature matrix.
        
        Returns:
            Predicted scores (higher = better ranking).
        
        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.model.predict(X.values)
    
    def predict_with_index(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions preserving the input index.
        
        Args:
            X: Feature matrix with index.
        
        Returns:
            Series with predictions and original index.
        """
        predictions = self.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")
    
    def rank_cross_sectionally(self, X: pd.DataFrame) -> pd.Series:
        """Predict and convert to cross-sectional ranks.
        
        For each date, ranks stocks by predicted score (higher prediction = lower rank).
        
        Args:
            X: Feature matrix with MultiIndex (date, ticker).
        
        Returns:
            Series with cross-sectional ranks (1 = best predicted).
        """
        predictions = self.predict_with_index(X)
        
        # Rank within each date (higher score = lower rank number = better)
        ranks = predictions.groupby(level=0).rank(ascending=False, method="first")
        return ranks.rename("rank")
    
    def get_feature_importance(
        self,
        importance_type: str = "gain",
        as_dataframe: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Get feature importance scores.
        
        Args:
            importance_type: Type of importance:
                - "gain": Average gain of splits using the feature
                - "weight": Number of times a feature appears in trees
                - "cover": Average coverage of splits using the feature
            as_dataframe: If True, return DataFrame instead of Series.
        
        Returns:
            Series or DataFrame with feature importances.
        
        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get raw importance dict from XGBoost
        importance_dict = self.model.get_booster().get_score(
            importance_type=importance_type
        )
        
        # Map to feature names (XGBoost uses f0, f1, ... internally)
        importance = np.zeros(len(self.feature_names))
        for key, value in importance_dict.items():
            # Extract feature index from "f0", "f1", etc.
            if key.startswith("f"):
                try:
                    idx = int(key[1:])
                    if 0 <= idx < len(importance):
                        importance[idx] = value
                except ValueError:
                    continue
        
        series = pd.Series(
            importance, 
            index=self.feature_names, 
            name="importance"
        ).sort_values(ascending=False)
        
        if as_dataframe:
            return pd.DataFrame({
                "feature_name": series.index,
                "importance": series.values,
            }).reset_index(drop=True)
        
        return series
    
    def get_feature_importance_by_family(self) -> pd.DataFrame:
        """Get feature importance aggregated by factor family.
        
        Factor families are determined by feature name prefixes:
        - tech_: Technical features (momentum, volatility, etc.)
        - sector_: Sector indicators
        - value_: Value factors
        - quality_: Quality factors
        - growth_: Growth factors
        - prof_: Profitability factors
        - size_: Size factor
        
        Returns:
            DataFrame with columns ['family', 'importance', 'pct'].
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Define family prefixes
        family_prefixes = {
            "tech": ["tech_", "mom_", "vol_", "rsi_", "ma_", "ret_"],
            "sector": ["sector_"],
            "value": ["value_"],
            "quality": ["quality_"],
            "growth": ["growth_"],
            "profitability": ["prof_"],
            "size": ["size_"],
        }
        
        importance = self.get_feature_importance().to_dict()
        family_importance: dict[str, float] = {family: 0.0 for family in family_prefixes}
        family_importance["other"] = 0.0
        
        for feat_name, imp in importance.items():
            assigned = False
            for family, prefixes in family_prefixes.items():
                if any(feat_name.startswith(p) for p in prefixes):
                    family_importance[family] += imp
                    assigned = True
                    break
            if not assigned:
                family_importance["other"] += imp
        
        # Remove families with zero importance
        family_importance = {k: v for k, v in family_importance.items() if v > 0}
        
        # Normalize to sum to 1.0
        total = sum(family_importance.values())
        if total > 0:
            normalized = {k: v / total for k, v in family_importance.items()}
        else:
            normalized = family_importance
        
        df = pd.DataFrame([
            {"family": k, "importance": family_importance[k], "pct": normalized.get(k, 0)}
            for k in family_importance
        ])
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        return df
    
    def cross_validate_temporal(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, list[float]]:
        """Perform time-series cross-validation for ranking.
        
        Uses expanding window: train on all data up to split, test on next fold.
        Evaluates using rank correlation (Spearman) which is more appropriate
        for ranking models.
        
        Args:
            X: Feature matrix with MultiIndex (date, ticker).
            y: Target values.
            n_splits: Number of CV splits.
        
        Returns:
            Dictionary with 'train_ic' and 'test_ic' (information coefficients).
        """
        from scipy.stats import spearmanr
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        train_ics = []
        test_ics = []
        
        # Get unique dates for splitting
        dates = X.index.get_level_values(0).unique()
        
        for train_dates_idx, test_dates_idx in tscv.split(dates):
            train_dates = dates[train_dates_idx]
            test_dates = dates[test_dates_idx]
            
            # Filter by dates
            train_mask = X.index.get_level_values(0).isin(train_dates)
            test_mask = X.index.get_level_values(0).isin(test_dates)
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            # Compute groups
            groups_train = X_train.groupby(level=0).size().values
            
            # Create and fit model
            model = self._create_model()
            model.fit(X_train.values, y_train.values, group=groups_train)
            
            # Predict
            train_pred = model.predict(X_train.values)
            test_pred = model.predict(X_test.values)
            
            # Compute rank correlation (IC)
            train_ic, _ = spearmanr(y_train, train_pred)
            test_ic, _ = spearmanr(y_test, test_pred)
            
            train_ics.append(train_ic if not np.isnan(train_ic) else 0.0)
            test_ics.append(test_ic if not np.isnan(test_ic) else 0.0)
        
        return {
            "train_ic": train_ics,
            "test_ic": test_ics,
        }
    
    def compute_ic_by_date(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """Compute information coefficient (rank correlation) for each date.
        
        Args:
            X: Feature matrix with MultiIndex (date, ticker).
            y: Actual target values.
        
        Returns:
            Series with IC values indexed by date.
        """
        from scipy.stats import spearmanr
        
        predictions = self.predict_with_index(X)
        
        ics = {}
        for date in X.index.get_level_values(0).unique():
            y_date = y.loc[date]
            pred_date = predictions.loc[date]
            
            if len(y_date) > 2:
                ic, _ = spearmanr(y_date, pred_date)
                ics[date] = ic if not np.isnan(ic) else 0.0
            else:
                ics[date] = np.nan
        
        return pd.Series(ics, name="IC")
    
    def compute_ic_ir(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """Compute IC and IC-IR (Information Ratio).
        
        IC-IR = mean(IC) / std(IC), a measure of IC consistency.
        Higher IC-IR indicates more reliable predictive signal.
        
        Args:
            X: Feature matrix.
            y: Actual target values.
        
        Returns:
            Dictionary with 'mean_ic', 'std_ic', 'ic_ir', 'hit_rate'.
        """
        ic_series = self.compute_ic_by_date(X, y).dropna()
        
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
        hit_rate = (ic_series > 0).mean()  # Fraction of dates with positive IC
        
        return {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ic_ir": ic_ir,
            "hit_rate": hit_rate,
            "n_dates": len(ic_series),
        }
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model (should end in .json or .ubj).
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> "XGBRankingModel":
        """Load a model from disk.
        
        Args:
            path: Path to the saved model.
        
        Returns:
            Self for method chaining.
        """
        self.model = xgb.XGBRanker()
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")
        
        return self


# =============================================================================
# XGBOOST REGRESSION MODEL
# =============================================================================

@dataclass
class XGBRegressionConfig:
    """Configuration for XGBoost regression model.
    
    Attributes:
        objective: Regression objective to use.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage (eta).
        reg_lambda: L2 regularization term (lambda).
        reg_alpha: L1 regularization term (alpha).
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns for each tree.
        min_child_weight: Minimum sum of instance weight in a child.
        gamma: Minimum loss reduction for further partition.
        early_stopping_rounds: Stop if no improvement for this many rounds.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel threads (-1 for all cores).
        verbose: Verbosity level for training.
    """
    
    objective: str = "reg:squarederror"
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    reg_lambda: float = 2.0
    reg_alpha: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    gamma: float = 0.0
    early_stopping_rounds: Optional[int] = 50
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0


class XGBRegressionModel:
    """XGBoost regression model for continuous target prediction.
    
    This model predicts continuous values (e.g., vol-normalized forward returns)
    and can be used for ranking by sorting predictions.
    
    The key advantage over ranking models:
    - Can learn absolute magnitudes, not just relative ordering
    - More interpretable predictions
    - Easier to diagnose and debug
    
    Example:
        >>> config = XGBRegressionConfig()
        >>> model = XGBRegressionModel(config)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, config: Optional[XGBRegressionConfig] = None):
        """Initialize the XGBoost regression model.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )
        
        self.config = config or XGBRegressionConfig()
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: Optional[list[str]] = None
        self._eval_results: dict = {}
    
    def _create_model(self) -> xgb.XGBRegressor:
        """Create a fresh XGBRegressor instance with configured parameters."""
        return xgb.XGBRegressor(
            objective=self.config.objective,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            reg_lambda=self.config.reg_lambda,
            reg_alpha=self.config.reg_alpha,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=self.config.verbose,
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[np.ndarray] = None,  # Ignored, kept for API compatibility
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        groups_val: Optional[np.ndarray] = None,  # Ignored
    ) -> "XGBRegressionModel":
        """Fit the regression model.
        
        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target values.
            groups: Ignored (kept for API compatibility with ranking model).
            X_val: Optional validation feature matrix.
            y_val: Optional validation targets.
            groups_val: Ignored.
        
        Returns:
            Self for method chaining.
        """
        self.feature_names = list(X.columns)
        
        # Log training info
        n_groups = None
        if isinstance(X.index, pd.MultiIndex):
            n_groups = X.groupby(level=0).size().shape[0]
        
        logger.info(
            f"Training XGBRegressor with {len(X)} samples, {len(self.feature_names)} features"
            + (f", {n_groups} dates" if n_groups else "")
        )
        
        self.model = self._create_model()
        
        # Prepare fit kwargs
        fit_kwargs = {
            "X": X.values,
            "y": y.values,
        }
        
        # Add validation set if provided
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]
            
            if self.config.early_stopping_rounds:
                fit_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds
        
        # Fit the model
        self.model.fit(**fit_kwargs)
        
        # Log training completion
        if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
            logger.info(f"Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate regression predictions.
        
        Args:
            X: Feature matrix.
        
        Returns:
            Predicted values.
        
        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.model.predict(X.values)
    
    def predict_with_index(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions preserving the input index.
        
        Args:
            X: Feature matrix with index.
        
        Returns:
            Series with predictions and original index.
        """
        predictions = self.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")
    
    def rank_cross_sectionally(self, X: pd.DataFrame) -> pd.Series:
        """Predict and convert to cross-sectional ranks.
        
        For each date, ranks stocks by predicted value (higher prediction = lower rank).
        
        Args:
            X: Feature matrix with MultiIndex (date, ticker).
        
        Returns:
            Series with cross-sectional ranks (1 = best predicted).
        """
        predictions = self.predict_with_index(X)
        
        # Rank within each date (higher score = lower rank number = better)
        ranks = predictions.groupby(level=0).rank(ascending=False, method="first")
        return ranks.rename("rank")
    
    def get_feature_importance(
        self,
        importance_type: str = "gain",
        as_dataframe: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Get feature importance scores.
        
        Args:
            importance_type: Type of importance ("gain", "weight", "cover").
            as_dataframe: If True, return a DataFrame with additional info.
        
        Returns:
            Feature importance scores.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Map back to feature names
        result = {}
        for i, name in enumerate(self.feature_names or []):
            key = f"f{i}"
            result[name] = importance.get(key, 0.0)
        
        series = pd.Series(result, name=importance_type).sort_values(ascending=False)
        
        if as_dataframe:
            return series.reset_index()
        return series
    
    def compute_ic_by_date(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """Compute Information Coefficient (IC) by date.
        
        IC = Spearman correlation between predictions and targets.
        
        Args:
            X: Feature matrix with MultiIndex (date, ticker).
            y: Actual target values.
        
        Returns:
            Series of IC values indexed by date.
        """
        predictions = self.predict_with_index(X)
        
        dates = X.index.get_level_values(0).unique()
        ics = {}
        
        for date in dates:
            pred = predictions.loc[date]
            actual = y.loc[date]
            
            if len(pred) >= 3:
                ic = stats.spearmanr(pred, actual)[0]
                ics[date] = ic if not np.isnan(ic) else 0.0
            else:
                ics[date] = np.nan
        
        return pd.Series(ics, name="IC")
    
    def compute_ic_ir(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """Compute IC and IC-IR (Information Ratio).
        
        IC-IR = mean(IC) / std(IC), a measure of IC consistency.
        
        Args:
            X: Feature matrix.
            y: Actual target values.
        
        Returns:
            Dictionary with 'mean_ic', 'std_ic', 'ic_ir', 'hit_rate'.
        """
        ic_series = self.compute_ic_by_date(X, y).dropna()
        
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
        hit_rate = (ic_series > 0).mean()  # Fraction of dates with positive IC
        
        return {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ic_ir": ic_ir,
            "hit_rate": hit_rate,
            "n_dates": len(ic_series),
        }


def create_model(
    model_type: Literal["regression", "rank_pairwise", "rank_ndcg"] = "regression",
    **kwargs,
):
    """Factory function to create appropriate model based on type.
    
    Args:
        model_type: Type of model to create:
            - "regression": XGBoost regression model (default)
            - "rank_pairwise": XGBoost pairwise ranking model
            - "rank_ndcg": XGBoost NDCG ranking model
        **kwargs: Additional arguments passed to model config.
    
    Returns:
        Configured model instance.
    """
    if model_type == "regression":
        config = XGBRegressionConfig(**kwargs) if kwargs else None
        return XGBRegressionModel(config)
    
    elif model_type == "rank_pairwise":
        config = XGBRankingConfig(objective="rank:pairwise", **kwargs)
        return XGBRankingModel(config)
    
    elif model_type == "rank_ndcg":
        config = XGBRankingConfig(objective="rank:ndcg", **kwargs)
        return XGBRankingModel(config)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
