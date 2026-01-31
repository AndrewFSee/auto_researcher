"""
Configuration module for Auto-Researcher.

This module provides centralized configuration using dataclasses
for type safety and easy extension.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional


@dataclass(frozen=True)
class EnhancedModelConfig:
    """
    Configuration for enhanced model.
    
    Attributes:
        enabled: Whether to use enhanced model features.
        model_type: Type of model to use:
            - "regression": XGBoost regression (default for enhanced mode)
            - "rank_pairwise": XGBoost pairwise ranking
            - "rank_ndcg": XGBoost NDCG ranking
        target_mode: Target engineering mode:
            - "vol_norm": Volatility-normalized forward returns (default)
            - "raw": Raw forward returns
            - "rank": Cross-sectional rank transformation
            - "ortho": Orthogonalized returns (market/sector adjusted)
        use_short_reversal: Include 5d/10d short-term reversal features.
        use_residual_momentum: Include beta-adjusted residual momentum.
        use_idio_vol: Include idiosyncratic volatility features.
        use_mad: Include median absolute deviation features.
        use_sector_ohe: Include sector one-hot encoding.
        cross_sectional_normalize: Apply cross-sectional z-score normalization.
        normalize_type: Type of normalization ("robust" or "standard").
        normalize_by_sector: Normalize within sectors.
        invert_signal_if_negative: If True, invert predictions when IC is consistently negative.
        rolling_window: If True, use rolling window for training instead of expanding window.
            Rolling windows adapt to regime changes faster but have less training data.
        rolling_window_days: Size of rolling window in trading days (default: 504 = ~2 years).
            Only used when rolling_window=True.
        use_shap_selection: If True, use SHAP values to select top features for reduced overfitting.
        shap_top_k_features: Number of top features to keep when using SHAP selection.
    """
    
    enabled: bool = False
    model_type: Literal["regression", "rank_pairwise", "rank_ndcg"] = "regression"
    target_mode: Literal["vol_norm", "raw", "rank", "ortho"] = "vol_norm"
    use_short_reversal: bool = True
    use_residual_momentum: bool = True
    use_idio_vol: bool = True
    use_mad: bool = True
    use_sector_ohe: bool = True
    cross_sectional_normalize: bool = True
    normalize_type: Literal["robust", "standard"] = "robust"
    normalize_by_sector: bool = False
    invert_signal_if_negative: bool = False
    rolling_window: bool = False
    rolling_window_days: int = 504  # ~2 years of trading days
    robust_objective: bool = False  # Use reg:pseudohubererror instead of reg:squarederror
    use_shap_selection: bool = False  # Use SHAP to select top features
    shap_top_k_features: int = 15  # Number of top features to keep when using SHAP
    
    # XGBoost hyperparameters (None means use defaults)
    xgb_max_depth: int | None = None
    xgb_learning_rate: float | None = None
    xgb_n_estimators: int | None = None
    xgb_subsample: float | None = None
    xgb_colsample_bytree: float | None = None
    xgb_reg_lambda: float | None = None
    xgb_reg_alpha: float | None = None


@dataclass(frozen=True)
class EnhancedPortfolioConfig:
    """
    Configuration for enhanced portfolio construction.
    
    Attributes:
        top_k: Number of stocks to include (default 25 for enhanced mode).
        weighting_scheme: How to weight stocks:
            - "equal": Equal weight all positions
            - "rank": Exponential decay based on rank
            - "score": Weight proportional to prediction score
        rank_tau: Temperature for rank weighting (lower = more concentrated).
        neutralization: Risk neutralization method:
            - "none": No neutralization
            - "vol": Scale by inverse volatility
            - "beta": Target zero portfolio beta
            - "vol_beta": Apply both
        max_position_weight: Maximum weight per position.
        sector_neutral_ranking: If True, rank within sectors first to ensure
            sector diversity. Prevents model from picking all stocks from one sector.
    """
    
    top_k: int = 25
    weighting_scheme: Literal["equal", "rank", "score"] = "rank"
    rank_tau: float = 5.0
    neutralization: Literal["none", "vol", "beta", "vol_beta"] = "none"
    max_position_weight: float = 0.10
    sector_neutral_ranking: bool = False  # NEW


@dataclass(frozen=True)
class ResearchConfig:
    """
    Configuration for the research pipeline.

    Attributes:
        universe: List of tickers to analyze.
        start_date: Start date for historical data (YYYY-MM-DD).
        end_date: End date for historical data (YYYY-MM-DD).
        rebalance_frequency: How often to rebalance ('M' for monthly, 'W' for weekly).
        horizon_days: Forward return horizon in trading days (63 ≈ 3 months).
        top_k: Number of stocks to select for the portfolio.
        benchmark: Ticker to use as benchmark (e.g., 'SPY').
        lookback_days: Minimum lookback days for feature computation.
    """

    universe: tuple[str, ...] = ("AAPL", "MSFT", "SPY")
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
    rebalance_frequency: Literal["M", "W", "D"] = "M"
    horizon_days: int = 63  # ~3 months of trading days
    top_k: int = 10
    benchmark: str = "SPY"
    lookback_days: int = 252  # ~1 year for training


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for feature engineering.

    Attributes:
        momentum_windows: Windows (in days) for momentum features.
        volatility_windows: Windows (in days) for volatility features.
        include_technical: Whether to include technical features (momentum, vol, etc.).
            Set to False for fundamentals-only (Quality) model.
        include_fundamentals: Whether to include fundamental features.
        include_sentiment: Whether to include sentiment features.
        fundamentals_csv_path: Path to CSV file with fundamental data (optional).
        fundamental_factor_families: Dict controlling which factor families to use.
            Keys: 'value', 'quality', 'growth', 'profitability', 'size'
            Values: True/False
            If None, all families are enabled by default.
        fundamentals_source: Data source for fundamentals ("csv", "finagg", "yfinance").
            - "csv": Load from CSV file (default)
            - "finagg": Fetch from SEC EDGAR via finagg library
            - "yfinance": Fetch from yfinance (limited history)
        finagg_mode: Mode for finagg data fetching.
            - "quarterly_refined": Use local SQL database (faster)
            - "quarterly_api": Fetch directly from SEC API (slower)
            - "annual_refined": Annual data from local SQL database
        tech_only_tickers: Tickers to mask fundamentals for (hybrid mode).
            For these tickers, fundamental features will be set to NaN,
            allowing them to use only technical features.
    """

    momentum_windows: tuple[int, ...] = (21, 63, 126, 252)
    volatility_windows: tuple[int, ...] = (21, 63)
    include_technical: bool = True  # Set to False for fundamentals-only model
    include_fundamentals: bool = False  # TODO: Enable when fundamentals ready
    include_sentiment: bool = False  # TODO: Enable when sentiment ready
    fundamentals_csv_path: str | None = None
    fundamental_factor_families: dict | None = None
    fundamentals_source: str = "csv"  # "csv" | "finagg" | "yfinance"
    finagg_mode: str = "quarterly_refined"
    # Core features mode: train on a stable subset of features
    use_core_features_only: bool = False
    core_features: tuple[str, ...] | None = None  # List of feature names to use
    # Hybrid mode: mask fundamentals for these tickers
    tech_only_tickers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for the GBDT model.

    Attributes:
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate for boosting.
        max_depth: Maximum tree depth.
        num_leaves: Maximum number of leaves per tree.
        min_child_samples: Minimum samples in a leaf.
        random_state: Random seed for reproducibility.
        early_stopping_rounds: Rounds for early stopping (None to disable).
        reg_lambda: L2 regularization term (LightGBM lambda_l2).
        subsample: Fraction of samples used per tree (LightGBM bagging_fraction).
        colsample_bytree: Fraction of features used per tree (LightGBM feature_fraction).
    """

    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 20
    random_state: int = 42
    early_stopping_rounds: int | None = 10
    reg_lambda: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0


# Simplified model config for reduced overfitting with core factors
SIMPLE_MODEL_CONFIG = ModelConfig(
    n_estimators=75,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=16,
    min_child_samples=10,
    random_state=42,
    early_stopping_rounds=10,
)


def build_model_config_from_params(
    base_config: ModelConfig | None = None,
    **overrides,
) -> ModelConfig:
    """
    Build a ModelConfig by merging base config with parameter overrides.
    
    This is useful for hyperparameter sweeps where you want to vary
    specific parameters while keeping others at their base values.
    
    Args:
        base_config: Base configuration to start from. Defaults to SIMPLE_MODEL_CONFIG.
        **overrides: Parameter overrides (e.g., max_depth=2, reg_lambda=5.0).
    
    Returns:
        New ModelConfig with overrides applied.
    
    Example:
        >>> config = build_model_config_from_params(max_depth=2, reg_lambda=5.0)
        >>> config.max_depth
        2
        >>> config.reg_lambda
        5.0
    """
    if base_config is None:
        base_config = SIMPLE_MODEL_CONFIG
    
    # Get all fields from base config
    params = {
        "n_estimators": base_config.n_estimators,
        "learning_rate": base_config.learning_rate,
        "max_depth": base_config.max_depth,
        "num_leaves": base_config.num_leaves,
        "min_child_samples": base_config.min_child_samples,
        "random_state": base_config.random_state,
        "early_stopping_rounds": base_config.early_stopping_rounds,
        "reg_lambda": base_config.reg_lambda,
        "subsample": base_config.subsample,
        "colsample_bytree": base_config.colsample_bytree,
    }
    
    # Apply overrides
    for key, value in overrides.items():
        if key in params:
            params[key] = value
        else:
            raise ValueError(f"Unknown ModelConfig parameter: {key}")
    
    return ModelConfig(**params)


@dataclass
class PipelineConfig:
    """
    Master configuration combining all sub-configs.

    This is mutable to allow runtime overrides.
    """

    research: ResearchConfig = field(default_factory=ResearchConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    enhanced_model: EnhancedModelConfig = field(default_factory=EnhancedModelConfig)
    enhanced_portfolio: EnhancedPortfolioConfig = field(default_factory=EnhancedPortfolioConfig)


@dataclass(frozen=True)
class RegularizationSweepConfig:
    """
    Configuration for hyperparameter sweep over regularization settings.
    
    Used to find more conservative (less overfit) model configurations
    for the Core Factors [Simple] model.
    
    Attributes:
        enable: Whether the sweep is enabled.
        max_depth_grid: Grid of max_depth values to try.
        n_estimators_grid: Grid of n_estimators values to try.
        reg_lambda_grid: Grid of L2 regularization values.
        subsample_grid: Grid of row subsampling fractions.
        colsample_grid: Grid of column subsampling fractions.
    """
    
    enable: bool = False
    max_depth_grid: tuple[int, ...] = (2, 3)
    n_estimators_grid: tuple[int, ...] = (50, 75)
    reg_lambda_grid: tuple[float, ...] = (0.0, 5.0)
    subsample_grid: tuple[float, ...] = (0.6, 0.8)
    colsample_grid: tuple[float, ...] = (0.5, 0.8)
    
    def grid_size(self) -> int:
        """Return total number of configurations in the grid."""
        return (
            len(self.max_depth_grid)
            * len(self.n_estimators_grid)
            * len(self.reg_lambda_grid)
            * len(self.subsample_grid)
            * len(self.colsample_grid)
        )


# Default sweep config (conservative grid of 32 configs)
DEFAULT_REG_SWEEP_CONFIG = RegularizationSweepConfig(
    enable=True,
    max_depth_grid=(2, 3),
    n_estimators_grid=(50, 75),
    reg_lambda_grid=(0.0, 5.0),
    subsample_grid=(0.6, 0.8),
    colsample_grid=(0.5, 0.8),
)


# Default configurations for quick access
DEFAULT_UNIVERSE: list[str] = ["AAPL", "MSFT", "SPY"]
START_DATE: str = "2021-01-01"
END_DATE: str = "2024-12-31"
REBALANCE_FREQUENCY: str = "M"
HORIZON_DAYS: int = 63
TOP_K: int = 10
BENCHMARK: str = "SPY"


def get_default_config() -> PipelineConfig:
    """
    Get the default pipeline configuration.

    Returns:
        PipelineConfig with default settings.
    """
    return PipelineConfig()


def create_test_config() -> PipelineConfig:
    """
    Create a minimal configuration for testing.

    Uses a small universe and short date range for fast test execution.

    Returns:
        PipelineConfig optimized for testing.
    """
    return PipelineConfig(
        research=ResearchConfig(
            universe=("AAPL", "MSFT", "SPY"),
            start_date="2022-01-01",
            end_date="2024-01-01",
            horizon_days=21,
            top_k=2,
            lookback_days=63,
        ),
        features=FeatureConfig(
            momentum_windows=(21, 63),
            volatility_windows=(21,),
        ),
        model=ModelConfig(
            n_estimators=10,
            early_stopping_rounds=None,
        ),
    )
