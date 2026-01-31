"""Feature engineering modules."""

from .enhanced import (
    EnhancedFeatureConfig,
    compute_all_enhanced_features,
    compute_short_reversal_features,
    compute_residual_momentum_features,
    compute_idio_volatility_features,
    compute_mad_features,
    compute_sector_features,
    compute_beta_features,
    normalize_cross_sectionally,
    DEFAULT_SECTOR_MAPPING,
)

from .targets import (
    TargetConfig,
    build_targets,
    prepare_training_targets,
    compute_rank_targets,
    compute_ortho_targets,
)

__all__ = [
    # Enhanced features
    "EnhancedFeatureConfig",
    "compute_all_enhanced_features",
    "compute_short_reversal_features",
    "compute_residual_momentum_features",
    "compute_idio_volatility_features",
    "compute_mad_features",
    "compute_sector_features",
    "compute_beta_features",
    "normalize_cross_sectionally",
    "DEFAULT_SECTOR_MAPPING",
    # Targets
    "TargetConfig",
    "build_targets",
    "prepare_training_targets",
    "compute_rank_targets",
    "compute_ortho_targets",
]
