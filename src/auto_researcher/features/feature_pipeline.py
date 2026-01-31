"""
Feature pipeline orchestration.

This module combines all feature sources (technical, fundamental, sentiment)
into a unified feature matrix for model training and prediction.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from auto_researcher.features.technical import (
    compute_returns,
    compute_all_technical_features,
)
from auto_researcher.features.fundamentals import (
    compute_all_fundamental_features,
    load_raw_fundamentals,
    build_fundamental_factors,
    FundamentalFactorConfig,
)
from auto_researcher.features.sentiment import compute_all_sentiment_features
from auto_researcher.config import FeatureConfig, get_default_config

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class FeaturePipelineConfig:
    """
    Configuration for the full feature pipeline.

    This extends the basic FeatureConfig with options for controlling
    which feature sources to include and how to configure them.

    Attributes:
        use_technical: Whether to include technical features (momentum, vol, etc.).
        use_fundamentals: Whether to include fundamental factors.
        use_sentiment: Whether to include sentiment features.
        fundamentals_csv_path: Path to CSV with fundamental data (optional).
        fundamental_factor_config: Configuration for fundamental factors.
        momentum_windows: Windows for momentum feature computation.
        volatility_windows: Windows for volatility feature computation.
        use_core_features_only: When True, filter to only core_features.
        core_features: Tuple of feature names to use when use_core_features_only=True.
        tech_only_tickers: Tickers to mask fundamentals for (hybrid mode).
            For these tickers, fundamental features will be set to NaN.
    """

    use_technical: bool = True
    use_fundamentals: bool = False
    use_sentiment: bool = False
    fundamentals_csv_path: str | None = None
    fundamental_factor_config: FundamentalFactorConfig = field(
        default_factory=FundamentalFactorConfig
    )
    momentum_windows: tuple[int, ...] = (21, 63, 126, 252)
    volatility_windows: tuple[int, ...] = (21, 63)
    use_core_features_only: bool = False
    core_features: tuple[str, ...] | None = None
    tech_only_tickers: tuple[str, ...] = ()


def _convert_feature_config(config: FeatureConfig) -> FeaturePipelineConfig:
    """
    Convert legacy FeatureConfig to FeaturePipelineConfig.

    Args:
        config: Legacy FeatureConfig.

    Returns:
        Equivalent FeaturePipelineConfig.
    """
    # Build FundamentalFactorConfig from dictionary if provided
    factor_config = FundamentalFactorConfig()
    factor_families = getattr(config, 'fundamental_factor_families', None)
    
    # Get source settings from config
    source = getattr(config, 'fundamentals_source', 'csv')
    finagg_mode = getattr(config, 'finagg_mode', 'quarterly_refined')
    csv_path = getattr(config, 'fundamentals_csv_path', None)
    
    # For finagg data, disable z-scoring since filing dates are sparse
    # and each date may only have 1-2 companies, making cross-sectional
    # normalization ineffective. The raw ratios (ROE, ROA, etc.) are
    # already comparable across companies.
    zscore_enabled = source != 'finagg'
    
    if factor_families is not None:
        factor_config = FundamentalFactorConfig(
            use_value=factor_families.get('value', True),
            use_quality=factor_families.get('quality', True),
            use_growth=factor_families.get('growth', True),
            use_profitability=factor_families.get('profitability', True),
            use_size=factor_families.get('size', True),
            source=source,
            csv_path=csv_path,
            finagg_mode=finagg_mode,
            zscore_by_cross_section=zscore_enabled,
        )
    else:
        factor_config = FundamentalFactorConfig(
            source=source,
            csv_path=csv_path,
            finagg_mode=finagg_mode,
            zscore_by_cross_section=zscore_enabled,
        )
    
    # Check for include_technical flag (defaults to True for backward compat)
    use_technical = getattr(config, 'include_technical', True)
    
    # Get core features settings
    use_core_features_only = getattr(config, 'use_core_features_only', False)
    core_features = getattr(config, 'core_features', None)
    
    # Get tech-only tickers for hybrid mode
    tech_only_tickers = getattr(config, 'tech_only_tickers', ())
    
    return FeaturePipelineConfig(
        use_technical=use_technical,
        use_fundamentals=config.include_fundamentals,
        use_sentiment=config.include_sentiment,
        fundamentals_csv_path=getattr(config, 'fundamentals_csv_path', None),
        fundamental_factor_config=factor_config,
        momentum_windows=config.momentum_windows,
        volatility_windows=config.volatility_windows,
        use_core_features_only=use_core_features_only,
        core_features=core_features,
        tech_only_tickers=tech_only_tickers,
    )


# =============================================================================
# Feature Matrix Building
# =============================================================================


def build_feature_matrix(
    prices: pd.DataFrame,
    config: FeatureConfig | FeaturePipelineConfig | None = None,
) -> pd.DataFrame:
    """
    Build a complete feature matrix from price data.

    Combines technical, fundamental, and sentiment features based on config.
    Returns a DataFrame with rows indexed by date and columns organized
    as (ticker, feature_name) MultiIndex.

    Technical feature columns are prefixed with 'tech_' when fundamentals are enabled.
    Fundamental feature columns use their family prefix (value_, quality_, etc.).

    Args:
        prices: Adjusted close prices with tickers as columns, DatetimeIndex as rows.
        config: Feature configuration. Accepts either:
            - FeatureConfig (legacy): Converted internally to FeaturePipelineConfig
            - FeaturePipelineConfig: Full control over all feature sources
            Uses defaults if not provided.

    Returns:
        Feature matrix with MultiIndex columns (ticker, feature).

    Examples:
        >>> import numpy as np
        >>> dates = pd.date_range('2023-01-01', periods=300)
        >>> prices = pd.DataFrame(
        ...     np.random.lognormal(size=(300, 3)).cumprod(axis=0) * 100,
        ...     index=dates,
        ...     columns=['AAPL', 'MSFT', 'GOOG']
        ... )
        >>> features = build_feature_matrix(prices)
        >>> features.shape[0] == 300
        True
    """
    # Handle config conversion
    if config is None:
        pipeline_config = FeaturePipelineConfig()
    elif isinstance(config, FeaturePipelineConfig):
        pipeline_config = config
    else:
        # Convert legacy FeatureConfig
        pipeline_config = _convert_feature_config(config)

    logger.info("Building feature matrix...")

    all_features: list[pd.DataFrame] = []

    # -------------------------------------------------------------------------
    # Technical features
    # -------------------------------------------------------------------------
    if pipeline_config.use_technical:
        tech_features = compute_all_technical_features(
            prices,
            momentum_windows=pipeline_config.momentum_windows,
            volatility_windows=pipeline_config.volatility_windows,
        )

        # Prefix technical features with 'tech_' when fundamentals are enabled
        if pipeline_config.use_fundamentals:
            tech_features = _prefix_features(tech_features, "tech_")

        all_features.append(tech_features)
        logger.info(f"Computed {len(tech_features.columns)} technical features")

    # -------------------------------------------------------------------------
    # Fundamental features
    # -------------------------------------------------------------------------
    if pipeline_config.use_fundamentals:
        # Handle MultiIndex columns (e.g., from yfinance with ('Adj Close', 'AAPL'))
        if isinstance(prices.columns, pd.MultiIndex):
            tickers = prices.columns.get_level_values(-1).unique().tolist()
        else:
            tickers = prices.columns.tolist()
        start = prices.index[0].strftime("%Y-%m-%d")
        end = prices.index[-1].strftime("%Y-%m-%d")

        # Load raw fundamentals (pass config for source selection)
        raw_fund = load_raw_fundamentals(
            tickers=tickers,
            start=start,
            end=end,
            fundamentals_csv_path=pipeline_config.fundamentals_csv_path,
            config=pipeline_config.fundamental_factor_config,
        )

        # Compute factors
        fund_factors = build_fundamental_factors(
            raw_fund,
            config=pipeline_config.fundamental_factor_config,
        )

        if not fund_factors.empty:
            # Align fundamentals to price dates (forward-fill monthly to daily)
            fund_features = _align_fundamentals_to_prices(fund_factors, prices)
            all_features.append(fund_features)
            logger.info(
                f"Computed {len(fund_features.columns)} fundamental features"
            )
        else:
            logger.warning("No fundamental factors computed")

    # -------------------------------------------------------------------------
    # Sentiment features (stub)
    # -------------------------------------------------------------------------
    if pipeline_config.use_sentiment:
        tickers = prices.columns.tolist()
        start = prices.index[0].strftime("%Y-%m-%d")
        end = prices.index[-1].strftime("%Y-%m-%d")
        sent_features = compute_all_sentiment_features(tickers, start, end)
        if not sent_features.empty:
            all_features.append(sent_features)
            logger.info(f"Computed {len(sent_features.columns)} sentiment features")

    # -------------------------------------------------------------------------
    # Combine all features
    # -------------------------------------------------------------------------
    if not all_features:
        raise ValueError("No features computed - check configuration")

    if len(all_features) == 1:
        result = all_features[0]
    else:
        # Merge features on date index
        result = all_features[0]
        for other in all_features[1:]:
            result = _merge_feature_matrices(result, other)

    # -------------------------------------------------------------------------
    # Filter to core features if requested
    # -------------------------------------------------------------------------
    if pipeline_config.use_core_features_only and pipeline_config.core_features:
        result = _filter_to_core_features(result, pipeline_config.core_features)
        logger.info(
            f"Filtered to {len(pipeline_config.core_features)} core features"
        )

    # -------------------------------------------------------------------------
    # Mask fundamentals for tech-only tickers (hybrid mode)
    # -------------------------------------------------------------------------
    if pipeline_config.tech_only_tickers and pipeline_config.use_fundamentals:
        from auto_researcher.features.coverage import (
            identify_fundamental_columns,
            mask_fundamentals_for_tech_only_tickers,
        )
        fund_cols = identify_fundamental_columns(result)
        if fund_cols:
            result = mask_fundamentals_for_tech_only_tickers(
                result, 
                list(pipeline_config.tech_only_tickers), 
                fund_cols
            )
            logger.info(
                f"Masked {len(fund_cols)} fundamental features for "
                f"{len(pipeline_config.tech_only_tickers)} tech-only tickers"
            )

    return result


def _filter_to_core_features(
    features: pd.DataFrame,
    core_features: tuple[str, ...],
) -> pd.DataFrame:
    """
    Filter feature matrix to only include core features.

    Args:
        features: DataFrame with MultiIndex columns (ticker, feature).
        core_features: Tuple of feature names to keep.

    Returns:
        DataFrame filtered to only core features.

    Raises:
        ValueError: If no core features are found in the feature matrix.
    """
    # Get unique feature names from the MultiIndex
    if isinstance(features.columns, pd.MultiIndex):
        all_feature_names = features.columns.get_level_values("feature").unique()
        core_set = set(core_features)
        found_features = [f for f in all_feature_names if f in core_set]

        if not found_features:
            raise ValueError(
                f"None of the core features {core_features} found in feature matrix. "
                f"Available features: {list(all_feature_names)[:10]}..."
            )

        # Filter columns: keep (ticker, feature) where feature is in core_set
        mask = features.columns.get_level_values("feature").isin(core_set)
        return features.loc[:, mask]
    else:
        # Flat columns (non-MultiIndex case)
        core_set = set(core_features)
        found_features = [f for f in features.columns if f in core_set]

        if not found_features:
            raise ValueError(
                f"None of the core features {core_features} found in feature matrix. "
                f"Available features: {list(features.columns)[:10]}..."
            )

        return features[found_features]


def _prefix_features(
    features: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Add a prefix to all feature names in a MultiIndex column DataFrame.

    Args:
        features: DataFrame with MultiIndex columns (ticker, feature).
        prefix: Prefix to add to feature names.

    Returns:
        DataFrame with prefixed feature names.
    """
    new_columns = []
    for ticker, feature in features.columns:
        new_columns.append((ticker, f"{prefix}{feature}"))

    features.columns = pd.MultiIndex.from_tuples(new_columns, names=["ticker", "feature"])
    return features


def _align_fundamentals_to_prices(
    fund_factors: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align fundamental factors to price dates.

    Fundamentals are typically updated monthly/quarterly, so we forward-fill
    to daily frequency to align with price data.

    Args:
        fund_factors: DataFrame with MultiIndex (date, ticker) and factor columns.
        prices: Price DataFrame with DatetimeIndex and ticker columns.

    Returns:
        DataFrame with MultiIndex columns (ticker, feature), aligned to prices.index.
    """
    tickers = prices.columns.tolist()
    price_dates = prices.index

    # Build a (date, ticker) -> factor mapping
    # For each ticker, create a Series for each factor and reindex to price dates
    aligned_data = {}

    for ticker in tickers:
        if ticker not in fund_factors.index.get_level_values("ticker"):
            # No fundamental data for this ticker, skip
            continue

        ticker_data = fund_factors.xs(ticker, level="ticker")

        for factor in fund_factors.columns:
            factor_series = ticker_data[factor]

            # Reindex to price dates with forward-fill
            aligned_series = factor_series.reindex(price_dates, method="ffill")
            aligned_data[(ticker, factor)] = aligned_series

    if not aligned_data:
        return pd.DataFrame(index=price_dates)

    result = pd.DataFrame(aligned_data, index=price_dates)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["ticker", "feature"])

    return result


def _merge_feature_matrices(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge two feature matrices with MultiIndex columns.

    Both matrices should have the same DatetimeIndex.

    Args:
        left: First feature matrix.
        right: Second feature matrix.

    Returns:
        Combined feature matrix.
    """
    # Ensure both have the same index
    common_dates = left.index.intersection(right.index)
    left = left.loc[common_dates]
    right = right.loc[common_dates]

    # Concatenate columns
    result = pd.concat([left, right], axis=1)

    return result


def build_training_data(
    prices: pd.DataFrame,
    horizon_days: int,
    benchmark: str | None = "SPY",
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix and forward return labels for training.

    Creates aligned features (at time t) and labels (return from t+1 to t+horizon).
    No lookahead bias: features use only data up to time t.

    IMPORTANT: Labels (y) are EXCESS RETURNS relative to the benchmark.
    -----------------------------------------------------------------------
    - If benchmark="SPY", y = stock_forward_return - SPY_forward_return
    - This means y represents alpha vs benchmark, NOT raw returns
    - The benchmark ticker (e.g., SPY) is EXCLUDED from the label columns
      because its excess return vs itself would always be 0
    - For predicting absolute returns, set benchmark=None

    Args:
        prices: Adjusted close prices with tickers as columns.
        horizon_days: Forward return horizon in trading days.
        benchmark: Ticker to use for excess return calculation. If None, use raw returns.
        config: Feature configuration.

    Returns:
        Tuple of (features, labels) where:
        - features: DataFrame with MultiIndex columns (ticker, feature)
        - labels: DataFrame with tickers as columns, containing:
            - EXCESS forward returns (vs benchmark) if benchmark is specified
            - RAW forward returns if benchmark is None
          Note: Benchmark ticker is excluded from labels when benchmark is specified.

    Raises:
        ValueError: If benchmark ticker not in prices.
    """
    if benchmark is not None and benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} not found in prices")

    # Build features
    features = build_feature_matrix(prices, config)

    # Compute forward returns (shift by -horizon to align with current features)
    forward_prices = prices.shift(-horizon_days)
    forward_returns = (forward_prices / prices) - 1

    # Compute excess returns vs benchmark
    if benchmark is not None:
        benchmark_return = forward_returns[benchmark]
        labels = forward_returns.drop(columns=[benchmark]).sub(benchmark_return, axis=0)
    else:
        labels = forward_returns

    # Remove rows where we don't have complete forward returns
    # NOTE: We only filter on labels here, not features.
    # Features are filtered per-ticker after stacking in prepare_training_data.
    # This allows tickers with partial fundamental coverage to still be used.
    valid_mask = labels.notna().all(axis=1)
    
    features = features[valid_mask]
    labels = labels[valid_mask]

    logger.info(f"Built training data: {len(features)} samples, {len(features.columns)} features")

    return features, labels


def reshape_features_for_model(features: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape MultiIndex feature matrix into format suitable for ML model.

    Converts from (date, (ticker, feature)) to (date_ticker, feature).
    This creates a panel/stacked format where each row is a date-ticker observation.

    Args:
        features: Feature matrix with MultiIndex columns (ticker, feature).

    Returns:
        Stacked DataFrame with features as columns, MultiIndex (date, ticker) as rows.
    """
    # Stack ticker level to rows
    stacked = features.stack(level="ticker", future_stack=True)
    stacked.index.names = ["date", "ticker"]
    
    return stacked


def reshape_labels_for_model(labels: pd.DataFrame) -> pd.Series:
    """
    Reshape labels DataFrame into format suitable for ML model.

    Args:
        labels: Labels DataFrame with tickers as columns.

    Returns:
        Series with MultiIndex (date, ticker).
    """
    stacked = labels.stack(future_stack=True)
    stacked.index.names = ["date", "ticker"]
    stacked.name = "forward_return"
    
    return stacked


def align_features_and_labels(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Ensure features and labels have matching indices.

    Args:
        features: Stacked feature DataFrame.
        labels: Stacked label Series.

    Returns:
        Aligned (features, labels) tuple.
    """
    common_idx = features.index.intersection(labels.index)
    return features.loc[common_idx], labels.loc[common_idx]


def prepare_training_data(
    prices: pd.DataFrame,
    horizon_days: int,
    benchmark: str | None = "SPY",
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline to prepare data for model training.

    Combines feature building, label creation, reshaping, and alignment.

    Args:
        prices: Adjusted close prices.
        horizon_days: Forward return horizon.
        benchmark: Benchmark ticker for excess returns.
        config: Feature configuration.

    Returns:
        Tuple of (X, y) ready for model training.
    """
    features, labels = build_training_data(prices, horizon_days, benchmark, config)
    
    X = reshape_features_for_model(features)
    y = reshape_labels_for_model(labels)
    
    X, y = align_features_and_labels(X, y)
    
    # Identify technical vs fundamental columns
    # Technical features start with 'tech_', fundamentals with value_, quality_, growth_, prof_, size_
    tech_cols = [c for c in X.columns if c.startswith('tech_')]
    fund_cols = [c for c in X.columns if not c.startswith('tech_')]
    
    # For fundamental features, fill NaN with 0 (neutral value)
    # This allows training even when some tickers lack fundamentals
    if fund_cols:
        X[fund_cols] = X[fund_cols].fillna(0.0)
    
    # Drop rows where technical features or labels are NaN
    valid_mask = X[tech_cols].notna().all(axis=1) & y.notna() if tech_cols else y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y


def get_features_for_date(
    features: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Extract features for a specific date.

    Args:
        features: Feature matrix with MultiIndex columns (ticker, feature).
        date: Date to extract features for.

    Returns:
        DataFrame with tickers as rows and features as columns.
    """
    if date not in features.index:
        raise KeyError(f"Date {date} not found in features")
    
    row = features.loc[date]
    
    # Reshape to (ticker, feature) format
    result = row.unstack(level="feature")
    
    return result
