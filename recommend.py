"""
Stock Recommendation Generator.

This module generates stock recommendations using the XGBoost ranking model.
It computes enhanced technical features and generates ML-based scores for
universe screening.

Usage:
    from recommend import generate_recommendations, UNIVERSES
    
    recommendations, scores, prices = generate_recommendations(
        tickers=UNIVERSES["sp100"],
        top_k=25,
        explain=True,
    )
    
    for rec in recommendations:
        print(f"{rec.ticker}: {rec.score:.2f} ({rec.percentile:.0f}th percentile)")
"""

import gc
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from auto_researcher.models.regimes import RegimeMode, add_regime_feature, get_regime_aware_features

logger = logging.getLogger(__name__)

# Module-level variable to store last model IC for external access
_last_model_ic = 0.15


def clear_memory():
    """Force garbage collection to free memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Recommendation:
    """A stock recommendation from the ML model."""
    ticker: str
    score: float  # Raw ML score
    percentile: float  # Percentile rank in universe (0-100)
    rank: int  # Absolute rank in universe (1 = best)
    drivers: list[str] = field(default_factory=list)  # Top contributing features
    sector: str = ""
    predicted_return: float = 0.0  # Predicted forward return


# ==============================================================================
# UNIVERSES
# ==============================================================================

def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return [t for t in tickers if isinstance(t, str) and len(t) <= 5]
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500: {e}, using fallback")
        return get_sp100_tickers()


def get_sp100_tickers() -> list[str]:
    """S&P 100 constituents."""
    return [
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMZN", "AVGO",
        "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT", "CHTR",
        "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "DE",
        "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD", "GE",
        "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
        "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ",
        "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
        "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX",
        "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN",
        "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
    ]


def get_large_cap_tickers() -> list[str]:
    """Large cap tech + finance tickers."""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
        "JPM", "V", "MA", "JNJ", "UNH", "HD", "PG", "XOM", "BAC", "CVX",
        "ABBV", "MRK", "COST", "PEP", "KO", "LLY", "TMO", "AVGO", "ORCL",
        "CRM", "MCD", "ADBE", "AMD", "NFLX", "INTC", "CSCO", "QCOM",
    ]


UNIVERSES = {
    "sp500": get_sp500_tickers,
    "sp100": get_sp100_tickers,
    "large_cap": get_large_cap_tickers,
    "core_tech": lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "AMD", "INTC", "CRM", "ORCL", "ADBE", "NFLX", "PYPL",
    ],
}


# ==============================================================================
# DATA FETCHING
# ==============================================================================

def fetch_prices(
    tickers: list[str],
    lookback_days: int = 1260,  # ~5 years for sufficient training data
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.
    
    Args:
        tickers: List of ticker symbols.
        lookback_days: Number of calendar days to fetch.
        benchmark: Benchmark ticker to include.
    
    Returns:
        DataFrame with date index and ticker columns.
    """
    # Include benchmark
    all_tickers = list(set(tickers + [benchmark]))
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"Fetching prices for {len(all_tickers)} tickers...")
    
    try:
        # For large universes, fetch in batches to reduce memory pressure
        if len(all_tickers) > 100:
            batch_size = 50
            all_dfs = []
            
            for i in range(0, len(all_tickers), batch_size):
                batch_tickers = all_tickers[i:i + batch_size]
                logger.info(f"  Fetching batch {i//batch_size + 1}/{(len(all_tickers) + batch_size - 1)//batch_size}...")
                
                batch_df = yf.download(
                    batch_tickers,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                
                if isinstance(batch_df.columns, pd.MultiIndex):
                    batch_df = batch_df["Close"]
                
                all_dfs.append(batch_df)
                clear_memory()
            
            # Combine all batches
            df = pd.concat(all_dfs, axis=1)
            # Remove duplicate columns (benchmark may appear multiple times)
            df = df.loc[:, ~df.columns.duplicated()]
        else:
            df = yf.download(
                all_tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"]
        
        # Drop tickers with too many missing values
        missing_pct = df.isna().sum() / len(df)
        valid_cols = missing_pct[missing_pct < 0.1].index.tolist()
        df = df[valid_cols]
        
        # Forward fill small gaps, then drop remaining NaN
        df = df.ffill(limit=5)
        df = df.dropna(axis=1, how="any")
        
        logger.info(f"Fetched {len(df)} days for {len(df.columns)} tickers")
        return df
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch prices: {e}")
        raise


# ==============================================================================
# FEATURE COMPUTATION
# ==============================================================================

def compute_features_for_ranking(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    normalization_prices: pd.DataFrame | None = None,
    use_sector_neutral_norm: bool = False,
    regime_mode: RegimeMode = RegimeMode.FEATURE,
) -> pd.DataFrame:
    """
    Compute enhanced features for ML ranking.
    
    Uses the enhanced feature pipeline but returns features for the
    latest date only.
    
    Args:
        prices: DataFrame with date index and ticker columns.
        benchmark: Benchmark ticker for beta calculations.
        regime_mode: Whether to add regime_id as a feature.
    
    Returns:
        DataFrame with ticker index and feature columns (for latest date).
    """
    from auto_researcher.features.enhanced import (
        EnhancedFeatureConfig,
        compute_all_enhanced_features,
    )
    
    config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,  # Skip sector encoding for speed
        use_cross_sec_norm=True,
        cross_sec_norm_robust=True,
        cross_sec_norm_by_sector=use_sector_neutral_norm,
    )
    
    # Compute all enhanced features
    features_prices = normalization_prices if normalization_prices is not None else prices
    features_wide = compute_all_enhanced_features(
        features_prices,
        benchmark=benchmark,
        config=config,
    )
    
    # Stack to (date, ticker) x features format
    features_long = features_wide.stack(level=0, future_stack=True)
    features_long.index.names = ["date", "ticker"]
    
    # Get features for the latest date
    latest_date = features_long.index.get_level_values("date").max()
    latest_features = features_long.loc[latest_date]

    # If using a larger normalization universe, keep only target tickers
    if normalization_prices is not None:
        target_tickers = [t for t in prices.columns if t != benchmark]
        latest_features = latest_features.loc[latest_features.index.intersection(target_tickers)]
    
    # Drop any columns with NaN (features that need more history)
    latest_features = latest_features.dropna(axis=1, how="all")
    
    # Drop rows (tickers) with any NaN
    latest_features = latest_features.dropna(axis=0)
    
    # Cap extreme values at ±3 z-scores to prevent outliers from distorting predictions
    for col in latest_features.columns:
        latest_features[col] = latest_features[col].clip(-3, 3)
    
    # Remove benchmark from features
    if benchmark in latest_features.index:
        latest_features = latest_features.drop(benchmark)
    
    # Add regime feature if enabled
    if regime_mode == RegimeMode.FEATURE:
        latest_features = get_regime_aware_features(
            latest_features, latest_date, regime_mode
        )
    
    return latest_features


def compute_targets_for_training(
    prices: pd.DataFrame,
    horizon_days: int = 21,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Compute forward returns as training targets.
    
    Args:
        prices: DataFrame with date index and ticker columns.
        horizon_days: Forward return horizon in trading days.
        benchmark: Benchmark ticker.
    
    Returns:
        DataFrame with date index and ticker columns containing forward returns.
    """
    from auto_researcher.features.targets import TargetConfig, build_targets
    
    config = TargetConfig(
        mode="vol_norm",  # Volatility-normalized forward returns
        horizon_days=horizon_days,
    )
    
    targets = build_targets(prices, config, benchmark=benchmark)
    return targets


# ==============================================================================
# MODEL TRAINING & PREDICTION
# ==============================================================================

def train_ranking_model(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    horizon_days: int = 21,
    min_train_days: int = 252,
    training_window_months: int = 6,
    normalization_prices: pd.DataFrame | None = None,
    min_feature_ic: float = 0.01,
    use_dynamic_feature_ic: bool = True,
    feature_ic_quantile: float = 0.4,
    use_sector_neutral_norm: bool = False,
    regime_mode: RegimeMode = RegimeMode.FEATURE,
) -> tuple:
    """
    Train XGBoost regression model on historical data.
    
    Uses a rolling training window (default 6 months) rather than
    expanding window. Backtests show 6-month windows outperform
    longer windows across 13+ years of data (IC: 0.259 vs 0.129).
    
    Args:
        prices: Full price history.
        benchmark: Benchmark ticker.
        horizon_days: Forward return horizon for targets.
        min_train_days: Minimum training samples.
        training_window_months: Rolling window size in months (default 6).
    
    Returns:
        Tuple of (model, feature_columns, historical_ic).
    """
    from auto_researcher.models.xgb_ranking_model import (
        XGBRegressionModel,
        XGBRegressionConfig,
    )
    from auto_researcher.features.enhanced import (
        EnhancedFeatureConfig,
        compute_all_enhanced_features,
    )
    from auto_researcher.features.targets import TargetConfig, build_targets
    
    logger.info("Training ML ranking model on historical data...")
    
    # Feature config
    feature_config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,
        use_cross_sec_norm=True,
        cross_sec_norm_robust=True,
        cross_sec_norm_by_sector=use_sector_neutral_norm,
    )
    
    # Target config
    target_config = TargetConfig(
        mode="vol_norm",
        horizon_days=horizon_days,
    )
    
    # Exclude last horizon_days from training (we don't have targets for those)
    train_prices = prices.iloc[:-horizon_days]
    
    # Compute features (optionally normalize against a larger universe)
    features_prices = normalization_prices if normalization_prices is not None else train_prices
    features_wide = compute_all_enhanced_features(
        features_prices,
        benchmark=benchmark,
        config=feature_config,
    )
    
    # Stack to (date, ticker) x features
    features_long = features_wide.stack(level=0, future_stack=True)
    features_long.index.names = ["date", "ticker"]

    # If using a larger normalization universe, keep only target tickers
    if normalization_prices is not None:
        target_tickers = [t for t in train_prices.columns if t != benchmark]
        features_long = features_long[features_long.index.get_level_values("ticker").isin(target_tickers)]
    
    # Apply 6-month rolling window (backtest shows 6m IC=0.259 vs 24m IC=0.129)
    # Shorter windows adapt faster to regime changes
    training_window_days = training_window_months * 21
    all_dates = features_long.index.get_level_values("date").unique().sort_values()
    if len(all_dates) > training_window_days:
        cutoff_date = all_dates[-training_window_days]
        features_long = features_long[features_long.index.get_level_values("date") >= cutoff_date]
        logger.info(f"Using {training_window_months}-month rolling window: {len(features_long.index.get_level_values('date').unique())} training dates")
    
    # Clear intermediate data
    del features_wide
    clear_memory()
    
    # Add regime feature if enabled (before alignment with targets)
    if regime_mode == RegimeMode.FEATURE:
        features_long = add_regime_feature(features_long)
        logger.info(f"Added regime_id feature (mode={regime_mode.value})")
    
    # Compute targets
    targets_df = build_targets(train_prices, target_config, benchmark=benchmark)
    targets_stacked = targets_df.stack()
    targets_stacked.index.names = ["date", "ticker"]
    
    # Clear intermediate data
    del targets_df
    clear_memory()
    
    # Align features and targets
    common_idx = features_long.index.intersection(targets_stacked.index)
    X_train = features_long.loc[common_idx]
    y_train = targets_stacked.loc[common_idx]
    
    # Keep a copy for IC calculation before clearing
    features_for_ic = features_long.copy() if len(features_long) > 0 else None
    targets_for_ic = targets_stacked.copy() if len(targets_stacked) > 0 else None
    
    # Clear intermediate data
    del features_long, targets_stacked
    clear_memory()
    
    # Drop NaN
    valid_cols = X_train.columns[X_train.notna().any()]
    X_train = X_train[valid_cols]
    
    valid_rows = X_train.dropna().index.intersection(y_train.dropna().index)
    X_train = X_train.loc[valid_rows]
    y_train = y_train.loc[valid_rows]
    
    # Cap extreme feature values at ±3 z-scores for consistent training/inference
    for col in X_train.columns:
        X_train[col] = X_train[col].clip(-3, 3)
    
    # Feature IC pruning (remove weak or unstable features)
    if min_feature_ic is not None and len(X_train.columns) > 1:
        feature_ic = X_train.apply(lambda col: col.corr(y_train))
        feature_ic = feature_ic.dropna()

        ic_abs = feature_ic.abs()
        dynamic_threshold = min_feature_ic
        if use_dynamic_feature_ic and len(ic_abs) > 0:
            quantile_threshold = ic_abs.quantile(feature_ic_quantile)
            dynamic_threshold = max(min_feature_ic, float(quantile_threshold))

        keep_mask = ic_abs >= dynamic_threshold
        kept_features = feature_ic[keep_mask].index.tolist()

        # Ensure we keep at least 4 features (avoid overly aggressive pruning)
        if len(kept_features) < 4:
            kept_features = feature_ic.abs().sort_values(ascending=False).head(4).index.tolist()

        removed = [c for c in X_train.columns if c not in kept_features]
        if removed:
            logger.info(
                f"Pruned {len(removed)} low-IC features (< {dynamic_threshold:.3f})"
            )
        X_train = X_train[kept_features]

    logger.info(f"Training on {len(X_train)} samples with {len(X_train.columns)} features")
    
    # Adjust minimum based on sample availability
    actual_min = min(min_train_days, len(X_train) // 2)
    if len(X_train) < actual_min:
        raise ValueError(f"Insufficient training data: {len(X_train)} < {actual_min}")
    
    if len(X_train) < min_train_days:
        logger.warning(f"Training with limited data: {len(X_train)} samples (ideally {min_train_days})")
    
    # Compute recency weights - give more importance to recent data
    # This helps the model adapt to regime changes (e.g., crash-bounce worked in 2023 but not 2025)
    dates = X_train.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    sample_dates_idx = dates.map(date_to_idx).values
    
    # Exponential decay weight: recent samples get 2x weight vs oldest
    decay_rate = 0.5 / len(unique_dates)  # Half-life roughly at midpoint
    sample_weights = np.exp(decay_rate * sample_dates_idx)
    sample_weights = sample_weights / sample_weights.mean()  # Normalize
    
    # Create and train model
    config = XGBRegressionConfig(
        objective="reg:pseudohubererror",  # Robust to outliers
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        reg_lambda=2.0,
        reg_alpha=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    # Calculate historical IC (out-of-sample)
    # Use last month of training dates as validation, train on earlier dates only
    historical_ic = 0.15  # Default IC if calculation fails
    try:
        train_dates = X_train.index.get_level_values("date").unique().sort_values()
        if len(train_dates) >= 42:  # Need at least 2 months of dates
            val_dates = train_dates[-21:]
            ic_train_dates = train_dates[:-21]

            val_mask = X_train.index.get_level_values("date").isin(val_dates)
            train_mask = X_train.index.get_level_values("date").isin(ic_train_dates)

            X_ic_train = X_train[train_mask]
            y_ic_train = y_train.loc[X_ic_train.index]
            X_val = X_train[val_mask]
            y_val = y_train.loc[X_val.index]

            if len(X_val) > 20 and len(X_ic_train) > 100:
                dates_ic = X_ic_train.index.get_level_values("date")
                unique_dates_ic = dates_ic.unique().sort_values()
                date_to_idx_ic = {d: i for i, d in enumerate(unique_dates_ic)}
                sample_dates_idx_ic = dates_ic.map(date_to_idx_ic).values

                decay_rate_ic = 0.5 / len(unique_dates_ic)
                sample_weights_ic = np.exp(decay_rate_ic * sample_dates_idx_ic)
                sample_weights_ic = sample_weights_ic / sample_weights_ic.mean()

                ic_model = XGBRegressionModel(config)
                ic_model.fit(X_ic_train, y_ic_train, sample_weight=sample_weights_ic)
                preds = ic_model.predict(X_val)
                historical_ic = pd.Series(preds).corr(y_val.reset_index(drop=True))
                if pd.isna(historical_ic):
                    historical_ic = 0.15
                logger.info(f"Historical OOS IC: {historical_ic:.3f}")
    except Exception as e:
        logger.warning(f"Failed to calculate historical IC: {e}")

    # Train final model on full data
    model = XGBRegressionModel(config)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    logger.info("Model training complete")

    return model, X_train.columns.tolist(), historical_ic


def get_feature_importance(
    model,
    feature_names: list[str],
    top_k: int = 5,
) -> dict[str, list[str]]:
    """
    Get top feature drivers for each stock based on model feature importance.
    
    Args:
        model: Trained XGBoost model.
        feature_names: List of feature column names.
        top_k: Number of top features to return.
    
    Returns:
        Dict mapping ticker to list of top feature names.
    """
    try:
        importance = model.model.get_booster().get_score(importance_type="gain")
        
        # Map feature indices to names
        importance_named = {}
        for key, value in importance.items():
            if key.startswith("f"):
                idx = int(key[1:])
                if idx < len(feature_names):
                    importance_named[feature_names[idx]] = value
        
        # Sort by importance
        sorted_features = sorted(
            importance_named.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = [f[0] for f in sorted_features[:top_k]]
        return top_features
        
    except Exception as e:
        logger.warning(f"Failed to get feature importance: {e}")
        return []


# ==============================================================================
# MAIN RECOMMENDATION FUNCTION
# ==============================================================================

def generate_recommendations(
    tickers: Optional[list[str]] = None,
    universe: str = "sp100",
    top_k: int = 25,
    explain: bool = False,
    lookback_days: int = 1260,  # ~5 years to allow 252+ training dates after feature lookback
    benchmark: str = "SPY",
    horizon_days: int = 21,
    use_normalization_universe: bool = True,
    use_feature_ic_pruning: bool = True,
    feature_ic_threshold: float = 0.01,
    use_dynamic_feature_ic: bool = True,
    feature_ic_quantile: float = 0.4,
    use_sector_neutral_norm: bool = False,
    use_drawdown_penalty: bool = True,
    drawdown_threshold: float = -40.0,
    drawdown_penalty_scale: float = 0.5,
    use_score_normalization: bool = True,
    score_norm_robust: bool = True,
    use_hard_exclusion: bool = True,
    exclusion_return_threshold: float = -30.0,
    exclusion_drawdown_threshold: float = -50.0,
) -> tuple[list[Recommendation], pd.Series, pd.DataFrame]:
    """
    Generate stock recommendations using ML model.
    
    Args:
        tickers: List of tickers to analyze. If None, uses universe.
        universe: Universe name from UNIVERSES dict.
        top_k: Number of top recommendations to return.
        explain: Whether to include feature drivers in recommendations.
        lookback_days: Days of price history to fetch.
        benchmark: Benchmark ticker.
        horizon_days: Forward return horizon for model training.
    
    Returns:
        Tuple of:
        - List of top_k Recommendation objects
        - Series of all scores (ticker -> score)
        - DataFrame of prices used
    """
    logger.info(f"Generating recommendations for top {top_k} stocks...")
    
    # Get tickers
    if tickers is None:
        if universe in UNIVERSES:
            tickers = UNIVERSES[universe]()
        else:
            raise ValueError(f"Unknown universe: {universe}")
    
    logger.info(f"Universe: {len(tickers)} tickers")
    
    # Fetch prices
    prices = fetch_prices(tickers, lookback_days=lookback_days, benchmark=benchmark)

    # Use larger normalization universe for small universes to avoid distorted z-scores
    normalization_prices = None
    if use_normalization_universe and len(tickers) < 30:
        ref_tickers = UNIVERSES["sp100"]()
        try:
            normalization_prices = fetch_prices(
                ref_tickers,
                lookback_days=lookback_days,
                benchmark=benchmark,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch normalization universe: {e}")
            normalization_prices = None
    
    # Train model on historical data (6-month rolling window)
    model, feature_columns, model_ic = train_ranking_model(
        prices,
        benchmark=benchmark,
        horizon_days=horizon_days,
        training_window_months=6,  # Backtests show 6m window has 2x better IC than 24m
        normalization_prices=normalization_prices,
        min_feature_ic=feature_ic_threshold if use_feature_ic_pruning else None,
        use_dynamic_feature_ic=use_dynamic_feature_ic,
        feature_ic_quantile=feature_ic_quantile,
        use_sector_neutral_norm=use_sector_neutral_norm,
        regime_mode=RegimeMode.FEATURE,
    )
    
    # Store IC for external access (used by run_ranking_low_memory.py)
    global _last_model_ic
    _last_model_ic = model_ic
    
    logger.info(f"ML model historical IC: {model_ic:.3f}")
    
    # Compute current features
    current_features = compute_features_for_ranking(
        prices,
        benchmark=benchmark,
        normalization_prices=normalization_prices,
        use_sector_neutral_norm=use_sector_neutral_norm,
        regime_mode=RegimeMode.FEATURE,
    )

    # Hard exclusion filter for extreme underperformers
    if use_hard_exclusion:
        try:
            prices_df = prices if isinstance(prices, pd.DataFrame) else prices
            if len(prices_df) >= 252:
                returns_1y = (prices_df.iloc[-1] / prices_df.iloc[-252] - 1) * 100
                returns_1y = returns_1y.reindex(current_features.index)

                max_dd = {}
                window = prices_df.iloc[-252:]
                for ticker in current_features.index:
                    if ticker not in window.columns:
                        continue
                    series = window[ticker].dropna()
                    if len(series) < 30:
                        continue
                    running_max = series.cummax()
                    drawdown = (series / running_max - 1.0) * 100
                    max_dd[ticker] = float(drawdown.min())

                max_dd_series = pd.Series(max_dd).reindex(current_features.index)
                exclude_mask = (returns_1y <= exclusion_return_threshold) | (
                    max_dd_series <= exclusion_drawdown_threshold
                )

                excluded = exclude_mask[exclude_mask].index.tolist()
                if excluded:
                    logger.info(
                        f"Excluded {len(excluded)} stocks for extreme underperformance"
                    )
                    current_features = current_features.drop(index=excluded, errors="ignore")
        except Exception as e:
            logger.warning(f"Could not apply hard exclusion filter: {e}")
    
    # Ensure we only use trained feature columns
    available_cols = [c for c in feature_columns if c in current_features.columns]
    current_features = current_features[available_cols]
    current_features = current_features.dropna()
    
    logger.info(f"Generating predictions for {len(current_features)} tickers")
    
    # Generate predictions
    scores = model.predict_with_index(current_features)
    
    # Apply STRONG penalty for extreme long-term underperformers
    # The model gives high scores to crash stocks expecting reversal,
    # but stocks down 40%+ in 1 year often have structural problems.
    # Apply aggressive penalty that scales quadratically for extreme cases.
    if "tech_resid_mom_252" in current_features.columns:
        resid_mom_252 = current_features["tech_resid_mom_252"]
        # Penalty kicks in below -1 std (about -20% relative to market)
        # Linear component: 0.5 per std below -1
        # Quadratic component: extra 0.2 * (excess)^2 for extreme cases
        excess = np.maximum(0, -1 - resid_mom_252)  # How far below -1 std
        linear_penalty = excess * 0.5
        quadratic_penalty = 0.2 * (excess ** 2)
        penalty = -(linear_penalty + quadratic_penalty)
        # UNH at -2.5 std: excess=1.5, penalty = -(0.75 + 0.45) = -1.2
        penalized_tickers = (penalty < 0).sum()
        if penalized_tickers > 0:
            logger.info(f"Applied momentum penalty to {penalized_tickers} extreme underperformers")
        scores = scores + penalty
    
    # Additional safety check: Apply absolute performance penalty
    # Cross-sectional normalization can hide extreme underperformance in small universes.
    # Check raw 1-year returns and penalize severely underperforming stocks.
    try:
        prices_df = prices if isinstance(prices, pd.DataFrame) else prices
        if len(prices_df) >= 252:
            # Calculate raw 1-year returns
            returns_1y = (prices_df.iloc[-1] / prices_df.iloc[-252] - 1) * 100
            returns_1y = returns_1y.reindex(scores.index)
            
            # Severe penalty for stocks down > 30% in 1 year
            severe_underperformers = returns_1y < -30
            if severe_underperformers.any():
                # Additional penalty: -0.5 for every 10% below -30%
                extra_penalty = np.minimum(0, (returns_1y + 30) / 10 * 0.5)
                extra_penalty = extra_penalty.fillna(0)
                scores = scores + extra_penalty
                n_severe = severe_underperformers.sum()
                logger.info(f"Applied absolute return penalty to {n_severe} stocks down >30% in 1Y")
    except Exception as e:
        logger.warning(f"Could not apply absolute return penalty: {e}")

    # Drawdown penalty: penalize deep peak-to-trough drawdowns over the last 1 year
    if use_drawdown_penalty:
        try:
            prices_df = prices if isinstance(prices, pd.DataFrame) else prices
            if len(prices_df) >= 252:
                window = prices_df.iloc[-252:]
                max_dd = {}
                for ticker in scores.index:
                    if ticker not in window.columns:
                        continue
                    series = window[ticker].dropna()
                    if len(series) < 30:
                        continue
                    running_max = series.cummax()
                    drawdown = (series / running_max - 1.0) * 100
                    max_dd[ticker] = float(drawdown.min())

                if max_dd:
                    max_dd_series = pd.Series(max_dd).reindex(scores.index)
                    deep_dd = max_dd_series <= drawdown_threshold
                    if deep_dd.any():
                        # Penalty increases by drawdown_penalty_scale per 10% beyond threshold
                        extra = ((max_dd_series - drawdown_threshold) / 10.0).clip(upper=0)
                        dd_penalty = extra * drawdown_penalty_scale
                        dd_penalty = dd_penalty.fillna(0)
                        scores = scores + dd_penalty
                        logger.info(
                            f"Applied drawdown penalty to {int(deep_dd.sum())} stocks below {drawdown_threshold:.0f}%"
                        )
        except Exception as e:
            logger.warning(f"Could not apply drawdown penalty: {e}")

    # Normalize score distribution to reduce outlier influence across universes
    if use_score_normalization and len(scores) > 0:
        try:
            if score_norm_robust:
                center = scores.median()
                mad = (scores - center).abs().median()
                scale = mad * 1.4826
            else:
                center = scores.mean()
                scale = scores.std()

            if scale and scale > 0:
                scores = (scores - center) / scale
                scores = scores.clip(-5, 5)
        except Exception as e:
            logger.warning(f"Could not normalize score distribution: {e}")
    
    # Sort by score (higher = better)
    scores_sorted = scores.sort_values(ascending=False)
    
    # Compute percentiles
    percentiles = scores.rank(pct=True) * 100
    
    # Get feature drivers if requested
    top_drivers = []
    if explain:
        top_drivers = get_feature_importance(model, feature_columns, top_k=5)
    
    # Create recommendations
    recommendations = []
    for rank, (ticker, score) in enumerate(scores_sorted.head(top_k).items(), start=1):
        rec = Recommendation(
            ticker=ticker,
            score=float(score),
            percentile=float(percentiles.get(ticker, 50)),
            rank=rank,
            drivers=top_drivers,
            predicted_return=float(score),  # Score approximates predicted return
        )
        recommendations.append(rec)
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    
    return recommendations, scores, prices


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Generate ML stock recommendations")
    parser.add_argument("--universe", default="sp100", choices=list(UNIVERSES.keys()))
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument(
        "--no-normalization-universe",
        action="store_true",
        help="Disable larger reference universe for cross-sectional normalization",
    )
    parser.add_argument(
        "--no-feature-ic-pruning",
        action="store_true",
        help="Disable pruning of low-IC features",
    )
    parser.add_argument(
        "--feature-ic-threshold",
        type=float,
        default=0.01,
        help="IC threshold for feature pruning (default: 0.01)",
    )
    parser.add_argument(
        "--no-dynamic-feature-ic",
        action="store_true",
        help="Disable dynamic IC thresholding for feature pruning",
    )
    parser.add_argument(
        "--feature-ic-quantile",
        type=float,
        default=0.4,
        help="Quantile for dynamic IC threshold (default: 0.4)",
    )
    parser.add_argument(
        "--sector-neutral-norm",
        action="store_true",
        help="Enable sector-neutral cross-sectional normalization",
    )
    parser.add_argument(
        "--no-drawdown-penalty",
        action="store_true",
        help="Disable drawdown-based penalty",
    )
    parser.add_argument(
        "--drawdown-threshold",
        type=float,
        default=-40.0,
        help="Drawdown threshold for penalty (default: -40)",
    )
    parser.add_argument(
        "--drawdown-penalty-scale",
        type=float,
        default=0.5,
        help="Penalty per additional 10% drawdown beyond threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-score-normalization",
        action="store_true",
        help="Disable score distribution normalization",
    )
    parser.add_argument(
        "--score-norm-standard",
        action="store_true",
        help="Use mean/std instead of median/MAD for score normalization",
    )
    parser.add_argument(
        "--no-hard-exclusion",
        action="store_true",
        help="Disable hard exclusion filter for extreme underperformers",
    )
    parser.add_argument(
        "--exclusion-return-threshold",
        type=float,
        default=-30.0,
        help="Exclude stocks below this 1Y return (default: -30)",
    )
    parser.add_argument(
        "--exclusion-drawdown-threshold",
        type=float,
        default=-50.0,
        help="Exclude stocks below this 1Y max drawdown (default: -50)",
    )
    
    args = parser.parse_args()
    
    recommendations, all_scores, prices = generate_recommendations(
        universe=args.universe,
        top_k=args.top_k,
        explain=args.explain,
        use_normalization_universe=not args.no_normalization_universe,
        use_feature_ic_pruning=not args.no_feature_ic_pruning,
        feature_ic_threshold=args.feature_ic_threshold,
        use_dynamic_feature_ic=not args.no_dynamic_feature_ic,
        feature_ic_quantile=args.feature_ic_quantile,
        use_sector_neutral_norm=args.sector_neutral_norm,
        use_drawdown_penalty=not args.no_drawdown_penalty,
        drawdown_threshold=args.drawdown_threshold,
        drawdown_penalty_scale=args.drawdown_penalty_scale,
        use_score_normalization=not args.no_score_normalization,
        score_norm_robust=not args.score_norm_standard,
        use_hard_exclusion=not args.no_hard_exclusion,
        exclusion_return_threshold=args.exclusion_return_threshold,
        exclusion_drawdown_threshold=args.exclusion_drawdown_threshold,
    )
    
    print("\n" + "=" * 60)
    print(f"  TOP {args.top_k} STOCK RECOMMENDATIONS ({args.universe.upper()})")
    print("=" * 60)
    print(f"  {'Rank':<6}{'Ticker':<8}{'Score':>10}{'Percentile':>12}")
    print(f"  {'-'*36}")
    
    for rec in recommendations:
        print(f"  {rec.rank:<6}{rec.ticker:<8}{rec.score:>10.4f}{rec.percentile:>12.1f}")
    
    print("=" * 60)
