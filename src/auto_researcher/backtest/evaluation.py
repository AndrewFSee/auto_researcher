"""
Evaluation utilities for enhanced rank-based models.

This module provides metrics and logging utilities specific to:
- Cross-sectional ranking evaluation (IC, IC_IR, rank correlation)
- Feature importance analysis for enhanced features
- Model comparison and reporting
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ICMetrics:
    """Summary of Information Coefficient metrics.
    
    Attributes:
        mean_ic: Mean IC across all periods.
        std_ic: Standard deviation of IC.
        ic_ir: Information ratio = mean / std.
        hit_rate: Fraction of periods with positive IC.
        n_periods: Number of periods evaluated.
        ic_series: Full IC time series.
    """
    mean_ic: float
    std_ic: float
    ic_ir: float
    hit_rate: float
    n_periods: int
    ic_series: Optional[pd.Series] = None


def compute_cross_sectional_ic(
    predictions: pd.Series,
    actuals: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Compute information coefficient (rank correlation) for each date.
    
    Args:
        predictions: Predicted scores with MultiIndex (date, ticker).
        actuals: Actual returns with matching index.
        method: Correlation method ("spearman" or "pearson").
    
    Returns:
        Series of IC values indexed by date.
    """
    from scipy.stats import spearmanr, pearsonr
    
    # Align indices
    common_idx = predictions.index.intersection(actuals.index)
    predictions = predictions.loc[common_idx]
    actuals = actuals.loc[common_idx]
    
    ics = {}
    dates = predictions.index.get_level_values(0).unique()
    
    for date in dates:
        try:
            pred_date = predictions.loc[date]
            actual_date = actuals.loc[date]
            
            if len(pred_date) < 3:
                continue
            
            # Remove NaN values
            mask = ~(pred_date.isna() | actual_date.isna())
            pred_clean = pred_date[mask]
            actual_clean = actual_date[mask]
            
            if len(pred_clean) < 3:
                continue
            
            if method == "spearman":
                ic, _ = spearmanr(pred_clean, actual_clean)
            else:
                ic, _ = pearsonr(pred_clean, actual_clean)
            
            if not np.isnan(ic):
                ics[date] = ic
                
        except Exception as e:
            logger.debug(f"Error computing IC for {date}: {e}")
            continue
    
    return pd.Series(ics, name="IC")


def compute_ic_metrics(
    predictions: pd.Series,
    actuals: pd.Series,
    method: str = "spearman",
) -> ICMetrics:
    """Compute comprehensive IC metrics.
    
    Args:
        predictions: Predicted scores with MultiIndex (date, ticker).
        actuals: Actual returns with matching index.
        method: Correlation method.
    
    Returns:
        ICMetrics dataclass with summary statistics.
    """
    ic_series = compute_cross_sectional_ic(predictions, actuals, method)
    
    if len(ic_series) == 0:
        return ICMetrics(
            mean_ic=0.0,
            std_ic=0.0,
            ic_ir=0.0,
            hit_rate=0.0,
            n_periods=0,
            ic_series=ic_series,
        )
    
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit_rate = (ic_series > 0).mean()
    
    return ICMetrics(
        mean_ic=mean_ic,
        std_ic=std_ic,
        ic_ir=ic_ir,
        hit_rate=hit_rate,
        n_periods=len(ic_series),
        ic_series=ic_series,
    )


def compute_rolling_ic(
    predictions: pd.Series,
    actuals: pd.Series,
    window: int = 12,
) -> pd.Series:
    """Compute rolling average IC.
    
    Args:
        predictions: Predicted scores.
        actuals: Actual returns.
        window: Rolling window size (number of periods).
    
    Returns:
        Rolling mean IC series.
    """
    ic_series = compute_cross_sectional_ic(predictions, actuals)
    return ic_series.rolling(window=window, min_periods=1).mean()


def compute_quantile_returns(
    predictions: pd.Series,
    actuals: pd.Series,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute average returns by prediction quantile.
    
    Args:
        predictions: Predicted scores.
        actuals: Actual returns.
        n_quantiles: Number of quantiles.
    
    Returns:
        DataFrame with quantile-level return statistics.
    """
    # Align
    common_idx = predictions.index.intersection(actuals.index)
    predictions = predictions.loc[common_idx]
    actuals = actuals.loc[common_idx]
    
    results = []
    dates = predictions.index.get_level_values(0).unique()
    
    for date in dates:
        try:
            pred_date = predictions.loc[date]
            actual_date = actuals.loc[date]
            
            # Assign quantiles
            quantiles = pd.qcut(
                pred_date.rank(method="first"),
                q=n_quantiles,
                labels=list(range(1, n_quantiles + 1)),
            )
            
            for q in range(1, n_quantiles + 1):
                mask = quantiles == q
                if mask.sum() > 0:
                    results.append({
                        "date": date,
                        "quantile": q,
                        "return": actual_date[mask].mean(),
                        "n_stocks": mask.sum(),
                    })
        except Exception:
            continue
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Summarize by quantile
    summary = df.groupby("quantile").agg({
        "return": ["mean", "std"],
        "n_stocks": "mean",
    }).reset_index()
    
    summary.columns = ["quantile", "mean_return", "std_return", "avg_n_stocks"]
    
    return summary


def compute_long_short_spread(
    predictions: pd.Series,
    actuals: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """Compute long-short spread (top quantile - bottom quantile).
    
    Args:
        predictions: Predicted scores.
        actuals: Actual returns.
        n_quantiles: Number of quantiles.
    
    Returns:
        Series of long-short spreads by date.
    """
    common_idx = predictions.index.intersection(actuals.index)
    predictions = predictions.loc[common_idx]
    actuals = actuals.loc[common_idx]
    
    spreads = {}
    dates = predictions.index.get_level_values(0).unique()
    
    for date in dates:
        try:
            pred_date = predictions.loc[date]
            actual_date = actuals.loc[date]
            
            # Assign quantiles
            quantiles = pd.qcut(
                pred_date.rank(method="first"),
                q=n_quantiles,
                labels=list(range(1, n_quantiles + 1)),
            )
            
            top_return = actual_date[quantiles == n_quantiles].mean()
            bottom_return = actual_date[quantiles == 1].mean()
            
            spreads[date] = top_return - bottom_return
            
        except Exception:
            continue
    
    return pd.Series(spreads, name="long_short_spread")


def analyze_feature_importance_by_category(
    importance: pd.Series,
) -> pd.DataFrame:
    """Analyze feature importance grouped by category.
    
    Categories:
    - tech_mom: Momentum features
    - tech_vol: Volatility features
    - tech_resid: Residual/beta-adjusted features
    - tech_mad: MAD features
    - sector: Sector indicators
    - other: Uncategorized
    
    Args:
        importance: Feature importance series.
    
    Returns:
        DataFrame with category-level importance.
    """
    categories = {
        "tech_mom": ["tech_mom_", "mom_"],
        "tech_vol": ["tech_vol_", "vol_"],
        "tech_resid": ["tech_resid_", "tech_beta_"],
        "tech_mad": ["tech_mad_"],
        "tech_reversal": ["tech_rev_", "tech_mom_5d", "tech_mom_10d"],
        "sector": ["sector_"],
        "value": ["value_"],
        "quality": ["quality_"],
        "growth": ["growth_"],
        "profitability": ["prof_"],
        "size": ["size_"],
    }
    
    category_importance = {cat: 0.0 for cat in categories}
    category_importance["other"] = 0.0
    category_counts = {cat: 0 for cat in categories}
    category_counts["other"] = 0
    
    for feat, imp in importance.items():
        assigned = False
        for cat, prefixes in categories.items():
            if any(feat.startswith(p) for p in prefixes):
                category_importance[cat] += imp
                category_counts[cat] += 1
                assigned = True
                break
        if not assigned:
            category_importance["other"] += imp
            category_counts["other"] += 1
    
    # Remove empty categories
    category_importance = {k: v for k, v in category_importance.items() if v > 0}
    category_counts = {k: v for k, v in category_counts.items() if k in category_importance}
    
    # Normalize
    total = sum(category_importance.values())
    if total > 0:
        pct = {k: v / total for k, v in category_importance.items()}
    else:
        pct = {k: 0.0 for k in category_importance}
    
    df = pd.DataFrame([
        {
            "category": k,
            "importance": category_importance[k],
            "pct": pct[k],
            "n_features": category_counts[k],
        }
        for k in category_importance
    ])
    
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def log_model_evaluation(
    ic_metrics: ICMetrics,
    feature_importance: Optional[pd.Series] = None,
    model_type: str = "unknown",
    target_mode: str = "unknown",
) -> None:
    """Log comprehensive model evaluation results.
    
    Args:
        ic_metrics: IC metrics from evaluation.
        feature_importance: Optional feature importance series.
        model_type: Type of model used.
        target_mode: Target engineering mode used.
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Target Mode: {target_mode}")
    logger.info("-" * 40)
    
    logger.info("Information Coefficient Metrics:")
    logger.info(f"  Mean IC:    {ic_metrics.mean_ic:>8.4f}")
    logger.info(f"  Std IC:     {ic_metrics.std_ic:>8.4f}")
    logger.info(f"  IC IR:      {ic_metrics.ic_ir:>8.4f}")
    logger.info(f"  Hit Rate:   {ic_metrics.hit_rate * 100:>8.2f}%")
    logger.info(f"  N Periods:  {ic_metrics.n_periods:>8d}")
    
    # Interpretation
    logger.info("-" * 40)
    if ic_metrics.ic_ir > 0.5:
        logger.info("✓ Strong IC_IR (>0.5): Consistent predictive signal")
    elif ic_metrics.ic_ir > 0.3:
        logger.info("○ Moderate IC_IR (0.3-0.5): Reasonable signal")
    else:
        logger.info("✗ Weak IC_IR (<0.3): Signal needs improvement")
    
    if ic_metrics.hit_rate > 0.55:
        logger.info("✓ Good hit rate (>55%): More positive IC periods")
    elif ic_metrics.hit_rate < 0.45:
        logger.info("✗ Poor hit rate (<45%): Many negative IC periods")
    
    if feature_importance is not None and len(feature_importance) > 0:
        logger.info("-" * 40)
        logger.info("Top 10 Important Features:")
        for i, (feat, imp) in enumerate(feature_importance.head(10).items()):
            logger.info(f"  {i+1:2d}. {feat:<40} {imp:>8.4f}")
        
        # Category analysis
        category_df = analyze_feature_importance_by_category(feature_importance)
        logger.info("-" * 40)
        logger.info("Importance by Category:")
        for _, row in category_df.iterrows():
            logger.info(
                f"  {row['category']:<20} {row['pct']*100:>6.1f}% "
                f"({row['n_features']} features)"
            )
    
    logger.info("=" * 60)


def format_evaluation_report(
    ic_metrics: ICMetrics,
    quantile_returns: Optional[pd.DataFrame] = None,
    feature_importance: Optional[pd.Series] = None,
) -> str:
    """Format a text-based evaluation report.
    
    Args:
        ic_metrics: IC metrics.
        quantile_returns: Optional quantile return analysis.
        feature_importance: Optional feature importance.
    
    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        "ENHANCED MODEL EVALUATION REPORT",
        "=" * 60,
        "",
        "INFORMATION COEFFICIENT ANALYSIS",
        "-" * 40,
        f"  Mean IC:           {ic_metrics.mean_ic:>10.4f}",
        f"  IC Std Dev:        {ic_metrics.std_ic:>10.4f}",
        f"  IC Information Ratio: {ic_metrics.ic_ir:>10.4f}",
        f"  Hit Rate:          {ic_metrics.hit_rate*100:>10.2f}%",
        f"  Periods Evaluated: {ic_metrics.n_periods:>10d}",
        "",
    ]
    
    if quantile_returns is not None and len(quantile_returns) > 0:
        lines.append("QUANTILE RETURN ANALYSIS")
        lines.append("-" * 40)
        for _, row in quantile_returns.iterrows():
            lines.append(
                f"  Q{int(row['quantile'])}: Mean={row['mean_return']*100:>7.2f}% "
                f"Std={row['std_return']*100:>7.2f}%"
            )
        
        # Monotonicity check
        if len(quantile_returns) >= 2:
            top_q = quantile_returns["mean_return"].iloc[-1]
            bottom_q = quantile_returns["mean_return"].iloc[0]
            spread = top_q - bottom_q
            lines.append(f"\n  Long-Short Spread: {spread*100:>7.2f}%")
        lines.append("")
    
    if feature_importance is not None and len(feature_importance) > 0:
        lines.append("TOP FEATURES")
        lines.append("-" * 40)
        for i, (feat, imp) in enumerate(feature_importance.head(10).items()):
            lines.append(f"  {i+1:2d}. {feat:<35} {imp:>8.4f}")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
