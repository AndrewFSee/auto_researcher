"""
Stage-1 ML screening: train the cross-sectional ranking model and score a universe.

This module replaces the repository-root ``recommend.py``, which the package
imported through ``sys.path`` hacks (and which therefore broke as soon as the
package was installed or run from another directory).

The training recipe lives in ``ScreeningModel`` so the *same object* is used
for live scoring and for walk-forward evaluation
(``scripts/ml_walkforward_backtest.py --models screening``). Evidence to date:
evaluated with the purged walk-forward harness on 103 US large caps
(2017-2025, monthly rebalances) the price-feature recipe has a mean IC
indistinguishable from zero. Treat the screen as a weak prior for which names
the agents look at, not as a source of alpha. See ``docs/AUDIT.md``.

Usage::

    from auto_researcher.screening import generate_recommendations, UNIVERSES

    recs, scores, prices = generate_recommendations(tickers=UNIVERSES["sp100"](), top_k=25)
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from auto_researcher.models.xgb_ranking_model import XGBRegressionConfig, XGBRegressionModel

logger = logging.getLogger(__name__)

# Most recent purged holdout IC from ``generate_recommendations`` (NaN if unknown).
_last_model_ic: float = float("nan")


def get_last_model_ic() -> float:
    """Purged holdout IC of the most recently trained screening model (NaN if unknown)."""
    return _last_model_ic


def clear_memory() -> None:
    """Force garbage collection to free memory between heavy stages."""
    gc.collect()


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class Recommendation:
    """A stock recommendation from the ML screen."""

    ticker: str
    score: float  # Robust z-score of the model output (after overlays)
    percentile: float  # Percentile rank in the scored universe (0-100)
    rank: int  # Absolute rank in universe (1 = best)
    drivers: list[str] = field(default_factory=list)  # This stock's largest feature contributions
    sector: str = ""
    predicted_return: float = 0.0  # Raw model output (target units, not a return forecast)


# ==============================================================================
# UNIVERSES
# ==============================================================================
# All universes are *current* constituent lists. Backtests on them are
# survivorship-biased (today's winners are in the list by construction), so
# compare strategies with the equal-weight portfolio of the same universe,
# never only with SPY.


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia (falls back to S&P 100)."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return [t for t in tickers if isinstance(t, str) and len(t) <= 5]
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500: {e}, using fallback")
        return get_sp100_tickers()


def get_sp100_tickers() -> list[str]:
    """S&P 100 constituents (current list)."""
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
    """Large-cap tech + finance tickers."""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
        "JPM", "V", "MA", "JNJ", "UNH", "HD", "PG", "XOM", "BAC", "CVX",
        "ABBV", "MRK", "COST", "PEP", "KO", "LLY", "TMO", "AVGO", "ORCL",
        "CRM", "MCD", "ADBE", "AMD", "NFLX", "INTC", "CSCO", "QCOM",
    ]


def get_core_tech_tickers() -> list[str]:
    """Mega-cap technology names."""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "AMD", "INTC", "CRM", "ORCL", "ADBE", "NFLX", "PYPL",
    ]


UNIVERSES = {
    "sp500": get_sp500_tickers,
    "sp100": get_sp100_tickers,
    "large_cap": get_large_cap_tickers,
    "core_tech": get_core_tech_tickers,
}


# ==============================================================================
# DATA FETCHING
# ==============================================================================


def fetch_prices(
    tickers: list[str],
    lookback_days: int = 1260,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for ``tickers`` plus the benchmark from yfinance.

    Tickers missing more than 10% of days are dropped, short gaps (<= 5 days)
    are forward-filled, and any ticker still incomplete is dropped.

    Args:
        tickers: Ticker symbols.
        lookback_days: Calendar days of history to fetch.
        benchmark: Benchmark ticker to include.

    Returns:
        DataFrame with a date index and one column per ticker.
    """
    import yfinance as yf

    all_tickers = list(dict.fromkeys([*tickers, benchmark]))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    logger.info(f"Fetching prices for {len(all_tickers)} tickers...")

    batch_size = 50 if len(all_tickers) > 100 else len(all_tickers)
    frames = []
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        raw = yf.download(
            batch,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw["Close"]
        frames.append(raw)
        clear_memory()

    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    missing_pct = df.isna().mean()
    df = df[missing_pct[missing_pct < 0.1].index]
    df = df.ffill(limit=5).dropna(axis=1, how="any")

    logger.info(f"Fetched {len(df)} days for {len(df.columns)} tickers")
    return df


# ==============================================================================
# FEATURES
# ==============================================================================


def build_feature_panel(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    normalization_prices: pd.DataFrame | None = None,
    use_sector_neutral_norm: bool = False,
) -> pd.DataFrame:
    """
    Price-based feature panel indexed by ``(date, ticker)``.

    Features are the causal technical set from ``features.enhanced``
    (short-term reversal, residual momentum, idiosyncratic volatility, MAD
    dispersion, MA-ratio trend), cross-sectionally robust-z-scored per date
    and clipped to +/-3.

    Args:
        prices: Wide price panel including ``benchmark``.
        benchmark: Benchmark column used for betas/residuals.
        normalization_prices: Optional larger panel whose cross-section is used
            for z-scoring (helps small universes); rows are then filtered back
            to ``prices``' tickers.
        use_sector_neutral_norm: Z-score within sectors instead of globally.
    """
    from auto_researcher.features.enhanced import (
        EnhancedFeatureConfig,
        compute_all_enhanced_features,
    )

    source = normalization_prices if normalization_prices is not None else prices
    if benchmark not in source.columns:
        # Without it every residual-momentum / idiosyncratic-vol feature is
        # silently skipped, which changes the model's inputs.
        raise ValueError(f"benchmark {benchmark!r} is missing from the price panel")

    config = EnhancedFeatureConfig(
        use_short_reversal=True,
        use_residual_mom=True,
        use_idio_vol=True,
        use_mad_metrics=True,
        use_sector_ohe=False,
        use_cross_sec_norm=True,
        cross_sec_norm_robust=True,
        cross_sec_norm_by_sector=use_sector_neutral_norm,
    )
    wide = compute_all_enhanced_features(source, benchmark=benchmark, config=config)
    panel = wide.stack(level=0, future_stack=True)
    panel.index = panel.index.set_names(["date", "ticker"])

    keep = [t for t in prices.columns if t != benchmark]
    panel = panel[panel.index.get_level_values("ticker").isin(keep)]
    panel = panel.dropna(axis=1, how="all")
    return panel.clip(-3, 3)


# ==============================================================================
# MODEL
# ==============================================================================


@dataclass
class ScreeningModelConfig:
    """
    The screening model's training recipe.

    Attributes:
        xgb: XGBoost hyperparameters.
        min_feature_ic: Drop features whose |mean per-date Spearman IC| with the
            training target is below this (``None`` disables pruning).
        use_dynamic_feature_ic: Raise the pruning threshold to the
            ``feature_ic_quantile`` quantile of |IC| across features.
        feature_ic_quantile: Quantile used by the dynamic threshold.
        min_features: Never keep fewer features than this.
        recency_weighting: Weight recent training dates more (the newest date
            gets ~e^0.5 = 1.65x the weight of the oldest).
    """

    xgb: XGBRegressionConfig = field(
        default_factory=lambda: XGBRegressionConfig(
            objective="reg:pseudohubererror",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            reg_lambda=2.0,
            reg_alpha=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=None,
        )
    )
    min_feature_ic: float | None = 0.01
    use_dynamic_feature_ic: bool = True
    feature_ic_quantile: float = 0.4
    min_features: int = 4
    recency_weighting: bool = True


def per_date_spearman_ic(values: pd.Series, target: pd.Series) -> float:
    """Mean over dates of the cross-sectional Spearman correlation (the IC)."""
    frame = pd.DataFrame({"x": values, "y": target}).dropna()
    if frame.empty:
        return float("nan")
    by_date = frame.groupby(level=0)
    ranks = by_date.rank()
    centered = ranks - ranks.groupby(level=0).transform("mean")
    dx, dy = centered["x"], centered["y"]
    num = (dx * dy).groupby(level=0).sum()
    den = np.sqrt((dx**2).groupby(level=0).sum() * (dy**2).groupby(level=0).sum())
    per_date = (num / den.replace(0.0, np.nan)).where(by_date.size() >= 3)
    return float(per_date.mean())


class ScreeningModel:
    """
    Feature pruning + recency-weighted XGBoost regression, with a
    scikit-learn style interface so the walk-forward harness can evaluate
    exactly what runs live.
    """

    def __init__(self, config: ScreeningModelConfig | None = None) -> None:
        self.config = config or ScreeningModelConfig()
        self.features_: list[str] = []
        self.feature_ic_: pd.Series = pd.Series(dtype=float)
        self.model_: XGBRegressionModel | None = None

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        cfg = self.config
        if cfg.min_feature_ic is None or X.shape[1] <= 1:
            return list(X.columns)
        if isinstance(X.index, pd.MultiIndex):
            ic = pd.Series({c: per_date_spearman_ic(X[c], y) for c in X.columns})
        else:
            ic = X.apply(lambda col: col.corr(y, method="spearman"))
        ic = ic.dropna()
        self.feature_ic_ = ic
        if ic.empty:
            return list(X.columns)
        threshold = cfg.min_feature_ic
        if cfg.use_dynamic_feature_ic:
            threshold = max(threshold, float(ic.abs().quantile(cfg.feature_ic_quantile)))
        kept = ic[ic.abs() >= threshold].index.tolist()
        if len(kept) < cfg.min_features:
            kept = ic.abs().sort_values(ascending=False).head(cfg.min_features).index.tolist()
        return [str(c) for c in kept]

    @staticmethod
    def _recency_weights(index: pd.Index) -> np.ndarray | None:
        if not isinstance(index, pd.MultiIndex):
            return None
        dates = index.get_level_values(0)
        codes, uniques = pd.factorize(dates, sort=True)
        decay = 0.5 / max(len(uniques), 1)
        weights = np.exp(decay * codes)
        return np.asarray(weights / weights.mean(), dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ScreeningModel:
        valid = y.notna().to_numpy() & X.notna().all(axis=1).to_numpy()
        X, y = X[valid], y[valid]
        self.features_ = self._select_features(X, y)
        weights = self._recency_weights(X.index) if self.config.recency_weighting else None
        self.model_ = XGBRegressionModel(self.config.xgb)
        self.model_.fit(X[self.features_], y, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("model has not been fitted")
        return self.model_.predict(X[self.features_])

    def feature_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row SHAP contributions (XGBoost ``pred_contribs``), one column per feature."""
        if self.model_ is None or self.model_.model is None:
            raise ValueError("model has not been fitted")
        import xgboost as xgb

        booster = self.model_.model.get_booster()
        contrib = booster.predict(xgb.DMatrix(X[self.features_].to_numpy()), pred_contribs=True)
        return pd.DataFrame(contrib[:, :-1], index=X.index, columns=self.features_)


def estimate_holdout_ic(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    train_dates: int = 126,
    holdout_dates: int = 63,
    config: ScreeningModelConfig | None = None,
) -> float:
    """
    One purged holdout estimate of the recipe's IC.

    Fits on ``train_dates`` dates, skips ``horizon_days`` dates so no training
    label overlaps the holdout, then scores the last ``holdout_dates`` dates and
    returns the mean per-date Spearman IC. (The previous estimate validated on
    the 21 dates immediately after training with 21-day labels and no gap, and
    reported a pooled Pearson correlation, so it measured label overlap more
    than skill.) With overlapping labels a 63-date holdout holds only about
    three independent observations, so treat the number as a noisy diagnostic;
    the walk-forward report is the evidence.
    """
    dates = pd.DatetimeIndex(X.index.get_level_values(0).unique()).sort_values()
    needed = train_dates + horizon_days + holdout_dates
    if len(dates) < needed:
        return float("nan")
    holdout = dates[-holdout_dates:]
    fit_dates = dates[-needed : -(horizon_days + holdout_dates)]
    level = X.index.get_level_values(0)
    fit_mask, hold_mask = level.isin(fit_dates), level.isin(holdout)
    model = ScreeningModel(config).fit(X[fit_mask], y[fit_mask])
    preds = pd.Series(model.predict(X[hold_mask]), index=X.index[hold_mask])
    return per_date_spearman_ic(preds, y[hold_mask])


def train_ranking_model(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    horizon_days: int = 21,
    training_window_months: int = 6,
    normalization_prices: pd.DataFrame | None = None,
    use_sector_neutral_norm: bool = False,
    config: ScreeningModelConfig | None = None,
) -> tuple[ScreeningModel, list[str], float, pd.DataFrame]:
    """
    Train the screening model on the most recent ``training_window_months``.

    Targets are 21-day vol-normalized forward returns, so the last
    ``horizon_days`` dates (whose labels are not yet realized) are excluded.

    Returns:
        ``(model, feature_columns, holdout_ic, feature_panel)``. The feature
        panel covers all dates so callers can score the latest date without
        recomputing features.
    """
    from auto_researcher.features.targets import TargetConfig, build_targets

    logger.info("Training ML screening model...")
    panel = build_feature_panel(
        prices,
        benchmark=benchmark,
        normalization_prices=normalization_prices,
        use_sector_neutral_norm=use_sector_neutral_norm,
    )
    targets = build_targets(
        prices, TargetConfig(mode="vol_norm", horizon_days=horizon_days), benchmark=benchmark
    ).stack(future_stack=True)
    targets.index = targets.index.set_names(["date", "ticker"])

    data = panel.join(targets.rename("target"), how="inner").dropna()
    X_all, y_all = data.drop(columns="target"), data["target"]

    holdout_ic = estimate_holdout_ic(
        X_all, y_all, horizon_days, train_dates=training_window_months * 21, config=config
    )
    if np.isfinite(holdout_ic):
        logger.info(f"Purged holdout IC (diagnostic): {holdout_ic:+.3f}")

    dates = pd.DatetimeIndex(X_all.index.get_level_values(0).unique()).sort_values()
    window = training_window_months * 21
    recent = dates[-window:] if len(dates) > window else dates
    mask = X_all.index.get_level_values(0).isin(recent)
    if mask.sum() < 100:
        raise ValueError(f"Insufficient training data: {int(mask.sum())} rows")

    model = ScreeningModel(config).fit(X_all[mask], y_all[mask])
    logger.info(
        f"Trained on {int(mask.sum())} rows, {len(recent)} dates, "
        f"{len(model.features_)} features: {model.features_}"
    )
    return model, list(model.features_), holdout_ic, panel


# ==============================================================================
# RECOMMENDATIONS
# ==============================================================================


def _trailing_drawdown_pct(prices: pd.DataFrame, tickers: pd.Index, days: int = 252) -> pd.Series:
    """Worst peak-to-trough drawdown (percent) over the trailing ``days`` for each ticker."""
    window = prices.iloc[-days:]
    out = {}
    for ticker in tickers:
        if ticker not in window.columns:
            continue
        series = window[ticker].dropna()
        if len(series) < 30:
            continue
        out[ticker] = float((series / series.cummax() - 1.0).min() * 100)
    return pd.Series(out, dtype=float).reindex(tickers)


def generate_recommendations(
    tickers: list[str] | None = None,
    universe: str = "sp100",
    top_k: int = 25,
    explain: bool = False,
    lookback_days: int = 1260,
    benchmark: str = "SPY",
    horizon_days: int = 21,
    use_normalization_universe: bool = True,
    use_sector_neutral_norm: bool = False,
    apply_heuristic_overlays: bool = True,
    drawdown_threshold: float = -40.0,
    drawdown_penalty_scale: float = 0.5,
    exclusion_return_threshold: float = -30.0,
    exclusion_drawdown_threshold: float = -50.0,
    score_norm_robust: bool = True,
    model_config: ScreeningModelConfig | None = None,
) -> tuple[list[Recommendation], pd.Series, pd.DataFrame]:
    """
    Score a universe with the ML screen and return the top ``top_k``.

    ``apply_heuristic_overlays`` keeps the legacy post-model rules: exclude
    names down more than 30% over a year or with a >50% drawdown, penalize
    extreme negative residual momentum, deep one-year losses and deep
    drawdowns. They were hand-tuned on a few recent anecdotes and have **not**
    been validated in a walk-forward backtest; they are risk filters, not alpha.

    Returns:
        ``(recommendations, scores, prices)``: the top-k ``Recommendation``
        objects, robust z-scores for every scored ticker, and the price panel.
    """
    global _last_model_ic

    if tickers is None:
        if universe not in UNIVERSES:
            raise ValueError(f"Unknown universe: {universe}")
        tickers = UNIVERSES[universe]()
    logger.info(f"Screening {len(tickers)} tickers for the top {top_k}...")

    prices = fetch_prices(tickers, lookback_days=lookback_days, benchmark=benchmark)

    normalization_prices = None
    if use_normalization_universe and len(tickers) < 30:
        try:
            ref = fetch_prices(UNIVERSES["sp100"](), lookback_days=lookback_days, benchmark=benchmark)
            normalization_prices = ref
        except Exception as e:
            logger.warning(f"Failed to fetch normalization universe: {e}")

    model, feature_columns, holdout_ic, panel = train_ranking_model(
        prices,
        benchmark=benchmark,
        horizon_days=horizon_days,
        normalization_prices=normalization_prices,
        use_sector_neutral_norm=use_sector_neutral_norm,
        config=model_config,
    )
    _last_model_ic = holdout_ic

    latest_date = panel.index.get_level_values("date").max()
    current = panel.xs(latest_date, level="date")[feature_columns].dropna()

    if apply_heuristic_overlays and len(prices) >= 252:
        returns_1y = (prices.iloc[-1] / prices.iloc[-252] - 1) * 100
        drawdown_1y = _trailing_drawdown_pct(prices, current.index)
        exclude = (returns_1y.reindex(current.index) <= exclusion_return_threshold) | (
            drawdown_1y <= exclusion_drawdown_threshold
        )
        if exclude.any():
            logger.info(f"Excluded {int(exclude.sum())} extreme underperformers")
            current = current[~exclude]

    logger.info(f"Scoring {len(current)} tickers as of {latest_date.date()}")
    raw = pd.Series(model.predict(current), index=current.index, name="score")
    scores = raw.copy()

    if apply_heuristic_overlays:
        if "tech_resid_mom_252" in current.columns:
            excess = np.maximum(0.0, -1.0 - current["tech_resid_mom_252"])
            scores -= 0.5 * excess + 0.2 * excess**2
        if len(prices) >= 252:
            returns_1y = ((prices.iloc[-1] / prices.iloc[-252] - 1) * 100).reindex(scores.index)
            scores += np.minimum(0.0, (returns_1y + 30) / 10 * 0.5).fillna(0.0)
            drawdown_1y = _trailing_drawdown_pct(prices, scores.index)
            extra = ((drawdown_1y - drawdown_threshold) / 10.0).clip(upper=0)
            scores += (extra * drawdown_penalty_scale).fillna(0.0)

    if len(scores) > 1:
        if score_norm_robust:
            center = scores.median()
            scale = (scores - center).abs().median() * 1.4826
        else:
            center, scale = scores.mean(), scores.std()
        if scale and np.isfinite(scale) and scale > 0:
            scores = ((scores - center) / scale).clip(-5, 5)

    ranked = scores.sort_values(ascending=False)
    percentiles = scores.rank(pct=True) * 100

    drivers: dict[str, list[str]] = {}
    if explain and len(current):
        contrib = model.feature_contributions(current)
        for ticker, row in contrib.iterrows():
            top = row.abs().sort_values(ascending=False).head(3).index
            drivers[ticker] = [f"{f} ({row[f]:+.3f})" for f in top]

    recommendations = [
        Recommendation(
            ticker=ticker,
            score=float(score),
            percentile=float(percentiles.get(ticker, 50.0)),
            rank=rank,
            drivers=drivers.get(ticker, []),
            predicted_return=float(raw.get(ticker, math.nan)),
        )
        for rank, (ticker, score) in enumerate(ranked.head(top_k).items(), start=1)
    ]
    logger.info(f"Generated {len(recommendations)} recommendations")
    return recommendations, scores, prices


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m auto_researcher.screening --universe sp100 --top-k 25``."""
    import argparse

    from auto_researcher.console import use_utf8_output

    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ML stock screen (stage 1)")
    parser.add_argument("--universe", default="sp100", choices=list(UNIVERSES))
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--explain", action="store_true", help="Show per-stock feature drivers")
    parser.add_argument(
        "--no-overlays", action="store_true", help="Disable the unvalidated heuristic overlays"
    )
    args = parser.parse_args(argv)

    recs, _, _ = generate_recommendations(
        universe=args.universe,
        top_k=args.top_k,
        explain=args.explain,
        apply_heuristic_overlays=not args.no_overlays,
    )
    print(f"\nTop {args.top_k} ({args.universe}); purged holdout IC {get_last_model_ic():+.3f}")
    print(f"  {'Rank':<6}{'Ticker':<8}{'Score':>8}{'Pctl':>8}  Drivers")
    for r in recs:
        print(f"  {r.rank:<6}{r.ticker:<8}{r.score:>8.2f}{r.percentile:>8.1f}  {', '.join(r.drivers)}")


if __name__ == "__main__":
    main()
