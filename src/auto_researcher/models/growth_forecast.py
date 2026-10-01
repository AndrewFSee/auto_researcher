"""
Purged walk-forward forecasts of 3-year revenue growth and future FCF margin.

A model refitted on date R trains only on rows whose labels were known before
R (``label_date < R``) and predicts rows dated in [R, next refit). Settings are
the ones pre-registered in ``docs/results/fundamental_factors_protocol.md``;
there is deliberately no tuning.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from auto_researcher.features.fundamental_factors import GROWTH_FEATURES

logger = logging.getLogger(__name__)

HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=50,
                  l2_regularization=1.0, random_state=0, categorical_features="from_dtype")
GROWTH_CLIP = (-0.5, 1.0)
MARGIN_CLIP = (-0.5, 0.8)


def baseline_forecasts(X: pd.DataFrame) -> pd.DataFrame:
    """
    B1: sector median past 3-year growth; B2: half own past growth, half B1.
    M1: current FCF-to-firm margin; M2: half current, half sector median.
    """
    dates = X.index.get_level_values("date")
    sector_margin = X.groupby([dates, "sector"], observed=True)["fcff_margin"].transform("median")
    b1 = X["sector_median_growth"]
    b2 = (0.5 * X["rev_cagr_3y"] + 0.5 * b1).fillna(b1)
    m1 = X["fcff_margin"]
    m2 = (0.5 * m1 + 0.5 * sector_margin).fillna(sector_margin)
    return pd.DataFrame({
        "growth_b1": b1.clip(*GROWTH_CLIP), "growth_b2": b2.clip(*GROWTH_CLIP),
        "margin_m1": m1.clip(*MARGIN_CLIP), "margin_m2": m2.clip(*MARGIN_CLIP),
    }, index=X.index)


def walk_forward_forecasts(
    train: pd.DataFrame,
    predict_X: pd.DataFrame,
    refit_dates: list[pd.Timestamp],
    min_rows: int = 1000,
) -> pd.DataFrame:
    """
    ``train``: features plus ``y_growth``, ``y_margin``, ``label_date``, indexed
    by (date, symbol). ``predict_X``: features to score. Returns
    ``growth_ml``, ``margin_ml`` and ``model_date`` for rows of ``predict_X``
    dated on or after the first usable refit.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    refits = sorted(pd.Timestamp(d) for d in refit_dates)
    pdates = predict_X.index.get_level_values("date")
    out = []
    for i, refit in enumerate(refits):
        nxt = refits[i + 1] if i + 1 < len(refits) else pd.Timestamp.max
        target_rows = (pdates >= refit) & (pdates < nxt)
        if not target_rows.any():
            continue
        rows = train[(train["label_date"] < refit) & train["y_growth"].notna() & train["y_margin"].notna()]
        if len(rows) < min_rows:
            logger.info("refit %s skipped: %d training rows", refit.date(), len(rows))
            continue
        Xt = rows[GROWTH_FEATURES]
        Xp = predict_X.loc[target_rows, GROWTH_FEATURES]
        g = HistGradientBoostingRegressor(**HGB_PARAMS).fit(Xt, rows["y_growth"])
        m = HistGradientBoostingRegressor(**HGB_PARAMS).fit(Xt, rows["y_margin"])
        out.append(pd.DataFrame({
            "growth_ml": np.clip(g.predict(Xp), *GROWTH_CLIP),
            "margin_ml": np.clip(m.predict(Xp), *MARGIN_CLIP),
            "model_date": refit,
            "train_rows": len(rows),
        }, index=Xp.index))
        logger.info("refit %s: %d training rows, %d predictions", refit.date(), len(rows), len(Xp))
    if not out:
        return pd.DataFrame(columns=["growth_ml", "margin_ml", "model_date", "train_rows"])
    return pd.concat(out).sort_index()
