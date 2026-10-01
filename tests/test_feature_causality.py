"""
Causality tests for the price-feature pipelines.

A feature dated t may only use prices up to the close of t. These tests
perturb every price strictly after a cut-off date and require that all
features on or before the cut-off are unchanged. Any look-ahead (a rolling
window centered on t, a full-sample normalization, a negative shift) fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_researcher.config import FeatureConfig
from auto_researcher.features.enhanced import EnhancedFeatureConfig, compute_all_enhanced_features
from auto_researcher.features.feature_pipeline import build_feature_matrix


def _prices(n_days: int = 420, n_stocks: int = 12, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rets = rng.normal(0.0004, 0.018, size=(n_days, n_stocks + 1))
    cols = [f"T{i:02d}" for i in range(n_stocks)] + ["SPY"]
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols)


def _perturb_after(prices: pd.DataFrame, cut: pd.Timestamp, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = prices.copy()
    after = out.index > cut
    out.loc[after] = out.loc[after].to_numpy() * rng.uniform(0.5, 1.5, size=(after.sum(), out.shape[1]))
    return out


@pytest.mark.parametrize("cross_sec_norm", [True, False])
def test_enhanced_features_are_causal(cross_sec_norm: bool) -> None:
    prices = _prices()
    cut = prices.index[330]
    cfg = EnhancedFeatureConfig(use_sector_ohe=False, use_cross_sec_norm=cross_sec_norm)

    base = compute_all_enhanced_features(prices, benchmark="SPY", config=cfg)
    moved = compute_all_enhanced_features(_perturb_after(prices, cut), benchmark="SPY", config=cfg)

    # Sanity: the perturbation does change later features...
    assert not base.loc[base.index > cut].equals(moved.loc[moved.index > cut])
    # ...but nothing on or before the cut-off.
    pd.testing.assert_frame_equal(base.loc[:cut], moved.loc[:cut])


def test_library_feature_matrix_is_causal() -> None:
    prices = _prices()
    cut = prices.index[330]
    cfg = FeatureConfig(include_fundamentals=False, include_sentiment=False)

    base = build_feature_matrix(prices, cfg)
    moved = build_feature_matrix(_perturb_after(prices, cut), cfg)

    pd.testing.assert_frame_equal(base.loc[:cut], moved.loc[:cut])


def test_alpha_factors_are_causal() -> None:
    from auto_researcher.features.alpha_factors import compute_alpha_factors

    prices = _prices(n_days=1400)
    rng = np.random.default_rng(5)
    volume = pd.DataFrame(rng.lognormal(14, 0.4, size=prices.shape), index=prices.index,
                          columns=prices.columns)
    sectors = pd.Series({t: ("A" if i % 2 else "B") for i, t in enumerate(prices.columns)})
    cut = prices.index[1300]

    base = compute_alpha_factors(prices, volume, benchmark="SPY", sectors=sectors)
    moved_volume = volume.copy()
    moved_volume.loc[moved_volume.index > cut] *= 3.0
    moved = compute_alpha_factors(_perturb_after(prices, cut), moved_volume, benchmark="SPY",
                                  sectors=sectors)

    early = base.index.get_level_values("date") <= cut
    late_m = moved.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(base[early], moved[late_m])
    # Every factor is populated by the end of the sample.
    last = base.xs(prices.index[-1], level="date")
    assert last.notna().all().all(), last.columns[last.isna().any()].tolist()
