"""
Tests for the purged walk-forward splitter and evaluation harness.

The most important test here is the leakage canary: on random-walk prices
nothing is predictable, so any positive IC under the purged protocol would
mean the harness leaks. The legacy split (``purge=False``) must, by contrast,
show a large spurious IC; that is the mechanism behind the retracted
"IC = +0.145" headline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from auto_researcher.backtest.baselines import FeatureScoreModel, LinearICModel
from auto_researcher.backtest.walk_forward import (
    WalkForwardConfig,
    _Window,
    forward_returns,
    make_training_target,
    overlap_lag,
    random_selection_null,
    run_walk_forward,
    simulate_portfolio,
)
from auto_researcher.validation.splits import (
    label_overlap_days,
    last_trainable_position,
    purged_walk_forward_splits,
)

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def random_walk_prices(n_days: int = 800, n_stocks: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.02, size=(n_days, n_stocks))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        index=dates,
        columns=[f"S{i:02d}" for i in range(n_stocks)],
    )
    prices["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
    return prices


def persistent_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Slow-moving, stock-specific features: a stock's feature vector barely moves
    from one day to the next, so a flexible model can recognize "the same stock
    a few days ago" and copy its (leaked) label.
    """
    stock = prices.drop(columns="SPY")
    daily = stock.pct_change()
    frames = {
        "mom_120": stock.pct_change(120),
        "mom_250": stock.pct_change(250),
        "vol_120": daily.rolling(120).std(),
    }
    panel = pd.concat({k: v.stack() for k, v in frames.items()}, axis=1)
    panel.index = panel.index.set_names(["date", "ticker"])
    return panel


def alpha_prices(n_days: int = 900, n_stocks: int = 40, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices whose drift depends on a persistent per-stock alpha, plus a noisy feature of it."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    alpha = rng.normal(size=n_stocks)
    rets = 0.0008 * alpha + rng.normal(0.0, 0.012, size=(n_days, n_stocks))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    prices["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
    obs = alpha[None, :] + rng.normal(0.0, 0.3, size=(n_days, n_stocks))
    feats = pd.DataFrame(obs, index=dates, columns=tickers).stack().rename("alpha_obs").to_frame()
    feats["noise"] = rng.normal(size=len(feats))
    feats.index = feats.index.set_names(["date", "ticker"])
    return prices, feats


FAST = dict(
    horizon=21,
    rebalance_every=21,
    train_window=252,
    min_train_dates=126,
    top_k=8,
    n_random_paths=50,
    min_names=10,
)


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


class TestPurgedSplits:
    calendar = pd.bdate_range("2020-01-01", periods=400)

    @pytest.mark.parametrize("horizon,lag,embargo", [(21, 0, 0), (21, 1, 0), (63, 1, 5), (5, 2, 3)])
    def test_no_training_label_overlaps_test_date(self, horizon, lag, embargo):
        tests = self.calendar[150::20]
        splits = list(
            purged_walk_forward_splits(
                self.calendar, tests, horizon=horizon, execution_lag=lag, embargo=embargo
            )
        )
        assert splits
        pos = {d: i for i, d in enumerate(self.calendar)}
        for sp in splits:
            assert label_overlap_days(sp.train_dates, sp.test_date, self.calendar, horizon, lag) == 0
            # The purge is tight: the last trainable date is exactly h + lag + embargo back.
            assert pos[sp.train_dates[-1]] == pos[sp.test_date] - horizon - lag - embargo

    def test_legacy_mode_leaks_by_construction(self):
        tests = self.calendar[150::20]
        for sp in purged_walk_forward_splits(self.calendar, tests, horizon=21, purge=False):
            # Latest training label covers 20 of the 21 test-label days.
            assert label_overlap_days(sp.train_dates, sp.test_date, self.calendar, 21, 0) == 20

    def test_rolling_window_length(self):
        tests = self.calendar[300:301]
        (sp,) = purged_walk_forward_splits(self.calendar, tests, horizon=21, train_window=100)
        assert len(sp.train_dates) == 100

    def test_min_train_dates_capped_by_window(self):
        tests = self.calendar[300:301]
        splits = list(
            purged_walk_forward_splits(
                self.calendar, tests, horizon=21, train_window=50, min_train_dates=252
            )
        )
        assert len(splits) == 1 and len(splits[0].train_dates) == 50

    def test_skips_dates_without_enough_history(self):
        tests = self.calendar[[10, 300]]
        splits = list(purged_walk_forward_splits(self.calendar, tests, horizon=21, min_train_dates=100))
        assert [s.test_date for s in splits] == [self.calendar[300]]

    def test_unknown_test_date_raises(self):
        with pytest.raises(KeyError):
            list(purged_walk_forward_splits(self.calendar, [pd.Timestamp("1999-01-01")], horizon=5))

    def test_last_trainable_position(self):
        assert last_trainable_position(100, horizon=21, execution_lag=1, embargo=2) == 76
        with pytest.raises(ValueError):
            last_trainable_position(100, horizon=-1)


# ---------------------------------------------------------------------------
# Labels and accounting
# ---------------------------------------------------------------------------


class TestLabelsAndAccounting:
    def test_forward_returns_timing(self):
        prices = pd.DataFrame({"A": [1.0, 2.0, 4.0, 8.0, 16.0]}, index=pd.bdate_range("2020-01-01", periods=5))
        f1 = forward_returns(prices, horizon=1, execution_lag=0)["A"]
        assert f1.iloc[:4].tolist() == [1.0, 1.0, 1.0, 1.0] and np.isnan(f1.iloc[4])
        f2 = forward_returns(prices, horizon=2, execution_lag=1)["A"]
        assert f2.iloc[0] == pytest.approx(3.0)  # 8 / 2 - 1
        assert f2.iloc[2:].isna().all()

    def test_training_targets(self):
        idx = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-01")], ["A", "B", "C", "D"]],
                                         names=["date", "ticker"])
        fwd = pd.Series([0.04, 0.01, -0.02, 0.03], index=idx)
        rank = make_training_target(fwd, "rank")
        assert rank.tolist() == pytest.approx([0.5, 0.0, -0.25, 0.25])
        assert make_training_target(fwd, "demeaned").mean() == pytest.approx(0.0)
        vol = pd.Series([0.02] * 4, index=idx)
        assert make_training_target(fwd, "vol_norm", vol).iloc[0] == pytest.approx(2.0)
        with pytest.raises(ValueError):
            make_training_target(fwd, "vol_norm")

    def test_overlap_lag(self):
        assert overlap_lag(21, 21) == 0
        assert overlap_lag(63, 21) == 2
        assert overlap_lag(21, 5) == 4

    def test_portfolio_simulation_exact(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame(
            {"A": [100, 110, 121, 121, 121], "B": [100, 100, 100, 50, 50], "C": [100] * 5},
            index=dates, dtype=float,
        )
        windows = [_Window(dates[0], 0, 2), _Window(dates[2], 2, 4)]
        weights = {dates[0]: pd.Series({"A": 0.5, "B": 0.5}), dates[2]: pd.Series({"C": 1.0})}
        gross, net, turnover = simulate_portfolio(prices, windows, weights, cost_bps=10)

        assert gross.tolist() == pytest.approx([0.05, 1.105 / 1.05 - 1, 0.0, 0.0])
        # Initial build trades 100% of capital; the switch sells A+B and buys C (200%).
        assert turnover.tolist() == pytest.approx([0.5, 1.0])
        assert net.iloc[0] == pytest.approx(1.05 * (1 - 0.001) - 1)
        assert net.iloc[2] == pytest.approx(-0.002)
        # B's crash on day 3 happens after it was sold, so it must not appear.
        assert gross.iloc[2] == 0.0

    def test_random_null_shape_and_degenerate_case(self):
        dates = pd.bdate_range("2020-01-01", periods=60)
        prices = pd.DataFrame(
            np.tile(np.linspace(100, 130, 60)[:, None], (1, 6)), index=dates, columns=list("ABCDEF")
        )
        windows = [_Window(dates[i], i, i + 20) for i in (0, 20)]
        universes = {w.rebalance_date: list("ABCDEF") for w in windows}
        null = random_selection_null(prices, windows, universes, k=2, n_paths=30)
        assert null.shape == (30,)
        assert np.allclose(null, null[0])  # identical stocks -> identical paths


# ---------------------------------------------------------------------------
# Harness behaviour
# ---------------------------------------------------------------------------


def _knn():
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3))


class TestLeakageCanary:
    """Random walk => no signal. Purged IC must be ~0; the legacy split must not be."""

    # Calibrated over seeds 0-2: leaky IC +0.38..+0.46 (t 11-17), purged |IC| < 0.05.
    @pytest.fixture(scope="class")
    def data(self):
        prices = random_walk_prices(n_days=900)
        return prices, persistent_features(prices)

    def test_purged_protocol_finds_nothing_in_noise(self, data):
        prices, feats = data
        cfg = WalkForwardConfig(**{**FAST, "n_random_paths": 0})
        s = run_walk_forward(feats, prices, _knn, cfg, name="purged").summary()
        assert s["n_periods"] >= 20
        assert abs(s["ic_mean"]) < 0.1
        assert abs(s["ic_t_nw"]) < 3.0

    def test_legacy_split_manufactures_signal_from_noise(self, data):
        prices, feats = data
        cfg = WalkForwardConfig(**{**FAST, "n_random_paths": 0, "purge": False, "execution_lag": 0})
        s = run_walk_forward(feats, prices, _knn, cfg, name="leaky").summary()
        assert s["ic_mean"] > 0.25
        assert s["ic_t_nw"] > 5.0


class TestSignalRecovery:
    @pytest.fixture(scope="class")
    def data(self):
        return alpha_prices()

    def test_oracle_feature_has_positive_ic_and_beats_equal_weight(self, data):
        prices, feats = data
        s = run_walk_forward(
            feats, prices, lambda: FeatureScoreModel("alpha_obs"), WalkForwardConfig(**FAST)
        ).summary()
        assert s["ic_mean"] > 0.1
        assert s["net_active_vs_equal_weight_ann"] > 0
        assert s["gross_sharpe_percentile_vs_random"] > 0.9

    def test_learned_linear_model_recovers_signal(self, data):
        prices, feats = data
        s = run_walk_forward(feats, prices, LinearICModel, WalkForwardConfig(**FAST)).summary()
        assert s["ic_mean"] > 0.1

    def test_sign_flip_reverses_ic(self, data):
        prices, feats = data
        cfg = WalkForwardConfig(**{**FAST, "n_random_paths": 0})
        up = run_walk_forward(feats, prices, lambda: FeatureScoreModel("alpha_obs"), cfg).summary()
        down = run_walk_forward(
            feats, prices, lambda: FeatureScoreModel("alpha_obs", sign=-1), cfg
        ).summary()
        assert down["ic_mean"] == pytest.approx(-up["ic_mean"], abs=1e-9)


class TestHarnessOutputs:
    @pytest.fixture(scope="class")
    def result(self):
        prices, feats = alpha_prices(n_days=700)
        return run_walk_forward(feats, prices, LinearICModel, WalkForwardConfig(**FAST), name="lin")

    def test_summary_has_headline_fields(self, result):
        s = result.summary()
        for key in (
            "ic_mean", "ic_t_nw", "spread_mean", "gross_sharpe", "net_sharpe",
            "net_max_drawdown", "equal_weight_sharpe", "benchmark_sharpe",
            "net_ir_vs_equal_weight", "net_ir_vs_equal_weight_deflated_prob",
            "gross_sharpe_percentile_vs_random", "avg_turnover",
        ):
            assert key in s and np.isfinite(s[key]), key

    def test_daily_returns_are_contiguous_and_costs_reduce_returns(self, result):
        daily = result.daily_returns
        assert daily.index.is_monotonic_increasing and daily.index.is_unique
        assert (daily["net"] <= daily["gross"] + 1e-12).all()
        assert result.summary()["net_cagr"] < result.summary()["gross_cagr"]

    def test_predictions_only_on_test_dates(self, result):
        pred_dates = set(result.predictions.index.get_level_values("date"))
        assert pred_dates == set(result.n_train_rows.index)
        assert set(result.ic.index) <= pred_dates
        # One portfolio holding window per prediction date, each rebalance_every days long.
        assert len(result.daily_returns) == len(pred_dates) * result.config.rebalance_every

    def test_ic_by_year(self, result):
        table = result.ic_by_year()
        assert table["n"].sum() == len(result.ic)

    def test_rejects_flat_index(self):
        prices = random_walk_prices(n_days=300, n_stocks=12)
        flat = persistent_features(prices).reset_index(drop=True)
        with pytest.raises(ValueError, match="indexed by"):
            run_walk_forward(flat, prices, _knn)

    def test_config_validation(self):
        with pytest.raises(ValueError):
            WalkForwardConfig(horizon=0)
        with pytest.raises(ValueError):
            WalkForwardConfig(top_k=0)
