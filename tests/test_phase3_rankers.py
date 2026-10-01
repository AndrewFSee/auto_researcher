"""Phase 3 tests: TransformerRanker + GNNRanker + ICWeightedEnsemble.

Each test uses a synthetic cross-sectional fixture where the target is a
known, noisy linear combination of the features — any working ranker
should recover IC > 0 on a held-out fold. Tiny net sizes keep CI fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from auto_researcher.models.transformer_ranker import (
    TransformerRanker,
    TransformerRankerConfig,
)
from auto_researcher.models.gnn_ranker import (
    GNNRanker,
    GNNRankerConfig,
    _build_adjacency,
    _row_normalize,
)
from auto_researcher.models.ensemble_ranker import (
    ICWeightedEnsemble,
    _mean_cross_sectional_ic,
    _zscore_per_date,
)


# ---------------------------------------------------------------------------
# Fixture: synthetic cross-sectional panel with a learnable signal
# ---------------------------------------------------------------------------

def _make_panel(
    n_dates: int = 60,
    n_tickers: int = 15,
    n_features: int = 8,
    noise: float = 0.6,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    X = rng.normal(size=(len(idx), n_features))
    # Only the first 3 features carry signal; the rest are noise.
    true_beta = np.array([1.0, -0.6, 0.4] + [0.0] * (n_features - 3))
    y = X @ true_beta + rng.normal(scale=noise, size=len(idx))

    X_df = pd.DataFrame(
        X, index=idx, columns=[f"f{i}" for i in range(n_features)]
    )
    y_s = pd.Series(y, index=idx, name="target")
    return X_df, y_s


def _make_returns(tickers: list[str], dates: pd.DatetimeIndex, seed: int = 1):
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=0.02, size=(len(dates), len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _ic_on(model, X, y) -> float:
    preds = model.predict_with_index(X)
    return _mean_cross_sectional_ic(preds, y)


# ---------------------------------------------------------------------------
# TransformerRanker
# ---------------------------------------------------------------------------

def test_transformer_fits_and_recovers_positive_ic() -> None:
    X, y = _make_panel(n_dates=40, n_tickers=12, n_features=6, noise=0.4)
    split = pd.Timestamp("2023-02-20")
    X_tr = X.loc[X.index.get_level_values("date") < split]
    y_tr = y.loc[X_tr.index]
    X_te = X.loc[X.index.get_level_values("date") >= split]
    y_te = y.loc[X_te.index]

    model = TransformerRanker(
        TransformerRankerConfig(
            d_model=16, n_heads=2, n_layers=1, n_epochs=30,
            lr=5e-3, batch_dates=4, random_state=0,
        )
    )
    model.fit(X_tr, y_tr)

    preds = model.predict_with_index(X_te)
    assert preds.index.equals(X_te.index)
    ic = _mean_cross_sectional_ic(preds, y_te)
    assert ic > 0.05, f"expected positive OOS IC, got {ic:.3f}"


def test_transformer_requires_multiindex() -> None:
    X = pd.DataFrame(np.random.randn(10, 3))
    y = pd.Series(np.random.randn(10))
    with pytest.raises(ValueError, match="MultiIndex"):
        TransformerRanker().fit(X, y)


def test_transformer_is_deterministic_for_same_seed() -> None:
    X, y = _make_panel(n_dates=20, n_tickers=8, n_features=5)
    cfg = TransformerRankerConfig(
        d_model=16, n_heads=2, n_layers=1, n_epochs=5,
        batch_dates=4, random_state=123,
    )
    p1 = TransformerRanker(cfg).fit(X, y).predict(X)
    p2 = TransformerRanker(cfg).fit(X, y).predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-4)


# ---------------------------------------------------------------------------
# GNNRanker
# ---------------------------------------------------------------------------

def test_build_adjacency_correlation_edges() -> None:
    tickers = ["A", "B", "C"]
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    rng = np.random.default_rng(3)
    base = rng.normal(size=len(dates))
    ret = pd.DataFrame({
        "A": base + rng.normal(scale=0.01, size=len(dates)),
        "B": base + rng.normal(scale=0.01, size=len(dates)),  # highly correlated w/ A
        "C": rng.normal(size=len(dates)),                      # independent
    }, index=dates)

    adj = _build_adjacency(
        tickers, pd.Timestamp("2023-06-01"),
        returns_df=ret, corr_window=60, corr_threshold=0.5,
        ticker_sector=None, use_sector_edges=False,
    )
    # A ↔ B must be connected; C should be isolated.
    assert adj[0, 1] == 1.0 and adj[1, 0] == 1.0
    assert adj[0, 2] == 0.0 and adj[2, 0] == 0.0


def test_build_adjacency_sector_edges() -> None:
    tickers = ["AAPL", "MSFT", "XOM"]
    adj = _build_adjacency(
        tickers, pd.Timestamp("2023-06-01"),
        returns_df=None, corr_window=60, corr_threshold=0.5,
        ticker_sector={"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"},
        use_sector_edges=True,
    )
    assert adj[0, 1] == 1.0 and adj[1, 0] == 1.0  # Tech peers connected
    assert adj[0, 2] == 0.0 and adj[2, 0] == 0.0  # cross-sector not


def test_row_normalize_handles_isolates() -> None:
    adj = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=float)
    norm = _row_normalize(adj)
    np.testing.assert_allclose(norm[0], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(norm[1], [1.0, 0.0, 0.0])
    # Isolate row (no neighbors): all zeros.
    np.testing.assert_allclose(norm[2], [0.0, 0.0, 0.0])


def test_gnn_fits_and_recovers_positive_ic() -> None:
    X, y = _make_panel(n_dates=40, n_tickers=12, n_features=6, noise=0.4)
    tickers = X.index.get_level_values("ticker").unique().tolist()
    dates = X.index.get_level_values("date").unique()
    ret = _make_returns(tickers, dates)

    split = pd.Timestamp("2023-02-20")
    X_tr = X.loc[X.index.get_level_values("date") < split]
    y_tr = y.loc[X_tr.index]
    X_te = X.loc[X.index.get_level_values("date") >= split]
    y_te = y.loc[X_te.index]

    model = GNNRanker(
        GNNRankerConfig(
            d_model=16, n_layers=2, n_epochs=30, lr=5e-3, random_state=0,
            corr_window=30, corr_threshold=0.3,
        ),
        returns_df=ret,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict_with_index(X_te)
    assert preds.index.equals(X_te.index)
    ic = _mean_cross_sectional_ic(preds, y_te)
    assert ic > 0.05, f"expected positive OOS IC, got {ic:.3f}"


# ---------------------------------------------------------------------------
# ICWeightedEnsemble
# ---------------------------------------------------------------------------

def test_ensemble_drops_models_with_nonpositive_ic() -> None:
    X, y = _make_panel(n_dates=30, n_tickers=8, n_features=5, noise=0.4)
    good = TransformerRanker(
        TransformerRankerConfig(
            d_model=16, n_heads=2, n_layers=1, n_epochs=20,
            batch_dates=4, random_state=0,
        )
    ).fit(X, y)

    # Deterministic anti-signal: predict the negation of y → IC ≈ -1.
    # (Random-noise "IC==0" can drift positive on small panels.)
    class _AntiModel:
        def __init__(self, y): self._y = y
        def predict_with_index(self, X):
            return -self._y.reindex(X.index)

    ens = ICWeightedEnsemble({"good": good, "anti": _AntiModel(y)})
    w = ens.fit_weights(X, y)
    # Good model must survive with nonzero weight; anti-signal must be dropped.
    assert w.weights.get("good", 0.0) > 0
    assert "anti" in w.dropped
    # predict_with_index should work and preserve the index.
    preds = ens.predict_with_index(X)
    assert preds.index.equals(X.index)


def test_ensemble_fallback_when_all_models_drop() -> None:
    X, y = _make_panel(n_dates=10, n_tickers=6, n_features=4)

    class _ConstModel:
        def predict_with_index(self, X):
            return pd.Series(1.0, index=X.index)  # zero-IC every date

    ens = ICWeightedEnsemble({"a": _ConstModel(), "b": _ConstModel()})
    w = ens.fit_weights(X, y)
    # No positive-IC survivors → equal-weight fallback.
    assert set(w.weights) == {"a", "b"}
    assert abs(sum(w.weights.values()) - 1.0) < 1e-6


def test_zscore_per_date_handles_constant_date() -> None:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2023-01-02"), "A"),
         (pd.Timestamp("2023-01-02"), "B"),
         (pd.Timestamp("2023-01-03"), "A"),
         (pd.Timestamp("2023-01-03"), "B")],
        names=["date", "ticker"],
    )
    preds = pd.Series([1.0, 1.0, 2.0, 4.0], index=idx)
    z = _zscore_per_date(preds)
    # Day 1: both equal → z == 0. Day 2: proper z-score.
    assert z.loc[(pd.Timestamp("2023-01-02"), "A")] == 0.0
    assert abs(z.loc[(pd.Timestamp("2023-01-03"), "A")] + 0.7071) < 1e-3
