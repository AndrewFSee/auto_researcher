"""
Relational GNN ranker.

Why a GNN alongside the transformer ranker
------------------------------------------
The :class:`TransformerRanker` lets stocks influence each other's scores
on a given date via cross-sectional attention, but its attention is
dense: every stock attends to every other. A GNN uses a sparser,
semantically meaningful adjacency — e.g. "these two stocks are in the
same GICS sector" or "their returns have been tightly correlated over
the past 60 days." That sparsity is itself information: a cluster of
correlated names moving together should pull each other's scores more
than two unrelated names.

Graph construction
------------------
For each rebalance date we build a graph with:

* **Nodes** — the tickers with features on that date.
* **Correlation edges** — any pair of tickers whose trailing
  ``corr_window``-day return correlation exceeds ``corr_threshold``.
  The correlation is computed on the returns strictly before the date,
  so the adjacency itself is causal.
* **Sector edges** (optional) — if a ``ticker_sector`` map is supplied,
  tickers in the same sector are connected.

Model
-----
A 2-layer GraphSAGE-style aggregator:

1. Per-node MLP encoder ``(n_features,) → (d_model,)``.
2. For each layer: ``h_i ← σ(W_self h_i + W_neigh mean(h_j for j in N(i)))``.
3. Linear head → scalar score per node.

Trained with the same listwise ListNet loss as the transformer ranker, so
they can be blended cleanly in the IC-weighted ensemble.

Dependency model
----------------
``torch`` is a hard dependency. ``torch_geometric`` is NOT required —
we implement message passing by hand because the aggregation we need
(mean-neighbor) is ~10 lines of dense ops, and pulling in PyG wheels
on Windows is fiddly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


@dataclass
class GNNRankerConfig:
    """Hyperparameters for :class:`GNNRanker`.

    Attributes:
        d_model: Hidden dim for node embeddings.
        n_layers: Number of GraphSAGE aggregation layers.
        dropout: Dropout applied after each layer.
        n_epochs: Training epochs.
        lr: Adam learning rate.
        weight_decay: L2 regularization.
        corr_window: Trailing window (days) for the correlation adjacency.
        corr_threshold: Absolute correlation above which we emit an edge.
            0.5 is a common "strongly comoving" threshold.
        use_sector_edges: Add edges between tickers sharing a sector.
        min_stocks_per_date: Drop dates with too few tickers to train on.
        device: ``"cpu"`` / ``"cuda"`` / ``"auto"``.
        random_state: Seed.
    """

    d_model: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    n_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    corr_window: int = 60
    corr_threshold: float = 0.5
    use_sector_edges: bool = True
    min_stocks_per_date: int = 3
    device: str = "auto"
    random_state: int = 42


def _resolve_device(pref: str) -> "torch.device":
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


if HAS_TORCH:

    class _SAGELayer(nn.Module):
        """Mean-aggregator GraphSAGE layer on a dense adjacency matrix."""

        def __init__(self, in_dim: int, out_dim: int, dropout: float):
            super().__init__()
            self.w_self = nn.Linear(in_dim, out_dim)
            self.w_neigh = nn.Linear(in_dim, out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, h, adj_norm):
            # adj_norm: (n, n) with rows normalized so sum == 1 where the node
            # has any neighbors (and 0 for isolates). Mean-aggregation.
            neigh = adj_norm @ h
            out = self.w_self(h) + self.w_neigh(neigh)
            return self.dropout(F.relu(out))

    class _GNNNet(nn.Module):
        def __init__(self, n_features: int, cfg: "GNNRankerConfig"):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.layers = nn.ModuleList(
                [_SAGELayer(cfg.d_model, cfg.d_model, cfg.dropout)
                 for _ in range(cfg.n_layers)]
            )
            self.head = nn.Linear(cfg.d_model, 1)

        def forward(self, x, adj_norm):
            h = self.encoder(x)
            for layer in self.layers:
                h = layer(h, adj_norm)
            return self.head(h).squeeze(-1)


def _listnet_loss(scores, targets):
    log_p = F.log_softmax(scores, dim=0)
    q = F.softmax(targets, dim=0)
    return -(q * log_p).sum()


def _build_adjacency(
    tickers: list[str],
    date: pd.Timestamp,
    returns_df: Optional[pd.DataFrame],
    corr_window: int,
    corr_threshold: float,
    ticker_sector: Optional[Mapping[str, str]],
    use_sector_edges: bool,
) -> np.ndarray:
    """Binary adjacency matrix for the given date / ticker set.

    All edges are strictly causal: correlations are measured on returns
    before ``date``, and sector membership is assumed static.
    """
    n = len(tickers)
    adj = np.zeros((n, n), dtype=np.float32)

    # Correlation edges.
    if returns_df is not None:
        # Take the last ``corr_window`` rows strictly before `date`.
        past = returns_df.loc[returns_df.index < date].tail(corr_window)
        common = [t for t in tickers if t in past.columns]
        if common and len(past) >= max(10, corr_window // 4):
            corr = past[common].corr().fillna(0.0)
            idx_of = {t: i for i, t in enumerate(tickers)}
            for i, a in enumerate(common):
                for j, b in enumerate(common):
                    if i == j:
                        continue
                    if abs(corr.iat[i, j]) >= corr_threshold:
                        adj[idx_of[a], idx_of[b]] = 1.0

    # Sector edges.
    if use_sector_edges and ticker_sector:
        for i, a in enumerate(tickers):
            sa = ticker_sector.get(a)
            if not sa:
                continue
            for j, b in enumerate(tickers):
                if i == j:
                    continue
                if ticker_sector.get(b) == sa:
                    adj[i, j] = 1.0

    return adj


def _row_normalize(adj: np.ndarray) -> np.ndarray:
    """Row-normalize so each row sums to 1 (or 0 for isolates)."""
    deg = adj.sum(axis=1, keepdims=True)
    safe_deg = np.where(deg == 0, 1.0, deg)
    return adj / safe_deg


class GNNRanker:
    """GraphSAGE-style ranker, API-compatible with XGB / transformer models."""

    def __init__(
        self,
        config: Optional[GNNRankerConfig] = None,
        returns_df: Optional[pd.DataFrame] = None,
        ticker_sector: Optional[Mapping[str, str]] = None,
    ):
        """
        Args:
            config: Hyperparameters.
            returns_df: Daily returns ``DataFrame`` indexed by date with
                tickers as columns. Used to build per-date correlation
                adjacency. Supply an empty frame to disable correlation
                edges (sector edges still work).
            ticker_sector: Optional ``{ticker: sector}`` map for sector
                edges. Skipped if ``use_sector_edges=False``.
        """
        if not HAS_TORCH:
            raise ImportError(
                "torch is required for GNNRanker. Install with `pip install torch`."
            )
        self.config = config or GNNRankerConfig()
        self.returns_df = returns_df
        self.ticker_sector = dict(ticker_sector) if ticker_sector else None
        self.model: Optional[_GNNNet] = None
        self.feature_names: Optional[list[str]] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._device = _resolve_device(self.config.device)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "GNNRanker":
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must have MultiIndex (date, ticker)")

        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        self.feature_names = list(X.columns)
        self._feature_mean = X.values.mean(axis=0)
        self._feature_std = X.values.std(axis=0)
        self._feature_std = np.where(self._feature_std < 1e-8, 1.0, self._feature_std)

        groups = self._group_by_date(X, y)
        groups = [g for g in groups if g[0].shape[0] >= self.config.min_stocks_per_date]
        if not groups:
            raise ValueError(
                f"No dates with ≥ {self.config.min_stocks_per_date} stocks"
            )
        logger.info(
            "GNNRanker training: %d dates, %d features, device=%s",
            len(groups), len(self.feature_names), self._device,
        )

        self.model = _GNNNet(len(self.feature_names), self.config).to(self._device)
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        for epoch in range(self.config.n_epochs):
            self.model.train()
            order = np.random.permutation(len(groups))
            running = 0.0
            for gi in order:
                xg, adj, yg = groups[gi]
                opt.zero_grad()
                scores = self.model(xg, adj)
                loss = _listnet_loss(scores, yg)
                loss.backward()
                opt.step()
                running += float(loss.item())
            logger.debug(
                "epoch %d/%d  listnet=%.4f",
                epoch + 1, self.config.n_epochs, running / max(1, len(groups)),
            )
        return self

    @torch.no_grad() if HAS_TORCH else (lambda f: f)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        self.model.eval()
        out = np.zeros(len(X), dtype=np.float64)
        pos_of = pd.Series(np.arange(len(X)), index=X.index)
        for date, sub in X.groupby(level=0, sort=False):
            tickers = sub.index.get_level_values(1).tolist()
            x = self._to_tensor(sub.values)
            adj = self._adj_tensor(tickers, pd.Timestamp(date))
            scores = self.model(x, adj).cpu().numpy()
            out[pos_of.loc[sub.index].to_numpy()] = scores
        return out

    def predict_with_index(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.predict(X), index=X.index, name="prediction")

    def rank_cross_sectionally(self, X: pd.DataFrame) -> pd.Series:
        preds = self.predict_with_index(X)
        return preds.groupby(level=0).rank(ascending=False, method="first").rename("rank")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_fitted(self):
        if self.model is None:
            raise ValueError("GNNRanker has not been fit")

    def _to_tensor(self, values: np.ndarray) -> "torch.Tensor":
        z = (values - self._feature_mean) / self._feature_std
        return torch.tensor(z, dtype=torch.float32, device=self._device)

    def _adj_tensor(
        self, tickers: list[str], date: pd.Timestamp
    ) -> "torch.Tensor":
        adj = _build_adjacency(
            tickers, date,
            self.returns_df, self.config.corr_window, self.config.corr_threshold,
            self.ticker_sector, self.config.use_sector_edges,
        )
        adj_norm = _row_normalize(adj)
        return torch.tensor(adj_norm, dtype=torch.float32, device=self._device)

    def _group_by_date(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]:
        groups = []
        y_aligned = y.reindex(X.index)
        for date, sub in X.groupby(level=0, sort=False):
            yg = y_aligned.loc[sub.index].values
            mask = np.isfinite(yg)
            if mask.sum() < self.config.min_stocks_per_date:
                continue
            tickers = sub.index.get_level_values(1)[mask].tolist()
            xg = self._to_tensor(sub.values[mask])
            adj = self._adj_tensor(tickers, pd.Timestamp(date))
            yt = torch.tensor(yg[mask], dtype=torch.float32, device=self._device)
            groups.append((xg, adj, yt))
        return groups
