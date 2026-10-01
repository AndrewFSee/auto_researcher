"""
Cross-sectional Transformer ranker.

Motivation
----------
XGBoost ranks stocks by making a local decision for each (date, ticker) row
using only that row's features. Attention-based rankers can do something
XGBoost structurally cannot: let stocks influence each other's scores
within the same date. "AAPL looks great in isolation, but every other
tech stock is also top-quintile today — discount AAPL because the theme
is crowded" is the kind of adjustment that falls out of cross-sectional
attention.

Architecture
------------
* Per-stock MLP encoder ``(n_features,) → (d_model,)``. Our feature matrix
  is already a summary of time-series (rolling means, momentum, etc.), so
  we skip the per-stock temporal encoder that the plan mentions and treat
  each row as a fixed-size token.
* ``n_layers`` self-attention blocks across the stocks present on a given
  date. Each date is a separate "sequence" — stocks attend to each other,
  not across time.
* Linear head → scalar score per stock.

Training objective
------------------
Listwise ListNet — cross-entropy between ``softmax(predicted_scores)`` and
``softmax(target_returns)`` within each date. Works well for rank-only
targets (we only care that the best stocks get the highest scores, not
that we hit absolute return magnitudes).

Dependency model
----------------
``torch`` is a hard dependency for this module — ImportError if absent.
The rest of the repo stays torch-free, so callers opt in by importing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

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
class TransformerRankerConfig:
    """Hyperparameters for :class:`TransformerRanker`.

    Attributes:
        d_model: Width of the per-stock embedding and attention channels.
        n_heads: Attention heads in each block. Must divide ``d_model``.
        n_layers: Number of self-attention blocks applied across stocks.
        dropout: Dropout probability inside the attention blocks.
        n_epochs: Training epochs (one pass per date per epoch).
        lr: Adam learning rate.
        weight_decay: L2 regularization.
        batch_dates: Dates processed per gradient step. Keep small —
            the cross-sectional attention already aggregates across tickers.
        min_stocks_per_date: Dates with fewer stocks are dropped from
            training (listwise loss is ill-defined with n=1).
        device: ``"cpu"`` / ``"cuda"`` / ``"auto"``.
        random_state: Seed for torch / numpy determinism.
    """

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    n_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_dates: int = 4
    min_stocks_per_date: int = 3
    device: str = "auto"
    random_state: int = 42


def _resolve_device(pref: str) -> "torch.device":
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


if HAS_TORCH:

    class _CrossSectionalBlock(nn.Module):
        """One self-attention block across stocks on a single date."""

        def __init__(self, d_model: int, n_heads: int, dropout: float):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )

        def forward(self, x):  # x: (n_stocks, d_model)
            h = x.unsqueeze(0)                       # (1, n_stocks, d_model)
            attn_out, _ = self.attn(h, h, h)
            h = self.norm1(h + attn_out)
            ff_out = self.ff(h)
            h = self.norm2(h + ff_out)
            return h.squeeze(0)

    class _TransformerNet(nn.Module):
        def __init__(self, n_features: int, cfg: "TransformerRankerConfig"):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.blocks = nn.ModuleList(
                [_CrossSectionalBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
                 for _ in range(cfg.n_layers)]
            )
            self.head = nn.Linear(cfg.d_model, 1)

        def forward(self, x):  # x: (n_stocks, n_features)
            h = self.encoder(x)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h).squeeze(-1)          # (n_stocks,)


def _listnet_loss(scores, targets):
    """Top-1 ListNet: cross-entropy between score-softmax and target-softmax.

    Reference: Cao et al., "Learning to Rank: From Pairwise Approach to
    Listwise Approach" (ICML 2007). Works well when the target is a
    noisy score; the softmax damps the tails so one outlier day doesn't
    dominate the gradient.
    """
    log_p = F.log_softmax(scores, dim=0)
    q = F.softmax(targets, dim=0)
    return -(q * log_p).sum()


class TransformerRanker:
    """Cross-sectional transformer ranker, API-compatible with XGB models.

    Usage mirrors :class:`XGBRegressionModel` so it can slot into the
    existing ensemble harness without special-casing::

        model = TransformerRanker(TransformerRankerConfig(n_epochs=10))
        model.fit(X_train, y_train)               # MultiIndex (date, ticker)
        preds = model.predict_with_index(X_test)  # pd.Series, same index
    """

    def __init__(self, config: Optional[TransformerRankerConfig] = None):
        if not HAS_TORCH:
            raise ImportError(
                "torch is required for TransformerRanker. "
                "Install with `pip install torch`."
            )
        self.config = config or TransformerRankerConfig()
        self.model: Optional[_TransformerNet] = None
        self.feature_names: Optional[list[str]] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._device = _resolve_device(self.config.device)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "TransformerRanker":
        """Train with listwise loss across dates.

        Args:
            X: Feature matrix with MultiIndex ``(date, ticker)``.
            y: Target aligned to ``X`` (e.g., vol-normalized forward return).
            X_val, y_val: Optional eval set used only for early-epoch
                diagnostics — no early stopping, since small nets overfit
                their own val sets too readily.
        """
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must have MultiIndex (date, ticker)")

        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        self.feature_names = list(X.columns)
        # Feature standardization — attention layers train faster and
        # more stably on ~unit-variance inputs.
        self._feature_mean = X.values.mean(axis=0)
        self._feature_std = X.values.std(axis=0)
        self._feature_std = np.where(self._feature_std < 1e-8, 1.0, self._feature_std)

        # Group rows by date — each group is one listwise example.
        groups = self._group_by_date(X, y)
        groups = [g for g in groups if g[0].shape[0] >= self.config.min_stocks_per_date]
        if not groups:
            raise ValueError(
                f"No dates with ≥ {self.config.min_stocks_per_date} stocks"
            )
        logger.info(
            "TransformerRanker training: %d dates, %d features, device=%s",
            len(groups), len(self.feature_names), self._device,
        )

        self.model = _TransformerNet(len(self.feature_names), self.config).to(
            self._device
        )
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        for epoch in range(self.config.n_epochs):
            self.model.train()
            order = np.random.permutation(len(groups))
            running_loss = 0.0
            for start in range(0, len(order), self.config.batch_dates):
                batch_idx = order[start:start + self.config.batch_dates]
                opt.zero_grad()
                loss = torch.zeros(1, device=self._device)
                for gi in batch_idx:
                    xg, yg = groups[gi]
                    scores = self.model(xg)
                    loss = loss + _listnet_loss(scores, yg)
                (loss / max(1, len(batch_idx))).backward()
                opt.step()
                running_loss += float(loss.item())
            logger.debug(
                "epoch %d/%d  listnet=%.4f",
                epoch + 1, self.config.n_epochs, running_loss / max(1, len(groups)),
            )

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @torch.no_grad() if HAS_TORCH else (lambda f: f)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        self.model.eval()
        out = np.zeros(len(X), dtype=np.float64)
        pos_of = pd.Series(np.arange(len(X)), index=X.index)
        for date, sub in X.groupby(level=0, sort=False):
            x = self._to_tensor(sub.values)
            scores = self.model(x).cpu().numpy()
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
            raise ValueError("TransformerRanker has not been fit")

    def _to_tensor(self, values: np.ndarray) -> "torch.Tensor":
        z = (values - self._feature_mean) / self._feature_std
        return torch.tensor(z, dtype=torch.float32, device=self._device)

    def _group_by_date(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[tuple["torch.Tensor", "torch.Tensor"]]:
        groups = []
        y_aligned = y.reindex(X.index)
        for date, sub in X.groupby(level=0, sort=False):
            yg = y_aligned.loc[sub.index].values
            mask = np.isfinite(yg)
            if mask.sum() < self.config.min_stocks_per_date:
                continue
            xg = self._to_tensor(sub.values[mask])
            yt = torch.tensor(yg[mask], dtype=torch.float32, device=self._device)
            groups.append((xg, yt))
        return groups
