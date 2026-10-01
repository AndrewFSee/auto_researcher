# Purged walk-forward evaluation

Generated 2026-09-29 by `scripts/ml_walkforward_backtest.py --offline --leak-ablation --window-sweep 126,252,504`.

* **Universe.** 103 stocks (local price cache), 2015-01-02 to 2026-01-15. The list is today's constituents, so every backtest here is survivorship-biased: judge models against the equal-weight portfolio of the same names, not only against SPY.
* **Protocol.** Rebalance every 21 trading days; signal at the close of t, trade at the close of t+1; 21-day labels; training rows purged so no label overlaps the test date; rolling 504-day training window. Long-only top-10 equal weight, 10 bps per unit traded.
* **Statistics.** IC t-stats use Newey-West with lag ceil(horizon / rebalance) - 1. The deflated probability assumes 6 configuration(s) were tried. The random-selection percentile compares the gross Sharpe with 500 random top-10 portfolios drawn from the same cross-sections.

## Models and baselines

| Metric | `xgb` | `screening` | `linear_ic` | `momentum` | `reversal` | `low_vol` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rebalance periods | 105 | 105 | 105 | 105 | 105 | 105 |
| Mean IC (Spearman) | -0.0147 | -0.0253 | -0.0038 | +0.0113 | -0.0058 | -0.0184 |
| IC t-stat (Newey-West) | -0.85 | -1.80 | -0.17 | +0.52 | -0.30 | -0.95 |
| Periods with IC > 0 | 41% | 47% | 48% | 54% | 47% | 45% |
| Top-bottom quintile fwd return / period | -0.27% | -0.38% | +0.32% | +0.09% | +0.06% | -0.95% |
| Top-k CAGR (gross) | +19.4% | +18.4% | +21.0% | +27.3% | +19.2% | +13.6% |
| Top-k CAGR (net of costs) | +17.5% | +16.1% | +19.5% | +26.5% | +16.7% | +12.6% |
| Top-k Sharpe (net) | 0.75 | 0.80 | 0.74 | 1.03 | 0.75 | 0.78 |
| Top-k max drawdown (net) | -42.4% | -29.9% | -41.2% | -31.3% | -31.8% | -38.6% |
| Avg one-way turnover / rebalance | 68% | 81% | 52% | 26% | 87% | 38% |
| Equal-weight universe Sharpe | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| SPY Sharpe | 0.84 | 0.84 | 0.84 | 0.84 | 0.84 | 0.84 |
| Net active return vs equal-weight / yr | +0.57% | -1.70% | +3.44% | +8.04% | -0.48% | -5.69% |
| Info ratio vs equal-weight | +0.04 | -0.17 | +0.19 | +0.49 | -0.04 | -0.62 |
| P(true IR vs EW > 0), deflated | 0.12 | 0.04 | 0.23 | 0.55 | 0.08 | 0.00 |
| Sharpe percentile vs random top-k | 23% | 45% | 17% | 84% | 29% | 29% |

Model descriptions:

* `xgb`: XGBoost regression on the per-date rank of the forward return
* `screening`: Live screening recipe (auto_researcher.screening): feature pruning, recency weights, pseudo-Huber XGBoost, 126-day window
* `linear_ic`: Features weighted by their training-window IC (no tuning)
* `momentum`: 12-month residual momentum, no model
* `reversal`: 5-day reversal, no model
* `low_vol`: Low idiosyncratic volatility, no model

## Leak ablation (xgb)

| Metric | `legacy_leaky` | `purged_same_close` | `purged_next_close` |
| --- | ---: | ---: | ---: |
| Rebalance periods | 105 | 105 | 105 |
| Mean IC (Spearman) | +0.1116 | -0.0110 | -0.0147 |
| IC t-stat (Newey-West) | +7.35 | -0.66 | -0.85 |
| Periods with IC > 0 | 80% | 48% | 41% |
| Top-bottom quintile fwd return / period | +2.22% | -0.14% | -0.27% |
| Top-k CAGR (gross) | +48.2% | +21.9% | +19.4% |
| Top-k CAGR (net of costs) | +45.8% | +20.0% | +17.5% |
| Top-k Sharpe (net) | 1.65 | 0.84 | 0.75 |
| Top-k max drawdown (net) | -30.2% | -46.9% | -42.4% |
| Avg one-way turnover / rebalance | 67% | 67% | 68% |
| Equal-weight universe Sharpe | 0.99 | 0.99 | 1.00 |
| SPY Sharpe | 0.83 | 0.83 | 0.84 |
| Net active return vs equal-weight / yr | +21.93% | +2.59% | +0.57% |
| Info ratio vs equal-weight | +1.66 | +0.20 | +0.04 |
| P(true IR vs EW > 0), deflated | 1.00 | 0.24 | 0.12 |
| Sharpe percentile vs random top-k | 100% | 50% | 23% |

## Training-window sweep (xgb)

| Metric | `w126_leaky` | `w126_purged` | `w252_leaky` | `w252_purged` | `w504_leaky` | `w504_purged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rebalance periods | 105 | 105 | 105 | 105 | 105 | 105 |
| Mean IC (Spearman) | +0.2309 | -0.0177 | +0.1578 | -0.0205 | +0.1116 | -0.0147 |
| IC t-stat (Newey-West) | +16.30 | -1.11 | +10.66 | -1.25 | +7.35 | -0.85 |
| Periods with IC > 0 | 94% | 41% | 85% | 44% | 80% | 41% |
| Top-bottom quintile fwd return / period | +4.89% | -0.19% | +3.42% | -0.17% | +2.22% | -0.27% |
| Top-k CAGR (gross) | +88.0% | +21.4% | +60.2% | +19.7% | +48.2% | +19.4% |
| Top-k CAGR (net of costs) | +84.8% | +19.4% | +57.8% | +17.8% | +45.8% | +17.5% |
| Top-k Sharpe (net) | 2.74 | 0.82 | 2.02 | 0.79 | 1.65 | 0.75 |
| Top-k max drawdown (net) | -25.1% | -38.3% | -30.3% | -35.7% | -30.2% | -42.4% |
| Avg one-way turnover / rebalance | 70% | 71% | 65% | 67% | 67% | 68% |
| Equal-weight universe Sharpe | 0.99 | 1.00 | 0.99 | 1.00 | 0.99 | 1.00 |
| SPY Sharpe | 0.83 | 0.84 | 0.83 | 0.84 | 0.83 | 0.84 |
| Net active return vs equal-weight / yr | +45.38% | +2.02% | +29.63% | +0.47% | +21.93% | +0.57% |
| Info ratio vs equal-weight | +3.42 | +0.15 | +2.20 | +0.04 | +1.66 | +0.04 |
| P(true IR vs EW > 0), deflated | 1.00 | 0.19 | 1.00 | 0.12 | 1.00 | 0.12 |
| Sharpe percentile vs random top-k | 100% | 45% | 100% | 36% | 100% | 23% |

## Mean IC by year

| Year | `xgb` | `screening` | `linear_ic` | `momentum` | `reversal` | `low_vol` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | -0.010 | -0.012 | +0.005 | +0.026 | -0.045 | -0.008 |
| 2018 | -0.055 | -0.088 | -0.016 | +0.035 | +0.007 | +0.015 |
| 2019 | +0.005 | -0.033 | -0.013 | -0.072 | +0.116 | -0.056 |
| 2020 | +0.022 | -0.010 | +0.019 | +0.068 | -0.055 | -0.031 |
| 2021 | -0.009 | -0.058 | -0.028 | -0.111 | -0.093 | -0.005 |
| 2022 | -0.077 | -0.018 | -0.044 | +0.018 | +0.057 | -0.021 |
| 2023 | +0.028 | +0.028 | +0.037 | +0.107 | +0.036 | -0.053 |
| 2024 | -0.022 | +0.008 | -0.035 | +0.028 | +0.019 | +0.020 |
| 2025 | -0.015 | -0.044 | +0.045 | +0.004 | -0.110 | -0.025 |
