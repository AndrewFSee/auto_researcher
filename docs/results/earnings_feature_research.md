# Earnings-surprise features: round 2 of the feature research

Generated 2026-09-30 by `scripts/earnings_feature_research.py`.

Pre-registered protocol as in [feature_research.md](feature_research.md): fixed candidates, selection by mean IC on test dates through 2022-12-31, one holdout run from 2023-01-01. Deflated probabilities assume 30 configurations (audit + both rounds).

Earnings inputs are point-in-time: time-series SUE from quarterly EPS history (no analyst estimates), announcement dates inferred from SEC 8-K/10-Q filings and usable two trading days after filing. EPS histories are a current vintage, so rare restatements are a residual look-ahead risk.

Candidates:

* `ext_linear`: Round-1 reference: price + 13 factors, linear
* `earn_linear`: + earnings features, linear
* `earn_linear_sn`: + earnings features, linear, sector-neutral target
* `earn_xgb`: + earnings features, XGBoost
* `earn_only_linear`: Earnings features only, linear
* `sue_only`: SUE alone, no model (baseline)

## Narrow universe (103 stocks, top-10 portfolios)

Rows with a live (non-stale) surprise: 98%.

### Development period (test dates through 2022-12-31)

| Metric | `ext_linear` | `earn_linear` | `earn_linear_sn` | `earn_xgb` | `earn_only_linear` | `sue_only` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Periods | 71 | 71 | 71 | 71 | 71 | 71 |
| Mean IC | -0.0081 | -0.0147 | -0.0084 | -0.0232 | -0.0389 | +0.0151 |
| IC t-stat (NW) | -0.31 | -0.55 | -0.31 | -0.98 | -2.48 | +0.79 |
| IC > 0 | 45% | 41% | 55% | 44% | 35% | 55% |
| Top-k net Sharpe | 0.46 | 0.42 | 0.39 | 0.48 | 0.52 | 0.74 |
| Equal-weight Sharpe | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 |
| Net active vs EW / yr | -3.42% | -4.65% | -7.39% | -3.40% | -5.32% | -0.32% |
| IR vs EW | -0.21 | -0.29 | -0.57 | -0.24 | -0.55 | -0.03 |
| P(IR > 0), deflated | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| Pctl vs random top-k | 1% | 1% | 0% | 3% | 7% | 41% |
| Turnover / rebalance | 57% | 57% | 51% | 67% | 58% | 34% |

Selected by development-period mean IC (excluding the reference): **`earn_linear_sn`**.

### Holdout (test dates from 2023-01-01, run once)

| Metric | `earn_linear_sn` | `ext_linear` | `sue_only` |
| --- | ---: | ---: | ---: |
| Periods | 44 | 44 | 44 |
| Mean IC | +0.0089 | +0.0317 | +0.0073 |
| IC t-stat (NW) | +0.28 | +0.89 | +0.32 |
| IC > 0 | 64% | 55% | 59% |
| Top-k net Sharpe | 1.07 | 0.83 | 1.52 |
| Equal-weight Sharpe | 1.48 | 1.48 | 1.48 |
| Net active vs EW / yr | +14.19% | +7.00% | +6.78% |
| IR vs EW | +0.56 | +0.27 | +0.65 |
| P(IR > 0), deflated | 0.16 | 0.06 | 0.20 |
| Pctl vs random top-k | 32% | 7% | 88% |
| Turnover / rebalance | 46% | 45% | 29% |

### Single-feature ICs, development period (diagnostic only)

| Feature | Mean IC | t (NW) |
| --- | ---: | ---: |
| `sue` | +0.0151 | +0.79 |
| `earnings_ann_return` | -0.0063 | -0.48 |
| `eps_beat_streak` | +0.0048 | +0.22 |
| `days_since_ann` | +0.0082 | +0.65 |

## Broad universe (500 stocks, top-50 portfolios)

Rows with a live (non-stale) surprise: 99%.

### Development period (test dates through 2022-12-31)

| Metric | `ext_linear` | `earn_linear` | `earn_linear_sn` | `earn_xgb` | `earn_only_linear` | `sue_only` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Periods | 71 | 71 | 71 | 71 | 71 | 71 |
| Mean IC | +0.0003 | -0.0027 | -0.0030 | -0.0182 | -0.0231 | +0.0000 |
| IC t-stat (NW) | +0.01 | -0.11 | -0.11 | -0.89 | -2.08 | +0.00 |
| IC > 0 | 51% | 49% | 44% | 46% | 44% | 55% |
| Top-k net Sharpe | 0.59 | 0.55 | 0.65 | 0.46 | 0.48 | 0.63 |
| Equal-weight Sharpe | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 |
| Net active vs EW / yr | +1.11% | -0.27% | +0.86% | -3.30% | -4.65% | -1.89% |
| IR vs EW | +0.09 | -0.02 | +0.08 | -0.32 | -0.98 | -0.33 |
| P(IR > 0), deflated | 0.03 | 0.02 | 0.03 | 0.00 | 0.00 | 0.00 |
| Pctl vs random top-k | 11% | 3% | 36% | 0% | 0% | 20% |
| Turnover / rebalance | 55% | 55% | 50% | 63% | 52% | 30% |

Selected by development-period mean IC (excluding the reference): **`earn_linear`**.

### Holdout (test dates from 2023-01-01, run once)

| Metric | `earn_linear` | `ext_linear` | `sue_only` |
| --- | ---: | ---: | ---: |
| Periods | 44 | 44 | 44 |
| Mean IC | -0.0268 | -0.0272 | +0.0143 |
| IC t-stat (NW) | -0.94 | -0.92 | +1.19 |
| IC > 0 | 43% | 43% | 59% |
| Top-k net Sharpe | 0.57 | 0.56 | 1.29 |
| Equal-weight Sharpe | 1.08 | 1.08 | 1.08 |
| Net active vs EW / yr | -1.51% | -1.52% | +3.77% |
| IR vs EW | -0.09 | -0.09 | +0.61 |
| P(IR > 0), deflated | 0.01 | 0.01 | 0.19 |
| Pctl vs random top-k | 0% | 0% | 98% |
| Turnover / rebalance | 52% | 53% | 27% |

### Single-feature ICs, development period (diagnostic only)

| Feature | Mean IC | t (NW) |
| --- | ---: | ---: |
| `sue` | +0.0000 | +0.00 |
| `earnings_ann_return` | -0.0063 | -0.68 |
| `eps_beat_streak` | -0.0072 | -0.45 |
| `days_since_ann` | +0.0010 | +0.10 |

## Event-time check (broad universe, 2014 onward)

22,387 announcements, 50 quarters. Mean quarterly Spearman IC of SUE against market-adjusted returns from the availability date; the reaction runs from the close before the filing to availability.

| Window | Mean quarterly IC | t | Top - bottom decile |
| --- | ---: | ---: | ---: |
| `reaction` | +0.0987 | +11.16 | +1.81% |
| `drift_20d` | +0.0353 | +2.50 | +0.43% |
| `drift_40d` | +0.0318 | +2.64 | +0.57% |
| `drift_60d` | +0.0221 | +1.86 | +1.14% |
