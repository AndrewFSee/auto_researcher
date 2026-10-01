# Feature research: does the ML ranker gain an edge?

Generated 2026-09-30 by `scripts/feature_research.py`.

Pre-registered protocol (see the script docstring): five candidates per universe, selection by mean IC on test dates through 2022-12-31, then one holdout run from 2023-01-01. Purged walk-forward, 21-day labels, next-close execution, 504-day rolling training window, 10 bps costs. Deflated probabilities assume 18 configurations were tried. Every candidate uses the same (date, ticker) rows, so they differ only in features, model and target.

Candidates:

* `base_xgb`: Audit model: price features, XGBoost
* `ext_xgb`: + published factors, XGBoost
* `ext_xgb_sn`: + factors, XGBoost, sector-neutral target
* `ext_linear`: + factors, linear IC-weighted
* `ext_linear_sn`: + factors, linear IC-weighted, sector-neutral target
* `momentum_12_1`: 12-1 momentum, no model (baseline)

Universes use current constituents (survivorship-biased). In the broad universe each stock enters on its S&P 500 add date, which removes the bias from future additions but not from companies that later left the index.

## Narrow universe (103 stocks, top-10 portfolios)

### Development period (test dates through 2022-12-31)

| Metric | `base_xgb` | `ext_xgb` | `ext_xgb_sn` | `ext_linear` | `ext_linear_sn` | `momentum_12_1` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Periods | 71 | 71 | 71 | 71 | 71 | 71 |
| Mean IC | -0.0148 | -0.0146 | -0.0266 | -0.0081 | -0.0035 | +0.0034 |
| IC t-stat (NW) | -0.80 | -0.60 | -1.32 | -0.31 | -0.13 | +0.11 |
| IC > 0 | 44% | 46% | 41% | 45% | 56% | 52% |
| Top-k net Sharpe | 0.64 | 0.46 | 0.58 | 0.46 | 0.52 | 0.68 |
| Equal-weight Sharpe | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 |
| Net active vs EW / yr | -0.78% | -4.60% | -1.98% | -3.42% | -4.00% | +1.97% |
| IR vs EW | -0.07 | -0.34 | -0.17 | -0.21 | -0.31 | +0.12 |
| P(IR > 0), deflated | 0.02 | 0.00 | 0.01 | 0.01 | 0.00 | 0.06 |
| Pctl vs random top-k | 26% | 1% | 13% | 1% | 6% | 26% |
| Turnover / rebalance | 71% | 70% | 70% | 57% | 49% | 28% |

Selected by development-period mean IC: **`ext_linear_sn`**.

### Holdout (test dates from 2023-01-01, run once)

| Metric | `ext_linear_sn` | `base_xgb` | `momentum_12_1` |
| --- | ---: | ---: | ---: |
| Periods | 44 | 44 | 44 |
| Mean IC | +0.0103 | -0.0071 | +0.0050 |
| IC t-stat (NW) | +0.31 | -0.31 | +0.14 |
| IC > 0 | 64% | 57% | 52% |
| Top-k net Sharpe | 1.18 | 1.19 | 1.27 |
| Equal-weight Sharpe | 1.48 | 1.48 | 1.48 |
| Net active vs EW / yr | +18.13% | +9.92% | +17.86% |
| IR vs EW | +0.71 | +0.53 | +0.74 |
| P(IR > 0), deflated | 0.31 | 0.20 | 0.33 |
| Pctl vs random top-k | 48% | 55% | 62% |
| Turnover / rebalance | 44% | 62% | 31% |

### Single-factor ICs, development period (diagnostic only)

| Factor | Mean IC | t (NW) |
| --- | ---: | ---: |
| `beta_252` | +0.0223 | +0.62 |
| `vol_63` | +0.0092 | +0.29 |
| `abn_volume` | +0.0048 | +0.33 |
| `seasonal` | +0.0048 | +0.19 |
| `mom_12_1` | +0.0034 | +0.11 |
| `max_ret_21` | +0.0008 | +0.03 |
| `mom_12_1_in_sector` | -0.0097 | -0.51 |
| `sector_mom_6_1` | -0.0112 | -0.40 |
| `dollar_volume_trend` | -0.0170 | -1.07 |
| `mom_6_1` | -0.0196 | -0.66 |
| `ret_21_in_sector` | -0.0223 | -1.29 |
| `high_52w` | -0.0353 | -1.12 |
| `ret_21` | -0.0597 | -2.21 |

## Broad universe (500 stocks, top-50 portfolios)

### Development period (test dates through 2022-12-31)

| Metric | `base_xgb` | `ext_xgb` | `ext_xgb_sn` | `ext_linear` | `ext_linear_sn` | `momentum_12_1` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Periods | 71 | 71 | 71 | 71 | 71 | 71 |
| Mean IC | -0.0076 | -0.0164 | -0.0111 | +0.0003 | -0.0001 | -0.0119 |
| IC t-stat (NW) | -0.43 | -0.78 | -0.61 | +0.01 | -0.00 | -0.43 |
| IC > 0 | 42% | 46% | 48% | 51% | 46% | 45% |
| Top-k net Sharpe | 0.58 | 0.37 | 0.56 | 0.59 | 0.73 | 0.54 |
| Equal-weight Sharpe | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 |
| Net active vs EW / yr | -0.56% | -5.64% | -0.83% | +1.11% | +2.60% | -2.51% |
| IR vs EW | -0.06 | -0.55 | -0.08 | +0.09 | +0.23 | -0.21 |
| P(IR > 0), deflated | 0.02 | 0.00 | 0.02 | 0.05 | 0.10 | 0.01 |
| Pctl vs random top-k | 12% | 0% | 7% | 11% | 77% | 1% |
| Turnover / rebalance | 67% | 63% | 64% | 55% | 50% | 32% |

Selected by development-period mean IC: **`ext_linear`**.

### Holdout (test dates from 2023-01-01, run once)

| Metric | `ext_linear` | `base_xgb` | `momentum_12_1` |
| --- | ---: | ---: | ---: |
| Periods | 44 | 44 | 44 |
| Mean IC | -0.0272 | -0.0114 | +0.0098 |
| IC t-stat (NW) | -0.92 | -0.70 | +0.36 |
| IC > 0 | 43% | 45% | 52% |
| Top-k net Sharpe | 0.56 | 0.97 | 1.12 |
| Equal-weight Sharpe | 1.08 | 1.08 | 1.08 |
| Net active vs EW / yr | -1.52% | +3.12% | +10.54% |
| IR vs EW | -0.09 | +0.28 | +0.63 |
| P(IR > 0), deflated | 0.02 | 0.10 | 0.26 |
| Pctl vs random top-k | 0% | 50% | 78% |
| Turnover / rebalance | 53% | 56% | 31% |

### Single-factor ICs, development period (diagnostic only)

| Factor | Mean IC | t (NW) |
| --- | ---: | ---: |
| `beta_252` | +0.0090 | +0.26 |
| `vol_63` | +0.0053 | +0.18 |
| `abn_volume` | -0.0034 | -0.34 |
| `max_ret_21` | -0.0057 | -0.28 |
| `mom_12_1_in_sector` | -0.0102 | -0.47 |
| `dollar_volume_trend` | -0.0117 | -1.02 |
| `mom_12_1` | -0.0119 | -0.43 |
| `ret_21_in_sector` | -0.0140 | -0.78 |
| `sector_mom_6_1` | -0.0154 | -0.85 |
| `high_52w` | -0.0223 | -0.75 |
| `seasonal` | -0.0240 | -1.13 |
| `mom_6_1` | -0.0247 | -0.98 |
| `ret_21` | -0.0413 | -1.72 |
