# Round 3: fundamental factors and an ML valuation gap

Generated 2026-10-01 by `scripts/fundamental_factor_research.py`, following the pre-registered [protocol](fundamental_factors_protocol.md) without changes. Signs are applied before scoring, so a positive IC always means the factor worked as published. Deflated probabilities assume 20 trials.

Run log: the first complete run crashed while writing this report (a naming bug in the report code, after all results were computed). The fix touched only report formatting; the rerun reproduced every logged statistic exactly.

Universe: 420 current S&P 500 members (from their add dates; no financials or captive-finance industrials). Survivorship bias remains.

## Verdicts (21-day horizon)

* `gross_profitability` (expected sign +): dev IC +0.0040 (t +0.34), holdout IC +0.0101 (t +0.60): **not supported**
* `accruals` (expected sign −): dev IC +0.0200 (t +2.34), holdout IC +0.0198 (t +1.47): **supported**
* `asset_growth` (expected sign −): dev IC -0.0040 (t -0.36), holdout IC -0.0194 (t -1.22): **not supported**
* `net_issuance` (expected sign −): dev IC +0.0144 (t +1.44), holdout IC +0.0083 (t +0.68): **not supported**
* `fcf_yield` (expected sign +): dev IC +0.0143 (t +1.15), holdout IC -0.0036 (t -0.19): **not supported**
* `ebit_ev` (expected sign +): dev IC -0.0122 (t -0.84), holdout IC +0.0002 (t +0.01): **not supported**
* `composite` (expected sign +): dev IC +0.0107 (t +0.87), holdout IC +0.0046 (t +0.30): **not supported**

* ML growth forecast useful (beats B2 on revenue-growth rank correlation, paired t ≥ 2): **yes** (difference +0.170, t +2.18).
* ML valuation gap adds value over the naive gap and the reverse DCF: **no** (dev IC difference vs naive +0.0084, t +0.80; vs reverse DCF +0.0155, t +1.18).

## Part A: six factors (189 dates, median 179 stocks)

### 21-day horizon (primary), development (through 2022-12-31)

| Metric | `gross_profitability` | `accruals` | `asset_growth` | `net_issuance` | `fcf_yield` | `ebit_ev` | `composite` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Periods | 141 | 141 | 141 | 141 | 141 | 141 | 141 |
| Mean IC | +0.0040 | +0.0200 | -0.0040 | +0.0144 | +0.0143 | -0.0122 | +0.0107 |
| IC t (NW) | +0.34 | +2.34 | -0.36 | +1.44 | +1.15 | -0.84 | +0.87 |
| IC > 0 | 50% | 60% | 48% | 56% | 50% | 50% | 51% |
| Top-50 net vs EW / yr | -1.06% | +1.87% | +0.75% | +1.34% | +1.98% | +0.02% | +1.37% |
| IR vs EW | -0.26 | +0.47 | +0.20 | +0.38 | +0.43 | +0.00 | +0.33 |
| P(IR > 0), deflated | 0.00 | 0.39 | 0.11 | 0.28 | 0.33 | 0.03 | 0.22 |
| Pctl vs random top-50 | 10% | 92% | 98% | 94% | 99% | 23% | 97% |

### 21-day horizon (primary), holdout (from 2023-01-01)

| Metric | `gross_profitability` | `accruals` | `asset_growth` | `net_issuance` | `fcf_yield` | `ebit_ev` | `composite` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Periods | 44 | 44 | 44 | 44 | 44 | 44 | 44 |
| Mean IC | +0.0101 | +0.0198 | -0.0194 | +0.0083 | -0.0036 | +0.0002 | +0.0046 |
| IC t (NW) | +0.60 | +1.47 | -1.22 | +0.68 | -0.19 | +0.01 | +0.30 |
| IC > 0 | 57% | 57% | 43% | 48% | 43% | 45% | 48% |
| Top-50 net vs EW / yr | -4.09% | +6.14% | -0.25% | +2.03% | +2.73% | +2.24% | +2.21% |
| IR vs EW | -0.78 | +1.00 | -0.05 | +0.41 | +0.37 | +0.33 | +0.34 |
| P(IR > 0), deflated | 0.00 | 0.50 | 0.02 | 0.13 | 0.12 | 0.10 | 0.10 |
| Pctl vs random top-50 | 2% | 98% | 39% | 87% | 96% | 74% | 87% |

### 63-day horizon (secondary), development (through 2022-12-31)

| Metric | `gross_profitability` | `accruals` | `asset_growth` | `net_issuance` | `fcf_yield` | `ebit_ev` | `composite` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Periods | 139 | 139 | 139 | 139 | 139 | 139 | 139 |
| Mean IC | -0.0030 | +0.0295 | -0.0032 | +0.0262 | +0.0251 | -0.0134 | +0.0182 |
| IC t (NW) | -0.17 | +2.42 | -0.19 | +1.88 | +1.36 | -0.66 | +0.96 |
| IC > 0 | 50% | 55% | 45% | 55% | 55% | 45% | 52% |
| Top-50 net vs EW / yr | -1.12% | +1.90% | +0.76% | +1.28% | +1.94% | -0.05% | +1.24% |
| IR vs EW | -0.27 | +0.48 | +0.20 | +0.37 | +0.42 | -0.01 | +0.30 |
| P(IR > 0), deflated | 0.00 | 0.39 | 0.11 | 0.26 | 0.32 | 0.03 | 0.19 |
| Pctl vs random top-50 | 9% | 93% | 98% | 94% | 99% | 19% | 97% |

### 63-day horizon (secondary), holdout (from 2023-01-01)

| Metric | `gross_profitability` | `accruals` | `asset_growth` | `net_issuance` | `fcf_yield` | `ebit_ev` | `composite` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Periods | 42 | 42 | 42 | 42 | 42 | 42 | 42 |
| Mean IC | +0.0091 | +0.0297 | -0.0198 | +0.0009 | -0.0035 | -0.0047 | +0.0041 |
| IC t (NW) | +0.44 | +1.96 | -1.14 | +0.06 | -0.14 | -0.19 | +0.20 |
| IC > 0 | 57% | 67% | 43% | 55% | 40% | 48% | 52% |
| Top-50 net vs EW / yr | -3.28% | +5.14% | -0.40% | +1.62% | +2.25% | +1.75% | +1.62% |
| IR vs EW | -0.64 | +0.90 | -0.08 | +0.35 | +0.32 | +0.27 | +0.27 |
| P(IR > 0), deflated | 0.00 | 0.41 | 0.02 | 0.10 | 0.10 | 0.08 | 0.08 |
| Pctl vs random top-50 | 8% | 97% | 42% | 84% | 93% | 66% | 84% |

### Mean IC by year (21-day, development)

| Year | `gross_profitability` | `accruals` | `asset_growth` | `net_issuance` | `fcf_yield` | `ebit_ev` | `composite` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2011 | +0.028 | +0.054 | +0.052 | +0.071 | +0.056 | +0.052 | +0.102 |
| 2012 | -0.061 | -0.011 | -0.010 | -0.038 | -0.019 | -0.073 | -0.049 |
| 2013 | -0.034 | +0.059 | -0.009 | +0.045 | +0.072 | +0.024 | +0.045 |
| 2014 | +0.040 | +0.029 | +0.001 | +0.013 | +0.052 | +0.031 | +0.047 |
| 2015 | +0.037 | +0.007 | -0.074 | -0.043 | -0.085 | -0.112 | -0.075 |
| 2016 | -0.069 | -0.005 | +0.039 | +0.029 | +0.024 | +0.003 | +0.005 |
| 2017 | +0.009 | +0.026 | -0.033 | -0.001 | +0.049 | -0.008 | -0.001 |
| 2018 | +0.055 | +0.016 | -0.052 | -0.003 | -0.011 | -0.046 | -0.013 |
| 2019 | +0.019 | +0.010 | -0.016 | +0.009 | -0.027 | -0.047 | -0.018 |
| 2020 | +0.046 | +0.031 | -0.034 | +0.012 | -0.010 | -0.040 | -0.006 |
| 2021 | +0.029 | +0.046 | +0.034 | +0.076 | +0.042 | +0.038 | +0.080 |
| 2022 | -0.045 | -0.013 | +0.067 | +0.018 | +0.039 | +0.048 | +0.034 |

## Part B: growth forecast

Out-of-sample forecasts at quarter-ends from 2016 whose 3-year outcome is known. Rank correlation is per date, averaged.

| Target | Forecast | Dates | Rows | Mean rank corr. | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| growth | `growth_ml` | 27 | 8,723 | +0.354 | 0.077 |
| growth | `growth_b1` | 27 | 8,723 | +0.121 | 0.082 |
| growth | `growth_b2` | 27 | 8,723 | +0.184 | 0.086 |
| margin | `margin_ml` | 27 | 8,709 | +0.749 | 0.070 |
| margin | `margin_m1` | 27 | 8,709 | +0.748 | 0.074 |
| margin | `margin_m2` | 27 | 8,709 | +0.720 | 0.078 |

Paired difference ML − B2 (growth): +0.170 (t +2.18); ML − M2 (margin): +0.029 (t +1.76).

## Part B: valuation gap (117 dates, median 181 stocks; positive FCF only)

* `valuation_gap_ml`: intrinsic EV from the ML growth and margin forecasts / EV
* `valuation_gap_naive`: intrinsic EV from the naive forecasts (B2 growth, M2 margin) / EV
* `reverse_dcf`: reverse DCF implied growth (scored as FCF-to-firm / EV, same ranking)
* `composite`: Part A composite on the same rows (comparison, not a Part B hypothesis)

### 21-day horizon (primary), development

| Metric | `valuation_gap_ml` | `valuation_gap_naive` | `reverse_dcf` | `composite` |
| --- | ---: | ---: | ---: | ---: |
| Periods | 69 | 69 | 69 | 69 |
| Mean IC | +0.0291 | +0.0207 | +0.0136 | +0.0156 |
| IC t (NW) | +1.59 | +0.97 | +0.64 | +0.80 |
| IC > 0 | 61% | 58% | 48% | 54% |
| Top-50 net vs EW / yr | +2.86% | +1.98% | +2.78% | +1.56% |
| IR vs EW | +0.56 | +0.33 | +0.51 | +0.32 |
| P(IR > 0), deflated | 0.29 | 0.13 | 0.25 | 0.13 |
| Pctl vs random top-50 | 94% | 72% | 97% | 83% |

### 21-day horizon (primary), holdout

| Metric | `valuation_gap_ml` | `valuation_gap_naive` | `reverse_dcf` | `composite` |
| --- | ---: | ---: | ---: | ---: |
| Periods | 44 | 44 | 44 | 44 |
| Mean IC | +0.0253 | +0.0279 | +0.0035 | +0.0057 |
| IC t (NW) | +1.15 | +1.39 | +0.17 | +0.36 |
| IC > 0 | 52% | 57% | 50% | 45% |
| Top-50 net vs EW / yr | +7.09% | +2.14% | +3.26% | +2.68% |
| IR vs EW | +0.96 | +0.33 | +0.48 | +0.42 |
| P(IR > 0), deflated | 0.47 | 0.10 | 0.16 | 0.14 |
| Pctl vs random top-50 | 100% | 69% | 95% | 90% |

### 63-day horizon (secondary), development

| Metric | `valuation_gap_ml` | `valuation_gap_naive` | `reverse_dcf` | `composite` |
| --- | ---: | ---: | ---: | ---: |
| Periods | 67 | 67 | 67 | 67 |
| Mean IC | +0.0489 | +0.0451 | +0.0329 | +0.0402 |
| IC t (NW) | +1.83 | +1.46 | +1.00 | +1.45 |
| IC > 0 | 63% | 60% | 54% | 51% |
| Top-50 net vs EW / yr | +2.98% | +2.33% | +2.95% | +1.97% |
| IR vs EW | +0.58 | +0.38 | +0.54 | +0.40 |
| P(IR > 0), deflated | 0.30 | 0.16 | 0.26 | 0.17 |
| Pctl vs random top-50 | 94% | 80% | 97% | 90% |

### 63-day horizon (secondary), holdout

| Metric | `valuation_gap_ml` | `valuation_gap_naive` | `reverse_dcf` | `composite` |
| --- | ---: | ---: | ---: | ---: |
| Periods | 42 | 42 | 42 | 42 |
| Mean IC | +0.0364 | +0.0398 | +0.0092 | +0.0074 |
| IC t (NW) | +1.53 | +1.65 | +0.35 | +0.35 |
| IC > 0 | 60% | 67% | 52% | 55% |
| Top-50 net vs EW / yr | +5.45% | +1.02% | +2.27% | +1.88% |
| IR vs EW | +0.78 | +0.16 | +0.35 | +0.32 |
| P(IR > 0), deflated | 0.33 | 0.05 | 0.11 | 0.09 |
| Pctl vs random top-50 | 99% | 50% | 90% | 84% |

### Part B verdicts

* `valuation_gap_ml`: dev IC +0.0291 (t +1.59), holdout IC +0.0253 (t +1.15): **not supported**
* `valuation_gap_naive`: dev IC +0.0207 (t +0.97), holdout IC +0.0279 (t +1.39): **not supported**
* `reverse_dcf`: dev IC +0.0136 (t +0.64), holdout IC +0.0035 (t +0.17): **not supported**
* `composite`: dev IC +0.0156 (t +0.80), holdout IC +0.0057 (t +0.36): **not supported**

