# Customer momentum in S&P 500 suppliers

Generated 2026-11-17 18:37 by `scripts/customer_momentum.py`, under the rules pre-registered in [customer_momentum_protocol.md](customer_momentum_protocol.md). Deviations from the protocol: none.

**Result: not supported.** The tercile long-short net return averaged +0.13% a month (Newey-West t = 0.33, 104 months); the decision rule needs t ≥ 2.0.

## Data

* 10-Ks filed 2018 to October 2026: 4594; with a link: 606 (912 supplier-customer links in all, 127 of them from XBRL tags only).
* Suppliers with a link at some point: 119; distinct customers: 92.
* Price sources: tradier 155, yfinance 25, tiingo 21, none 1. Without prices (left out): METRY.
* Source check (sentiq vs yfinance, 157 tickers in both): monthly returns differing by more than 0.5 points in 125 of 13636 ticker-months; tickers over 10% of their months: FLR (3/21), MCD (18/106), SWK (11/106) (sentiq used).
* Suppliers in the sort each month: min 1, median 60, max 66; distinct customers behind them: median 39. Supplier-months whose signal includes Walmart: 29%. Months skipped (fewer than 30 suppliers): 1.
* Mean monthly turnover of the tercile portfolio: 2.56 (sum of absolute weight changes; costs 10 bps per unit, 0.26% a month).

## Primary test

| Portfolio | Mean / month | NW t | Months | Ann. mean | Sharpe | Deflated P(SR>0) | Max drawdown | Months > 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Terciles, net (primary) | +0.13% | 0.33 | 104 | +1.5% | 0.09 | 0.22 | -26.0% | 52% |
| Terciles, gross | +0.38% | 1.01 | 104 | +4.6% | 0.29 | 0.41 | -22.9% | 55% |
| Quintiles, net | -0.04% | -0.07 | 104 | -0.5% | -0.02 | 0.13 | -44.7% | 50% |
| Without Walmart's links, net | +0.20% | 0.40 | 103 | +2.4% | 0.12 | 0.24 | -37.7% | 52% |

Only the first row decides; the others are reported to explain it.

## Secondary measures

| Measure | Mean | NW t | Months |
| --- | ---: | ---: | ---: |
| Long leg minus SPY (gross) | +0.20% | 0.67 | 104 |
| Short leg minus SPY (gross) | -0.18% | -0.60 | 104 |
| Fama-MacBeth slope on customer momentum, controlling for own month-t return | +0.075 | 2.40 | 104 |
| Fama-MacBeth slope on customer momentum, no control | +0.057 | 1.64 | 104 |
| Rank correlation of customer momentum with next month's return | +0.024 | 1.15 | 104 |
| Terciles net, holding months 2018-02 to 2023-12 | +0.33% | 0.65 | 70 |
| Terciles net, holding months 2024-01 to 2026-10 | -0.30% | -0.58 | 34 |

The Fama-MacBeth slope is the next-month return per unit of customer return (0.10 means a supplier whose customers rose 10% earns 1 point more the next month).

## By year (terciles, net)

| Year | Months | Mean / month | Suppliers (median) |
| --- | ---: | ---: | ---: |
| 2018 | 10 | -1.35% | 44 |
| 2019 | 12 | -0.06% | 60 |
| 2020 | 12 | +1.82% | 64 |
| 2021 | 12 | +0.20% | 64 |
| 2022 | 12 | +0.48% | 60 |
| 2023 | 12 | +0.64% | 62 |
| 2024 | 12 | -0.30% | 59 |
| 2025 | 12 | +0.05% | 55 |
| 2026 | 10 | -0.71% | 51 |
