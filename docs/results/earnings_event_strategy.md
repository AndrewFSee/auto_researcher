# Event-driven earnings drift strategy

Generated 2026-10-01 by `scripts/earnings_event_strategy.py`.

Rules fixed before running: rank each surprise against the previous year's announcements, long the top 20% and short the bottom 20%, enter one day after the surprise is usable (two trading days after the SEC filing), hold 40 trading days (20/60 as sensitivity), equal weight per leg, 10 bps per unit traded. Universe: current S&P 500 members from their index add dates. Deflated Sharpe probabilities assume 6 variants.

Costs are charged on daily changes in equal-weight targets, which re-weights every open position when one enters or exits; a buy-and-hold-per-position implementation would pay roughly half. Gross results are shown for that reason.

## Long-short, gross of costs

| Strategy | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ts_sue_20d` (from 2014-08-05) | +1.7% | 15.1% | 0.11 | +0.38 | -46.1% |
| `ts_sue_40d` (from 2014-08-05) | +1.0% | 9.9% | 0.10 | +0.34 | -29.7% |
| `ts_sue_60d` (from 2014-08-05) | +1.7% | 8.6% | 0.20 | +0.70 | -24.3% |
| `analyst_20d` (from 2023-11-06) | +4.9% | 13.7% | 0.36 | +0.61 | -15.9% |
| `analyst_40d` (from 2023-11-06) | +10.7% | 9.2% | 1.17 | +1.98 | -8.5% |
| `analyst_60d` (from 2023-11-06) | +11.6% | 11.1% | 1.04 | +1.77 | -9.8% |

## Long-short, net of costs

| Strategy | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ts_sue_20d` (from 2014-08-05) | -9.7% | 15.1% | -0.64 | -2.24 | -81.6% |
| `ts_sue_40d` (from 2014-08-05) | -5.9% | 9.9% | -0.59 | -2.06 | -58.9% |
| `ts_sue_60d` (from 2014-08-05) | -1.2% | 8.6% | -0.13 | -0.47 | -35.3% |
| `analyst_20d` (from 2023-11-06) | -4.9% | 13.7% | -0.36 | -0.60 | -31.1% |
| `analyst_40d` (from 2023-11-06) | +4.8% | 9.2% | 0.52 | +0.89 | -11.9% |
| `analyst_60d` (from 2023-11-06) | +8.8% | 11.1% | 0.79 | +1.34 | -11.4% |
| `ts_sue_40d_same_window` (from 2023-11-06) | +0.1% | 9.4% | 0.01 | +0.02 | -14.9% |

## Legs, 40-day hold (excess returns)

| Strategy | Leg | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ts_sue` | long vs SPY (net) | -2.5% | 6.3% | -0.39 | -1.37 | -30.8% |
| `ts_sue` | long vs equal-weight (net) | -2.8% | 5.7% | -0.49 | -1.71 | -34.0% |
| `ts_sue` | shorted names vs equal-weight | -0.5% | 6.6% | -0.08 | -0.28 | -18.9% |
| `ts_sue` | avg positions | 51 long, 51 short | | | | |
| `analyst` | long vs SPY (net) | +1.1% | 8.8% | 0.12 | +0.19 | -15.4% |
| `analyst` | long vs equal-weight (net) | +2.6% | 7.1% | 0.37 | +0.56 | -10.6% |
| `analyst` | shorted names vs equal-weight | -8.4% | 7.2% | -1.16 | -1.81 | -20.2% |
| `analyst` | avg positions | 42 long, 42 short | | | | |

## Long-short net return by year, 40-day hold

| Year | `ts_sue` | `analyst` |
| --- | ---: | ---: |
| 2014 | -2.0% |  |
| 2015 | +4.9% |  |
| 2016 | -25.0% |  |
| 2017 | +0.2% |  |
| 2018 | -11.3% |  |
| 2019 | -10.6% |  |
| 2020 | -16.0% |  |
| 2021 | +6.1% |  |
| 2022 | -12.9% |  |
| 2023 | -8.5% | -0.1% |
| 2024 | +10.7% | -1.4% |
| 2025 | -10.2% | +9.4% |
| 2026 | +3.1% | +5.9% |

Survivorship caveat: companies that left the S&P 500 are missing, which flatters the short leg (failed companies with bad surprises are absent).
