# Event-driven earnings drift strategy

Generated 2026-10-30 by `scripts/earnings_event_strategy.py`.

Rules fixed before running: rank each surprise against the previous year's announcements, long the top 20% and short the bottom 20%, enter one day after the surprise is usable (two trading days after the SEC filing), hold 40 trading days (20/60 as sensitivity), equal weight per leg, 10 bps per unit traded. Universe: current S&P 500 members from their index add dates. Deflated Sharpe probabilities assume 9 variants.

`fmp_consensus` and `av_consensus` use vendor earnings histories (actual vs. consensus EPS from FMP and Alpha Vantage; validated in [fmp_earnings_validation.md](fmp_earnings_validation.md) and [av_earnings_validation.md](av_earnings_validation.md)), each with its own price panel from 1994 and equal-weight benchmark; a surprise needs 60 (not 200) earlier announcements in the past year to be ranked. Universe: current S&P 500 members, so early years carry survivorship bias.

Costs are charged on daily changes in equal-weight targets, which re-weights every open position when one enters or exits; a buy-and-hold-per-position implementation would pay roughly half. Gross results are shown for that reason.

## Long-short, gross of costs

| Strategy | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ts_sue_20d` (from 2014-08-05) | +2.1% | 15.5% | 0.13 | +0.47 | -46.3% |
| `ts_sue_40d` (from 2014-08-05) | +0.6% | 10.2% | 0.06 | +0.20 | -30.1% |
| `ts_sue_60d` (from 2014-08-05) | +1.4% | 8.7% | 0.16 | +0.55 | -24.9% |
| `analyst_20d` (from 2023-11-06) | +3.2% | 13.5% | 0.23 | +0.40 | -16.2% |
| `analyst_40d` (from 2023-11-06) | +9.7% | 9.1% | 1.07 | +1.85 | -8.7% |
| `analyst_60d` (from 2023-11-06) | +10.8% | 10.9% | 0.99 | +1.71 | -9.9% |
| `av_consensus_20d` (from 1996-04-26) | +6.6% | 22.5% | 0.29 | +1.61 | -57.0% |
| `av_consensus_40d` (from 1996-04-26) | +4.5% | 14.4% | 0.31 | +1.73 | -52.2% |
| `av_consensus_60d` (from 1996-04-26) | +1.5% | 9.3% | 0.16 | +0.87 | -43.5% |

## Long-short, net of costs

| Strategy | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ts_sue_20d` (from 2014-08-05) | -9.3% | 15.5% | -0.60 | -2.09 | -81.7% |
| `ts_sue_40d` (from 2014-08-05) | -6.3% | 10.2% | -0.62 | -2.16 | -59.7% |
| `ts_sue_60d` (from 2014-08-05) | -1.5% | 8.7% | -0.18 | -0.62 | -35.7% |
| `analyst_20d` (from 2023-11-06) | -6.4% | 13.5% | -0.47 | -0.81 | -32.6% |
| `analyst_40d` (from 2023-11-06) | +4.0% | 9.1% | 0.44 | +0.76 | -12.1% |
| `analyst_60d` (from 2023-11-06) | +8.1% | 10.9% | 0.74 | +1.28 | -11.5% |
| `av_consensus_20d` (from 1996-04-26) | -4.3% | 22.5% | -0.19 | -1.06 | -94.1% |
| `av_consensus_40d` (from 1996-04-26) | -2.3% | 14.4% | -0.16 | -0.89 | -85.0% |
| `av_consensus_60d` (from 1996-04-26) | -1.4% | 9.3% | -0.15 | -0.83 | -71.1% |
| `ts_sue_40d_same_window` (from 2023-11-06) | -1.5% | 10.4% | -0.14 | -0.25 | -16.2% |

## Legs, 40-day hold (excess returns)

| Strategy | Leg | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ts_sue` | long vs SPY (net) | -2.5% | 6.3% | -0.40 | -1.41 | -31.9% |
| `ts_sue` | long vs equal-weight (net) | -3.2% | 5.9% | -0.55 | -1.91 | -35.8% |
| `ts_sue` | shorted names vs equal-weight | -0.5% | 6.8% | -0.08 | -0.28 | -19.0% |
| `ts_sue` | avg positions | 51 long, 51 short | | | | |
| `analyst` | long vs SPY (net) | +1.0% | 8.9% | 0.11 | +0.18 | -15.5% |
| `analyst` | long vs equal-weight (net) | +2.6% | 7.2% | 0.36 | +0.56 | -10.6% |
| `analyst` | shorted names vs equal-weight | -7.5% | 7.3% | -1.03 | -1.61 | -18.8% |
| `analyst` | avg positions | 41 long, 41 short | | | | |
| `av_consensus` | long vs SPY (net) | -2.1% | 8.0% | -0.27 | -0.96 | -34.4% |
| `av_consensus` | long vs equal-weight (net) | +0.4% | 9.7% | 0.04 | +0.21 | -55.2% |
| `av_consensus` | shorted names vs equal-weight | -0.6% | 11.4% | -0.06 | -0.30 | -54.8% |
| `av_consensus` | avg positions | 35 long, 34 short | | | | |

## Long-short net return by year, 40-day hold

| Year | `ts_sue` | `analyst` | `av_consensus` |
| --- | ---: | ---: | ---: |
| 1996 |  |  | +14.9% |
| 1997 |  |  | +13.3% |
| 1998 |  |  | +24.9% |
| 1999 |  |  | +3.6% |
| 2000 |  |  | +18.9% |
| 2001 |  |  | +5.3% |
| 2002 |  |  | -13.0% |
| 2003 |  |  | +4.1% |
| 2004 |  |  | +5.6% |
| 2005 |  |  | -5.3% |
| 2006 |  |  | -0.9% |
| 2007 |  |  | +11.4% |
| 2008 |  |  | -29.8% |
| 2009 |  |  | -30.8% |
| 2010 |  |  | -9.3% |
| 2011 |  |  | -17.1% |
| 2012 |  |  | -3.7% |
| 2013 |  |  | -10.7% |
| 2014 | -2.0% |  | -6.9% |
| 2015 | +4.9% |  | -3.4% |
| 2016 | -25.0% |  | -7.9% |
| 2017 | +0.2% |  | +2.0% |
| 2018 | -11.3% |  | -0.4% |
| 2019 | -11.1% |  | -4.5% |
| 2020 | -16.1% |  | -1.5% |
| 2021 | +5.5% |  | -4.8% |
| 2022 | -13.0% |  | -17.2% |
| 2023 | -8.0% | -0.4% | -9.9% |
| 2024 | +11.2% | -2.2% | -1.7% |
| 2025 | -10.8% | +9.6% | +13.5% |
| 2026 | -1.3% | +4.9% | -9.0% |

## Exploratory: vendor histories by period, 40-day hold

Not part of the pre-registered test: the periods were chosen after seeing the yearly returns. 1996-2001 is the classic drift era; before 2010 some consensus estimates look backfilled (excluded); "analyst window" is the `analyst` sample's period. Plain statistics, no multiple-testing deflation.

| Strategy | Period | Gross / yr | Gross t | Net / yr | Net t | Sharpe (net) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `av_consensus` | 1996-2001 (from 1996-04-26) | +20.6% | +2.10 | +14.3% | +1.45 | 0.61 |
| `av_consensus` | 2002-2009 (from 2002-01-02) | -0.7% | -0.12 | -7.4% | -1.34 | -0.47 |
| `av_consensus` | 2010 on (from 2010-01-04) | +1.5% | +0.73 | -5.5% | -2.64 | -0.65 |
| `av_consensus` | analyst window (from 2023-11-06) | +6.9% | +1.27 | +0.0% | +0.01 | 0.00 |

Survivorship caveat: companies that left the S&P 500 are missing, which flatters the short leg (failed companies with bad surprises are absent).
