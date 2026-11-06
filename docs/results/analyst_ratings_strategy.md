# Analyst rating changes and price-target revisions

Generated 2026-10-31 by `scripts/analyst_ratings_strategy.py`. Rules pre-registered in [analyst_ratings_protocol.md](analyst_ratings_protocol.md); data checks in [analyst_ratings_validation.md](analyst_ratings_validation.md). Deflated Sharpe probabilities assume 12 variants.

## Verdict (20-day hold, no earnings filter)

| Signal | Development t | Holdout return / yr | Supported (t ≥ 2 and holdout > 0) |
| --- | ---: | ---: | --- |
| `rating` | -3.75 | -3.4% | no |
| `target` | -0.80 | +1.6% | no |

## Events

| Signal | Events (long / short positions opened) | Near earnings (±3 days) | No earnings history |
| --- | ---: | ---: | ---: |
| `rating` | 21,933 | 4,622 (21%) | 0 |
| `target` | 18,540 | 5,954 (32%) | 2 |

## Development: long-short, net of costs

| Variant | Hold | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rating` | 5d | -28.1% | 11.7% | -2.41 | -8.63 | -97.5% |
| `rating` | 20d | -7.1% | 6.8% | -1.04 | -3.75 | -61.2% |
| `rating` | 60d | -3.8% | 4.3% | -0.87 | -3.15 | -39.8% |
| `rating_ex_earnings` | 5d | -26.9% | 13.7% | -1.96 | -7.02 | -97.3% |
| `rating_ex_earnings` | 20d | -6.7% | 7.7% | -0.87 | -3.13 | -61.5% |
| `rating_ex_earnings` | 60d | -4.1% | 4.7% | -0.86 | -3.12 | -42.4% |
| `target` | 5d | -20.7% | 25.9% | -0.80 | -2.86 | -96.8% |
| `target` | 20d | -4.1% | 18.3% | -0.22 | -0.80 | -68.2% |
| `target` | 60d | -4.6% | 14.8% | -0.31 | -1.13 | -65.4% |
| `target_ex_earnings` | 5d | -19.3% | 28.8% | -0.67 | -2.40 | -96.7% |
| `target_ex_earnings` | 20d | -1.8% | 20.6% | -0.09 | -0.30 | -69.1% |
| `target_ex_earnings` | 60d | -6.0% | 16.1% | -0.37 | -1.33 | -70.8% |

## Development: long-short, gross of costs

| Variant | Hold | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rating` | 5d | -1.9% | 11.6% | -0.16 | -0.58 | -47.5% |
| `rating` | 20d | -0.1% | 6.8% | -0.01 | -0.04 | -17.8% |
| `rating` | 60d | -1.3% | 4.3% | -0.30 | -1.10 | -22.8% |
| `rating_ex_earnings` | 5d | +0.4% | 13.7% | 0.03 | +0.11 | -43.5% |
| `rating_ex_earnings` | 20d | +0.8% | 7.7% | 0.10 | +0.37 | -19.9% |
| `rating_ex_earnings` | 60d | -1.5% | 4.7% | -0.31 | -1.13 | -25.1% |
| `target` | 5d | +3.0% | 25.9% | 0.12 | +0.41 | -67.8% |
| `target` | 20d | +3.6% | 18.3% | 0.19 | +0.70 | -54.0% |
| `target` | 60d | -1.8% | 14.8% | -0.12 | -0.43 | -54.9% |
| `target_ex_earnings` | 5d | +4.1% | 28.8% | 0.14 | +0.51 | -58.9% |
| `target_ex_earnings` | 20d | +6.2% | 20.6% | 0.30 | +1.08 | -54.2% |
| `target_ex_earnings` | 60d | -2.9% | 16.1% | -0.18 | -0.66 | -60.6% |

## Holdout: long-short, net of costs

| Variant | Hold | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rating` | 5d | -25.6% | 16.3% | -1.57 | -1.56 | -25.0% |
| `rating` | 20d | -3.4% | 9.2% | -0.37 | -0.37 | -7.6% |
| `rating` | 60d | -4.2% | 7.6% | -0.55 | -0.55 | -7.5% |
| `rating_ex_earnings` | 5d | -45.7% | 19.7% | -2.32 | -2.30 | -39.2% |
| `rating_ex_earnings` | 20d | -8.0% | 10.7% | -0.74 | -0.74 | -12.0% |
| `rating_ex_earnings` | 60d | -4.5% | 7.6% | -0.60 | -0.59 | -8.1% |
| `target` | 5d | -17.2% | 33.3% | -0.52 | -0.51 | -42.6% |
| `target` | 20d | +1.6% | 29.7% | 0.06 | +0.05 | -28.0% |
| `target` | 60d | +5.0% | 26.7% | 0.19 | +0.18 | -23.7% |
| `target_ex_earnings` | 5d | +1.3% | 35.6% | 0.04 | +0.04 | -43.8% |
| `target_ex_earnings` | 20d | -4.8% | 31.5% | -0.15 | -0.15 | -31.0% |
| `target_ex_earnings` | 60d | +0.9% | 27.2% | 0.03 | +0.03 | -23.8% |

## Holdout: long-short, gross of costs

| Variant | Hold | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rating` | 5d | +0.1% | 16.3% | 0.01 | +0.01 | -15.9% |
| `rating` | 20d | +4.0% | 9.2% | 0.44 | +0.43 | -6.5% |
| `rating` | 60d | -0.9% | 7.6% | -0.11 | -0.11 | -6.3% |
| `rating_ex_earnings` | 5d | -18.8% | 19.7% | -0.95 | -0.95 | -23.8% |
| `rating_ex_earnings` | 20d | -0.1% | 10.7% | -0.01 | -0.01 | -9.0% |
| `rating_ex_earnings` | 60d | -1.0% | 7.6% | -0.13 | -0.13 | -7.2% |
| `target` | 5d | +5.7% | 33.3% | 0.17 | +0.17 | -38.0% |
| `target` | 20d | +8.3% | 29.7% | 0.28 | +0.28 | -26.3% |
| `target` | 60d | +7.9% | 26.7% | 0.29 | +0.29 | -23.6% |
| `target_ex_earnings` | 5d | +25.7% | 35.6% | 0.72 | +0.72 | -39.1% |
| `target_ex_earnings` | 20d | +2.6% | 31.5% | 0.08 | +0.08 | -29.2% |
| `target_ex_earnings` | 60d | +4.3% | 27.2% | 0.16 | +0.16 | -23.6% |

## Legs, 20-day hold, development (excess over SPY)

| Variant | Leg | Return / yr | Vol | Sharpe | t | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rating` | long (net) − SPY | -2.7% | 6.6% | -0.41 | -1.48 | -32.8% |
| `rating` | shorted names − SPY | +0.8% | 8.5% | 0.10 | +0.35 | -24.2% |
| `rating` | avg positions | 64 long, 61 short | | | | |
| `rating_ex_earnings` | long (net) − SPY | -3.6% | 7.1% | -0.51 | -1.85 | -40.5% |
| `rating_ex_earnings` | shorted names − SPY | -0.4% | 9.3% | -0.04 | -0.16 | -34.1% |
| `rating_ex_earnings` | avg positions | 51 long, 47 short | | | | |
| `target` | long (net) − SPY | -2.3% | 9.0% | -0.25 | -0.91 | -42.5% |
| `target` | shorted names − SPY | -1.8% | 15.8% | -0.12 | -0.41 | -48.1% |
| `target` | avg positions | 59 long, 37 short | | | | |
| `target_ex_earnings` | long (net) − SPY | -3.3% | 10.1% | -0.33 | -1.18 | -49.5% |
| `target_ex_earnings` | shorted names − SPY | -5.3% | 18.0% | -0.30 | -1.03 | -64.1% |
| `target_ex_earnings` | avg positions | 39 long, 26 short | | | | |

## Long-short net return by year, 20-day hold, development

| Year | `rating` | `rating_ex_earnings` | `target` | `target_ex_earnings` |
| --- | ---: | ---: | ---: | ---: |
| 2013 | -8.2% | -12.9% | +11.1% | -0.2% |
| 2014 | -9.7% | -12.2% | -10.8% | -6.8% |
| 2015 | +1.4% | -0.5% | +18.9% | +21.8% |
| 2016 | -14.8% | -13.3% | -23.7% | -8.3% |
| 2017 | -6.9% | -3.3% | +22.9% | +28.2% |
| 2018 | -4.0% | -2.0% | -1.1% | +5.8% |
| 2019 | -14.7% | -13.3% | -14.3% | -11.8% |
| 2020 | -8.4% | -3.0% | +2.6% | +0.6% |
| 2021 | -6.0% | -7.2% | -19.3% | -20.1% |
| 2022 | +3.7% | +5.4% | -2.3% | +4.7% |
| 2023 | -11.5% | -13.2% | -24.7% | -20.1% |
| 2024 | -0.5% | -0.7% | -3.7% | -6.2% |
| 2025 | -12.3% | -10.4% | -7.9% | -10.2% |

## Findings (first run, 2026-10-31; no deviations from the protocol)

* Neither signal is supported. Rating changes: development t = −3.75 net, holdout −3.4% a year. Price targets: t = −0.80, holdout +1.6% a year.
* Gross of costs, rating changes earn nothing after day +1 (−0.1% a year over 20 days, t = −0.04); the net loss is trading costs. The stock moves on day 0 (+1.6% for upgrades, −1.9% for downgrades vs SPY in the validation), and by the next close the news is priced. Excluding actions near earnings changes nothing (+0.8% gross, t = +0.37).
* Price-target revisions: +3.6% a year gross over 20 days (t = +0.70), with 18% volatility and a −54% drawdown; noise, not a signal.
* The long legs trail SPY (rating upgrades −2.7% a year net), and the shorted names do not trail it: there is no drift on either side. Rating changes lose money in 11 of 13 development years.
* Costs: the engine charges for re-weighting every open position whenever one enters or exits, so net figures overstate costs (roughly double a buy-and-hold-per-position implementation); the conclusion rests on the gross figures, which are zero.

Survivorship caveat: companies that left the S&P 500 are missing, which flatters the short leg (failed companies that were downgraded are absent).
