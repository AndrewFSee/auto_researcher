# Customer momentum regression in S&P 500 suppliers, 2010-2017

Generated 2026-11-18 17:24 by `scripts/customer_momentum.py run-early`, under the rules pre-registered in [customer_momentum_2010_protocol.md](customer_momentum_2010_protocol.md). Deviations from the protocol: none. The first run (2026-11-18 17:17) gave the same primary result; its pooled row, information only, was wrong (191 months, +0.074) because the 2018-2026 month-ends were re-read from sentiq's SPY file, which had been cut to a rolling window that morning. They now come from the cached 2018-2026 returns, which reproduce that test exactly.

**Result: not supported.** The monthly Fama-MacBeth slope of suppliers' next-month return on their customers' return, controlling for their own month-t return, averaged +0.051 (Newey-West t = 1.60, 96 months); the decision rule needs a positive mean with t ≥ 2.0. In 2018-2026, where it was found, it was +0.075 (t 2.40).

## Data

* 10-Ks filed 2009-2017 by 583 companies of sentiq's S&P 500 list (members from 2018 on): 4892; with a link: 595 (831 supplier-customer links, text only). Filings with an Exhibit 13 (not read): 225.
* Suppliers with a link at some point: 130; distinct customers: 94. Without yfinance prices (left out): AGN, ALXN, ATVI, AVB, CEG, CELG, CSGNF, CTXS, CXO, EA, EXE, GPJA, HBI, HFC, HOLX, INGM, JNPR, K, METRY, MRO, MXIM, MYL, NBL, NFX, NWCYY, PXD, QRVO, SKHY, TSS, WBA, XEC, XLNX.
* Most common customer names with 10% or more that match no current listed company: U.S. Government (45), United States (37), Hewlett-Packard Company (26), China (24), Europe (22), Japan (20), Dell Inc. (19), Germany (17), Sprint (15), Texas (13), Australia (12), Asia Pacific (12), U.S. government (12), CVS Caremark Corporation (11), Northern California (11).
* Suppliers in the regression each month: min 43, median 50, max 60; distinct customers behind them: median 33. Supplier-months whose signal includes Walmart: 30%. Months skipped (fewer than 30 suppliers): 0.

| Year filed | 10-Ks | With a link | Links |
| --- | ---: | ---: | ---: |
| 2009 | 510 | 63 | 83 |
| 2010 | 514 | 63 | 83 |
| 2011 | 524 | 66 | 87 |
| 2012 | 532 | 62 | 86 |
| 2013 | 543 | 66 | 92 |
| 2014 | 554 | 64 | 90 |
| 2015 | 567 | 65 | 94 |
| 2016 | 571 | 74 | 111 |
| 2017 | 577 | 72 | 105 |

## Primary test

| Measure | Mean | NW t | Months | Months > 0 |
| --- | ---: | ---: | ---: | ---: |
| Fama-MacBeth slope on customer momentum, controlling for own month-t return (primary) | +0.051 | 1.60 | 96 | 61% |

Only this row decides. The slope is the next-month return per unit of customer return (0.10 means a supplier whose customers rose 10% earns 1 point more the next month).

## Secondary measures

| Measure | Mean | NW t | Months | Months > 0 |
| --- | ---: | ---: | ---: | ---: |
| Slope without the own-return control | +0.043 | 1.41 | 96 | 61% |
| Own month-t return slope (in the primary regression) | -0.025 | -1.06 | 96 | 49% |
| Terciles on customer momentum, net | +0.06% | 0.20 | 96 | 53% |
| Terciles on customer momentum, gross | +0.32% | 1.16 | 96 | 55% |
| Terciles on the residual of customer momentum after own return, net | +0.23% | 0.76 | 96 | 50% |
| Terciles on the residual, gross | +0.50% | 1.66 | 96 | 61% |
| Rank correlation of customer momentum with next month's return | +0.022 | 1.30 | 96 | 57% |
| Primary slope, holding months 2010-02 to 2013-12 | +0.032 | 0.67 | 47 | 62% |
| Primary slope, holding months 2014-01 to 2018-01 | +0.070 | 1.64 | 49 | 61% |
| Primary slope pooled with 2018-2026 (information only) | +0.064 | 2.83 | 200 | 61% |

Mean monthly turnover: terciles 2.66, residual terciles 2.70 (costs 10 bps per unit).

## By year (primary slope)

| Year | Months | Mean slope | Suppliers (median) |
| --- | ---: | ---: | ---: |
| 2010 | 11 | +0.038 | 51 |
| 2011 | 12 | +0.080 | 48 |
| 2012 | 12 | -0.051 | 48 |
| 2013 | 12 | +0.061 | 45 |
| 2014 | 12 | +0.044 | 50 |
| 2015 | 12 | -0.063 | 50 |
| 2016 | 12 | +0.096 | 56 |
| 2017 | 12 | +0.204 | 57 |
| 2018 | 1 | +0.042 | 54 |
