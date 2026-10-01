# Reverse DCF: what growth is priced in?

Generated 2026-10-01 by `scripts/reverse_dcf.py`, prices as of 2026-10-01. Full table: [reverse_dcf.csv](reverse_dcf.csv).

For each company: the annual growth in free cash flow to the firm (operating cash flow − capex + after-tax interest, trailing twelve months, from SEC filings known today) over the next 10 years that makes a DCF equal today's enterprise value, with 2.5% growth afterwards. Main column uses a 9% discount rate; the CSV also has 8% and 10%. This is a description of market expectations, not a tested signal.

Companies: included 380, financial 76, negative FCF 36, captive finance 5, share count mismatch 2, debt not tagged 1.

## By sector

| Sector | Companies | Median implied growth | Median FCFF / EV | Median past 3y FCF growth |
| --- | ---: | ---: | ---: | ---: |
| Energy | 21 | -0.3% | 7.8% | +4.4% |
| Consumer Discretionary | 43 | +5.3% | 5.1% | +7.5% |
| Consumer Staples | 30 | +5.8% | 4.9% | +14.8% |
| Utilities | 14 | +6.1% | 4.8% | -3.8% |
| Health Care | 57 | +6.3% | 4.8% | +10.4% |
| Real Estate | 28 | +6.5% | 4.7% | +4.5% |
| Communication Services | 21 | +6.5% | 4.7% | +8.6% |
| Materials | 23 | +7.9% | 4.2% | +9.1% |
| Industrials | 75 | +8.2% | 4.1% | +13.3% |
| Information Technology | 68 | +11.5% | 3.2% | +17.5% |

## Lowest expectations (market prices in decline or slow growth)

| Company | Sector | Market cap ($bn) | FCFF / EV | Implied growth (9%) | FCF growth, past 3y | Revenue growth, past 3y |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CNC | Health Care | 30 | 49.8% | -29.5% | +5.4% | +11.1% |
| APA | Energy | 15 | 25.9% | -17.6% | +5.5% | n/a |
| OMC | Communication Services | 21 | 18.8% | -12.7% | +54.8% | +11.3% |
| EOG | Energy | 74 | 17.6% | -11.7% | -1.0% | +1.4% |
| MGM | Consumer Discretionary | 8 | 16.2% | -10.5% | +8.1% | +6.2% |
| FANG | Energy | 52 | 16.1% | -10.4% | +17.2% | +27.2% |
| EXPE | Consumer Discretionary | 32 | 15.4% | -9.7% | +24.3% | +8.6% |
| LULU | Consumer Discretionary | 10 | 15.3% | -9.6% | +11.8% | +7.9% |
| CMCSA | Communication Services | 77 | 14.9% | -9.3% | +10.2% | +1.2% |
| COP | Energy | 152 | 13.6% | -7.9% | +15.0% | -1.7% |
| VZ | Communication Services | 191 | 12.6% | -6.8% | +1.1% | +1.0% |
| GDDY | Information Technology | 12 | 12.4% | -6.7% | +24.2% | +7.1% |
| HPQ | Information Technology | 29 | 12.4% | -6.6% | +10.2% | +2.6% |
| DECK | Consumer Discretionary | 11 | 12.3% | -6.6% | +23.6% | +14.4% |
| ADBE | Information Technology | 93 | 12.1% | -6.3% | +11.6% | +11.2% |

## Highest expectations

| Company | Sector | Market cap ($bn) | FCFF / EV | Implied growth (9%) | FCF growth, past 3y | Revenue growth, past 3y |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P | Information Technology | 44 | 0.3% | +45.0% | -34.5% | +15.5% |
| LITE | Information Technology | 95 | 0.3% | +43.0% | +80.2% | +19.5% |
| BE | Industrials | 79 | 0.3% | +43.0% | n/a | +22.5% |
| IRM | Real Estate | 33 | 0.4% | +40.3% | n/a | +13.0% |
| RCL | Consumer Discretionary | 71 | 0.4% | +39.4% | n/a | +15.9% |
| TSLA | Consumer Discretionary | 1,415 | 0.4% | +39.0% | -2.2% | +3.3% |
| INTC | Information Technology | 601 | 0.6% | +34.4% | n/a | +1.8% |
| CRWD | Information Technology | 271 | 0.6% | +34.2% | +24.1% | +26.9% |
| AXON | Industrials | 34 | 0.6% | +33.5% | +6.3% | +33.1% |
| PLTR | Information Technology | 486 | 0.7% | +32.2% | +107.6% | +44.4% |
| MRVL | Information Technology | 233 | 0.8% | +30.2% | +30.6% | +18.9% |
| CEG | Utilities | 93 | 0.8% | +29.9% | n/a | +5.8% |
| AMD | Information Technology | 996 | 0.9% | +29.3% | +65.1% | +23.6% |
| MPWR | Information Technology | 67 | 0.9% | +28.7% | +24.1% | +21.0% |
| WMB | Energy | 84 | 0.9% | +28.1% | n/a | +2.0% |

## Caveats

* Terminal value is usually 60–80% of the DCF, so small changes in the discount rate move the implied growth a lot (compare the 8% and 10% columns).
* Trailing free cash flow can be temporarily depressed (heavy investment) or inflated (working-capital release); a low or negative base inflates implied growth.
* Debt is the tagged current plus non-current debt (or a reported total); leases and pensions are left out. Companies with no debt figure but sizeable non-current liabilities are excluded, as are CAT, DE, F, GM, PCAR (captive finance arms).
* FCFF / EV is the yield the growth rate is solved from: free cash flow plus after-tax interest, over enterprise value (market cap + debt − cash).
* Past growth uses the figures known three years ago, so it matches what investors saw.
