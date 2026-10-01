# SEC point-in-time fundamentals: validation

Generated 2026-10-01 by `scripts/build_fundamentals.py`. Companies with facts: 503 of 503; 601,678 point-in-time values.

## Timing

Each value becomes usable the day after the filing that first reported it. Days from period end to availability, for periods ending 2011 or later. The long tail is periods first disclosed later (pre-IPO years in a prospectus-era 10-K, spin-offs, late filers); earlier periods were mostly first tagged as comparatives during the 2009–2011 XBRL phase-in and become usable only then.

| Values | 10th pct | Median | 90th pct |
| --- | ---: | ---: | ---: |
| TTM revenue / cash flow, December period ends (mostly 10-K) | 35 | 52 | 59 |
| TTM revenue / cash flow, other period ends | 24 | 33 | 41 |

Values available before their period ended: 0.

## Restatements

Share of periods whose figure in the latest filing differs from the first-reported one. The table keeps first-reported values; using the latest would leak later corrections into backtests.

| Item | Periods | Changed > 1% | Changed > 5% | Median delay to last version (days) |
| --- | ---: | ---: | ---: | ---: |
| net_income | 31,611 | 5.9% | 3.0% | 568 |
| operating_cash_flow | 31,504 | 9.7% | 4.8% | 568 |
| revenue | 30,237 | 11.4% | 7.1% | 567 |
| shares_diluted | 31,241 | 4.2% | 3.3% | 364 |
| total_assets | 31,246 | 0.9% | 0.2% | 0 |

## Agreement with Yahoo (fiscal-year figures)

Latest-reported SEC figures vs. Yahoo's annual statements (DefeatBeta), matched within 7 days of the fiscal year end. Differences are mostly definitions (e.g. Yahoo's net income to common holders).

| Item | Matched years | Within 1% | Within 5% |
| --- | ---: | ---: | ---: |
| revenue | 3,027 | 93% | 96% |
| net_income | 3,068 | 87% | 94% |
| operating_cash_flow | 2,986 | 98% | 99% |
| total_assets | 2,977 | 100% | 100% |

## Share counts

SEC share count (cover page, else diluted average; split-adjusted) vs. Yahoo's latest, 503 companies compared. Outside 0.80–1.25×: 7. These are excluded from market-cap ratios (`data/research_cache/sec_share_mismatch.json`); most are multi-class companies.

| Symbol | SEC / Yahoo |
| --- | ---: |
| BX | 0.603 |
| ARES | nan |
| BRK-B | nan |
| ERIE | nan |
| HSY | nan |
| STZ | nan |
| V | nan |

## Coverage

Share of companies with a current value (as of today):

| Item | Coverage |
| --- | ---: |
| total_assets | 100% |
| equity | 100% |
| cash | 100% |
| net_income | 100% |
| operating_cash_flow | 100% |
| revenue | 98% |
| shares_diluted | 97% |
| depreciation | 97% |
| stock_comp | 93% |
| shares_cover | 90% |
| capex | 88% |
| current_liabilities | 84% |
| current_assets | 84% |
| debt_total | 82% |
| buybacks | 81% |
| operating_income | 76% |
| debt_current | 76% |
| dividends_paid | 76% |
| debt_noncurrent | 74% |
| interest_expense | 73% |
| total_liabilities | 72% |
| cost_of_revenue | 58% |
| gross_profit | 37% |
| short_term_investments | 31% |

Companies with a value, by period-end year:

| Year | operating_cash_flow | revenue | shares_diluted | total_assets |
| --- | ---: | ---: | ---: | ---: |
| 2008 | 345 | 314 | 302 | 218 |
| 2009 | 402 | 374 | 382 | 356 |
| 2010 | 419 | 394 | 400 | 412 |
| 2011 | 425 | 402 | 409 | 419 |
| 2012 | 434 | 413 | 419 | 427 |
| 2013 | 442 | 420 | 426 | 435 |
| 2014 | 446 | 423 | 424 | 445 |
| 2015 | 452 | 430 | 429 | 448 |
| 2016 | 459 | 453 | 434 | 454 |
| 2017 | 470 | 467 | 447 | 462 |
| 2018 | 480 | 478 | 455 | 473 |
| 2019 | 485 | 481 | 467 | 481 |
| 2020 | 489 | 483 | 476 | 485 |
| 2021 | 492 | 485 | 479 | 490 |
| 2022 | 498 | 491 | 484 | 493 |
| 2023 | 501 | 494 | 489 | 498 |
| 2024 | 502 | 495 | 489 | 501 |
| 2025 | 501 | 493 | 488 | 503 |
| 2026 | 499 | 489 | 487 | 502 |
