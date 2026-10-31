# Alpha Vantage earnings data validation

Generated 2026-10-30 by `scripts/download_av_earnings.py`. Symbols downloaded: 502 of 503 (unknown to Alpha Vantage: 2). Earliest report: 1996-01-31.

## Report dates

Against SEC-inferred announcement dates (40,012 quarters): same day 84%, within one day 91%, Alpha Vantage two or more days earlier 8.2%.

Against Yahoo report dates (`data/sentiment_500.csv`, 5,332 quarters): same day 85%, within one day 99%.

## Consensus vs. Yahoo (2023+)

4,531 quarters. Estimates within 1 cent: 84%; within 5%: 91%. Rank correlation of surprises: +0.91.

## Cross-check against FMP

Same report matched within 5 days. *AV exact* is the share of quarters where Alpha Vantage's estimate equals the actual; the last two columns describe those quarters: the share FMP flags as backfilled (revenue estimate equals actual too), and the share where FMP shows a real surprise (so Alpha Vantage's estimate was overwritten).

| Period | Quarters | Stocks | Same date | Estimate within 1c | AV exact | of which FMP-flagged | of which FMP real surprise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2010 on | 1,430 | 24 | 100% | 67% | 4% | 37% | 17% |
| before 2010 | 870 | 18 | 100% | 83% | 17% | 29% | 8% |

## Coverage by year

Flagged = estimate equals actual before 2010; excluded from surprise events as probable backfill.

| Year | Reported quarters | Symbols | With estimate | Estimate = actual | Flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1996 | 827 | 273 | 97% | 34% | 34% |
| 1997 | 1,127 | 292 | 97% | 34% | 34% |
| 1998 | 1,318 | 335 | 86% | 30% | 30% |
| 1999 | 1,348 | 343 | 88% | 26% | 26% |
| 2000 | 1,401 | 359 | 90% | 25% | 25% |
| 2001 | 1,429 | 372 | 93% | 29% | 29% |
| 2002 | 1,491 | 376 | 94% | 26% | 26% |
| 2003 | 1,509 | 379 | 95% | 26% | 26% |
| 2004 | 1,520 | 385 | 96% | 20% | 20% |
| 2005 | 1,551 | 392 | 97% | 18% | 18% |
| 2006 | 1,583 | 398 | 97% | 14% | 14% |
| 2007 | 1,608 | 408 | 98% | 16% | 16% |
| 2008 | 1,646 | 413 | 98% | 13% | 13% |
| 2009 | 1,654 | 418 | 98% | 12% | 12% |
| 2010 | 1,689 | 427 | 98% | 11% | 0% |
| 2011 | 1,727 | 436 | 98% | 12% | 0% |
| 2012 | 1,754 | 443 | 98% | 13% | 0% |
| 2013 | 1,796 | 454 | 98% | 12% | 0% |
| 2014 | 1,826 | 460 | 99% | 12% | 0% |
| 2015 | 1,852 | 466 | 99% | 10% | 0% |
| 2016 | 1,869 | 471 | 99% | 10% | 0% |
| 2017 | 1,889 | 474 | 99% | 9% | 0% |
| 2018 | 1,898 | 480 | 99% | 7% | 0% |
| 2019 | 1,932 | 486 | 99% | 8% | 0% |
| 2020 | 1,950 | 490 | 99% | 4% | 0% |
| 2021 | 1,969 | 493 | 99% | 4% | 0% |
| 2022 | 1,972 | 495 | 99% | 4% | 0% |
| 2023 | 1,981 | 496 | 100% | 4% | 0% |
| 2024 | 1,996 | 500 | 99% | 5% | 0% |
| 2025 | 2,002 | 501 | 99% | 4% | 0% |
| 2026 | 1,550 | 502 | 100% | 3% | 0% |
