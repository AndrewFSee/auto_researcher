# Alpha Vantage earnings data validation

Generated 2026-10-26 by `scripts/download_av_earnings.py`. Symbols downloaded: 430 of 503 (unknown to Alpha Vantage: 9). Earliest report: 1996-01-31.

## Report dates

Against SEC-inferred announcement dates (34,321 quarters): same day 85%, within one day 91%, Alpha Vantage two or more days earlier 8.1%.

Against Yahoo report dates (`data/sentiment_500.csv`, 4,652 quarters): same day 85%, within one day 99%.

## Consensus vs. Yahoo (2023+)

3,885 quarters. Estimates within 1 cent: 84%; within 5%: 91%. Rank correlation of surprises: +0.92.

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
| 1996 | 712 | 235 | 97% | 35% | 35% |
| 1997 | 978 | 253 | 97% | 35% | 35% |
| 1998 | 1,132 | 288 | 87% | 30% | 30% |
| 1999 | 1,162 | 296 | 88% | 27% | 27% |
| 2000 | 1,209 | 310 | 90% | 26% | 26% |
| 2001 | 1,234 | 320 | 94% | 30% | 30% |
| 2002 | 1,283 | 323 | 95% | 27% | 27% |
| 2003 | 1,297 | 326 | 95% | 26% | 26% |
| 2004 | 1,309 | 332 | 96% | 21% | 21% |
| 2005 | 1,339 | 339 | 98% | 19% | 19% |
| 2006 | 1,368 | 344 | 98% | 15% | 15% |
| 2007 | 1,388 | 352 | 98% | 16% | 16% |
| 2008 | 1,419 | 356 | 99% | 13% | 13% |
| 2009 | 1,425 | 360 | 98% | 12% | 12% |
| 2010 | 1,455 | 368 | 98% | 11% | 0% |
| 2011 | 1,486 | 375 | 98% | 12% | 0% |
| 2012 | 1,509 | 381 | 99% | 13% | 0% |
| 2013 | 1,543 | 390 | 99% | 12% | 0% |
| 2014 | 1,569 | 395 | 99% | 12% | 0% |
| 2015 | 1,591 | 400 | 99% | 10% | 0% |
| 2016 | 1,603 | 404 | 99% | 10% | 0% |
| 2017 | 1,617 | 406 | 99% | 9% | 0% |
| 2018 | 1,621 | 410 | 100% | 7% | 0% |
| 2019 | 1,648 | 415 | 100% | 8% | 0% |
| 2020 | 1,666 | 419 | 100% | 4% | 0% |
| 2021 | 1,685 | 422 | 100% | 4% | 0% |
| 2022 | 1,688 | 424 | 100% | 5% | 0% |
| 2023 | 1,695 | 424 | 100% | 4% | 0% |
| 2024 | 1,708 | 428 | 100% | 5% | 0% |
| 2025 | 1,714 | 429 | 100% | 4% | 0% |
| 2026 | 1,309 | 430 | 100% | 3% | 0% |
