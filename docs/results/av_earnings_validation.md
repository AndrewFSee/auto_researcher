# Alpha Vantage earnings data validation

Generated 2026-10-01 by `scripts/download_av_earnings.py`. Symbols downloaded: 19 of 503 (unknown to Alpha Vantage: 0). Earliest report: 1996-01-31.

## Report dates

Against SEC-inferred announcement dates (1,504 quarters): same day 85%, within one day 87%, Alpha Vantage two or more days earlier 10.8%.

Against Yahoo report dates (`data/sentiment_500.csv`, 279 quarters): same day 94%, within one day 96%.

## Consensus vs. Yahoo (2023+)

182 quarters. Estimates within 1 cent: 78%; within 5%: 85%. Rank correlation of surprises: +0.89.

## Cross-check against FMP

Same report matched within 5 days. *AV exact* is the share of quarters where Alpha Vantage's estimate equals the actual; the last two columns describe those quarters: the share FMP flags as backfilled (revenue estimate equals actual too), and the share where FMP shows a real surprise (so Alpha Vantage's estimate was overwritten).

| Period | Quarters | Stocks | Same date | Estimate within 1c | AV exact | of which FMP-flagged | of which FMP real surprise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2010 on | 1,151 | 19 | 100% | 71% | 5% | 38% | 15% |
| before 2010 | 813 | 16 | 100% | 83% | 18% | 29% | 8% |

## Coverage by year

Flagged = estimate equals actual before 2010; excluded from surprise events as probable backfill.

| Year | Reported quarters | Symbols | With estimate | Estimate = actual | Flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1996 | 43 | 14 | 88% | 21% | 21% |
| 1997 | 59 | 16 | 97% | 25% | 25% |
| 1998 | 64 | 16 | 94% | 22% | 22% |
| 1999 | 63 | 16 | 98% | 17% | 17% |
| 2000 | 63 | 16 | 100% | 17% | 17% |
| 2001 | 64 | 16 | 94% | 22% | 22% |
| 2002 | 63 | 16 | 97% | 17% | 17% |
| 2003 | 64 | 16 | 92% | 9% | 9% |
| 2004 | 64 | 16 | 100% | 19% | 19% |
| 2005 | 63 | 16 | 100% | 22% | 22% |
| 2006 | 64 | 16 | 100% | 16% | 16% |
| 2007 | 64 | 16 | 97% | 11% | 11% |
| 2008 | 63 | 16 | 100% | 14% | 14% |
| 2009 | 63 | 16 | 100% | 11% | 11% |
| 2010 | 63 | 16 | 100% | 10% | 0% |
| 2011 | 62 | 16 | 100% | 8% | 0% |
| 2012 | 64 | 16 | 100% | 5% | 0% |
| 2013 | 67 | 17 | 99% | 6% | 0% |
| 2014 | 68 | 17 | 100% | 10% | 0% |
| 2015 | 68 | 17 | 100% | 1% | 0% |
| 2016 | 67 | 17 | 100% | 3% | 0% |
| 2017 | 67 | 17 | 100% | 9% | 0% |
| 2018 | 68 | 17 | 99% | 6% | 0% |
| 2019 | 72 | 18 | 99% | 7% | 0% |
| 2020 | 75 | 19 | 99% | 3% | 0% |
| 2021 | 77 | 19 | 99% | 3% | 0% |
| 2022 | 76 | 19 | 99% | 1% | 0% |
| 2023 | 76 | 19 | 100% | 1% | 0% |
| 2024 | 76 | 19 | 100% | 3% | 0% |
| 2025 | 76 | 19 | 100% | 3% | 0% |
| 2026 | 56 | 19 | 100% | 4% | 0% |
