# FMP earnings data validation

Generated 2026-10-01 by `scripts/download_fmp_earnings.py`. Symbols downloaded: 24 of 503; refused by the current FMP plan (HTTP 402): 216. The free plan serves only a fixed list of popular stocks, too few for the event strategy, which needs at least 100.

## Report dates vs. SEC-inferred announcement dates

1,844 reported quarters matched within 10 days (60%). Same day: 83%; within one day: 86%; FMP two or more days earlier: 12.0%. (The SEC rule deliberately errs late, so FMP being earlier is expected.)

Against independent report dates (`data/sentiment_500.csv`, 319 quarters): same day 97%, within one day 98%, FMP two or more days earlier 0.9%.

## Consensus vs. Yahoo (overlapping quarters, 2023+)

230 quarters. Estimates within 1 cent: 45%; within 5%: 77%. Actuals within 1 cent: 82% (the rest are mostly GAAP vs. adjusted EPS on quarters with one-off items). Rank correlation of surprises: +0.88.

## Coverage by year

Backfilled = estimate equals actual down to the dollar of revenue (excluded from surprise events).

| Year | Reported quarters | Symbols | Backfilled |
| --- | ---: | ---: | ---: |
| 1993 | 54 | 13 | 2% |
| 1994 | 52 | 14 | 0% |
| 1995 | 55 | 15 | 2% |
| 1996 | 57 | 15 | 0% |
| 1997 | 60 | 16 | 13% |
| 1998 | 61 | 16 | 7% |
| 1999 | 65 | 17 | 8% |
| 2000 | 66 | 17 | 8% |
| 2001 | 61 | 16 | 13% |
| 2002 | 64 | 17 | 12% |
| 2003 | 61 | 16 | 8% |
| 2004 | 68 | 18 | 12% |
| 2005 | 71 | 18 | 6% |
| 2006 | 70 | 18 | 4% |
| 2007 | 71 | 18 | 0% |
| 2008 | 71 | 18 | 4% |
| 2009 | 71 | 18 | 4% |
| 2010 | 72 | 18 | 3% |
| 2011 | 79 | 20 | 6% |
| 2012 | 80 | 20 | 2% |
| 2013 | 83 | 21 | 4% |
| 2014 | 82 | 21 | 4% |
| 2015 | 84 | 21 | 2% |
| 2016 | 83 | 21 | 2% |
| 2017 | 82 | 21 | 4% |
| 2018 | 84 | 21 | 2% |
| 2019 | 84 | 21 | 5% |
| 2020 | 86 | 22 | 3% |
| 2021 | 95 | 24 | 1% |
| 2022 | 96 | 24 | 0% |
| 2023 | 96 | 24 | 0% |
| 2024 | 96 | 24 | 0% |
| 2025 | 96 | 24 | 0% |
| 2026 | 71 | 24 | 0% |
