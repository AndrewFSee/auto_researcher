# Analyst rating actions: data validation

Generated 2026-10-30 by `scripts/download_analyst_ratings.py`. Yahoo: 502 of 503 stocks, 170,811 actions. FMP (free plan, cross-check only): 24 stocks, 14,370 actions.

## Coverage by year (Yahoo)

| Year | Actions | Stocks | Upgrades | Downgrades | Initiations | Maintained / reiterated | With price target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2011 | 4 | 4 | 0 | 2 | 0 | 2 | 50% |
| 2012 | 5,658 | 380 | 558 | 627 | 512 | 3,961 | 90% |
| 2013 | 5,999 | 388 | 983 | 1,176 | 937 | 2,903 | 81% |
| 2014 | 6,047 | 390 | 893 | 761 | 863 | 3,530 | 87% |
| 2015 | 5,878 | 401 | 930 | 892 | 901 | 3,155 | 81% |
| 2016 | 6,513 | 405 | 923 | 1,170 | 994 | 3,426 | 66% |
| 2017 | 4,426 | 412 | 884 | 774 | 824 | 1,944 | 65% |
| 2018 | 7,581 | 418 | 1,049 | 762 | 854 | 4,916 | 82% |
| 2019 | 7,398 | 431 | 828 | 1,096 | 840 | 4,634 | 85% |
| 2020 | 13,590 | 447 | 1,380 | 1,105 | 761 | 10,344 | 94% |
| 2021 | 10,995 | 452 | 1,010 | 706 | 817 | 8,462 | 96% |
| 2022 | 15,231 | 460 | 860 | 1,122 | 789 | 12,460 | 97% |
| 2023 | 18,461 | 472 | 996 | 946 | 897 | 15,622 | 98% |
| 2024 | 20,782 | 482 | 981 | 1,004 | 969 | 17,828 | 97% |
| 2025 | 21,388 | 488 | 1,027 | 847 | 876 | 18,638 | 98% |
| 2026 | 20,860 | 500 | 856 | 697 | 787 | 18,520 | 99% |

## Time of day (Yahoo, New York time)

| Before the open | During the session | After the close |
| ---: | ---: | ---: |
| 57% | 39% | 4% |

0% of timestamps are exactly midnight (probably a date without a time).

## Yahoo vs FMP (upgrades, downgrades, initiations)

On the 24 stocks FMP serves, from 2019-01-01: 1,048 FMP actions, 93% found in Yahoo (same stock and broker, within 2 days); of those, same day 100%, FMP a day earlier 0%, a day later 0%; same action 100%, same new rating 94%.


## Timing check: returns around each action

Day 0 is the first session that could react: the action's day if it came before the close, the next session if after (FMP: its date). Mean return in excess of SPY; a right timestamp puts the move on day 0, not day -1. Later days are not looked at here (that is where a signal would be).

| Source | Action | When | N | Day -2 | Day -1 | Day 0 | Day +1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Yahoo | upgrade | all | 14,115 | +0.08% | +0.34% | +1.56% | +0.03% |
| Yahoo | upgrade | pre-market | 11,704 | +0.06% | +0.32% | +1.58% | +0.05% |
| Yahoo | upgrade | session | 2,108 | +0.14% | +0.32% | +1.65% | -0.04% |
| Yahoo | upgrade | after close | 303 | +0.24% | +1.11% | +0.30% | -0.06% |
| Yahoo | downgrade | all | 13,639 | -0.09% | -0.55% | -1.91% | -0.04% |
| Yahoo | downgrade | pre-market | 11,337 | -0.10% | -0.51% | -1.95% | -0.04% |
| Yahoo | downgrade | session | 2,029 | -0.05% | -0.62% | -1.91% | -0.03% |
| Yahoo | downgrade | after close | 273 | -0.18% | -1.46% | -0.22% | +0.07% |
| FMP | upgrade | all | 1,006 | +0.25% | +0.43% | +1.85% | +0.17% |
| FMP | downgrade | all | 1,038 | +0.02% | -0.29% | -1.80% | -0.18% |

Universe: current S&P 500 members (survivorship: companies that left the index are missing).
