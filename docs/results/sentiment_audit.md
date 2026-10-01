# Sentiment agent audit

Generated 2026-09-30 by `scripts/sentiment_audit.py`.

* **Data.** 197,435 FinBERT-scored articles for 453 S&P 500 tickers from `data/news.db`; coverage is dense only from 2024 (median 436 names per day with a 30-day score). Test dates run from February 2024.
* **Timing.** An article dated day d is used from the close of the next trading day; positions are entered one trading day later. Long-only top-50 portfolios, 10 bps costs.
* **Multiple testing.** 8 pre-specified signal/horizon pairs; deflated probabilities account for all of them.

## Previous evidence

The composite weight used IC +0.020 over 90 periods from `scripts/backtest_news_combined.py`, which aligned articles by calendar date with returns starting at that day's close (after-close news leaks into the window), chose its signal weights on the full sample including the test period, and used an 80/20 split without a purge gap.

## Point-in-time results, weekly (5d)

| Metric | `sent_7d` | `sent_30d` | `sent_change` | `news_surge` |
| --- | ---: | ---: | ---: | ---: |
| Periods | 100 | 103 | 100 | 111 |
| Mean IC | +0.0052 | +0.0113 | +0.0015 | -0.0037 |
| IC t (NW) | +0.61 | +1.00 | +0.23 | -0.55 |
| IC > 0 | 55% | 59% | 55% | 49% |
| Top-bottom quintile / period | +0.06% | +0.06% | +0.03% | -0.07% |
| Top-50 IR vs EW (net) | -1.04 | -0.07 | -2.23 | -1.40 |
| P(IR > 0), deflated | 0.00 | 0.06 | 0.00 | 0.00 |
| Turnover / rebalance | 84% | 41% | 87% | 69% |

## Point-in-time results, monthly (21d)

| Metric | `sent_7d` | `sent_30d` | `sent_change` | `news_surge` |
| --- | ---: | ---: | ---: | ---: |
| Periods | 23 | 24 | 23 | 26 |
| Mean IC | -0.0111 | +0.0163 | -0.0166 | +0.0213 |
| IC t (NW) | -0.55 | +0.81 | -1.15 | +1.58 |
| IC > 0 | 48% | 71% | 48% | 54% |
| Top-bottom quintile / period | -0.35% | +0.38% | -0.59% | +0.31% |
| Top-50 IR vs EW (net) | -0.60 | +0.52 | -1.43 | +0.24 |
| P(IR > 0), deflated | 0.01 | 0.24 | 0.00 | 0.14 |
| Turnover / rebalance | 86% | 79% | 90% | 79% |

## Timing ablation (`sent_7d`)

Legacy alignment: articles usable at the close of their own calendar day and traded at that close.

| Metric | `legacy weekly (5d)` | `point-in-time weekly (5d)` | `legacy monthly (21d)` | `point-in-time monthly (21d)` |
| --- | ---: | ---: | ---: | ---: |
| Periods | 100 | 100 | 23 | 23 |
| Mean IC | -0.0050 | +0.0052 | -0.0082 | -0.0111 |
| IC t (NW) | -0.56 | +0.61 | -0.37 | -0.55 |
| IC > 0 | 52% | 55% | 65% | 48% |
| Top-bottom quintile / period | -0.03% | +0.06% | -0.34% | -0.35% |
| Top-50 IR vs EW (net) | -1.06 | -1.04 | -0.69 | -0.60 |
| P(IR > 0), deflated | 0.00 | 0.00 | 0.01 | 0.01 |
| Turnover / rebalance | 84% | 84% | 85% | 86% |
