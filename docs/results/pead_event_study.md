# PEAD event study (announcement-dated)

Generated 2026-09-29 by `scripts/pead_event_study.py --offline`.

* **Events.** 1,050 earnings events for 119 tickers with cached prices, 2023-10-17 to 2026-01-16. Surprise = (actual - estimate) / |estimate| from `data/pead_backtest_results.parquet`; report dates from `sentiment_500.csv`.
* **Timing.** Reports arrive a median 18 trading days after the fiscal quarter end (10th-90th percentile 11-26). All returns are in excess of SPY.
* **IC.** Pooled Spearman over all events, and the mean of per-quarter cross-sectional ICs (t-stat across quarters). The last two columns compare mean returns for surprises beyond +/-20% and beyond +/-5%.

| Return window | Events | Pooled IC | Mean quarterly IC | t | Quarters | Beat - miss (20%) | Beat - miss (5%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy: 40d from fiscal quarter end (contains announcement) | 896 | +0.214 | +0.210 | +4.91 | 9 | +12.22% | +9.17% |
| Announcement reaction (close before -> next close) | 913 | +0.192 | +0.190 | +4.38 | 9 | +4.85% | +3.88% |
| Post-announcement drift, 5 trading days | 908 | +0.045 | +0.046 | +1.05 | 9 | +0.15% | +1.02% |
| Post-announcement drift, 20 trading days | 903 | +0.100 | +0.090 | +2.66 | 9 | +2.43% | +2.13% |
| Post-announcement drift, 40 trading days | 882 | +0.140 | +0.137 | +3.14 | 9 | +5.15% | +4.54% |
| Post-announcement drift, 60 trading days | 808 | +0.127 | +0.164 | +2.26 | 9 | +3.59% | +4.24% |

Survivorship caveat: tickers are today's large caps, so firms that collapsed after bad surprises are missing, which biases the short side toward smaller losses.
