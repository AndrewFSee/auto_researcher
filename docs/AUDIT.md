# Audit report: September 2026

## Summary

The project's headline performance numbers were artifacts of evaluation bugs,
not properties of the models. The walk-forward IC of +0.145, the "out-of-sample"
Sharpe of 3.53, and the PEAD IC of +0.22 all trace back to look-ahead or to a
test that did not measure what it claimed. Once the evaluation was fixed,
**none of the ML rankers shows a statistically significant IC** on 103 US large
caps over 2017-2025, and simple 12-month momentum does as well as any model.
Post-earnings drift survives re-measurement on announcement dates, but only
over nine quarters, so it is tentative.

Alongside the research fixes, the audit repaired a repository that could not be
installed from git (an entire package was never committed), a CI pipeline that
could not pass, a Docker setup that could not build, and 29 failing tests. It also
removed about 17,100 lines of dead or broken code.

All numbers below can be regenerated with the commands under
[Reproducing the results](#reproducing-the-results).

## Headline claims vs. re-measurement

| Previous claim (README) | Cause | Re-measured |
| --- | --- | --- |
| ML walk-forward IC **+0.145** (t = 7.6), IC > 0 in 88% of periods | Training used every date before the test date; 21-day labels overlapped the test label by up to 20 days | IC **-0.015** (t = -0.85), IC > 0 in 41% ([walkforward.md](results/walkforward.md)). The live screening recipe: **-0.025** (t = -1.8) |
| Top-minus-bottom quintile "**+36.2%** per rebalance period" | Spread measured in vol-normalized target units, printed as a return | Top-minus-bottom quintile **-0.27%** per 21-day period |
| ICs are "Spearman rank correlations" | The walk-forward script used Pearson | All ICs are now per-date Spearman |
| Out-of-sample 2023-2025: Sharpe **3.53**, max drawdown **-1.3%** | Not a holdout: a fresh walk-forward on a 2-year window that left ~9 monthly returns after warm-up, plus a max-drawdown bug that ignored initial capital | Replaced by a full purged walk-forward: net Sharpe **0.75** vs **1.00** for an equal-weight portfolio of the same stocks |
| "OOS Sharpe exceeds in-sample, so no overfitting" | Invalid inference; OOS > IS is not evidence against overfitting | Removed |
| "Beats random selection by +0.24 Sharpe (proves ranking skill)" | The random baseline averaged 50 random portfolios into one diversified return stream | **23rd percentile** of 500 random top-10 portfolios |
| 6-month training window "IC 0.259 vs 0.129" for 24 months | The leak is stronger when the leaked rows are a larger share of a short window | Leaky: +0.231 / +0.158 / +0.112 for 126 / 252 / 504 days. Purged: **-0.018 / -0.021 / -0.015** |
| PEAD IC **+0.220** (40 days), L/S **+14.9%** | Events dated at the fiscal quarter end; reports arrive a median 18 trading days later, so the window contained the announcement reaction | Announcement reaction IC +0.19 (not tradeable). **Post-announcement 40-day drift IC +0.137** (t = 3.1, 9 quarters), big beats minus big misses +5.2% ([pead_event_study.md](results/pead_event_study.md)) |

The leak ablation isolates the cause: on identical data and model, removing
only the purge takes the IC from +0.112 to -0.011; moving execution to the next
close changes it little further (-0.015).

## Findings

Severity: **Critical** = invalidates a published result; **High** = wrong
numbers or a broken workflow; **Medium** = misleading or fragile.

### Research validity

| # | Severity | Finding | Fix |
| --- | --- | --- | --- |
| 1 | Critical | No purge between training labels and the test date in `ml_walkforward_backtest.py`, `ml_window_comparison.py`, `ml_window_extended.py`, and the live model's "historical IC" in `recommend.py` | New purged splitter (`validation/splits.py`) and harness (`backtest/walk_forward.py`); scripts rewritten or removed; the live IC estimate uses a purged holdout |
| 2 | Critical | `run_oos_alpha_validation.py` "holdout" was an independent short backtest (~9 periods) | Removed; the walk-forward report covers 105 monthly out-of-sample periods |
| 3 | Critical | PEAD events keyed on fiscal quarter ends (`backtest_pead.py`), then re-validated as "stronger" (`validate_signal_ic.py`), cited in the README, used as the earnings agent's calibrated IC (0.107) and turned into a "+20.8% expected return" by `pead_enhanced.py` | New announcement-dated event study; `validate_signal_ic.py` refuses period-end-dated events (`validation/event_dates.py`); PEAD constants replaced; the composite ignores the contaminated calibration |
| 4 | Critical | Composite weights used `abs(IC)`, so anti-predictive agents (momentum, IC -0.026) got positive weight; ML weight floored at an IC of 0.05; "default ICs" (0.15, 0.12, 0.10, ...) were guesses; the thematic agent got 20% weight from 31 events of a proxy signal | `composite.py`: signed ICs clipped at zero and shrunk toward a small prior by sample size (`(n * IC + 36 * 0.02) / (n + 36)`); agents without measured evidence get the prior |
| 5 | High | Vol-normalized spreads reported as percent returns; Pearson reported as Spearman | Harness reports Spearman IC and quantile returns in raw return units |
| 6 | High | Random baseline averaged paths before computing Sharpe | `random_selection_null`: Sharpe distribution of 500 random portfolios; the model's percentile is reported |
| 7 | Medium | Every universe is today's constituent list (survivorship bias) | Not fixable without point-in-time membership data; reports now compare against the equal-weight portfolio of the same names and say so |
| 8 | Medium | Four post-model score penalties in the live screen were tuned on anecdotes (e.g. "UNH at -2.5 std") and never backtested | Kept for continuity but documented as unvalidated risk overlays; `--no-overlays` disables them |
| 9 | Medium | "Drivers" attached to each recommendation were the model's *global* feature importances | Per-stock SHAP contributions from XGBoost (`pred_contribs`) |
| 10 | Medium | Live screen used hindsight calendar regime labels ("2024-2026") as a feature | Removed from the recipe |

### Statistics bugs

| # | Severity | Finding | Fix |
| --- | --- | --- | --- |
| 11 | High | `compute_max_drawdown` ignored the initial capital: returns of -10%, +5% reported a 0% drawdown | Drawdown measured from a starting wealth of 1.0 |
| 12 | High | Deflated Sharpe applied the per-period variance formula to annualized Sharpe ratios, overstating z-scores by about sqrt(252) for daily data | `periods_per_year` argument; null dispersion defaults to the Sharpe sampling s.d. |
| 13 | High | CPCV purged `horizon` *calendar* days (21 calendar ~ 15 trading) and only before each test group | Purge in trading days on both sides of each test group |
| 14 | Medium | Backtest runner used Newey-West lag 62 on ~100 non-overlapping monthly ICs (via `locals().get("horizon_days")`) | Lag 0 for non-overlapping samples; the harness uses `ceil(horizon / step) - 1` |
| 15 | Medium | Sharpe was CAGR / volatility, inconsistent with the DSR and IR statistics | Standard mean / s.d. x sqrt(periods) |

### Engineering

| # | Severity | Finding | Fix |
| --- | --- | --- | --- |
| 16 | Critical | `src/auto_researcher/data/` (price loader, universes, scrapers, vector stores, alt-data; 41 importers) was never committed: `.gitignore`'s `data/` matched the package | Ignore rules anchored to the repo root (`/data/`, `/results/`) |
| 17 | High | Package code imported the repo-root `recommend.py` through `sys.path` hacks (one only worked with forward-slash paths) | Moved into the package as `auto_researcher.screening` |
| 18 | High | CI could not pass: 6,101 ruff findings, a black check over 117 files, Python 3.10 in the matrix vs `requires-python >= 3.11`, undeclared dependencies (xgboost, scipy, ...), a 70% coverage gate | Correctness-focused ruff rules (clean), 3.11-3.13 matrix, accurate dependencies and extras, type-checked evaluation core |
| 19 | High | Docker could not build or run: `COPY configs/` (missing), `python -m auto_researcher` (no `__main__`), a scheduler module that does not exist; Postgres and Redis were never used by any code | Two-stage image serving the dashboard; single-service compose file |
| 20 | High | 29 failing tests. 27 mocked `litellm` but not the availability flag; fixing that exposed a real bug (`skip_hold_signals` was ignored). 2 needed `chromadb` without skipping | Fixed; optional-dependency tests skip cleanly |
| 21 | High | `XGBRegressionModel`/`XGBRankingModel` passed `early_stopping_rounds` to `fit()`, a `TypeError` on XGBoost >= 2 whenever a validation set was given | Passed to the estimator constructor |
| 22 | High | The dashboard launched the pipeline with a hard-coded `.venv/Scripts/python.exe` | Uses `sys.executable` |
| 23 | Medium | Pipeline paths were relative to the working directory; the report used a second hard-coded directory; the ranking table crashed on non-UTF-8 consoles after results were saved | Paths anchored to the project root (override: `AUTO_RESEARCHER_RESULTS_DIR`); UTF-8 console output |
| 24 | Medium | Insider agent read `SEC_USER_AGENT` while `.env` and all other modules use `SEC_API_USER_AGENT`, and it never loaded `.env` | Reads the standard variable and loads `.env` |
| 25 | Medium | Dead code: `ranking_system.py` (its `--ml-stack` option never ran), orchestrator v1, `attribution`, `audit`, `backtest/evaluation.py`, three unused models, three unused agents; 41 debug/one-off/broken/leaky scripts (one did not parse) | Removed (57 files, 17,102 lines; recoverable from git history) |
| 26 | Low | Rolling MAD features took 79 s for 103 stocks | Vectorized with identical output: 6 s |

## What was added

* **`validation/splits.py`**: purged walk-forward splits in trading days, with an
  explicit execution lag and embargo. `purge=False` reproduces the legacy leak
  for measurement only.
* **`backtest/walk_forward.py`**: one harness for every model. It reports
  per-date Spearman IC with a Newey-West t-stat, quantile returns, a daily
  top-k portfolio net of costs next to the benchmark and the equal-weight universe,
  deflated information ratios, and a random-selection null distribution.
* **`backtest/baselines.py`**: single-factor and linear-IC baselines.
* **`screening.py`**: the live ML recipe as a model object, so the backtested
  object is the one that runs live.
* **`composite.py`**: evidence-based composite weights.
* **`validation/event_dates.py`**: rejects event datasets keyed on period ends.
* **Tests** (613 total, 0 failing): a leakage canary (a memorizing model on a
  random walk reaches IC ~+0.4 under the legacy split and ~0 when purged),
  feature-causality tests (perturbing future prices must not change past
  features), exact portfolio accounting, CPCV/DSR regressions, and composite
  weighting.

## Evidence by agent

| Agent | Evidence | Status |
| --- | --- | --- |
| ML screen | Purged walk-forward IC -0.025 (t = -1.8, 105 periods) | No skill: zero weight once `calibrate_ic_weights.py` is rerun (the current `data/agent_ic.json` still holds a 0.15 placeholder, which is treated as "no evidence") |
| Earnings (PEAD) | Announcement-dated drift IC +0.090 at 20 days, +0.137 at 40 days (9 quarters, survivorship-biased) | Tentative; the largest weight after recalibration |
| Sentiment | Point-in-time audit: all 8 signal/horizon pairs \|t\| < 1.6 ([sentiment_audit.md](results/sentiment_audit.md)); the old +0.020 came from a backtest with after-close news in its return windows and weights fit on the test period | No edge (prior weight) |
| Fundamental | IC +0.017 over 5 periods | Inconclusive (prior weight) |
| Momentum (sector) | IC -0.026 over 33 periods | Zero weight (its calibration downloads live ETF data and falls back to the prior if that fails) |
| Insider, filing tone, earnings-call quality | Literature priors only | Unvalidated (prior weight) |
| Thematic / early adopter | 31 events of a proxy signal | Unvalidated (prior weight) |

## Follow-up studies

After the audit, three pre-registered studies tested whether the ML ranker can
be rescued ([feature_research.md](results/feature_research.md),
[earnings_feature_research.md](results/earnings_feature_research.md)): it
cannot with price, factor or earnings features. An event-driven earnings
strategy ([earnings_event_strategy.md](results/earnings_event_strategy.md))
found that surprises against analyst consensus earn +10.7%/yr gross long-short
(Sharpe 1.2, t 2.0) over 2.3 years, while year-over-year surprises earn about
+1% gross. The sentiment audit ([sentiment_audit.md](results/sentiment_audit.md))
found no edge in the agent's FinBERT signal.

## Recommended next steps

1. **Point-in-time universes.** Historical index membership (including delisted
   names) is the only real fix for survivorship bias.
2. **Re-audit the remaining agents** (insider, filing tone, earnings-call
   quality, thematic) through the harness or an event study. Until then they
   carry only the prior weight.
3. **Extend the analyst-surprise sample.** Announcement dates now come from SEC
   filings back to the 1990s; the bottleneck is point-in-time consensus
   estimates, available locally only from late 2023. FMP's history (back to
   the 1990s) validates well against Yahoo (surprise rank correlation 0.88,
   report dates 97% exact; [fmp_earnings_validation.md](results/fmp_earnings_validation.md)),
   but the free plan covers only a few dozen S&P 500 stocks. Alpha Vantage's
   free key covers the whole index back to about 1996 at 25 stocks a day;
   `scripts/download_av_earnings.py` runs daily and agrees with Yahoo
   (surprise rank correlation 0.89, report dates 94% exact;
   [av_earnings_validation.md](results/av_earnings_validation.md)).
   `scripts/earnings_event_strategy.py` adds it as `av_consensus` once 100
   stocks are cached. Survivorship bias remains: the universe is today's index.
4. **Decide what the ML screen is for.** A pre-registered follow-up with 13
   published factors, sector-neutral targets and a 500-stock universe found no
   edge either ([feature_research.md](results/feature_research.md)); price and
   volume data alone are unlikely to be enough for large caps. A second round
   with point-in-time earnings features also found no edge
   ([earnings_feature_research.md](results/earnings_feature_research.md)). With an IC of
   about zero, stage 1 is close to a random filter on which names the agents
   analyze. A momentum or liquidity screen may be a better default until a
   model beats the baselines.
5. **Validate or drop the heuristic overlays** in `screening.py`.
6. **Test fundamental factors point-in-time.** The pre-audit DCF/FCF result
   (IC +0.10, falling to +0.02 once filing lags were applied) used statements
   dated by period end. `data/sec_fundamentals.py` now provides first-reported
   SEC figures from 2009, usable only after their filing date
   ([sec_fundamentals_validation.md](results/sec_fundamentals_validation.md):
   11% of revenue figures were later restated by more than 1%). A
   pre-registered round 3 ([fundamental_factors.md](results/fundamental_factors.md))
   found only accruals supported, weakly; an ML growth forecast beats naive
   forecasts, and its valuation gap was the strongest signal tested but did
   not clear the pre-registered bar against a naive-forecast gap. Re-test the
   gap once more holdout accrues (no changes to the model) before relying on
   it. The legacy, non-point-in-time `data/finagg_fundamentals.py` has been removed.
7. **Formatting.** Run `ruff format` in a dedicated formatting-only commit so it
   does not bury functional changes.
8. **Commit history.** Done: the audit, the data and fundamentals work, and a
   cleanup that removed the legacy ML stack (backtest runner, CLI, GBDT/GNN/
   Transformer/ensemble rankers, hyperparameter tuner, regime and risk
   modules, alt-data adapters, unused agents and pre-audit scripts) are
   separate commits; anything removed is in the history before the cleanup.

## Reproducing the results

```bash
# Purged walk-forward, leak ablation and training-window sweep (~20 min, offline)
python scripts/ml_walkforward_backtest.py --offline --leak-ablation --window-sweep 126,252,504

# Announcement-dated PEAD event study (offline)
python scripts/pead_event_study.py --offline

# Refresh the agent calibration the pipeline uses (writes data/agent_ic.json)
python scripts/calibrate_ic_weights.py

# Tests
pytest -q
```

`--offline` uses the local price cache in `data/price_cache/` (not in git). On a
fresh clone, drop `--offline` to download prices with yfinance, e.g.
`--universe sp100 --lookback-years 10`.
