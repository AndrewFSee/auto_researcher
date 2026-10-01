# Scripts

Run from the repository root. Every script puts `src/` on the path itself, or
you can install the package with `pip install -e .`. On Windows, set
`PYTHONUTF8=1` if you redirect a script's output to a file; some scripts print
characters the default code page cannot encode.

## Pipeline

| Script | Purpose |
| --- | --- |
| `run_ranking_low_memory.py` | Full three-stage ranking pipeline (ML screen, agents, composite, report) |
| `run_pipeline_subprocess.py` | The same pipeline driven by the Streamlit dashboard, with a progress file |
| `calibrate_ic_weights.py` | Writes `data/agent_ic.json`, the per-agent evidence behind the composite weights |

## Evaluation (leakage-safe)

These use the purged harness in `auto_researcher.backtest.walk_forward` or
announcement-dated events. Their outputs are the numbers the README cites.

| Script | Purpose |
| --- | --- |
| `ml_walkforward_backtest.py` | Purged walk-forward of the ML models and baselines, with `--leak-ablation` and `--window-sweep`; writes `docs/results/walkforward.*` |
| `feature_research.py` | Pre-registered test of published factors, sector-neutral targets and a 500-stock universe; writes `docs/results/feature_research.*` |
| `earnings_feature_research.py` | Round 2: point-in-time earnings features (SEC-filing announcement dates, time-series SUE) plus an event-time drift check; writes `docs/results/earnings_feature_research.*` |
| `download_fmp_earnings.py` | Download FMP earnings history (actual vs. consensus EPS, ~250 requests/day on the free plan, resumable) and write `docs/results/fmp_earnings_validation.md`; `earnings_event_strategy.py` uses it once 100+ stocks are cached |
| `download_av_earnings.py` | Download Alpha Vantage earnings history (consensus back to ~1996; 25 requests/day, resumable; refreshes files older than 30 days once complete) and write `docs/results/av_earnings_validation.md`, including a cross-check against FMP. Used by `earnings_event_strategy.py` once 100+ stocks are cached |
| `schedule_av_download.ps1` | Register a daily Windows scheduled task for `download_av_earnings.py` (`-Time 07:30` to change the time, `-Remove` to delete it); logs to `data/logs/av_download.log` |
| `build_fundamentals.py` | Download SEC XBRL company facts for the S&P 500 and build the point-in-time fundamentals table (first-reported values, usable after the filing date, TTM from year-to-date figures); writes `docs/results/sec_fundamentals_validation.md` |
| `reverse_dcf.py` | Reverse DCF: the 10-year FCF growth each company's enterprise value implies, beside its past growth and value/quality ratios; writes `docs/results/reverse_dcf.*`, or `--tickers AAPL KO` for a console view |
| `fundamental_factor_research.py` | Round 3, pre-registered in `docs/results/fundamental_factors_protocol.md`: six fundamental factors and an ML growth-forecast valuation gap through the walk-forward harness; writes `docs/results/fundamental_factors.*` |
| `earnings_event_strategy.py` | Event-driven long/short earnings-drift strategy (analyst and year-over-year surprises, 20/40/60-day holds); writes `docs/results/earnings_event_strategy.*` |
| `sentiment_audit.py` | Point-in-time audit of the sentiment agent's FinBERT signal with a timing ablation; writes `docs/results/sentiment_audit.*` |
| `pead_event_study.py` | Earnings surprise vs. announcement reaction and post-announcement drift; writes `docs/results/pead_event_study.*` |
| `validate_signal_ic.py` | Walk-forward IC, Newey-West and deflated Sharpe for any `(date, ticker, score)` signal or event file; refuses events dated at fiscal period ends |
| `run_cpcv_report.py` | Combinatorial purged CV report for the ML model |
| `ml_stack_backtest.py` | CPCV comparison of XGBoost, Transformer, GNN and their IC-weighted ensemble |
| `altdata_dump.py` | Materialize an alt-data adapter to a signal file for `validate_signal_ic.py` |
| `diagnostic_finbert_signal.py` | Lead/lag profile of FinBERT news sentiment vs. returns |

## Backtests through the library runner

`auto_researcher.backtest.runner` purges training labels, so these are sound
apart from the survivorship bias of current-constituent universes.

| Script | Purpose |
| --- | --- |
| `run_large_cap_backtest.py` | Configurable backtest CLI (fundamentals, regimes, costs, ensembles) |
| `run_baseline_backtest.py`, `run_baseline_comparisons.py` | Model vs. equal-weight, momentum and random baselines |
| `run_experiment_grid.py`, `run_universe_scaling_experiments.py`, `run_random_universe_validation.py` | Experiment grids via `auto_researcher.cli.main` |
| `hyperparam_optuna_cv.py` | Optuna search scored by full backtests on sequential folds |

## Signal research (pre-audit)

Exploratory backtests for individual signals: `backtest_*.py`,
`build_earnings_model_v2.py`, `pead_enhancement.py`, `early_adopter_detection.py`,
`ic_backtest_v2.py`, `historical_tech_backtest.py`, `fulltext_sentiment_backtest.py`,
`signal_decay.py`, `significance_test.py`, `feature_stability_analysis.py`,
`spy_comparison.py`, `analyze_*.py` and the forward-bias checks `audit_*.py`.
They build their own train/test splits and predate the September 2026 audit.
Re-run a signal through `validate_signal_ic.py` or the walk-forward harness
before citing any number they print.

## Data ingestion

| Script | Purpose |
| --- | --- |
| `scrape_sp500.py`, `async_scraper.py`, `update_news.py`, `fill_missing_articles.py` | Build and refresh `data/news.db` |
| `score_news_sentiment.py` | FinBERT-score scraped articles |
| `download_fundamentals.py`, `check_fundamentals_coverage.py` | Fetch fundamentals (FMP, Alpha Vantage) and check coverage |
