# Scripts

Run from the repository root. Every script puts `src/` on the path itself, or
you can install the package with `pip install -e .`. On Windows, set
`PYTHONUTF8=1` if you redirect a script's output to a file; some scripts print
characters the default code page cannot encode.

## Research workstation

| Script | Purpose |
| --- | --- |
| `workstation.py` | A cited, gated memo on one company as of a date, for human review (`docs/workstation/DESIGN.md`): `memo TICKER --as-of DATE` (add `--dry-run` to see what the tools return without calling Claude), `runs` lists runs and decisions, `decide RUN_ID ACTION` records yours. Reads sentiq's EDGAR lake, TickerLake and market-compass's published data, all read-only |
| `workstation_review.py` | Builds the review page for a pilot (`data/workstation/review/<pilot>.html`) from its saved runs: each memo with its quotes, numbers and calculations, and a verdict per paragraph and per memo; the published page stores the marks in its database (collection `reviews`) |
| `workstation_verify.py` | Runs the verifier (Sonnet 5.5, batch) on a pilot's saved memos: a verdict per analysis paragraph against only what it cites, saved to `data/workstation/verify/<run>.json`; with `--marks`, its agreement with a reviewer's marks from the review page |

## Analyst pipeline

| Script | Purpose |
| --- | --- |
| `run_ranking_low_memory.py` | Full three-stage ranking pipeline (screen, agents, composite, report) |
| `run_pipeline_subprocess.py` | The same pipeline driven by the Streamlit dashboard, with a progress file |
| `calibrate_ic_weights.py` | Writes `data/agent_ic.json`, the per-agent evidence behind the composite weights |

## Research (leakage-safe)

These use the purged harness in `auto_researcher.backtest.walk_forward` or
announcement-dated events. Their outputs are the reports in `docs/results/`
that the README cites.

| Script | Purpose |
| --- | --- |
| `ml_walkforward_backtest.py` | Purged walk-forward of the ML ranker and baselines, with `--leak-ablation` and `--window-sweep`; writes `docs/results/walkforward.*` |
| `feature_research.py` | Round 1, pre-registered: published factors, sector-neutral targets and a 500-stock universe; writes `docs/results/feature_research.*` |
| `earnings_feature_research.py` | Round 2: point-in-time earnings features (SEC-filing announcement dates, time-series SUE) plus an event-time drift check; writes `docs/results/earnings_feature_research.*` |
| `fundamental_factor_research.py` | Round 3, pre-registered in `docs/results/fundamental_factors_protocol.md`: six fundamental factors and an ML growth-forecast valuation gap; writes `docs/results/fundamental_factors.*` |
| `pead_event_study.py` | Earnings surprise vs. announcement reaction and post-announcement drift; writes `docs/results/pead_event_study.*` |
| `earnings_event_strategy.py` | Event-driven long/short earnings-drift strategy (analyst and year-over-year surprises, plus vendor histories once 100+ stocks are cached); writes `docs/results/earnings_event_strategy.*` |
| `sentiment_audit.py` | Point-in-time audit of the sentiment agent's FinBERT signal with a timing ablation; writes `docs/results/sentiment_audit.*` |
| `analyst_ratings_strategy.py` | Pre-registered in `docs/results/analyst_ratings_protocol.md`: rating changes and price-target revisions as long/short event signals; writes `docs/results/analyst_ratings_strategy.*` |
| `validate_signal_ic.py` | Walk-forward IC, Newey-West and deflated Sharpe for any `(date, ticker, score)` signal or event file; refuses events dated at fiscal period ends |

## Fundamentals and valuation

| Script | Purpose |
| --- | --- |
| `build_fundamentals.py` | Download SEC XBRL company facts for the S&P 500 and build the point-in-time fundamentals table (first-reported values, usable after the filing date, TTM from year-to-date figures); writes `docs/results/sec_fundamentals_validation.md` |
| `reverse_dcf.py` | Reverse DCF: the 10-year FCF growth each company's enterprise value implies, beside its past growth and value/quality ratios; writes `docs/results/reverse_dcf.*`, or `--tickers AAPL KO` for a console view |

## Filing tools (local models in Ollama)

| Script | Purpose |
| --- | --- |
| `finqa_baseline.py` | Score an Ollama model on FinQA (`--kind finqa`) or the filing questions (`--kind filings`) in `program`, `cot_program` or `direct` mode; resumable; writes `docs/results/finqa_baseline.*` (`--report` for another path, such as `filing_qa.md`) |
| `build_filing_qa.py` | Auto-graded FinQA-style questions from the inline XBRL of 10-K / 10-Q filings stored by sentiq (read-only) |
| `paraphrase_filing_questions.py` | Rewrite those questions the way an analyst would ask them, with mechanical checks that the calculation is unchanged |
| `filing_rag_eval.py` | Retrieval within a filing (BM25, embeddings, hybrid), then answering from the top table; writes `docs/results/filing_rag.*` |
| `filing_rerank_eval.py` | An LLM re-ranks BM25's top tables (`--think` for the thinking variant), then the fine-tuned model answers; writes `docs/results/filing_rerank*.md` |
| `moat_pilot.py` | Moat tagger: `sample` companies, `tag` their 10-Ks (Ollama or Claude), `report` the tags and the hand checks; writes `docs/results/moat_*.md` |
| `customer_links.py` | Customer-supplier links: `extract` each 10-K's major customers and their share of revenue (quotes checked word for word), read the `xbrl` concentration tags, `report` the links with the XBRL check and the hand check; writes `docs/results/customer_links_count.md` |
| `customer_momentum.py` | Customer momentum test (pre-registered in `docs/results/customer_momentum_protocol.md`): `prices` builds monthly total returns with a source check, `run` builds the links in force, the signal and the long-short; writes `docs/results/customer_momentum.*`; `prices-early` and `run-early` run the 2010-2017 retest (`docs/results/customer_momentum_2010_protocol.md`) on filings from `download_early_10ks.py` |
| `download_early_10ks.py` | 10-Ks filed 2009-2017 by S&P 500 companies, from EDGAR, as text made the way sentiq's lake makes it; `customer_links.py` reads them for years before 2018 |

## Data ingestion

| Script | Purpose |
| --- | --- |
| `download_av_earnings.py` | Alpha Vantage earnings history (consensus back to ~1996; 25 requests/day, resumable; refreshes files older than 30 days once complete); writes `docs/results/av_earnings_validation.md` |
| `schedule_av_download.ps1` | Register a daily Windows scheduled task for `download_av_earnings.py` (`-Time 07:30` to change the time, `-Remove` to delete it); logs to `data/logs/av_download.log` |
| `download_analyst_ratings.py` | Yahoo analyst rating actions for the S&P 500, cross-checked against FMP; writes `docs/results/analyst_ratings_validation.md` |
| `download_fmp_earnings.py` | FMP earnings history (~250 requests/day on the free plan, which serves only popular stocks); writes `docs/results/fmp_earnings_validation.md` |
| `scrape_sp500.py`, `async_scraper.py`, `update_news.py`, `fill_missing_articles.py` | Build and refresh `data/news.db` (the scrapers need `aiohttp`, in the `research` extra) |
| `score_news_sentiment.py` | FinBERT-score scraped articles |

Pre-audit experiment scripts (the legacy backtest runner, CPCV and stacked
rankers, alt-data adapters, signal backtests) were removed in the cleanup
after the audit; they are in the git history before that commit.
