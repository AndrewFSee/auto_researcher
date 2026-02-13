# Auto-Researcher: AI-Powered Equity Research Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20check-mypy-blue.svg)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Institutional-grade quantitative research platform** combining ML-based stock ranking, multi-agent fundamental analysis, NLP-powered earnings/filing signals, RAG-augmented transcript search, regime-aware factor rotation, and sector rotation overlays — all unified in a 3-stage ranking pipeline with IC-calibrated signal weighting and an interactive Streamlit dashboard.

---

## Table of Contents

- [Pipeline Overview](#-pipeline-overview)
- [Model Performance Summary](#-model-performance-summary)
- [Alpha Models](#-alpha-models)
- [RAG Systems (ChromaDB)](#-rag-systems-chromadb)
- [Sector Rotation Overlay](#-sector-rotation-overlay)
- [Factor Rotation Model](#-factor-rotation-model)
- [IC Calibration System](#-ic-calibration-system)
- [Percentile-Based Signal Assignment](#-percentile-based-signal-assignment)
- [LLM Review Agent](#-llm-review-agent)
- [Deep Research Agent](#-deep-research-agent)
- [Risk Management](#%EF%B8%8F-risk-management)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Performance Tracking](#-performance-tracking)
- [Project Structure](#%EF%B8%8F-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#configuration)
- [Academic Foundations](#-academic-foundations)
- [Testing](#-testing)
- [Data Sources](#-data-sources)
- [Model Performance & Alpha Validation](#-model-performance--alpha-validation)

---

## 🔄 Pipeline Overview

The ranking pipeline (`scripts/run_ranking_low_memory.py`) processes stocks through three sequential stages, each designed to add orthogonal information while managing memory efficiently:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STOCK RANKING PIPELINE                              │
│                                                                       │
│  Stage 1: ML Screening                                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Load XGBoost model → compute features for full universe         │ │
│  │  (technical + fundamental + sentiment) → rank all stocks →       │ │
│  │  select top-K candidates → unload model to free memory           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              ↓  top-K stocks                          │
│  Stage 2: Multi-Agent Deep Analysis                                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  For each stock (one at a time, memory-cleared between):         │ │
│  │    → 8 non-ML agents score the stock (-1 to +1)                  │ │
│  │    → Post-processing: context bands, conflict resolution,       │ │
│  │      evidence budget, consistency checks, cross-validation,     │ │
│  │      risk qualifiers, freshness gates                            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              ↓  agent scores                          │
│  Stage 3: IC-Weighted Composite Scoring                               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Load calibrated ICs from data/agent_ic.json                     │ │
│  │  → factor rotation: detect regime via leading indicators         │ │
│  │    (VIX term structure, credit spreads, dispersion, breadth)     │ │
│  │  → adjust IC weights by regime (risk_on → risk_off profiles)    │ │
│  │  → composite = Σ(adjusted_ic_weight_i × score_i)                │ │
│  │  → apply missing-data penalties                                  │ │
│  │  → apply sector rotation overlay (0.8x — 1.2x tilt)             │ │
│  │  → percentile-based signal assignment (cross-sectional ranks)   │ │
│  │  → optional: LLM red-team review + deep research                │ │
│  │  → generate markdown report                                      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Running the Pipeline

```bash
# Set PYTHONPATH
$env:PYTHONPATH = "C:\Users\Andrew\projects\auto_researcher\src"  # Windows
export PYTHONPATH=src  # Linux/Mac

# Full pipeline: screen S&P 100, pass top 25 to agents, display top 10
python scripts/run_ranking_low_memory.py --universe sp100 --ml-top 25 --final-top 10

# Verbose mode with detailed agent logging
python scripts/run_ranking_low_memory.py --universe sp100 --ml-top 25 -v

# Skip ML stage (reuse previous screening results)
python scripts/run_ranking_low_memory.py --skip-ml --final-top 15

# S&P 500 universe, larger screening pool
python scripts/run_ranking_low_memory.py --universe sp500 --ml-top 50 --final-top 25
```

**Command-line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--universe` | `sp100` | Stock universe: `sp500`, `sp100`, `large_cap`, `core_tech` |
| `--ml-top` | `25` | Number of stocks to pass from ML screening to agents |
| `--final-top` | `10` | Top N stocks to display in final ranking |
| `--ml-weight` | `0.35` | ML score weight override in composite (default: IC-proportional) |
| `--batch-size` | `5` | Batch size for memory clearing between stocks |
| `--skip-ml` | — | Skip ML screening, load from previous results |
| `--skip-agents` | — | Skip agent analysis, load from previous results |
| `--output` | auto | Custom output JSON path |
| `--verbose` / `-v` | — | Enable detailed per-stock agent logging |

### Pipeline Output

The pipeline produces:
1. **JSON files** in `data/ranking_results/` — ML screening, agent analysis, and final rankings
2. **Markdown report** in `data/ranking_results/` — human-readable report with all agent scores, context bands, conflict resolution, and final rankings
3. **Console output** — summary table with rankings, scores, and signal breakdown

```
  FINAL STOCK RANKINGS
  ════════════════════════════════════════════════════════════════════════════
  Rank  Ticker  Score   Signal     ML%   Fund  Earn  Sent   Ins  Them   Mom  Tone   CaQ
  ──────────────────────────────────────────────────────────────────────────────────────
  1     NVDA    +0.487  buy       92.0  0.35  0.42  0.28  0.15  0.65  0.18  0.12  0.22
  2     AAPL    +0.412  buy       88.0  0.28  0.35  0.32  0.08  0.45  0.22  0.18  0.15
  ...
```

---

## 📊 Model Performance Summary

| Model | Signal | IC | t-stat | Quintile Spread | Hold Period |
|-------|--------|-----|--------|-----------------|-------------|
| **Early Adopter** | Tech pioneer detection | **+0.36** | 2.1 | **+25%** | 12 months |
| **Enhanced PEAD** | Big earnings surprises | **+0.152** | 2.8 | **+3.55%** | 60 days |
| **Topic Sentiment** | Earnings-topic news | **+0.021** | 3.4 | **+0.48%** | 10 days |
| **Insider Cluster** | Multi-insider buying | **+0.08** | 1.9 | **+3-5%** | 90 days |
| **ML Ranking (XGBoost)** | Technical + fundamental | **+0.12** | 2.5 | **+16.4%** | 63 days |
| **Filing Tone** | 10-K tone change (YoY) | **+0.04** | 1.5 | **+3-4%** | 90 days |
| **Sector Momentum** | Relative strength rotation | **+0.07** | 1.8 | **+2-4%** | 60 days |
| **Quality-Value** | Profitability + value | **+0.10** | 2.2 | **+2-3%** | 90 days |
| **Earnings Call Qual** | Transcript tone analysis | **+0.05** | 1.6 | **+2-3%** | 60 days |

> All ICs are Spearman rank correlations with forward returns. Statistical significance at p<0.05.

---

## 📈 Alpha Models

The platform has **19 production models** organized into categories. In the ranking pipeline, 9 of these are used as scoring agents (ML + 8 fundamental/alternative agents).

### Stage 1: ML Ranking Model

#### XGBoost Ranking Model (`xgb_ranking_model.py`)

The ML screening stage uses an XGBoost model for cross-sectional stock ranking. It supports both pairwise ranking (`rank:pairwise`) and regression (`reg:squarederror` with volatility-normalized returns).

```
Out-of-Sample Performance (2023-2025):
  Sharpe Ratio: 3.53 (improved from 1.44 in-sample)
  Annual Return: +45.3%
  Max Drawdown: -1.3%

vs Baselines:
  Equal-Weight: Sharpe 1.33    ML beats by +0.21
  Momentum Top-5: Sharpe 1.29  ML beats by +0.25
  Random Top-5: Sharpe 1.30    ML beats by +0.24 (proves ranking skill)
  SPY Buy-Hold: Sharpe 0.83    ML beats by +0.71
```

**Features Used (40+):**
- **Technical**: Momentum (1M, 3M, 6M, 12M), short-term reversal, volatility, idiosyncratic vol, volume trends
- **Fundamental**: Value (P/E, P/B, P/S), quality (ROE, ROA, gross margin), growth rates
- **Sentiment**: FinBERT scores, topic-adjusted sentiment
- **Alternative**: Insider signals, early adopter scores

**Training Protocol:**
- 6-month rolling window with walk-forward validation
- Volatility-normalized returns as target (reduces large-cap bias)
- SHAP-based feature selection to prevent overfitting
- Regime-aware training option (calendar-based regime labels as features)

**Key Finding:** The model's Sharpe ratio *improved* out-of-sample (1.44 → 3.53), indicating robust generalization and no overfitting.

A **LightGBM variant** (`gbdt_model.py`) is also available for comparison.

### Stage 2: Agent Scoring Models (8 Agents)

Each agent independently scores a stock from **-1** (strong sell) to **+1** (strong buy).

#### 1. Fundamental Agent — Quality-Value Model (IC: +0.10)

Multi-factor fundamental model based on Novy-Marx (2013) gross profitability, Fama-French (2015) five factors, and Piotroski F-Score.

**Sub-factors:**
- **Profitability**: Gross margin, operating margin, net margin, ROE, ROA
- **Value**: P/E, P/B, P/S relative to sector peers
- **Quality**: Earnings stability, balance sheet strength, cash conversion
- Sector-relative Z-score benchmarking with letter grades (A–F)

#### 2. Earnings Agent — Enhanced PEAD Model (IC: +0.152)

Post-Earnings Announcement Drift targeting **big surprises only** (>20% vs expectations).

```
Base Signal (all earnings):      IC = +0.049 (N=1,948)
Enhanced Signal (>20% surprise): IC = +0.152 (N=334) ← 3x improvement
Combined (big + consecutive):    L/S Spread = +5.08% at 60d
```

**Signal Logic:**
- LONG: Big positive surprise (>20%) + optional consecutive beat pattern
- SHORT: Big negative surprise (<-20%) + optional consecutive miss
- Hold period: 40–60 days for full drift capture

Also includes **Revenue Surprise** signals for revenue beats/misses.

#### 3. Insider Agent — Cluster Detection (IC: +0.08)

Based on Cohen, Malloy & Pomorski (2012). Detects cluster buying/selling by multiple insiders.

- Cluster event = 3+ insiders trade within 30 days
- Weights by role: CEO/CFO = 1.0, Director = 0.6, VP = 0.5
- Filters out 10b5-1 pre-planned trades and options exercises
- **Buying is informative, selling is less so** (many non-information reasons to sell)

#### 4. Thematic Agent — Sector-Aware Early Adopter (IC: +0.36)

**The highest-conviction signal in the platform.** Combines:

- **Early Adopter / Pioneer Score**: Detects companies discussing emerging technologies BEFORE their peers in earnings calls. Tracks 40+ emerging tech terms. Companies that mention AI, quantum computing, etc. months ahead of sector median outperform by +25% over 12 months.
- **Economic Moat Scoring**: Network effects, switching costs, scale advantages
- **Theme Exposure**: AI/ML, energy transition, cybersecurity, digital payments
- **Sector Rotation Signals**: Consumes `SectorMomentumModel` internally for sector-level context

**Sector-Aware Enhancement**: The early adopter model normalizes pioneer scores within each GICS sector rather than globally. A healthcare company mentioning "digital therapeutics" early is evaluated against healthcare peers, not tech companies.

#### 5. Momentum Agent — Sector Momentum (IC: +0.07)

Sector-relative momentum based on Moskowitz & Grinblatt (1999).

- Weighted relative strength: `0.30 × RS_1m + 0.50 × RS_3m + 0.20 × RS_6m` (vs SPY)
- Tracks 11 SPDR sector ETFs (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC)
- Generates stock-level tailwind/headwind signal based on sector membership
- Detects market regime: risk_on / risk_off / neutral / rotating

#### 6. Sentiment Agent — Topic-Classified News Sentiment (IC: +0.021)

Not all news is equal. Topic classification + FinBERT sentiment dramatically improves signal.

```
Generic FinBERT Sentiment: IC = +0.0004 (essentially noise)
Earnings-Topic Sentiment:  IC = +0.0210 ← 52x improvement
```

**Pipeline:**
1. Scrape business news → store in SQLite (`news.db`)
2. Index into ChromaDB vector store (`data/news_chroma/`, 210K+ articles)
3. Classify by topic (earnings, litigation, M&A, management changes)
4. Apply topic-specific sentiment multipliers
5. IC-weighted topic aggregation using `topic_ic.json`

| Topic | Sentiment Multiplier | Signal Strength |
|-------|---------------------|-----------------|
| Litigation/Regulatory | 1.5x | Negative sentiment hits harder |
| Earnings/Guidance | 1.2x | Immediate impact |
| M&A | 1.0x | Depends on acquirer vs target |
| Management Changes | 1.3x | CEO departures especially impactful |

#### 7. Filing Tone Agent — 10-K Tone Change (IC: +0.04)

Analyzes SEC 10-K/10-Q filings using the Loughran-McDonald financial dictionary.

**6 Tone Metrics:** Negative, Positive, Uncertainty, Litigious, Net Tone, Constraining

The key signal is **year-over-year tone change** — a filing that becomes significantly more negative or uncertain relative to the prior year signals deteriorating fundamentals before they show up in earnings.

- Expected alpha: +3–4% annually
- Loughran & McDonald (2011) dictionary specifically designed for financial text

#### 8. Earnings Call Qualitative Agent (IC: +0.05)

Deep analysis of earnings call transcripts with **6 sub-signals**:

| Sub-Signal | Weight | Description |
|------------|--------|-------------|
| Management Tone | 25% | FinBERT sentiment on management Q&A responses |
| Analyst-Mgmt Tone Gap | 12% | Divergence between analyst questions and management answers |
| Hedging Language | 13% | Frequency of uncertainty/hedging phrases |
| Guidance Specificity | 17% | How specific vs vague is forward guidance |
| QoQ Tone Change | 18% | Quarter-over-quarter shift in management tone |
| Peer Tone Delta | 15% | RAG-powered cross-company tone comparison via transcript vectorstore |

The **Peer Tone Delta** sub-signal uses the transcript ChromaDB vectorstore to compare a company's management tone against sector peers, detecting relative optimism or pessimism.

---

## 🔍 RAG Systems (ChromaDB)

Two vector stores provide retrieval-augmented generation (RAG) for the analysis agents.

### News Vector Store (`data/news_vectorstore.py`)

- **Backend**: ChromaDB with `all-MiniLM-L6-v2` embeddings (sentence-transformers)
- **Content**: 210,000+ business news articles
- **Storage**: `data/news_chroma/`
- **Source**: Business Insider scraper → SQLite → ChromaDB
- **Features**: Filtered retrieval by ticker, date range, and topic. IC-based topic weighting.
- **Used by**: Sentiment Agent for topic-classified news retrieval

### Transcript Vector Store (`data/transcript_vectorstore.py`)

- **Backend**: ChromaDB with `all-MiniLM-L6-v2` embeddings
- **Content**: Earnings call transcripts chunked by speaker turn
- **Storage**: `data/transcript_chroma/`
- **Source**: Parquet file (FMP API transcripts) → chunked and embedded
- **Rich metadata per chunk**: ticker, quarter, year, speaker name, speaker role, is_management, is_analyst, is_qa_section
- **Query methods**:
  - `query()` — semantic search within a ticker's transcripts
  - `query_by_theme()` — cross-company thematic search (e.g., "tariff impact")
  - `query_peer_comparison()` — compare management tone across sector peers
  - `query_ticker_history()` — track how a company's discussion evolves over time
- **Auto-indexing**: Pipeline automatically indexes any missing tickers via `ensure_tickers_indexed()` before agent analysis
- **Used by**: Earnings Call Qualitative Agent (peer_tone_delta sub-signal)

```bash
# Build/rebuild transcript index for specific tickers
python -m auto_researcher.data.transcript_vectorstore build \
    --rebuild --tickers AAPL,MSFT,GOOG --min-year 2023

# Check index stats
python -m auto_researcher.data.transcript_vectorstore stats

# Semantic query
python -m auto_researcher.data.transcript_vectorstore query \
    --ticker AAPL --query "AI revenue growth guidance" --n 5

# Cross-company theme search
python -m auto_researcher.data.transcript_vectorstore theme \
    --theme "tariff impact supply chain" --n 5
```

---

## 🔄 Sector Rotation Overlay

A post-agent overlay applied in Stage 3 that detects **divergence between fundamental breadth and sector ETF price momentum** to tilt composite scores toward sectors with leading fundamental improvement.

**This is NOT a separate agent.** It multiplicatively adjusts composite scores (0.8x–1.2x) after all agents have scored.

**How It Works:**
1. **Aggregate** existing agent scores (earnings, fundamental, filing_tone, earnings_call_qual) by GICS sector across all pipeline stocks
2. **Compute breadth** = fraction of stocks in each sector with improving signals
3. **Fetch sector ETF price momentum** (11 SPDR ETFs vs SPY, 1m + 3m weighted)
4. **Detect divergence**: z(breadth) − z(price_momentum)
   - Positive divergence: breadth improving but prices haven't moved yet → bullish tilt
   - Negative divergence: prices rising but fundamentals deteriorating → bearish tilt
5. **Apply tilt**: `composite_score *= tilt` where tilt ∈ [0.80, 1.20] via tanh mapping

**Academic Basis:** Chan, Jegadeesh & Lakonishok (1996) — earnings revision breadth leads sector returns by 1–3 months.

Sectors with fewer than 3 stocks in the pipeline get neutral tilt (1.0x) to avoid unreliable estimates.

---

## 🔄 Factor Rotation Model

**File:** `src/auto_researcher/models/factor_rotation.py`

While the sector rotation overlay tilts *which sectors* to favour, the factor rotation model adjusts *which agents to trust more* based on the current macro regime. Traditional regime detection (e.g., 21-day volatility + 200-day MA) is **lagging** — by the time it confirms a regime shift, the move has already happened.

The factor rotation model uses **leading indicators** that detect regime transitions 2–6 weeks ahead, enabling proactive weight adjustment rather than reactive confirmation.

### Leading Indicators

| Indicator | Source | What It Detects |
|-----------|--------|-----------------|
| **VIX Term Structure** | VIX / VIX3M ratio + 5-day change | Near-term fear spike vs calm (inverted curve = stress) |
| **Credit Spread Momentum** | HYG / IEF ratio (5-day change) | Risk appetite shift in bond markets |
| **Cross-Sectional Dispersion** | Stdev of 11 sector ETF returns | Regime break: low dispersion = consensus, high = uncertainty |
| **Breadth Thrust** | Fraction of pipeline stocks with improving signals | Bottom-up confirmation of broad recovery |
| **Factor Momentum** | Rolling IC trends from `data/agent_ic.json` | Which agents are currently predictive (momentum in alpha) |

### Regime States

The model classifies the market into **5 regime states**, including transition phases that lagging models miss:

| Regime | Description | Factor Profile |
|--------|-------------|----------------|
| `RISK_ON_EARLY` | Recovery beginning — breadth improving, VIX declining | ↑ Momentum, ↑ Thematic, ↑ Earnings |
| `RISK_ON_LATE` | Expansion mature — dispersion low, credit stable | ↑ ML, ↑ Fundamental, neutral others |
| `NEUTRAL` | No clear directional signal | Equal weights (no adjustment) |
| `RISK_OFF_EARLY` | Stress emerging — VIX curve inverting, credit widening | ↑ Fundamental, ↑ Filing Tone, ↓ Momentum |
| `RISK_OFF_LATE` | Full risk-off — high vol, wide spreads | ↑ Insider, ↑ Fundamental, ↓ Thematic |

### How It Integrates

```
Stage 3 (IC-Weighted Scoring):
  1. Load base ICs from data/agent_ic.json
  2. FactorRotationModel.detect_regime()
     → scores 5 leading indicators
     → classifies into FactorRegime enum
     → outputs transition_probability + transition_direction
  3. FactorRotationModel.adjust_ic_weights(base_ics)
     → applies regime-specific multipliers (0.6x – 1.4x per agent)
     → tilt_strength parameter (default 0.5) controls aggressiveness
     → re-normalizes weights to sum to 1.0
  4. Adjusted weights used for composite scoring
  5. Regime state stored in agent_rationales["factor_regime"]
```

**Graceful Degradation:** If market data feeds fail (VIX, HYG, sector ETFs unavailable), the model falls back to neutral weights with no adjustment rather than crashing.

**Academic Basis:** Asness, Moskowitz & Pedersen (2013) — value and momentum everywhere; Arnott et al. (2016) — factor timing via macro indicators; Bender et al. (2018) — regime-conditional factor allocation.

---

## ⚖ IC Calibration System

The pipeline uses empirically calibrated Information Coefficients to weight each agent proportionally to its historical predictive power.

```bash
# Run IC calibration (computes Spearman IC for all 9 agents)
python scripts/calibrate_ic_weights.py --verbose

# Output: data/agent_ic.json
```

**Default ICs (fallback when no calibration file exists):**

| Agent | Default IC | Weight (normalized) |
|-------|-----------|---------------------|
| ML | 0.15 | ~20.5% |
| Earnings | 0.12 | ~16.4% |
| Fundamental | 0.10 | ~13.7% |
| Sentiment | 0.08 | ~11.0% |
| Momentum | 0.07 | ~9.6% |
| Insider | 0.06 | ~8.2% |
| Thematic | 0.05 | ~6.8% |
| Earnings Call Qual | 0.05 | ~6.8% |
| Filing Tone | 0.04 | ~5.5% |

When `data/agent_ic.json` exists (from calibration), the pipeline uses the empirical ICs instead. A floor of 0.02 ensures no agent gets zero weight. The ML IC is computed at runtime from out-of-sample predictions.

---

## 📊 Percentile-Based Signal Assignment

After computing IC-weighted composite scores, the pipeline assigns buy/sell signals using **cross-sectional percentile ranks** rather than absolute thresholds. This is critical because IC-weighted composites compress scores into narrow ranges (typically ±0.30), making fixed thresholds unreliable across different universe sizes.

| Percentile | Signal | Condition |
|------------|--------|-----------|
| Top 10% | `strong_buy` | Composite > 0 |
| Top 25% | `buy` | Composite > 0 |
| Middle 50% | `hold` | — |
| Bottom 25% | `sell` | Composite < 0 |
| Bottom 10% | `strong_sell` | Composite < 0 |

The **sign guard** (composite > 0 for buys, < 0 for sells) prevents a stock from receiving a buy signal purely because it's "less bad" than peers — it must have a genuinely positive composite score.

This approach is standard practice in quantitative finance (see Grinold & Kahn, 2000) and automatically adapts to any universe size without manual threshold tuning.

---

## 🤖 LLM Review Agent

**File:** `src/auto_researcher/agents/llm_review_agent.py` | **Flag:** `--llm-review`

An optional post-ranking narrative layer using a frontier LLM (GPT-4o / Claude). The LLM review agent **never changes scores** — it adds qualitative commentary for human consumption.

**What It Does:**
1. **Stock Selection** — picks the highest-conviction buy and sell candidates by composite score. Falls back to top-N by composite when no clear buy/sell signals exist.
2. **Red-Team Critique** — for each selected stock, generates adversarial arguments against the ranking (e.g., "the momentum signal may be a dead-cat bounce because...")
3. **Conflict Narration** — when agents disagree sharply (e.g., fundamental = +0.8, sentiment = -0.6), narrates which signal is likely more informative given the current context
4. **Risk Flagging** — identifies hidden risks not captured by the quantitative agents (liquidity, regulatory, key-person dependency)

**Reflexion Rounds:** The agent runs 1–2 reflexion passes, critiquing its own initial review and refining it. This reduces overconfidence in LLM-generated commentary.

```bash
# Enable LLM review in the pipeline
python scripts/run_ranking_low_memory.py --universe sp100 --ml-top 25 --llm-review
```

---

## 🔬 Deep Research Agent

**File:** `src/auto_researcher/agents/deep_research_agent.py`

Integrates [GPT-Researcher](https://github.com/assafelovic/gpt-researcher) for comprehensive web-based research on individual stocks. Produces long-form markdown reports with cited sources.

**How It Works:**
1. Accepts a ticker and optional research question
2. GPT-Researcher conducts autonomous web search (via Tavily API)
3. Synthesizes findings into a structured research report
4. Returns markdown with source URLs

**Modes:**
- **Standard** — quick summary of key developments and risks
- **Deep** — multi-step research with follow-up queries, cross-referencing, and longer output

**Requirements:** `OPENAI_API_KEY` and `TAVILY_API_KEY` environment variables.

```python
from auto_researcher.agents.deep_research_agent import DeepResearchAgent

agent = DeepResearchAgent()
report = agent.research_stock_sync("NVDA", query="AI chip competition and margin outlook")
print(report)  # Markdown report with sources
```

---

## 🖥️ Streamlit Dashboard

**File:** `app.py` | **Launch:** `streamlit run app.py`

A full web-based interface for running the ranking pipeline interactively without touching the CLI.

### Features

- **Sidebar Controls** — universe selection (S&P 100/500/custom), ML top-K, agent weight overrides, skip toggles (ML, sentiment, etc.)
- **Subprocess Architecture** — pipeline runs in a separate process (`scripts/run_pipeline_subprocess.py`) to avoid blocking the Streamlit event loop. Progress is polled from a JSON file.
- **Live Progress** — real-time progress bar and status messages as stocks are processed
- **Results Tabs** — ranking table with color-coded signals, per-stock agent score breakdowns, composite score distributions
- **LLM Review & Deep Research** — optional post-pipeline steps triggered from the dashboard (runs inline after subprocess completes)
- **Dark Mode** — native dark theme via `.streamlit/config.toml`

### Quick Launch

```bash
# Set Python path and run
$env:PYTHONPATH = "src"  # Windows PowerShell
streamlit run app.py

# Or with custom port
streamlit run app.py --server.port 8502
```

The dashboard writes results to `data/ranking_results/` and the Streamlit config (`.streamlit/config.toml`) disables file watching to prevent reruns when the pipeline writes output files.

---

## � Performance Tracking

**File:** `src/auto_researcher/performance_tracker.py`

Measure the realised returns of any previous pipeline run by comparing stock prices from the run date to the present (or any user-specified end date). Available in the dashboard's **📊 Performance** tab.

### How It Works

1. **Load a previous run** — select any `final_ranking_*.json` from the dropdown in the sidebar.
2. **Date extraction** — the run date is parsed automatically from the filename (e.g. `final_ranking_sp100_20260203_0856.json` → 2026-02-03).
3. **Price fetching** — yfinance downloads daily closes for every ranked stock + SPY from the first trading day after the run through the end date.
4. **Return computation** — total return, excess return vs SPY, and signal correctness (buy → positive return, sell → negative return, hold → not counted).

### Metrics Displayed

| Metric | Description |
|--------|-------------|
| **Portfolio Return** | Equal-weight return across all ranked stocks |
| **SPY Return** | S&P 500 benchmark over the same window |
| **Alpha vs SPY** | Portfolio return − SPY return |
| **Signal Hit Rate** | Fraction of buy/sell signals that were directionally correct |
| **Avg Return by Signal** | Mean return grouped by signal bucket (strong_buy, buy, hold, sell, strong_sell) |
| **Best / Worst Performer** | Highlighted with ticker, return %, and signal |
| **Composite vs Realised Scatter** | Spearman rank correlation between composite score and realised return |

### Usage (Dashboard)

1. Open the Streamlit dashboard (`streamlit run app.py`)
2. Expand **"Load Previous Results"** in the sidebar
3. Select a historical run from the dropdown and click **✅ Load Selected Run**
4. Navigate to the **📊 Performance** tab
5. Adjust the date range if needed, then click **📊 Compute Performance**
6. Download the per-stock results as CSV via the **📥 Download** button

---

## �🛡️ Risk Management

### Position Sizing Methods

```python
from auto_researcher.risk import PositionSizer, PositionSizingMethod

sizer = PositionSizer(
    method=PositionSizingMethod.VOLATILITY_TARGET,
    target_volatility=0.15,  # 15% portfolio vol target
    limits=PositionLimit(
        max_position_pct=0.10,  # Max 10% single position
        max_sector_pct=0.30,    # Max 30% sector concentration
    )
)

positions = sizer.size_portfolio(
    tickers=["AAPL", "MSFT", "GOOGL"],
    signals=[0.8, 0.6, 0.4],
    prices={"AAPL": 150, "MSFT": 350, "GOOGL": 140},
    portfolio_value=1_000_000
)
```

**Available Methods:**
| Method | Description | Use Case |
|--------|-------------|----------|
| `EQUAL_WEIGHT` | 1/N allocation | Baseline, diversification |
| `KELLY` | Full Kelly criterion | Aggressive, high conviction |
| `FRACTIONAL_KELLY` | Kelly × fraction (default 0.25) | Reduced variance |
| `VOLATILITY_TARGET` | Inverse-vol weighting scaled to target | Risk-controlled |
| `EQUAL_RISK` | Equal marginal risk contribution | Risk parity |

### Drawdown Circuit Breakers

```python
from auto_researcher.risk import DrawdownController

controller = DrawdownController(
    initial_value=1_000_000,
    max_drawdown=0.20
)

# Update daily
state = controller.update(portfolio_value=950_000)

if controller.is_trading_halted():
    print(f"Trading halted: {state.active_circuit_breakers}")
    print(f"Target exposure: {controller.get_target_exposure():.0%}")
```

**Default Circuit Breakers:**
| Trigger | Threshold | Action |
|---------|-----------|--------|
| Daily Loss | -2% | Alert |
| Daily Loss | -3% | Halt new positions |
| Daily Loss | -5% | Reduce to 50% exposure |
| Peak-to-Trough | -10% | Reduce to 75% exposure |
| Peak-to-Trough | -15% | Close losing positions |
| Peak-to-Trough | -20% | Full de-leverage to 25% |

### Exposure Monitoring

```python
from auto_researcher.risk import ExposureLimitChecker, LimitType

checker = ExposureLimitChecker()

# Pre-trade compliance check
allowed, violations = checker.check_proposed_trade(
    current_positions={"AAPL": 0.08, "MSFT": 0.06},
    trade_ticker="AAPL",
    trade_weight_change=0.05,  # Would exceed 10% limit
    sectors={"AAPL": "Technology", "MSFT": "Technology"}
)

if not allowed:
    print("Trade blocked:", violations[0].recommended_action)
```

---

## 🏗️ Project Structure

```
auto_researcher/
├── src/auto_researcher/
│   ├── models/                         # 18 Alpha Models
│   │   ├── xgb_ranking_model.py        # XGBoost ranking (Stage 1 ML)
│   │   ├── gbdt_model.py              # LightGBM ranking (alternative)
│   │   ├── quality_value.py            # Quality-Value composite (fundamental agent)
│   │   ├── pead_enhanced.py            # Enhanced PEAD (earnings agent)
│   │   ├── insider_cluster.py          # Insider cluster detection (insider agent)
│   │   ├── early_adopter.py            # Tech pioneer detection (thematic agent)
│   │   ├── sector_momentum.py          # Sector rotation signals (momentum agent)
│   │   ├── topic_sentiment.py          # Topic-classified news sentiment
│   │   ├── earnings_topic_model.py     # Earnings-focused topic model
│   │   ├── filing_tone.py              # 10-K tone change (filing tone agent)
│   │   ├── earnings_call_qual.py       # Earnings call qualitative (6 sub-signals)
│   │   ├── sector_rotation_overlay.py  # Breadth-divergence sector overlay
│   │   ├── emerging_tech.py            # Emerging technology adoption signals
│   │   ├── earnings_tech_signal.py     # Technology signals from transcripts
│   │   ├── filing_tech_signal.py       # Innovation signals from filings
│   │   ├── earnings_return_model.py    # Post-earnings return prediction
│   │   ├── patent_innovation.py        # Patent-based innovation signals
│   │   ├── regimes.py                  # Regime-aware ML training/inference
│   │   ├── factor_rotation.py          # Leading-indicator regime-aware factor rotation
│   │   └── fundamentals_alpha.py       # (disabled — forward bias detected)
│   │
│   ├── agents/                         # 12 Analysis Agents
│   │   ├── fundamental_agent.py        # Quality-Value analysis with sector benchmarks
│   │   ├── earnings_agent.py           # Earnings call transcript analysis (LLM)
│   │   ├── insider_trading_agent.py    # SEC Form 4 insider trading patterns
│   │   ├── thematic_agent.py           # Moat, themes, sector rotation, early adopter
│   │   ├── sentiment_agent.py          # News sentiment (LLM + FinBERT + topic)
│   │   ├── sec_filing_agent.py         # 10-K/10-Q/8-K EDGAR analysis
│   │   ├── finbert_sentiment.py        # FinBERT financial sentiment (VADER fallback)
│   │   ├── model_agent.py              # GBDT model lifecycle management
│   │   ├── feature_agent.py            # Feature engineering orchestration
│   │   ├── backtest_agent.py           # Backtesting and portfolio construction
│   │   ├── llm_review_agent.py          # LLM red-team review (optional post-ranking)
│   │   ├── deep_research_agent.py      # GPT-Researcher deep web research
│   │   ├── orchestrator.py             # Research orchestrator v1
│   │   └── research_orchestrator.py    # Research orchestrator v2 (parallel agents)
│   │
│   ├── data/                           # Data Layer
│   │   ├── news_vectorstore.py         # ChromaDB RAG for news (210K+ articles)
│   │   ├── transcript_vectorstore.py   # ChromaDB RAG for earnings transcripts
│   │   ├── news_scraper.py             # Business Insider scraper → SQLite
│   │   ├── price_loader.py             # yfinance with disk caching
│   │   ├── universe.py                 # Ticker universe management
│   │   ├── fundamentals_sources.py     # FMP + Alpha Vantage fundamentals
│   │   ├── finagg_fundamentals.py      # SEC EDGAR via finagg
│   │   └── defeatbeta.py              # HuggingFace financial data
│   │
│   ├── performance_tracker.py           # Realised return tracking for past runs
│   │
│   ├── risk/                           # Risk Management Suite
│   │   ├── position_sizing.py          # Kelly, vol-target, equal-risk
│   │   ├── exposure_limits.py          # Concentration limits
│   │   ├── drawdown_control.py         # Circuit breakers
│   │   └── risk_attribution.py         # Factor decomposition, MCTR
│   │
│   ├── features/                       # Feature Engineering
│   │   ├── technical.py                # Momentum, volatility, reversal
│   │   ├── fundamentals.py             # Value, quality, growth factors
│   │   └── feature_pipeline.py         # Feature orchestration
│   │
│   ├── backtest/                       # Backtesting Framework
│   │   ├── runner.py                   # Walk-forward backtest
│   │   ├── portfolio.py                # Portfolio construction
│   │   └── metrics.py                  # IC, Sharpe, drawdown
│   │
│   ├── attribution/                    # Performance Attribution
│   ├── audit/                          # Forward-bias audit tools
│   ├── validation/                     # Pandera data validation schemas
│   ├── cli/                            # CLI entry point
│   └── config.py                       # Centralized configuration
│
├── scripts/                            # Pipeline & Analysis Scripts
│   ├── run_ranking_low_memory.py       # Main 3-stage ranking pipeline
│   ├── run_pipeline_subprocess.py      # Subprocess runner for Streamlit
│   ├── calibrate_ic_weights.py         # IC calibration for agent weights
│   ├── backtest_*.py                   # Various backtesting scripts
│   └── ...                             # 70+ analysis/debug scripts
│
├── tests/                              # 22+ test files
│   ├── test_factor_rotation.py         # Factor rotation model tests (35 tests)
│   ├── test_sector_rotation_overlay.py
│   ├── test_transcript_vectorstore.py
│   ├── test_risk.py
│   ├── test_technical_features.py
│   └── ...
│
├── data/                               # Data Storage
│   ├── news_chroma/                    # ChromaDB news vector store
│   ├── transcript_chroma/              # ChromaDB transcript vector store
│   ├── ranking_results/                # Pipeline output (JSON + reports)
│   ├── agent_ic.json                   # Calibrated IC weights
│   └── price_cache/                    # yfinance price cache
│
├── app.py                              # Streamlit web dashboard
├── .streamlit/config.toml              # Streamlit server config
├── .github/workflows/                  # CI/CD
├── Dockerfile                          # Multi-stage build
├── docker-compose.yml                  # Full stack (Postgres + Redis)
├── pyproject.toml                      # Project configuration
├── requirements.lock                   # Pinned dependencies
└── mypy.ini                            # Type checking config
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourorg/auto_researcher.git
cd auto_researcher

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install with all dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, FMP, Polygon, etc.)
```

### Docker Deployment

```bash
# Start full stack (app + PostgreSQL + Redis)
docker-compose up -d

# Development mode with Jupyter
docker-compose --profile dev up -d

# View logs
docker-compose logs -f app
```

### Running the Ranking Pipeline

```bash
# Set Python path
$env:PYTHONPATH = "src"  # Windows PowerShell
export PYTHONPATH=src     # Linux/Mac

# Full pipeline: S&P 100 universe, top 25 from ML, display top 10
python scripts/run_ranking_low_memory.py --universe sp100 --ml-top 25 --final-top 10

# With verbose agent output
python scripts/run_ranking_low_memory.py --universe sp100 --ml-top 25 -v

# Skip ML (reuse previous screening)
python scripts/run_ranking_low_memory.py --skip-ml --final-top 15

# With LLM review enabled
python scripts/run_ranking_low_memory.py --universe sp100 --ml-top 25 --llm-review
```

### Web Dashboard (Streamlit)

```bash
# Launch the interactive dashboard
$env:PYTHONPATH = "src"  # Windows PowerShell
streamlit run app.py
```

The dashboard provides the same pipeline functionality with a point-and-click interface — see [Streamlit Dashboard](#-streamlit-dashboard) for details.

### Running Individual Agents

```python
from auto_researcher.agents import ThematicAnalysisAgent
from auto_researcher.models import EarlyAdopterModel, EnhancedPEADModel

# Thematic analysis with early adopter signals
agent = ThematicAnalysisAgent()
result = agent.analyze_ticker("NVDA")

print(f"Forward Score: {result.forward_score}")
print(f"Pioneer Score: {result.ea_pioneer_score}")
print(f"Moat Rating: {result.moat_rating}")

# PEAD signal
pead = EnhancedPEADModel()
signal = pead.get_signal("AAPL")

if signal.is_actionable:
    print(f"Direction: {signal.direction}")
    print(f"Hold Period: {signal.recommended_days}d")
```

### Running IC Calibration

```bash
# Calibrate agent IC weights (updates data/agent_ic.json)
python scripts/calibrate_ic_weights.py --verbose
```

### Using the Transcript Vector Store

```bash
# Build transcript index
python -m auto_researcher.data.transcript_vectorstore build \
    --rebuild --tickers AAPL,MSFT,NVDA --min-year 2023

# Query transcripts
python -m auto_researcher.data.transcript_vectorstore query \
    --ticker AAPL --query "AI revenue growth" --n 5

# Cross-company theme search
python -m auto_researcher.data.transcript_vectorstore theme \
    --theme "tariff impact supply chain" --n 5
```

### Running Backtests

```bash
# Full alpha validation
python run_oos_alpha_validation.py --universe large_cap --top-k 5

# Topic sentiment backtest
python scripts/backtest_topic_sentiment.py

# Early adopter IC calculation
python scripts/ic_backtest_v2.py
```

---

## Configuration

### Environment Variables

```bash
# API Keys (required for full functionality)
OPENAI_API_KEY=sk-...           # LLM-based agent analysis
ANTHROPIC_API_KEY=sk-ant-...    # Alternative LLM provider
FMP_API_KEY=...                 # Financial Modeling Prep (transcripts, fundamentals)
POLYGON_API_KEY=...             # Polygon.io (news, prices)
ALPHAVANTAGE_API_KEY=...        # Alpha Vantage (fundamentals fallback)
TAVILY_API_KEY=tvly-...         # Tavily (deep research web search)
SEC_API_USER_AGENT=...          # SEC EDGAR (format: "Name email@example.com")

# Risk Settings (optional overrides)
MAX_POSITION_SIZE=0.10          # 10% max single position
MAX_SECTOR_EXPOSURE=0.30        # 30% max sector
MAX_DRAWDOWN=0.20               # 20% circuit breaker
TARGET_VOLATILITY=0.15          # 15% portfolio vol target
```

### Model Configuration

Edit `src/auto_researcher/config.py` for model-level settings:

```python
@dataclass(frozen=True)
class EnhancedModelConfig:
    model_type: str = "regression"        # regression | rank_pairwise | rank_ndcg
    target_mode: str = "vol_norm"         # vol_norm | raw | rank | ortho
    rolling_window: int = 504             # ~2 years training window
    include_short_reversal: bool = True
    include_residual_momentum: bool = True
    include_idio_vol: bool = True
    use_shap_selection: bool = False       # SHAP-based feature selection
```

### Using Fundamentals Data

The framework supports multiple sources for fundamental data:

#### Option 1: yfinance (Default)

Used by the fundamental agent for real-time quality-value analysis. No API key required.

#### Option 2: Financial Modeling Prep (FMP)

Primary source for earnings transcripts and detailed fundamentals. Requires `FMP_API_KEY`.

#### Option 3: finagg (SEC EDGAR)

Direct access to SEC EDGAR quarterly filings:

```bash
pip install -e ".[finagg]"
export SEC_API_USER_AGENT="YourName your.email@example.com"

python scripts/run_large_cap_backtest.py \
    --use-fundamentals --fundamentals-source finagg --finagg-mode quarterly_refined
```

| Mode | Description |
|------|-------------|
| `quarterly_refined` | Local SQL database (requires `finagg sec install`) |
| `quarterly_api` | Direct SEC API (no local DB) |
| `annual_refined` | Annual data from local SQL database |

---

## 📚 Academic Foundations

| Model | Key Papers |
|-------|-----------|
| **PEAD** | Ball & Brown (1968), Bernard & Thomas (1989) |
| **Insider Trading** | Lakonishok & Lee (2001), Cohen, Malloy & Pomorski (2012) |
| **Topic Sentiment** | Garcia (2013), Boudoukh et al. (2019), Loughran & McDonald (2011) |
| **Sector Momentum** | Moskowitz & Grinblatt (1999), Hong, Torous & Valkanov (2007) |
| **Factor Models** | Fama & French (1993, 2015), Asness et al. (2019), Novy-Marx (2013) |
| **Filing Tone** | Loughran & McDonald (2011), Li (2010) |
| **Earnings Call Analysis** | Hobson et al. (2012), Price et al. (2012) |
| **Sector Rotation Overlay** | Chan, Jegadeesh & Lakonishok (1996), Kakushadze & Yu (2017) |
| **Regime Detection** | Hamilton (1989), Ang & Bekaert (2002) |
| **Factor Rotation** | Asness, Moskowitz & Pedersen (2013), Arnott et al. (2016), Bender et al. (2018) |
| **Signal Assignment** | Grinold & Kahn (2000) — cross-sectional percentile ranking |

---

## 🧪 Testing

```bash
# Run all tests (22+ test files)
pytest tests/ -v

# Run specific test suite
pytest tests/test_factor_rotation.py -v
pytest tests/test_sector_rotation_overlay.py -v
pytest tests/test_transcript_vectorstore.py -v

# With coverage
pytest tests/ --cov=src/auto_researcher --cov-report=html

# Type checking
mypy src/

# Linting & formatting
ruff check src/ tests/
black src/ tests/
```

---

## 🔗 Data Sources

| Source | Data Type | API Key Required |
|--------|-----------|------------------|
| [yfinance](https://pypi.org/project/yfinance/) | Prices, fundamentals, earnings dates | No |
| [Financial Modeling Prep](https://financialmodelingprep.com/) | Earnings transcripts, fundamentals | Yes |
| [Polygon.io](https://polygon.io/) | News, tick data | Yes |
| [Alpha Vantage](https://www.alphavantage.co/) | Fundamentals (fallback) | Yes |
| [SEC EDGAR](https://www.sec.gov/edgar) | 10-K/10-Q/8-K filings (via finagg) | No |
| [Business Insider](https://www.businessinsider.com/) | News articles (scraper) | No |
| [ChromaDB](https://www.trychroma.com/) | Local vector stores (news + transcripts) | No |
| [HuggingFace](https://huggingface.co/) | FinBERT model, sentence-transformers | No |
| [DefeatBeta (HuggingFace)](https://huggingface.co/datasets/DefeatBeta/) | 223K+ earnings call transcripts (parquet) | No |
| [Tavily](https://tavily.com/) | Web search for deep research agent | Yes |

---

## 📊 Model Performance & Alpha Validation

The model has been rigorously tested to verify genuine alpha generation beyond simple strategies.

### Out-of-Sample Holdout

| Period | Sharpe | Ann. Return | Max Drawdown |
|--------|--------|-------------|--------------|
| **Train (2016-2022)** | 1.44 | +31.5% | -21.9% |
| **Test OOS (2023-2025)** | **3.53** | +45.3% | -1.3% |

The model's Sharpe ratio *improved* out-of-sample (1.44 → 3.53), indicating **no overfitting**.

### Baseline Comparisons

| Strategy | Sharpe | Ann. Return | Max DD | Excess vs SPY |
|----------|--------|-------------|--------|---------------|
| **ML Model** | **1.54** | 31.2% | -23.0% | +16.4% |
| Equal-Weight | 1.33 | 23.1% | -30.7% | +8.3% |
| Momentum Top-5 | 1.29 | 33.4% | -32.6% | +18.6% |
| Random Top-5 | 1.30 | 22.6% | -31.0% | +7.8% |
| Buy-Hold SPY | 0.83 | 14.8% | -33.7% | — |

**Key Findings:**
- ML beats all baselines on risk-adjusted returns (Sharpe)
- +0.24 Sharpe vs random selection (proves ranking skill)
- Lower drawdowns (-23% vs -31% to -34% for baselines)
- Higher hit rate (59.4% of months beat SPY)

### Running Validation Tests

```bash
python run_oos_alpha_validation.py --universe large_cap --top-k 5
python run_oos_alpha_validation.py --universe sp100 --top-k 10
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
