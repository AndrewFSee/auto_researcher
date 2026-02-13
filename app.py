"""
Auto Researcher — Stock Ranking Dashboard.

Streamlit UI for the multi-agent stock ranking pipeline.

Usage:
    streamlit run app.py
"""

import asyncio
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

# Disable ChromaDB telemetry before any chromadb import (posthog compat issue)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Silence noisy third-party warnings & errors
import logging
import warnings
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
warnings.filterwarnings("ignore", message=".*Pandas4Warning.*")

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — same as the CLI pipeline
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline functions (lazy — deferred until needed)
RESULTS_DIR = PROJECT_ROOT / "data" / "ranking_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Auto Researcher",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Signal badges */
    .signal-strong_buy { background: #15803d; color: white; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
    .signal-buy { background: #22c55e; color: white; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
    .signal-hold { background: #a3a3a3; color: white; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
    .signal-sell { background: #ef4444; color: white; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
    .signal-strong_sell { background: #991b1b; color: white; padding: 2px 10px; border-radius: 12px; font-weight: 600; }

    /* Confidence badges */
    .conf-high { color: #15803d; font-weight: 600; }
    .conf-medium { color: #ca8a04; font-weight: 600; }
    .conf-low { color: #dc2626; font-weight: 600; }

    /* Agent score cell colouring */
    .score-pos { color: #15803d; }
    .score-neg { color: #dc2626; }
    .score-zero { color: #737373; }

    /* Section headers */
    .stage-header {
        background: linear-gradient(90deg, #1e3a5f, #2563eb);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 12px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    div[data-testid="stMetric"] {
        background: rgba(30, 58, 95, 0.35);
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    div[data-testid="stMetric"] label { color: rgba(255, 255, 255, 0.7) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def signal_badge(signal: str) -> str:
    """Render a signal as a coloured HTML badge."""
    label = signal.replace("_", " ").upper()
    return f'<span class="signal-{signal}">{label}</span>'


def score_color(val: float) -> str:
    """Return CSS class for a score value."""
    if val > 0.05:
        return "score-pos"
    if val < -0.05:
        return "score-neg"
    return "score-zero"


def format_score(val: float) -> str:
    """Format a score as a coloured HTML span."""
    cls = score_color(val)
    return f'<span class="{cls}">{val:+.3f}</span>'


def load_cached_results(filepath: Path) -> list[dict] | None:
    """Load JSON results if file exists."""
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def rankings_to_df(data: list[dict]) -> pd.DataFrame:
    """Convert list of ranking dicts to a clean DataFrame."""
    df = pd.DataFrame(data)
    cols_order = [
        "final_rank", "ticker", "company_name", "sector", "signal",
        "composite_score", "ml_percentile",
        "sentiment_score", "fundamental_score", "earnings_score",
        "insider_score", "thematic_score", "momentum_score",
        "filing_tone_score", "earnings_call_qual_score",
        "composite_confidence",
    ]
    available = [c for c in cols_order if c in df.columns]
    return df[available].copy()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("Auto Researcher")
    st.caption("Multi-Agent Stock Ranking Pipeline")
    st.divider()

    # --- Universe & Scope ---
    st.subheader("🎯 Universe & Scope")
    universe = st.selectbox(
        "Stock Universe",
        ["sp100", "sp500", "large_cap", "core_tech"],
        index=0,
        help="Which stock universe to screen",
        key="sidebar_universe",
    )
    ml_top = st.slider("ML Top-K", 5, 100, 25, 5, help="Stocks to pass from ML screening to agents", key="sidebar_ml_top")
    final_top = st.slider("Display Top-N", 5, 50, 10, 5, help="Top N stocks to display in results", key="sidebar_final_top")

    # --- Weights ---
    st.subheader("⚖️ Weights")
    ml_weight = st.slider("ML Weight Override", 0.0, 1.0, 0.35, 0.05, help="Override ML weight in composite (0 = auto IC-proportional)", key="sidebar_ml_weight")
    
    # --- Execution ---
    st.subheader("⚡ Execution")
    batch_size = st.slider("Memory Batch Size", 1, 20, 5, 1, help="Clear memory every N stocks", key="sidebar_batch_size")
    verbose = st.checkbox("Verbose Logging", value=False, key="sidebar_verbose")
    
    col_skip1, col_skip2 = st.columns(2)
    with col_skip1:
        skip_ml = st.checkbox("Skip ML", help="Load ML results from previous run", key="sidebar_skip_ml")
    with col_skip2:
        skip_agents = st.checkbox("Skip Agents", help="Load agent results from previous run", key="sidebar_skip_agents")

    # --- Deep Research ---
    st.subheader("🔍 Deep Research (GPT-Researcher)")
    deep_research = st.checkbox("Enable Deep Research", value=False, key="sidebar_deep_research")
    if deep_research:
        deep_research_type = st.selectbox(
            "Research Depth",
            ["research_report", "deep"],
            format_func=lambda x: "Standard (~1 min)" if x == "research_report" else "Deep Recursive (~5 min)",
            help="Standard does web research. Deep uses recursive tree exploration (higher cost).",
            key="sidebar_deep_research_type",
        )
        deep_research_tickers = st.text_input(
            "Tickers to Research (comma-separated, or blank for top-N)",
            placeholder="AAPL, MSFT, NVDA",
            help="Leave blank to auto-research your top-ranked stocks",
            key="sidebar_deep_research_tickers",
        )
        deep_research_top_n = st.slider("Auto-Research Top-N", 1, 10, 3, help="If no tickers specified, research the top N ranked stocks", key="sidebar_deep_research_top_n")
    else:
        deep_research_type = "research_report"
        deep_research_tickers = ""
        deep_research_top_n = 3

    # --- LLM Review ---
    st.subheader("🤖 LLM Red-Team Review")
    llm_review = st.checkbox("Enable LLM Review", value=False, key="sidebar_llm_review")
    if llm_review:
        review_model = st.selectbox(
            "Review Model",
            [
                "o3",
                "o4-mini",
                "gpt-5.2",
                "gpt-4.1",
                "gpt-4o",
                "claude-opus-4",
                "claude-sonnet-4",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
            ],
            index=0,
            help="Reasoning models (o3, o4-mini) recommended for adversarial review",
            key="sidebar_review_model",
        )
        review_top_n = st.slider("Stocks to Review", 1, 15, 5, key="sidebar_review_top_n")
        reflexion_rounds = st.slider("Reflexion Rounds", 0, 3, 1, help="Self-critique iterations", key="sidebar_reflexion_rounds")
    else:
        review_model = "o3"
        review_top_n = 5
        reflexion_rounds = 1

    st.divider()

    # --- Run Button ---
    run_pipeline = st.button("🚀 Run Pipeline", use_container_width=True, type="primary", key="sidebar_run_pipeline")

    # --- Load Previous ---
    st.caption("— or —")
    load_previous = st.button("📂 Load Previous Results", use_container_width=True, key="sidebar_load_previous")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════

# Title
st.markdown("# 📊 Stock Ranking Dashboard")

# ───────────────────────────────────────────────────────────────────────────
# Load Previous Results
# ───────────────────────────────────────────────────────────────────────────
if load_previous:
    # Find all available final ranking files
    final_files = sorted(RESULTS_DIR.glob("final_ranking_*.json"), reverse=True)

    if final_files:
        # Store the available files so the selector persists
        st.session_state["_available_runs"] = final_files
    else:
        st.warning("No previous results found. Run the pipeline first.")

# Show run selector if we have available runs (persists across reruns)
if st.session_state.get("_available_runs") and not st.session_state.get("pipeline_running"):
    final_files = st.session_state["_available_runs"]

    # Build human-readable labels
    def _run_label(f: Path) -> str:
        import re as _re
        m = _re.search(r"ranking_(\w+)_(\d{8})_(\d{4})", f.name)
        if m:
            univ, date_s, time_s = m.group(1), m.group(2), m.group(3)
            date_fmt = f"{date_s[:4]}-{date_s[4:6]}-{date_s[6:]}"
            time_fmt = f"{time_s[:2]}:{time_s[2:]}"
            return f"{univ.upper()} — {date_fmt} {time_fmt}"
        return f.stem

    selected_run = st.selectbox(
        "Select a previous run",
        final_files,
        format_func=_run_label,
        key="run_selector",
    )

    if st.button("✅ Load Selected Run", key="confirm_load_run"):
        final_data = load_cached_results(selected_run)

        if final_data:
            # Try to find matching ML/agent files
            import re as _re
            m = _re.search(r"ranking_(\w+)_\d{8}", selected_run.name)
            univ_tag = m.group(1) if m else universe
            ml_results_file = RESULTS_DIR / f"ml_screening_{univ_tag}_{ml_top}.json"
            agent_results_file = RESULTS_DIR / f"agent_analysis_{univ_tag}_{ml_top}.json"

            st.session_state["ml_data"] = load_cached_results(ml_results_file)
            st.session_state["agent_data"] = load_cached_results(agent_results_file)
            st.session_state["final_data"] = final_data
            st.session_state["review_data"] = None
            st.session_state["deep_research_data"] = None
            st.session_state["pipeline_ran"] = True
            st.session_state["run_filename"] = selected_run.name
            st.session_state["_available_runs"] = None  # Hide selector

            # Check if reviews are embedded
            if final_data and any(
                "llm_review" in (r.get("agent_rationales") or {}) for r in final_data
            ):
                st.session_state["review_data"] = [
                    r["agent_rationales"]["llm_review"]
                    for r in final_data
                    if "llm_review" in (r.get("agent_rationales") or {})
                ]

            st.success(f"Loaded previous results ({len(final_data)} stocks) from {_run_label(selected_run)}")
            st.rerun()
        else:
            st.warning("Failed to load the selected results file.")


# ───────────────────────────────────────────────────────────────────────────
# Run Pipeline  (subprocess-based — immune to Streamlit reruns)
# ───────────────────────────────────────────────────────────────────────────
_PROGRESS_FILE = RESULTS_DIR / "_pipeline_progress.json"

def _read_progress() -> dict | None:
    """Read the subprocess progress file (returns None if missing/corrupt)."""
    try:
        if _PROGRESS_FILE.exists():
            return json.loads(_PROGRESS_FILE.read_text())
    except Exception:
        pass
    return None

# Launch pipeline subprocess when button is clicked
if run_pipeline:
    import subprocess as _sp

    # Clean up any stale progress file
    if _PROGRESS_FILE.exists():
        _PROGRESS_FILE.unlink()

    # Build command
    cmd = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        str(PROJECT_ROOT / "scripts" / "run_pipeline_subprocess.py"),
        "--universe", universe,
        "--ml-top", str(ml_top),
        "--ml-weight", str(ml_weight),
        "--batch-size", str(batch_size),
        "--progress-file", str(_PROGRESS_FILE),
        "--results-file", str(RESULTS_DIR / "_final.json"),
    ]
    if verbose:
        cmd.append("--verbose")
    if skip_ml:
        cmd.append("--skip-ml")
    if skip_agents:
        cmd.append("--skip-agents")

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "ANONYMIZED_TELEMETRY": "False"}
    proc = _sp.Popen(cmd, env=env, cwd=str(PROJECT_ROOT),
                     stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    st.session_state["pipeline_pid"] = proc.pid
    st.session_state["pipeline_running"] = True
    st.session_state["pipeline_ran"] = False
    time.sleep(1)
    st.rerun()

# Poll running pipeline for progress
if st.session_state.get("pipeline_running"):
    progress_bar = st.progress(0, text="Initializing pipeline...")
    status_area = st.empty()
    log_expander = st.expander("📋 Pipeline Log", expanded=True)
    log_area = log_expander.empty()

    prog = _read_progress()

    if prog is None:
        # Subprocess hasn't written progress yet — wait and rerun
        status_area.info("⏳ Waiting for pipeline subprocess to start...")
        time.sleep(2)
        st.rerun()

    elif prog.get("status") == "error":
        progress_bar.progress(100, text="Pipeline failed!")
        status_area.error(f"💥 Pipeline error: {prog.get('error', 'unknown')}")
        if prog.get("log"):
            log_area.markdown("\n\n".join(f"`{l}`" for l in prog["log"]))
        st.session_state["pipeline_running"] = False
        if _PROGRESS_FILE.exists():
            _PROGRESS_FILE.unlink()

    elif prog.get("stage") == "done":
        progress_bar.progress(100, text="Pipeline complete!")
        status_area.success(f"✅ Pipeline complete!")
        if prog.get("log"):
            log_area.markdown("\n\n".join(f"`{l}`" for l in prog["log"]))

        # Store results in session state
        st.session_state["ml_data"] = prog.get("ml_data")
        st.session_state["agent_data"] = prog.get("agent_data")
        st.session_state["final_data"] = prog.get("final_data")
        st.session_state["review_data"] = None
        st.session_state["deep_research_data"] = None
        st.session_state["pipeline_running"] = False
        st.session_state["pipeline_ran"] = True
        st.session_state["run_filename"] = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M')}"
        st.session_state["perf_summary"] = None  # Reset performance cache

        if _PROGRESS_FILE.exists():
            _PROGRESS_FILE.unlink()

        # Optional Stage 4: LLM Review (fast, safe to run inline)
        if llm_review:
            status_area.info(f"🧠 Running LLM review ({review_model})...")
            try:
                from scripts.run_ranking_low_memory import load_ml_results, StockRanking
                from dataclasses import asdict as _asdict

                # Load rankings from final_data
                final_data = st.session_state.get("final_data", [])
                rankings = [StockRanking(**d) for d in final_data]

                from auto_researcher.agents.llm_review_agent import (
                    LLMReviewAgent,
                    LLMReviewConfig,
                )
                review_config = LLMReviewConfig(
                    model=review_model,
                    top_n_review=review_top_n,
                    reflexion_rounds=reflexion_rounds,
                )
                review_agent = LLMReviewAgent(config=review_config)
                review_summary = review_agent.review_rankings(rankings, verbose=verbose)
                review_agent.inject_reviews_into_rankings(rankings, review_summary)

                st.session_state["final_data"] = [_asdict(r) for r in rankings]
                st.session_state["review_data"] = [r.to_dict() for r in review_summary.reviews]
                del review_agent
                gc.collect()
            except ImportError:
                st.warning("⚠️ litellm not installed — skipping LLM review")
            except Exception as e:
                st.warning(f"⚠️ LLM review failed: {e}")

        # Optional Stage 5: Deep Research (GPT-Researcher)
        if deep_research:
            status_area.info(f"🔍 Running deep research ({deep_research_type})...")
            try:
                from auto_researcher.agents.deep_research_agent import (
                    DeepResearchAgent,
                    DeepResearchConfig,
                )
                dr_config = DeepResearchConfig(
                    report_type=deep_research_type,
                    verbose=verbose,
                )
                dr_agent = DeepResearchAgent(config=dr_config)

                # Determine which tickers to research
                final_data = st.session_state.get("final_data", [])
                if deep_research_tickers.strip():
                    dr_tickers = [t.strip().upper() for t in deep_research_tickers.split(",") if t.strip()]
                    dr_stocks = [
                        s for s in final_data if s["ticker"] in dr_tickers
                    ] or [{"ticker": t} for t in dr_tickers]
                else:
                    # Auto-select top-N ranked stocks
                    dr_stocks = sorted(
                        final_data,
                        key=lambda s: s.get("composite_score", 0),
                        reverse=True,
                    )[:deep_research_top_n]

                dr_summary = dr_agent.research_batch_sync(dr_stocks, max_concurrent=2)
                st.session_state["deep_research_data"] = [
                    {
                        "ticker": r.ticker,
                        "report": r.report,
                        "sources": r.sources,
                        "costs": r.costs,
                        "latency_ms": r.latency_ms,
                        "error": r.error,
                    }
                    for r in dr_summary.results
                ]
                del dr_agent
                gc.collect()
            except ImportError:
                st.warning("⚠️ gpt-researcher not installed — skipping deep research")
            except Exception as e:
                st.warning(f"⚠️ Deep research failed: {e}")

        status_area.success(f"✅ Pipeline complete!")

    else:
        # Still running — show progress and poll again
        pct = prog.get("pct", 0)
        stage = prog.get("stage", "unknown")
        stage_labels = {
            "ml_screening": "Stage 1: ML Screening",
            "agent_analysis": "Stage 2: Agent Analysis",
            "composite": "Stage 3: Composite Scoring",
        }
        label = stage_labels.get(stage, stage)

        current_ticker = prog.get("current_ticker", "")
        current_idx = prog.get("current_idx", 0)
        total = prog.get("total", 0)
        if current_ticker and total:
            label = f"Stage 2: Analyzing {current_ticker} ({current_idx}/{total})"

        progress_bar.progress(min(pct, 99), text=f"{label}...")
        status_area.info(f"🔄 {label}")

        if prog.get("log"):
            log_area.markdown("\n\n".join(f"`{l}`" for l in prog["log"]))

        time.sleep(3)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY (tabs)
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.get("pipeline_ran"):
    ml_data = st.session_state.get("ml_data")
    agent_data = st.session_state.get("agent_data")
    final_data = st.session_state.get("final_data")
    review_data = st.session_state.get("review_data")
    deep_research_data = st.session_state.get("deep_research_data")

    tab_names = ["📈 Final Rankings", "🔬 ML Screening", "🤖 Agent Scores", "📋 Stock Details"]
    if review_data:
        tab_names.append("🧠 LLM Review")
    if deep_research_data:
        tab_names.append("🔍 Deep Research")
    # Performance tab available when we have a run filename (loaded previous results)
    run_filename = st.session_state.get("run_filename")
    tab_names.append("📊 Performance")
    tab_names.append("📄 Raw Data")

    tabs = st.tabs(tab_names)

    # ── Tab: Final Rankings ───────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="stage-header">📈 Final Rankings — IC-Weighted Composite</div>', unsafe_allow_html=True)

        if final_data:
            df = rankings_to_df(final_data[:final_top])

            # Summary metrics
            signal_counts = df["signal"].value_counts().to_dict() if "signal" in df.columns else {}
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Ranked", len(final_data))
            col2.metric("Strong Buy", signal_counts.get("strong_buy", 0))
            col3.metric("Buy", signal_counts.get("buy", 0))
            col4.metric("Hold", signal_counts.get("hold", 0))
            col5.metric("Sell / Strong Sell", signal_counts.get("sell", 0) + signal_counts.get("strong_sell", 0))

            st.divider()

            # Rankings table
            for _, row in df.iterrows():
                signal = row.get("signal", "hold")
                composite = row.get("composite_score", 0)
                ticker = row.get("ticker", "???")
                company = row.get("company_name", "")
                rank = int(row.get("final_rank", 0))

                col_rank, col_info, col_signal, col_score = st.columns([1, 4, 2, 2])
                with col_rank:
                    st.markdown(f"### #{rank}")
                with col_info:
                    st.markdown(f"**{ticker}** — {company}")
                    st.caption(f"Sector: {row.get('sector', 'N/A')}")
                with col_signal:
                    st.markdown(signal_badge(signal), unsafe_allow_html=True)
                with col_score:
                    st.metric("Composite", f"{composite:+.3f}")

            # Bar chart
            st.divider()
            st.subheader("Score Distribution")
            chart_df = pd.DataFrame(final_data[:final_top])[["ticker", "composite_score"]].set_index("ticker")
            st.bar_chart(chart_df, color=["#2563eb"])
        else:
            st.info("No final ranking data available.")

    # ── Tab: ML Screening ─────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="stage-header">🔬 Stage 1: ML Screening (XGBoost)</div>', unsafe_allow_html=True)

        if ml_data:
            ml_df = pd.DataFrame(ml_data)[["ml_rank", "ticker", "ml_score", "ml_percentile"]].copy()
            ml_df.columns = ["Rank", "Ticker", "ML Score", "Percentile"]
            ml_df = ml_df.sort_values("Rank")

            col1, col2, col3 = st.columns(3)
            col1.metric("Stocks Screened", len(ml_df))
            col2.metric("ML IC", f"{ml_data[0].get('ml_ic', 0.15):.3f}" if ml_data else "N/A")
            col3.metric("Top Score", f"{ml_df['ML Score'].max():.4f}")

            st.divider()
            st.dataframe(
                ml_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ML Score": st.column_config.NumberColumn(format="%.4f"),
                    "Percentile": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
            )
        else:
            st.info("No ML screening data available.")

    # ── Tab: Agent Scores ─────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="stage-header">🤖 Stage 2: Multi-Agent Analysis</div>', unsafe_allow_html=True)

        source = final_data or agent_data
        if source:
            agent_cols = [
                "ticker", "sentiment_score", "fundamental_score", "earnings_score",
                "insider_score", "thematic_score", "momentum_score",
                "filing_tone_score", "earnings_call_qual_score",
            ]
            available = [c for c in agent_cols if c in source[0]]
            agent_df = pd.DataFrame(source)[available].copy()
            agent_df.columns = [
                c.replace("_score", "").replace("_", " ").title() for c in available
            ]

            # Heatmap-style table
            st.dataframe(
                agent_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    col: st.column_config.NumberColumn(format="%+.3f")
                    for col in agent_df.columns
                    if col != "Ticker"
                },
            )

            # Radar-ish breakdown for selected stock
            st.divider()
            st.subheader("Individual Stock Breakdown")
            tickers = [s.get("ticker", "???") for s in source]
            selected_ticker = st.selectbox("Select stock", tickers, key="agent_selected_ticker")

            if selected_ticker:
                stock = next((s for s in source if s.get("ticker") == selected_ticker), None)
                if stock:
                    score_fields = [
                        ("ML (normalized)", (stock.get("ml_percentile", 50) - 50) / 50),
                        ("Sentiment", stock.get("sentiment_score", 0)),
                        ("Fundamental", stock.get("fundamental_score", 0)),
                        ("Earnings/PEAD", stock.get("earnings_score", 0)),
                        ("Insider", stock.get("insider_score", 0)),
                        ("Thematic", stock.get("thematic_score", 0)),
                        ("Momentum", stock.get("momentum_score", 0)),
                        ("Filing Tone", stock.get("filing_tone_score", 0)),
                        ("Call Quality", stock.get("earnings_call_qual_score", 0)),
                    ]
                    breakdown_df = pd.DataFrame(score_fields, columns=["Agent", "Score"])
                    st.bar_chart(breakdown_df.set_index("Agent"), color=["#2563eb"], horizontal=True)
        else:
            st.info("No agent data available.")

    # ── Tab: Stock Details ────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="stage-header">📋 Stock Details & Agent Rationales</div>', unsafe_allow_html=True)

        source = final_data or agent_data
        if source:
            tickers = [s.get("ticker", "???") for s in source]
            detail_ticker = st.selectbox("Select stock for detailed view", tickers, key="detail_ticker")

            stock = next((s for s in source if s.get("ticker") == detail_ticker), None)
            if stock:
                # Header
                signal = stock.get("signal", "hold")
                st.markdown(f"## {detail_ticker} — {stock.get('company_name', '')}")
                st.markdown(
                    f"{signal_badge(signal)} &nbsp; Composite: **{stock.get('composite_score', 0):+.3f}** "
                    f"| Rank: **#{stock.get('final_rank', 'N/A')}** "
                    f"| Sector: {stock.get('sector', 'N/A')} "
                    f"| Confidence: {stock.get('composite_confidence', 0):.2f}",
                    unsafe_allow_html=True,
                )

                st.divider()
                rationales = stock.get("agent_rationales", {})

                # Conflict resolution
                conflict = rationales.get("conflict_resolution", {})
                if conflict:
                    summary = conflict.get("summary", "No material disagreement.")
                    bullish = conflict.get("bullish_agents", [])
                    bearish = conflict.get("bearish_agents", [])
                    if bullish or bearish:
                        st.warning(f"⚔️ **Conflict:** {summary}")
                    else:
                        st.success(f"✅ {summary}")

                # Key factors
                key_factors = stock.get("key_factors", [])
                if key_factors:
                    st.markdown("**Key ML Factors:** " + " • ".join(key_factors))

                # Agent rationales
                st.subheader("Agent Rationales")
                agent_configs = [
                    ("Fundamental", "fundamental", "📊"),
                    ("Sentiment", "sentiment", "📰"),
                    ("Earnings / PEAD", "earnings", "💰"),
                    ("Insider", "insider", "👤"),
                    ("Thematic", "thematic", "🔮"),
                    ("Momentum", "momentum", "🚀"),
                    ("Filing Tone", "filing_tone", "📄"),
                    ("Earnings Call Quality", "earnings_call_qual", "🎤"),
                ]

                for label, key, icon in agent_configs:
                    data = rationales.get(key, {})
                    score_attr = f"{key}_score"
                    score_val = stock.get(score_attr, 0.0)

                    with st.expander(f"{icon} **{label}** — Score: {score_val:+.3f}", expanded=False):
                        if data.get("missing"):
                            st.caption(f"No data — {data.get('reason', 'unknown')}")
                        else:
                            # Evidence
                            evidence = data.get("evidence", [])
                            if evidence:
                                for e in evidence:
                                    st.markdown(f"- {e}")

                            # Summary
                            summary = data.get("summary", "")
                            if summary:
                                st.markdown(f"**Summary:** {summary}")

                            # Strengths / Concerns (fundamental)
                            strengths = data.get("strengths", [])
                            concerns = data.get("concerns", [])
                            if strengths:
                                st.markdown("**Strengths:** " + ", ".join(str(s) for s in strengths[:4]))
                            if concerns:
                                st.markdown("**Concerns:** " + ", ".join(str(c) for c in concerns[:4]))

                            # Raw data toggle
                            if st.checkbox(f"Show raw {key} data", key=f"raw_{detail_ticker}_{key}"):
                                st.json(data)

                # Errors
                errors = stock.get("errors", [])
                if errors:
                    st.error("**Errors during analysis:**")
                    for err in errors:
                        st.markdown(f"- `{err}`")
        else:
            st.info("No data available.")

    # ── Tab: LLM Review ───────────────────────────────────────────────
    if review_data:
        with tabs[4]:
            st.markdown('<div class="stage-header">🧠 Stage 4: LLM Red-Team Review</div>', unsafe_allow_html=True)

            for review in review_data:
                ticker = review.get("ticker", "???")
                signal = review.get("signal", "hold")
                composite = review.get("composite_score", 0)
                confidence = review.get("confidence_in_signal", "MEDIUM")
                review_text = review.get("review_text", "")
                risk_flags = review.get("risk_flags", [])
                reflexion = review.get("reflexion_applied", False)

                conf_color = {"HIGH": "conf-high", "MEDIUM": "conf-medium", "LOW": "conf-low"}.get(confidence, "")
                conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")

                st.markdown(
                    f"### {ticker} — {signal_badge(signal)} &nbsp; "
                    f"Composite: {composite:+.3f} &nbsp; "
                    f"Signal Confidence: {conf_emoji} "
                    f'<span class="{conf_color}">{confidence}</span>',
                    unsafe_allow_html=True,
                )

                if reflexion:
                    st.caption("🔄 Reflexion-refined review")

                st.markdown(review_text)

                if risk_flags:
                    flag_str = " • ".join(f"`{f}`" for f in risk_flags)
                    st.warning(f"**Risk Flags:** {flag_str}")

                st.divider()

    # ── Tab: Deep Research ────────────────────────────────────────────
    if deep_research_data:
        dr_tab_idx = 5 if review_data else 4
        with tabs[dr_tab_idx]:
            st.markdown('<div class="stage-header">🔍 Deep Stock Research Reports</div>', unsafe_allow_html=True)

            # Cost summary
            total_cost = sum(r.get("costs", 0) for r in deep_research_data)
            total_time = sum(r.get("latency_ms", 0) for r in deep_research_data) / 1000
            col1, col2, col3 = st.columns(3)
            col1.metric("Stocks Researched", len(deep_research_data))
            col2.metric("Total Cost", f"${total_cost:.2f}")
            col3.metric("Total Time", f"{total_time:.0f}s")

            st.divider()

            for dr in deep_research_data:
                ticker = dr.get("ticker", "???")
                report = dr.get("report", "")
                sources = dr.get("sources", [])
                error = dr.get("error")
                cost = dr.get("costs", 0)

                with st.expander(f"📄 **{ticker}** — {'❌ Error' if error else f'{len(sources)} sources, ${cost:.2f}'}", expanded=len(deep_research_data) == 1):
                    if error:
                        st.error(f"Research failed: {error}")
                    elif report:
                        st.markdown(report)

                        if sources:
                            st.divider()
                            st.caption(f"**Sources ({len(sources)}):**")
                            for i, url in enumerate(sources[:20], 1):
                                st.caption(f"{i}. {url}")

                        # Download individual report
                        st.download_button(
                            f"⬇️ Download {ticker} Report",
                            report,
                            file_name=f"deep_research_{ticker}_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown",
                            key=f"dl_dr_{ticker}",
                        )
                    else:
                        st.info("No report generated.")

    # ── Tab: Performance Tracking ────────────────────────────────────
    perf_tab_idx = len(tab_names) - 2  # second-to-last (before Raw Data)
    with tabs[perf_tab_idx]:
        st.markdown('<div class="stage-header">📊 Performance Tracking — Realised Returns</div>', unsafe_allow_html=True)

        if not final_data:
            st.info("No ranking data available to compute performance.")
        else:
            # Determine run date from filename or let user override
            run_fname = st.session_state.get("run_filename", "")
            auto_date = None
            if run_fname:
                import re as _re
                m = _re.search(r"(\d{8})", run_fname)
                if m:
                    d = m.group(1)
                    auto_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"

            col_date1, col_date2 = st.columns(2)
            with col_date1:
                run_date_input = st.date_input(
                    "Run Date (pipeline execution date)",
                    value=datetime.strptime(auto_date, "%Y-%m-%d").date() if auto_date else datetime.now().date() - timedelta(days=7),
                    key="perf_run_date",
                )
            with col_date2:
                end_date_input = st.date_input(
                    "End Date (measure returns through)",
                    value=datetime.now().date(),
                    key="perf_end_date",
                )

            if st.button("📊 Compute Performance", type="primary", key="compute_perf_btn"):
                with st.spinner("Fetching prices and computing returns..."):
                    try:
                        from auto_researcher.performance_tracker import compute_performance

                        summary = compute_performance(
                            final_data=final_data,
                            run_date_override=run_date_input.strftime("%Y-%m-%d"),
                            end_date=end_date_input.strftime("%Y-%m-%d"),
                        )
                        st.session_state["perf_summary"] = summary
                    except Exception as e:
                        st.error(f"Performance computation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            summary = st.session_state.get("perf_summary")
            if summary:
                # ── Summary Metrics ──
                st.divider()
                st.subheader("Summary")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(
                    "Portfolio Return",
                    f"{summary.portfolio_return_pct:+.2f}%" if summary.portfolio_return_pct is not None else "N/A",
                )
                m2.metric(
                    "SPY Return",
                    f"{summary.spy_return_pct:+.2f}%" if summary.spy_return_pct is not None else "N/A",
                )
                m3.metric(
                    "Alpha vs SPY",
                    f"{summary.alpha_pct:+.2f}%" if summary.alpha_pct is not None else "N/A",
                    delta=f"{summary.alpha_pct:+.2f}%" if summary.alpha_pct is not None else None,
                    delta_color="normal",
                )
                m4.metric(
                    "Signal Hit Rate",
                    f"{summary.hit_rate:.1%}" if summary.hit_rate is not None else "N/A",
                    help=f"{summary.correct_count}/{summary.total_actionable} actionable signals correct",
                )
                m5.metric(
                    "Trading Days",
                    f"{summary.trading_days}",
                    help=f"{summary.calendar_days} calendar days",
                )

                # ── Average Return by Signal ──
                if summary.avg_return_by_signal:
                    st.divider()
                    st.subheader("Average Return by Signal")

                    signal_order = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
                    sig_data = []
                    for sig in signal_order:
                        if sig in summary.avg_return_by_signal:
                            count = len([sp for sp in summary.stocks if sp.signal == sig])
                            sig_data.append({
                                "Signal": sig.replace("_", " ").upper(),
                                "Avg Return %": round(summary.avg_return_by_signal[sig], 2),
                                "Median Return %": round(summary.median_return_by_signal.get(sig, 0), 2),
                                "Count": count,
                            })

                    if sig_data:
                        sig_df = pd.DataFrame(sig_data)
                        st.dataframe(sig_df, use_container_width=True, hide_index=True)

                        # Bar chart of avg return by signal
                        chart_sig_df = sig_df.set_index("Signal")[["Avg Return %"]]
                        st.bar_chart(chart_sig_df, color=["#2563eb"])

                # ── Best & Worst Performers ──
                if summary.best_performer or summary.worst_performer:
                    st.divider()
                    col_best, col_worst = st.columns(2)
                    with col_best:
                        if summary.best_performer:
                            bp = summary.best_performer
                            st.success(
                                f"🏆 **Best:** {bp.ticker} ({bp.company_name}) — "
                                f"{bp.total_return_pct:+.2f}% "
                                f"(Signal: {bp.signal.replace('_', ' ').upper()})"
                            )
                    with col_worst:
                        if summary.worst_performer:
                            wp = summary.worst_performer
                            st.error(
                                f"📉 **Worst:** {wp.ticker} ({wp.company_name}) — "
                                f"{wp.total_return_pct:+.2f}% "
                                f"(Signal: {wp.signal.replace('_', ' ').upper()})"
                            )

                # ── Per-Stock Returns Table ──
                st.divider()
                st.subheader("Per-Stock Returns")

                perf_rows = []
                for sp in summary.stocks:
                    row = {
                        "Rank": sp.final_rank,
                        "Ticker": sp.ticker,
                        "Company": sp.company_name,
                        "Signal": sp.signal.replace("_", " ").upper(),
                        "Composite": round(sp.composite_score, 3),
                        "Price (Run)": sp.price_at_run,
                        "Price (Now)": sp.price_current,
                        "Return %": sp.total_return_pct,
                        "vs SPY %": sp.excess_return_pct,
                        "Correct?": {True: "✅", False: "❌", None: "—"}.get(sp.signal_correct, "—"),
                    }
                    perf_rows.append(row)

                perf_df = pd.DataFrame(perf_rows)
                if not perf_df.empty:
                    # Sortable table
                    sort_col = st.selectbox(
                        "Sort by",
                        ["Return %", "vs SPY %", "Rank", "Composite"],
                        key="perf_sort_col",
                    )
                    ascending = sort_col == "Rank"
                    perf_df_sorted = perf_df.sort_values(sort_col, ascending=ascending, na_position="last")

                    st.dataframe(
                        perf_df_sorted,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Return %": st.column_config.NumberColumn(format="%+.2f"),
                            "vs SPY %": st.column_config.NumberColumn(format="%+.2f"),
                            "Composite": st.column_config.NumberColumn(format="%+.3f"),
                            "Price (Run)": st.column_config.NumberColumn(format="$%.2f"),
                            "Price (Now)": st.column_config.NumberColumn(format="$%.2f"),
                        },
                    )

                    # Return distribution chart
                    st.divider()
                    st.subheader("Return Distribution")
                    chart_perf = perf_df_sorted[perf_df_sorted["Return %"].notna()][["Ticker", "Return %"]].copy()
                    chart_perf = chart_perf.set_index("Ticker")
                    st.bar_chart(chart_perf, color=["#2563eb"])

                    # Scatter: composite score vs actual return
                    st.divider()
                    st.subheader("Composite Score vs Realised Return")
                    scatter_df = perf_df[perf_df["Return %"].notna()][["Composite", "Return %"]].copy()
                    if len(scatter_df) > 2:
                        st.scatter_chart(scatter_df, x="Composite", y="Return %", color=["#2563eb"])

                        # Rank correlation
                        try:
                            from scipy import stats as _stats
                            corr, pval = _stats.spearmanr(scatter_df["Composite"], scatter_df["Return %"])
                            st.caption(
                                f"Spearman rank correlation: **{corr:.3f}** (p={pval:.4f}) — "
                                + ("statistically significant" if pval < 0.05 else "not significant")
                            )
                        except ImportError:
                            pass

                    # Download performance report
                    st.divider()
                    st.download_button(
                        "⬇️ Download Performance Report (CSV)",
                        perf_df_sorted.to_csv(index=False),
                        file_name=f"performance_{summary.run_date}_to_{summary.end_date}.csv",
                        mime="text/csv",
                        key="dl_perf_csv",
                    )

    # ── Tab: Raw Data ─────────────────────────────────────────────────
    raw_tab_idx = len(tab_names) - 1
    with tabs[raw_tab_idx]:
        st.markdown('<div class="stage-header">📄 Raw JSON Data</div>', unsafe_allow_html=True)

        raw_source = st.radio("Select data source", ["Final Rankings", "Agent Analysis", "ML Screening"], horizontal=True, key="raw_data_source")

        if raw_source == "Final Rankings" and final_data:
            st.json(final_data[:final_top])
        elif raw_source == "Agent Analysis" and agent_data:
            st.json(agent_data[:final_top])
        elif raw_source == "ML Screening" and ml_data:
            st.json(ml_data)
        else:
            st.info("No data available for this source.")

        # Download buttons
        st.divider()
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        if final_data:
            col_dl1.download_button(
                "⬇️ Download Final Rankings",
                json.dumps(final_data, indent=2),
                file_name=f"final_ranking_{universe}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )
        if agent_data:
            col_dl2.download_button(
                "⬇️ Download Agent Data",
                json.dumps(agent_data, indent=2),
                file_name=f"agent_analysis_{universe}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )
        if ml_data:
            col_dl3.download_button(
                "⬇️ Download ML Screening",
                json.dumps(ml_data, indent=2),
                file_name=f"ml_screening_{universe}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )

else:
    # ── Welcome screen ────────────────────────────────────────────────
    st.markdown("""
    Welcome to the **Auto Researcher** stock ranking dashboard!
    
    This tool runs a multi-stage pipeline to rank stocks:
    
    | Stage | Description |
    |-------|-------------|
    | **1. ML Screening** | XGBoost model screens the universe and selects top candidates |
    | **2. Agent Analysis** | 8 specialized agents score each stock on different dimensions |
    | **3. Composite Scoring** | IC-weighted combination produces final rankings |
    | **4. LLM Review** *(optional)* | Frontier reasoning model red-teams the top/bottom picks |
    | **5. Deep Research** *(optional)* | GPT-Researcher generates comprehensive web-based research reports |
    
    Configure options in the **sidebar** and click **🚀 Run Pipeline** to begin,  
    or click **📂 Load Previous Results** to view cached results and track performance.
    
    💡 **Tip:** Load a previous run and go to the **📊 Performance** tab to see realised returns for each stock since the pipeline ran.
    """)

    # Show available cached results
    cached_files = list(RESULTS_DIR.glob("final_ranking_*.json"))
    if cached_files:
        st.subheader("📂 Available Previous Runs")
        for f in sorted(cached_files, reverse=True)[:5]:
            size_kb = f.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.caption(f"• `{f.name}` — {size_kb:.0f} KB — {mod_time}")
