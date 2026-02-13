"""
Subprocess pipeline runner for Streamlit.

Runs the full ranking pipeline (Stages 1-3) in a separate process,
writing progress and results to JSON files that the Streamlit app polls.

Usage (called by app.py via subprocess):
    python scripts/run_pipeline_subprocess.py --universe sp100 --ml-top 25 \
        --ml-weight 0.35 --batch-size 5 --verbose \
        --skip-ml --skip-agents \
        --progress-file data/ranking_results/_progress.json \
        --results-file data/ranking_results/_final.json
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Silence chromadb telemetry before any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
import warnings
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
warnings.filterwarnings("ignore", message=".*Pandas4Warning.*")

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ranking_low_memory import (
    run_ml_screening,
    load_ml_results,
    save_results,
    compute_composite_scores,
    add_context_bands,
    add_conflict_resolution,
    add_evidence_budget,
    add_consistency_checks,
    add_lightweight_cross_validation,
    add_risk_qualifiers,
    apply_freshness_gates,
    analyze_single_stock_lightweight,
    clear_memory,
    StockRanking,
)

RESULTS_DIR = PROJECT_ROOT / "data" / "ranking_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_progress(progress_file: Path, data: dict):
    """Atomically write progress JSON (write to temp, then rename)."""
    tmp = progress_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    tmp.replace(progress_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="sp100")
    parser.add_argument("--ml-top", type=int, default=25)
    parser.add_argument("--ml-weight", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-agents", action="store_true")
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--results-file", required=True)
    args = parser.parse_args()

    progress_file = Path(args.progress_file)
    results_file = Path(args.results_file)

    log_lines = []

    def log(msg: str):
        log_lines.append(f"{datetime.now().strftime('%H:%M:%S')} — {msg}")
        # Keep only last 50
        if len(log_lines) > 50:
            log_lines.pop(0)

    def update_progress(stage: str, pct: int, status: str, **extra):
        write_progress(progress_file, {
            "stage": stage,
            "pct": pct,
            "status": status,
            "log": log_lines[-30:],
            "timestamp": datetime.now().isoformat(),
            **extra,
        })

    try:
        # ── Stage 1: ML Screening ─────────────────────────────────────
        log("**Stage 1: ML Screening** started")
        update_progress("ml_screening", 5, "running")

        ml_results_file = RESULTS_DIR / f"ml_screening_{args.universe}_{args.ml_top}.json"
        agent_results_file = RESULTS_DIR / f"agent_analysis_{args.universe}_{args.ml_top}.json"
        ml_ic = 0.15

        if args.skip_ml and ml_results_file.exists():
            rankings = load_ml_results(ml_results_file)
            if rankings and hasattr(rankings[0], "ml_ic"):
                ml_ic = rankings[0].ml_ic
            log(f"Loaded cached ML results ({len(rankings)} stocks)")
        else:
            t0 = time.time()
            rankings, ml_ic = run_ml_screening(universe=args.universe, top_k=args.ml_top)
            save_results(rankings, ml_results_file)
            log(f"ML screening complete: {len(rankings)} candidates in {time.time()-t0:.1f}s (IC={ml_ic:.3f})")

        clear_memory()
        update_progress("ml_screening", 25, "complete",
                        ml_data=[asdict(r) for r in rankings])

        # ── Stage 2: Agent Analysis ───────────────────────────────────
        log("**Stage 2: Agent Analysis** started")
        update_progress("agent_analysis", 30, "running")

        if args.skip_agents and agent_results_file.exists():
            rankings = load_ml_results(agent_results_file)
            log(f"Loaded cached agent results ({len(rankings)} stocks)")
        else:
            total_stocks = len(rankings)
            for i, stock in enumerate(rankings, 1):
                pct = 30 + int(45 * i / total_stocks)
                log(f"Analyzing **{stock.ticker}**... ({i}/{total_stocks})")
                update_progress("agent_analysis", pct, "running",
                                current_ticker=stock.ticker,
                                current_idx=i, total=total_stocks)

                try:
                    results = analyze_single_stock_lightweight(
                        ticker=stock.ticker,
                        ml_percentile=stock.ml_percentile,
                        verbose=args.verbose,
                    )
                    stock.sentiment_score = results["sentiment_score"]
                    stock.fundamental_score = results["fundamental_score"]
                    stock.earnings_score = results["earnings_score"]
                    stock.insider_score = results["insider_score"]
                    stock.thematic_score = results["thematic_score"]
                    stock.momentum_score = results["momentum_score"]
                    stock.filing_tone_score = results["filing_tone_score"]
                    stock.earnings_call_qual_score = results["earnings_call_qual_score"]
                    stock.company_name = results["company_name"]
                    stock.sector = results["sector"]
                    stock.errors = results["errors"]
                    stock.agent_rationales = results.get("agent_rationales", {})
                    log(f"✓ {stock.ticker}: sent={results['sentiment_score']:.3f} fund={results['fundamental_score']:.3f}")
                except Exception as e:
                    log(f"⚠️ {stock.ticker} failed: {e}")
                    stock.errors.append(str(e))

                if i % args.batch_size == 0:
                    clear_memory()

        # Post-processing enrichments
        log("Running post-processing enrichments...")
        try:
            add_context_bands(rankings)
            add_conflict_resolution(rankings)
            add_evidence_budget(rankings)
            add_consistency_checks(rankings)
            add_lightweight_cross_validation(rankings)
            add_risk_qualifiers(rankings)
            apply_freshness_gates(rankings)
        except Exception as e:
            log(f"⚠️ Post-processing error: {e}")

        save_results(rankings, agent_results_file)
        clear_memory()
        log("Agent analysis & enrichments complete")
        update_progress("agent_analysis", 75, "complete",
                        agent_data=[asdict(r) for r in rankings])

        # ── Stage 3: Composite Scoring ────────────────────────────────
        log("**Stage 3: Composite Scoring** started")
        update_progress("composite", 78, "running")

        try:
            rankings = compute_composite_scores(
                rankings, ml_weight=args.ml_weight, verbose=args.verbose, ml_ic=ml_ic
            )
        except Exception as e:
            log(f"⚠️ Composite scoring error: {e}")

        final_out = RESULTS_DIR / f"final_ranking_{args.universe}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        save_results(rankings, final_out)

        log(f"Composite scoring complete — {len(rankings)} stocks ranked")
        log("**Pipeline complete!** ✅")

        # Write final results
        update_progress("done", 100, "complete",
                        ml_data=[asdict(r) for r in rankings],
                        agent_data=[asdict(r) for r in rankings],
                        final_data=[asdict(r) for r in rankings],
                        final_file=str(final_out))

    except Exception as e:
        log(f"💥 FATAL: {e}")
        log(traceback.format_exc())
        update_progress("error", 0, "error",
                        error=str(e),
                        traceback=traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
