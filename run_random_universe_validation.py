#!/usr/bin/env python
"""
Random Universe Validation Script for auto_researcher.

This script performs Monte Carlo validation of the stock-picking model
by running backtests on random subsets of S&P 500 tickers. This tests
whether the model generalizes beyond a specific hand-picked universe.

For each (universe_size, top_k) combination, we sample n_universes random
subsets and compute aggregate statistics to assess robustness.

Usage:
    python run_random_universe_validation.py
    python run_random_universe_validation.py --runs-per-config 50
"""

import argparse
import hashlib
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Fixed experiment parameters
START_DATE = "2016-01-01"
END_DATE = "2024-12-31"
BENCHMARK = "SPY"
HORIZON = 63
REBALANCE = "M"

# Core tickers to exclude from random sampling
CORE_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "SPY"]

# Monte Carlo parameters
UNIVERSE_SIZES = [20, 50, 100]
TOP_K_VALUES = [3, 5]
DEFAULT_RUNS_PER_CONFIG = 50  # Default number of random draws per (universe_size, top_k)
RNG_SEED = 42

# Output directory
RESULTS_DIR = Path("results")


# =============================================================================
# Fallback S&P 500 Ticker List (same as run_universe_scaling_experiments.py)
# =============================================================================

SP500_FALLBACK = [
    # Technology (excluding core)
    "AVGO", "ORCL", "CRM", "CSCO", "ACN", "ADBE", "IBM", "INTC", "AMD", "TXN",
    "QCOM", "AMAT", "ADI", "LRCX", "MU", "KLAC", "MCHP", "CDNS", "SNPS", "FTNT",
    "PANW", "NOW", "PLTR", "CRWD",
    # Communication Services (excluding core)
    "GOOGL", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO",
    "WBD", "PARA", "FOXA", "LYV", "OMC", "IPG",
    # Consumer Discretionary (excluding core)
    "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR", "ORLY",
    "AZO", "ROST", "CMG", "DHI", "LEN", "GM", "F", "APTV", "EBAY", "ETSY",
    "YUM", "DPZ", "HLT", "WYNN", "LVS", "RCL", "CCL", "EXPE",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "K", "HSY", "SJM", "CAG", "CPB", "MKC", "HRL", "TSN", "KHC",
    "STZ", "TAP", "EL", "CHD", "CLX", "KVUE",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD", "OXY",
    "HES", "DVN", "FANG", "HAL", "BKR", "KMI", "WMB", "OKE", "TRGP",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "USB",
    "PNC", "TFC", "COF", "BK", "STT", "FITB", "HBAN", "CFG", "RF", "KEY",
    "MTB", "NTRS", "ZION", "CMA", "AIG", "PRU", "MET", "AFL", "TRV", "ALL",
    "CB", "PGR", "AON", "MMC", "AJG", "CINF", "L", "WRB",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT", "SYK", "BSX", "EW", "ZBH",
    "BDX", "BAX", "IDXX", "IQV", "A", "MTD", "WAT", "HOLX", "ALGN", "DXCM",
    "BIIB", "MRNA", "CVS", "CI", "ELV", "HUM", "CNC", "MOH", "HCA",
    # Industrials
    "CAT", "DE", "UNP", "RTX", "HON", "BA", "GE", "LMT", "UPS", "MMM",
    "ETN", "ITW", "EMR", "ROK", "PH", "PCAR", "CMI", "ODFL", "CSX", "NSC",
    "FDX", "DAL", "UAL", "LUV", "AAL", "WM", "RSG", "GD", "NOC", "TDG",
    "TXT", "HWM", "IR", "DOV", "SWK", "FAST", "GWW", "CTAS", "PAYX", "ADP",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "STLD", "VMC",
    "MLM", "ALB", "PPG", "DOW", "LYB", "CF", "MOS", "FMC", "CE", "EMN",
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    "EQR", "VTR", "ARE", "ESS", "MAA", "UDR", "HST", "INVH", "KIM",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PEG", "ED",
    "WEC", "ES", "AWK", "DTE", "ETR", "FE", "PPL", "AEE", "CMS", "CNP",
    "EVRG", "NI", "ATO", "LNT", "NRG", "PNW",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a single backtest run.
    
    Attributes:
        universe_size: Size of the ticker universe.
        top_k: Number of top stocks held.
        run_index: Index of this run within the configuration.
        tickers: List of tickers in the universe.
        ann_ret: Annualized return (%).
        sharpe: Sharpe ratio.
        max_dd: Maximum drawdown (%).
        avg_ic: Average information coefficient.
        hit_rate: Hit rate (%).
        excess_ret: Excess return over benchmark (%).
        benchmark_ret: Benchmark return (%).
        error: Error message if run failed.
        is_degenerate: Whether this run is degenerate (likely all-cash).
        degenerate_reason: Reason for degenerate classification.
        n_rebalance_points: Number of rebalance dates (estimated from date range).
    """
    universe_size: int
    top_k: int
    run_index: int
    tickers: list[str]
    ann_ret: Optional[float] = None
    sharpe: Optional[float] = None
    max_dd: Optional[float] = None
    avg_ic: Optional[float] = None
    hit_rate: Optional[float] = None
    excess_ret: Optional[float] = None
    benchmark_ret: Optional[float] = None
    error: Optional[str] = None
    # Degenerate run detection fields
    is_degenerate: bool = False
    degenerate_reason: Optional[str] = None
    n_rebalance_points: int = 0


# =============================================================================
# Universe Pool Construction
# =============================================================================

def fetch_sp500_tickers_from_wikipedia() -> Optional[list[str]]:
    """
    Attempt to fetch S&P 500 tickers from Wikipedia.
    
    Returns:
        List of ticker symbols, or None if fetch fails.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        symbol_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker'
        tickers = df[symbol_col].tolist()
        # Clean up tickers (BRK.B -> BRK-B for yfinance compatibility)
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"  [OK] Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"  [WARN] Failed to fetch from Wikipedia: {e}")
        return None


def get_sp500_pool() -> list[str]:
    """
    Get S&P 500 tickers pool (excluding core tickers).
    
    Returns:
        List of S&P 500 ticker symbols available for random sampling.
    """
    print("\n[DATA] Loading S&P 500 ticker list...")
    
    # Try Wikipedia first
    tickers = fetch_sp500_tickers_from_wikipedia()
    
    if tickers is None:
        print(f"  [INFO] Using fallback list of {len(SP500_FALLBACK)} tickers")
        tickers = SP500_FALLBACK.copy()
    
    # Remove core tickers
    tickers = [t for t in tickers if t not in CORE_TICKERS]
    print(f"  [INFO] {len(tickers)} tickers available for sampling (excluding core)")
    
    # Filter tickers with sufficient data history
    print("\n[DATA] Validating ticker data availability...")
    valid_tickers = validate_ticker_data(tickers)
    print(f"  [INFO] {len(valid_tickers)} tickers have sufficient data history")
    
    return valid_tickers


def validate_ticker_data(tickers: list[str], min_coverage: float = 0.95) -> list[str]:
    """
    Validate that tickers have sufficient data history.
    
    Filters out delisted stocks, recent IPOs, and tickers with too many missing days.
    
    Args:
        tickers: List of ticker symbols to validate.
        min_coverage: Minimum fraction of trading days required (0.95 = 95% coverage).
        
    Returns:
        List of validated ticker symbols with sufficient data.
    """
    import yfinance as yf
    
    # Download price data for all tickers
    print(f"  [INFO] Downloading data for {len(tickers)} tickers to validate...")
    
    try:
        data = yf.download(
            tickers,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            threads=True,
        )
        
        # Get Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data[['Close']]
            prices.columns = tickers[:1]
        
        # Calculate coverage for each ticker
        total_days = len(prices)
        valid_tickers = []
        skipped_tickers = []
        
        for ticker in tickers:
            if ticker in prices.columns:
                coverage = prices[ticker].notna().sum() / total_days
                if coverage >= min_coverage:
                    valid_tickers.append(ticker)
                else:
                    skipped_tickers.append((ticker, coverage))
            else:
                skipped_tickers.append((ticker, 0.0))
        
        # Report skipped tickers
        if skipped_tickers:
            n_skipped = len(skipped_tickers)
            print(f"  [WARN] Skipped {n_skipped} tickers with insufficient data:")
            # Show first 10
            for ticker, cov in skipped_tickers[:10]:
                print(f"         {ticker}: {cov:.1%} coverage")
            if n_skipped > 10:
                print(f"         ... and {n_skipped - 10} more")
        
        return valid_tickers
        
    except Exception as e:
        print(f"  [ERROR] Failed to validate ticker data: {e}")
        print(f"  [INFO] Using unvalidated ticker list")
        return tickers


# =============================================================================
# Reproducible Random Sampling
# =============================================================================

def get_deterministic_seed(base_seed: int, universe_size: int, top_k: int, run_index: int) -> int:
    """
    Generate a deterministic seed for a specific (universe_size, top_k, run_index) combination.
    
    Uses a hash function to ensure reproducibility across runs.
    
    Args:
        base_seed: Base random seed.
        universe_size: Size of the universe.
        top_k: Top-K value.
        run_index: Index of the run within this configuration.
        
    Returns:
        Deterministic integer seed.
    """
    key = f"{base_seed}_{universe_size}_{top_k}_{run_index}"
    hash_bytes = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')


def sample_universe(pool: list[str], size: int, seed: int) -> list[str]:
    """
    Sample a random subset of tickers from the pool.
    
    Args:
        pool: Full list of available tickers.
        size: Number of tickers to sample.
        seed: Random seed for reproducibility.
        
    Returns:
        Sorted list of sampled ticker symbols.
    """
    rng = random.Random(seed)
    if size > len(pool):
        size = len(pool)
    sampled = rng.sample(pool, size)
    sampled.sort()  # Sort for consistency
    return sampled


# =============================================================================
# Metric Parsing
# =============================================================================

# =============================================================================
# Degenerate Run Detection
# =============================================================================

# Thresholds for degenerate run detection
DEGEN_ANN_RET_THRESHOLD = 0.5  # Abs annualized return below 0.5% is suspicious
DEGEN_BENCHMARK_THRESHOLD = 5.0  # Benchmark return above 5% annualized


def detect_degenerate_run(result: BacktestResult) -> BacktestResult:
    """
    Inspect result metrics and mark is_degenerate/degenerate_reason when
    the run looks like a 'do nothing / all cash / failed' backtest.
    
    Heuristics for degenerate runs:
    - Zero or near-zero annualized return while benchmark had positive returns
    - Zero Sharpe ratio with meaningful benchmark performance
    - Very large negative excess return combined with near-zero portfolio return
    
    Args:
        result: BacktestResult to inspect.
        
    Returns:
        The same BacktestResult with is_degenerate and degenerate_reason populated.
    """
    # Skip if there was an error (already flagged)
    if result.error is not None:
        return result
    
    # Skip if metrics are missing
    if result.ann_ret is None or result.sharpe is None:
        return result
    
    reasons = []
    
    # Check for near-zero return while benchmark was positive
    if abs(result.ann_ret) < DEGEN_ANN_RET_THRESHOLD:
        if result.benchmark_ret is not None and result.benchmark_ret > DEGEN_BENCHMARK_THRESHOLD:
            reasons.append(f"Near-zero return ({result.ann_ret:.2f}%) while benchmark up {result.benchmark_ret:.1f}%")
    
    # Check for zero Sharpe with meaningful benchmark performance
    if abs(result.sharpe) < 0.01:
        if result.benchmark_ret is not None and result.benchmark_ret > DEGEN_BENCHMARK_THRESHOLD:
            reasons.append(f"Zero Sharpe ({result.sharpe:.2f}) while benchmark positive")
    
    # Check for large negative excess return combined with near-zero portfolio return
    if result.excess_ret is not None and result.excess_ret < -15.0:
        if result.ann_ret is not None and abs(result.ann_ret) < DEGEN_ANN_RET_THRESHOLD:
            reasons.append(f"Large negative excess ({result.excess_ret:.1f}%) with near-zero portfolio return")
    
    # Set degenerate flag if any reason found
    if reasons:
        result.is_degenerate = True
        result.degenerate_reason = "; ".join(reasons)
    
    return result


# =============================================================================
# Metric Parsing
# =============================================================================

def parse_float_from_line(pattern: str, text: str) -> Optional[float]:
    """
    Extract a float value from text using a regex pattern.
    
    Args:
        pattern: Regex pattern with one capture group for the number.
        text: Text to search.
        
    Returns:
        Extracted float value, or None if not found.
    """
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None


def parse_metrics(stdout: str) -> dict[str, Optional[float]]:
    """
    Parse performance metrics from CLI stdout.
    
    Expected lines:
        "  Annualized Return:        23.02%"
        "  Sharpe Ratio:              0.66"
        "  Max Drawdown:            -46.69%"
        "  Average IC:              0.1352"
        "  Benchmark Return:         15.00%"
        "  Excess Return:             8.02%"
        "  Hit Rate:                 55.00%"
    
    Args:
        stdout: Full stdout from the CLI run.
        
    Returns:
        Dict with metric values (floats or None).
    """
    metrics = {}
    
    # Annualized Return: XX.XX%
    metrics["ann_ret"] = parse_float_from_line(
        r"Annualized Return:\s+([-\d.]+)%", stdout
    )
    
    # Sharpe Ratio: X.XX
    metrics["sharpe"] = parse_float_from_line(
        r"Sharpe Ratio:\s+([-\d.]+)", stdout
    )
    
    # Max Drawdown: -XX.XX%
    metrics["max_dd"] = parse_float_from_line(
        r"Max Drawdown:\s+([-\d.]+)%", stdout
    )
    
    # Average IC: X.XXXX
    metrics["avg_ic"] = parse_float_from_line(
        r"Average IC:\s+([-\d.]+)", stdout
    )
    
    # Benchmark Return: XX.XX%
    metrics["benchmark_ret"] = parse_float_from_line(
        r"Benchmark Return:\s+([-\d.]+)%", stdout
    )
    
    # Excess Return: XX.XX%
    metrics["excess_ret"] = parse_float_from_line(
        r"Excess Return:\s+([-\d.]+)%", stdout
    )
    
    # Hit Rate: XX.XX%
    metrics["hit_rate"] = parse_float_from_line(
        r"Hit Rate:\s+([-\d.]+)%", stdout
    )
    
    return metrics


# =============================================================================
# Backtest Runner
# =============================================================================

def build_command(tickers: list[str], top_k: int, use_enhanced: bool = False) -> list[str]:
    """
    Build the CLI command for a backtest.
    
    Args:
        tickers: List of ticker symbols for the universe.
        top_k: Number of top stocks to hold.
        use_enhanced: Whether to use the enhanced model.
        
    Returns:
        List of command arguments for subprocess.
    """
    cmd = [
        sys.executable, "-m", "auto_researcher.cli.main",
        "--tickers", *tickers,
        "--start-date", START_DATE,
        "--end-date", END_DATE,
        "--benchmark", BENCHMARK,
        "--horizon", str(HORIZON),
        "--top-k", str(top_k),
        "--rebalance", REBALANCE,
    ]
    
    if use_enhanced:
        cmd.append("--enhanced-model")
    
    return cmd


def run_single_backtest(
    tickers: list[str],
    top_k: int,
    universe_size: int,
    run_index: int,
    use_enhanced: bool = False,
) -> BacktestResult:
    """
    Run a single backtest for a given universe.
    
    Args:
        tickers: List of ticker symbols.
        top_k: Number of top stocks to hold.
        universe_size: Size of the universe (for result labeling).
        run_index: Index of this run (for result labeling).
        use_enhanced: Whether to use the enhanced model.
        
    Returns:
        BacktestResult with parsed metrics or error.
    """
    result = BacktestResult(
        universe_size=universe_size,
        top_k=top_k,
        run_index=run_index,
        tickers=tickers,
    )
    
    cmd = build_command(tickers, top_k, use_enhanced=use_enhanced)
    
    # Enhanced model takes much longer due to additional feature computation
    timeout_seconds = 1800 if use_enhanced else 600  # 30 min vs 10 min
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds,
        )
        
        # Parse metrics from stdout
        metrics = parse_metrics(proc.stdout)
        result.ann_ret = metrics.get("ann_ret")
        result.sharpe = metrics.get("sharpe")
        result.max_dd = metrics.get("max_dd")
        result.avg_ic = metrics.get("avg_ic")
        result.hit_rate = metrics.get("hit_rate")
        result.excess_ret = metrics.get("excess_ret")
        result.benchmark_ret = metrics.get("benchmark_ret")
        
    except subprocess.CalledProcessError as e:
        result.error = f"Exit code {e.returncode}"
    except subprocess.TimeoutExpired:
        result.error = f"Timeout ({timeout_seconds}s)"
    except Exception as e:
        result.error = str(e)
    
    # Estimate n_rebalance_points (monthly rebalance over the date range)
    # From 2016-01-01 to 2024-12-31 is about 9 years = ~108 monthly rebalances
    from datetime import datetime
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    months_diff = (end.year - start.year) * 12 + (end.month - start.month)
    result.n_rebalance_points = max(0, months_diff)
    
    # Detect degenerate run
    result = detect_degenerate_run(result)
    
    return result


# =============================================================================
# Monte Carlo Loop
# =============================================================================

def run_monte_carlo_validation(
    pool: list[str],
    runs_per_config: int,
    use_enhanced: bool = False,
) -> list[BacktestResult]:
    """
    Run the full Monte Carlo validation experiment.
    
    Args:
        pool: Pool of tickers available for sampling.
        runs_per_config: Number of random universes per (universe_size, top_k) config.
        use_enhanced: Whether to use the enhanced model.
        
    Returns:
        List of BacktestResult objects from all runs.
    """
    total_runs = len(UNIVERSE_SIZES) * len(TOP_K_VALUES) * runs_per_config
    model_label = "enhanced" if use_enhanced else "base"
    print(f"\n[RUN] Starting Monte Carlo validation ({model_label} model)")
    print(f"      Universe sizes: {UNIVERSE_SIZES}")
    print(f"      Top-K values:   {TOP_K_VALUES}")
    print(f"      Runs per config: {runs_per_config}")
    print(f"      Total runs:      {total_runs}")
    print()
    
    results = []
    run_count = 0
    
    for universe_size in UNIVERSE_SIZES:
        for top_k in TOP_K_VALUES:
            print(f"\n{'='*70}")
            print(f"  Universe Size: {universe_size} | Top-K: {top_k}")
            print(f"{'='*70}")
            
            for run_idx in range(runs_per_config):
                run_count += 1
                
                # Get deterministic seed for this run
                seed = get_deterministic_seed(RNG_SEED, universe_size, top_k, run_idx)
                
                # Sample universe
                tickers = sample_universe(pool, universe_size, seed)
                
                # Run backtest
                result = run_single_backtest(
                    tickers, top_k, universe_size, run_idx,
                    use_enhanced=use_enhanced,
                )
                results.append(result)
                
                # Print compact progress line
                if result.error:
                    print(f"  [{run_count:3d}/{total_runs}] U{universe_size}/K{top_k}/#{run_idx:02d}: ERROR - {result.error}")
                elif result.is_degenerate:
                    # Flag degenerate runs prominently
                    sharpe_str = f"{result.sharpe:.2f}" if result.sharpe is not None else "N/A"
                    ann_ret_str = f"{result.ann_ret:.1f}%" if result.ann_ret is not None else "N/A"
                    excess_str = f"{result.excess_ret:.1f}%" if result.excess_ret is not None else "N/A"
                    reason_short = result.degenerate_reason[:60] + "..." if len(result.degenerate_reason or "") > 60 else (result.degenerate_reason or "")
                    print(f"  [{run_count:3d}/{total_runs}] [DEGEN] U{universe_size}/K{top_k}/#{run_idx:02d}: "
                          f"Sharpe={sharpe_str}, AnnRet={ann_ret_str}, Excess={excess_str} ({reason_short})")
                else:
                    sharpe_str = f"{result.sharpe:.2f}" if result.sharpe is not None else "N/A"
                    ann_ret_str = f"{result.ann_ret:.1f}%" if result.ann_ret is not None else "N/A"
                    excess_str = f"{result.excess_ret:.1f}%" if result.excess_ret is not None else "N/A"
                    print(f"  [{run_count:3d}/{total_runs}] U{universe_size}/K{top_k}/#{run_idx:02d}: "
                          f"Sharpe={sharpe_str}, AnnRet={ann_ret_str}, Excess={excess_str}")
    
    return results


# =============================================================================
# Results Aggregation
# =============================================================================

def build_results_dataframe(results: list[BacktestResult]) -> pd.DataFrame:
    """
    Convert results to a pandas DataFrame.
    
    Args:
        results: List of BacktestResult objects.
        
    Returns:
        DataFrame with all per-run results.
    """
    rows = []
    for r in results:
        rows.append({
            "UniverseSize": r.universe_size,
            "TopK": r.top_k,
            "RunIndex": r.run_index,
            "Tickers": ",".join(r.tickers),
            "AnnRet": r.ann_ret,
            "Sharpe": r.sharpe,
            "MaxDD": r.max_dd,
            "AvgIC": r.avg_ic,
            "HitRate": r.hit_rate,
            "ExcessRet": r.excess_ret,
            "BenchmarkRet": r.benchmark_ret,
            "Error": r.error,
            # Degenerate run fields
            "IsDegenerate": r.is_degenerate,
            "DegenerateReason": r.degenerate_reason,
            "NRebalancePoints": r.n_rebalance_points,
        })
    
    return pd.DataFrame(rows)


def compute_summary_statistics(
    df: pd.DataFrame,
    exclude_degenerate: bool = False,
) -> pd.DataFrame:
    """
    Compute aggregate statistics for each (universe_size, top_k) combination.
    
    Args:
        df: Per-run results DataFrame.
        exclude_degenerate: If True, exclude degenerate runs from statistics.
        
    Returns:
        Summary DataFrame with aggregated statistics including min/max/percentiles.
    """
    # Filter out error rows
    valid = df[df["Error"].isna()].copy()
    
    # Optionally filter out degenerate runs
    if exclude_degenerate and "IsDegenerate" in valid.columns:
        valid = valid[~valid["IsDegenerate"]].copy()
    
    summary_rows = []
    
    for universe_size in UNIVERSE_SIZES:
        for top_k in TOP_K_VALUES:
            subset = valid[(valid["UniverseSize"] == universe_size) & (valid["TopK"] == top_k)]
            
            if len(subset) == 0:
                continue
            
            row = {
                # Core identifiers
                "UniverseSize": universe_size,
                "TopK": top_k,
                "Count": len(subset),
                # AnnRet stats
                "MeanAnnRet": subset["AnnRet"].mean(),
                "MedianAnnRet": subset["AnnRet"].median(),
                # Sharpe stats (extended)
                "MeanSharpe": subset["Sharpe"].mean(),
                "MedianSharpe": subset["Sharpe"].median(),
                "StdSharpe": subset["Sharpe"].std(),
                "MinSharpe": subset["Sharpe"].min(),
                "MaxSharpe": subset["Sharpe"].max(),
                "SharpeQ25": subset["Sharpe"].quantile(0.25),
                "SharpeQ75": subset["Sharpe"].quantile(0.75),
                # MaxDD stats
                "MeanMaxDD": subset["MaxDD"].mean(),
                "MedianMaxDD": subset["MaxDD"].median(),
                # IC stats
                "MeanAvgIC": subset["AvgIC"].mean(),
                "MedianAvgIC": subset["AvgIC"].median(),
                # ExcessRet stats (extended)
                "MeanExcessRet": subset["ExcessRet"].mean(),
                "MedianExcessRet": subset["ExcessRet"].median(),
                "MinExcessRet": subset["ExcessRet"].min(),
                "MaxExcessRet": subset["ExcessRet"].max(),
                # Hit rate
                "MeanHitRate": subset["HitRate"].mean(),
                # Success percentages
                "PctSharpeGt1": (subset["Sharpe"] > 1.0).mean() * 100,
                "PctExcessPos": (subset["ExcessRet"] > 0).mean() * 100,
            }
            summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def compute_degenerate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute degenerate run statistics for each (universe_size, top_k) combination.
    
    Args:
        df: Per-run results DataFrame with IsDegenerate column.
        
    Returns:
        DataFrame with degenerate run counts per configuration.
    """
    if "IsDegenerate" not in df.columns:
        return pd.DataFrame()
    
    # Filter out error rows
    valid = df[df["Error"].isna()].copy()
    
    stats_rows = []
    
    for universe_size in UNIVERSE_SIZES:
        for top_k in TOP_K_VALUES:
            subset = valid[(valid["UniverseSize"] == universe_size) & (valid["TopK"] == top_k)]
            
            if len(subset) == 0:
                continue
            
            n_total = len(subset)
            n_degen = subset["IsDegenerate"].sum()
            pct_degen = (n_degen / n_total * 100) if n_total > 0 else 0.0
            
            stats_rows.append({
                "UniverseSize": universe_size,
                "TopK": top_k,
                "TotalRuns": n_total,
                "DegenerateRuns": n_degen,
                "NonDegenerateRuns": n_total - n_degen,
                "PctDegenerate": pct_degen,
            })
    
    return pd.DataFrame(stats_rows)


# =============================================================================
# Output and Reporting
# =============================================================================

def print_summary_tables(
    summary_df: pd.DataFrame,
    degen_stats_df: pd.DataFrame | None = None,
    summary_label: str = "ALL RUNS",
) -> None:
    """
    Print summary tables in plain text and Markdown formats.
    
    Args:
        summary_df: Summary statistics DataFrame.
        degen_stats_df: Degenerate run statistics (optional).
        summary_label: Label for the summary section (e.g., "ALL RUNS" or "NON-DEGENERATE ONLY").
    """
    print("\n" + "=" * 80)
    print(f"AGGREGATED SUMMARY BY (UniverseSize, TopK) - {summary_label}")
    print("=" * 80)
    
    # Print degenerate stats if provided
    if degen_stats_df is not None and not degen_stats_df.empty:
        print("\n--- Degenerate Run Counts ---\n")
        degen_display = degen_stats_df.copy()
        degen_display["PctDegenerate"] = degen_display["PctDegenerate"].apply(lambda x: f"{x:.1f}%")
        print(degen_display.to_string(index=False))
    
    # Plain text table - Core metrics
    print("\n--- Core Summary ---\n")
    display_cols = [
        "UniverseSize", "TopK", "Count",
        "MeanSharpe", "MedianSharpe", "StdSharpe",
        "PctSharpeGt1", "MeanExcessRet", "PctExcessPos", "MeanMaxDD"
    ]
    print(summary_df[display_cols].to_string(index=False))
    
    # Extended stats table
    print("\n--- Extended Sharpe & Excess Return Stats ---\n")
    extended_cols = [
        "UniverseSize", "TopK", "Count",
        "MinSharpe", "SharpeQ25", "MedianSharpe", "SharpeQ75", "MaxSharpe",
        "MinExcessRet", "MeanExcessRet", "MaxExcessRet"
    ]
    # Format for display
    ext_df = summary_df[extended_cols].copy()
    for col in ["MinSharpe", "SharpeQ25", "MedianSharpe", "SharpeQ75", "MaxSharpe"]:
        ext_df[col] = ext_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    for col in ["MinExcessRet", "MeanExcessRet", "MaxExcessRet"]:
        ext_df[col] = ext_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    print(ext_df.to_string(index=False))
    
    # Markdown table
    print("\n--- Markdown Table ---\n")
    md_df = summary_df.copy()
    md_df["MeanSharpe"] = md_df["MeanSharpe"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    md_df["MedianSharpe"] = md_df["MedianSharpe"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    md_df["MinSharpe"] = md_df["MinSharpe"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    md_df["MaxSharpe"] = md_df["MaxSharpe"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    md_df["PctSharpeGt1"] = md_df["PctSharpeGt1"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    md_df["MeanExcessRet"] = md_df["MeanExcessRet"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    md_df["MinExcessRet"] = md_df["MinExcessRet"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    md_df["MaxExcessRet"] = md_df["MaxExcessRet"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    md_df["PctExcessPos"] = md_df["PctExcessPos"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    md_df["MeanMaxDD"] = md_df["MeanMaxDD"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    
    md_cols = ["UniverseSize", "TopK", "MeanSharpe", "MedianSharpe", "MinSharpe", "MaxSharpe",
               "PctSharpeGt1", "MeanExcessRet", "MinExcessRet", "MaxExcessRet", "PctExcessPos", "MeanMaxDD"]
    print(md_df[md_cols].to_markdown(index=False))


def print_best_configurations(summary_df: pd.DataFrame) -> None:
    """
    Print highlights of best-performing configurations.
    
    Args:
        summary_df: Summary statistics DataFrame.
    """
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    
    if summary_df.empty:
        print("\nNo valid results to summarize.")
        return
    
    # Best by mean Sharpe
    best_sharpe = summary_df.loc[summary_df["MeanSharpe"].idxmax()]
    print(f"\n[Best Mean Sharpe]")
    print(f"    UniverseSize={int(best_sharpe['UniverseSize'])}, TopK={int(best_sharpe['TopK'])}")
    print(f"    Mean Sharpe: {best_sharpe['MeanSharpe']:.3f}")
    print(f"    Median Sharpe: {best_sharpe['MedianSharpe']:.3f}")
    
    # Best by mean ExcessRet
    best_excess = summary_df.loc[summary_df["MeanExcessRet"].idxmax()]
    print(f"\n[Best Mean Excess Return]")
    print(f"    UniverseSize={int(best_excess['UniverseSize'])}, TopK={int(best_excess['TopK'])}")
    print(f"    Mean ExcessRet: {best_excess['MeanExcessRet']:.2f}%")
    
    # Configurations with >= 70% of runs having Sharpe > 1.0
    print(f"\n[Robust Configurations (PctSharpeGt1 >= 70%)]")
    robust = summary_df[summary_df["PctSharpeGt1"] >= 70]
    if robust.empty:
        print("    None - no configuration had >= 70% of runs with Sharpe > 1.0")
    else:
        for _, row in robust.iterrows():
            print(f"    UniverseSize={int(row['UniverseSize'])}, TopK={int(row['TopK'])}: "
                  f"{row['PctSharpeGt1']:.1f}% of runs > 1.0 Sharpe")
    
    # Configurations with >= 70% positive excess return
    print(f"\n[Alpha Generators (PctExcessPos >= 70%)]")
    alpha_gen = summary_df[summary_df["PctExcessPos"] >= 70]
    if alpha_gen.empty:
        print("    None - no configuration had >= 70% of runs with positive excess return")
    else:
        for _, row in alpha_gen.iterrows():
            print(f"    UniverseSize={int(row['UniverseSize'])}, TopK={int(row['TopK'])}: "
                  f"{row['PctExcessPos']:.1f}% of runs beat benchmark")


def save_results(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """
    Save results to CSV files.
    
    Args:
        results_df: Per-run results DataFrame.
        summary_df: Summary statistics DataFrame.
    """
    # Create results directory if needed
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Save per-run results
    results_path = RESULTS_DIR / "random_universe_validation.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[SAVE] Per-run results saved to: {results_path}")
    
    # Save summary
    summary_path = RESULTS_DIR / "random_universe_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVE] Summary saved to: {summary_path}")


# =============================================================================
# CLI Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Random Universe Validation for auto_researcher. "
                    "Tests model generalization across random S&P 500 subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-per-config",
        type=int,
        default=DEFAULT_RUNS_PER_CONFIG,
        help="Number of random universes to run per (UniverseSize, TopK) configuration",
    )
    parser.add_argument(
        "--exclude-degenerate",
        action="store_true",
        help="Exclude degenerate runs (all-cash/zero-return) from aggregated summary statistics",
    )
    parser.add_argument(
        "--use-enhanced-model",
        action="store_true",
        help="Use the enhanced model with additional features (short reversal, residual momentum, idio vol, etc.)",
    )
    parser.add_argument(
        "--universe-sizes",
        type=str,
        default=None,
        help="Comma-separated list of universe sizes to test (default: 20,50,100). E.g., --universe-sizes 20,30",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default=None,
        help="Comma-separated list of top-K values to test (default: 3,5). E.g., --top-k-values 3,5,10",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    global UNIVERSE_SIZES, TOP_K_VALUES
    
    # Parse CLI arguments
    args = parse_args()
    runs_per_config = args.runs_per_config
    exclude_degenerate = args.exclude_degenerate
    use_enhanced = args.use_enhanced_model
    
    # Parse custom universe sizes if provided
    if args.universe_sizes:
        UNIVERSE_SIZES = [int(x.strip()) for x in args.universe_sizes.split(",")]
    
    # Parse custom top-k values if provided
    if args.top_k_values:
        TOP_K_VALUES = [int(x.strip()) for x in args.top_k_values.split(",")]
    
    model_label = "ENHANCED" if use_enhanced else "base"
    
    print("=" * 80)
    print("  RANDOM UNIVERSE VALIDATION")
    print("  Testing generalization across random S&P 500 subsets")
    print("=" * 80)
    print()
    print(f"Date Range:         {START_DATE} to {END_DATE}")
    print(f"Benchmark:          {BENCHMARK}")
    print(f"Horizon:            {HORIZON} days")
    print(f"Rebalance:          Monthly")
    print(f"Model:              {model_label}")
    print(f"Universe Sizes:     {UNIVERSE_SIZES}")
    print(f"Top-K Values:       {TOP_K_VALUES}")
    print(f"Runs per Config:    {runs_per_config}")
    print(f"RNG Seed:           {RNG_SEED}")
    print(f"Core Excluded:      {CORE_TICKERS}")
    print(f"Exclude Degenerate: {exclude_degenerate}")
    
    # Load S&P 500 pool
    pool = get_sp500_pool()
    
    # Shuffle pool once using base seed (for reproducibility)
    random.seed(RNG_SEED)
    random.shuffle(pool)
    
    # Run Monte Carlo validation
    results = run_monte_carlo_validation(pool, runs_per_config, use_enhanced=use_enhanced)
    
    # Build results DataFrame
    results_df = build_results_dataframe(results)
    
    # Compute degenerate run statistics
    degen_stats_df = compute_degenerate_stats(results_df)
    
    # Count degenerate runs
    n_degen_total = results_df["IsDegenerate"].sum() if "IsDegenerate" in results_df.columns else 0
    
    # Compute summary statistics (primary - respecting --exclude-degenerate flag)
    summary_df = compute_summary_statistics(results_df, exclude_degenerate=exclude_degenerate)
    
    # Print summary tables
    if exclude_degenerate:
        # Only show non-degenerate summary
        print_summary_tables(
            summary_df,
            degen_stats_df=degen_stats_df,
            summary_label="NON-DEGENERATE RUNS ONLY",
        )
        if n_degen_total > 0:
            valid_count = len(results_df[results_df["Error"].isna()])
            print(f"\n  Note: {n_degen_total}/{valid_count} runs ({n_degen_total/valid_count*100:.1f}%) "
                  f"were degenerate and excluded from statistics.")
    else:
        # Show all-runs summary first
        print_summary_tables(
            summary_df,
            degen_stats_df=degen_stats_df,
            summary_label="ALL RUNS",
        )
        
        # Also show non-degenerate-only summary for comparison
        if n_degen_total > 0:
            clean_summary_df = compute_summary_statistics(results_df, exclude_degenerate=True)
            print("\n")
            print_summary_tables(
                clean_summary_df,
                degen_stats_df=None,  # Already shown above
                summary_label="NON-DEGENERATE RUNS ONLY (for comparison)",
            )
    
    # Print best configurations (using the primary summary)
    print_best_configurations(summary_df)
    
    # Save results
    save_results(results_df, summary_df)
    
    # Final summary
    error_count = results_df["Error"].notna().sum()
    success_count = len(results_df) - error_count
    degen_count = results_df["IsDegenerate"].sum() if "IsDegenerate" in results_df.columns else 0
    
    print("\n" + "=" * 80)
    print(f"[DONE] Validation complete!")
    print(f"       Successful runs:  {success_count}/{len(results_df)}")
    if error_count > 0:
        print(f"       Failed runs:      {error_count}")
    if degen_count > 0:
        print(f"       Degenerate runs:  {degen_count} ({degen_count/success_count*100:.1f}% of successful)")
    print("=" * 80)


if __name__ == "__main__":
    main()
