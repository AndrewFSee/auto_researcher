#!/usr/bin/env python
"""
Universe Scaling Experiment Runner for auto_researcher.

This script runs a grid of experiments to test whether the stock-picking
model has real alpha beyond a small universe of known mega-cap winners.

It defines three universes of increasing size, runs the base model on each,
parses performance metrics from stdout, and generates summary tables.

Usage:
    python run_universe_scaling_experiments.py
"""

import random
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Date range for all experiments
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# Fixed experiment parameters
BENCHMARK = "SPY"
HORIZON = 63
REBALANCE = "M"

# Core tickers to exclude from random sampling
CORE_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "SPY"]

# Universe sizes for random sampling
U20_SIZE = 20
U50_SIZE = 50

# Top-K values to test
TOP_KS = [3, 5]


# =============================================================================
# Fallback S&P 500 Ticker List
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
class Universe:
    """Represents a ticker universe for experiments."""
    id: str
    name: str
    tickers: list[str]

    @property
    def size(self) -> int:
        return len(self.tickers)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    id: int
    universe: Universe
    top_k: int


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    id: int
    universe_name: str
    universe_size: int
    top_k: int
    ann_ret: Optional[float] = None
    sharpe: Optional[float] = None
    max_dd: Optional[float] = None
    avg_ic: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
# Universe Construction
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


def get_sp500_tickers() -> list[str]:
    """
    Get S&P 500 tickers, with fallback to local list.
    
    Returns:
        List of S&P 500 ticker symbols (excluding core tickers).
    """
    print("\n[DATA] Loading S&P 500 ticker list...")
    
    # Try Wikipedia first
    tickers = fetch_sp500_tickers_from_wikipedia()
    
    if tickers is None:
        print(f"  [INFO] Using fallback list of {len(SP500_FALLBACK)} tickers")
        tickers = SP500_FALLBACK.copy()
    
    # Remove core tickers
    tickers = [t for t in tickers if t not in CORE_TICKERS]
    print(f"  [INFO] {len(tickers)} tickers available for sampling (ex-core)")
    
    return tickers


def build_universes() -> list[Universe]:
    """
    Build the three test universes.
    
    Returns:
        List of Universe objects: [U7_core, U20_random, U50_random]
    """
    # Set seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Universe A: Core mega-cap names
    universe_a = Universe(
        id="U7_core",
        name="U7_core",
        tickers=CORE_TICKERS.copy(),
    )
    
    # Get S&P 500 pool (excluding core)
    sp500_pool = get_sp500_tickers()
    
    # Shuffle pool for random sampling
    random.shuffle(sp500_pool)
    
    # Universe B: 20 random S&P 500 names
    sample_b = sp500_pool[:U20_SIZE]
    sample_b.sort()  # Sort for readability
    
    universe_b = Universe(
        id="U20_random",
        name="U20_random",
        tickers=sample_b,
    )
    
    # Universe C: 50 random S&P 500 names
    sample_c = sp500_pool[:U50_SIZE]
    sample_c.sort()  # Sort for readability
    
    universe_c = Universe(
        id="U50_random",
        name="U50_random",
        tickers=sample_c,
    )
    
    return [universe_a, universe_b, universe_c]


def print_universes(universes: list[Universe]) -> None:
    """Print universe definitions for verification."""
    print("\n" + "=" * 70)
    print("UNIVERSE DEFINITIONS")
    print("=" * 70)
    
    for u in universes:
        print(f"\n[{u.id}] {u.name} ({u.size} tickers)")
        # Print tickers in rows of 10
        for i in range(0, len(u.tickers), 10):
            chunk = u.tickers[i:i+10]
            print(f"  {', '.join(chunk)}")


# =============================================================================
# Experiment Grid
# =============================================================================

def build_experiment_grid(universes: list[Universe]) -> list[ExperimentConfig]:
    """
    Build the experiment grid.
    
    Args:
        universes: List of Universe objects.
        
    Returns:
        List of ExperimentConfig objects.
    """
    experiments = []
    exp_id = 1
    
    for universe in universes:
        for top_k in TOP_KS:
            experiments.append(ExperimentConfig(
                id=exp_id,
                universe=universe,
                top_k=top_k,
            ))
            exp_id += 1
    
    return experiments


# =============================================================================
# Running Experiments
# =============================================================================

def build_command(config: ExperimentConfig) -> list[str]:
    """
    Build the CLI command for an experiment.
    
    Args:
        config: Experiment configuration.
        
    Returns:
        List of command arguments for subprocess.
    """
    cmd = [
        sys.executable, "-m", "auto_researcher.cli.main",
        "--tickers", *config.universe.tickers,
        "--start-date", START_DATE,
        "--end-date", END_DATE,
        "--benchmark", BENCHMARK,
        "--horizon", str(HORIZON),
        "--top-k", str(config.top_k),
        "--rebalance", REBALANCE,
    ]
    return cmd


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run a single experiment via subprocess.
    
    Args:
        config: Experiment configuration.
        
    Returns:
        ExperimentResult with parsed metrics or error.
    """
    result = ExperimentResult(
        id=config.id,
        universe_name=config.universe.name,
        universe_size=config.universe.size,
        top_k=config.top_k,
    )
    
    print(f"\n{'─' * 70}")
    print(f"Running exp {config.id}: universe={config.universe.name}, "
          f"size={config.universe.size}, top_k={config.top_k}")
    print(f"{'─' * 70}")
    
    cmd = build_command(config)
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,  # 10 minute timeout
        )
        
        # Parse metrics from stdout
        metrics = parse_metrics(proc.stdout)
        result.ann_ret = metrics.get("ann_ret")
        result.sharpe = metrics.get("sharpe")
        result.max_dd = metrics.get("max_dd")
        result.avg_ic = metrics.get("avg_ic")
        
        print(f"  [OK] AnnRet={result.ann_ret}%, Sharpe={result.sharpe}, "
              f"MaxDD={result.max_dd}%, AvgIC={result.avg_ic}")
        
    except subprocess.CalledProcessError as e:
        result.error = f"Exit code {e.returncode}"
        print(f"  [ERROR] {result.error}")
        if e.stderr:
            print(f"  stderr: {e.stderr[:500]}")
    except subprocess.TimeoutExpired:
        result.error = "Timeout (600s)"
        print(f"  [ERROR] {result.error}")
    except Exception as e:
        result.error = str(e)
        print(f"  [ERROR] {result.error}")
    
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
        "  Average IC:             -0.1352"
    
    Args:
        stdout: Full stdout from the CLI run.
        
    Returns:
        Dict with keys: ann_ret, sharpe, max_dd, avg_ic (floats or None).
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
    
    return metrics


# =============================================================================
# Results Summary
# =============================================================================

def build_results_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    """
    Convert results to a pandas DataFrame.
    
    Args:
        results: List of ExperimentResult objects.
        
    Returns:
        DataFrame with columns: ID, Universe, Size, TopK, AnnRet, Sharpe, MaxDD, AvgIC
    """
    rows = []
    for r in results:
        rows.append({
            "ID": r.id,
            "Universe": r.universe_name,
            "Size": r.universe_size,
            "TopK": r.top_k,
            "AnnRet": r.ann_ret,
            "Sharpe": r.sharpe,
            "MaxDD": r.max_dd,
            "AvgIC": r.avg_ic,
        })
    
    return pd.DataFrame(rows)


def format_pct(val: Optional[float]) -> str:
    """Format a float as percentage string."""
    if val is None:
        return ""
    return f"{val:.2f}%"


def format_float4(val: Optional[float]) -> str:
    """Format a float with 4 decimal places."""
    if val is None:
        return ""
    return f"{val:.4f}"


def print_results_tables(df: pd.DataFrame) -> None:
    """
    Print results as plain-text and markdown tables.
    
    Args:
        df: Results DataFrame.
    """
    # Sort by Sharpe descending
    df_sorted = df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 70)
    print("RESULTS (sorted by Sharpe)")
    print("=" * 70)
    
    # Plain text table
    print("\n--- Plain Text Table ---\n")
    print(df_sorted.to_string(index=False))
    
    # Create formatted DataFrame for markdown
    df_md = df_sorted.copy()
    df_md["AnnRet"] = df_md["AnnRet"].apply(format_pct)
    df_md["Sharpe"] = df_md["Sharpe"].apply(format_float4)
    df_md["MaxDD"] = df_md["MaxDD"].apply(format_pct)
    df_md["AvgIC"] = df_md["AvgIC"].apply(format_float4)
    
    # Rename columns for markdown
    df_md = df_md.rename(columns={"TopK": "Top-K"})
    
    print("\n--- Markdown Table ---\n")
    print(df_md.to_markdown(index=False))


def print_best_configs(df: pd.DataFrame) -> None:
    """
    Print summary of best configurations.
    
    Args:
        df: Results DataFrame.
    """
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)
    
    # Filter out rows with missing metrics
    valid = df.dropna(subset=["Sharpe", "AnnRet", "MaxDD", "AvgIC"])
    
    if valid.empty:
        print("\nNo valid results to summarize.")
        return
    
    # Best by Sharpe
    best_sharpe = valid.loc[valid["Sharpe"].idxmax()]
    print(f"\n[Best Sharpe] {best_sharpe['Universe']} / TopK={best_sharpe['TopK']}")
    print(f"    Sharpe = {best_sharpe['Sharpe']:.4f}")
    
    # Best by AnnRet
    best_ret = valid.loc[valid["AnnRet"].idxmax()]
    print(f"\n[Best AnnRet] {best_ret['Universe']} / TopK={best_ret['TopK']}")
    print(f"    AnnRet = {best_ret['AnnRet']:.2f}%")
    
    # Best by |AvgIC|
    valid_with_absic = valid.copy()
    valid_with_absic["AbsIC"] = valid_with_absic["AvgIC"].abs()
    best_ic = valid_with_absic.loc[valid_with_absic["AbsIC"].idxmax()]
    print(f"\n[Best |AvgIC|] {best_ic['Universe']} / TopK={best_ic['TopK']}")
    print(f"    AvgIC = {best_ic['AvgIC']:.4f}")
    
    # Best by MaxDD (least negative = highest value)
    best_dd = valid.loc[valid["MaxDD"].idxmax()]
    print(f"\n[Best MaxDD] {best_dd['Universe']} / TopK={best_dd['TopK']}")
    print(f"    MaxDD = {best_dd['MaxDD']:.2f}%")
    
    # Check for errors
    errors = df[df["Sharpe"].isna()]
    if not errors.empty:
        print(f"\n[Errors] {len(errors)} experiment(s) had errors or missing data")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("  UNIVERSE SCALING EXPERIMENT RUNNER")
    print("  Testing alpha beyond known mega-cap winners")
    print("=" * 70)
    print()
    print(f"Date Range:   {START_DATE} to {END_DATE}")
    print(f"Benchmark:    {BENCHMARK}")
    print(f"Horizon:      {HORIZON} days")
    print(f"Rebalance:    Monthly")
    print(f"Model:        base (no --enhanced-model)")
    print(f"Top-K values: {TOP_KS}")
    print(f"RNG Seed:     {RANDOM_SEED}")
    
    # Build universes
    universes = build_universes()
    print_universes(universes)
    
    # Build experiment grid
    experiments = build_experiment_grid(universes)
    
    print("\n" + "=" * 70)
    print(f"RUNNING {len(experiments)} EXPERIMENTS")
    print("=" * 70)
    
    # Run all experiments
    results = []
    for config in experiments:
        result = run_experiment(config)
        results.append(result)
    
    # Build results DataFrame
    df = build_results_dataframe(results)
    
    # Print tables
    print_results_tables(df)
    
    # Print best configs
    print_best_configs(df)
    
    print("\n" + "=" * 70)
    print("[DONE] All experiments completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
