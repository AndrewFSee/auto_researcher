#!/usr/bin/env python
"""
Universe Scaling Experiment Generator for auto_researcher.

This script generates CLI commands to test whether the stock-picking model
has real alpha beyond a small universe of known winners.

It defines three universes of increasing size and generates commands for
each combination of universe × top_k. Commands are printed for manual
execution along with a Markdown results table template.

Usage:
    python scripts/run_universe_scaling_experiments.py
"""

import random
import sys
from dataclasses import dataclass
from typing import Optional


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
MODEL_TYPE = "base"

# Top-K values to test
TOP_KS = [3, 5]

# Universe A: Known mega-caps (original test universe)
UNIVERSE_A = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "SPY"]

# Target sizes for randomly sampled universes
UNIVERSE_B_SIZE = 20
UNIVERSE_C_SIZE = 50


# =============================================================================
# S&P 500 Ticker List (Fallback)
# =============================================================================

# Fallback list of S&P 500 tickers (as of late 2024)
# This is used if yfinance fetch fails
SP500_FALLBACK = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "ACN", "ADBE", "IBM",
    "INTC", "AMD", "TXN", "QCOM", "AMAT", "ADI", "LRCX", "MU", "KLAC", "MCHP",
    "CDNS", "SNPS", "FTNT", "PANW", "NOW", "PLTR", "CRWD",
    # Communication Services
    "GOOG", "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "WBD", "PARA", "FOXA", "LYV", "OMC", "IPG",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR",
    "ORLY", "AZO", "ROST", "CMG", "DHI", "LEN", "GM", "F", "APTV", "EBAY",
    "ETSY", "YUM", "DPZ", "DARDEN", "HLT", "WYNN", "LVS", "RCL", "CCL", "EXPE",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "K", "HSY", "SJM", "CAG", "CPB", "MKC", "HRL", "TSN", "KHC",
    "STZ", "TAP", "BF.B", "EL", "CHD", "CLX", "KVUE",
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
    "BDX", "BAX", "IDXX", "IQV", "A", "MTD", "WAT", "HOLX", "TECH", "ALGN",
    "DXCM", "BIIB", "MRNA", "CVS", "CI", "ELV", "HUM", "CNC", "MOH", "HCA",
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
    "EQR", "VTR", "ARE", "ESS", "MAA", "UDR", "PEAK", "HST", "INVH", "KIM",
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
    name: str
    tickers: list[str]
    description: str
    
    @property
    def size(self) -> int:
        return len(self.tickers)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    id: int
    universe: Universe
    top_k: int
    
    def build_command(self) -> str:
        """Build the CLI command string for this experiment."""
        tickers_str = " ".join(self.universe.tickers)
        
        cmd_parts = [
            "python -m auto_researcher.cli.main",
            f"--tickers {tickers_str}",
            f"--start-date {START_DATE}",
            f"--end-date {END_DATE}",
            f"--benchmark {BENCHMARK}",
            f"--horizon {HORIZON}",
            f"--top-k {self.top_k}",
            f"--rebalance {REBALANCE}",
        ]
        
        return " \\\n  ".join(cmd_parts)
    
    def build_command_oneline(self) -> str:
        """Build the CLI command as a single line."""
        tickers_str = " ".join(self.universe.tickers)
        
        return (
            f"python -m auto_researcher.cli.main "
            f"--tickers {tickers_str} "
            f"--start-date {START_DATE} "
            f"--end-date {END_DATE} "
            f"--benchmark {BENCHMARK} "
            f"--horizon {HORIZON} "
            f"--top-k {self.top_k} "
            f"--rebalance {REBALANCE}"
        )


# =============================================================================
# S&P 500 Ticker Fetching
# =============================================================================

def fetch_sp500_tickers() -> Optional[list[str]]:
    """
    Attempt to fetch S&P 500 tickers from Wikipedia via pandas.
    
    Returns:
        List of ticker symbols, or None if fetch fails.
    """
    try:
        import pandas as pd
        
        # Wikipedia maintains a table of S&P 500 constituents
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        
        # First table contains the current constituents
        df = tables[0]
        
        # Symbol column may be named 'Symbol' or 'Ticker'
        symbol_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker'
        tickers = df[symbol_col].tolist()
        
        # Clean up tickers (remove dots, e.g., BRK.B -> BRK-B for yfinance)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"  [OK] Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
        
    except Exception as e:
        print(f"  [WARN] Failed to fetch S&P 500 tickers: {e}")
        return None


def get_sp500_tickers() -> list[str]:
    """
    Get S&P 500 tickers, with fallback to local list.
    
    Returns:
        List of S&P 500 ticker symbols.
    """
    print("\n[DATA] Loading S&P 500 ticker list...")
    
    # Try to fetch from Wikipedia
    tickers = fetch_sp500_tickers()
    
    if tickers is None:
        print(f"  [INFO] Using fallback list of {len(SP500_FALLBACK)} tickers")
        tickers = SP500_FALLBACK.copy()
    
    return tickers


# =============================================================================
# Universe Generation
# =============================================================================

def create_universes(sp500_tickers: list[str]) -> list[Universe]:
    """
    Create the three test universes.
    
    Args:
        sp500_tickers: Full list of S&P 500 tickers.
        
    Returns:
        List of Universe objects.
    """
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Universe A: Fixed mega-cap list
    universe_a = Universe(
        name="A",
        tickers=UNIVERSE_A.copy(),
        description="Fixed mega-cap (known winners)",
    )
    
    # Filter out SPY and tickers already in Universe A for sampling
    available_tickers = [
        t for t in sp500_tickers 
        if t != "SPY" and t not in UNIVERSE_A
    ]
    
    # Shuffle for random sampling
    random.shuffle(available_tickers)
    
    # Universe B: ~20 random S&P 500 tickers + SPY
    sample_b = available_tickers[:UNIVERSE_B_SIZE - 1]  # -1 to leave room for SPY
    sample_b.append("SPY")  # Always include benchmark
    sample_b.sort()  # Sort for readability
    
    universe_b = Universe(
        name="B",
        tickers=sample_b,
        description=f"Random S&P 500 sample ({UNIVERSE_B_SIZE} tickers)",
    )
    
    # Universe C: ~50 random S&P 500 tickers + SPY
    # Use different slice to get different tickers than B
    sample_c = available_tickers[:UNIVERSE_C_SIZE - 1]  # -1 to leave room for SPY
    sample_c.append("SPY")  # Always include benchmark
    sample_c.sort()  # Sort for readability
    
    universe_c = Universe(
        name="C",
        tickers=sample_c,
        description=f"Random S&P 500 sample ({UNIVERSE_C_SIZE} tickers)",
    )
    
    return [universe_a, universe_b, universe_c]


# =============================================================================
# Experiment Grid Generation
# =============================================================================

def create_experiment_grid(universes: list[Universe]) -> list[ExperimentConfig]:
    """
    Create the full experiment grid.
    
    Args:
        universes: List of Universe objects to test.
        
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
# Output Formatting
# =============================================================================

def print_universes(universes: list[Universe]) -> None:
    """Print universe definitions."""
    print("\n" + "=" * 70)
    print("UNIVERSE DEFINITIONS")
    print("=" * 70)
    
    for u in universes:
        print(f"\n[Universe {u.name}] {u.description}")
        print(f"  Size: {u.size} tickers")
        print(f"  Tickers: {', '.join(u.tickers[:10])}", end="")
        if u.size > 10:
            print(f", ... (+{u.size - 10} more)")
        else:
            print()


def print_experiments(experiments: list[ExperimentConfig]) -> None:
    """Print experiment commands."""
    print("\n" + "=" * 70)
    print("EXPERIMENT COMMANDS")
    print("=" * 70)
    print(f"\nGenerated {len(experiments)} experiments to run:")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Horizon: {HORIZON} days")
    print(f"  Rebalance: Monthly")
    print(f"  Benchmark: {BENCHMARK}")
    print(f"  Date Range: {START_DATE} to {END_DATE}")
    
    for exp in experiments:
        print(f"\n{'─' * 70}")
        print(f"[Experiment {exp.id}] Universe {exp.universe.name} | "
              f"Size={exp.universe.size} | Top-K={exp.top_k}")
        print(f"{'─' * 70}")
        print()
        print(exp.build_command())
        print()


def print_experiment_table(experiments: list[ExperimentConfig]) -> None:
    """Print experiments as a summary table."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY TABLE")
    print("=" * 70)
    print()
    print("| ID | Universe | Size | Top-K | Description |")
    print("|----|----------|------|-------|-------------|")
    
    for exp in experiments:
        print(f"| {exp.id:2d} | {exp.universe.name:8s} | {exp.universe.size:4d} | "
              f"{exp.top_k:5d} | {exp.universe.description} |")
    print()


def print_results_template(experiments: list[ExperimentConfig]) -> None:
    """Print a Markdown template for recording results."""
    print("\n" + "=" * 70)
    print("RESULTS TABLE TEMPLATE (copy and fill in)")
    print("=" * 70)
    print()
    print("```markdown")
    print("## Universe Scaling Experiment Results")
    print()
    print(f"**Configuration:** Model={MODEL_TYPE}, Horizon={HORIZON}, "
          f"Rebalance=Monthly, Dates={START_DATE} to {END_DATE}")
    print()
    print("| ID | Universe | Size | Top-K | AnnRet | Sharpe | MaxDD | AvgIC | Notes |")
    print("|----|----------|------|-------|--------|--------|-------|-------|-------|")
    
    for exp in experiments:
        print(f"| {exp.id:2d} | {exp.universe.name:8s} | {exp.universe.size:4d} | "
              f"{exp.top_k:5d} |        |        |       |       |       |")
    
    print()
    print("### Key Questions:")
    print("1. Does Sharpe degrade as universe size increases?")
    print("2. Does IC improve with more diverse tickers?")
    print("3. Is the model just picking mega-caps, or does it generalize?")
    print("```")
    print()


def print_copy_paste_commands(experiments: list[ExperimentConfig]) -> None:
    """Print all commands in a copy-paste friendly format."""
    print("\n" + "=" * 70)
    print("COPY-PASTE COMMANDS (one per line)")
    print("=" * 70)
    print()
    
    for exp in experiments:
        print(f"# Experiment {exp.id}: Universe {exp.universe.name}, "
              f"Size={exp.universe.size}, Top-K={exp.top_k}")
        print(exp.build_command_oneline())
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("  UNIVERSE SCALING EXPERIMENT GENERATOR")
    print("  Testing alpha beyond known mega-cap winners")
    print("=" * 70)
    
    # Get S&P 500 tickers
    sp500_tickers = get_sp500_tickers()
    
    # Create universes
    universes = create_universes(sp500_tickers)
    
    # Print universe definitions
    print_universes(universes)
    
    # Create experiment grid
    experiments = create_experiment_grid(universes)
    
    # Print experiment summary
    print_experiment_table(experiments)
    
    # Print full commands
    print_experiments(experiments)
    
    # Print copy-paste friendly commands
    print_copy_paste_commands(experiments)
    
    # Print results template
    print_results_template(experiments)
    
    print("\n[DONE] Generated all experiment commands.")
    print("       Run each command and record results in the table above.")


if __name__ == "__main__":
    main()
