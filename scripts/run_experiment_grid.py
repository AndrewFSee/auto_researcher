#!/usr/bin/env python
"""
Experiment grid runner for auto_researcher CLI.

Runs a grid of model configurations and generates a Markdown results table.

Usage:
    python scripts/run_experiment_grid.py
"""

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "SPY"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
BENCHMARK = "SPY"
REBALANCE = "M"

# Model families
MODEL_BASE = "base"
MODEL_ENH_RANK = "enh_rank"
MODEL_ENH_REG = "enh_reg"

# Grid parameters
HORIZONS = [21, 63]
TOP_KS = [3, 5]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    id: int
    model: str
    horizon: int
    top_k: int


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    id: int
    model: str
    horizon: int
    top_k: int
    annualized_return: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    avg_ic: Optional[float] = None
    error: Optional[str] = None


def build_experiment_grid() -> list[ExperimentConfig]:
    """Build the full experiment grid."""
    configs = []
    exp_id = 1
    
    for model in [MODEL_BASE, MODEL_ENH_RANK, MODEL_ENH_REG]:
        for horizon in HORIZONS:
            for top_k in TOP_KS:
                configs.append(ExperimentConfig(
                    id=exp_id,
                    model=model,
                    horizon=horizon,
                    top_k=top_k,
                ))
                exp_id += 1
    
    return configs


# =============================================================================
# Command Building
# =============================================================================

def build_command(config: ExperimentConfig) -> list[str]:
    """Build the CLI command for an experiment configuration."""
    cmd = [
        sys.executable, "-m", "auto_researcher.cli.main",
        "--tickers", *TICKERS,
        "--start-date", START_DATE,
        "--end-date", END_DATE,
        "--benchmark", BENCHMARK,
        "--horizon", str(config.horizon),
        "--top-k", str(config.top_k),
        "--rebalance", REBALANCE,
    ]
    
    if config.model == MODEL_BASE:
        # Base model: no --enhanced-model flag
        pass
    
    elif config.model == MODEL_ENH_RANK:
        # Enhanced ranking model (pairwise)
        cmd.extend([
            "--enhanced-model",
            "--model-type", "rank_pairwise",
            "--target-mode", "rank",
        ])
    
    elif config.model == MODEL_ENH_REG:
        # Enhanced regression model with vol-normalized targets
        cmd.extend([
            "--enhanced-model",
            "--model-type", "regression",
            "--target-mode", "vol_norm",
        ])
    
    return cmd


# =============================================================================
# Metric Parsing
# =============================================================================

def parse_float(pattern: str, text: str) -> Optional[float]:
    """Extract a float value using a regex pattern."""
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None


def parse_metrics(stdout: str) -> dict:
    """Parse metrics from CLI stdout."""
    metrics = {
        "annualized_return": None,
        "sharpe": None,
        "max_drawdown": None,
        "avg_ic": None,
    }
    
    # Annualized Return: XX.XX%
    metrics["annualized_return"] = parse_float(
        r"Annualized Return:\s+([-\d.]+)%", stdout
    )
    
    # Sharpe Ratio: X.XX
    metrics["sharpe"] = parse_float(
        r"Sharpe Ratio:\s+([-\d.]+)", stdout
    )
    
    # Max Drawdown: -XX.XX%
    metrics["max_drawdown"] = parse_float(
        r"Max Drawdown:\s+([-\d.]+)%", stdout
    )
    
    # Average IC: 0.XXXX (in [MODEL] section)
    metrics["avg_ic"] = parse_float(
        r"Average IC:\s+([-\d.]+)", stdout
    )
    
    return metrics


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single experiment and return the result."""
    cmd = build_command(config)
    
    print(f"\n{'='*60}")
    print(f"[{config.id:2d}/12] Running: {config.model} | horizon={config.horizon} | top_k={config.top_k}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd[:6])} ... (truncated)")
    
    result = ExperimentResult(
        id=config.id,
        model=config.model,
        horizon=config.horizon,
        top_k=config.top_k,
    )
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per experiment
        )
        
        if proc.returncode != 0:
            result.error = f"Exit code {proc.returncode}"
            print(f"  [ERROR] {result.error}")
            if proc.stderr:
                print(f"  stderr: {proc.stderr[:500]}")
            return result
        
        # Parse metrics from stdout
        metrics = parse_metrics(proc.stdout)
        result.annualized_return = metrics["annualized_return"]
        result.sharpe = metrics["sharpe"]
        result.max_drawdown = metrics["max_drawdown"]
        result.avg_ic = metrics["avg_ic"]
        
        print(f"  [OK] AnnRet={result.annualized_return}%, Sharpe={result.sharpe}, "
              f"MaxDD={result.max_drawdown}%, AvgIC={result.avg_ic}")
        
    except subprocess.TimeoutExpired:
        result.error = "Timeout (600s)"
        print(f"  [ERROR] {result.error}")
    except Exception as e:
        result.error = str(e)
        print(f"  [ERROR] {result.error}")
    
    return result


def run_all_experiments(configs: list[ExperimentConfig]) -> list[ExperimentResult]:
    """Run all experiments in the grid."""
    results = []
    for config in configs:
        result = run_experiment(config)
        results.append(result)
    return results


# =============================================================================
# Markdown Table Generation
# =============================================================================

def format_pct(value: Optional[float]) -> str:
    """Format a percentage value."""
    if value is None:
        return ""
    return f"{value:.2f}%"


def format_float(value: Optional[float]) -> str:
    """Format a float value."""
    if value is None:
        return ""
    return f"{value:.2f}"


def format_ic(value: Optional[float]) -> str:
    """Format IC value with 4 decimal places."""
    if value is None:
        return ""
    return f"{value:.4f}"


def print_markdown_table(results: list[ExperimentResult]) -> None:
    """Print results as a Markdown table."""
    # Sort by ID
    results = sorted(results, key=lambda r: r.id)
    
    print("\n")
    print("## Experiment Results")
    print()
    print("| ID | Model     | Horizon | Top-K | AnnRet   | Sharpe | MaxDD    | AvgIC  |")
    print("|----|-----------|---------|-------|----------|--------|----------|--------|")
    
    for r in results:
        row = (
            f"| {r.id:2d} "
            f"| {r.model:9s} "
            f"| {r.horizon:7d} "
            f"| {r.top_k:5d} "
            f"| {format_pct(r.annualized_return):>8s} "
            f"| {format_float(r.sharpe):>6s} "
            f"| {format_pct(r.max_drawdown):>8s} "
            f"| {format_ic(r.avg_ic):>6s} |"
        )
        print(row)
    
    print()


def print_summary(results: list[ExperimentResult]) -> None:
    """Print a summary of best results."""
    print("## Summary")
    print()
    
    # Best Sharpe
    valid_sharpe = [r for r in results if r.sharpe is not None]
    if valid_sharpe:
        best_sharpe = max(valid_sharpe, key=lambda r: r.sharpe)
        print(f"**Best Sharpe:** {best_sharpe.model} / horizon={best_sharpe.horizon} / "
              f"top_k={best_sharpe.top_k} → **{best_sharpe.sharpe:.2f}**")
    
    # Best Annualized Return
    valid_ret = [r for r in results if r.annualized_return is not None]
    if valid_ret:
        best_ret = max(valid_ret, key=lambda r: r.annualized_return)
        print(f"**Best AnnRet:** {best_ret.model} / horizon={best_ret.horizon} / "
              f"top_k={best_ret.top_k} → **{best_ret.annualized_return:.2f}%**")
    
    # Best IC (highest absolute IC)
    valid_ic = [r for r in results if r.avg_ic is not None]
    if valid_ic:
        best_ic = max(valid_ic, key=lambda r: abs(r.avg_ic))
        print(f"**Best |IC|:** {best_ic.model} / horizon={best_ic.horizon} / "
              f"top_k={best_ic.top_k} → **{best_ic.avg_ic:.4f}**")
    
    # Smallest Max Drawdown (least negative)
    valid_dd = [r for r in results if r.max_drawdown is not None]
    if valid_dd:
        best_dd = max(valid_dd, key=lambda r: r.max_drawdown)  # least negative
        print(f"**Best MaxDD:** {best_dd.model} / horizon={best_dd.horizon} / "
              f"top_k={best_dd.top_k} → **{best_dd.max_drawdown:.2f}%**")
    
    # Count errors
    errors = [r for r in results if r.error is not None]
    if errors:
        print(f"\n**Errors:** {len(errors)} experiment(s) failed")
        for r in errors:
            print(f"  - ID {r.id} ({r.model}): {r.error}")
    
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("  AUTO-RESEARCHER EXPERIMENT GRID")
    print("=" * 60)
    print()
    print(f"Universe: {', '.join(TICKERS)}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Benchmark: {BENCHMARK}")
    print(f"Rebalance: Monthly")
    print()
    print("Grid:")
    print(f"  Models:   {MODEL_BASE}, {MODEL_ENH_RANK}, {MODEL_ENH_REG}")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Top-Ks:   {TOP_KS}")
    print(f"  Total:    {len(HORIZONS) * len(TOP_KS) * 3} experiments")
    
    # Build grid
    configs = build_experiment_grid()
    
    # Run all experiments
    results = run_all_experiments(configs)
    
    # Print Markdown table
    print_markdown_table(results)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
