"""
XGBoost Hyperparameter Grid Search

Full grid search over key hyperparameters for the stock selection model.
Results are logged to a CSV file for analysis.

Estimated runtime: 2-4 hours depending on hardware.
"""

import itertools
import subprocess
import re
import csv
import time
from datetime import datetime
from pathlib import Path

# Output files
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"hyperparam_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
BEST_PARAMS_FILE = RESULTS_DIR / "best_hyperparams.txt"

# Hyperparameter grid
PARAM_GRID = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# Fixed parameters
FIXED_PARAMS = {
    'reg_lambda': 2.0,
    'reg_alpha': 0.1,
    'min_child_weight': 1.0,
    'gamma': 0.0,
}

# Backtest settings
BACKTEST_ARGS = [
    '--enhanced-model',
    '--rolling-window',
    '--shap-selection',
    '--shap-top-k', '15',
    '--sp500',
    '--top-k', '15',
    '--benchmark', 'SPY',
    '--start-date', '2021-01-01',  # Use full period for tuning
]


def run_backtest(params: dict) -> dict:
    """Run a single backtest with given hyperparameters."""
    # Build command with hyperparameters
    cmd = [
        'python', '-m', 'auto_researcher.cli.main',
        *BACKTEST_ARGS,
        '--xgb-max-depth', str(params['max_depth']),
        '--xgb-learning-rate', str(params['learning_rate']),
        '--xgb-n-estimators', str(params['n_estimators']),
        '--xgb-subsample', str(params['subsample']),
        '--xgb-colsample-bytree', str(params['colsample_bytree']),
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start_time
    
    output = result.stdout + result.stderr
    
    # Parse results
    ret_match = re.search(r'Total Return:\s+([\d.-]+)%', output)
    sharpe_match = re.search(r'Sharpe Ratio:\s+([\d.-]+)', output)
    dd_match = re.search(r'Max Drawdown:\s+([\d.-]+)%', output)
    
    return {
        'total_return': float(ret_match.group(1)) if ret_match else None,
        'sharpe_ratio': float(sharpe_match.group(1)) if sharpe_match else None,
        'max_drawdown': float(dd_match.group(1)) if dd_match else None,
        'elapsed_seconds': elapsed,
        'success': ret_match is not None and sharpe_match is not None,
    }


def main():
    print("="*70)
    print("XGBOOST HYPERPARAMETER GRID SEARCH")
    print("="*70)
    print()
    
    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(itertools.product(*param_values))
    
    total_runs = len(all_combinations)
    print(f"Grid size: {' x '.join(str(len(v)) for v in param_values)} = {total_runs} combinations")
    print(f"Estimated time: {total_runs * 10 / 60:.1f} - {total_runs * 20 / 60:.1f} minutes")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print()
    
    # Initialize results file
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_id', 'max_depth', 'learning_rate', 'n_estimators', 
            'subsample', 'colsample_bytree',
            'total_return', 'sharpe_ratio', 'max_drawdown', 
            'elapsed_seconds', 'success'
        ])
    
    best_sharpe = -999
    best_params = None
    best_results = None
    
    start_time = time.time()
    
    for i, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))
        
        # Progress update
        elapsed_total = time.time() - start_time
        if i > 0:
            avg_time = elapsed_total / i
            remaining = avg_time * (total_runs - i)
            eta = f"{remaining/60:.1f} min remaining"
        else:
            eta = "calculating..."
        
        print(f"[{i+1}/{total_runs}] Testing: depth={params['max_depth']}, "
              f"lr={params['learning_rate']}, n_est={params['n_estimators']}, "
              f"sub={params['subsample']}, col={params['colsample_bytree']} ... ", 
              end='', flush=True)
        
        # Run backtest
        results = run_backtest(params)
        
        if results['success']:
            print(f"Sharpe={results['sharpe_ratio']:.2f}, Return={results['total_return']:.1f}% "
                  f"({results['elapsed_seconds']:.0f}s) [{eta}]")
            
            # Track best
            if results['sharpe_ratio'] > best_sharpe:
                best_sharpe = results['sharpe_ratio']
                best_params = params.copy()
                best_results = results.copy()
                print(f"  *** NEW BEST! Sharpe={best_sharpe:.2f} ***")
        else:
            print(f"FAILED ({results['elapsed_seconds']:.0f}s)")
        
        # Save result
        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i+1, params['max_depth'], params['learning_rate'], 
                params['n_estimators'], params['subsample'], params['colsample_bytree'],
                results['total_return'], results['sharpe_ratio'], 
                results['max_drawdown'], results['elapsed_seconds'], results['success']
            ])
    
    total_elapsed = time.time() - start_time
    
    # Print summary
    print()
    print("="*70)
    print("GRID SEARCH COMPLETE")
    print("="*70)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_FILE}")
    print()
    
    if best_params:
        print("BEST CONFIGURATION:")
        print("-"*40)
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print()
        print("BEST RESULTS:")
        print("-"*40)
        print(f"  Sharpe Ratio:  {best_results['sharpe_ratio']:.2f}")
        print(f"  Total Return:  {best_results['total_return']:.2f}%")
        print(f"  Max Drawdown:  {best_results['max_drawdown']:.2f}%")
        
        # Save best params
        with open(BEST_PARAMS_FILE, 'w') as f:
            f.write("Best Hyperparameters from Grid Search\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total combinations tested: {total_runs}\n")
            f.write("\n")
            f.write("Parameters:\n")
            for k, v in best_params.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
            f.write("Results:\n")
            f.write(f"  Sharpe Ratio:  {best_results['sharpe_ratio']:.2f}\n")
            f.write(f"  Total Return:  {best_results['total_return']:.2f}%\n")
            f.write(f"  Max Drawdown:  {best_results['max_drawdown']:.2f}%\n")
        
        print(f"\nBest params saved to: {BEST_PARAMS_FILE}")
    else:
        print("No successful runs!")


if __name__ == '__main__':
    main()
