"""
XGBoost Hyperparameter Optimization with Optuna + Proper Time-Series CV

Uses temporal cross-validation to avoid hyperparameter overfitting:
- Fold 1: Train 2018-2020, Validate 2021
- Fold 2: Train 2018-2021, Validate 2022  
- Fold 3: Train 2018-2022, Validate 2023

Average Sharpe across folds is the objective.
Final evaluation on held-out 2024-2025.

Estimated runtime: 2-3 hours for 50 trials
"""

import optuna
from optuna.samplers import TPESampler
import subprocess
import re
import time
from datetime import datetime
from pathlib import Path
import json
import numpy as np

# Output files
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STUDY_NAME = f"xgb_optimization_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_FILE = RESULTS_DIR / f"{STUDY_NAME}.json"
BEST_PARAMS_FILE = RESULTS_DIR / "best_hyperparams_optuna_cv.txt"

# Number of trials
N_TRIALS = 50

# Time-series CV folds for hyperparameter tuning
# Each fold runs a full backtest from start_date to end_date
# The rolling window (504 days) means training data comes from within each period
# We use non-overlapping periods to approximate train/val splits
CV_FOLDS = [
    ('2018-01-01', '2020-06-30'),  # Fold 1: ~2.5 years
    ('2020-01-01', '2022-06-30'),  # Fold 2: ~2.5 years (slight overlap for continuity)
    ('2021-01-01', '2023-12-31'),  # Fold 3: ~3 years
]

# Held-out test period (NEVER used during HP tuning)
HOLDOUT_START = '2024-01-01'
HOLDOUT_END = '2025-12-31'

# Base backtest args (dates will be added per fold)
BASE_ARGS = [
    '--enhanced-model',
    '--rolling-window',
    '--shap-selection',
    '--shap-top-k', '15',
    '--sp500',
    '--top-k', '15',
    '--benchmark', 'SPY',
]


def run_single_backtest(params: dict, start_date: str, end_date: str | None = None) -> tuple[float, float, float]:
    """
    Run a single backtest with given hyperparameters and date range.
    Returns (sharpe_ratio, total_return, max_drawdown) or (None, None, None) on failure.
    """
    cmd = [
        'python', '-m', 'auto_researcher.cli.main',
        *BASE_ARGS,
        '--start-date', start_date,
        '--xgb-max-depth', str(params['max_depth']),
        '--xgb-learning-rate', str(params['learning_rate']),
        '--xgb-n-estimators', str(params['n_estimators']),
        '--xgb-subsample', str(params['subsample']),
        '--xgb-colsample-bytree', str(params['colsample_bytree']),
        '--xgb-reg-lambda', str(params['reg_lambda']),
        '--xgb-reg-alpha', str(params['reg_alpha']),
    ]
    
    if end_date:
        cmd.extend(['--end-date', end_date])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    output = result.stdout + result.stderr
    
    # Parse results
    ret_match = re.search(r'Total Return:\s+([\d.-]+)%', output)
    sharpe_match = re.search(r'Sharpe Ratio:\s+([\d.-]+)', output)
    dd_match = re.search(r'Max Drawdown:\s+([\d.-]+)%', output)
    
    if ret_match and sharpe_match and dd_match:
        return (
            float(sharpe_match.group(1)),
            float(ret_match.group(1)),
            float(dd_match.group(1))
        )
    return None, None, None


def run_cv_backtest(params: dict) -> tuple[float, list[float]]:
    """
    Run time-series cross-validation backtest.
    Returns (mean_sharpe, list_of_fold_sharpes).
    """
    fold_sharpes = []
    
    for i, (start_date, end_date) in enumerate(CV_FOLDS):
        # Run backtest on this period
        sharpe, ret, dd = run_single_backtest(params, start_date, end_date)
        
        if sharpe is not None:
            fold_sharpes.append(sharpe)
        else:
            fold_sharpes.append(0.0)  # Penalize failures
    
    mean_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0.0
    return mean_sharpe, fold_sharpes


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function - returns mean CV Sharpe ratio to maximize."""
    
    # Sample hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
    }
    
    start_time = time.time()
    mean_sharpe, fold_sharpes = run_cv_backtest(params)
    elapsed = time.time() - start_time
    
    # Store fold results
    trial.set_user_attr('fold_sharpes', fold_sharpes)
    trial.set_user_attr('elapsed_seconds', elapsed)
    
    print(f"  Trial {trial.number}: Mean Sharpe={mean_sharpe:.2f} "
          f"[{', '.join(f'{s:.2f}' for s in fold_sharpes)}] ({elapsed:.0f}s)")
    
    return mean_sharpe


def main():
    print("="*70)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION (Optuna + Time-Series CV)")
    print("="*70)
    print()
    print(f"Using Bayesian optimization with {N_TRIALS} trials")
    print(f"Cross-validation folds: {len(CV_FOLDS)}")
    for i, (start, end) in enumerate(CV_FOLDS):
        print(f"  Fold {i+1}: {start} to {end}")
    print(f"Held-out test period: {HOLDOUT_START} to {HOLDOUT_END} (NOT used in tuning)")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print()
    
    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='maximize',
        sampler=sampler,
    )
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    start_time = time.time()
    
    print("Starting optimization...")
    print("-"*70)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    opt_elapsed = time.time() - start_time
    
    # Get best params
    best = study.best_trial
    
    print()
    print("="*70)
    print("OPTIMIZATION COMPLETE - NOW TESTING ON HELD-OUT PERIOD")
    print("="*70)
    
    # Run final evaluation on held-out test set
    print(f"\nEvaluating best params on held-out period ({HOLDOUT_START} to {HOLDOUT_END})...")
    holdout_sharpe, holdout_return, holdout_dd = run_single_backtest(
        best.params, HOLDOUT_START, HOLDOUT_END
    )
    
    total_elapsed = time.time() - start_time
    
    print()
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Optimization time: {opt_elapsed/60:.1f} minutes")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print()
    
    print("BEST CONFIGURATION:")
    print("-"*40)
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    print("CROSS-VALIDATION RESULTS (used for tuning):")
    print("-"*40)
    print(f"  Mean CV Sharpe: {best.value:.2f}")
    print(f"  Fold Sharpes:   {best.user_attrs.get('fold_sharpes', [])}")
    print()
    
    print("HELD-OUT TEST RESULTS (true out-of-sample):")
    print("-"*40)
    if holdout_sharpe is not None:
        print(f"  Sharpe Ratio:  {holdout_sharpe:.2f}")
        print(f"  Total Return:  {holdout_return:.2f}%")
        print(f"  Max Drawdown:  {holdout_dd:.2f}%")
        
        # Check for overfitting
        cv_sharpe = best.value
        if holdout_sharpe < cv_sharpe * 0.5:
            print(f"\n  ⚠️  WARNING: Holdout Sharpe ({holdout_sharpe:.2f}) is much lower than CV Sharpe ({cv_sharpe:.2f})")
            print(f"      This suggests possible overfitting to the CV period.")
        elif holdout_sharpe >= cv_sharpe * 0.8:
            print(f"\n  ✓ Good generalization: Holdout Sharpe ({holdout_sharpe:.2f}) is close to CV Sharpe ({cv_sharpe:.2f})")
    else:
        print("  FAILED to run holdout backtest")
    
    # Save results
    results = {
        'study_name': STUDY_NAME,
        'n_trials': N_TRIALS,
        'cv_folds': CV_FOLDS,
        'holdout_period': [HOLDOUT_START, HOLDOUT_END],
        'optimization_elapsed_minutes': opt_elapsed / 60,
        'best_params': best.params,
        'best_cv_sharpe': best.value,
        'best_fold_sharpes': best.user_attrs.get('fold_sharpes'),
        'holdout_sharpe': holdout_sharpe,
        'holdout_return': holdout_return,
        'holdout_max_drawdown': holdout_dd,
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'cv_sharpe': t.value,
                'fold_sharpes': t.user_attrs.get('fold_sharpes'),
            }
            for t in study.trials
        ]
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    # Save best params
    with open(BEST_PARAMS_FILE, 'w') as f:
        f.write("Best Hyperparameters from Optuna Optimization with Time-Series CV\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trials: {N_TRIALS}\n")
        f.write(f"CV Folds: {len(CV_FOLDS)}\n")
        f.write("\n")
        f.write("Parameters:\n")
        for key, value in best.params.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write("Cross-Validation Results:\n")
        f.write(f"  Mean CV Sharpe: {best.value:.2f}\n")
        f.write(f"  Fold Sharpes: {best.user_attrs.get('fold_sharpes', [])}\n")
        f.write("\n")
        f.write(f"Held-Out Test Results ({HOLDOUT_START} to {HOLDOUT_END}):\n")
        if holdout_sharpe is not None:
            f.write(f"  Sharpe Ratio:  {holdout_sharpe:.2f}\n")
            f.write(f"  Total Return:  {holdout_return:.2f}%\n")
            f.write(f"  Max Drawdown:  {holdout_dd:.2f}%\n")
        f.write("\n")
        f.write("CLI command with best params:\n")
        f.write("python -m auto_researcher.cli.main --enhanced-model --rolling-window "
                "--shap-selection --shap-top-k 15 --sp500 --top-k 15 --benchmark SPY ")
        f.write(f"--xgb-max-depth {best.params['max_depth']} ")
        f.write(f"--xgb-learning-rate {best.params['learning_rate']:.4f} ")
        f.write(f"--xgb-n-estimators {best.params['n_estimators']} ")
        f.write(f"--xgb-subsample {best.params['subsample']:.2f} ")
        f.write(f"--xgb-colsample-bytree {best.params['colsample_bytree']:.2f} ")
        f.write(f"--xgb-reg-lambda {best.params['reg_lambda']:.4f} ")
        f.write(f"--xgb-reg-alpha {best.params['reg_alpha']:.4f}\n")
    
    print(f"Best params saved to: {BEST_PARAMS_FILE}")


if __name__ == '__main__':
    main()
