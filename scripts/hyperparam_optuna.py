"""
XGBoost Hyperparameter Optimization with Optuna

Uses Bayesian optimization to efficiently search the hyperparameter space.
Much faster than grid search - typically finds good params in 50-100 trials.

Estimated runtime: 1-2 hours for 50 trials
"""

import optuna
from optuna.samplers import TPESampler
import subprocess
import re
import time
from datetime import datetime
from pathlib import Path
import json

# Output files
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STUDY_NAME = f"xgb_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_FILE = RESULTS_DIR / f"{STUDY_NAME}.json"
BEST_PARAMS_FILE = RESULTS_DIR / "best_hyperparams_optuna.txt"

# Number of trials
N_TRIALS = 50  # Usually sufficient for good results

# Backtest settings (use shorter period for faster optimization)
BACKTEST_ARGS = [
    '--enhanced-model',
    '--rolling-window',
    '--shap-selection',
    '--shap-top-k', '15',
    '--sp500',
    '--top-k', '15',
    '--benchmark', 'SPY',
    '--start-date', '2022-01-01',  # Use 3 years for optimization
]


def run_backtest(params: dict) -> tuple[float, float, float]:
    """
    Run a single backtest with given hyperparameters.
    Returns (sharpe_ratio, total_return, max_drawdown) or (None, None, None) on failure.
    """
    cmd = [
        'python', '-m', 'auto_researcher.cli.main',
        *BACKTEST_ARGS,
        '--xgb-max-depth', str(params['max_depth']),
        '--xgb-learning-rate', str(params['learning_rate']),
        '--xgb-n-estimators', str(params['n_estimators']),
        '--xgb-subsample', str(params['subsample']),
        '--xgb-colsample-bytree', str(params['colsample_bytree']),
        '--xgb-reg-lambda', str(params['reg_lambda']),
        '--xgb-reg-alpha', str(params['reg_alpha']),
    ]
    
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


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function - returns Sharpe ratio to maximize."""
    
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
    sharpe, total_return, max_dd = run_backtest(params)
    elapsed = time.time() - start_time
    
    if sharpe is None:
        print(f"  Trial {trial.number}: FAILED ({elapsed:.0f}s)")
        return float('-inf')  # Return worst possible value
    
    # Store additional metrics as user attributes
    trial.set_user_attr('total_return', total_return)
    trial.set_user_attr('max_drawdown', max_dd)
    trial.set_user_attr('elapsed_seconds', elapsed)
    
    print(f"  Trial {trial.number}: Sharpe={sharpe:.2f}, Return={total_return:.1f}%, DD={max_dd:.1f}% ({elapsed:.0f}s)")
    
    return sharpe


def main():
    print("="*70)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION (Optuna)")
    print("="*70)
    print()
    print(f"Using Bayesian optimization with {N_TRIALS} trials")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print()
    
    # Create Optuna study (maximize Sharpe ratio)
    sampler = TPESampler(seed=42)  # Tree-structured Parzen Estimator
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='maximize',
        sampler=sampler,
    )
    
    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    start_time = time.time()
    
    # Run optimization
    print("Starting optimization...")
    print("-"*70)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    total_elapsed = time.time() - start_time
    
    # Print results
    print()
    print("="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Trials completed: {len(study.trials)}")
    print()
    
    # Best trial
    best = study.best_trial
    print("BEST CONFIGURATION:")
    print("-"*40)
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    print("BEST RESULTS:")
    print("-"*40)
    print(f"  Sharpe Ratio:  {best.value:.2f}")
    print(f"  Total Return:  {best.user_attrs.get('total_return', 'N/A'):.2f}%")
    print(f"  Max Drawdown:  {best.user_attrs.get('max_drawdown', 'N/A'):.2f}%")
    
    # Save results
    results = {
        'study_name': STUDY_NAME,
        'n_trials': N_TRIALS,
        'total_elapsed_minutes': total_elapsed / 60,
        'best_params': best.params,
        'best_sharpe': best.value,
        'best_total_return': best.user_attrs.get('total_return'),
        'best_max_drawdown': best.user_attrs.get('max_drawdown'),
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'sharpe': t.value if t.value != float('-inf') else None,
                'total_return': t.user_attrs.get('total_return'),
                'max_drawdown': t.user_attrs.get('max_drawdown'),
            }
            for t in study.trials
        ]
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    # Save best params in readable format
    with open(BEST_PARAMS_FILE, 'w') as f:
        f.write("Best Hyperparameters from Optuna Optimization\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trials: {N_TRIALS}\n")
        f.write(f"Optimization time: {total_elapsed/60:.1f} minutes\n")
        f.write("\n")
        f.write("Parameters:\n")
        for key, value in best.params.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write("Results:\n")
        f.write(f"  Sharpe Ratio:  {best.value:.2f}\n")
        f.write(f"  Total Return:  {best.user_attrs.get('total_return', 'N/A'):.2f}%\n")
        f.write(f"  Max Drawdown:  {best.user_attrs.get('max_drawdown', 'N/A'):.2f}%\n")
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
    
    # Print top 5 trials
    print("\nTOP 5 CONFIGURATIONS:")
    print("-"*70)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value != float('-inf') else float('-inf'), reverse=True)
    for i, t in enumerate(sorted_trials[:5]):
        if t.value != float('-inf'):
            print(f"{i+1}. Sharpe={t.value:.2f}, Return={t.user_attrs.get('total_return', 0):.1f}%")
            print(f"   depth={t.params['max_depth']}, lr={t.params['learning_rate']:.3f}, "
                  f"n_est={t.params['n_estimators']}, sub={t.params['subsample']:.2f}")


if __name__ == '__main__':
    main()
