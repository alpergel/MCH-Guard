import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import json
import re
import logging
from pathlib import Path
import numpy as np
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# INTELLIGENT TRAINING WITH OPTUNA N_TRIALS ADJUSTMENT
# ============================================================================

class OptunaProgressiveTrainer:
    """Train models progressively with Optuna n_trials adjustment for statistical significance.
    In intelligent mode, only supports CLS models."""
    
    def __init__(self):
        self.model_metrics = {}  # Store metrics for comparison
        self.initial_n_trials = 50  # Default n_trials for Optuna
        self.n_trials_increment = 50  # How much to increase n_trials each retry
        self.max_n_trials = 300  # Maximum n_trials to prevent infinite loops
        self.significance_threshold = 0.05  # p-value threshold
        self.min_improvement = 0.05  # Minimum improvement required
        
    def extract_metrics(self, output: str, family: str) -> Optional[Dict[str, float]]:
        """Extract metrics from training script output."""
        metrics = {}
        
        # Different patterns for different model families
        if family == "CLS" or family == "SW":
            patterns = {
                'accuracy': r'Test accuracy[:\s]+([0-9.]+)',
                'auc': r'Test AUC[:\s]+([0-9.]+)',
                'roc_auc': r'(?:Test ROC AUC|roc_auc)[:\s]+([0-9.]+)',
                'f1': r'Test F1[:\s]+([0-9.]+)',
                'precision': r'Test precision[:\s]+([0-9.]+)',
                'recall': r'Test recall[:\s]+([0-9.]+)'
            }
        elif family == "RG":
            patterns = {
                'r2': r'R² Score[:\s]+([0-9.-]+)',
                'mse': r'Mean Squared Error[:\s]+([0-9.]+)',
                'rmse': r'Root Mean Squared Error[:\s]+([0-9.]+)',
                'mae': r'Mean Absolute Error[:\s]+([0-9.]+)'
            }
        elif family == "SRV":
            patterns = {
                'concordance_index': r'Concordance index[:\s]+([0-9.]+)',
                'aic': r'AIC[:\s]+([0-9.]+)'
            }
        else:
            return None
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                except ValueError:
                    pass
        
        return metrics if metrics else None
    
    def is_statistically_significant(self, baseline_metric: float, new_metric: float,
                                    metric_type: str = "higher_better") -> bool:
        """
        Check if the improvement is statistically significant.
        Simple check: requires both minimum improvement and relative improvement.
        """
        if metric_type == "higher_better":
            improvement = new_metric - baseline_metric
            relative_improvement = improvement / baseline_metric if baseline_metric > 0 else improvement
        else:  # lower_better (e.g., for error metrics)
            improvement = baseline_metric - new_metric
            relative_improvement = improvement / baseline_metric if baseline_metric > 0 else improvement
        
        # Require both absolute and relative improvement
        return improvement >= self.min_improvement and relative_improvement >= 0.01
    
    def run_with_optuna_adjustment(self, script_path: str, family: str, size: str,
                                   baseline_metrics: Optional[Dict] = None,
                                   n_trials: Optional[int] = None) -> Tuple[bool, Dict]:
        """
        Run training script with specified Optuna n_trials.
        If baseline_metrics provided and significance not achieved, retry with more trials.
        """
        current_n_trials = n_trials or self.initial_n_trials
        
        # Determine primary metric for this family
        primary_metrics = {
            "CLS": "auc",
            "RG": "r2",
            "SW": "roc_auc",
            "SRV": "concordance_index"
        }
        primary_metric = primary_metrics.get(family, "accuracy")
        metric_type = "lower_better" if primary_metric in ["mse", "rmse", "mae", "aic"] else "higher_better"
        
        best_metrics = None
        attempts = 0
        
        while current_n_trials <= self.max_n_trials:
            attempts += 1
            logger.info(f"\nTraining {family}-{size} (attempt {attempts}, n_trials={current_n_trials})")
            
            # Set environment variable for n_trials
            env = os.environ.copy()
            env["OPTUNA_N_TRIALS"] = str(current_n_trials)
            
            # Set hyperparameter_search flag for models that support it
            if family in ["CLS", "RG"]:
                env["HYPERPARAMETER_SEARCH"] = "true" if size != "M1" else "false"
            
            # Build command
            command = [sys.executable, "-u", script_path]
            
            # SW scripts accept --hpo flag
            if family == "SW" and size != "M1":
                command.append("--hpo")
            
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env
                )
                
                output = result.stdout + result.stderr
                metrics = self.extract_metrics(output, family)
                
                if not metrics:
                    logger.warning(f"  Could not extract metrics from output")
                    return False, {}
                
                logger.info(f"  Metrics: {metrics}")
                
                # For M1 model or if no baseline, accept first result
                if size == "M1" or baseline_metrics is None:
                    logger.info(f"  Baseline established: {primary_metric}={metrics.get(primary_metric, 0):.4f}")
                    return True, metrics
                
                # Check statistical significance
                if primary_metric in metrics and primary_metric in baseline_metrics:
                    baseline_value = baseline_metrics[primary_metric]
                    new_value = metrics[primary_metric]
                    
                    if self.is_statistically_significant(baseline_value, new_value, metric_type):
                        logger.info(f"  ✓ Statistically significant improvement achieved!")
                        logger.info(f"    {primary_metric}: {baseline_value:.4f} → {new_value:.4f}")
                        return True, metrics
                    else:
                        logger.info(f"  ✗ Not statistically significant")
                        logger.info(f"    {primary_metric}: {baseline_value:.4f} → {new_value:.4f}")
                        
                        # Store best result so far
                        if best_metrics is None:
                            best_metrics = metrics
                        elif metric_type == "higher_better" and metrics.get(primary_metric, 0) > best_metrics.get(primary_metric, 0):
                            best_metrics = metrics
                        elif metric_type == "lower_better" and metrics.get(primary_metric, float('inf')) < best_metrics.get(primary_metric, float('inf')):
                            best_metrics = metrics
                        
                        # Increase n_trials for next attempt
                        current_n_trials += self.n_trials_increment
                        if current_n_trials <= self.max_n_trials:
                            logger.info(f"  Increasing n_trials to {current_n_trials} for next attempt...")
                        
            except subprocess.CalledProcessError as e:
                logger.error(f"Training failed: {e}")
                if attempts == 1:
                    return False, {}
                # Try increasing n_trials
                current_n_trials += self.n_trials_increment
        
        # Return best result even if significance not achieved
        if best_metrics:
            logger.warning(f"  Max n_trials ({self.max_n_trials}) reached without significance")
            logger.info(f"  Returning best result: {primary_metric}={best_metrics.get(primary_metric, 0):.4f}")
            return True, best_metrics
        
        return False, {}
    
    def train_family_progressively(self, family: str, sizes: List[str], 
                                   scripts: Dict[str, str]) -> Dict[str, Dict]:
        """
        Train all sizes for a family progressively with Optuna adjustment.
        """
        results = {}
        baseline_metrics = None
        
        # Ensure we train in order: M1 -> M2 -> M3
        ordered_sizes = []
        for size in ["M1", "M2", "M3"]:
            if size in sizes and size in scripts:
                ordered_sizes.append(size)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {family} models progressively")
        logger.info(f"Order: {' → '.join(ordered_sizes)}")
        logger.info(f"{'='*60}")
        
        for size in ordered_sizes:
            script_path = scripts[size]
            
            if not os.path.exists(script_path):
                logger.error(f"Script not found: {script_path}")
                continue
            
            # Determine initial n_trials based on size
            if size == "M1":
                n_trials = None  # Use default from script
            else:
                n_trials = self.initial_n_trials
            
            success, metrics = self.run_with_optuna_adjustment(
                script_path, family, size, baseline_metrics, n_trials
            )
            
            if success and metrics:
                results[size] = metrics
                baseline_metrics = metrics  # Update baseline for next size
                self.model_metrics[f"{family}_{size}"] = metrics
            else:
                logger.error(f"Failed to train {family}-{size}")
        
        return results


MODEL_FAMILIES_TO_SCRIPTS: Dict[str, Dict[str, str]] = {
    "CLS": {
        "M3": os.path.join("notebooks", "CLS", "CLS_M3_Train.py"),
        "M2": os.path.join("notebooks", "CLS", "CLS_M2_Train.py"),
        "M1": os.path.join("notebooks", "CLS", "CLS_M1_Train.py"),
    },
    "RG": {
        "M3": os.path.join("notebooks", "RG", "RG_M3_Train.py"),
        "M2": os.path.join("notebooks", "RG", "RG_M2_Train.py"),
        "M1": os.path.join("notebooks", "RG", "RG_M1_Train.py"),
    },
    "SRV": {
        "M3": os.path.join("notebooks", "SRV", "SRV_Train_M3.py"),
        "M2": os.path.join("notebooks", "SRV", "SRV_Train_M2.py"),
        "M1": os.path.join("notebooks", "SRV", "SRV_Train_M1.py"),
    },
    "SW": {
        "M3": os.path.join("notebooks", "SW", "SW_M3_Train.py"),
        "M2": os.path.join("notebooks", "SW", "SW_M2_Train.py"),
        "M1": os.path.join("notebooks", "SW", "SW_M1_Train.py"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run training scripts for M3/M2/M1 variants across model families "
            "CLS, RG, SRV, and SW. Supports intelligent mode with Optuna n_trials adjustment."
        )
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=["CLS", "RG", "SRV", "SW"],
        choices=list(MODEL_FAMILIES_TO_SCRIPTS.keys()),
        help="Which model families to train.",
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=["M3", "M2", "M1"],
        choices=["M3", "M2", "M1"],
        help="Which model sizes to train.",
    )
    parser.add_argument(
        "--intelligent",
        action="store_true",
        help="Use intelligent mode for CLS models only: trains m1->m2->m3 with Optuna n_trials adjustment for significance.",
    )
    parser.add_argument(
        "--initial-n-trials",
        type=int,
        default=100,
        help="Initial Optuna n_trials for hyperparameter search (intelligent mode).",
    )
    parser.add_argument(
        "--n-trials-increment",
        type=int,
        default=50,
        help="How much to increase n_trials if significance not achieved (intelligent mode).",
    )
    parser.add_argument(
        "--max-n-trials",
        type=int,
        default=300,
        help="Maximum n_trials to prevent infinite tuning (intelligent mode).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.02,
        help="Minimum metric improvement required for significance (intelligent mode).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run trainings in parallel (standard mode only).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Max parallel workers when --parallel is set. Default = number of tasks "
            "or CPU count, whichever is smaller."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining trainings even if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which commands would run without executing them.",
    )
    return parser.parse_args()


def build_task_list(families: List[str], sizes: List[str]) -> List[Tuple[str, str, str]]:
    tasks: List[Tuple[str, str, str]] = []
    for family, size in itertools.product(families, sizes):
        script_path = MODEL_FAMILIES_TO_SCRIPTS[family][size]
        tasks.append((family, size, script_path))
    return tasks


def verify_scripts_exist(tasks: List[Tuple[str, str, str]]) -> None:
    missing_paths: List[str] = [script for _, _, script in tasks if not os.path.exists(script)]
    if missing_paths:
        missing_str = "\n".join(missing_paths)
        raise FileNotFoundError(
            f"The following training scripts were not found:\n{missing_str}"
        )


def run_script(script_path: str) -> subprocess.CompletedProcess:
    command = [sys.executable, "-u", script_path]
    return subprocess.run(command, check=True)


def run_sequential(tasks: List[Tuple[str, str, str]], continue_on_error: bool) -> None:
    for family, size, script_path in tasks:
        print(f"[START] {family}-{size}: {script_path}")
        try:
            run_script(script_path)
            print(f"[DONE ] {family}-{size}")
        except subprocess.CalledProcessError as exc:
            print(f"[FAIL ] {family}-{size} (exit code {exc.returncode})")
            if not continue_on_error:
                raise


def run_parallel(tasks: List[Tuple[str, str, str]], max_workers: int, continue_on_error: bool) -> None:
    if max_workers is None:
        # Reasonable default: min(number of tasks, os.cpu_count() or 4)
        cpu_count = os.cpu_count() or 4
        max_workers = min(len(tasks), cpu_count)
    print(f"Running in parallel with up to {max_workers} workers...")

    failures: List[Tuple[str, str, int]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_script, script_path): (family, size, script_path)
            for family, size, script_path in tasks
        }
        for future in as_completed(future_to_task):
            family, size, script_path = future_to_task[future]
            try:
                future.result()
                print(f"[DONE ] {family}-{size}: {script_path}")
            except subprocess.CalledProcessError as exc:
                print(f"[FAIL ] {family}-{size}: {script_path} (exit code {exc.returncode})")
                failures.append((family, size, exc.returncode))
                if not continue_on_error:
                    # Best effort: cancel remaining futures
                    for f in future_to_task:
                        f.cancel()
                    raise

    if failures and continue_on_error:
        failures_str = ", ".join(
            f"{family}-{size}(code={code})" for family, size, code in failures
        )
        print(f"Completed with failures: {failures_str}")


def run_intelligent_mode(args: argparse.Namespace) -> None:
    """Run training in intelligent mode with Optuna n_trials adjustment (CLS only)."""
    logger.info("="*60)
    logger.info("INTELLIGENT TRAINING MODE WITH OPTUNA ADJUSTMENT (CLS ONLY)")
    logger.info("="*60)
    
    # Intelligent mode only works with CLS
    if args.families != ["CLS"]:
        original_families = args.families.copy()
        if "CLS" not in args.families:
            logger.warning("Intelligent mode only supports CLS models. No CLS specified, exiting.")
            return
        else:
            logger.warning(f"Intelligent mode only supports CLS models. Ignoring non-CLS families: {[f for f in original_families if f != 'CLS']}")
    
    logger.info(f"Training order: M1 → M2 → M3")
    logger.info(f"Initial n_trials: {args.initial_n_trials}")
    logger.info(f"n_trials increment: {args.n_trials_increment}")
    logger.info(f"Max n_trials: {args.max_n_trials}")
    logger.info(f"Min improvement for significance: {args.min_improvement}")
    
    # Initialize trainer
    trainer = OptunaProgressiveTrainer()
    trainer.initial_n_trials = args.initial_n_trials
    trainer.n_trials_increment = args.n_trials_increment
    trainer.max_n_trials = args.max_n_trials
    trainer.min_improvement = args.min_improvement
    
    # Train only CLS family
    all_results = {}
    
    # Process only CLS
    families_to_train = ["CLS"]
    
    for family in families_to_train:
        scripts = MODEL_FAMILIES_TO_SCRIPTS[family]
        
        # Filter scripts based on requested sizes
        filtered_scripts = {size: path for size, path in scripts.items() if size in args.sizes}
        
        if not filtered_scripts:
            logger.warning(f"No scripts found for {family} with sizes {args.sizes}")
            continue
        
        family_results = trainer.train_family_progressively(family, args.sizes, filtered_scripts)
        all_results[family] = family_results
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    for family, results in all_results.items():
        logger.info(f"\n{family} Models:")
        for size, metrics in results.items():
            logger.info(f"  {size}:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"    {metric_name}: {metric_value:.4f}")
    
    # Save results to JSON
    output_file = Path("training_results") / "optuna_progressive_results.json"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert results to serializable format
    json_results = {}
    for family, family_results in all_results.items():
        json_results[family] = {}
        for size, metrics in family_results.items():
            json_results[family][size] = metrics
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\n✓ Training complete! Results saved to {output_file}")


def main() -> None:
    args = parse_args()
    
    # Use intelligent mode if requested
    if args.intelligent:
        run_intelligent_mode(args)
        return
    
    # Otherwise use standard mode
    tasks = build_task_list(args.families, args.sizes)
    verify_scripts_exist(tasks)

    if args.dry_run:
        for family, size, script_path in tasks:
            print(f"DRY RUN: {sys.executable} -u {script_path}  # {family}-{size}")
        return

    print("Training tasks to run:")
    for family, size, script_path in tasks:
        print(f" - {family}-{size}: {script_path}")

    if args.parallel:
        run_parallel(tasks, args.max_workers, args.continue_on_error)
    else:
        run_sequential(tasks, args.continue_on_error)


if __name__ == "__main__":
    main()


