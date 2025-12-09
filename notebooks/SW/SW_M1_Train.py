"""CLS_Switch_Train.py

Features
========
1. Robust data loading / preprocessing (missing values, scaling, SMOTE optional).
2. Baseline XGBoost model + optional Optuna hyper-parameter optimisation.
3. Evaluation utilities (ROC-AUC, Accuracy, F1, confusion matrix).
4. Visualisation utilities (feature importance, Optuna plots).
5. Logging, error handling, model persistence (joblib) and CLI entry-point.

"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    auc,
)
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# GLOBALS & LOGGING CONFIGURATION
# ----------------------------------------------------------------------------

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # Optuna parallelism
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

# colour palette
COLORS = {
    "xgb": "#1f77b4",
}

# ----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------

def gpu_available() -> bool:
    """Return True if an NVIDIA GPU is available (checked via nvidia-smi)."""
    try:
        if platform.system() == "Windows":
            subprocess.check_output("nvidia-smi", stderr=subprocess.STDOUT)
        else:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ----------------------------------------------------------------------------
# DATA LOADING / PREPROCESSING
# ----------------------------------------------------------------------------

def load_and_preprocess_data(filepath: str = "./processed/longitudinal_progression.csv"):
    """Load the longitudinal dataset and return split features/target/groups.

    Switch prediction uses `SWITCH_STATUS` as target.
    Certain identifier columns are dropped.
    Missing numerics are imputed with the median.
    """
    logger.info("Loading data from %s", filepath)
    data = pd.read_csv(filepath)

    # Basic missing value logging
    missing = data.isnull().sum()
    if missing.any():
        logger.info("Missing values per column (non-zero):\n%s", missing[missing > 0])

    # Median impute numeric columns
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data[num_cols].apply(lambda col: col.fillna(col.median()))

    # Drop rows with any still-missing
    before = len(data)
    data = data.dropna()
    logger.info("Dropped %d rows with remaining NaNs", before - len(data))

    # Features / target
    y = data["SWITCH_STATUS"]
    X = data.drop(columns=[
        "RID",
        "SCANDATE",
        'MCH_pos', 'MCH_count',
        "SWITCH_STATUS",
    ])
    groups = data["RID"]  # For grouped CV

    # Class distribution
    dist = y.value_counts(normalize=True)
    logger.info("Class distribution: %s (imbalance 1:%s)", dist.to_dict(), round(dist.min() / dist.max(), 2))
    return X, y, groups


def split_and_scale_data(X, y, groups=None):
    """Train/Val/Test split (80/10/10) + StandardScaler."""
    X_temp, X_test, y_temp, y_test, groups_temp, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=123, stratify=y)

    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_temp, y_temp, groups_temp, test_size=0.125, random_state=123, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, list(X.columns), groups_train


# ----------------------------------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val, perform_hpo: bool = False):
    """Train XGBoost classifier with optional Optuna search."""

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "random_state": 123,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, pred_proba)

    if perform_hpo:
        logger.info("Starting Optuna hyper-parameter search…")
        study = optuna.create_study(direction="maximize", study_name="xgb_switch")
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({"objective": "binary:logistic", "eval_metric": "logloss", "random_state": 123, "n_jobs": -1})
        model = XGBClassifier(**best_params)
        model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))  # retrain on full train+val
        # Visualization disabled to avoid plotly-induced kernel hangs
        log_optuna_results(study, "xgboost")
    else:
        logger.info("Training baseline XGBoost model…")
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.2,
            max_depth=7,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
            random_state=123,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
    return model

# ----------------------------------------------------------------------------
# EVALUATION & VISUALS
# ----------------------------------------------------------------------------

def find_best_threshold(model, X_val, y_val, beta: float = 2.0, min_precision: float = 0.0):
    """Return threshold that maximises F-beta score (beta>1 emphasises recall).
    Optionally enforce a minimum precision.
    """
    proba = model.predict_proba(X_val)[:, 1]
    best_thr, best_score = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        preds = (proba >= thr).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        if prec < min_precision:
            continue
        score = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-9)
        if score > best_score:
            best_score, best_thr = score, thr
    logger.info("Selected threshold %.2f with F%.0f=%.3f (min_precision=%.2f)", best_thr, beta, best_score, min_precision)
    return best_thr


def evaluate_model(model, X_test, y_test, threshold: float | None = None):
    """Return dict of evaluation metrics and plot ROC/PR curves."""
    proba = model.predict_proba(X_test)[:, 1]
    if threshold is None:
        threshold = 0.5
    preds = (proba >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    logger.info("Test Accuracy: %.3f | ROC-AUC: %.3f", acc, roc_auc)
    logger.info("Confusion Matrix:\n%s", cm)

    # Confusion Matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Switch', 'Switch'],
                yticklabels=['No Switch', 'Switch'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    _save_plot("confusion_matrix_small.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}", color=COLORS["xgb"])
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(); plt.tight_layout()
    _save_plot("roc_curve_small.png")
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}", color=COLORS["xgb"])
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend(); plt.tight_layout()
    _save_plot("pr_curve_small.png")
    plt.close()

    return {"accuracy": acc, "roc_auc": roc_auc, "confusion_matrix": cm, "report": report}


def plot_feature_importance(model, feature_names: list[str], top_n: int = 20):
    """Plot a horizontal bar chart of feature importances (same style as CLS_Large_Train)."""
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_. Skipping plot.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    selected_importances = importances[idx]
    selected_features = np.array(feature_names)[idx]

    plt.figure(figsize=(12, 7))
    bars = plt.barh(range(top_n), selected_importances, color="#1f77b4")
    #plt.gca().invert_yaxis()  # highest importance at the top
    plt.yticks(range(top_n), selected_features)
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Features - XGBoost")

    # annotate bars with importance values
    for bar, val in zip(bars, selected_importances):
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    _save_plot("feature_importance_small.png")
    plt.close()


# ------------------------------ OPTUNA HELPERS --------------------------------


def _save_plotly(fig, out_path: Path):
    """Save a plotly Figure robustly. PNG if kaleido present otherwise HTML."""
    try:
        # Lazy import to avoid unnecessary dependency if user does not need PNGs
        import kaleido  # noqa: F401  # type: ignore
        fig.write_image(str(out_path.with_suffix(".png")))
    except (ImportError, ValueError) as exc:
        logger.info("Falling back to HTML for plotly figure (%s)", exc)
        fig.write_html(str(out_path.with_suffix(".html")))




# ----------------------------------------------------------------------------
# MISC HELPERS
# ----------------------------------------------------------------------------

def _save_plot(filename: str, out_dir: str = "viz/switch_results"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(out_dir) / filename)


# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------

def main(args: argparse.Namespace | None = None):
    if args is None:
        parser = argparse.ArgumentParser(description="Train SWITCH_STATUS classifier (XGBoost)")
        parser.add_argument("--data", default="./processed/worsening_small.csv", help="Path to CSV data file")
        parser.add_argument("--hpo", action="store_true", help="Run Optuna hyper-parameter optimisation")
        parser.add_argument("--model-out", default="./models/sw_small_model.joblib", help="Output path for trained model")
        parsed = parser.parse_args()
    else:
        parsed = args

    X, y, groups = load_and_preprocess_data(parsed.data)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, _ = split_and_scale_data(X, y, groups)

    # Save test data for statistical analysis
    test_data = pd.DataFrame(X_test, columns=X.columns)
    test_data['SWITCH_STATUS'] = y_test.reset_index(drop=True)
    test_csv_path = "./processed/SW_small_test.csv"
    
    os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)
    test_data.to_csv(test_csv_path, index=False)
    print(f"Test data saved to {test_csv_path}")

    model = train_xgboost(X_train, y_train, X_val, y_val, perform_hpo=parsed.hpo)

    # optimise threshold on validation for higher recall (beta=2)
    best_thr = find_best_threshold(model, X_val, y_val, beta=2.0, min_precision=0.6)

    metrics = evaluate_model(model, X_test, y_test, threshold=best_thr)
    plot_feature_importance(model, feature_names)

    joblib.dump(model, parsed.model_out)
    logger.info("Model persisted to %s", parsed.model_out)

    logger.info("Pipeline complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
