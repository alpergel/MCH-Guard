import os
from pathlib import Path
from typing import Dict, Tuple, List, Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import joblib
import logging


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SizeName = Literal["m1", "m2", "m3"]


def _dataset_paths(size: SizeName) -> Tuple[Path, Path]:
    data_path = {
        "m1": Path("processed/classification_m1.csv"),
        "m2": Path("processed/classification_m2.csv"),
        "m3": Path("processed/classification_m3.csv"),
    }[size]

    model_path = {
        "m1": Path("models/m1_rf_model.pkl"),
        "m2": Path("models/m2_rf_model.pkl"),
        "m3": Path("models/m3_rf_model.pkl"),
    }[size]
    return data_path, model_path


def load_and_preprocess(size: SizeName) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    data_path, _ = _dataset_paths(size)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"[{size}] Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"[{size}] Loaded shape: {df.shape}")

    # Impute numerics using median (to mirror training scripts)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda c: c.fillna(c.median()))
    before_drop = len(df)
    df = df.dropna()
    dropped = before_drop - len(df)
    if dropped > 0:
        logger.info(f"[{size}] Dropped {dropped} rows after imputation due to remaining NaNs")

    # Split features/target
    drop_cols = ["MCH_pos", "MCH_count", "RID", "SCANDATE"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["MCH_pos"].astype(int)
    groups = df["RID"]
    meta_cols = [c for c in ["RID", "SCANDATE"] if c in df.columns]
    meta = df[meta_cols].copy()
    meta.index = df.index

    logger.info(f"[{size}] Features shape: {X.shape}; target positive rate: {y.mean():.3f}")
    return X, y, groups, meta


def split_and_scale(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler, Tuple[pd.Index, pd.Index, pd.Index]]:
    logger.info("Splitting into train/val/test (80/10/10, stratified)")
    X_temp, X_test, y_temp, y_test, groups_temp, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=123, stratify=y
    )
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_temp, y_temp, groups_temp, test_size=0.1, random_state=123, stratify=y_temp
    )
    logger.info(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    feature_names = list(X.columns)

    scaler = StandardScaler()
    logger.info("Scaling features (StandardScaler) for model scoring")
    X_train_scaled = scaler.fit_transform(np.asarray(X_train))
    X_val_scaled = scaler.transform(np.asarray(X_val))
    X_test_scaled = scaler.transform(np.asarray(X_test))

    return (
        X_train_scaled,
        y_train.to_numpy(),
        X_val_scaled,
        y_val.to_numpy(),
        X_test_scaled,
        y_test.to_numpy(),
        feature_names,
        scaler,
        (X_train.index, X_val.index, X_test.index),
    )


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 37)
    best_thr, best_f1 = 0.5, -np.inf
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        # F1 for the positive class (minimize FN)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def compute_error_masks(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    mask_tp = (y_true == 1) & (y_pred == 1)
    mask_tn = (y_true == 0) & (y_pred == 0)
    mask_fp = (y_true == 0) & (y_pred == 1)
    mask_fn = (y_true == 1) & (y_pred == 0)
    return {"TP": mask_tp, "TN": mask_tn, "FP": mask_fp, "FN": mask_fn}


def summarize_feature_differences(X_test_raw: pd.DataFrame, mask_fail: np.ndarray, mask_correct: np.ndarray, top_k: int = 50) -> pd.DataFrame:
    numeric_cols = X_test_raw.select_dtypes(include=[np.number]).columns
    rows: List[Dict[str, float]] = []
    for col in numeric_cols:
        x_fail = X_test_raw.loc[mask_fail, col].values
        x_corr = X_test_raw.loc[mask_correct, col].values
        if x_fail.size < 2 or x_corr.size < 2:
            continue
        mean_fail = float(np.mean(x_fail))
        mean_corr = float(np.mean(x_corr))
        std_fail = float(np.std(x_fail, ddof=1))
        std_corr = float(np.std(x_corr, ddof=1))
        diff = mean_fail - mean_corr
        # Cohen's d (pooled SD)
        n1, n2 = x_fail.size, x_corr.size
        denom = ((n1 - 1) * (std_fail ** 2) + (n2 - 1) * (std_corr ** 2))
        pooled_var = denom / (n1 + n2 - 2) if (n1 + n2 - 2) > 0 else 0.0
        pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else np.nan
        cohens_d = diff / pooled_sd if pooled_sd and not np.isnan(pooled_sd) and pooled_sd != 0 else np.nan
        rows.append({
            "feature": col,
            "n_fail": n1,
            "n_correct": n2,
            "mean_fail": mean_fail,
            "mean_correct": mean_corr,
            "diff": diff,
            "abs_diff": abs(diff),
            "std_fail": std_fail,
            "std_correct": std_corr,
            "cohens_d": float(cohens_d) if cohens_d is not np.nan else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("abs_diff", ascending=False).head(top_k)
    return df


def summarize_feature_bins(X_test_raw: pd.DataFrame, mask_fail: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    numeric_cols = X_test_raw.select_dtypes(include=[np.number]).columns
    records: List[Dict[str, float]] = []
    for col in numeric_cols:
        series = X_test_raw[col]
        try:
            binned = pd.qcut(series, q=n_bins, duplicates='drop')
        except Exception:
            # If not enough unique values to bin, skip
            continue
        grp = pd.DataFrame({
            "bin": binned.astype(str),
            "is_fail": mask_fail.astype(int)
        })
        agg = grp.groupby("bin").agg(n=("is_fail", "size"), n_fail=("is_fail", "sum")).reset_index()
        agg["fail_rate"] = agg["n_fail"] / agg["n"].replace(0, np.nan)
        agg["feature"] = col
        records.append(agg)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    # Order columns
    out = out[["feature", "bin", "n", "n_fail", "fail_rate"]]
    return out


def run_for_size(size: SizeName, out_dir: Path, top_k: int = 50, n_bins: int = 5) -> None:
    logger.info(f"==== {size.upper()} ====")
    X, y, groups, meta = load_and_preprocess(size)
    (
        X_train_s,
        y_train,
        X_val_s,
        y_val,
        X_test_s,
        y_test,
        feature_names,
        scaler,
        (idx_train, idx_val, idx_test),
    ) = split_and_scale(X, y, groups)

    # Raw test DataFrame (unscaled) aligned to test indices
    X_test_raw = X.loc[idx_test, :].copy()

    # Load model
    _, model_path = _dataset_paths(size)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info(f"[{size}] Loading model from {model_path}")
    model = joblib.load(model_path)

    # Threshold: match embeddings pipeline (F1-tuned for m2/m1; 0.5 for m3)
    logger.info(f"[{size}] Scoring validation set for threshold selection")
    y_val_proba = model.predict_proba(X_val_s)[:, 1]
    thr = find_best_threshold(y_val, y_val_proba) if size in ("m2", "m1") else 0.5
    logger.info(f"[{size}] Using decision threshold: {thr:.3f}")

    logger.info(f"[{size}] Scoring test set")
    y_test_proba = model.predict_proba(X_test_s)[:, 1]
    y_test_pred = (y_test_proba >= thr).astype(int)

    masks = compute_error_masks(y_test, y_test_pred)

    # Save failed cases
    df_failed = pd.DataFrame({
        "RID": meta.loc[idx_test, "RID"].values if "RID" in meta.columns else idx_test.values,
        "SCANDATE": meta.loc[idx_test, "SCANDATE"].values if "SCANDATE" in meta.columns else None,
        "y_true": y_test,
        "y_pred": y_test_pred,
        "y_proba": y_test_proba,
        "error_type": np.where(masks["FP"], "FP", np.where(masks["FN"], "FN", np.where(masks["TP"], "TP", "TN")))
    })
    failed_path = out_dir / f"failed_cases_{size}.csv"
    logger.info(f"[{size}] Saving failed cases → {failed_path}")
    df_failed.to_csv(failed_path, index=False)

    # Separate masks
    mask_fp = masks["FP"]
    mask_fn = masks["FN"]
    mask_correct = masks["TP"] | masks["TN"]

    # Summaries for FP
    logger.info(f"[{size}] Summarizing feature differences for FP")
    diffs_fp = summarize_feature_differences(X_test_raw, mask_fp, mask_correct, top_k=top_k)
    if not diffs_fp.empty:
        diffs_fp_path = out_dir / f"feature_diffs_{size}_FP.csv"
        diffs_fp.to_csv(diffs_fp_path, index=False)
        logger.info(f"[{size}] Saved → {diffs_fp_path}")

    bins_fp = summarize_feature_bins(X_test_raw, mask_fp, n_bins=n_bins)
    if not bins_fp.empty:
        bins_fp_path = out_dir / f"feature_bins_{size}_FP.csv"
        bins_fp.to_csv(bins_fp_path, index=False)
        logger.info(f"[{size}] Saved → {bins_fp_path}")

    # Summaries for FN
    logger.info(f"[{size}] Summarizing feature differences for FN")
    diffs_fn = summarize_feature_differences(X_test_raw, mask_fn, mask_correct, top_k=top_k)
    if not diffs_fn.empty:
        diffs_fn_path = out_dir / f"feature_diffs_{size}_FN.csv"
        diffs_fn.to_csv(diffs_fn_path, index=False)
        logger.info(f"[{size}] Saved → {diffs_fn_path}")

    bins_fn = summarize_feature_bins(X_test_raw, mask_fn, n_bins=n_bins)
    if not bins_fn.empty:
        bins_fn_path = out_dir / f"feature_bins_{size}_FN.csv"
        bins_fn.to_csv(bins_fn_path, index=False)
        logger.info(f"[{size}] Saved → {bins_fn_path}")


def main():
    out_dir = Path("viz") / "classification_results" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes: List[SizeName] = ["m1", "m2", "m3"]
    for size in sizes:
        try:
            run_for_size(size, out_dir=out_dir, top_k=50, n_bins=5)
            logger.info(f"Completed error analysis for: {size} (output: {out_dir})")
        except Exception as e:
            logger.warning(f"Skipped {size}: {e}")


if __name__ == "__main__":
    main()


