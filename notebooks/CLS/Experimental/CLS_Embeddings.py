import os
from pathlib import Path
from typing import Dict, Tuple, List, Literal
import logging
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.manifold import TSNE

import joblib

try:
    import umap  # type: ignore
    try:
        from umap import UMAP as _UMAP_CLASS  # type: ignore
    except Exception:
        try:
            from umap.umap_ import UMAP as _UMAP_CLASS  # type: ignore
        except Exception:
            _UMAP_CLASS = None
    _UMAP_AVAILABLE = _UMAP_CLASS is not None
except Exception:
    _UMAP_CLASS = None
    _UMAP_AVAILABLE = False

import plotly.express as px


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
SAVE_PNG = False


SizeName = Literal["m1", "m2", "m3"]
EmbedMethod = Literal["umap", "tsne"]


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
    meta = df[[c for c in ["RID", "SCANDATE"] if c in df.columns]].copy()
    meta.index = df.index

    logger.info(f"[{size}] Features shape: {X.shape}; target positive rate: {y.mean():.3f}")
    return X, y, groups, meta


def split_and_scale(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler, Tuple[pd.Index, pd.Index, pd.Index]]:
    # 80/10/10 split with stratification and fixed seed to match training scripts
    logger.info("Splitting into train/val/test (80/10/10, stratified)")
    X_temp, X_test, y_temp, y_test, groups_temp, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=123, stratify=y
    )
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_temp, y_temp, groups_temp, test_size=0.1, random_state=123, stratify=y_temp
    )
    logger.info(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    feature_names = list(X.columns)

    logger.info("Scaling features (StandardScaler)")
    scaler = StandardScaler()
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


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: Literal["f1", "spec"] = "f1") -> float:
    thresholds = np.linspace(0.05, 0.95, 37)
    best_thr, best_score = 0.5, -np.inf
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            score = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr)


def compute_embedding(X: np.ndarray, method: EmbedMethod = "umap", random_state: int = 123) -> np.ndarray:
    if method == "umap":
        if not _UMAP_AVAILABLE or _UMAP_CLASS is None:
            # Fallback: use TSNE if UMAP class is unavailable
            logger.info("UMAP unavailable; falling back to t-SNE")
            tsne = TSNE(n_components=3, perplexity=30, learning_rate="auto", init="pca", random_state=random_state, verbose=1)
            return tsne.fit_transform(X)
        logger.info("Computing UMAP embedding (3D)")
        reducer = _UMAP_CLASS(n_components=3, n_neighbors=15, min_dist=0.1, metric="euclidean", verbose=True)
        try:
            start = time.time()
            return reducer.fit_transform(X)
        except Exception:
            # Last resort fallback to TSNE
            logger.info("UMAP failed; falling back to t-SNE")
            tsne = TSNE(n_components=3, perplexity=30, learning_rate="auto", init="pca", random_state=random_state, verbose=1)
            return tsne.fit_transform(X)
    else:
        logger.info("Computing t-SNE embedding (3D)")
        tsne = TSNE(n_components=3, perplexity=30, learning_rate="auto", init="pca", random_state=random_state, verbose=1)
        return tsne.fit_transform(X)


def build_plot_df(
    embed: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    meta: pd.DataFrame,
    test_indices: pd.Index,
    feature_names: List[str],
    X_test_scaled: np.ndarray,
    top_feature_indices: List[int] | None = None,
) -> pd.DataFrame:
    plot_df = pd.DataFrame({
        "x": embed[:, 0],
        "y": embed[:, 1],
        "z": embed[:, 2],
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }, index=test_indices)

    # Error type for coloring
    def _etype(t, p):
        if t == 1 and p == 1:
            return "TP"
        if t == 0 and p == 0:
            return "TN"
        if t == 0 and p == 1:
            return "FP"
        return "FN"

    plot_df["error_type"] = [
        _etype(int(t), int(p)) for t, p in zip(plot_df["y_true"].values, plot_df["y_pred"].values)
    ]

    # Attach meta (RID/SCANDATE if present)
    plot_df = plot_df.join(meta, how="left")

    # Optionally attach top feature values for hover
    if top_feature_indices is not None and len(top_feature_indices) > 0:
        for i in top_feature_indices:
            fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            plot_df[f"feat::{fname}"] = X_test_scaled[:, i]

    return plot_df.reset_index(names=["index"])  # preserve original index as a column


def plot_and_save(plot_df: pd.DataFrame, size: SizeName, method: EmbedMethod, out_dir: Path) -> None:
    # Ensure base and embeddings subdirectory exist
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / "embeddings"
    save_dir.mkdir(parents=True, exist_ok=True)

    color_map = {"TP": "#2ca02c", "TN": "#1f77b4", "FP": "#d62728", "FN": "#ff7f0e"}

    hover_cols = [c for c in ["RID", "SCANDATE", "y_true", "y_pred", "y_proba", "error_type"] if c in plot_df.columns]
    # include up to 5 feature columns if present
    hover_cols += [c for c in plot_df.columns if c.startswith("feat::")][:5]

    fig = px.scatter_3d(
        plot_df,
        x="x",
        y="y",
        z="z",
        color="error_type",
        color_discrete_map=color_map,
        symbol="y_true",
        opacity=0.85,
        hover_data=hover_cols,
        title=f"{size.capitalize()} CLS – 3D {method.upper()} (colored by error type)",
        height=800,
    )
    fig.update_traces(marker=dict(size=4))

    html_path = save_dir /  f"embeddings_{size}_{method}.html"
    csv_path = save_dir / f"embeddings_{size}_{method}.csv"

    logger.info(f"Saving HTML → {html_path}")
    fig.write_html(str(html_path))
    logger.info(f"Saving CSV → {csv_path}")
    plot_df.to_csv(csv_path, index=False)

    # Optional static image if explicitly enabled; disabled by default to avoid Windows kaleido hangs
    if SAVE_PNG:
        try:
            png_path = save_dir / f"embeddings_{size}_{method}.png"
            logger.info(f"Saving PNG → {png_path}")
            fig.write_image(str(png_path), scale=2)
        except Exception as e:
            logger.warning(f"PNG export failed: {e}")


def top_important_feature_indices(model, k: int = 5) -> List[int]:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        order = np.argsort(importances)[::-1]
        return [int(i) for i in order[:k]]
    return []


def run_for_size(size: SizeName, methods: List[EmbedMethod] | None = None) -> Dict[str, Path]:
    if methods is None:
        methods = ["umap" if _UMAP_AVAILABLE else "tsne"]

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

    # Load model
    _, model_path = _dataset_paths(size)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info(f"[{size}] Loading model from {model_path}")
    model = joblib.load(model_path)

    # Preds on validation to optionally tune threshold (m2/m1 used threshold optimisation in training eval)
    logger.info(f"[{size}] Scoring validation set for threshold selection")
    y_val_proba = model.predict_proba(X_val_s)[:, 1]
    # Default strategy: use F1-tuned threshold for m2/m1, 0.5 for m3
    if size in ("m2", "m1"):
        thr = find_best_threshold(y_val, y_val_proba, metric="f1")
    else:
        thr = 0.5
    logger.info(f"[{size}] Using decision threshold: {thr:.3f}")

    logger.info(f"[{size}] Scoring test set")
    y_test_proba = model.predict_proba(X_test_s)[:, 1]
    y_test_pred = (y_test_proba >= thr).astype(int)

    # Prepare feature list for hover (top-k important)
    top_idx = top_important_feature_indices(model, k=5)

    results: Dict[str, Path] = {}
    out_dir = Path("viz") / "classification_results"

    for method in methods:
        logger.info(f"[{size}] Computing embedding with method={method}")
        t0 = time.time()
        embed = compute_embedding(X_test_s, method=method)
        logger.info(f"[{size}] Embedding done in {time.time() - t0:.1f}s; building plot dataframe")
        plot_df = build_plot_df(
            embed=embed,
            y_true=y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
            meta=meta,
            test_indices=idx_test,
            feature_names=feature_names,
            X_test_scaled=X_test_s,
            top_feature_indices=top_idx,
        )
        plot_and_save(plot_df, size=size, method=method, out_dir=out_dir)
        results[method] = out_dir

    return results


def main():
    sizes: List[SizeName] = ["m1", "m2", "m3"]
    for size in sizes:
        try:
            logger.info(f"Starting embeddings for: {size}")
            run_for_size(size)
            logger.info(f"Completed embeddings for: {size} (output: viz/classification_results/)")
        except Exception as e:
            logger.warning(f"Skipped {size}: {e}")


if __name__ == "__main__":
    main()


