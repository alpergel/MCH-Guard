import logging
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def load_and_preprocess(size: SizeName) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    data_path, _ = _dataset_paths(size)
    df = pd.read_csv(data_path)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].apply(lambda c: c.fillna(c.median()))
    df = df.dropna()
    X = df.drop(columns=[c for c in ["MCH_pos", "MCH_count", "RID", "SCANDATE"] if c in df.columns])
    y = df["MCH_pos"].astype(int)
    return X, y, df


def split_and_scale(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Tuple[pd.Index, pd.Index, pd.Index]]:
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=123, stratify=y_temp)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(np.asarray(X_train))
    X_val_s = scaler.transform(np.asarray(X_val))
    X_test_s = scaler.transform(np.asarray(X_test))
    return X_train_s, y_train.to_numpy(), X_val_s, y_val.to_numpy(), X_test_s, y_test.to_numpy(), list(X.columns), (X_train.index, X_val.index, X_test.index)


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 37)
    best_thr, best_f1 = 0.5, -np.inf
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
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


def compute_masks(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "TP": (y_true == 1) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
    }


def load_feature_summaries(size: SizeName, base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diffs_fp = pd.read_csv(base_dir / f"feature_diffs_{size}_FP.csv") if (base_dir / f"feature_diffs_{size}_FP.csv").exists() else pd.DataFrame()
    diffs_fn = pd.read_csv(base_dir / f"feature_diffs_{size}_FN.csv") if (base_dir / f"feature_diffs_{size}_FN.csv").exists() else pd.DataFrame()
    bins_fp = pd.read_csv(base_dir / f"feature_bins_{size}_FP.csv") if (base_dir / f"feature_bins_{size}_FP.csv").exists() else pd.DataFrame()
    bins_fn = pd.read_csv(base_dir / f"feature_bins_{size}_FN.csv") if (base_dir / f"feature_bins_{size}_FN.csv").exists() else pd.DataFrame()
    return diffs_fp, diffs_fn, bins_fp, bins_fn


def select_top_features(diffs_fp: pd.DataFrame, diffs_fn: pd.DataFrame, top_k: int = 10, for_violin: int = 4) -> Tuple[List[str], List[str]]:
    def score_df(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if "cohens_d" in df.columns and df["cohens_d"].notna().any():
            df["score"] = df["cohens_d"].abs()
        else:
            df["score"] = df["abs_diff"].abs() if "abs_diff" in df.columns else df["diff"].abs()
        return df.sort_values("score", ascending=False)

    fp_s = score_df(diffs_fp)
    fn_s = score_df(diffs_fn)
    top_bar = list(pd.unique(pd.concat([fp_s.head(top_k)["feature"], fn_s.head(top_k)["feature"]], ignore_index=True))) if not fp_s.empty or not fn_s.empty else []
    top_violin = list(pd.unique(pd.concat([fp_s.head(for_violin)["feature"], fn_s.head(for_violin)["feature"]], ignore_index=True))) if not fp_s.empty or not fn_s.empty else []
    return top_bar, top_violin


def fig_diverging_bars(df: pd.DataFrame, title: str) -> go.Figure:
    if df.empty:
        return go.Figure()
    df = df.copy()
    df = df.sort_values("abs_diff" if "abs_diff" in df.columns else "diff", ascending=True)
    color = np.where(df["diff"] >= 0, "#2ca02c", "#d62728")
    fig = go.Figure(go.Bar(
        x=df["diff"],
        y=df["feature"],
        orientation='h',
        marker_color=color,
        hovertemplate="feature=%{y}<br>diff=%{x:.3f}<br>mean_fail=%{customdata[0]:.3f}<br>mean_correct=%{customdata[1]:.3f}<extra></extra>",
        customdata=np.stack([
            df.get("mean_fail", pd.Series([np.nan]*len(df))),
            df.get("mean_correct", pd.Series([np.nan]*len(df)))
        ], axis=1)
    ))
    fig.update_layout(title=title, xaxis_title="mean_fail - mean_correct", yaxis_title="feature", height=500)
    return fig


def fig_heatmap_effects(diffs_fp: pd.DataFrame, diffs_fn: pd.DataFrame, features: List[str], title: str) -> go.Figure:
    if not features:
        return go.Figure()
    mat = []
    for feat in features:
        row = []
        for df in (diffs_fp, diffs_fn):
            if not df.empty and feat in set(df["feature"]):
                val = float(df.loc[df["feature"] == feat, "cohens_d"].values[0]) if "cohens_d" in df.columns else np.nan
            else:
                val = np.nan
            row.append(val)
        mat.append(row)
    z = np.array(mat)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["FP", "FN"],
        y=features,
        colorscale="RdBu",
        zmid=0,
        colorbar_title="Cohen's d"
    ))
    fig.update_layout(title=title, height=300 + 20 * len(features))
    return fig


def fig_violin_feature(X_test_raw: pd.DataFrame, mask_fail: np.ndarray, mask_correct: np.ndarray, feature: str, title: str) -> go.Figure:
    df = pd.DataFrame({
        "value": X_test_raw[feature].values,
        "group": np.where(mask_fail, "Failed (FP|FN)", np.where(mask_correct, "Correct (TP|TN)", "Other")),
    })
    df = df[df["group"] != "Other"]
    fig = px.violin(df, x="group", y="value", box=True, points="all", hover_data=["group"], color="group", color_discrete_map={"Failed (FP|FN)": "#d62728", "Correct (TP|TN)": "#1f77b4"})
    fig.update_layout(title=title, xaxis_title="", yaxis_title=feature, height=350)
    return fig


def fig_failrate_bins(df_bins: pd.DataFrame, feature: str, title: str) -> go.Figure:
    if df_bins.empty or feature not in set(df_bins["feature"]):
        return go.Figure()
    sub = df_bins[df_bins["feature"] == feature].copy()
    sub = sub.sort_values("bin")
    fig = go.Figure(go.Bar(x=sub["bin"], y=sub["fail_rate"], marker_color="#ff7f0e"))
    fig.update_layout(title=title, xaxis_title="quantile bin", yaxis_title="fail rate", height=350)
    return fig


def build_and_save_report(size: SizeName, out_dir: Path, top_k: int = 10, violin_k: int = 4) -> None:
    base_dir = out_dir
    diffs_fp, diffs_fn, bins_fp, bins_fn = load_feature_summaries(size, base_dir)

    if diffs_fp.empty and diffs_fn.empty:
        logger.info(f"[{size}] No diffs found; skipping report")
        return

    X, y, df_full = load_and_preprocess(size)
    X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, feats, (idx_tr, idx_va, idx_te) = split_and_scale(X, y)

    _, model_path = _dataset_paths(size)
    model = joblib.load(model_path)

    y_val_proba = model.predict_proba(X_val_s)[:, 1]
    thr = find_best_threshold(y_val, y_val_proba) if size in ("m2", "m1") else 0.5
    y_test_proba = model.predict_proba(X_test_s)[:, 1]
    y_test_pred = (y_test_proba >= thr).astype(int)
    masks = compute_masks(y_test, y_test_pred)

    X_test_raw = X.loc[idx_te, :]
    mask_fail = (masks["FP"] | masks["FN"]).astype(bool)
    mask_corr = (masks["TP"] | masks["TN"]).astype(bool)

    # Top features
    top_bar, top_violin = select_top_features(diffs_fp, diffs_fn, top_k=top_k, for_violin=violin_k)

    figs: List[Tuple[str, go.Figure]] = []

    if not diffs_fp.empty:
        figs.append((f"{size.upper()} – FP: top {top_k} mean differences", fig_diverging_bars(diffs_fp.sort_values(("abs_diff" if "abs_diff" in diffs_fp.columns else "diff"), ascending=False).head(top_k), title="FP: mean_fail - mean_correct")))
    if not diffs_fn.empty:
        figs.append((f"{size.upper()} – FN: top {top_k} mean differences", fig_diverging_bars(diffs_fn.sort_values(("abs_diff" if "abs_diff" in diffs_fn.columns else "diff"), ascending=False).head(top_k), title="FN: mean_fail - mean_correct")))

    heat_features = top_bar[: min(20, len(top_bar))]
    if heat_features:
        figs.append((f"{size.upper()} – Effect sizes (Cohen's d) for top features", fig_heatmap_effects(diffs_fp, diffs_fn, heat_features, title="Effect sizes (FP/FN)")))

    for feat in top_violin:
        if feat in X_test_raw.columns:
            figs.append((f"{size.upper()} – Distribution: {feat}", fig_violin_feature(X_test_raw, mask_fail, mask_corr, feat, title=f"{feat} – Failed vs Correct")))
            if not bins_fp.empty and feat in set(bins_fp["feature"]):
                figs.append((f"{size.upper()} – FP fail rate by bin: {feat}", fig_failrate_bins(bins_fp, feat, title=f"FP fail rate – {feat}")))
            if not bins_fn.empty and feat in set(bins_fn["feature"]):
                figs.append((f"{size.upper()} – FN fail rate by bin: {feat}", fig_failrate_bins(bins_fn, feat, title=f"FN fail rate – {feat}")))

    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"failure_report_{size}.html"

    parts: List[str] = []
    parts.append("<html><head><meta charset='utf-8'><title>CLS Failure Report</title></head><body>")
    parts.append(f"<h1>CLS Failure Report – {size.upper()}</h1>")
    for title, fig in figs:
        parts.append(f"<h2>{title}</h2>")
        parts.append(pio.to_html(fig, include_plotlyjs='cdn', full_html=False))
    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info(f"Saved report → {out_path}")


def main():
    base_out = Path("viz") / "classification_results" / "error_analysis"
    base_out.mkdir(parents=True, exist_ok=True)
    for size in ["m1", "m2", "m3"]:
        try:
            logger.info(f"Building failure report for: {size}")
            build_and_save_report(size, base_out, top_k=10, violin_k=4)
        except Exception as e:
            logger.warning(f"Skipped {size}: {e}")


if __name__ == "__main__":
    main()


