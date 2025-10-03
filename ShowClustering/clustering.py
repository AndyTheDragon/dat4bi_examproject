"""
clustering.py

A small toolbox to run MeanShift clustering on restaurant-style POS data
with sensible preprocessing and easy visuals. Designed to be imported and
used from a Jupyter notebook (.ipynb).

Key features:
- Ignores columns like 'datetime' and 'order_id' by default.
- Handles numeric + categorical features (one-hot) with imputation and robust scaling.
- Optional log1p transform for heavy-tailed positive numeric columns.
- PCA for decorrelation.
- MeanShift bandwidth selection via simple quantile grid search.
- Post-processing: merges tiny clusters and fuses near-duplicate centroids.
- Easy plotting helpers (matplotlib-only).

Dependencies: pandas, numpy, scikit-learn, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

__all__ = [
    "DEFAULT_IGNORE_COLS",
    "DEFAULT_NUMERIC_COLS",
    "DEFAULT_CATEGORICAL_COLS",
    "load_table",
    "prepare_features",
    "MeanShiftResult",
    "run_meanshift",
    "attach_labels",
    "plot_cluster_sizes",
    "plot_pca_scatter_2d",              # if you export it elsewhere
    "plot_pca_scatter_2d_named",
    "plot_centroids_over_pca",
    "plot_pca_scatter_3d_interactive_named",
    "plot_feature_scatter_3d_interactive",
    # one-liner API
    "show_cluster_sizes",
    "show_pca_scatter_2d",
    "show_pca_scatter_3d_interactive",
    "show_feature_scatter_2d",
    "show_feature_scatter_3d_interactive",
    "show_cluster_overview",
]

# ---------------------------- Data & Features ----------------------------

DEFAULT_IGNORE_COLS = ["datetime", "order_id"]
DEFAULT_NUMERIC_COLS = [
    "order_total",
    "number_of_maindishes",
    "number_of_snacks",
    "number_of_drinks",
    "number_of_soups",
    "number_of_extras",
]
DEFAULT_CATEGORICAL_COLS = ["day_of_week", "payment_method", "is_takeaway"]


def _unique_list(seq: Sequence[str]) -> List[str]:
    """Deduplicate while preserving order."""
    return list(dict.fromkeys(seq))


def _make_ohe() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder with dense output."""
    try:
        # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_table(path: str, sheet: Optional[str | int] = None, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load a CSV or Excel table. If path ends with .xlsx/.xls, uses read_excel; otherwise read_csv.
    For CSV, pass read_csv_kwargs like sep=";", encoding="utf-8-sig".
    """
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    return pd.read_csv(path, **read_csv_kwargs)


def _should_log1p(series: pd.Series) -> bool:
    """Heuristic: log1p only if mostly positive and heavy-tailed."""
    s = pd.to_numeric(series, errors="coerce")
    if (s > 0).mean() <= 0.9:
        return False
    s_pos = s[s > 0]
    if s_pos.empty:
        return False
    # Heavy-tailed if max/median is large
    max_val = np.nanmax(s.values.astype(float))
    median_pos = float(np.median(s_pos.values.astype(float)))
    return (max_val / max(1e-9, median_pos)) > 20


def prepare_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    ignore_cols: Optional[List[str]] = None,
    force_categorical: Optional[List[str]] = None,
    auto_detect_categoricals: bool = False,
    max_auto_cat_cardinality: int = 20,
    apply_log1p: bool = True,
    force_log1p: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build a clean feature table:
    - drops ignore_cols if present
    - numeric columns: optional log1p if heavy-tailed
    - categorical columns: cast to category dtype
    - optionally auto-detect categoricals by low cardinality
    Returns (X, num_cols_used, cat_cols_used)
    """
    ignore_cols = _unique_list((ignore_cols or []) + DEFAULT_IGNORE_COLS)
    keep_df = df.drop(columns=ignore_cols, errors="ignore").copy()

    # Choose columns
    if numeric_cols is None:
        numeric_cols = [c for c in DEFAULT_NUMERIC_COLS if c in keep_df.columns]
    if categorical_cols is None:
        categorical_cols = [c for c in DEFAULT_CATEGORICAL_COLS if c in keep_df.columns]

    if auto_detect_categoricals:
        for c in keep_df.columns:
            if c in numeric_cols or c in categorical_cols:
                continue
            nunique = keep_df[c].nunique(dropna=True)
            if keep_df[c].dtype.kind in "OSbU":
                if nunique <= max_auto_cat_cardinality:
                    categorical_cols.append(c)
            else:
                if nunique <= max_auto_cat_cardinality:
                    categorical_cols.append(c)

    # Ensure forced categoricals included
    if force_categorical:
        for c in force_categorical:
            if c in keep_df.columns:
                categorical_cols.append(c)

    # Remove overlaps & duplicates
    categorical_cols = [c for c in _unique_list(categorical_cols) if c not in set(numeric_cols)]

    X = pd.DataFrame(index=keep_df.index)

    # Numeric
    used_numeric: List[str] = []
    force_log1p_set = set(force_log1p or [])
    for c in numeric_cols:
        if c not in keep_df.columns:
            continue
        s = pd.to_numeric(keep_df[c], errors="coerce")
        should_log = (c in force_log1p_set) or (apply_log1p and _should_log1p(s))
        if should_log:
            X[f"{c}_log1p"] = np.log1p(s.clip(lower=0))
            used_numeric.append(f"{c}_log1p")
        else:
            X[c] = s
            used_numeric.append(c)

    # Categorical
    used_categorical: List[str] = []
    for c in categorical_cols:
        if c not in keep_df.columns:
            continue
        X[c] = keep_df[c].astype("category")
        used_categorical.append(c)

    # Drop rows with all-NaN across selected features
    X = X.dropna(how="all")

    return X, used_numeric, used_categorical


# ---------------------------- Clustering Core ----------------------------

@dataclass
class MeanShiftResult:
    labels: np.ndarray
    n_clusters: int
    cluster_sizes: List[int]
    selected_quantile: float
    selected_bandwidth: float
    pipeline: Pipeline
    Z: np.ndarray
    grid_results: List[Dict[str, Any]]


def run_meanshift(
    X: pd.DataFrame,
    quantiles: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    pca_variance: float = 0.95,
    min_cluster_size_ratio: float = 0.01,
    centroid_merge_eps: float = 0.75,
    sample_for_bw: int = 5000,
    random_state: int = 42,
) -> MeanShiftResult:
    """End-to-end MeanShift with preprocessing, bandwidth search, and post-merging."""
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Pipelines with imputation for robustness
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_ohe()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    steps = [("prep", pre)]
    if pca_variance and pca_variance > 0:
        steps.append(("pca", PCA(n_components=pca_variance, random_state=random_state)))
    pipe = Pipeline(steps)

    Z = pipe.fit_transform(X)
    Z = np.asarray(Z, dtype=float, order="C")
    n = Z.shape[0]

    # Bandwidth grid search
    results: List[Dict[str, Any]] = []
    for q in quantiles:
        try:
            bw = estimate_bandwidth(Z, quantile=float(q), n_samples=min(sample_for_bw, n), random_state=random_state)
        except Exception:
            continue
        if not np.isfinite(bw) or bw <= 0:
            continue
        ms = MeanShift(bandwidth=float(bw), bin_seeding=True, cluster_all=True)
        labels = ms.fit_predict(Z).astype(int, copy=False)
        k = int(np.unique(labels).size)
        sizes = np.bincount(labels)
        top_share = float(sizes.max() / n)
        db = float(davies_bouldin_score(Z, labels)) if k >= 2 else float("inf")
        results.append(
            {
                "quantile": float(q),
                "bandwidth": float(bw),
                "k": k,
                "top_share": top_share,
                "db": db,
                "labels": labels,
            }
        )

    if not results:
        raise RuntimeError("No usable bandwidth found. Try widening quantiles or adjusting features.")

    # Select best by balanced score
    def score(r: Dict[str, Any]) -> float:
        k = int(r["k"])
        if k < 2:
            return float("inf")
        penalty_k = 0.0 if 2 <= k <= 30 else float(abs(k - 16))
        return float(r["db"]) + 2.5 * float(r["top_share"]) + 0.1 * penalty_k

    best = min(results, key=score)
    labels = np.array(best["labels"], dtype=int, copy=True)

    # Merge tiny clusters into nearest big one
    sizes = np.bincount(labels)
    min_size = max(1, int(min_cluster_size_ratio * n))
    small = np.where(sizes < min_size)[0]
    if small.size > 0:
        nc = NearestCentroid().fit(Z, labels)
        centroids = nc.centroids_
        big = np.where(sizes >= min_size)[0]
        for c in small:
            if c in big:
                continue
            d = np.linalg.norm(centroids[big] - centroids[c], axis=1)
            target = int(big[np.argmin(d)])
            labels[labels == c] = target
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(int, copy=False)

    # Merge near-duplicate centroids (version-safe AgglomerativeClustering)
    if np.unique(labels).size > 1:
        nc = NearestCentroid().fit(Z, labels)
        C = nc.centroids_

        kwargs = dict(n_clusters=None, distance_threshold=float(centroid_merge_eps), linkage="average")
        try:
            ag = AgglomerativeClustering(metric="euclidean", **kwargs)  # sklearn >= 1.2
        except TypeError:
            ag = AgglomerativeClustering(affinity="euclidean", **kwargs)  # sklearn < 1.2

        parent = ag.fit_predict(C)
        mapping = {old: new for old, new in enumerate(parent)}
        labels = np.vectorize(mapping.get)(labels)
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(int, copy=False)

    grid_results = [{k: v for k, v in r.items() if k != "labels"} for r in results]
    return MeanShiftResult(
        labels=labels,
        n_clusters=int(np.unique(labels).size),
        cluster_sizes=np.bincount(labels).astype(int).tolist(),
        selected_quantile=float(best["quantile"]),
        selected_bandwidth=float(best["bandwidth"]),
        pipeline=pipe,
        Z=Z,
        grid_results=grid_results,
    )


def attach_labels(df: pd.DataFrame, labels: np.ndarray, column: str = "cluster") -> pd.DataFrame:
    """Return a copy of df with cluster labels attached."""
    out = df.copy()
    out[column] = pd.Series(np.asarray(labels, dtype=int), index=df.index)
    return out


# ---------------------------- Plotting Helpers ----------------------------

def plot_cluster_sizes(labels: np.ndarray) -> None:
    """Bar plot of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    fig = plt.figure()
    plt.bar(unique, counts)
    plt.title("Cluster sizes")
    plt.xlabel("Cluster label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    return fig

def _get_encoded_feature_names_from_prep(pre: ColumnTransformer) -> List[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        names = []
        try:
            num_name, num_trans, num_cols = pre.transformers_[0]
            cat_name, cat_trans, cat_cols = pre.transformers_[1]
        except Exception:
            return names
        if isinstance(num_cols, list):
            names.extend([f"num__{c}" for c in num_cols])
        try:
            cat_feature_names = cat_trans.get_feature_names_out(cat_cols)
            names.extend([f"cat__{n}" for n in cat_feature_names])
        except Exception:
            for c in cat_cols:
                names.append(f"cat__{c}")
        return names

def _pretty_name(n: str) -> str:
    if n.startswith("num__"): n = n[5:]
    if n.startswith("cat__"): n = n[5:]
    return n

def _top_features_for_pc(pca: PCA, feature_names: List[str], pc_index: int, top_k: int = 3) -> List[str]:
    comp = pca.components_[pc_index]
    idx = np.argsort(np.abs(comp))[::-1][:top_k]
    return [_pretty_name(feature_names[i]) for i in idx]

def plot_pca_scatter_2d_named(
    pipeline: Pipeline,
    X: pd.DataFrame,
    labels: np.ndarray,
    top_k: int = 3,
    title_prefix: str = "Clusters (PCA 2D)",
    annotate: bool = True,  # kept for signature compatibility; ignored
    cluster_names: Optional[Dict[int, str] | Sequence[str]] = None,
):
    pre = pipeline.named_steps["prep"]
    X_encoded = pre.transform(X)
    pca = pipeline.named_steps.get("pca")
    if pca is None or getattr(pca, "n_components_", None) is None:
        pca = PCA(n_components=2, random_state=42).fit(X_encoded)
        Z2 = pca.transform(X_encoded)
    else:
        Z_full = pca.transform(X_encoded)
        Z2 = Z_full[:, :2] if Z_full.shape[1] >= 2 else PCA(n_components=2, random_state=42).fit_transform(X_encoded)

    feature_names = _get_encoded_feature_names_from_prep(pre)
    top1 = ", ".join(_top_features_for_pc(pca, feature_names, 0, top_k))
    top2 = ", ".join(_top_features_for_pc(pca, feature_names, 1, top_k))

    # Helper to resolve names
    def name_for(c: int) -> str:
        if isinstance(cluster_names, dict):
            return cluster_names.get(int(c), f"Cluster {int(c)}")
        if isinstance(cluster_names, (list, tuple)) and int(c) < len(cluster_names):
            return str(cluster_names[int(c)])
        return f"Cluster {int(c)}"

    fig = plt.figure()
    plt.title(title_prefix)
    plt.xlabel(f"PC1 (top: {top1})")
    plt.ylabel(f"PC2 (top: {top2})")

    clusters = np.unique(labels)
    cmap = plt.get_cmap("tab10" if len(clusters) <= 10 else "tab20")
    for i, c in enumerate(clusters):
        m = labels == c
        plt.scatter(Z2[m, 0], Z2[m, 1], s=12, color=cmap(i % cmap.N), label=name_for(int(c)), alpha=0.9)

    plt.legend(loc="upper right", frameon=True, title="Clusters")
    plt.tight_layout()
    plt.show()
    return fig

def plot_pca_scatter_3d_named(pipeline: Pipeline, X: pd.DataFrame, labels: np.ndarray, top_k: int = 3, title_prefix: str = "Clusters (PCA 3D)"):
    pre = pipeline.named_steps["prep"]
    X_encoded = pre.transform(X)
    pca = pipeline.named_steps.get("pca")
    if pca is None or getattr(pca, "n_components_", None) is None or getattr(pca, "n_components_", 0) < 3:
        pca = PCA(n_components=3, random_state=42).fit(X_encoded)
        Z3 = pca.transform(X_encoded)
    else:
        Z_full = pca.transform(X_encoded)
        Z3 = Z_full[:, :3] if Z_full.shape[1] >= 3 else PCA(n_components=3, random_state=42).fit_transform(X_encoded)
    feature_names = _get_encoded_feature_names_from_prep(pre)
    top1 = ", ".join(_top_features_for_pc(pca, feature_names, 0, top_k))
    top2 = ", ".join(_top_features_for_pc(pca, feature_names, 1, top_k))
    top3 = ", ".join(_top_features_for_pc(pca, feature_names, 2, top_k))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=12, c=labels)
    ax.set_title(title_prefix)
    ax.set_xlabel(f"PC1 (top: {top1})")
    ax.set_ylabel(f"PC2 (top: {top2})")
    ax.set_zlabel(f"PC3 (top: {top3})")
    plt.tight_layout()
    plt.show()

def plot_feature_scatter_2d(
    X: pd.DataFrame,
    labels: np.ndarray,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
    annotate: bool = True,  # kept for signature compatibility; ignored
    cluster_names: Optional[Dict[int, str] | Sequence[str]] = None,
):
    if title is None:
        title = f"{x_col} vs {y_col}"
    x = X[x_col].values
    y = X[y_col].values

    # Helper to resolve names
    def name_for(c: int) -> str:
        if isinstance(cluster_names, dict):
            return cluster_names.get(int(c), f"Cluster {int(c)}")
        if isinstance(cluster_names, (list, tuple)) and int(c) < len(cluster_names):
            return str(cluster_names[int(c)])
        return f"Cluster {int(c)}"

    fig = plt.figure()
    plt.title(title)
    plt.xlabel(x_col); plt.ylabel(y_col)

    clusters = np.unique(labels)
    cmap = plt.get_cmap("tab10" if len(clusters) <= 10 else "tab20")
    for i, c in enumerate(clusters):
        m = labels == c
        plt.scatter(x[m], y[m], s=12, color=cmap(i % cmap.N), label=name_for(int(c)), alpha=0.9)

    plt.legend(loc="upper right", frameon=True, title="Clusters")
    plt.tight_layout()
    plt.show()
    return fig

def plot_feature_scatter_3d(X: pd.DataFrame, labels: np.ndarray, x_col: str, y_col: str, z_col: str, title: Optional[str] = None):
    if title is None:
        title = f"{x_col} / {y_col} / {z_col}"
    x = X[x_col].values
    y = X[y_col].values
    z = X[z_col].values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=12, c=labels)
    ax.set_title(title)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_zlabel(z_col)
    plt.tight_layout()
    plt.show()

def plot_centroids_over_pca(X_encoded: np.ndarray, labels: np.ndarray, title: str = "Clusters with centroids (2D PCA)") -> None:
    """Like plot_pca_scatter_2d but overlays centroids."""
    pca2 = PCA(n_components=2, random_state=42)
    Z2 = pca2.fit_transform(X_encoded)

    # centroids in this 2D space
    nc = NearestCentroid().fit(Z2, labels)
    C2 = nc.centroids_

    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=12, alpha=0.8)
    plt.scatter(C2[:, 0], C2[:, 1], marker="X", s=120, edgecolors="black")
    for i, (x, y) in enumerate(C2):
        plt.text(x, y, str(i), ha="center", va="center")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def plot_pca_scatter_3d_interactive_named(
    pipeline: Pipeline,
    X: pd.DataFrame,
    labels: np.ndarray,
    top_k: int = 3,
    title_prefix: str = "Clusters (PCA 3D, interactive)",
    show: bool = True,
):
    """
    Interactive 3D PCA scatter with per-cluster legend and centroid overlay.
    Requires: plotly (pip install plotly)

    If show=True (default), displays the figure and returns None.
    If show=False, returns the plotly Figure without displaying (no double output).
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("plotly is required for interactive plots. Install with: pip install plotly") from e

    labels = np.asarray(labels)
    pre = pipeline.named_steps["prep"]
    X_encoded = pre.transform(X)

    # Get or fit a 3D PCA projection
    pca = pipeline.named_steps.get("pca")
    if pca is None or getattr(pca, "n_components_", None) is None or getattr(pca, "n_components_", 0) < 3:
        from sklearn.decomposition import PCA as _PCA
        pca = _PCA(n_components=3, random_state=42).fit(X_encoded)
        Z3 = pca.transform(X_encoded)
    else:
        Z_full = pca.transform(X_encoded)
        Z3 = Z_full[:, :3] if Z_full.shape[1] >= 3 else PCA(n_components=3, random_state=42).fit_transform(X_encoded)

    # Axis labels with top features
    feature_names = _get_encoded_feature_names_from_prep(pre)
    top1 = ", ".join(_top_features_for_pc(pca, feature_names, 0, top_k))
    top2 = ", ".join(_top_features_for_pc(pca, feature_names, 1, top_k))
    top3 = ", ".join(_top_features_for_pc(pca, feature_names, 2, top_k))

    # Build per-cluster traces for legend clarity
    fig = go.Figure()
    Z3x, Z3y, Z3z = Z3[:, 0], Z3[:, 1], Z3[:, 2]
    clusters = np.unique(labels)
    for c in clusters:
        mask = labels == c
        fig.add_scatter3d(
            x=Z3x[mask],
            y=Z3y[mask],
            z=Z3z[mask],
            mode="markers",
            name=f"Cluster {int(c)}",
            marker=dict(size=3, opacity=0.85),
            hovertemplate="Cluster=%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
            text=[str(int(c))] * int(mask.sum()),
        )

    # Centroids in the same 3D space
    nc = NearestCentroid().fit(Z3, labels)
    C3 = nc.centroids_
    fig.add_scatter3d(
        x=C3[:, 0], y=C3[:, 1], z=C3[:, 2],
        mode="markers+text",
        name="Centroids",
        marker=dict(size=8, symbol="x", color="black", line=dict(width=1, color="white")),
        text=[str(i) for i in range(C3.shape[0])],
        textposition="middle center",
        hovertemplate="Centroid %{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
    )

    fig.update_layout(
        title=title_prefix,
        scene=dict(
            xaxis_title=f"PC1 (top: {top1})",
            yaxis_title=f"PC2 (top: {top2})",
            zaxis_title=f"PC3 (top: {top3})",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if show:
        fig.show()
        return None
    return fig

def plot_feature_scatter_3d_interactive(
    X: pd.DataFrame,
    labels: np.ndarray,
    x_col: str,
    y_col: str,
    z_col: str,
    title: Optional[str] = None,
    show: bool = True,
):
    """
    Interactive 3D scatter of raw features with per-cluster legend.
    Requires: plotly (pip install plotly)

    If show=True (default), displays the figure and returns None.
    If show=False, returns the plotly Figure without displaying (no double output).
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("plotly is required for interactive plots. Install with: pip install plotly") from e

    if title is None:
        title = f"{x_col} / {y_col} / {z_col} (interactive)"

    labels = np.asarray(labels)
    x = X[x_col].values
    y = X[y_col].values
    z = X[z_col].values

    fig = go.Figure()
    clusters = np.unique(labels)
    for c in clusters:
        mask = labels == c
        fig.add_scatter3d(
            x=x[mask], y=y[mask], z=z[mask],
            mode="markers",
            name=f"Cluster {int(c)}",
            marker=dict(size=3, opacity=0.85),
            hovertemplate="Cluster=%{text}<br>"
                          f"{x_col}=%{{x}}<br>"
                          f"{y_col}=%{{y}}<br>"
                          f"{z_col}=%{{z}}<extra></extra>",
            text=[str(int(c))] * int(mask.sum()),
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if show:
        fig.show()
        return None
    return fig

# ---------------------------- One-liner "show_*" API ----------------------------

def _default_feature_kwargs() -> Dict[str, Any]:
    return dict(
        numeric_cols=DEFAULT_NUMERIC_COLS,
        categorical_cols=DEFAULT_CATEGORICAL_COLS,
        ignore_cols=DEFAULT_IGNORE_COLS,
        force_categorical=["day_of_week", "payment_method", "is_takeaway"],
        auto_detect_categoricals=False,
        force_log1p=["order_total"],
    )

def _default_cluster_kwargs() -> Dict[str, Any]:
    return dict(
        quantiles=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        pca_variance=0.95,
        min_cluster_size_ratio=0.01,
        centroid_merge_eps=0.75,
        sample_for_bw=5000,
        random_state=42,
    )

def _prepare_and_cluster(df: pd.DataFrame, **kwargs) -> Tuple[MeanShiftResult, pd.DataFrame]:
    # Split kwargs into feature vs clustering
    fkw = _default_feature_kwargs()
    ckw = _default_cluster_kwargs()
    for k in list(kwargs.keys()):
        if k in fkw:
            fkw[k] = kwargs.pop(k)
        if k in ckw:
            ckw[k] = kwargs.pop(k)
    X, _, _ = prepare_features(df, **fkw)
    res = run_meanshift(X, **ckw)
    return res, X

def show_cluster_sizes(df: pd.DataFrame, **kwargs) -> Tuple[MeanShiftResult, pd.DataFrame]:
    res, X = _prepare_and_cluster(df, **kwargs)
    fig = plot_cluster_sizes(res.labels)
    return res, X, fig

def show_pca_scatter_2d(
    df: pd.DataFrame,
    *,
    cluster_names: Optional[Dict[int, str] | Sequence[str]] = None,
    top_k: int = 3,
    title_prefix: str = "Clusters (PCA 2D)",
    **kwargs,
) -> Tuple[MeanShiftResult, pd.DataFrame]:
    res, X = _prepare_and_cluster(df, **kwargs)
    fig = plot_pca_scatter_2d_named(res.pipeline, X, res.labels, top_k=top_k, title_prefix=title_prefix, cluster_names=cluster_names)
    return res, X, fig

def show_pca_scatter_3d_interactive(
    df: pd.DataFrame,
    *,
    cluster_names: Optional[Dict[int, str] | Sequence[str]] = None,  # reserved for future; colors/legend come from labels
    top_k: int = 3,
    title_prefix: str = "Clusters (PCA 3D, interactive)",
    show: bool = True,
    **kwargs,
) -> Tuple[MeanShiftResult, pd.DataFrame]:
    res, X = _prepare_and_cluster(df, **kwargs)
    plot_pca_scatter_3d_interactive_named(res.pipeline, X, res.labels, top_k=top_k, title_prefix=title_prefix, show=show)
    return res, X

def show_feature_scatter_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    cluster_names: Optional[Dict[int, str] | Sequence[str]] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Tuple[MeanShiftResult, pd.DataFrame]:
    res, X = _prepare_and_cluster(df, **kwargs)
    fig = plot_feature_scatter_2d(X, res.labels, x_col=x_col, y_col=y_col, title=title, cluster_names=cluster_names)
    return res, X, fig

def show_feature_scatter_3d_interactive(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    *,
    title: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Tuple[MeanShiftResult, pd.DataFrame]:
    res, X = _prepare_and_cluster(df, **kwargs)
    plot_feature_scatter_3d_interactive(X, res.labels, x_col=x_col, y_col=y_col, z_col=z_col, title=title, show=show)
    return res, X

def show_cluster_overview(
    df: pd.DataFrame,
    *,
    cluster_names: Optional[Dict[int, str] | Sequence[str]] = None,
    include_interactive_3d: bool = True,
    **kwargs,
) -> Tuple[MeanShiftResult, pd.DataFrame]:
    res, X = _prepare_and_cluster(df, **kwargs)
    # sizes
    plot_cluster_sizes(res.labels)
    # 2D PCA with legend
    plot_pca_scatter_2d_named(res.pipeline, X, res.labels, top_k=3, title_prefix="Clusters (PCA 2D)", cluster_names=cluster_names)
    # interactive 3D PCA
    if include_interactive_3d:
        plot_pca_scatter_3d_interactive_named(res.pipeline, X, res.labels, top_k=3, title_prefix="Clusters (PCA 3D, interactive)", show=True)
    return res, X
