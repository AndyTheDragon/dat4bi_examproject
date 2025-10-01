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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    "plot_pca_scatter_2d",
    "plot_centroids_over_pca",
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
    plt.figure()
    plt.bar(unique, counts)
    plt.title("Cluster sizes")
    plt.xlabel("Cluster label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_pca_scatter_2d(X_encoded: np.ndarray, labels: np.ndarray, title: str = "PCA (2D) of clusters") -> None:
    """
    Scatter plot of data projected to 2 principal components. Use after encoding/scaling.
    Pass the transformed matrix 'X_encoded' (e.g., result.pipeline.named_steps['prep'].transform(X)).
    """
    pca2 = PCA(n_components=2, random_state=42)
    Z2 = pca2.fit_transform(X_encoded)
    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=12)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
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
