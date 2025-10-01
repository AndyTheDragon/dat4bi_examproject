import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


def _to_matrix(df: pd.DataFrame, drop_cols=('datetime', 'order_id')):
    """
    Convert df to numeric matrix X after dropping common non-feature columns.
    Returns (X, feature_names).
    """
    cols = [c for c in df.columns if c not in set(drop_cols)]
    X = df[cols].select_dtypes(include=[np.number]).to_numpy()
    return X, cols


def meanshift_diagrams(df, bandwidth=None, quantile=0.2, bin_seeding=True,
                       drop_cols=('datetime','order_id')):
    # 1) Matrix + (optional) log1p for order_total
    cols = [c for c in df.columns if c not in set(drop_cols)]
    M = df[cols].select_dtypes(include=[np.number]).copy()
    if 'order_total' in M.columns:
        M['order_total'] = np.log1p(M['order_total'].clip(lower=0))

    # 2) Scale
    X = RobustScaler().fit_transform(M.to_numpy())

    # 3) Bandwidth search (if not provided)
    q_grid = [quantile] if bandwidth is not None else [0.10,0.15,0.20,0.25,0.30]
    best = None
    for q in q_grid:
        bw = bandwidth or estimate_bandwidth(X, quantile=q, n_samples=min(1000, len(X)))
        if not np.isfinite(bw) or bw <= 0: 
            continue
        ms = MeanShift(bandwidth=bw, bin_seeding=bin_seeding)
        lbl = ms.fit_predict(X)
        sizes = np.bincount(lbl)
        top = sizes.max()/len(lbl)
        score = (len(sizes) if len(sizes)>1 else 1e9) + 2.5*top  # simple balance heuristic
        cand = (score, q, bw, lbl, ms.cluster_centers_)
        best = min([best, cand], key=lambda t: t[0]) if best else cand

    _, q_sel, bandwidth, labels, centers = best
    n_clusters = len(np.unique(labels))

    # 4) PCA plot for clarity
    Z2 = PCA(n_components=2, random_state=42).fit_transform(X)
    plt.figure(); plt.scatter(Z2[:,0], Z2[:,1], c=labels, s=20)
    plt.title(f"MeanShift (PCA 2D) — {n_clusters} clusters (q={q_sel}, bw={bandwidth:.3f})")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()

    # 5) Optional 3D if you want
    if X.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        Z3 = PCA(n_components=3, random_state=42).fit_transform(X)
        fig = plt.figure(); ax = fig.add_subplot(projection='3d')
        ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], c=labels, s=10)
        ax.set_title("MeanShift (PCA 3D)"); plt.tight_layout(); plt.show()

    return {"labels": labels, "centers": centers, "n_clusters": n_clusters,
            "bandwidth": bandwidth, "selected_quantile": q_sel, "feature_names": cols}



def kmeans_elbow_diagram(
    df: pd.DataFrame,
    k_range=range(2, 10),
    n_init: int = 10,
    random_state: int | None = 42,
    drop_cols=('datetime', 'order_id', 'order_total')
):
    """
    Plot the elbow diagram (distortion vs K). Returns list of distortions.
    """
    X, _ = _to_matrix(df, drop_cols)
    distortions = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        model.fit(X)
        distortions.append(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1).sum() / X.shape[0])

    plt.figure()
    plt.plot(list(k_range), distortions, 'bx-')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('K'); plt.ylabel('Distortion'); plt.grid(True, alpha=.3)
    plt.show()
    return distortions


def kmeans_silhouette_diagram(
    df: pd.DataFrame,
    k_range=range(2, 10),
    n_init: int = 10,
    random_state: int | None = 42,
    drop_cols=('datetime', 'order_id', 'order_total')
):
    """
    Plot silhouette score vs K. Returns list of scores.
    """
    X, _ = _to_matrix(df, drop_cols)
    scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X)
        score = metrics.silhouette_score(X, labels, metric='euclidean')
        print(f'K={k}: silhouette={score:.3f}')
        scores.append(score)

    plt.figure()
    plt.plot(list(k_range), scores, 'bx-')
    plt.title('Silhouette Score by K')
    plt.xlabel('K'); plt.ylabel('Silhouette'); plt.grid(True, alpha=.3)
    plt.show()
    return scores


def kmeans_cluster_plots(
    df: pd.DataFrame,
    n_clusters: int = 5,
    n_init: int = 20,
    random_state: int | None = 42,
    drop_cols=('datetime', 'order_id', 'order_total')
):
    """
    Fit KMeans and plot each cluster as a separate 2D scatter (first two features).
    Returns dict with labels and the fitted model.
    """
    X, cols = _to_matrix(df, drop_cols)
    if X.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns to plot clusters.")

    km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)

    for i in range(n_clusters):
        cluster = X[labels == i]
        plt.figure()
        plt.scatter(cluster[:, 0], cluster[:, 1], s=30)
        plt.title(f'Cluster {i} (size={cluster.shape[0]})')
        plt.xlabel(cols[0]); plt.ylabel(cols[1]); plt.grid(True, alpha=.3)
        plt.show()

    return {'labels': labels, 'model': km}


def yellowbrick_silhouette_diagram(
    df: pd.DataFrame,
    k: int = 5,
    n_init: int = 10,
    random_state: int | None = 42,
    drop_cols=('datetime', 'order_id', 'order_total')
):
    """
    Visualize per-sample silhouette with Yellowbrick. Returns fitted model or None.
    """
    try:
        from yellowbrick.cluster import SilhouetteVisualizer
    except Exception:
        print("yellowbrick is not installed. Install with: pip install yellowbrick")
        return None

    X, _ = _to_matrix(df, drop_cols)
    model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(X)
    visualizer.show()
    return model