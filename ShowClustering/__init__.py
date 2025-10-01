import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
from sklearn import metrics
from scipy.spatial.distance import cdist


def _to_matrix(df: pd.DataFrame, drop_cols=('datetime', 'order_id', 'order_total')):
    """
    Convert df to numeric matrix X after dropping common non-feature columns.
    Returns (X, feature_names).
    """
    cols = [c for c in df.columns if c not in set(drop_cols)]
    X = df[cols].select_dtypes(include=[np.number]).to_numpy()
    return X, cols


def meanshift_diagrams(
    df: pd.DataFrame,
    bandwidth: float | None = None,
    quantile: float = 0.2,
    bin_seeding: bool = True,
    drop_cols=('datetime', 'order_id', 'order_total')
):
    """
    Fit MeanShift and plot:
      - 2D scatter (first two features)
      - 3D scatter (first three features), if available
    Returns dict with labels, centers, n_clusters, bandwidth, feature_names.
    """
    X, cols = _to_matrix(df, drop_cols)
    if X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("Need at least 1 row and 2 numeric columns after dropping.")

    if bandwidth is None:
        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=min(500, len(X)))

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
    labels = ms.fit_predict(X)  # fit + labels
    centers = ms.cluster_centers_
    n_clusters = len(np.unique(labels))

    # 2D plot (first two features)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', s=100, linewidths=2)
    plt.title(f'MeanShift: {n_clusters} clusters (bw={bandwidth:.3f})')
    plt.xlabel(cols[0]); plt.ylabel(cols[1]); plt.grid(True, alpha=.3)
    plt.show()

    # 3D plot (optional if >= 3 features)
    if X.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=15)
        cz = centers[:, 2] if centers.shape[1] >= 3 else np.zeros(len(centers))
        ax.scatter(centers[:, 0], centers[:, 1], cz, marker='x', c='red', s=100, linewidths=2)
        ax.set_title('MeanShift 3D (first 3 features)')
        ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1]); ax.set_zlabel(cols[2] if len(cols) >= 3 else 'z')
        plt.show()

    return {
        'labels': labels,
        'centers': centers,
        'n_clusters': n_clusters,
        'bandwidth': bandwidth,
        'feature_names': cols,
    }


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