"""
=============================================================
MODULE: clustering.py
PURPOSE: K-Means Clustering – the core ML algorithm
=============================================================
# Handles data scaling, K-Means elbow method, and cluster assignment
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FIGURES_DIR = "outputs"
PALETTE = {
    "primary"  : "#6C63FF",
    "secondary": "#FF6584",
    "accent"   : "#43B89C",
    "warning"  : "#F9A825",
    "dark"     : "#1E1E2E",
    "light"    : "#F5F5F5",
}

# Colour palette for clusters (up to 8 clusters)
CLUSTER_COLORS = [
    "#6C63FF", "#FF6584", "#43B89C", "#F9A825",
    "#A855F7", "#06B6D4", "#F97316", "#10B981"
]


def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 1 : Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(X: np.ndarray) -> tuple:
    """
    Standardise the feature matrix so that every feature has mean=0 and std=1.

    Parameters:
        X (np.ndarray): Raw feature matrix (n_customers × n_features)

    Returns:
        X_scaled   (np.ndarray): Scaled feature matrix
        scaler     (StandardScaler): Fitted scaler (needed later to inverse-transform)
    """
    print("\n✅ Scaling features with StandardScaler …")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"   Before scaling – Mean: {X.mean(axis=0).round(2)},  "
          f"Std: {X.std(axis=0).round(2)}")
    print(f"   After  scaling – Mean: {X_scaled.mean(axis=0).round(4)},  "
          f"Std: {X_scaled.std(axis=0).round(4)}")

    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 2 : Elbow Method
# ─────────────────────────────────────────────────────────────────────────────
def elbow_method(X_scaled: np.ndarray, max_k: int = 12) -> int:
    """
    Use the Elbow Method to find the optimal number of clusters (K).

    Parameters:
        X_scaled (np.ndarray): Scaled feature matrix
        max_k    (int)        : Maximum K value to try (default 12)

    Returns:
    """
    print("\n" + "=" * 60)
    print("  STEP 4: ELBOW METHOD – FINDING OPTIMAL K")
    print("=" * 60)

    wcss_values = []        # List to store inertia for each K
    silhouette_values = []  # List to store silhouette score for each K ≥ 2
    k_range = range(1, max_k + 1)

    print(f"\n  Training K-Means for K = 1 to {max_k} …\n")
    print(f"  {'K':>3} | {'WCSS':>12} | {'Silhouette':>12}")
    print("  " + "-" * 32)

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        wcss_values.append(km.inertia_)

        if k >= 2:
            labels = km.labels_
            sil = silhouette_score(X_scaled, labels)
            silhouette_values.append(sil)
            print(f"  {k:>3} | {km.inertia_:>12.2f} | {sil:>12.4f}")
        else:
            silhouette_values.append(None)
            print(f"  {k:>3} | {km.inertia_:>12.2f} | {'N/A':>12}")

    # ── Automatically detect the elbow using distance to line ──────────
    p1 = np.array([k_range[0], wcss_values[0]])
    p2 = np.array([k_range[-1], wcss_values[-1]])
    distances = []
    for i in range(len(k_range)):
        p = np.array([k_range[i], wcss_values[i]])
        dist = np.linalg.norm(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
    optimal_k = k_range[np.argmax(distances)]
    print(f"\n  📌 Suggested optimal K (auto-detected elbow) = {optimal_k}")

    # ── Plot the elbow curve ────────────────────────────────────────────────
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Elbow Method & Silhouette Scores",
                 fontsize=15, color=PALETTE["light"], fontweight="bold")
    fig.patch.set_facecolor(PALETTE["dark"])

    for ax in axes:
        ax.set_facecolor("#2A2A3E")
        ax.tick_params(colors=PALETTE["light"])
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

    # Left: WCSS vs K (Elbow curve)
    axes[0].plot(list(k_range), wcss_values,
                 color=PALETTE["primary"], linewidth=2.5, marker="o",
                 markerfacecolor=PALETTE["secondary"], markersize=7)
    axes[0].axvline(optimal_k, color=PALETTE["warning"], linestyle="--",
                    linewidth=2, label=f"Elbow @ K={optimal_k}")
    axes[0].set_title("WCSS vs Number of Clusters (Elbow Curve)",
                      color=PALETTE["light"], fontsize=12)
    axes[0].set_xlabel("Number of Clusters (K)", color=PALETTE["light"])
    axes[0].set_ylabel("WCSS (Within-Cluster Sum of Squares)", color=PALETTE["light"])
    axes[0].legend(facecolor="#2A2A3E", labelcolor=PALETTE["light"])

    # Right: Silhouette Score vs K
    sil_k   = list(range(2, max_k + 1))
    sil_val = [v for v in silhouette_values if v is not None]
    axes[1].plot(sil_k, sil_val,
                 color=PALETTE["accent"], linewidth=2.5, marker="s",
                 markerfacecolor=PALETTE["warning"], markersize=7)
    best_sil_k = sil_k[int(np.argmax(sil_val))]
    axes[1].axvline(best_sil_k, color=PALETTE["secondary"], linestyle="--",
                    linewidth=2, label=f"Best Silhouette @ K={best_sil_k}")
    axes[1].set_title("Silhouette Score vs Number of Clusters",
                      color=PALETTE["light"], fontsize=12)
    axes[1].set_xlabel("Number of Clusters (K)", color=PALETTE["light"])
    axes[1].set_ylabel("Silhouette Score (higher = better)", color=PALETTE["light"])
    axes[1].legend(facecolor="#2A2A3E", labelcolor=PALETTE["light"])

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "08_elbow_silhouette.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")

    return optimal_k


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 3 : Train Final K-Means Model
# ─────────────────────────────────────────────────────────────────────────────
def train_kmeans(X_scaled: np.ndarray, n_clusters: int) -> tuple:
    """
    Train the final K-Means model with the chosen number of clusters.

    Parameters:
        X_scaled   (np.ndarray): Scaled feature matrix
        n_clusters (int)       : Number of clusters (chosen from elbow method)

    Returns:
        tuple: (Trained KMeans model object, Silhouette score)
    """
    print("\n" + "=" * 60)
    print(f"  STEP 5: TRAINING K-MEANS WITH K = {n_clusters}")
    print("=" * 60)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42, max_iter=500)
    kmeans.fit(X_scaled)

    print(f"\n✅ K-Means trained successfully!")
    print(f"   Iterations to converge : {kmeans.n_iter_}")
    print(f"   Final WCSS (Inertia)   : {kmeans.inertia_:.2f}")

    sil = silhouette_score(X_scaled, kmeans.labels_)
    print(f"   Silhouette Score       : {sil:.4f}  (range -1 to 1; closer to 1 is better)")

    return kmeans, sil


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 4 : Assign Cluster Labels to DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def assign_clusters(df: pd.DataFrame, kmeans: KMeans,
                    X_scaled: np.ndarray) -> pd.DataFrame:
    """
    Add a 'Cluster' column to the DataFrame with each customer's cluster number.

    Parameters:
        df       (pd.DataFrame): Cleaned customer DataFrame
        kmeans   (KMeans)      : Trained K-Means model
        X_scaled (np.ndarray)  : The same scaled features used for training

    Returns:
        pd.DataFrame: DataFrame with a new 'Cluster' column (0-indexed integers)
    """
    df = df.copy()
    df["Cluster"] = kmeans.labels_

    print("\n✅ Cluster labels assigned to DataFrame.")
    print("\n--- Cluster sizes ---")
    cluster_counts = df["Cluster"].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = count / len(df) * 100
        print(f"   Cluster {cluster_id}: {count:>4} customers  ({pct:.1f}%)")

    return df
