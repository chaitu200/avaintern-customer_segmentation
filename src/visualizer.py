"""
=============================================================
MODULE: visualizer.py
PURPOSE: Visualize the K-Means clusters and provide insights
=============================================================
After clustering, we need to:
  1. Plot the clusters with different colours
  2. Show the cluster centroids
  3. Create a 3-D scatter
  4. Generate a radar/spider chart
  5. Print a cluster analysis report
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

FIGURES_DIR = "outputs"
PALETTE = {
    "dark"  : "#1E1E2E",
    "light" : "#F5F5F5",
    "warning": "#F9A825",
}

# One colour per cluster (up to 8)
CLUSTER_COLORS = [
    "#6C63FF",   # Purple
    "#FF6584",   # Pink
    "#43B89C",   # Teal
    "#F9A825",   # Amber
    "#A855F7",   # Violet
    "#06B6D4",   # Cyan
    "#F97316",   # Orange
    "#10B981",   # Emerald
]

# Human-readable names assigned AFTER looking at cluster statistics
# These will be set dynamically in the analysis function
CLUSTER_NAMES = {}


def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _dark_axes(fig, ax_list):
    fig.patch.set_facecolor(PALETTE["dark"])
    for ax in ax_list:
        ax.set_facecolor("#2A2A3E")
        ax.tick_params(colors=PALETTE["light"])
        ax.xaxis.label.set_color(PALETTE["light"])
        ax.yaxis.label.set_color(PALETTE["light"])
        ax.title.set_color(PALETTE["light"])
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 1 : 2-D Cluster Scatter (Income vs Spending Score)
# ─────────────────────────────────────────────────────────────────────────────
def plot_2d_clusters(df: pd.DataFrame, features: list, kmeans):
    """
    Scatter plot of clusters coloured by cluster label.
    """
    print("\n📊 Plotting 2-D cluster scatter …")
    _ensure_dir()

    x_col, y_col = features[0], features[1]
    n_clusters = df["Cluster"].nunique()

    fig, ax = plt.subplots(figsize=(11, 7))
    _dark_axes(fig, [ax])

    for cluster_id in sorted(df["Cluster"].unique()):
        mask = df["Cluster"] == cluster_id
        name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        ax.scatter(
            df.loc[mask, x_col],
            df.loc[mask, y_col],
            c=color, s=70, alpha=0.80,
            edgecolors="none", label=name
        )

    # Compute centroids in the original (unscaled) space
    centroids = df.groupby("Cluster")[features].mean().values
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="*",
        s=350,
        c=PALETTE["warning"],
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        label="Centroids"
    )

    ax.set_xlabel(x_col.replace("_", " "), fontsize=13)
    ax.set_ylabel(y_col.replace("_", " "), fontsize=13)
    ax.set_title(f"K-Means Customer Clusters  (K = {n_clusters})\n"
                 f"Axes: {x_col}  vs  {y_col}",
                 fontsize=14, fontweight="bold")

    ax.legend(
        facecolor="#2A2A3E", labelcolor=PALETTE["light"],
        fontsize=10, markerscale=1.2, framealpha=0.9
    )

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "09_2d_clusters.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 2 : 3-D Cluster Scatter (Age, Income, Spending)
# ─────────────────────────────────────────────────────────────────────────────
def plot_3d_clusters(df: pd.DataFrame):
    """
    3-D scatter plot using Age, Annual Income, and Spending Score.
    """
    print("\n📊 Plotting 3-D cluster scatter …")
    _ensure_dir()

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(PALETTE["dark"])

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#2A2A3E")
    ax.grid(color="#444466", linestyle="--", linewidth=0.5)

    for cluster_id in sorted(df["Cluster"].unique()):
        mask = df["Cluster"] == cluster_id
        name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        ax.scatter(
            df.loc[mask, "Age"],
            df.loc[mask, "Annual_Income"],
            df.loc[mask, "Spending_Score"],
            c=color, s=50, alpha=0.80, label=name
        )

    ax.set_xlabel("Age", color=PALETTE["light"], labelpad=10)
    ax.set_ylabel("Annual Income (k$)", color=PALETTE["light"], labelpad=10)
    ax.set_zlabel("Spending Score", color=PALETTE["light"], labelpad=10)
    ax.set_title("3-D Customer Cluster View\n(Age, Income, Spending Score)",
                 color=PALETTE["light"], fontsize=14, fontweight="bold", pad=20)

    ax.tick_params(colors=PALETTE["light"])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.legend(facecolor="#2A2A3E", labelcolor=PALETTE["light"], fontsize=9,
              loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "10_3d_clusters.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 3 : Cluster Statistics Bar Charts
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_statistics(df: pd.DataFrame):
    """
    Bar charts comparing the mean Age, Income, and Spending Score for each cluster.
    """
    print("\n📊 Plotting cluster statistics bar charts …")
    _ensure_dir()

    # Compute the mean of each numeric feature per cluster
    cluster_stats = df.groupby("Cluster")[["Age", "Annual_Income", "Spending_Score"]].mean().round(1)

    metrics = ["Age", "Annual_Income", "Spending_Score"]
    titles  = ["Average Age", "Average Annual Income (k$)", "Average Spending Score"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Cluster Comparison – Mean Feature Values",
                 fontsize=16, color=PALETTE["light"], fontweight="bold")
    _dark_axes(fig, axes)

    for ax, metric, title in zip(axes, metrics, titles):
        cluster_ids = cluster_stats.index.tolist()
        values      = cluster_stats[metric].values
        colors      = [CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] for cid in cluster_ids]
        x_labels    = [CLUSTER_NAMES.get(cid, f"C{cid}") for cid in cluster_ids]

        bars = ax.bar(x_labels, values, color=colors, edgecolor=PALETTE["dark"],
                      linewidth=1, width=0.6)

        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom",
                color=PALETTE["light"], fontsize=10, fontweight="bold"
            )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "11_cluster_statistics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 4 : Cluster Gender Composition
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_gender(df: pd.DataFrame):
    """
    Stacked bar chart: for each cluster, show the male/female breakdown.
    Useful to know if certain clusters skew towards a specific gender.
    """
    print("\n📊 Plotting cluster gender composition …")
    _ensure_dir()

    # Normalise gender values (handle 'M'/'F' and 'Male'/'Female')
    gender_map = {"M": "Male", "F": "Female"}
    gender_col = df["Gender"].map(gender_map).fillna(df["Gender"])

    # Cross-tabulate cluster vs gender
    ct = pd.crosstab(df["Cluster"], gender_col)
    cluster_ids = ct.index.tolist()
    x_labels    = [CLUSTER_NAMES.get(cid, f"C{cid}") for cid in cluster_ids]

    fig, ax = plt.subplots(figsize=(10, 6))
    _dark_axes(fig, [ax])
    fig.suptitle("Gender Composition per Cluster",
                 fontsize=15, color=PALETTE["light"], fontweight="bold")

    male_counts   = ct.get("Male",   pd.Series(0, index=cluster_ids)).values
    female_counts = ct.get("Female", pd.Series(0, index=cluster_ids)).values
    x = np.arange(len(cluster_ids))
    width = 0.35

    bars_m = ax.bar(x - width/2, male_counts,   width, label="Male",
                    color="#5B9BD5", edgecolor=PALETTE["dark"])
    bars_f = ax.bar(x + width/2, female_counts, width, label="Female",
                    color="#ED7D31", edgecolor=PALETTE["dark"])

    for bars in [bars_m, bars_f]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                        ha="center", va="bottom",
                        color=PALETTE["light"], fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.legend(facecolor="#2A2A3E", labelcolor=PALETTE["light"])

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "12_cluster_gender.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 5 : Radar / Spider Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar_chart(df: pd.DataFrame):
    """
    Radar (spider) chart: a polygon for each cluster showing
    normalised mean values of Age, Income, and Spending Score.

    A radar chart is great for comparing multiple groups across
    multiple dimensions all in one view.
    """
    print("\n📊 Plotting radar chart …")
    _ensure_dir()

    features = ["Age", "Annual_Income", "Spending_Score"]
    cluster_stats = df.groupby("Cluster")[features].mean()

    # Normalise to 0–1 scale so all features are comparable on the radar
    norm_stats = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())

    n_features  = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]   # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(PALETTE["dark"])
    ax.set_facecolor("#2A2A3E")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one polygon per cluster
    for cluster_id, row in norm_stats.iterrows():
        values = row.values.tolist()
        values += values[:1]   # Close the polygon
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        name  = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Age", "Annual Income", "Spending Score"],
                       color=PALETTE["light"], fontsize=12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"],
                       color="#888", fontsize=8)
    ax.grid(color="#444466", linewidth=0.8)
    ax.set_title("Radar Chart – Cluster Profiles (Normalised)",
                 color=PALETTE["light"], fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#2A2A3E", labelcolor=PALETTE["light"], fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "13_radar_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION : Cluster Analysis Report (printed to console)
# ─────────────────────────────────────────────────────────────────────────────
def cluster_analysis_report(df: pd.DataFrame):
    """
    Print a detailed text report describing each cluster.
    For each cluster we show:
      - Size & percentage of total customers
      - Mean Age, Income, Spending Score
      - Gender breakdown
      - A business interpretation
    """
    print("\n" + "=" * 60)
    print("  STEP 6: CLUSTER ANALYSIS REPORT")
    print("=" * 60)

    cluster_stats = df.groupby("Cluster").agg(
        Count=("CustomerID", "count"),
        Age_Mean=("Age", "mean"),
        Income_Mean=("Annual_Income", "mean"),
        Spending_Mean=("Spending_Score", "mean"),
    ).round(1)

    total = len(df)

    # Auto-assign descriptive names based on Income & Spending Score medians
    income_median  = df["Annual_Income"].median()
    spending_median = df["Spending_Score"].median()

    for cid, row in cluster_stats.iterrows():
        inc   = row["Income_Mean"]
        spend = row["Spending_Mean"]

        if   inc > income_median  and spend > spending_median:
            name = "High Value Targets 💎"
        elif inc > income_median  and spend <= spending_median:
            name = "Conservative Earners 🏦"
        elif inc <= income_median and spend > spending_median:
            name = "Impulse Buyers 🛍️"
        elif inc <= income_median and spend <= spending_median:
            name = "Budget Conscious 💰"
        else:
            name = "Average Shoppers 🛒"

        CLUSTER_NAMES[cid] = name

    print()
    for cluster_id in sorted(df["Cluster"].unique()):
        row   = cluster_stats.loc[cluster_id]
        name  = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        count = int(row["Count"])
        pct   = count / total * 100

        gender_breakdown = df[df["Cluster"] == cluster_id]["Gender"].value_counts()

        print(f"┌─ Cluster {cluster_id}: {name}")
        print(f"│   Size          : {count} customers  ({pct:.1f}% of total)")
        print(f"│   Avg Age       : {row['Age_Mean']:.1f} years")
        print(f"│   Avg Income    : ${row['Income_Mean']:.1f}k")
        print(f"│   Avg Spending  : {row['Spending_Mean']:.1f} / 100")
        print(f"│   Gender Split  : {gender_breakdown.to_dict()}")
        print(f"└{'─'*55}\n")

    return CLUSTER_NAMES


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_visualization(df: pd.DataFrame, features: list, kmeans):
    """
    Run all visualization and analysis steps.

    Parameters:
        df       (pd.DataFrame): DataFrame with 'Cluster' column
        features (list)        : Column names used for clustering
        kmeans                 : Trained KMeans model
    """
    print("\n" + "=" * 60)
    print("  STEP 6-7: VISUALIZATION & CLUSTER ANALYSIS")
    print("=" * 60)

    # 1. Analyze clusters first (this sets CLUSTER_NAMES)
    cluster_analysis_report(df)

    # 2. All plots
    plot_2d_clusters(df, features, kmeans)
    plot_3d_clusters(df)
    plot_cluster_statistics(df)
    plot_cluster_gender(df)
    plot_radar_chart(df)

    print("\n✅ All visualizations complete!  Saved to outputs/")
