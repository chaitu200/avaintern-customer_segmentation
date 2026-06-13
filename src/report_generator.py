"""
=============================================================
MODULE: report_generator.py
PURPOSE: Generate the final project report as a text file
=============================================================
After clustering and analysis, this module writes a complete
academic-style report to:
    reports/Project_Report.txt
=============================================================
"""

import os
import datetime
import pandas as pd


REPORTS_DIR = "reports"


def generate_report(df: pd.DataFrame, n_clusters: int,
                    features: list, cluster_names: dict,
                    silhouette_val: float):
    """
    Generate and save the full project report.

    Parameters:
        df             (pd.DataFrame): DataFrame with Cluster labels
        n_clusters     (int)         : Number of clusters used
        features       (list)        : Feature names used for clustering
        cluster_names  (dict)        : {cluster_id: descriptive_name}
        silhouette_val (float)       : Silhouette score of the final model
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    cluster_stats = df.groupby("Cluster").agg(
        Count=("CustomerID", "count"),
        Age_Mean=("Age", "mean"),
        Income_Mean=("Annual_Income", "mean"),
        Spending_Mean=("Spending_Score", "mean"),
    ).round(2)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    def w(text=""):
        lines.append(text)

    border = "=" * 65

    w(border)
    w("   CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
    w("   Internship Project – Final Report")
    w(f"   Generated on : {timestamp}")
    w(border)

    # ── PROJECT ABSTRACT ───────────────────────────────────────────────────
    w()
    w("■ PROJECT ABSTRACT")
    w("-" * 65)
    w("""
This project implements an end-to-end Customer Segmentation system using
the K-Means unsupervised machine-learning algorithm applied to the Mall
Customers dataset.  By analysing customer demographics (Age, Annual Income,
Spending Score, Gender) we partition the customer base into distinct,
actionable segments.  The optimal number of clusters was determined through
the Elbow Method and validated with the Silhouette Score.  The results
provide data-driven insights that marketing teams can use to tailor their
campaigns, improve customer retention, and maximise revenue.
    """.strip())

    # ── PROBLEM STATEMENT ─────────────────────────────────────────────────
    w()
    w("■ PROBLEM STATEMENT")
    w("-" * 65)
    w("""
Retail businesses serve thousands of customers with vastly different needs,
spending habits, and income levels.  A one-size-fits-all marketing strategy
wastes budget and generates poor results.  The problem is:

  "How can a mall identify meaningful groups of customers automatically,
   so that targeted marketing strategies can be designed for each group?"

This is an unsupervised learning problem because we have no predefined
labels – we let the data reveal its own structure.
    """.strip())

    # ── OBJECTIVES ────────────────────────────────────────────────────────
    w()
    w("■ OBJECTIVES")
    w("-" * 65)
    w("""
1. Load and clean the Mall Customers dataset.
2. Perform Exploratory Data Analysis (EDA) to understand the data.
3. Select the most informative features for clustering.
4. Determine the optimal number of clusters using the Elbow Method.
5. Train a K-Means model and assign cluster labels to each customer.
6. Visualize the clusters and their characteristics.
7. Derive business insights and actionable recommendations.
    """.strip())

    # ── METHODOLOGY ───────────────────────────────────────────────────────
    w()
    w("■ METHODOLOGY")
    w("-" * 65)
    w(f"""
Step 1 – Data Loading
  • Loaded Mall_Customers.csv using pandas.read_csv().
  • Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns.

Step 2 – Data Cleaning
  • Renamed columns to shorter, code-friendly names.
  • Identified and imputed missing values (median for numeric,
    mode for categorical columns).
  • Encoded Gender (Male=1, Female=0).
  • Removed duplicate rows.

Step 3 – Exploratory Data Analysis
  • Plotted histograms, box plots, scatter plots, and a correlation
    heatmap to understand data distributions and relationships.

Step 4 – Feature Selection
  • Selected features: {features}
  • Rationale: Annual Income and Spending Score together capture the
    most business-relevant distinctions between customer groups.

Step 5 – Feature Scaling
  • Applied StandardScaler (zero mean, unit variance).
  • Scaling ensures that features with larger numeric ranges do not
    dominate the distance calculations in K-Means.

Step 6 – Elbow Method
  • Trained K-Means for K = 1 to 12.
  • Plotted WCSS (inertia) and Silhouette Score vs K.
  • Determined optimal K = {n_clusters} from the elbow point.

Step 7 – K-Means Training
  • Trained the final model with K = {n_clusters}, n_init=20, random_state=42.
  • Silhouette Score = {silhouette_val:.4f}

Step 8 – Cluster Analysis & Visualisation
  • Produced 2-D and 3-D cluster scatter plots.
  • Created bar charts, radar charts, and gender breakdown plots.
  • Printed a detailed cluster statistics report.
    """.strip())

    # ── ALGORITHM ─────────────────────────────────────────────────────────
    w()
    w("■ ALGORITHM: K-MEANS CLUSTERING")
    w("-" * 65)
    w("""
K-Means is an iterative, centroid-based clustering algorithm.

Algorithm Steps:
  1. Choose K random data points as initial centroids.
  2. Assign each data point to the nearest centroid (Euclidean distance).
  3. Recompute each centroid as the mean of its assigned points.
  4. Repeat 2–3 until centroids do not move (convergence).

Objective Function (minimised):
  WCSS = Σ Σ ||x_i − μ_k||²
  where x_i is a data point and μ_k is its cluster centroid.

Time Complexity : O(n · k · t · d)
  n = number of data points
  k = number of clusters
  t = number of iterations
  d = number of dimensions (features)

Evaluation Metrics:
  • WCSS (Inertia)   – lower is better; used in the Elbow Method.
  • Silhouette Score – ranges [-1, 1]; higher is better.
    """.strip())

    # ── DATASET SUMMARY ───────────────────────────────────────────────────
    w()
    w("■ DATASET SUMMARY")
    w("-" * 65)
    w(f"  File          : Mall_Customers.csv")
    w(f"  Total Records : {df.shape[0]}")
    w(f"  Features Used : {features}")
    w()
    w("  Statistical Summary:")
    summary = df[["Age", "Annual_Income", "Spending_Score"]].describe().round(2)
    w(summary.to_string())

    # ── RESULTS ───────────────────────────────────────────────────────────
    w()
    w()
    w("■ RESULTS")
    w("-" * 65)
    w(f"  Optimal Number of Clusters : {n_clusters}")
    w(f"  Silhouette Score           : {silhouette_val:.4f}")
    w()
    w("  Cluster Summary:")
    w(f"  {'Cluster':<10} {'Name':<30} {'Size':>6} {'%':>6} "
      f"{'Avg Age':>8} {'Avg Inc':>9} {'Avg Spend':>10}")
    w("  " + "-" * 80)
    total = len(df)
    for cid, row in cluster_stats.iterrows():
        name  = cluster_names.get(cid, f"Cluster {cid}")
        count = int(row["Count"])
        pct   = count / total * 100
        w(f"  {cid:<10} {name:<30} {count:>6} {pct:>5.1f}% "
          f"{row['Age_Mean']:>8.1f} {row['Income_Mean']:>9.1f} "
          f"{row['Spending_Mean']:>10.1f}")

    # ── BUSINESS INSIGHTS ─────────────────────────────────────────────────
    w()
    w()
    w("■ BUSINESS INSIGHTS & RECOMMENDATIONS")
    w("-" * 65)
    insights = [
        ("High Value Targets 💎",
         "These customers have high income AND high spending score.\n"
         "  → STRATEGY: VIP loyalty programs, exclusive offers, premium products.\n"
         "  → They are the mall's most profitable segment – retain them at all costs."),

        ("Conservative Earners 🏦",
         "High income but low spending – they visit but don't buy much.\n"
         "  → STRATEGY: Personalised discounts on premium items they already browse.\n"
         "  → Emphasise quality and value to convert browsing into purchases."),

        ("Impulse Buyers 🛍️",
         "Low income but high spending – enthusiastic shoppers on a budget.\n"
         "  → STRATEGY: Flash sales, instalment payment options, loyalty points.\n"
         "  → They respond well to emotional marketing and FOMO campaigns."),

        ("Budget Conscious 💰",
         "Low income and low spending – price-sensitive customers.\n"
         "  → STRATEGY: Coupons, everyday-low-price promotions, bundle deals.\n"
         "  → Focus on volume and footfall rather than per-item margin."),

        ("Average Shoppers 🛒",
         "Middle-of-the-road on both income and spending.\n"
         "  → STRATEGY: Seasonal campaigns, gift cards, referral bonuses.\n"
         "  → Nurture this segment; with the right nudge they can move up."),
    ]
    for name, insight in insights:
        w(f"\n  [{name}]")
        w(f"  {insight}")

    # ── CONCLUSION ────────────────────────────────────────────────────────
    w()
    w()
    w("■ CONCLUSION")
    w("-" * 65)
    w(f"""
This project successfully applied K-Means Clustering to segment {df.shape[0]}
mall customers into {n_clusters} distinct groups.  The Elbow Method identified
K = {n_clusters} as the optimal number of clusters, achieving a Silhouette
Score of {silhouette_val:.4f}.

Key findings:
  • Customer behaviour is well-explained by just two features:
    Annual Income and Spending Score.
  • Clear separation exists between groups such as
    "High Value Targets" and "Budget Conscious" customers.
  • Gender does not strongly determine spending behaviour;
    both genders appear in all clusters.

These insights empower mall management to allocate marketing budgets
efficiently, design targeted campaigns, and improve overall customer
experience.
    """.strip())

    # ── FUTURE SCOPE ──────────────────────────────────────────────────────
    w()
    w()
    w("■ FUTURE SCOPE")
    w("-" * 65)
    w("""
1. Add more features such as purchase history, visit frequency,
   and product category preferences for richer segmentation.

2. Try alternative clustering algorithms:
   • DBSCAN  – handles irregular cluster shapes and noise.
   • Agglomerative Hierarchical Clustering – no need to specify K upfront.
   • Gaussian Mixture Models (GMM) – soft cluster assignments.

3. Incorporate real-time data pipelines (e.g., Apache Kafka + Spark)
   to update cluster assignments as new customers arrive.

4. Build a web dashboard (Flask/Streamlit) that lets business analysts
   explore cluster insights interactively.

5. Develop a personalised recommendation engine on top of the segments.
    """.strip())

    w()
    w(border)
    w("   END OF REPORT")
    w(border)

    # Write to file
    report_path = os.path.join(REPORTS_DIR, "Project_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Full project report saved → {report_path}")
    return report_path
