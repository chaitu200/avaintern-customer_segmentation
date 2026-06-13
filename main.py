"""
=============================================================
FILE: main.py  (Entry Point)
PROJECT: Customer Segmentation Using K-Means Clustering
AUTHOR: Chaitanya reddy manda
DATE: 2026

PURPOSE:
  This is the MAIN script that ties every module together.
  Run this file to execute the entire pipeline from start to finish.

HOW TO RUN:
  Open a terminal in the project root folder and type:
      python main.py

WHAT THIS SCRIPT DOES (in order):
  1.  Loads the Mall Customers dataset
  2.  Cleans the data (handles missing values, encoding)
  3.  Runs Exploratory Data Analysis (7 charts)
  4.  Scales the features for clustering
  5.  Finds the optimal K using the Elbow Method
  6.  Trains the final K-Means model
  7.  Assigns cluster labels to each customer
  8.  Visualizes and analyses the clusters (5 more charts)
  9.  Generates the full project report (TXT file)
  10. Saves the clustered dataset as a CSV
=============================================================
"""

# ── Standard Library Imports ────────────────────────────────────────────────
import os        # For working with file paths and directories
import time      # For measuring how long the pipeline takes

# ── Third-Party Imports ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')   # Run without opening GUI windows
import matplotlib.pyplot as plt
plt.show = lambda: None # Disable plt.show() so it doesn't block execution

# ── Project Module Imports ──────────────────────────────────────────────────
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.data_loader      import load_data, clean_data, get_feature_matrix
from src.eda              import run_eda
from src.clustering       import scale_features, elbow_method, train_kmeans, assign_clusters
from src.visualizer       import run_visualization
from src.report_generator import generate_report


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Path to the dataset (relative to project root)
    "data_path": os.path.join(BASE_DIR, "data", "Mall_Customers.csv"),

    # Features to use for K-Means clustering.
    # We use Annual_Income and Spending_Score because they best
    # separate customers into meaningful business groups.
    "cluster_features": ["Annual_Income", "Spending_Score"],

    # Maximum K value to test during the Elbow Method
    "max_k": 12,

    # Set to True to use the auto-detected elbow K,
    # or set "override_k" to a specific integer to force a K value.
    "use_auto_k": False,
    "override_k": 5,       # only used if use_auto_k = False

    # Where to save the final clustered CSV
    "output_csv": os.path.join("reports", "clustered_customers.csv"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    Main function that orchestrates the entire customer segmentation pipeline.
    Calling this function runs steps 1-10 in order.
    """
    start_time = time.time()   # Record start time

    print("\n" + "█" * 60)
    print("█   CUSTOMER SEGMENTATION – K-MEANS CLUSTERING PIPELINE   █")
    print("█" * 60)
    print(f"\n  Dataset  : {CONFIG['data_path']}")
    print(f"  Features : {CONFIG['cluster_features']}")
    print(f"  Max K    : {CONFIG['max_k']}")
    print()

    # ── STEP 1 & 2: Load and Clean Data ─────────────────────────────────
    raw_df    = load_data(CONFIG["data_path"])
    clean_df  = clean_data(raw_df)

    # ── STEP 3: Exploratory Data Analysis ────────────────────────────────
    run_eda(clean_df)

    # ── STEP 4: Feature Selection & Scaling ──────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: FEATURE SELECTION")
    print("=" * 60)
    print(f"\n  Selected features for clustering: {CONFIG['cluster_features']}")
    print("""
  WHY these two features?
    • Annual_Income  reflects purchasing power
    • Spending_Score reflects actual willingness to spend
    Together they directly reveal customer value and behaviour.
    Adding Age would make the clusters harder to visualise in 2-D
    and adds less discriminatory power for business segmentation.
  """)

    # Extract the feature matrix (NumPy array)
    X = get_feature_matrix(clean_df, CONFIG["cluster_features"])

    # Scale the features
    X_scaled, scaler = scale_features(X)

    # ── STEP 5: Elbow Method ─────────────────────────────────────────────
    auto_k = elbow_method(X_scaled, max_k=CONFIG["max_k"])

    # Choose K: auto-detected or user override
    if CONFIG["use_auto_k"]:
        chosen_k = auto_k
        print(f"\n  ✅ Using auto-detected K = {chosen_k}")
    else:
        chosen_k = CONFIG["override_k"]
        print(f"\n  ⚙️  Using user-override K = {chosen_k}")

    # ── STEP 5: Train K-Means ────────────────────────────────────────────
    kmeans, sil_score = train_kmeans(X_scaled, n_clusters=chosen_k)

    # ── STEP 6: Assign Cluster Labels ────────────────────────────────────
    clustered_df = assign_clusters(clean_df, kmeans, X_scaled)

    # ── STEP 7-8: Visualize & Analyse Clusters ───────────────────────────
    run_visualization(clustered_df, CONFIG["cluster_features"], kmeans)

    # ── STEP 9: Generate Report ──────────────────────────────────────────
    # Import cluster_names from visualizer (set during run_visualization)
    from src.visualizer import CLUSTER_NAMES
    generate_report(
        df            = clustered_df,
        n_clusters    = chosen_k,
        features      = CONFIG["cluster_features"],
        cluster_names = CLUSTER_NAMES,
        silhouette_val= sil_score
    )

    # ── SAVE CLUSTERED CSV ────────────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    clustered_df.to_csv(CONFIG["output_csv"], index=False)
    print(f"\n✅ Clustered dataset saved → {CONFIG['output_csv']}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "█" * 60)
    print("█              PIPELINE COMPLETE!                         █")
    print("█" * 60)
    print(f"\n  ⏱️  Total runtime     : {elapsed:.1f} seconds")
    print(f"  📦  Customers analysed: {len(clustered_df)}")
    print(f"  🔵  Clusters formed   : {chosen_k}")
    print(f"  📈  Silhouette Score  : {sil_score:.4f}")
    print(f"  📁  Charts saved to   : outputs/")
    print(f"  📄  Report saved to   : reports/Project_Report.txt")
    print(f"  💾  CSV saved to      : {CONFIG['output_csv']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point guard
# ─────────────────────────────────────────────────────────────────────────────
# This block only runs when you execute THIS file directly (python main.py).
# It does NOT run when this file is imported by another module.
if __name__ == "__main__":
    main()
