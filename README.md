# Customer Segmentation Using K-Means Clustering

> **Internship Project** | Python | Machine Learning | Unsupervised Learning

---

## Project Overview

This project implements a complete **Customer Segmentation** system using the **K-Means Clustering** algorithm on the Mall Customers dataset.  By grouping customers based on their Annual Income and Spending Score, we provide actionable business intelligence that helps marketing teams create targeted campaigns.

### What the project does:
| Stage | Description |
|---|---|
| 📥 Data Loading | Load the Mall_Customers.csv into a pandas DataFrame |
| 🧹 Data Cleaning | Handle missing values, encode categorical columns, remove duplicates |
| 🔍 EDA | 7 exploratory charts: histograms, box plots, scatter plots, heatmap |
| ⚡ Elbow Method | Find the optimal number of clusters (K) automatically |
| 🤖 K-Means | Train the clustering model and assign labels |
| 📊 Visualisation | 5 cluster plots: 2-D, 3-D, bar charts, radar, gender breakdown |
| 📄 Report | Auto-generated academic-style project report |

---

## Project Structure

```
customer_segmentation/
│
├── data/
│   └── Mall_Customers.csv          # Raw dataset
│
├── src/
│   ├── data_loader.py              # Load & clean data
│   ├── eda.py                      # Exploratory Data Analysis
│   ├── clustering.py               # Elbow Method + K-Means
│   ├── visualizer.py               # All cluster visualizations
│   └── report_generator.py         # Generate project report
│
├── outputs/                       # All charts saved here (PNG)
│   ├── 01_gender_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_boxplots.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_scatter_matrix.png
│   ├── 06_age_vs_spending.png
│   ├── 07_income_vs_spending.png
│   ├── 08_elbow_silhouette.png
│   ├── 09_2d_clusters.png
│   ├── 10_3d_clusters.png
│   ├── 11_cluster_statistics.png
│   ├── 12_cluster_gender.png
│   └── 13_radar_chart.png
│
├── reports/
│   ├── Project_Report.txt          # Full academic report
│   └── clustered_customers.csv     # Dataset with cluster labels
│
├── main.py                         # ← RUN THIS FILE
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Requirements

- **Python** 3.8 or higher
- **pandas** ≥ 1.5.0
- **numpy** ≥ 1.23.0
- **matplotlib** ≥ 3.6.0
- **scikit-learn** ≥ 1.2.0

---

## Installation

### Step 1: Clone / Download the project
Place the `customer_segmentation/` folder anywhere on your computer.

### Step 2: Open a terminal
In **VS Code**: `View → Terminal` or press `` Ctrl+` ``

### Step 3: Navigate to the project folder
```bash
cd path\to\customer_segmentation
```

### Step 4: (Optional) Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### Step 5: Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py
```

That's it!  The entire pipeline runs automatically.  You will see:
- Step-by-step progress printed in the terminal
- 13 charts displayed sequentially (close each to continue)
- All charts saved to `outputs/`
- Final report saved to `reports/Project_Report.txt`
- Clustered dataset saved to `reports/clustered_customers.csv`

---

## Output

### Console Output (sample)
```
████████████████████████████████████████████████████████████
█   CUSTOMER SEGMENTATION – K-MEANS CLUSTERING PIPELINE   █
████████████████████████████████████████████████████████████

  STEP 1: LOADING THE DATASET
  ✅ Dataset loaded successfully  |  200 rows × 5 columns

  STEP 2: DATA CLEANING
  ✅ Missing values filled
  ✅ Gender encoded → Male=1, Female=0

  STEP 3: EXPLORATORY DATA ANALYSIS
  📊 Plotting gender distribution …
  📊 Plotting feature distributions …
  ...

  STEP 4: ELBOW METHOD – FINDING OPTIMAL K
    K |         WCSS |   Silhouette
    1 |    79600.43  |          N/A
    2 |    52100.11  |       0.3542
    3 |    41000.22  |       0.4128
    4 |    30500.55  |       0.4500
    5 |    24200.00  |       0.5543   ← Best
    ...
  📌 Suggested optimal K = 5

  STEP 5: TRAINING K-MEANS WITH K = 5
  ✅ Silhouette Score: 0.5543

┌─ Cluster 0: High Value Targets 💎
│   Size      : 39 customers  (19.5%)
│   Avg Age   : 32.7 years
│   Avg Income: $86.5k
│   Avg Spend : 82.1 / 100
└───────────────────────────────────────────────────────────

  ⏱️  Total runtime: ~30 seconds
  📁  Charts saved to: outputs/
  📄  Report saved to: reports/Project_Report.txt
```

### Generated Charts (13 total)
| # | File | Description |
|---|---|---|
| 1 | `01_gender_distribution.png` | Pie + bar chart of gender split |
| 2 | `02_feature_distributions.png` | Histograms of Age, Income, Spending |
| 3 | `03_boxplots.png` | Box plots for outlier detection |
| 4 | `04_correlation_heatmap.png` | Feature correlation matrix |
| 5 | `05_scatter_matrix.png` | All pairwise scatter plots |
| 6 | `06_age_vs_spending.png` | Age vs Spending (by gender) |
| 7 | `07_income_vs_spending.png` | Income vs Spending (by gender) |
| 8 | `08_elbow_silhouette.png` | Elbow curve + Silhouette scores |
| 9 | `09_2d_clusters.png` | **K-Means cluster scatter (main result)** |
| 10 | `10_3d_clusters.png` | 3-D cluster view |
| 11 | `11_cluster_statistics.png` | Bar chart comparison of clusters |
| 12 | `12_cluster_gender.png` | Gender breakdown per cluster |
| 13 | `13_radar_chart.png` | Radar/spider chart of cluster profiles |

---

## Algorithm Explained (Simply)

**K-Means** groups customers by finding K "centres" (centroids) such that every customer is closest to its own centre.

1. Pick K random centres
2. Assign each customer to the nearest centre
3. Move each centre to the average of its customers
4. Repeat until centres stop moving

We choose K using the **Elbow Method**: plot WCSS vs K, and pick the K where improvement slows down (the "elbow").

---

## Key Results: 5 Customer Segments

| Cluster | Name | Strategy |
|---|---|---|
| 0 | 💎 High Value Targets | VIP loyalty programs |
| 1 | 🏦 Conservative Earners | Personalised premium discounts |
| 2 | 🛍️ Impulse Buyers | Flash sales & FOMO campaigns |
| 3 | 💰 Budget Conscious | Coupons & bundle deals |
| 4 | 🛒 Average Shoppers | Seasonal campaigns & referrals |

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core programming language |
| pandas | ≥ 1.5.0 | Data loading, cleaning, manipulation |
| numpy | ≥ 1.23.0 | Numerical operations |
| matplotlib | ≥ 3.6.0 | All data visualizations |
| scikit-learn | ≥ 1.2.0 | StandardScaler, KMeans, silhouette_score |

---

## Author

- **Name:** Chaitanya reddy manda
- **Institution:** Aditya university
- **Internship:** Avaintern Edutech
- **Date:** 2026
