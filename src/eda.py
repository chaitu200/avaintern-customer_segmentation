"""
=============================================================
MODULE: eda.py
PURPOSE: Exploratory Data Analysis (EDA) on the customer data
=============================================================
EDA means "get to know your data before modeling."
We look at:
  - Distributions of each feature (histograms)
  - Relationships between features (scatter plots, correlation heatmap)
  - Gender breakdown (pie chart / bar chart)
  - Outlier detection (box plots)

All charts are saved as PNG files in the outputs/ folder.
=============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt   # matplotlib: the standard Python plotting library
import os

# ── colour palette used throughout the project ──────────────────────────────
PALETTE = {
    "primary"   : "#6C63FF",   # Purple
    "secondary" : "#FF6584",   # Pink
    "accent"    : "#43B89C",   # Teal
    "warning"   : "#F9A825",   # Amber
    "dark"      : "#1E1E2E",   # Near-black background
    "light"     : "#F5F5F5",   # Off-white
    "male"      : "#5B9BD5",   # Blue for males
    "female"    : "#ED7D31",   # Orange for females
}

FIGURES_DIR = "outputs"


def _ensure_dir():
    """Create the figures directory if it does not exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _apply_dark_style(fig, ax_list):
    """Apply a consistent dark-mode style to a figure and its axes."""
    fig.patch.set_facecolor(PALETTE["dark"])
    for ax in ax_list:
        ax.set_facecolor("#2A2A3E")
        ax.tick_params(colors=PALETTE["light"])
        ax.xaxis.label.set_color(PALETTE["light"])
        ax.yaxis.label.set_color(PALETTE["light"])
        ax.title.set_color(PALETTE["light"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 1 : Gender Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_gender_distribution(df: pd.DataFrame):
    """
    Show how many male vs female customers are in the dataset.
    """
    print("\n📊 Plotting gender distribution …")
    _ensure_dir()

    # Count occurrences of each gender
    gender_counts = df["Gender"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Gender Distribution", fontsize=16, color=PALETTE["light"], fontweight="bold")
    _apply_dark_style(fig, axes)

    # --- Pie chart ---
    colors_pie = [PALETTE["male"], PALETTE["female"]]
    axes[0].pie(
        gender_counts.values,
        labels=gender_counts.index,
        autopct="%1.1f%%",       # Show percentage on each slice
        colors=colors_pie,
        startangle=90,
        wedgeprops={"edgecolor": PALETTE["dark"], "linewidth": 2},
        textprops={"color": PALETTE["light"]}
    )
    axes[0].set_title("Gender Split (Pie)", fontsize=13)

    # --- Bar chart ---
    bars = axes[1].bar(
        gender_counts.index,
        gender_counts.values,
        color=[PALETTE["male"], PALETTE["female"]],
        edgecolor=PALETTE["dark"],
        linewidth=1.5
    )
    axes[1].set_title("Gender Count (Bar)", fontsize=13)
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Number of Customers")

    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            str(int(height)),
            ha="center", va="bottom",
            color=PALETTE["light"], fontsize=12, fontweight="bold"
        )

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "01_gender_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 2 : Feature Distributions (Histograms)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame):
    """
    Draw histograms for Age, Annual Income, and Spending Score.
    """
    print("\n📊 Plotting feature distributions (histograms) …")
    _ensure_dir()

    numeric_cols = ["Age", "Annual_Income", "Spending_Score"]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    x_labels = ["Age (years)", "Annual Income (k$)", "Spending Score (1-100)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Distribution of Customer Features", fontsize=16,
                 color=PALETTE["light"], fontweight="bold")
    _apply_dark_style(fig, axes)

    for ax, col, color, xlabel in zip(axes, numeric_cols, colors, x_labels):
        ax.hist(df[col], bins=20, color=color, edgecolor=PALETTE["dark"],
                alpha=0.85, linewidth=0.8)

        # Draw a vertical line at the mean value
        mean_val = df[col].mean()
        ax.axvline(mean_val, color=PALETTE["warning"], linestyle="--",
                   linewidth=2, label=f"Mean = {mean_val:.1f}")

        ax.set_title(f"Distribution of {col}", fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of Customers")
        ax.legend(fontsize=9, facecolor="#2A2A3E", labelcolor=PALETTE["light"])

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "02_feature_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 3 : Box Plots (Outlier Detection)
# ─────────────────────────────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame):
    """
    Draw box plots for Age, Annual Income, and Spending Score.
    """
    print("\n📊 Plotting box plots (outlier detection) …")
    _ensure_dir()

    numeric_cols = ["Age", "Annual_Income", "Spending_Score"]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Box Plots – Spotting Outliers in Customer Data",
                 fontsize=16, color=PALETTE["light"], fontweight="bold")
    _apply_dark_style(fig, axes)

    for ax, col, color in zip(axes, numeric_cols, colors):
        ax.boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor=color, color=PALETTE["light"]),
            medianprops=dict(color=PALETTE["warning"], linewidth=2),
            whiskerprops=dict(color=PALETTE["light"]),
            capprops=dict(color=PALETTE["light"]),
            flierprops=dict(marker="o", color=PALETTE["secondary"],
                            markerfacecolor=PALETTE["secondary"], markersize=5)
        )
        ax.set_title(f"Box Plot: {col}", fontsize=12)
        ax.set_ylabel(col)
        ax.set_xticks([])

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "03_boxplots.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 4 : Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Draw a heatmap of correlations between numeric features.
    """
    print("\n📊 Plotting correlation heatmap …")
    _ensure_dir()

    numeric_df = df[["Age", "Annual_Income", "Spending_Score", "Gender_Encoded"]]
    corr_matrix = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PALETTE["dark"])
    ax.set_facecolor(PALETTE["dark"])
    ax.title.set_color(PALETTE["light"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    # Create the heatmap manually
    n = len(corr_matrix)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Colour bar (legend) on the right
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=PALETTE["light"])

    # Axis tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=30, ha="right",
                       color=PALETTE["light"], fontsize=10)
    ax.set_yticklabels(corr_matrix.index, color=PALETTE["light"], fontsize=10)

    # Annotate each cell with the correlation value
    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            text_color = "black" if 0.3 < abs(val) < 0.8 else PALETTE["light"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    ax.set_title("Feature Correlation Heatmap", fontsize=14,
                 color=PALETTE["light"], fontweight="bold", pad=15)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "04_correlation_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 5 : Scatter Plot Matrix (Pair Plot)
# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter_matrix(df: pd.DataFrame):
    """
    Draw scatter plots between every pair of numeric features, coloured by gender.
    """
    print("\n📊 Plotting scatter matrix …")
    _ensure_dir()

    cols = ["Age", "Annual_Income", "Spending_Score"]
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(14, 12))
    fig.suptitle("Scatter Matrix – Pairwise Feature Relationships",
                 fontsize=15, color=PALETTE["light"], fontweight="bold")
    _apply_dark_style(fig, [ax for row in axes for ax in row])

    # Separate male and female rows for coloured scatter
    male_mask   = df["Gender"] == "Male"
    female_mask = df["Gender"] == "Female"
    # also handle M/F notation
    if not male_mask.any():
        male_mask   = df["Gender"] == "M"
        female_mask = df["Gender"] == "F"

    for i, col_y in enumerate(cols):
        for j, col_x in enumerate(cols):
            ax = axes[i][j]
            if i == j:
                # Diagonal: histogram of the feature
                ax.hist(df[col_y], bins=15, color=PALETTE["primary"],
                        edgecolor=PALETTE["dark"], alpha=0.8)
            else:
                # Off-diagonal: scatter coloured by gender
                ax.scatter(df.loc[male_mask,   col_x],
                           df.loc[male_mask,   col_y],
                           color=PALETTE["male"],   alpha=0.6, s=20, label="Male")
                ax.scatter(df.loc[female_mask, col_x],
                           df.loc[female_mask, col_y],
                           color=PALETTE["female"], alpha=0.6, s=20, label="Female")

            if i == n - 1:
                ax.set_xlabel(col_x, color=PALETTE["light"], fontsize=9)
            if j == 0:
                ax.set_ylabel(col_y, color=PALETTE["light"], fontsize=9)

    # Add a legend in the bottom-right panel
    axes[n-1][n-1].legend(fontsize=8, facecolor="#2A2A3E", labelcolor=PALETTE["light"])

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "05_scatter_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 6 : Age vs Spending Score by Gender
# ─────────────────────────────────────────────────────────────────────────────
def plot_age_vs_spending(df: pd.DataFrame):
    """
    Dedicated scatter: Age (X) vs Spending Score (Y), split by gender.
    This directly answers: "Do younger customers spend more?"
    """
    print("\n📊 Plotting Age vs Spending Score …")
    _ensure_dir()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(PALETTE["dark"])
    ax.set_facecolor("#2A2A3E")

    male_mask   = df["Gender"].isin(["Male",   "M"])
    female_mask = df["Gender"].isin(["Female", "F"])

    ax.scatter(df.loc[male_mask,   "Age"], df.loc[male_mask,   "Spending_Score"],
               color=PALETTE["male"],   alpha=0.75, s=60, label="Male",   edgecolors="none")
    ax.scatter(df.loc[female_mask, "Age"], df.loc[female_mask, "Spending_Score"],
               color=PALETTE["female"], alpha=0.75, s=60, label="Female", edgecolors="none")

    ax.set_xlabel("Age (years)", color=PALETTE["light"], fontsize=12)
    ax.set_ylabel("Spending Score (1-100)", color=PALETTE["light"], fontsize=12)
    ax.set_title("Age vs Spending Score by Gender",
                 color=PALETTE["light"], fontsize=14, fontweight="bold")
    ax.tick_params(colors=PALETTE["light"])
    ax.legend(facecolor="#2A2A3E", labelcolor=PALETTE["light"], fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "06_age_vs_spending.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 7 : Income vs Spending Score
# ─────────────────────────────────────────────────────────────────────────────
def plot_income_vs_spending(df: pd.DataFrame):
    """
    Scatter: Annual Income (X) vs Spending Score (Y).
    """
    print("\n📊 Plotting Annual Income vs Spending Score …")
    _ensure_dir()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(PALETTE["dark"])
    ax.set_facecolor("#2A2A3E")

    male_mask   = df["Gender"].isin(["Male",   "M"])
    female_mask = df["Gender"].isin(["Female", "F"])

    ax.scatter(df.loc[male_mask,   "Annual_Income"], df.loc[male_mask,   "Spending_Score"],
               color=PALETTE["male"],   alpha=0.75, s=70, label="Male",   edgecolors="none")
    ax.scatter(df.loc[female_mask, "Annual_Income"], df.loc[female_mask, "Spending_Score"],
               color=PALETTE["female"], alpha=0.75, s=70, label="Female", edgecolors="none")

    ax.set_xlabel("Annual Income (k$)", color=PALETTE["light"], fontsize=12)
    ax.set_ylabel("Spending Score (1-100)", color=PALETTE["light"], fontsize=12)
    ax.set_title("Annual Income vs Spending Score\n(Pre-Clustering View)",
                 color=PALETTE["light"], fontsize=14, fontweight="bold")
    ax.tick_params(colors=PALETTE["light"])
    ax.legend(facecolor="#2A2A3E", labelcolor=PALETTE["light"], fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "07_income_vs_spending.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER EDA FUNCTION – call this from main.py
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    """
    Run all EDA steps in order.

    Parameters:
        df (pd.DataFrame): Cleaned customer DataFrame
    """
    print("\n" + "=" * 60)
    print("  STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # Print a statistical summary to the console
    print("\n--- Statistical Summary ---")
    print(df[["Age", "Annual_Income", "Spending_Score"]].describe().round(2))

    # Gender breakdown
    print("\n--- Gender Breakdown ---")
    print(df["Gender"].value_counts())
    print(f"   Male   : {df['Gender'].isin(['Male','M']).sum()} customers")
    print(f"   Female : {df['Gender'].isin(['Female','F']).sum()} customers")

    # Run all plot functions
    plot_gender_distribution(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_scatter_matrix(df)
    plot_age_vs_spending(df)
    plot_income_vs_spending(df)

    print("\n✅ EDA complete!  All charts saved to outputs/")
