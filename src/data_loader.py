"""
=============================================================
MODULE: data_loader.py
PURPOSE: Load and clean the Mall Customers dataset
=============================================================
This module handles:
  - Reading the CSV file into a pandas DataFrame
  - Checking and fixing missing values
  - Checking and removing duplicate rows
  - Printing a data quality report
=============================================================
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Mall Customers CSV file into a pandas DataFrame.

    What this function does:
      1. Reads the CSV file from the given path.
      2. Prints the first few rows so we can see what the data looks like.
      3. Prints the shape (rows, columns).
      4. Prints data types of each column.

    Parameters:
        filepath (str): Full or relative path to Mall_Customers.csv

    Returns:
        pd.DataFrame: Raw (not yet cleaned) customer data
    """
    print("=" * 60)
    print("  STEP 1: LOADING THE DATASET")
    print("=" * 60)

    df = pd.read_csv(filepath)

    print(f"\n✅ Dataset loaded successfully from: {filepath}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("--- First 5 rows of the dataset ---")
    print(df.head())

    print("\n--- Column names & data types ---")
    print(df.dtypes)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame by:
      - Renaming columns to shorter, code-friendly names
      - Encoding Gender as a number (Male=1, Female=0)
      - Handling missing values (NaN)
      - Removing duplicate rows

    Parameters:
        df (pd.DataFrame): Raw dataset returned by load_data()

    Returns:
        pd.DataFrame: Cleaned dataset ready for analysis
    """
    print("\n" + "=" * 60)
    print("  STEP 2: DATA CLEANING")
    print("=" * 60)

    # ----- 2a. Rename columns for easier typing -----
    df = df.rename(columns={
        "CustomerID"              : "CustomerID",
        "Gender"                  : "Gender",
        "Age"                     : "Age",
        "Annual Income (k$)"      : "Annual_Income",
        "Spending Score (1-100)"  : "Spending_Score"
    })
    print("\n✅ Columns renamed:")
    print("   ", list(df.columns))

    # ----- 2b. Check for missing values -----
    missing = df.isnull().sum()
    print("\n--- Missing values per column ---")
    print(missing)

    # If any missing values exist, fill them with the column's median.
    for col in ["Age", "Annual_Income", "Spending_Score"]:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   ⚠️  Filled {col} NaN values with median = {median_val:.1f}")

    # For Gender, fill missing with the most common value (mode).
    if df["Gender"].isnull().any():
        mode_val = df["Gender"].mode()[0]
        df["Gender"] = df["Gender"].fillna(mode_val)
        print(f"   ⚠️  Filled Gender NaN values with mode = {mode_val}")

    # ----- 2c. Encode Gender as a number -----
    df["Gender_Encoded"] = df["Gender"].map({"Male": 1, "Female": 0})
    # If the dataset used 'M'/'F' notation, handle that too
    df["Gender_Encoded"] = df["Gender_Encoded"].fillna(
        df["Gender"].map({"M": 1, "F": 0})
    )
    print("\n✅ Gender encoded → Male=1, Female=0")

    # ----- 2d. Remove duplicate rows -----
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"\n✅ Duplicate rows removed: {before - after}")
    print(f"   Clean dataset size: {df.shape[0]} rows × {df.shape[1]} columns")

    # ----- 2e. Reset the index after dropping rows -----
    df.reset_index(drop=True, inplace=True)

    print("\n--- Final cleaned dataset (first 5 rows) ---")
    print(df.head())

    print("\n--- Dataset Statistics Summary ---")
    print(df.describe().round(2))

    return df


def get_feature_matrix(df: pd.DataFrame, features: list) -> np.ndarray:
    """
    Extract the specified columns from the DataFrame and return as a NumPy array.

    Why NumPy array?
      scikit-learn's KMeans expects a 2-D array (matrix) of numbers, not a DataFrame.

    Parameters:
        df       (pd.DataFrame): Cleaned dataset
        features (list)        : Column names to use, e.g. ['Annual_Income', 'Spending_Score']

    Returns:
        np.ndarray: 2-D array of shape (n_customers, n_features)
    """
    print(f"\n✅ Feature matrix extracted with columns: {features}")
    return df[features].values
