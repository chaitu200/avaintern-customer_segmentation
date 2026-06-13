"""
=============================================================
FILE: test_pipeline.py
PURPOSE: Quick smoke test to verify all modules import correctly
         and the data file exists before running main.py
=============================================================
Run this first if you're unsure everything is set up:
    python test_pipeline.py
"""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Tests the pipeline components and dependencies

print("=" * 50)
print("  SMOKE TEST – Checking project setup")
print("=" * 50)

# 1. Check Python version
import platform
print(f"\n✅ Python version : {platform.python_version()}")

# 2. Check all required libraries
required_libs = {
    "pandas"      : "pd",
    "numpy"       : "np",
    "matplotlib"  : "matplotlib",
    "sklearn"     : "sklearn",
}

for lib, alias in required_libs.items():
    try:
        __import__(lib)
        print(f"✅ {lib:15} : installed")
    except ImportError:
        pkg = "scikit-learn" if lib == "sklearn" else lib
        print(f"❌ {lib:15} : MISSING! Run: pip install {pkg}")

# 3. Check data file
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "Mall_Customers.csv")
if os.path.exists(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"\n✅ Data file found : {data_path}")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
else:
    print(f"\n❌ Data file NOT found: {data_path}")
    print("   Please make sure Mall_Customers.csv is in the data/ folder.")

# 4. Check all src modules
modules = ["data_loader", "eda", "clustering", "visualizer", "report_generator"]
print()
for mod in modules:
    try:
        __import__(f"src.{mod}")
        print(f"✅ src/{mod}.py : OK")
    except Exception as e:
        print(f"❌ src/{mod}.py : ERROR – {e}")

print("\n" + "=" * 50)
print("  If all checks show ✅ → run: python main.py")
print("=" * 50)
