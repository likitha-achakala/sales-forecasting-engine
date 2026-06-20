"""
01_data_cleaning.py
--------------------
Loads raw transaction-level retail data, cleans it, and aggregates it
into a daily sales time series ready for feature engineering.

Input : data/raw/superstore.csv
Output: data/processed/daily_sales.csv
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "superstore.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "daily_sales.csv")
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df):,} raw transaction rows")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")

# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
# Parse dates
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Drop rows where the date itself failed to parse (can't use them in a time series)
before = len(df)
df = df.dropna(subset=["Order Date"])
print(f"\nDropped {before - len(df)} rows with unparseable dates")

# Remove exact duplicate transactions
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# Handle missing Discount: impute with 0 (no discount applied) rather than
# dropping rows, since Discount missingness doesn't invalidate the Sales figure
df["Discount"] = df["Discount"].fillna(0)

# Drop any rows still missing the core Sales figure (can't be repaired)
before = len(df)
df = df.dropna(subset=["Sales"])
print(f"Dropped {before - len(df)} rows with missing Sales")

# Remove non-physical values
df = df[df["Sales"] > 0]
df = df[df["Quantity"] > 0]

print(f"\nClean transaction-level dataset: {len(df):,} rows")

# ---------------------------------------------------------------------------
# 3. Aggregate to daily sales (this is what we actually forecast)
# ---------------------------------------------------------------------------
daily = (
    df.groupby(df["Order Date"].dt.date)
    .agg(
        total_sales=("Sales", "sum"),
        order_count=("Order ID", "nunique"),
        total_quantity=("Quantity", "sum"),
        avg_discount=("Discount", "mean"),
    )
    .reset_index()
    .rename(columns={"Order Date": "date"})
)
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

# Fill any calendar gaps (days with zero recorded sales) so the time series
# has no missing dates - critical for valid lag/rolling features later
full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
daily = daily.set_index("date").reindex(full_range).rename_axis("date").reset_index()
daily["total_sales"] = daily["total_sales"].fillna(0)
daily["order_count"] = daily["order_count"].fillna(0)
daily["total_quantity"] = daily["total_quantity"].fillna(0)
daily["avg_discount"] = daily["avg_discount"].ffill()

print(f"\nDaily series: {len(daily)} days, "
      f"{daily['date'].min().date()} to {daily['date'].max().date()}")
print(f"Days with zero sales (filled gaps): {(daily['total_sales'] == 0).sum()}")

daily.to_csv(PROCESSED_PATH, index=False)
print(f"\nSaved daily sales series -> {PROCESSED_PATH}")
print(daily.head())
