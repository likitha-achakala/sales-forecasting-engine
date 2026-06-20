"""
generate_sample_data.py
------------------------
Generates a realistic, Superstore-style retail transactions dataset for
offline development and testing of the forecasting pipeline.

This is a STAND-IN for the real Kaggle "Superstore Sales Dataset"
(https://www.kaggle.com/datasets/vivek468/superstore-dataset-final).

To use the REAL dataset instead (recommended for your actual submission):
  1. Download "Sample - Superstore.csv" from the Kaggle link above
  2. Place it at data/raw/superstore.csv
  3. Skip this script entirely - 01_data_cleaning.ipynb will pick it up

The synthetic data below mimics real retail patterns so the rest of the
pipeline (cleaning, feature engineering, modeling) runs unchanged on
either source:
  - Multi-year daily transactions across Regions and Categories
  - Upward revenue trend over time
  - Weekly seasonality (weekend dip in B2B-style retail)
  - Annual seasonality (Nov/Dec holiday spike)
  - Random noise + occasional promo spikes
  - A small percentage of missing values (realistic data quality issue)
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "superstore.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

START_DATE = "2021-01-01"
END_DATE = "2024-12-31"
REGIONS = ["East", "West", "Central", "South"]
CATEGORIES = {
    "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
    "Office Supplies": ["Binders", "Paper", "Storage", "Art"],
    "Technology": ["Phones", "Accessories", "Machines", "Copiers"],
}

dates = pd.date_range(START_DATE, END_DATE, freq="D")
rows = []
order_id_counter = 1

for date in dates:
    day_index = (date - pd.Timestamp(START_DATE)).days

    # Base demand with long-term growth trend
    trend = 1 + (day_index / len(dates)) * 0.6

    # Weekly seasonality: lower on weekends (typical B2B retail pattern)
    weekday_factor = 0.7 if date.dayofweek >= 5 else 1.0

    # Annual seasonality: holiday season spike (Nov-Dec), summer lull (Jun-Jul)
    month = date.month
    if month in (11, 12):
        season_factor = 1.8
    elif month in (6, 7):
        season_factor = 0.85
    else:
        season_factor = 1.0

    # Number of orders placed on this day
    n_orders = np.random.poisson(lam=8 * trend * weekday_factor * season_factor)

    for _ in range(max(n_orders, 0)):
        region = np.random.choice(REGIONS, p=[0.3, 0.28, 0.22, 0.2])
        category = np.random.choice(list(CATEGORIES.keys()), p=[0.25, 0.45, 0.30])
        sub_category = np.random.choice(CATEGORIES[category])

        base_price = {
            "Furniture": 250,
            "Office Supplies": 35,
            "Technology": 320,
        }[category]

        quantity = np.random.randint(1, 8)
        unit_noise = np.random.normal(1.0, 0.25)
        sales = max(base_price * quantity * unit_noise * season_factor, 5)

        # Occasional promo / bulk-order spike
        if np.random.rand() < 0.02:
            sales *= np.random.uniform(2, 4)

        discount = round(np.random.choice([0, 0, 0, 0.1, 0.15, 0.2, 0.3]), 2)
        profit = sales * np.random.uniform(0.05, 0.35) * (1 - discount)

        rows.append(
            {
                "Order ID": f"ORD-{order_id_counter:07d}",
                "Order Date": date,
                "Region": region,
                "Category": category,
                "Sub-Category": sub_category,
                "Quantity": quantity,
                "Discount": discount,
                "Sales": round(sales, 2),
                "Profit": round(profit, 2),
            }
        )
        order_id_counter += 1

df = pd.DataFrame(rows)

# Inject a small, realistic amount of missing data
missing_mask = np.random.rand(len(df)) < 0.01
df.loc[missing_mask, "Discount"] = np.nan

# Inject a few duplicate rows (realistic data-quality issue)
dupes = df.sample(frac=0.002, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Generated {len(df):,} transaction rows -> {OUTPUT_PATH}")
print(f"Date range: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
print(f"Missing Discount values: {df['Discount'].isna().sum()}")
