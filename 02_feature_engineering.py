"""
02_feature_engineering.py
--------------------------
Builds time-based features (calendar, lag, rolling-window) on top of the
cleaned daily sales series, ready for model training.

Input : data/processed/daily_sales.csv
Output: data/processed/features.csv
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "daily_sales.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")

df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon ... 6=Sun
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

# Holiday-season flag: Nov-Dec retail spike (tune to your real dataset's market)
df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

# Cyclical encoding for day-of-week and month so the model understands
# that e.g. Sunday (6) is "close to" Monday (0), and December is close to January
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Linear time trend index (days since series start) - lets linear models
# capture long-term growth directly
df["time_index"] = (df["date"] - df["date"].min()).dt.days

# ---------------------------------------------------------------------------
# Lag features (sales N days ago) - the single most important feature class
# for time-series regression, since recent sales predict near-future sales
# ---------------------------------------------------------------------------
for lag in [1, 7, 14, 30]:
    df[f"sales_lag_{lag}"] = df["total_sales"].shift(lag)

# ---------------------------------------------------------------------------
# Rolling window features (trend smoothing)
# NOTE: shift(1) before rolling so the window only sees PAST data relative to
# the row's own date - prevents leaking same-day sales into its own features
# ---------------------------------------------------------------------------
for window in [7, 30]:
    df[f"rolling_mean_{window}"] = df["total_sales"].shift(1).rolling(window).mean()
    df[f"rolling_std_{window}"] = df["total_sales"].shift(1).rolling(window).std()

# Day-over-day and week-over-week percent change (momentum indicators)
# NOTE: pct_change() produces +/-inf when the prior period's sales were 0
# (can happen on sparse/holiday days) - replace those with a large but
# finite value so models don't choke on infinities, while still signaling
# "sales surged from near-zero"
df["pct_change_1d"] = df["total_sales"].pct_change(1).replace([np.inf, -np.inf], np.nan)
df["pct_change_7d"] = df["total_sales"].pct_change(7).replace([np.inf, -np.inf], np.nan)
df["pct_change_1d"] = df["pct_change_1d"].clip(-5, 5)
df["pct_change_7d"] = df["pct_change_7d"].clip(-5, 5)

# ---------------------------------------------------------------------------
# Drop rows with NaNs introduced by lag/rolling windows
# (first 30 days of the series can't have a 30-day lookback - this is
# expected and correct, not a data quality bug)
# ---------------------------------------------------------------------------
before = len(df)
df_clean = df.dropna().reset_index(drop=True)
print(f"Dropped {before - len(df_clean)} early rows lacking full lag/rolling history")
print(f"Final feature set: {len(df_clean)} rows, {df_clean.shape[1]} columns")
print(f"\nFeature columns:\n{[c for c in df_clean.columns if c not in ['date', 'total_sales']]}")

df_clean.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved engineered features -> {OUTPUT_PATH}")
