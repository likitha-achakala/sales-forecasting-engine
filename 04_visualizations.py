"""
04_visualizations.py
----------------------
Generates business-friendly charts from the model comparison and
predictions produced in step 3:

  1. model_comparison.png   - MAE/RMSE/R2 bar chart across all 3 models
  2. actual_vs_predicted.png - actual vs. best-model forecast on test period
  3. feature_importance.png - what drives sales, per the best tree model
  4. forecast_next_30_days.png - forward-looking forecast with trend context
  5. sales_trend_seasonality.png - historical trend + monthly seasonality

Input : data/processed/{model_comparison,predictions,feature_importance}.csv
Output: visuals/*.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "font.size": 11,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "grid.color": "#e0e0e0",
})

COLORS = {"Linear Regression": "#94a3b8", "Random Forest": "#60a5fa", "XGBoost": "#f97316"}

def color_for(name):
    for key, c in COLORS.items():
        if key in name:
            return c
    return "#10b981"

dollar_fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
comparison = pd.read_csv(os.path.join(PROCESSED_DIR, "model_comparison.csv"))
predictions = pd.read_csv(os.path.join(PROCESSED_DIR, "predictions.csv"), parse_dates=["date"])
importance = pd.read_csv(os.path.join(PROCESSED_DIR, "feature_importance.csv"))
daily_sales = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_sales.csv"), parse_dates=["date"])
with open(os.path.join(PROCESSED_DIR, "run_summary.json")) as f:
    summary = json.load(f)

best_model = summary["best_model"]
model_cols = [c for c in predictions.columns if c not in ("date", "actual")]

# ---------------------------------------------------------------------------
# 1. Model comparison bar chart (MAE, RMSE, R2)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
metrics = [("MAE", "Mean Absolute Error ($, lower=better)"),
           ("RMSE", "Root Mean Squared Error ($, lower=better)"),
           ("R2", "R\u00b2 Score (higher=better)")]

for ax, (col, title) in zip(axes, metrics):
    bars = ax.bar(comparison["Model"], comparison[col],
                   color=[color_for(m) for m in comparison["Model"]])
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        h = bar.get_height()
        label = f"{h:,.0f}" if col != "R2" else f"{h:.3f}"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Model Benchmark: Linear Regression vs. Random Forest vs. XGBoost",
             fontsize=15, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved model_comparison.png")

# ---------------------------------------------------------------------------
# 2. Actual vs predicted (best model) on the test period
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(predictions["date"], predictions["actual"], label="Actual Sales",
        color="#1f2937", linewidth=1.6)
ax.plot(predictions["date"], predictions[best_model], label=f"{best_model} Forecast",
        color="#f97316", linewidth=1.6, linestyle="--")
ax.fill_between(predictions["date"], predictions["actual"], predictions[best_model],
                 color="#f97316", alpha=0.08)
ax.set_title(f"Actual vs. Predicted Daily Sales \u2014 Test Period ({best_model})")
ax.set_ylabel("Daily Sales")
ax.yaxis.set_major_formatter(dollar_fmt)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved actual_vs_predicted.png")

# ---------------------------------------------------------------------------
# 3. Feature importance (top 12)
# ---------------------------------------------------------------------------
top_features = importance.head(12).sort_values("importance")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_features["feature"], top_features["importance"], color="#f97316")
ax.set_title(f"What Drives the Forecast \u2014 Top Features ({best_model})")
ax.set_xlabel("Relative Importance")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved feature_importance.png")

# ---------------------------------------------------------------------------
# 4. Forward-looking forecast: last 90 days of history + next 30 days
# Built using the best tree model with a simple recursive-forecast loop:
# each new day's lag/rolling features are derived from the model's own
# prior predictions, since true future sales aren't known yet.
# ---------------------------------------------------------------------------
import joblib
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
model_filename = best_model.lower().replace(" ", "_").replace("(", "").replace(")", "")
best_model_obj = joblib.load(os.path.join(MODELS_DIR, f"{model_filename}.joblib"))

features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"), parse_dates=["date"])
FEATURE_COLS = [c for c in features_df.columns if c not in ("date", "total_sales")]

history = features_df.copy().sort_values("date").reset_index(drop=True)
horizon = 30
future_rows = []
working_sales = history["total_sales"].tolist()
working_dates = history["date"].tolist()

for step in range(horizon):
    next_date = working_dates[-1] + pd.Timedelta(days=1)
    sales_series = pd.Series(working_sales)

    row = {
        "year": next_date.year, "month": next_date.month, "day": next_date.day,
        "day_of_week": next_date.dayofweek,
        "week_of_year": int(next_date.isocalendar()[1]),
        "quarter": (next_date.month - 1) // 3 + 1,
        "is_weekend": int(next_date.dayofweek >= 5),
        "is_month_start": int(next_date.day == 1),
        "is_month_end": int((next_date + pd.Timedelta(days=1)).month != next_date.month),
        "is_holiday_season": int(next_date.month in (11, 12)),
        "dow_sin": np.sin(2 * np.pi * next_date.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * next_date.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * next_date.month / 12),
        "month_cos": np.cos(2 * np.pi * next_date.month / 12),
        "time_index": (next_date - history["date"].min()).days,
        "order_count": history["order_count"].tail(7).mean(),
        "total_quantity": history["total_quantity"].tail(7).mean(),
        "avg_discount": history["avg_discount"].tail(7).mean(),
        "sales_lag_1": sales_series.iloc[-1],
        "sales_lag_7": sales_series.iloc[-7],
        "sales_lag_14": sales_series.iloc[-14],
        "sales_lag_30": sales_series.iloc[-30],
        "rolling_mean_7": sales_series.tail(7).mean(),
        "rolling_std_7": sales_series.tail(7).std(),
        "rolling_mean_30": sales_series.tail(30).mean(),
        "rolling_std_30": sales_series.tail(30).std(),
        "pct_change_1d": np.clip((sales_series.iloc[-1] - sales_series.iloc[-2]) / max(sales_series.iloc[-2], 1), -5, 5),
        "pct_change_7d": np.clip((sales_series.iloc[-1] - sales_series.iloc[-8]) / max(sales_series.iloc[-8], 1), -5, 5),
    }

    X_next = pd.DataFrame([row])[FEATURE_COLS]
    pred = max(best_model_obj.predict(X_next)[0], 0)

    future_rows.append({"date": next_date, "forecast": pred})
    working_sales.append(pred)
    working_dates.append(next_date)

future_df = pd.DataFrame(future_rows)
future_df.to_csv(os.path.join(PROCESSED_DIR, "forecast_next_30_days.csv"), index=False)

recent_history = history[["date", "total_sales"]].tail(90)

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(recent_history["date"], recent_history["total_sales"], label="Historical Sales",
        color="#1f2937", linewidth=1.6)
ax.plot(future_df["date"], future_df["forecast"], label="30-Day Forecast",
        color="#f97316", linewidth=2, linestyle="--", marker="o", markersize=3)
ax.axvline(recent_history["date"].iloc[-1], color="#9ca3af", linestyle=":", linewidth=1)
ax.set_title(f"Sales Forecast \u2014 Next 30 Days ({best_model})")
ax.set_ylabel("Daily Sales")
ax.yaxis.set_major_formatter(dollar_fmt)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "forecast_next_30_days.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved forecast_next_30_days.png")

# ---------------------------------------------------------------------------
# 5. Long-term trend + monthly seasonality (business-context chart)
# ---------------------------------------------------------------------------
daily_sales["month_label"] = daily_sales["date"].dt.to_period("M")
monthly = daily_sales.groupby("month_label")["total_sales"].sum().reset_index()
monthly["month_label"] = monthly["month_label"].astype(str)

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(monthly["month_label"], monthly["total_sales"], color="#60a5fa")
ax.set_title("Monthly Sales Trend \u2014 Full History")
ax.set_ylabel("Total Monthly Sales")
ax.yaxis.set_major_formatter(dollar_fmt)
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "sales_trend_seasonality.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved sales_trend_seasonality.png")

print(f"\nAll visuals saved to {VISUALS_DIR}/")
