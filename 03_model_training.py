"""
03_model_training.py
---------------------
Trains and benchmarks three forecasting models on the engineered feature
set, using a strict time-based (non-shuffled) train/test split, and
evaluates each on MAE, RMSE, and R^2 - matching the resume claim:

  "benchmarked Linear Regression, Random Forest, and XGBoost on MAE,
   RMSE, and R^2 - XGBoost selected as best model."

XGBoost handling
-----------------
True XGBoost is used automatically if the `xgboost` package is installed
(it will be, on your own machine - just `pip install xgboost`).
If it is not installed (as in this sandboxed environment, which has no
internet access to install new packages), the script falls back to
scikit-learn's GradientBoostingRegressor, which is algorithmically the
same family of model (gradient-boosted decision trees) and a standard,
honest substitute for development/testing purposes. The fallback is
clearly labeled in all output and saved results so there is never any
ambiguity about which engine actually produced a given number.

Input : data/processed/features.csv
Output: data/processed/model_comparison.csv
        data/processed/predictions.csv
        models/ (saved trained models)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "model_comparison.csv")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "predictions.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Try real XGBoost first; fall back to GradientBoostingRegressor if unavailable
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor
    XGB_ENGINE = "xgboost.XGBRegressor (real XGBoost)"
    USING_REAL_XGBOOST = True
except ImportError:
    XGB_ENGINE = "sklearn.GradientBoostingRegressor (XGBoost fallback - install xgboost for the real thing)"
    USING_REAL_XGBOOST = False

print(f"Boosting engine in use: {XGB_ENGINE}\n")

# ---------------------------------------------------------------------------
# 1. Load features
# ---------------------------------------------------------------------------
df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

TARGET = "total_sales"
DROP_COLS = ["date", TARGET]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURE_COLS]
y = df[TARGET]

# ---------------------------------------------------------------------------
# 2. Time-based train/test split (CRITICAL: never shuffle time series data)
# Train on the earlier ~80% of days, test on the most recent ~20%, so the
# evaluation honestly reflects forecasting into the future, not interpolation
# ---------------------------------------------------------------------------
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df["date"].iloc[split_idx:]

print(f"Train period: {df['date'].iloc[0].date()} to {df['date'].iloc[split_idx-1].date()} "
      f"({len(X_train)} days)")
print(f"Test period:  {df['date'].iloc[split_idx].date()} to {df['date'].iloc[-1].date()} "
      f"({len(X_test)} days)\n")

# ---------------------------------------------------------------------------
# 3. Define models (resume-matching trio)
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    ),
}

if USING_REAL_XGBOOST:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
else:
    models["XGBoost (GB fallback)"] = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )

# ---------------------------------------------------------------------------
# 4. Train, predict, evaluate
# ---------------------------------------------------------------------------
results = []
predictions = {"date": dates_test.values, "actual": y_test.values}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # sales can't be negative

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / np.where(y_test == 0, 1, y_test))) * 100

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape})
    predictions[name] = preds

    model_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{model_filename}.joblib"))

    print(f"{name:30s}  MAE: {mae:9,.2f}   RMSE: {rmse:9,.2f}   R2: {r2:6.4f}   MAPE: {mape:6.2f}%")

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
results_df.to_csv(COMPARISON_PATH, index=False)

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(PREDICTIONS_PATH, index=False)

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model by RMSE: {best_model_name}")
print(f"\nSaved comparison table -> {COMPARISON_PATH}")
print(f"Saved test-set predictions -> {PREDICTIONS_PATH}")
print(f"Saved trained models -> {MODELS_DIR}/")

# Save feature importance from the best tree-based model for the visualization step
best_tree_model_name = [n for n in models if "Linear" not in n][
    [n for n in models if "Linear" not in n].index(
        next(n for n in models if "XGBoost" in n)
    )
]
best_tree_model = models[best_tree_model_name]
importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": best_tree_model.feature_importances_
}).sort_values("importance", ascending=False)
importance_df.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "feature_importance.csv"), index=False)
print(f"Saved feature importance -> data/processed/feature_importance.csv")

# Save a small JSON summary used by the README / LinkedIn writeup
summary = {
    "boosting_engine": XGB_ENGINE,
    "using_real_xgboost": USING_REAL_XGBOOST,
    "train_days": len(X_train),
    "test_days": len(X_test),
    "best_model": best_model_name,
    "best_model_metrics": results_df.iloc[0].to_dict(),
}
with open(os.path.join(PROJECT_ROOT, "data", "processed", "run_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
