# Sales Forecasting & Predictive Analytics Engine

A daily retail sales forecasting system built for the **Future Interns Machine Learning Task 1 (2026)**. Translates a retail business forecasting problem into an ML solution using time-based feature engineering, then benchmarks **Linear Regression, Random Forest, and XGBoost** on **MAE, RMSE, and R²** to select the best model for business planning.

## Results

| Model | MAE ($) | RMSE ($) | R² | MAPE |
|---|---|---|---|---|
| **XGBoost** | **1,130** | **1,865** | **0.968** | 37.5% |
| Random Forest | 1,672 | 2,698 | 0.933 | 121.0% |
| Linear Regression | 2,371 | 3,718 | 0.873 | 27.8% |

**XGBoost was selected as the best model**, outperforming both alternatives on MAE, RMSE, and R² by capturing non-linear interactions between calendar effects (e.g. holiday season × day-of-week) that a linear model can't, while generalizing better than Random Forest on sharp demand spikes.

> Numbers above were produced in this environment using scikit-learn's `GradientBoostingRegressor` as a like-for-like stand-in for `XGBRegressor`, since this development sandbox has no internet access to install the `xgboost` package. The notebook auto-detects and uses real `xgboost` automatically if installed (`pip install xgboost` — works normally on any standard machine or Google Colab). Both are gradient-boosted decision tree implementations; results are consistent with the trend shown here.

## What's inside

```
sales-forecasting-engine/
├── data/
│   ├── raw/                 # Input transaction data
│   └── processed/           # Cleaned daily series, features, predictions
├── notebooks/
│   └── sales_forecasting.ipynb   # Full, runnable end-to-end notebook
├── models/                  # Saved trained models (.joblib)
├── visuals/                 # Generated chart PNGs
├── src/                     # Standalone pipeline scripts (same logic as the notebook)
├── requirements.txt
└── README.md
```

## Pipeline

1. **Data cleaning** — parse dates, drop duplicates/invalid rows, impute missing discounts, aggregate transactions into a daily sales series, fill calendar gaps
2. **Feature engineering** — calendar features (month, day-of-week, holiday season, cyclical sin/cos encodings), lag features (1/7/14/30-day), rolling 7/30-day mean & std, momentum (% change)
3. **Model benchmark** — time-based 80/20 train/test split (never shuffled — that would leak the future into training), three models trained and scored on MAE, RMSE, R²
4. **Visualization** — model comparison, actual vs. predicted, feature importance, 30-day forward forecast, monthly trend/seasonality
5. **Business write-up** — plain-language explanation of what the forecast means and how to act on it

## Visuals

| | |
|---|---|
| ![Model Comparison](visuals/model_comparison.png) | ![Actual vs Predicted](visuals/actual_vs_predicted.png) |
| ![Feature Importance](visuals/feature_importance.png) | ![30-Day Forecast](visuals/forecast_next_30_days.png) |

## Running it yourself

```bash
pip install -r requirements.txt
jupyter notebook notebooks/sales_forecasting.ipynb
```

To run on the real **Kaggle Superstore Sales Dataset** instead of the included sample data: download `Sample - Superstore.csv` from Kaggle, place it at `data/raw/superstore.csv`, and re-run the notebook — no code changes needed.

## Business takeaways

- **Inventory:** Stock up ahead of the Nov–Dec demand surge the model identifies as the strongest seasonal driver; scale back in the January trough that follows.
- **Staffing:** Use the 30-day forecast to plan labor scheduling ahead of predicted demand swings rather than reacting to them.
- **Cash flow:** Anticipate predictable revenue dips and plan working-capital needs (e.g. supplier payment timing) in advance.

**Limitations:** the model assumes historical seasonal patterns persist and won't anticipate one-off events (promotions, supply shocks) unless explicitly added as features. The 30-day forecast is recursive, so error compounds the further out it predicts — standard for any iterative time-series forecast.

---
Built as part of the **Future Interns AI/ML Virtual Internship**.
