"""
=============================================================================
STEP 2: MODEL TRAINING — Teaching the Computer to Predict Stock Prices
=============================================================================

🎓 ML CONCEPT: "Supervised Learning"
We show the model thousands of examples:
  INPUT  → [13 features from today]
  OUTPUT → [tomorrow's actual closing price]

After seeing enough examples, it learns patterns and can make predictions
on NEW data it has never seen before.

Three algorithms compete for each stock — the best one wins:
  1. Linear Regression  — fast, simple, assumes linear relationships
  2. Random Forest      — 100 decision trees, handles complex patterns
  3. Gradient Boosting  — trees built sequentially, learns from errors
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_pipeline import STOCKS, FEATURE_COLUMNS, prepare_all_stocks


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    THE TRAINING LOOP

    1. SCALE features — normalise everything to mean=0, std=1 so no single
       feature dominates (e.g. Volume in millions vs daily_return ~0.01).
    2. TRAIN 3 models on the training data.
    3. EVALUATE each on the held-out test data.
    4. Return the best performer.

    Metrics we measure:
      MAE  — Mean Absolute Error:  average $ we're off by
      RMSE — Root Mean Squared Error:  penalises big misses
      R²   — Coefficient of determination:  1.0 perfect, 0.0 useless
      MAPE — Mean Absolute Percentage Error:  error as % of price
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    best_r2, best_model, best_name = -float("inf"), None, None

    for name, model in models.items():
        print(f"    Training {name}...")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Directional accuracy: did we predict the right direction (up/down)?
        direction_correct = np.mean(
            (y_pred > 0) == (y_test.values > 0)
        ) * 100

        results[name] = {
            "mae": round(mae * 100, 4),       # as percentage points
            "rmse": round(rmse * 100, 4),      # as percentage points
            "r2": round(r2, 4),
            "direction_acc": round(direction_correct, 1),
        }
        print(
            f"      MAE: {mae*100:.3f}% | RMSE: {rmse*100:.3f}% | "
            f"R²: {r2:.4f} | Direction: {direction_correct:.1f}%"
        )

        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, name

    print(f"    Winner: {best_name} (R² = {best_r2:.4f})")
    return best_model, best_name, scaler, results


def train_all_models():
    """
    FULL TRAINING PIPELINE
    For each stock: load data → chronological split → train → evaluate → save.
    """
    print("=" * 60)
    print(" MODEL TRAINING PIPELINE")
    print("=" * 60)

    if not os.path.exists("data/data_summary.json"):
        print("\n Running data pipeline first...\n")
        prepare_all_stocks()

    os.makedirs("models", exist_ok=True)
    report = {}

    for ticker, name in STOCKS.items():
        filepath = f"data/{ticker}_featured.csv"
        if not os.path.exists(filepath):
            print(f"\n  No data for {ticker}, skipping")
            continue

        print(f"\n{'─' * 50}")
        print(f"  {name} ({ticker})")
        print(f"{'─' * 50}")

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        # CHRONOLOGICAL SPLIT
        # We MUST use the past to predict the future, never the reverse.
        split = int(len(df) * 0.8)
        X, y = df[FEATURE_COLUMNS], df["next_return"]
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

        best_model, best_name, scaler, results = train_and_evaluate(
            X_train, X_test, y_train, y_test
        )

        # Save model + scaler to disk
        joblib.dump(best_model, f"models/{ticker}_model.pkl")
        joblib.dump(scaler, f"models/{ticker}_scaler.pkl")

        report[ticker] = {
            "stock_name": name,
            "best_model": best_name,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "results": results,
            "best_metrics": results[best_name],
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_used": FEATURE_COLUMNS,
        }

    with open("models/training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 72}")
    print(
        f"{'Stock':<8} {'Best Model':<22} {'MAE (%)':<10} "
        f"{'Dir. Acc':<10} {'R²':<10}"
    )
    print("─" * 72)
    for t, r in report.items():
        m = r["best_metrics"]
        print(
            f"{t:<8} {r['best_model']:<22} {m['mae']:.3f}%{'':<3} "
            f"{m['direction_acc']}%{'':<4} {m['r2']}"
        )
    print(f"\n All models saved to models/")
    return report


if __name__ == "__main__":
    train_all_models()