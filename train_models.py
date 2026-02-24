"""
=============================================================================
STEP 2: MODEL TRAINING — Teaching the Computer to Predict Stock Prices
=============================================================================

ML CONCEPT: "Supervised Learning"
We show the model thousands of examples:
  INPUT  → [13 features from today]
  OUTPUT → [tomorrow's percentage return]

After seeing enough examples, it learns patterns and can make predictions
on NEW data it has never seen before.

Three algorithms compete for each stock — the best one wins:
  1. Huber Regressor     — linear model, robust to outlier days
  2. Random Forest       — 100 decision trees, handles complex patterns
  3. Gradient Boosting   — trees built sequentially, learns from errors

🎓 WHY HUBER LOSS?
Stock returns are noisy. A single earnings gap (±8%) shouldn't dominate
the entire loss. Huber loss combines the best of MSE and MAE:

  L(e) = ½·e²           if |e| ≤ ε   (smooth quadratic for small errors)
  L(e) = ε·(|e| − ½·ε)  if |e| > ε   (linear for large errors / outliers)

Properties:
  ✓ Differentiable at ALL points (unlike MAE which has a kink at 0)
  ✓ Balances small & large errors (unlike MSE which squares outliers)
  ✓ Threshold ε controls the transition (we use ε=1.35, the standard)
  ✓ Gradient-based optimizers converge cleanly
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_pipeline import STOCKS, FEATURE_COLUMNS, prepare_all_stocks


def huber_loss(y_true, y_pred, epsilon=1.35):
    """
    HUBER LOSS — the metric all 3 models optimise.

    For small residuals (|e| ≤ ε): quadratic → smooth gradients near zero.
    For large residuals (|e| > ε): linear → outliers don't explode the loss.

    Returns the mean Huber loss across all samples.
    """
    residuals = np.abs(y_true - y_pred)
    quadratic = np.minimum(residuals, epsilon)
    linear = residuals - quadratic
    return np.mean(0.5 * quadratic**2 + epsilon * linear)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    THE TRAINING LOOP

    1. SCALE features — normalise to mean=0, std=1 so no single
       feature dominates.
    2. TRAIN 3 models on training data (all use Huber-based loss).
    3. EVALUATE each on the held-out test data.
    4. Return the best performer (lowest Huber loss on test set).

    Metrics we measure:
      Huber  — Huber loss: our primary metric, robust to outliers
      MAE    — Mean Absolute Error: average % points we're off by
      RMSE   — Root Mean Squared Error: penalises big misses
      Dir    — Directional accuracy: % of days we predict up/down correctly
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ε for Huber loss — 1.35 is the standard choice (95% efficiency
    # relative to OLS under normal errors, far more robust to outliers)
    EPSILON = 1.35

    models = {
        "Huber Regressor": HuberRegressor(
            epsilon=EPSILON, max_iter=500, alpha=0.0001
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            criterion="squared_error",  # RF doesn't have native Huber,
            random_state=42,            # but we SELECT the winner by Huber
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            loss="huber",               # ← native Huber loss!
            alpha=0.95,                 # quantile for Huber transition
            random_state=42,
        ),
    }

    results = {}
    best_huber, best_model, best_name = float("inf"), None, None

    for name, model in models.items():
        print(f"    Training {name}...")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        h_loss = huber_loss(y_test.values, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Directional accuracy: did we predict the right direction?
        direction_correct = np.mean(
            (y_pred > 0) == (y_test.values > 0)
        ) * 100

        results[name] = {
            "huber_loss": round(float(h_loss * 1e4), 4),  # scaled ×10⁴ for readability
            "mae": round(mae * 100, 4),       # as percentage points
            "rmse": round(rmse * 100, 4),      # as percentage points
            "r2": round(r2, 4),
            "direction_acc": round(direction_correct, 1),
        }
        print(
            f"      Huber: {h_loss*1e4:.4f}×10⁻⁴ | MAE: {mae*100:.3f}% | "
            f"RMSE: {rmse*100:.3f}% | Dir: {direction_correct:.1f}%"
        )

        # Select winner by LOWEST Huber loss (not R²)
        if h_loss < best_huber:
            best_huber, best_model, best_name = h_loss, model, name

    print(f"    Winner: {best_name} (Huber: {best_huber*1e4:.4f}×10⁻⁴)")
    return best_model, best_name, scaler, results


def train_all_models():
    """
    🎓 FULL TRAINING PIPELINE
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
    print(f"\n{'=' * 76}")
    print(
        f"{'Stock':<8} {'Best Model':<22} {'Huber (×10⁻⁴)':<15} "
        f"{'Dir. Acc':<10} {'MAE (%)':<10}"
    )
    print("─" * 76)
    for t, r in report.items():
        m = r["best_metrics"]
        print(
            f"{t:<8} {r['best_model']:<22} {m['huber_loss']:<15} "
            f"{m['direction_acc']}%{'':<4} {m['mae']:.3f}%"
        )
    print(f"\n All models saved to models/")
    return report


if __name__ == "__main__":
    train_all_models()