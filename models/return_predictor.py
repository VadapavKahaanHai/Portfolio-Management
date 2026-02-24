# models/return_predictor.py — Step 5: Expected Return Estimation (ML)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib, os, warnings
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

import sys
from config import RETURN_HORIZON, CV_SPLITS, RANDOM_STATE

# Feature columns to use for prediction (exclude leaking columns)
EXCLUDE_COLS = ["date", "stock", "fwd_return"]


def train_return_model(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    method: str = "lgbm",
    save_path: str = "models/return_model.pkl",
) -> dict:
    """
    Train a forward return prediction model.

    method options:
      - ridge    : Ridge regression (fast baseline)
      - lasso    : Lasso regression
      - lgbm     : LightGBM (recommended — best accuracy)
      - gbm      : Sklearn GradientBoosting

    Uses TimeSeriesSplit to avoid look-ahead bias.
    Returns {"model": ..., "scaler": ..., "feature_cols": ..., "cv_scores": ...}
    """
    print(f"\n{'─'*60}")
    print(f"  Training Return Predictor  (method={method}, horizon={RETURN_HORIZON}d)")
    print(f"{'─'*60}")

    feat_cols = [c for c in X_all.columns if c not in EXCLUDE_COLS]
    X = X_all[feat_cols].fillna(0).values
    y = y_all.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Time-series cross-validation ──────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_cv = _build_model(method)
        model_cv.fit(X_tr, y_tr)
        preds = model_cv.predict(X_val)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        ic    = np.corrcoef(preds, y_val)[0, 1]  # information coefficient
        cv_scores.append({"fold": fold+1, "rmse": rmse, "ic": ic})
        print(f"  Fold {fold+1}: RMSE={rmse:.4f}  IC={ic:.3f}")

    # ── Final model on full data ──────────────────────────────────────────
    final_model = _build_model(method)
    final_model.fit(X_scaled, y)

    avg_ic   = np.mean([s["ic"]   for s in cv_scores])
    avg_rmse = np.mean([s["rmse"] for s in cv_scores])
    print(f"\n  ✓ Final model | Avg IC={avg_ic:.3f} | Avg RMSE={avg_rmse:.4f}")

    result = {
        "model": final_model,
        "scaler": scaler,
        "feature_cols": feat_cols,
        "cv_scores": cv_scores,
        "avg_ic": avg_ic,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(result, save_path)
    print(f"  ✓ Model saved → {save_path}")

    return result


def predict_expected_returns(
    feature_table: pd.DataFrame,
    model_bundle: dict,
) -> pd.Series:
    """
    Given the latest feature snapshot, predict forward returns for all stocks.
    Returns a Series: stock → predicted_return
    """
    feat_cols = model_bundle["feature_cols"]
    X = feature_table[feat_cols].fillna(0)
    X_scaled = model_bundle["scaler"].transform(X)
    preds = model_bundle["model"].predict(X_scaled)
    mu_hat = pd.Series(preds, index=feature_table.index, name="mu_hat")
    print(f"\n  ✓ Predicted expected returns (top 10):")
    print(mu_hat.sort_values(ascending=False).head(10).to_string())
    return mu_hat


def _build_model(method: str):
    if method == "ridge":
        return Ridge(alpha=1.0)
    elif method == "lasso":
        return Lasso(alpha=0.001, max_iter=5000)
    elif method == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    elif method == "gbm":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
