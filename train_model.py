"""
train_model.py — Global Rainfall Engine & Climate Intelligence
==============================================================
Pipeline:
  1. Load & clean climate_change_dataset.csv
  2. Derive water-status labels from per-country Z-scores
  3. Feature engineering: lag-1, lag-2, rolling-3yr average per country
  4. Train Gradient Boosting Regressor on rainfall
  5. Evaluate with RMSE, MAE, R²
  6. Persist model + preprocessor via joblib → model_artifacts/

Run:
    python train_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH      = "climate_change_dataset.csv"
ARTIFACTS_DIR  = "model_artifacts"
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "rainfall_model.pkl")
META_PATH      = os.path.join(ARTIFACTS_DIR, "feature_meta.pkl")

FEATURE_COLS   = [
    "Avg Temperature (°C)",
    "CO2 Emissions (Tons/Capita)",
    "Extreme Weather Events",
    "Rainfall_lag1",
    "Rainfall_lag2",
    "Rainfall_roll3",
]
TARGET_COL     = "Rainfall (mm)"

# Z-score thresholds for water-status labelling
ZSCORE_LOW     = -0.60   # below this  → Low (Water Difficulty)
ZSCORE_HIGH    =  0.60   # above this  → Heavy (Surplus)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, enforce types, drop rows missing critical fields."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Coerce numeric columns
    numeric_cols = [
        "Year", TARGET_COL,
        "Avg Temperature (°C)", "CO2 Emissions (Tons/Capita)",
        "Sea Level Rise (mm)", "Population",
        "Renewable Energy (%)", "Extreme Weather Events", "Forest Area (%)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the target or key features are missing
    essential = ["Year", "Country", TARGET_COL, "Avg Temperature (°C)",
                 "CO2 Emissions (Tons/Capita)", "Extreme Weather Events"]
    df.dropna(subset=essential, inplace=True)
    df["Year"] = df["Year"].astype(int)

    print(f"[load]  {len(df):,} valid rows | "
          f"{df['Country'].nunique()} countries | "
          f"Years {df['Year'].min()}–{df['Year'].max()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — WATER-STATUS LABELS (Z-score per country)
# ─────────────────────────────────────────────────────────────────────────────
def assign_water_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each row's rainfall Z-score relative to the country's own
    historical mean and std, then label:
      z < ZSCORE_LOW  → 'Low (Water Difficulty)'
      z > ZSCORE_HIGH → 'Heavy (Surplus)'
      else            → 'Decent (Normal)'
    """
    stats = (
        df.groupby("Country")[TARGET_COL]
          .agg(country_mean="mean", country_std="std")
          .reset_index()
    )
    # Countries with only one observation get std=0 → treat as Decent
    stats["country_std"] = stats["country_std"].fillna(1.0).replace(0, 1.0)

    df = df.merge(stats, on="Country", how="left")
    df["rainfall_zscore"] = (
        (df[TARGET_COL] - df["country_mean"]) / df["country_std"]
    )

    def label(z):
        if z < ZSCORE_LOW:
            return "Low (Water Difficulty)"
        elif z > ZSCORE_HIGH:
            return "Heavy (Surplus)"
        return "Decent (Normal)"

    df["Water_Status"] = df["rainfall_zscore"].apply(label)

    dist = df["Water_Status"].value_counts().to_dict()
    print(f"[label] Water status distribution: {dist}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per country, sort by year and compute:
      - Rainfall_lag1   : rainfall value 1 year prior
      - Rainfall_lag2   : rainfall value 2 years prior
      - Rainfall_roll3  : 3-year rolling mean (min_periods=1)
    Fill any NaNs from lags with the country mean rainfall.
    """
    df = df.sort_values(["Country", "Year"]).copy()

    df["Rainfall_lag1"]  = (
        df.groupby("Country")[TARGET_COL].shift(1)
    )
    df["Rainfall_lag2"]  = (
        df.groupby("Country")[TARGET_COL].shift(2)
    )
    df["Rainfall_roll3"] = (
        df.groupby("Country")[TARGET_COL]
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Back-fill lag NaNs with the country historical mean
    country_means = df.groupby("Country")[TARGET_COL].transform("mean")
    for col in ["Rainfall_lag1", "Rainfall_lag2", "Rainfall_roll3"]:
        df[col] = df[col].fillna(country_means)

    print(f"[feats] Feature columns: {FEATURE_COLS}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    """
    Train a Gradient Boosting Regressor inside a StandardScaler pipeline.
    Returns the fitted pipeline and the test split for evaluation.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])

    print(f"[train] Training on {len(X_train):,} rows …")
    model_pipeline.fit(X_train, y_train)
    print("[train] Training complete.")
    return model_pipeline, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print("\n" + "═" * 45)
    print("  MODEL EVALUATION (held-out 20%)")
    print("═" * 45)
    print(f"  RMSE : {rmse:>10.2f} mm")
    print(f"  MAE  : {mae:>10.2f} mm")
    print(f"  R²   : {r2:>10.4f}")
    print("═" * 45 + "\n")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PERSIST ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(pipeline, df: pd.DataFrame, metrics: dict):
    """
    Persist:
      - rainfall_model.pkl  : fitted sklearn pipeline
      - feature_meta.pkl    : dict with feature list, country stats,
                               year range, full processed dataframe
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    country_stats = (
        df.groupby("Country")[TARGET_COL]
          .agg(mean="mean", std="std")
          .reset_index()
          .rename(columns={"mean": "country_mean", "std": "country_std"})
    )
    country_stats["country_std"] = (
        country_stats["country_std"].fillna(1.0).replace(0, 1.0)
    )

    meta = {
        "feature_cols"   : FEATURE_COLS,
        "target_col"     : TARGET_COL,
        "zscore_low"     : ZSCORE_LOW,
        "zscore_high"    : ZSCORE_HIGH,
        "country_stats"  : country_stats,
        "year_min"       : int(df["Year"].min()),
        "year_max"       : int(df["Year"].max()),
        "processed_df"   : df,
        "metrics"        : metrics,
    }
    joblib.dump(meta, META_PATH)

    print(f"[save]  Model  → {MODEL_PATH}")
    print(f"[save]  Meta   → {META_PATH}")
    print("[done]  All artifacts saved. Run `streamlit run app.py` to launch.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df          = load_and_clean(DATA_PATH)
    df          = assign_water_status(df)
    df          = engineer_features(df)
    pipeline, X_test, y_test = train(df)
    metrics     = evaluate(pipeline, X_test, y_test)
    save_artifacts(pipeline, df, metrics)
