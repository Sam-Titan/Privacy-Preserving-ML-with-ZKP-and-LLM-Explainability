"""
models/preprocess.py
────────────────────
Loads the UCI Heart Disease dataset, encodes categoricals,
imputes missing values, and fits/loads a StandardScaler.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MODELS_DIR

# ── Categorical encoding maps ─────────────────────────────────────────────────

SEX_MAP = {
    "Male": 1, "male": 1, "M": 1, "m": 1, 1: 1,
    "Female": 0, "female": 0, "F": 0, "f": 0, 0: 0,
}
CP_MAP = {
    "typical angina": 0, "atypical angina": 1,
    "non-anginal": 2, "asymptomatic": 3,
    0: 0, 1: 1, 2: 2, 3: 3,
}
RESTECG_MAP = {
    "normal": 0, "stt abnormality": 1,
    "lv hypertrophy": 2,
    0: 0, 1: 1, 2: 2,
}
EXANG_MAP = {
    "True": 1, "true": 1, True: 1, "Yes": 1, "yes": 1, 1: 1,
    "False": 0, "false": 0, False: 0, "No": 0, "no": 0, 0: 0,
}
SLOPE_MAP = {
    "upsloping": 0, "flat": 1, "downsloping": 2,
    0: 0, 1: 1, 2: 2,
}
THAL_MAP = {
    "normal": 0, "fixed defect": 1, "reversible defect": 2,
    0: 0, 1: 1, 2: 2,
}

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# ── Public helpers ─────────────────────────────────────────────────────────────

def load_and_preprocess(data_path: str = None):
    """
    Returns X (float32 ndarray), y (float32 ndarray), feature_cols (list).
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "heart_disease.csv")

    df = pd.read_csv(data_path)

    # Drop bookkeeping columns if present
    df.drop(columns=[c for c in ("id", "origin") if c in df.columns],
            inplace=True, errors="ignore")

    # ── Target ───────────────────────────────────────────────────────────────
    # 'num' > 0 → heart disease
    df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
    df.drop(columns=["num"], inplace=True)

    # ── Categorical encoding ──────────────────────────────────────────────────
    for col, mapping in [
        ("sex",     SEX_MAP),
        ("cp",      CP_MAP),
        ("restecg", RESTECG_MAP),
        ("exang",   EXANG_MAP),
        ("slope",   SLOPE_MAP),
        ("thal",    THAL_MAP),
    ]:
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # ── Coerce everything to numeric & impute with column median ──────────────
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)

    present_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[present_features].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    return X, y, present_features


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit and persist a StandardScaler."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    return scaler


def load_scaler() -> StandardScaler:
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}. Run train.py first.")
    return joblib.load(path)


def save_feature_cols(feature_cols):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)


def load_feature_cols():
    path = os.path.join(MODELS_DIR, "feature_cols.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature_cols.json not found. Run train.py first.")
    with open(path) as f:
        return json.load(f)