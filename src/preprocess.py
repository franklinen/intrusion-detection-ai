"""
preprocess.py
-------------
Preprocessing pipeline for the UNSW-NB15 intrusion detection dataset.
Handles scaling, one-hot encoding, and artifact persistence (joblib).
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ── Column configuration ──────────────────────────────────────────────────────
DROP_COLS = ["id", "attack_cat"]
TARGET_COL = "label"
CAT_COLS = ["proto", "service", "state"]

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
ENCODER_PATH = ARTIFACT_DIR / "encoder.joblib"
FEATURE_NAMES_PATH = ARTIFACT_DIR / "feature_names.joblib"


# ── Public helpers ────────────────────────────────────────────────────────────

def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw UNSW-NB15 CSV and return a clean DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    logger.info("Loaded %d rows × %d cols from %s", *df.shape, csv_path)
    return df


def fit_transform(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit preprocessors on *df* and return (X, y) as numpy arrays.
    Persists fitted scaler + encoder to ARTIFACT_DIR.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values.astype(np.int32)

    num_cols = [c for c in X_raw.select_dtypes(include=["float64", "int64"]).columns
                if c not in CAT_COLS]
    cat_cols = [c for c in CAT_COLS if c in X_raw.columns]

    # Fit & transform scaler
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_raw[num_cols])

    # Fit & transform encoder
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(X_raw[cat_cols])

    X = np.hstack([X_num, X_cat]).astype(np.float32)

    # Persist
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
    joblib.dump({"names": feature_names, "num_cols": num_cols, "cat_cols": cat_cols},
                FEATURE_NAMES_PATH)

    logger.info("Preprocessing complete → X shape %s", X.shape)
    return X, y


def transform(df: pd.DataFrame) -> np.ndarray:
    """
    Transform *df* using previously fitted artifacts.
    Suitable for inference-time preprocessing.
    """
    if not SCALER_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Preprocessing artifacts not found. Run fit_transform() first."
        )

    scaler: StandardScaler = joblib.load(SCALER_PATH)
    encoder: OneHotEncoder = joblib.load(ENCODER_PATH)
    meta: dict = joblib.load(FEATURE_NAMES_PATH)

    num_cols: list = meta["num_cols"]
    cat_cols: list = meta["cat_cols"]

    # Normalise column names
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    X_num = scaler.transform(df[num_cols])
    X_cat = encoder.transform(df[cat_cols])

    return np.hstack([X_num, X_cat]).astype(np.float32)