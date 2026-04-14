"""
train_autoencoder.py
--------------------
End-to-end training script for the three intrusion-detection models:
  1. LSTM classifier
  2. Random Forest
  3. XGBoost

Run:
    python train_autoencoder.py --data UNSW_NB15_training-set.csv

All trained model artefacts are written to ARTIFACT_DIR (default: ./artifacts).
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import xgboost as xgb

from preprocess import fit_transform, load_raw
from sequence_builder import build_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
LSTM_MODEL_PATH = ARTIFACT_DIR / "lstm_model.keras"
RF_MODEL_PATH = ARTIFACT_DIR / "rf_model.joblib"
XGB_MODEL_PATH = ARTIFACT_DIR / "xgb_model.joblib"


# ── Model builders ────────────────────────────────────────────────────────────

def build_lstm(n_features: int) -> Sequential:
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(1, n_features)),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="lstm_intrusion_detector",
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)
    return model


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    auc = roc_auc_score(y_true, y_prob)
    logger.info("\n=== %s ===\nROC-AUC: %.4f\n%s", name, auc,
                classification_report(y_true, y_pred))


# ── Main training routine ─────────────────────────────────────────────────────

def train(csv_path: str, test_size: float = 0.2, epochs: int = 50,
          batch_size: int = 1024, random_state: int = 42) -> None:

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & preprocess
    df = load_raw(csv_path)
    X, y = fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train: %s  |  Test: %s", X_train.shape, X_test.shape)

    # ── LSTM ─────────────────────────────────────────────────────────────────
    X_train_seq = build_sequences(X_train)
    X_test_seq = build_sequences(X_test)

    lstm_model = build_lstm(X_train.shape[1])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(str(LSTM_MODEL_PATH), save_best_only=True, monitor="val_loss"),
    ]

    lstm_model.fit(
        X_train_seq, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    lstm_probs = lstm_model.predict(X_test_seq, batch_size=batch_size).flatten()
    lstm_preds = (lstm_probs > 0.5).astype(int)
    evaluate("LSTM", y_test, lstm_preds, lstm_probs)
    logger.info("LSTM model saved → %s", LSTM_MODEL_PATH)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    evaluate("Random Forest", y_test, rf_preds, rf_probs)
    joblib.dump(rf, RF_MODEL_PATH)
    logger.info("Random Forest saved → %s", RF_MODEL_PATH)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    evaluate("XGBoost", y_test, xgb_preds, xgb_probs)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    logger.info("XGBoost saved → %s", XGB_MODEL_PATH)

    # ── Summary ───────────────────────────────────────────────────────────────
    comparison = pd.DataFrame(
        {
            "Model": ["LSTM", "Random Forest", "XGBoost"],
            "ROC_AUC": [
                roc_auc_score(y_test, lstm_probs),
                roc_auc_score(y_test, rf_probs),
                roc_auc_score(y_test, xgb_probs),
            ],
        }
    )
    logger.info("\nModel Comparison:\n%s", comparison.to_string(index=False))


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intrusion-detection models")
    parser.add_argument("--data", required=True, help="Path to UNSW_NB15 CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    train(
        csv_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
    )