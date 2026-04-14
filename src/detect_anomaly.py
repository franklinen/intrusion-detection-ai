"""
detect_anomaly.py
-----------------
Inference engine: loads all three trained models (LSTM, RF, XGBoost)
and exposes a unified `predict()` interface used by app.py and kafka_consumer.py.

Ensemble strategy: majority vote across the three classifiers.
Individual model scores are also returned for transparency.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess import transform
from sequence_builder import build_sequences

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
LSTM_MODEL_PATH = ARTIFACT_DIR / "lstm_model.keras"
RF_MODEL_PATH = ARTIFACT_DIR / "rf_model.joblib"
XGB_MODEL_PATH = ARTIFACT_DIR / "xgb_model.joblib"

THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    is_anomaly: bool
    ensemble_label: int
    confidence: float                  # mean probability across models
    model_scores: Dict[str, float] = field(default_factory=dict)
    model_labels: Dict[str, int] = field(default_factory=dict)


# ── Detector ─────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Thread-safe singleton that holds loaded model references.
    Instantiate once at application start-up.
    """

    def __init__(self) -> None:
        self._lstm: Optional[tf.keras.Model] = None
        self._rf = None
        self._xgb = None
        self._loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all model artefacts from ARTIFACT_DIR."""
        if self._loaded:
            return

        if not LSTM_MODEL_PATH.exists():
            raise FileNotFoundError(f"LSTM model not found at {LSTM_MODEL_PATH}")
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError(f"RF model not found at {RF_MODEL_PATH}")
        if not XGB_MODEL_PATH.exists():
            raise FileNotFoundError(f"XGBoost model not found at {XGB_MODEL_PATH}")

        logger.info("Loading LSTM model …")
        self._lstm = tf.keras.models.load_model(str(LSTM_MODEL_PATH))

        logger.info("Loading Random Forest model …")
        self._rf = joblib.load(RF_MODEL_PATH)

        logger.info("Loading XGBoost model …")
        self._xgb = joblib.load(XGB_MODEL_PATH)

        self._loaded = True
        logger.info("All models loaded successfully.")

    def unload(self) -> None:
        self._lstm = None
        self._rf = None
        self._xgb = None
        self._loaded = False

    # ── Core inference ────────────────────────────────────────────────────────

    def predict(
        self,
        records: List[Dict],
        threshold: float = THRESHOLD,
    ) -> List[DetectionResult]:
        """
        Run ensemble inference on a list of raw network-traffic record dicts.

        Parameters
        ----------
        records  : list of dicts with raw feature keys (same schema as training CSV)
        threshold: classification cut-off (default 0.5)

        Returns
        -------
        List[DetectionResult]
        """
        if not self._loaded:
            self.load()

        df = pd.DataFrame(records)
        X = transform(df)                          # (n, features) float32
        X_seq = build_sequences(X)                 # (n, 1, features)

        # LSTM
        lstm_probs: np.ndarray = self._lstm.predict(X_seq, verbose=0).flatten()

        # Random Forest
        rf_probs: np.ndarray = self._rf.predict_proba(X)[:, 1]

        # XGBoost
        xgb_probs: np.ndarray = self._xgb.predict_proba(X)[:, 1]

        results = []
        for i in range(len(records)):
            scores = {
                "lstm": float(lstm_probs[i]),
                "random_forest": float(rf_probs[i]),
                "xgboost": float(xgb_probs[i]),
            }
            labels = {k: int(v >= threshold) for k, v in scores.items()}
            mean_prob = float(np.mean(list(scores.values())))
            ensemble_label = int(sum(labels.values()) >= 2)  # majority vote

            results.append(
                DetectionResult(
                    is_anomaly=bool(ensemble_label),
                    ensemble_label=ensemble_label,
                    confidence=mean_prob,
                    model_scores=scores,
                    model_labels=labels,
                )
            )

        return results

    def predict_one(self, record: Dict, threshold: float = THRESHOLD) -> DetectionResult:
        """Convenience wrapper for single-record inference."""
        return self.predict([record], threshold=threshold)[0]


# ── Module-level singleton ────────────────────────────────────────────────────

detector = AnomalyDetector()





