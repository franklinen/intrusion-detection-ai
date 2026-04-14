"""
shap_explain.py
---------------
SHAP-based explainability for the XGBoost intrusion-detection model.

Exposes:
  - explain_batch()  → returns per-sample SHAP values + feature importance
  - explain_one()    → single-record explanation dict (used by /explain endpoint)
  - summary_plot()   → saves a SHAP beeswarm PNG (useful for offline analysis)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from preprocess import transform

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
XGB_MODEL_PATH = ARTIFACT_DIR / "xgb_model.joblib"
FEATURE_NAMES_PATH = ARTIFACT_DIR / "feature_names.joblib"

# Max samples sent to SHAP to keep latency acceptable at inference time
MAX_EXPLAIN_SAMPLES = int(os.getenv("MAX_EXPLAIN_SAMPLES", "200"))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_explainer():
    """Lazy-load the TreeExplainer (cached after first call)."""
    global _EXPLAINER  # noqa: PLW0603
    if _EXPLAINER is None:
        model = joblib.load(XGB_MODEL_PATH)
        _EXPLAINER = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialised.")
    return _EXPLAINER


def _load_feature_names() -> List[str]:
    meta: dict = joblib.load(FEATURE_NAMES_PATH)
    return meta["names"]


_EXPLAINER = None   # module-level cache


# ── Public API ────────────────────────────────────────────────────────────────

def explain_batch(
    records: List[Dict],
    max_samples: int = MAX_EXPLAIN_SAMPLES,
) -> Dict:
    """
    Compute SHAP values for a list of raw network-traffic record dicts.

    Returns
    -------
    dict with keys:
      - shap_values   : list of per-sample SHAP value lists
      - feature_names : ordered list of feature names
      - base_value    : SHAP base (expected) value
      - mean_abs_shap : mean |SHAP| per feature (global importance)
    """
    df = pd.DataFrame(records[:max_samples])
    X = transform(df)

    feature_names = _load_feature_names()
    explainer = _load_explainer()
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0).tolist()
    feature_importance = sorted(
        zip(feature_names, mean_abs),
        key=lambda t: t[1],
        reverse=True,
    )

    return {
        "shap_values": shap_values.tolist(),
        "feature_names": feature_names,
        "base_value": float(explainer.expected_value),
        "mean_abs_shap": [
            {"feature": f, "importance": round(v, 6)}
            for f, v in feature_importance
        ],
    }


def explain_one(record: Dict) -> Dict:
    """
    Single-record explanation suitable for the /explain API endpoint.

    Returns
    -------
    dict with keys:
      - feature_contributions: [{feature, shap_value, direction}]
      - base_value
      - predicted_log_odds
    """
    result = explain_batch([record], max_samples=1)
    shap_vals = result["shap_values"][0]
    feature_names = result["feature_names"]
    base_value = result["base_value"]

    contributions = sorted(
        [
            {
                "feature": feat,
                "shap_value": round(val, 6),
                "direction": "attack" if val > 0 else "normal",
            }
            for feat, val in zip(feature_names, shap_vals)
        ],
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )

    return {
        "feature_contributions": contributions,
        "base_value": base_value,
        "predicted_log_odds": round(base_value + sum(shap_vals), 6),
    }


def summary_plot(
    records: List[Dict],
    output_path: str = "shap_summary.png",
    max_samples: int = MAX_EXPLAIN_SAMPLES,
) -> str:
    """
    Save a SHAP beeswarm summary plot to *output_path* and return the path.
    Intended for offline/batch analysis, not real-time API calls.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    df = pd.DataFrame(records[:max_samples])
    X = transform(df)
    feature_names = _load_feature_names()

    explainer = _load_explainer()
    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved → %s", output_path)
    return output_path