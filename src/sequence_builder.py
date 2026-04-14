"""
sequence_builder.py
-------------------
Converts flat 2-D feature arrays into the 3-D tensors (samples, timesteps, features)
required by the LSTM model.

For the current single-timestep architecture, timesteps=1.
The module is written generically so it can be extended to sliding-window
sequences if the project evolves to streaming / stateful LSTM inference.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_sequences(
    X: np.ndarray,
    timesteps: int = 1,
) -> np.ndarray:
    """
    Reshape a 2-D feature matrix into a 3-D LSTM-compatible tensor.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    timesteps : int
        Number of time steps per sample.
        - timesteps=1  → single-step inference (current model).
        - timesteps>1  → sliding-window; the last (n_samples - timesteps + 1)
          samples are returned.

    Returns
    -------
    np.ndarray, shape (n_sequences, timesteps, n_features)
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {X.shape}")

    n_samples, n_features = X.shape

    if timesteps == 1:
        reshaped = X.reshape(n_samples, 1, n_features)
        logger.debug("Built single-step sequences: %s", reshaped.shape)
        return reshaped

    # Sliding-window approach
    if n_samples < timesteps:
        raise ValueError(
            f"Not enough samples ({n_samples}) for timesteps={timesteps}"
        )

    n_sequences = n_samples - timesteps + 1
    sequences = np.stack(
        [X[i : i + timesteps] for i in range(n_sequences)], axis=0
    )
    logger.debug("Built sliding-window sequences: %s", sequences.shape)
    return sequences


def flatten_sequences(X_seq: np.ndarray) -> np.ndarray:
    """
    Inverse of build_sequences for models that consume flat 2-D input
    (e.g. Random Forest, XGBoost).

    Parameters
    ----------
    X_seq : np.ndarray, shape (n_samples, timesteps, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples, timesteps * n_features)
    """
    if X_seq.ndim == 2:
        return X_seq  # already flat
    n_samples, timesteps, n_features = X_seq.shape
    return X_seq.reshape(n_samples, timesteps * n_features)