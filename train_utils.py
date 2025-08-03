"""Utility functions for training and evaluation."""

from __future__ import annotations

import pickle
from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def get_weights(model) -> List[np.ndarray]:
    """Serialize a scikit-learn model into a list of ``np.ndarray``."""
    return [np.frombuffer(pickle.dumps(model), dtype=np.uint8)]


def set_weights(weights: List[np.ndarray]):
    """Deserialize model weights back into a scikit-learn model."""
    return pickle.loads(weights[0].tobytes())


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute MSE and R² metrics for the given model and data."""
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    return float(mse), float(r2)


__all__ = ["get_weights", "set_weights", "evaluate_model"]
