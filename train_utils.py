"""Utility functions for training and evaluation."""

from __future__ import annotations

import pickle
from typing import Any, List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def get_weights(model: Any) -> List[np.ndarray]:
    """Serialize a full scikit-learn model into a list of ``np.ndarray``.

    The entire model object is pickled so that arbitrary estimators can be
    transported between Flower server and clients without having to manually
    extract their internal parameters.
    """

    return [np.frombuffer(pickle.dumps(model), dtype=np.uint8)]


def set_weights(model: Any, weights: List[np.ndarray]) -> Any:
    """Update ``model`` with the provided serialized parameters.

    ``weights`` is expected to be the output of :func:`get_weights`.  If the
    deserialised object is of a different type than ``model`` the deserialised
    object is returned unchanged and ``model`` is left untouched.  This makes
    the function robust against accidentally mixing different estimator types
    across clients.
    """

    new_model = pickle.loads(weights[0].tobytes())

    # If the types match we can simply copy over the attributes which updates
    # the model in-place.  Returning the original instance avoids the caller
    # having to keep track of a newly created object.
    if isinstance(new_model, type(model)):
        model.__dict__.update(new_model.__dict__)
        return model

    # If types do not match, return the new model so the caller can decide
    # what to do with it (e.g. replace the old model).  No exception is raised
    # to avoid breaking the FL pipeline when model types are mismatched.
    return new_model


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute MSE and R² metrics for the given model and data."""
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    return float(mse), float(r2)


__all__ = ["get_weights", "set_weights", "evaluate_model"]
