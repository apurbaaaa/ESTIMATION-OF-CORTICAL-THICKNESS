"""Flower client implementation for cortical thickness estimation."""

from __future__ import annotations

from typing import Tuple

import flwr as fl
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from model import ModelName, get_model
from train_utils import get_weights, set_weights


class ThicknessClient(fl.client.NumPyClient):
    """Flower ``NumPyClient`` that can train different regressors."""

    def __init__(
        self,
        model_name: ModelName,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.model_name = model_name
        self.model = get_model(model_name)
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data

    # Flower client API -------------------------------------------------
    def get_parameters(self, config):  # type: ignore[override]
        return get_weights(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        # Update the existing model instance with the received parameters.
        # ``set_weights`` will keep ``self.model`` in-place when the estimator
        # type matches, avoiding the creation of a brand new object every round.
        self.model = set_weights(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        preds = self.model.predict(self.X_val)
        mse = mean_squared_error(self.y_val, preds)
        r2 = r2_score(self.y_val, preds)
        return get_weights(self.model), len(self.X_train), {"mse": float(mse), "r2": float(r2)}

    def evaluate(self, parameters, config):  # type: ignore[override]
        # Update model with server-provided parameters before evaluation
        self.model = set_weights(self.model, parameters)
        preds = self.model.predict(self.X_val)
        mse = mean_squared_error(self.y_val, preds)
        r2 = r2_score(self.y_val, preds)
        return float(mse), len(self.X_val), {"mse": float(mse), "r2": float(r2)}


__all__ = ["ThicknessClient"]
