"""Flower server setup and custom strategy supporting multiple models."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from model import ModelName, get_model
from train_utils import evaluate_model, get_weights, set_weights


class SklearnStrategy(fl.server.strategy.FedAvg):
    """Strategy that can aggregate various scikit-learn style models.

    ``RandomForestRegressor`` models are aggregated by concatenating the
    decision trees from all participating clients.  For all other estimators we
    do **not** average parameters; instead the client model achieving the
    highest R² score in a round is chosen and broadcast to the rest of the
    federation in the following round.
    """

    def __init__(self, model_name: ModelName, eval_fn: Callable, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.eval_fn = eval_fn

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):  # type: ignore[override]
        model = get_model(self.model_name)
        return ndarrays_to_parameters(get_weights(model))

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, float]]:  # type: ignore[override]
        if not results:
            return ndarrays_to_parameters([]), {}

        if self.model_name == "random_forest":
            base_model = get_model(self.model_name)
            base_model = set_weights(
                base_model, parameters_to_ndarrays(results[0][1].parameters)
            )
            estimators = list(base_model.estimators_)
            for _, fit_res in results[1:]:
                client_model = get_model(self.model_name)
                client_model = set_weights(
                    client_model, parameters_to_ndarrays(fit_res.parameters)
                )
                estimators.extend(client_model.estimators_)
            base_model.estimators_ = estimators
            base_model.n_estimators = len(estimators)
            aggregated_parameters = ndarrays_to_parameters(get_weights(base_model))
            return aggregated_parameters, {}

        # For all other estimators the ``fit`` results already contain full
        # models.  We simply select the one with the best R² score and broadcast
        # its parameters in the next round instead of averaging coefficients.
        best = max(results, key=lambda r: r[1].metrics.get("r2", float("-inf")))
        return best[1].parameters, {"r2": best[1].metrics.get("r2", 0.0)}

    def evaluate(
        self, rnd: int, parameters: fl.common.Parameters
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:  # type: ignore[override]
        model = get_model(self.model_name)
        model = set_weights(model, parameters_to_ndarrays(parameters))
        loss, metrics = self.eval_fn(model)
        return loss, metrics


def get_evaluate_fn(X_test: np.ndarray, y_test: np.ndarray) -> Callable:
    """Create an evaluation function for server-side metrics."""

    def evaluate(model) -> Tuple[float, Dict[str, float]]:
        mse, r2 = evaluate_model(model, X_test, y_test)
        print(f"\nServer evaluation -- MSE: {mse:.4f}, R²: {r2:.4f}")
        return mse, {"mse": mse, "r2": r2}

    return evaluate


def get_strategy(X_test: np.ndarray, y_test: np.ndarray, model_name: ModelName) -> SklearnStrategy:
    eval_fn = get_evaluate_fn(X_test, y_test)
    strategy = SklearnStrategy(
        model_name=model_name,
        eval_fn=eval_fn,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )
    return strategy


__all__ = ["get_strategy"]
