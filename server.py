"""Flower server setup and custom strategy."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.ensemble import RandomForestRegressor

from model import get_model
from train_utils import evaluate_model, get_weights, set_weights


class RandomForestStrategy(fl.server.strategy.FedAvg):
    """Custom strategy that aggregates RandomForest models by merging trees."""

    def __init__(self, eval_fn: Callable, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eval_fn = eval_fn

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):  # type: ignore[override]
        model = get_model()
        return ndarrays_to_parameters(get_weights(model))

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, float]]:  # type: ignore[override]
        if not results:
            return ndarrays_to_parameters([]), {}

        # Start from the first client's model and extend its trees
        base_model = set_weights(parameters_to_ndarrays(results[0][1].parameters))
        estimators = list(base_model.estimators_)
        for _, fit_res in results[1:]:
            client_model = set_weights(parameters_to_ndarrays(fit_res.parameters))
            estimators.extend(client_model.estimators_)

        base_model.estimators_ = estimators
        base_model.n_estimators = len(estimators)

        aggregated_parameters = ndarrays_to_parameters(get_weights(base_model))
        return aggregated_parameters, {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, float]]:  # type: ignore[override]
        return super().aggregate_evaluate(rnd, results, failures)

    def evaluate(
        self, rnd: int, parameters: fl.common.Parameters
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:  # type: ignore[override]
        # Centralized evaluation using the provided evaluation function
        model = set_weights(parameters_to_ndarrays(parameters))
        loss, metrics = self.eval_fn(model)
        return loss, metrics


def get_evaluate_fn(X_test: np.ndarray, y_test: np.ndarray) -> Callable:
    """Create an evaluation function for server-side metrics."""

    def evaluate(model: RandomForestRegressor) -> Tuple[float, Dict[str, float]]:
        mse, r2 = evaluate_model(model, X_test, y_test)
        print(f"\nServer evaluation -- MSE: {mse:.4f}, R²: {r2:.4f}")
        return mse, {"mse": mse, "r2": r2}

    return evaluate


def get_strategy(X_test: np.ndarray, y_test: np.ndarray) -> RandomForestStrategy:
    eval_fn = get_evaluate_fn(X_test, y_test)
    strategy = RandomForestStrategy(
        eval_fn=eval_fn,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )
    return strategy


__all__ = ["get_strategy"]
