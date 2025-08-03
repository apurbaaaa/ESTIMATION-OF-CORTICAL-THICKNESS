"""Entry point for the federated learning simulation."""

from __future__ import annotations

import flwr as fl

from client import ThicknessClient
from data_loader import load_client_data
from model import ModelName
from server import get_strategy


def run_simulation(model_name: ModelName) -> None:
    """Run a federated simulation for the specified model."""

    client_partitions, test_set = load_client_data("data")
    X_test, y_test = test_set

    def client_fn(cid: str):
        cid_int = int(cid)
        X_tr, y_tr, X_val, y_val = client_partitions[cid_int]
        return ThicknessClient(model_name, (X_tr, y_tr), (X_val, y_val))

    strategy = get_strategy(X_test, y_test, model_name)

    print(f"\n>>> Federated training for model: {model_name}")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_partitions),
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


def main() -> None:
    for name in ["random_forest", "xgboost", "catboost", "knn"]:
        run_simulation(name)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
