"""Entry point for the federated learning simulation."""

from __future__ import annotations

import flwr as fl

from client import ThicknessClient
from data_loader import load_client_data
from model import get_model
from server import get_strategy


def main() -> None:
    # Load and partition the dataset
    client_partitions, test_set = load_client_data("data")
    X_test, y_test = test_set

    def client_fn(cid: str):
        cid_int = int(cid)
        X_tr, y_tr, X_val, y_val = client_partitions[cid_int]
        model = get_model()
        return ThicknessClient(model, (X_tr, y_tr), (X_val, y_val))

    strategy = get_strategy(X_test, y_test)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_partitions),
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
