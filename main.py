"""Entry point for the federated learning simulation."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable

import flwr as fl

from client import ThicknessClient
from data_loader import load_client_data
from model import ModelName
from server import get_strategy


def run_simulation(
    model_name: ModelName,
    num_rounds: int = 3,
    metrics_file: str | None = None,
) -> None:
    """Run a federated simulation for the specified model."""

    client_partitions, test_set = load_client_data("data")
    X_test, y_test = test_set

    def client_fn(cid: str):
        cid_int = int(cid)
        X_tr, y_tr, X_val, y_val = client_partitions[cid_int]
        return ThicknessClient(model_name, (X_tr, y_tr), (X_val, y_val))

    strategy = get_strategy(X_test, y_test, model_name)

    print(f"\n>>> Federated training for model: {model_name}")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_partitions),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # ------------------------------------------------------------------
    # Optional metric logging
    # ------------------------------------------------------------------
    if metrics_file:
        rounds: Iterable[int] = sorted(
            {r for v in history.metrics_centralized.values() for r, _ in v}
        )
        with open(metrics_file, "w", newline="") as fp:
            writer = csv.writer(fp)
            header = ["round"] + list(history.metrics_centralized.keys())
            writer.writerow(header)
            for rnd in rounds:
                row = [rnd]
                for key in history.metrics_centralized.keys():
                    val = next(
                        (v for r, v in history.metrics_centralized[key] if r == rnd),
                        "",
                    )
                    row.append(val)
                writer.writerow(row)
        print(f"Saved metrics to {metrics_file}")
    else:
        for key, values in history.metrics_centralized.items():
            for rnd, val in values:
                print(f"Round {rnd:>2d} {key}: {val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run federated simulations")
    parser.add_argument(
        "--model",
        type=str,
        choices=["random_forest", "xgboost", "catboost", "knn", "all"],
        default="random_forest",
        help="Which model to train. Use 'all' to iterate over every model sequentially.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of federated training rounds to run",
    )
    parser.add_argument(
        "--log-metrics",
        type=str,
        default=None,
        help="Optional path to a CSV file where global metrics will be saved",
    )
    args = parser.parse_args()

    model_list = (
        [args.model]
        if args.model != "all"
        else ["random_forest", "xgboost", "catboost", "knn"]
    )

    for name in model_list:
        metrics_file = None
        if args.log_metrics:
            base, ext = os.path.splitext(args.log_metrics)
            metrics_file = f"{base}_{name}{ext or '.csv'}"
        run_simulation(name, num_rounds=args.rounds, metrics_file=metrics_file)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
