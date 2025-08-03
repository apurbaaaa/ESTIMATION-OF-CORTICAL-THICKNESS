"""Data loading and preprocessing utilities for ADNI dataset.

This module loads multiple CSV files, merges them, performs cleaning,
feature/target extraction, scaling, and partitions the data for
federated learning simulations. The dataset is assumed to follow the
preprocessing steps performed in `Data_Cleaning_and_Extraction.ipynb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Target regions for cortical thickness estimation
TARGET_COLUMNS: List[str] = [
    "ST58TA",
    "ST117TA",
    "ST40TA",
    "ST99TA",
    "ST32TA",
    "ST91TA",
    "ST60TA",
    "ST119TA",
    "ST62TA",
    "ST121TA",
]


def load_client_data(
    data_dir: str | Path,
    test_size: float = 0.2,
    local_val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess the ADNI dataset.

    Parameters
    ----------
    data_dir: str or Path
        Directory containing the cleaned CSV files. All CSV files will be
        merged on the ``RID`` column. If a single ``cleaned_data.csv`` file
        exists it will be used directly.
    test_size: float
        Fraction of the full dataset reserved for the *central* test set.
    local_val_size: float
        Fraction of each client's data reserved for local evaluation.
    random_state: int
        Random seed used throughout the splitting operations.

    Returns
    -------
    client_partitions: list
        List containing four tuples ``(X_train, y_train, X_val, y_val)`` – one
        for each client.
    test_set: tuple
        The tuple ``(X_test, y_test)`` used for centralized evaluation on the
        server.
    """

    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Read and merge all CSV files on the 'RID' column
    df = pd.read_csv(csv_files[0])
    for csv_file in csv_files[1:]:
        df_other = pd.read_csv(csv_file)
        merge_cols = [c for c in ["RID"] if c in df.columns and c in df_other.columns]
        if not merge_cols:
            # Fallback to index-based concatenation if RID is missing
            df = pd.concat([df, df_other], axis=1)
        else:
            df = df.merge(df_other, on=merge_cols, how="inner")

    # Convert date of birth into separate numerical features if present
    if "PTDOB" in df.columns:
        df["PTDOB"] = pd.to_datetime(df["PTDOB"])
        df["year"] = df["PTDOB"].dt.year
        df["month"] = df["PTDOB"].dt.month
        df["day"] = df["PTDOB"].dt.day
        df.drop(columns=["PTDOB"], inplace=True)

    # Separate features and targets
    X = df.drop(columns=TARGET_COLUMNS + ["RID"], errors="ignore")
    y = df[TARGET_COLUMNS]

    # Remove obvious outliers using the IQR rule on selected biomarker columns
    outlier_cols = ["APVOLUME", "ABETA42", "TAU", "PTAU", "PLASMAPTAU181"]
    existing_cols = [c for c in outlier_cols if c in X.columns]
    if existing_cols:
        Q1 = X[existing_cols].quantile(0.25)
        Q3 = X[existing_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((X[existing_cols] < (Q1 - 1.5 * IQR)) | (X[existing_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        X = X.loc[mask]
        y = y.loc[mask]

    # Ensure only numeric columns are used as features
    X = X.select_dtypes(include=[np.number]).copy()

    # Min-Max scaling
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train/test split for a central server-side test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Partition the training data among the four clients
    X_parts = np.array_split(X_train, 4)
    y_parts = np.array_split(y_train, 4)

    client_partitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for X_part, y_part in zip(X_parts, y_parts):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_part, y_part, test_size=local_val_size, random_state=random_state
        )
        client_partitions.append(
            (
                X_tr.to_numpy(),
                y_tr.to_numpy(),
                X_val.to_numpy(),
                y_val.to_numpy(),
            )
        )

    return client_partitions, (X_test.to_numpy(), y_test.to_numpy())


__all__ = ["TARGET_COLUMNS", "load_client_data"]
