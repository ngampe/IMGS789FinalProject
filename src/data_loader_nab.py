from __future__ import annotations

from pathlib import Path
from typing import Tuple

import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_nab_series(
    csv_path: str | Path,
    labels_json_path: str | Path = "data/NAB/labels/combined_windows.json",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load one NAB CSV file and convert anomaly windows into point labels.

    Returns:
        df: DataFrame with timestamp and value columns
        point_labels: numpy array of shape [N], 0=normal, 1=anomaly
    """
    csv_path = Path(csv_path)
    labels_json_path = Path(labels_json_path)

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    point_labels = np.zeros(len(df), dtype=np.int64)

    with open(labels_json_path, "r") as f:
        labels = json.load(f)

    # NAB label keys are relative paths from the data folder
    key = str(csv_path).replace("\\", "/")
    key = key.split("data/NAB/data/")[-1]

    if key not in labels:
        raise ValueError(f"Could not find labels for key: {key}")

    windows = labels[key]

    for start_str, end_str in windows:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        point_labels[mask.to_numpy()] = 1

    return df, point_labels


def create_windows_from_series(
    values: np.ndarray,
    point_labels: np.ndarray,
    window_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a univariate series into sliding windows.

    A window is labeled anomalous if any point inside the window is anomalous.
    """
    X = []
    y = []

    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        window_labels = point_labels[i:i + window_size]

        X.append(window)
        y.append(1 if np.any(window_labels == 1) else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def prepare_nab_for_anomaly_detection(
    csv_path: str | Path = "data/NAB/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    labels_json_path: str | Path = "data/NAB/labels/combined_windows.json",
    window_size: int = 32,
    normalize: bool = True,
):
    """
    Prepare NAB data for the same anomaly detection pipeline used for ECG5000.

    Train on normal windows only.
    Test on all windows.
    """
    df, point_labels = load_nab_series(csv_path, labels_json_path)
    values = df["value"].to_numpy(dtype=np.float32)

    X, y = create_windows_from_series(values, point_labels, window_size=window_size)

    X_train = X[y == 0]
    X_test = X
    y_test = y

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

    return train_tensor, test_tensor, y_test


if __name__ == "__main__":
    train_tensor, test_tensor, test_labels = prepare_nab_for_anomaly_detection()

    print("Train tensor shape:", train_tensor.shape)
    print("Test tensor shape:", test_tensor.shape)
    print("Test labels shape:", test_labels.shape)
    print("Normal training windows:", train_tensor.shape[0])
    print("Anomalous test windows:", int(test_labels.sum()))