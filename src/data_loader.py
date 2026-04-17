from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.io import arff
from sklearn.preprocessing import StandardScaler


def load_arff_file(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data, _ = arff.loadarff(str(path))
    data = np.array(data.tolist(), dtype=np.float32)

    X = data[:, :-1]
    y = data[:, -1]
    y = np.array([int(v.decode() if isinstance(v, bytes) else v) for v in y], dtype=np.int64)

    return X, y


def load_ecg5000(data_dir: str | Path = "data/ECG5000") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)

    train_path = data_dir / "ECG5000_TRAIN.arff"
    test_path = data_dir / "ECG5000_TEST.arff"

    X_train, y_train = load_arff_file(train_path)
    X_test, y_test = load_arff_file(test_path)

    return X_train, y_train, X_test, y_test


def prepare_ecg5000_for_anomaly_detection(
    data_dir: str | Path = "data/ECG5000",
    normalize: bool = True,
):
    X_train, y_train, X_test, y_test = load_ecg5000(data_dir)

    # class 1 = normal, classes 2-5 = anomalies
    train_mask = y_train == 1
    test_binary = (y_test != 1).astype(np.int64)

    X_train_normal = X_train[train_mask]

    if normalize:
        scaler = StandardScaler()
        X_train_normal = scaler.fit_transform(X_train_normal)
        X_test = scaler.transform(X_test)

    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32).unsqueeze(-1)
    test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

    return train_tensor, test_tensor, test_binary


if __name__ == "__main__":
    train_tensor, test_tensor, test_labels = prepare_ecg5000_for_anomaly_detection()

    print("Train tensor shape:", train_tensor.shape)
    print("Test tensor shape:", test_tensor.shape)
    print("Test labels shape:", test_labels.shape)
    print("Normal training samples:", train_tensor.shape[0])
    print("Test anomalies:", int(test_labels.sum()))
