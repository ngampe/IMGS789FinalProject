from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data_loader import prepare_ecg5000_for_anomaly_detection
from model import LSTMAutoencoder


def compute_reconstruction_errors(model: torch.nn.Module, data: torch.Tensor, device: str = "cpu") -> np.ndarray:
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(data.shape[0]):
            x = data[i:i+1].to(device)  # shape [1, seq_len, 1]
            recon = model(x)
            err = torch.mean((x - recon) ** 2).item()
            errors.append(err)

    return np.array(errors, dtype=np.float32)


def fixed_threshold(train_errors: np.ndarray, k: float = 3.0) -> float:
    return float(train_errors.mean() + k * train_errors.std())


def percentile_threshold(train_errors: np.ndarray, q: float = 95.0) -> float:
    return float(np.percentile(train_errors, q))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return p, r, f1, cm


def save_score_plot(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    out_path: str,
    title: str,
):
    Path("figures").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Anomaly score")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")

    anomaly_idx = np.where(y_true == 1)[0]
    plt.scatter(anomaly_idx, scores[anomaly_idx], color="orange", s=8, label="True anomalies")

    plt.xlabel("Test sample index")
    plt.ylabel("Reconstruction error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_tensor, test_tensor, test_labels = prepare_ecg5000_for_anomaly_detection()

    model = LSTMAutoencoder()
    model.load_state_dict(torch.load("results/lstm_autoencoder.pt", map_location=device))
    model.to(device)

    print("Computing reconstruction errors...")
    train_errors = compute_reconstruction_errors(model, train_tensor, device=device)
    test_errors = compute_reconstruction_errors(model, test_tensor, device=device)

    # Baseline 1: fixed threshold
    thr_fixed = fixed_threshold(train_errors, k=3.0)
    pred_fixed = (test_errors > thr_fixed).astype(np.int64)
    p_fix, r_fix, f1_fix, cm_fix = evaluate_predictions(test_labels, pred_fixed)

    # Baseline 2: percentile threshold
    thr_pct = percentile_threshold(train_errors, q=95.0)
    pred_pct = (test_errors > thr_pct).astype(np.int64)
    p_pct, r_pct, f1_pct, cm_pct = evaluate_predictions(test_labels, pred_pct)

    print("\n=== Fixed Threshold (mean + 3*std) ===")
    print("Threshold:", thr_fixed)
    print(f"Precision: {p_fix:.4f}")
    print(f"Recall:    {r_fix:.4f}")
    print(f"F1-score:  {f1_fix:.4f}")
    print("Confusion matrix:\n", cm_fix)

    print("\n=== Percentile Threshold (95th) ===")
    print("Threshold:", thr_pct)
    print(f"Precision: {p_pct:.4f}")
    print(f"Recall:    {r_pct:.4f}")
    print(f"F1-score:  {f1_pct:.4f}")
    print("Confusion matrix:\n", cm_pct)

    Path("results").mkdir(exist_ok=True)

    np.save("results/train_errors.npy", train_errors)
    np.save("results/test_errors.npy", test_errors)

    save_score_plot(
        test_errors,
        test_labels,
        thr_fixed,
        "figures/test_scores_fixed_threshold.png",
        "ECG5000 Anomaly Scores with Fixed Threshold",
    )

    save_score_plot(
        test_errors,
        test_labels,
        thr_pct,
        "figures/test_scores_percentile_threshold.png",
        "ECG5000 Anomaly Scores with Percentile Threshold",
    )

    print("\nSaved:")
    print("- results/train_errors.npy")
    print("- results/test_errors.npy")
    print("- figures/test_scores_fixed_threshold.png")
    print("- figures/test_scores_percentile_threshold.png")
