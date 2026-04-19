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

def evaluate_multiple_percentiles(train_errors, test_errors, test_labels, percentiles):
    rows = []
    for q in percentiles:
        thr = percentile_threshold(train_errors, q=q)
        pred = (test_errors > thr).astype(np.int64)
        p, r, f1, cm = evaluate_predictions(test_labels, pred)
        rows.append((q, thr, p, r, f1, cm))
    return rows

def rolling_threshold(scores: np.ndarray, window: int = 50, k: float = 2.0) -> np.ndarray:
    thresholds = np.zeros_like(scores)
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        window_scores = scores[start:i+1]
        thresholds[i] = window_scores.mean() + k * window_scores.std()
    return thresholds

def save_dynamic_score_plot(scores, y_true, thresholds, out_path, title):
    Path("figures").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Anomaly score")
    plt.plot(thresholds, color="red", linestyle="--", label="Rolling threshold")

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

    percentile_rows = evaluate_multiple_percentiles(
    train_errors, test_errors, test_labels, [90.0, 95.0, 97.0, 99.0])

    print("\n=== Percentile Sweep ===")
    for q, thr, p, r, f1, cm in percentile_rows:
        print(f"{q:.0f}th percentile | thr={thr:.4f} | P={p:.4f} | R={r:.4f} | F1={f1:.4f}")
    
    rolling_thr = rolling_threshold(test_errors, window=50, k=2.0)
    pred_roll = (test_errors > rolling_thr).astype(np.int64)
    p_roll, r_roll, f1_roll, cm_roll = evaluate_predictions(test_labels, pred_roll)

    save_dynamic_score_plot(
        test_errors,
        test_labels,
        rolling_thr,
        "figures/test_scores_rolling_threshold.png",
        "ECG5000 Anomaly Scores with Rolling Threshold",
    )

    print("\n=== Rolling Threshold ===")
    print(f"Precision: {p_roll:.4f}")
    print(f"Recall:    {r_roll:.4f}")
    print(f"F1-score:  {f1_roll:.4f}")
    print("Confusion matrix:\n", cm_roll)    

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

    import csv

    with open("results/metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Threshold", "Precision", "Recall", "F1"])
        writer.writerow(["Fixed", thr_fixed, p_fix, r_fix, f1_fix])
        writer.writerow(["Percentile95", thr_pct, p_pct, r_pct, f1_pct])
        writer.writerow(["Rolling", "dynamic", p_roll, r_roll, f1_roll])

        for q, thr, p, r, f1, _ in percentile_rows:
            writer.writerow([f"Percentile{int(q)}", thr, p, r, f1])
