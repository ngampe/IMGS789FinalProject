from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data_loader import prepare_ecg5000_for_anomaly_detection


def fixed_threshold(train_scores: np.ndarray, k: float = 3.0) -> float:
    return float(train_scores.mean() + k * train_scores.std())


def percentile_threshold(train_scores: np.ndarray, q: float = 95.0) -> float:
    return float(np.percentile(train_scores, q))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return p, r, f1, cm


def compute_zscore_sequence_scores(train_tensor, test_tensor):
    """
    train_tensor: [N_train, seq_len, 1]
    test_tensor: [N_test, seq_len, 1]
    Returns anomaly scores for train and test based on distance from normal prototype.
    """
    X_train = train_tensor.squeeze(-1).numpy()  # [N_train, seq_len]
    X_test = test_tensor.squeeze(-1).numpy()    # [N_test, seq_len]

    mean_seq = X_train.mean(axis=0)
    std_seq = X_train.std(axis=0) + 1e-8

    train_z = (X_train - mean_seq) / std_seq
    test_z = (X_test - mean_seq) / std_seq

    # mean absolute z-score per sequence
    train_scores = np.mean(np.abs(train_z), axis=1)
    test_scores = np.mean(np.abs(test_z), axis=1)

    return train_scores, test_scores


def save_score_plot(scores, y_true, threshold, out_path, title):
    Path("figures").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Anomaly score")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")

    anomaly_idx = np.where(y_true == 1)[0]
    plt.scatter(anomaly_idx, scores[anomaly_idx], color="orange", s=8, label="True anomalies")

    plt.xlabel("Test sample index")
    plt.ylabel("Sequence z-score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    train_tensor, test_tensor, test_labels = prepare_ecg5000_for_anomaly_detection()

    train_scores, test_scores = compute_zscore_sequence_scores(train_tensor, test_tensor)

    thr_fixed = fixed_threshold(train_scores, k=3.0)
    pred_fixed = (test_scores > thr_fixed).astype(np.int64)
    p_fix, r_fix, f1_fix, cm_fix = evaluate_predictions(test_labels, pred_fixed)

    thr_90 = percentile_threshold(train_scores, q=90.0)
    pred_90 = (test_scores > thr_90).astype(np.int64)
    p_90, r_90, f1_90, cm_90 = evaluate_predictions(test_labels, pred_90)

    thr_95 = percentile_threshold(train_scores, q=95.0)
    pred_95 = (test_scores > thr_95).astype(np.int64)
    p_95, r_95, f1_95, cm_95 = evaluate_predictions(test_labels, pred_95)

    print("\n=== Z-Score Baseline ===")
    print("Fixed Threshold")
    print(f"Precision: {p_fix:.4f}")
    print(f"Recall:    {r_fix:.4f}")
    print(f"F1-score:  {f1_fix:.4f}")
    print("Confusion matrix:\n", cm_fix)

    print("\n90th Percentile")
    print(f"Precision: {p_90:.4f}")
    print(f"Recall:    {r_90:.4f}")
    print(f"F1-score:  {f1_90:.4f}")
    print("Confusion matrix:\n", cm_90)

    print("\n95th Percentile")
    print(f"Precision: {p_95:.4f}")
    print(f"Recall:    {r_95:.4f}")
    print(f"F1-score:  {f1_95:.4f}")
    print("Confusion matrix:\n", cm_95)

    Path("results").mkdir(exist_ok=True)
    with open("results/zscore_metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Threshold", "Precision", "Recall", "F1"])
        writer.writerow(["ZScoreFixed", thr_fixed, p_fix, r_fix, f1_fix])
        writer.writerow(["ZScorePercentile90", thr_90, p_90, r_90, f1_90])
        writer.writerow(["ZScorePercentile95", thr_95, p_95, r_95, f1_95])

    save_score_plot(
        test_scores,
        test_labels,
        thr_90,
        "figures/zscore_scores_percentile90.png",
        "ECG5000 Z-Score Baseline with 90th Percentile Threshold",
    )

    print("\nSaved:")
    print("- results/zscore_metrics_summary.csv")
    print("- figures/zscore_scores_percentile90.png")