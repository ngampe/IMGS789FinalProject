from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data_loader import prepare_ecg5000_for_anomaly_detection


def percentile_threshold(train_scores: np.ndarray, q: float = 90.0) -> float:
    return float(np.percentile(train_scores, q))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return p, r, f1, cm


def save_score_plot(scores, y_true, threshold, out_path, title):
    Path("figures").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Anomaly score")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")

    anomaly_idx = np.where(y_true == 1)[0]
    plt.scatter(anomaly_idx, scores[anomaly_idx], color="orange", s=8, label="True anomalies")

    plt.xlabel("Test sample index")
    plt.ylabel("Isolation Forest score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    train_tensor, test_tensor, test_labels = prepare_ecg5000_for_anomaly_detection()

    X_train = train_tensor.squeeze(-1).numpy()
    X_test = test_tensor.squeeze(-1).numpy()

    clf = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )
    clf.fit(X_train)

    # sklearn's decision_function: higher is more normal
    train_scores = -clf.decision_function(X_train)
    test_scores = -clf.decision_function(X_test)

    thr_90 = percentile_threshold(train_scores, q=90.0)
    pred_90 = (test_scores > thr_90).astype(np.int64)
    p_90, r_90, f1_90, cm_90 = evaluate_predictions(test_labels, pred_90)

    thr_95 = percentile_threshold(train_scores, q=95.0)
    pred_95 = (test_scores > thr_95).astype(np.int64)
    p_95, r_95, f1_95, cm_95 = evaluate_predictions(test_labels, pred_95)

    print("\n=== Isolation Forest Baseline ===")
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
    with open("results/isoforest_metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Threshold", "Precision", "Recall", "F1"])
        writer.writerow(["IsoForestPercentile90", thr_90, p_90, r_90, f1_90])
        writer.writerow(["IsoForestPercentile95", thr_95, p_95, r_95, f1_95])

    save_score_plot(
        test_scores,
        test_labels,
        thr_90,
        "figures/isoforest_scores_percentile90.png",
        "ECG5000 Isolation Forest with 90th Percentile Threshold",
    )

    print("\nSaved:")
    print("- results/isoforest_metrics_summary.csv")
    print("- figures/isoforest_scores_percentile90.png")