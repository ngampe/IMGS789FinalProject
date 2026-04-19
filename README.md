## Topic
Time Series Anomaly Detection

## Goal
This project studies time series anomaly detection using an LSTM autoencoder baseline and evaluates whether adaptive thresholding improves detection performance relative to a fixed threshold.

## Repository Structure

```text
time_series_anomaly_project/
├── data/
│   ├── ECG5000/
│   └── NAB/
├── figures/
├── results/
├── report/
├── src/
│   ├── data_loader.py
│   ├── data_loader_nab.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_nab.py
│   ├── evaluate_nab.py
│   ├── baseline_zscore.py
│   ├── baseline_isoforest.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt

## Planned Method
- Baseline: LSTM Autoencoder
- Improvement: Adaptive thresholding on anomaly scores
- Metrics: Precision, Recall, F1-score

The project was evaluated on two datasets:
ECG5000 as the primary dataset
NAB (`ec2_cpu_utilization_5f5533`) as the secondary 

Two simple baselines were tested on ECG5000:
z-score baseline
Isolation Forest baseline
