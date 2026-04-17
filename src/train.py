from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import prepare_ecg5000_for_anomaly_detection
from model import LSTMAutoencoder


def train_model(
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dim: int = 32,
    latent_dim: int = 16,
    device: str = "cpu",
):
    train_tensor, test_tensor, test_labels = prepare_ecg5000_for_anomaly_detection()

    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    return model, train_tensor, test_tensor, test_labels, losses


def save_loss_plot(losses, out_path: str = "figures/loss_curve.png"):
    Path("figures").mkdir(exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("LSTM Autoencoder Training Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model, train_tensor, test_tensor, test_labels, losses = train_model(device=device)
    save_loss_plot(losses)

    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "results/lstm_autoencoder.pt")

    print("Saved model to results/lstm_autoencoder.pt")
    print("Saved loss curve to figures/loss_curve.png")
