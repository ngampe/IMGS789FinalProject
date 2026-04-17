from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, latent_dim: int = 16, num_layers: int = 1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Encode sequence
        _, (hidden, _) = self.encoder(x)
        last_hidden = hidden[-1]  # [batch, hidden_dim]

        # Bottleneck
        latent = self.to_latent(last_hidden)          # [batch, latent_dim]
        decoded_seed = self.from_latent(latent)       # [batch, hidden_dim]

        # Repeat across sequence length for decoding
        repeated = decoded_seed.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden_dim]

        decoded_seq, _ = self.decoder(repeated)
        out = self.output_layer(decoded_seq)  # [batch, seq_len, input_dim]

        return out


if __name__ == "__main__":
    model = LSTMAutoencoder()
    x = torch.randn(8, 140, 1)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
