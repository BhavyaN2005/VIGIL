"""
VIGIL: FlowAutoencoder Deep Bottleneck Architecture
Trained strictly on Benign traffic for Unsupervised Zero-Day Anomaly Detection.
Reconstruction error exceeding adaptive threshold tau flags novel attacks.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class FlowAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 78,
        hidden_dims: List[int] = None,
    ):
        super(FlowAutoencoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [48, 24, 12]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Encoder: 78 -> 48 -> 24 -> 12
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.1))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: 12 -> 24 -> 48 -> 78
        decoder_layers = []
        rev_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        for h_dim in rev_dims[:-1]:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.1))
            prev_dim = h_dim
        # Final output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck: returns reconstructed feature vector."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-sample Mean Squared Error (MSE) between input and reconstruction.
        Returns 1D tensor of shape (batch_size,).
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            return mse

    def predict_anomaly(
        self, x: torch.Tensor, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates input against anomaly threshold.
        Returns:
            (is_anomaly_bool_array, reconstruction_errors_array)
        """
        errors = self.compute_reconstruction_error(x).cpu().numpy()
        is_anomaly = errors > threshold
        return is_anomaly, errors
