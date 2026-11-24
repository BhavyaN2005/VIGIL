"""
VIGIL: FlowLSTM Neural Network Architecture
Captures temporal flow correlations and inter-packet sequence statistics.
"""

from typing import Tuple

import torch
import torch.nn as nn


class FlowLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 78,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.2,
    ):
        super(FlowLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # LSTM Layer (batch_first=True)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Batch Normalization & Classification Head
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        if x.dim() == 2:
            # Reshape (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        # Take the last sequence output
        last_hidden = lstm_out[:, -1, :]

        # Normalize and project
        normed = self.bn(last_hidden)
        dense = self.drop(self.relu(self.fc1(normed)))
        logits = self.fc_out(dense)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Softmax probability distribution."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)
