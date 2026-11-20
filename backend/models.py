"""PyTorch model architectures and tabular baselines for AeroPulse prognostics."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


class TurbofanLSTM(nn.Module):
    """Causal sequence-to-one 2-layer LSTM regression network for RUL estimation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor of shape (batch_size,) representing predicted RUL in cycles
        """
        lstm_out, _ = self.lstm(x)
        endpoint_feat = lstm_out[:, -1, :]
        raw_pred = self.fc_head(endpoint_feat).squeeze(-1)
        # Using softplus ensures strictly smooth positive outputs and prevents dying zero-gradient deadlocks
        return nn.functional.softplus(raw_pred)


class TurbofanGRU(nn.Module):
    """Gated Recurrent Unit comparison model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        endpoint_feat = gru_out[:, -1, :]
        raw_pred = self.fc_head(endpoint_feat).squeeze(-1)
        return nn.functional.softplus(raw_pred)


def extract_tabular_features(X_windows: np.ndarray) -> np.ndarray:
    """Extract statistical summary features across sequence window for tabular baselines.

    Extracts last value, mean, standard deviation, min, max, and trend (last - first).
    Input: (N, seq_len, D) -> Output: (N, D * 6)
    """
    last_val = X_windows[:, -1, :]
    mean_val = np.mean(X_windows, axis=1)
    std_val = np.std(X_windows, axis=1)
    min_val = np.min(X_windows, axis=1)
    max_val = np.max(X_windows, axis=1)
    trend_val = X_windows[:, -1, :] - X_windows[:, 0, :]
    return np.hstack([last_val, mean_val, std_val, min_val, max_val, trend_val])


class TabularRidgeBaseline:
    """Rigorous tabular baseline using engineered window features and L2 regularization."""

    def __init__(self, alpha: float = 10.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X_windows: np.ndarray, y: np.ndarray):
        feats = extract_tabular_features(X_windows)
        self.model.fit(feats, y)
        return self

    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        feats = extract_tabular_features(X_windows)
        preds = self.model.predict(feats)
        return np.maximum(0.0, preds)


class PersistenceBaseline:
    """Constant/median baseline from training targets."""

    def __init__(self):
        self.constant_prediction = 100.0

    def fit(self, y_train: np.ndarray):
        self.constant_prediction = float(np.median(y_train))
        return self

    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X_windows),), fill_value=self.constant_prediction, dtype=np.float32)
