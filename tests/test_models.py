"""Unit tests for AeroPulse PyTorch models and baselines."""

import numpy as np
import torch
from backend.models import (
    TurbofanLSTM,
    TurbofanGRU,
    TabularRidgeBaseline,
    PersistenceBaseline,
    extract_tabular_features,
)


def test_lstm_forward_and_shapes():
    batch_size, seq_len, input_dim = 16, 30, 14
    model = TurbofanLSTM(input_dim=input_dim, hidden_dim=32, num_layers=2)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size,)
    assert (out >= 0.0).all(), "Predictions must be non-negative cycles"


def test_gru_forward_and_shapes():
    batch_size, seq_len, input_dim = 8, 30, 14
    model = TurbofanGRU(input_dim=input_dim, hidden_dim=32, num_layers=2)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size,)
    assert (out >= 0.0).all()


def test_tabular_baselines():
    N, seq_len, D = 50, 30, 14
    X = np.random.randn(N, seq_len, D).astype(np.float32)
    y = np.random.uniform(10, 125, size=(N,)).astype(np.float32)

    ridge = TabularRidgeBaseline().fit(X, y)
    preds = ridge.predict(X)
    assert preds.shape == (N,)
    assert (preds >= 0.0).all()

    pers = PersistenceBaseline().fit(y)
    p_preds = pers.predict(X)
    assert p_preds.shape == (N,)
    assert np.allclose(p_preds, np.median(y))
