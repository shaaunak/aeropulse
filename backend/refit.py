"""Optional Final-Refit Pipeline on All 100 Development Engines for AeroPulse.

Scientific protocol:
1. Architecture, hyperparameters, and optimal stopping epochs are selected
   strictly on the 80/20 development validation split.
2. In this final-refit stage, the chosen architecture is trained on all 100
   development engines for exactly the optimal number of epochs (best_epoch)
   determined during validation.
3. The official 100-engine test set is evaluated ONLY at the very end to
   assess out-of-sample generalization.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.data_loader import CMAPSSDataPipeline
from backend.metrics import compute_all_metrics
from backend.models import TurbofanLSTM
from backend.train import set_seed, get_device, train_epoch, evaluate_torch_model


def run_final_refit(
    epochs: int = 6,  # Best epoch identified on validation split
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    seed: int = 42,
    device_name: str = "auto",
    output_dir: str = "artifacts/refit",
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device(device_name)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[AeroPulse Refit] Training on ALL 100 development engines for {epochs} epochs (determined via validation)...")

    pipeline = CMAPSSDataPipeline(data_dir="data/raw", window_size=30, rul_cap=125.0, seed=seed).fit_and_prepare(refit_all_100=True)
    X_all, y_all_cap, y_all_uncap, _ = pipeline.get_full_development_windows()
    X_test_last, y_test_true, _ = pipeline.get_official_test_windows()

    input_dim = len(pipeline.selected_features)
    train_dataset = TensorDataset(torch.tensor(X_all, dtype=torch.float32), torch.tensor(y_all_cap, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TurbofanLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Refit Epoch {ep:02d}/{epochs:02d} | Train MSE: {loss:.2f}")

    preds, test_metrics = evaluate_torch_model(model, X_test_last, y_test_true, device)
    _, test_metrics_cap = evaluate_torch_model(model, X_test_last, np.minimum(y_test_true, 125.0), device)

    print(f"\n[AeroPulse Refit Evaluation on Official 100 Test Engines]:")
    print(f"  Test RMSE (uncapped): {test_metrics['rmse']:.4f}")
    print(f"  Test MAE (uncapped):  {test_metrics['mae']:.4f}")
    print(f"  NASA Asymmetric Score: {test_metrics['nasa_score']:.2f}")
    print(f"  Test RMSE (capped):   {test_metrics_cap['rmse']:.4f}")

    summary = {
        "description": "Final refit on 100 development engines using validation-selected epoch count",
        "fixed_epochs": epochs,
        "seed": seed,
        "train_engines_count": 100,
        "test_engines_count": 100,
        "official_test_uncapped": test_metrics,
        "official_test_capped": test_metrics_cap,
    }

    out_file = out_path / "refit_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_final_refit(epochs=args.epochs, seed=args.seed, device_name=args.device)
