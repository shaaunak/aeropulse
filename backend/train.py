"""AeroPulse model training pipeline for NASA C-MAPSS FD001.

Reproducible, leakage-safe training of PyTorch LSTM, GRU, and Tabular Baselines.
"""

import argparse
import json
from pathlib import Path
import random
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.data_loader import CMAPSSDataPipeline
from backend.metrics import compute_all_metrics, compute_rmse, compute_mae, compute_nasa_score
from backend.models import TurbofanLSTM, TurbofanGRU, TabularRidgeBaseline, PersistenceBaseline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)

        optimizer.zero_grad()
        preds = model(X_b)
        loss = criterion(preds, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


def evaluate_torch_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_b = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
            p = model(X_b).cpu().numpy()
            preds_list.append(p)
    preds = np.concatenate(preds_list) if preds_list else np.array([], dtype=np.float32)
    metrics = compute_all_metrics(y, preds)
    return preds, metrics


def run_training(
    data_dir: str = "data/raw",
    artifacts_dir: str = "artifacts",
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    patience: int = 10,
    seed: int = 42,
    device_name: str = "auto",
) -> Dict[str, Any]:
    start_time = time.time()
    set_seed(seed)
    device = get_device(device_name)
    print(f"[AeroPulse] Device: {device} | Seed: {seed} | Epochs: {epochs}")

    # Prepare data pipeline
    pipeline = CMAPSSDataPipeline(data_dir=data_dir, window_size=30, rul_cap=125.0, seed=seed).fit_and_prepare()
    (X_tr, y_tr_cap, y_tr_uncap, tr_ends), (X_va, y_va_cap, y_va_uncap, va_ends) = pipeline.get_train_and_val_windows()
    X_test_last, y_test_true, test_units = pipeline.get_official_test_windows()

    input_dim = len(pipeline.selected_features)
    print(f"[AeroPulse] Selected features ({input_dim}): {pipeline.selected_features}")
    print(f"[AeroPulse] Train samples: {len(X_tr)} (80 engines) | Val samples: {len(X_va)} (20 engines) | Test engines: {len(X_test_last)}")

    # Data loaders
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr_cap, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate PyTorch LSTM
    lstm_model = TurbofanLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = 0
    best_weights = None
    history = []

    print("[AeroPulse] Training LSTM...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(lstm_model, train_loader, optimizer, criterion, device)
        _, val_metrics_cap = evaluate_torch_model(lstm_model, X_va, y_va_cap, device)
        _, val_metrics_uncap = evaluate_torch_model(lstm_model, X_va, y_va_uncap, device)

        val_rmse = val_metrics_cap["rmse"]
        history.append({
            "epoch": epoch,
            "train_mse": round(train_loss, 4),
            "val_rmse_cap": val_metrics_cap["rmse"],
            "val_mae_cap": val_metrics_cap["mae"],
            "val_rmse_uncap": val_metrics_uncap["rmse"],
            "val_mae_uncap": val_metrics_uncap["mae"],
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.2f} | Val RMSE (capped): {val_rmse:.2f} | Best: {best_val_rmse:.2f} @ ep {best_epoch}")

        if epoch - best_epoch >= patience:
            print(f"[AeroPulse] Early stopping triggered at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    # Restore best checkpoint
    assert best_weights is not None
    lstm_model.load_state_dict(best_weights)
    lstm_model.to(device)

    # Save checkpoint
    chk_dir = Path(artifacts_dir) / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = chk_dir / "best_lstm.pt"
    torch.save({
        "model_state_dict": best_weights,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "selected_features": pipeline.selected_features,
        "scaler_mean": pipeline.scaler.mean_.tolist(),
        "scaler_scale": pipeline.scaler.scale_.tolist(),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
    }, checkpoint_path)
    print(f"[AeroPulse] Saved checkpoint to {checkpoint_path}")

    # Official Test Evaluation for LSTM (Uncapped Ground Truth)
    lstm_test_preds, lstm_test_metrics_uncap = evaluate_torch_model(lstm_model, X_test_last, y_test_true, device)
    # Also evaluate capped for reporting
    y_test_capped = np.minimum(y_test_true, 125.0)
    _, lstm_test_metrics_cap = evaluate_torch_model(lstm_model, X_test_last, y_test_capped, device)

    print(f"\n=======================================================")
    print(f"OFFICIAL NASA FD001 100-ENGINE TEST EVALUATION (LSTM):")
    print(f"  Test RMSE (official uncapped): {lstm_test_metrics_uncap['rmse']}")
    print(f"  Test MAE (official uncapped):  {lstm_test_metrics_uncap['mae']}")
    print(f"  NASA Asymmetric Score:         {lstm_test_metrics_uncap['nasa_score']}")
    print(f"  Test RMSE (capped at 125):     {lstm_test_metrics_cap['rmse']}")
    print(f"=======================================================\n")

    # Train GRU comparison model
    print("[AeroPulse] Training GRU comparison model...")
    gru_model = TurbofanGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    gru_optimizer = torch.optim.AdamW(gru_model.parameters(), lr=lr, weight_decay=1e-4)
    best_gru_val_rmse = float("inf")
    best_gru_weights = None

    for ep in range(1, min(epochs, 25) + 1):
        train_epoch(gru_model, train_loader, gru_optimizer, criterion, device)
        _, val_m = evaluate_torch_model(gru_model, X_va, y_va_cap, device)
        if val_m["rmse"] < best_gru_val_rmse:
            best_gru_val_rmse = val_m["rmse"]
            best_gru_weights = {k: v.cpu().clone() for k, v in gru_model.state_dict().items()}

    assert best_gru_weights is not None
    gru_model.load_state_dict(best_gru_weights)
    gru_test_preds, gru_test_metrics = evaluate_torch_model(gru_model, X_test_last, y_test_true, device)
    print(f"[AeroPulse] GRU Test RMSE (uncapped): {gru_test_metrics['rmse']} | NASA Score: {gru_test_metrics['nasa_score']}")

    # Train Tabular Ridge Baseline
    print("[AeroPulse] Training Tabular Ridge Baseline...")
    ridge = TabularRidgeBaseline(alpha=10.0).fit(X_tr, y_tr_cap)
    ridge_test_preds = ridge.predict(X_test_last)
    ridge_test_metrics = compute_all_metrics(y_test_true, ridge_test_preds)
    print(f"[AeroPulse] Ridge Test RMSE (uncapped): {ridge_test_metrics['rmse']} | NASA Score: {ridge_test_metrics['nasa_score']}")

    # Train Persistence / Median Baseline
    persistence = PersistenceBaseline().fit(y_tr_cap)
    pers_test_preds = persistence.predict(X_test_last)
    pers_test_metrics = compute_all_metrics(y_test_true, pers_test_preds)
    print(f"[AeroPulse] Persistence Test RMSE (uncapped): {pers_test_metrics['rmse']} | NASA Score: {pers_test_metrics['nasa_score']}")

    total_duration = time.time() - start_time

    results = {
        "dataset": "NASA C-MAPSS FD001",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(total_duration, 2),
        "seed": seed,
        "device": str(device),
        "pipeline": {
            "window_size": 30,
            "rul_cap": 125.0,
            "selected_features": pipeline.selected_features,
            "dropped_features": pipeline.dropped_features,
            "train_engines": len(pipeline.train_units),
            "val_engines": len(pipeline.val_units),
            "test_engines": len(test_units),
        },
        "models": {
            "lstm": {
                "name": "Causal 2-Layer LSTM",
                "id": "model_lstm_v1",
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "best_epoch": best_epoch,
                "best_val_rmse_cap": round(best_val_rmse, 4),
                "official_test_uncapped": lstm_test_metrics_uncap,
                "official_test_capped": lstm_test_metrics_cap,
                "predictions": [round(float(p), 2) for p in lstm_test_preds],
            },
            "gru": {
                "name": "Causal 2-Layer GRU",
                "id": "model_gru_v1",
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "best_val_rmse_cap": round(best_gru_val_rmse, 4),
                "official_test_uncapped": gru_test_metrics,
                "predictions": [round(float(p), 2) for p in gru_test_preds],
            },
            "ridge_tabular": {
                "name": "Engineered-Feature Ridge Regression",
                "id": "model_ridge_baseline",
                "official_test_uncapped": ridge_test_metrics,
                "predictions": [round(float(p), 2) for p in ridge_test_preds],
            },
            "persistence": {
                "name": "Median Target Baseline",
                "id": "model_persistence",
                "official_test_uncapped": pers_test_metrics,
                "predictions": [round(float(p), 2) for p in pers_test_preds],
            },
        },
        "ground_truth_test_rul": [round(float(y), 2) for y in y_test_true],
        "training_history": history,
    }

    # Save run summary
    metrics_path = Path(artifacts_dir) / "metrics" / "evaluation_summary.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[AeroPulse] Saved metrics summary to {metrics_path}")

    return results, pipeline, lstm_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AeroPulse Prognostics Models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_name=args.device,
    )
