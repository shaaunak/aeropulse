"""Multi-seed reproducibility and benchmarking experiment runner for AeroPulse.

Runs N independently seeded experiments for LSTM and GRU architectures,
logging validation metrics, official test performance, mean +- std deviations,
and producing publication-ready diagnostic charts.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.data_loader import CMAPSSDataPipeline
from backend.metrics import compute_all_metrics
from backend.models import TurbofanLSTM, TurbofanGRU, TabularRidgeBaseline, PersistenceBaseline
from backend.train import set_seed, get_device, train_epoch, evaluate_torch_model


def run_multi_seed_benchmark(
    seeds: List[int] = [42, 123, 456, 789, 1011],
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    device_name: str = "auto",
    output_dir: str = "artifacts/experiments",
) -> Dict[str, Any]:
    device = get_device(device_name)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[AeroPulse Benchmark] Running {len(seeds)} seeds: {seeds} on device: {device}")

    lstm_results = []
    gru_results = []

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        set_seed(seed)

        # 1. Pipeline preparation with strict train-only split & scaling
        pipeline = CMAPSSDataPipeline(data_dir="data/raw", window_size=30, rul_cap=125.0, seed=seed).fit_and_prepare()
        (X_tr, y_tr_cap, y_tr_uncap, _), (X_va, y_va_cap, y_va_uncap, _) = pipeline.get_train_and_val_windows()
        X_test_last, y_test_true, _ = pipeline.get_official_test_windows()

        input_dim = len(pipeline.selected_features)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr_cap, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
        )
        criterion = nn.MSELoss()

        # 2. Train LSTM
        lstm = TurbofanLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(lstm.parameters(), lr=lr, weight_decay=1e-4)

        best_lstm_val = float("inf")
        best_lstm_w = None
        for ep in range(1, epochs + 1):
            train_epoch(lstm, train_loader, optimizer, criterion, device)
            _, val_m = evaluate_torch_model(lstm, X_va, y_va_cap, device)
            if val_m["rmse"] < best_lstm_val:
                best_lstm_val = val_m["rmse"]
                best_lstm_w = {k: v.cpu().clone() for k, v in lstm.state_dict().items()}

        lstm.load_state_dict(best_lstm_w)
        lstm_preds, lstm_test_m = evaluate_torch_model(lstm, X_test_last, y_test_true, device)
        _, lstm_test_m_cap = evaluate_torch_model(lstm, X_test_last, np.minimum(y_test_true, 125.0), device)

        lstm_results.append({
            "seed": seed,
            "val_rmse": round(best_lstm_val, 4),
            "test_rmse": round(lstm_test_m["rmse"], 4),
            "test_mae": round(lstm_test_m["mae"], 4),
            "nasa_score": round(lstm_test_m["nasa_score"], 2),
            "test_rmse_cap": round(lstm_test_m_cap["rmse"], 4),
            "preds": [round(float(p), 2) for p in lstm_preds],
        })
        print(f"  LSTM | Val RMSE: {best_lstm_val:.2f} | Test RMSE: {lstm_test_m['rmse']:.2f} | MAE: {lstm_test_m['mae']:.2f} | NASA: {lstm_test_m['nasa_score']:.1f}")

        # 3. Train GRU
        gru = TurbofanGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
        gru_opt = torch.optim.AdamW(gru.parameters(), lr=lr, weight_decay=1e-4)

        best_gru_val = float("inf")
        best_gru_w = None
        for ep in range(1, epochs + 1):
            train_epoch(gru, train_loader, gru_opt, criterion, device)
            _, val_m = evaluate_torch_model(gru, X_va, y_va_cap, device)
            if val_m["rmse"] < best_gru_val:
                best_gru_val = val_m["rmse"]
                best_gru_w = {k: v.cpu().clone() for k, v in gru.state_dict().items()}

        gru.load_state_dict(best_gru_w)
        gru_preds, gru_test_m = evaluate_torch_model(gru, X_test_last, y_test_true, device)
        _, gru_test_m_cap = evaluate_torch_model(gru, X_test_last, np.minimum(y_test_true, 125.0), device)

        gru_results.append({
            "seed": seed,
            "val_rmse": round(best_gru_val, 4),
            "test_rmse": round(gru_test_m["rmse"], 4),
            "test_mae": round(gru_test_m["mae"], 4),
            "nasa_score": round(gru_test_m["nasa_score"], 2),
            "test_rmse_cap": round(gru_test_m_cap["rmse"], 4),
            "preds": [round(float(p), 2) for p in gru_preds],
        })
        print(f"  GRU  | Val RMSE: {best_gru_val:.2f} | Test RMSE: {gru_test_m['rmse']:.2f} | MAE: {gru_test_m['mae']:.2f} | NASA: {gru_test_m['nasa_score']:.1f}")

    def compute_stats(arr):
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    summary = {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "epochs": epochs,
        "lstm": {
            "val_rmse": compute_stats([r["val_rmse"] for r in lstm_results]),
            "test_rmse": compute_stats([r["test_rmse"] for r in lstm_results]),
            "test_mae": compute_stats([r["test_mae"] for r in lstm_results]),
            "nasa_score": compute_stats([r["nasa_score"] for r in lstm_results]),
            "test_rmse_cap": compute_stats([r["test_rmse_cap"] for r in lstm_results]),
            "runs": lstm_results,
        },
        "gru": {
            "val_rmse": compute_stats([r["val_rmse"] for r in gru_results]),
            "test_rmse": compute_stats([r["test_rmse"] for r in gru_results]),
            "test_mae": compute_stats([r["test_mae"] for r in gru_results]),
            "nasa_score": compute_stats([r["nasa_score"] for r in gru_results]),
            "test_rmse_cap": compute_stats([r["test_rmse_cap"] for r in gru_results]),
            "runs": gru_results,
        },
        "ground_truth_test_rul": [round(float(y), 2) for y in y_test_true],
    }

    out_file = out_path / "multi_seed_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[AeroPulse Benchmark] Completed! Results saved to {out_file}")
    print(f"LSTM Test RMSE: {summary['lstm']['test_rmse']['mean']} ± {summary['lstm']['test_rmse']['std']}")
    print(f"GRU  Test RMSE: {summary['gru']['test_rmse']['mean']} ± {summary['gru']['test_rmse']['std']}")
    print(f"LSTM NASA Score: {summary['lstm']['nasa_score']['mean']} ± {summary['lstm']['nasa_score']['std']}")
    print(f"GRU  NASA Score: {summary['gru']['nasa_score']['mean']} ± {summary['gru']['nasa_score']['std']}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_multi_seed_benchmark(epochs=args.epochs, device_name=args.device)
