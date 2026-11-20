"""AeroPulse export generator.

Exports precomputed, verified NASA C-MAPSS artifacts into compact, deterministic JSON
structures consumed directly by the React frontend and OpenAPI schemas.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

from backend.data_loader import CMAPSSDataPipeline
from backend.metrics import compute_all_metrics
from backend.models import TurbofanLSTM


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def export_all_artifacts(
    pipeline: CMAPSSDataPipeline,
    lstm_model: TurbofanLSTM,
    results: Dict[str, Any],
    output_dir: str = "frontend/public/data",
    device_name: str = "cpu",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    engines_dir = out_path / "engines" / "FD001" / "test"
    engines_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name)
    lstm_model.to(device)
    lstm_model.eval()

    print(f"[AeroPulse Export] Exporting 100 test engines to {engines_dir}...")

    # Load raw test DataFrame and official RUL series
    test_df = pipeline.raw_test_df
    rul_series = pipeline.test_rul_series
    assert test_df is not None and rul_series is not None

    test_units = sorted([int(u) for u in test_df["unit_id"].unique()])
    all_engine_summaries = []

    for u_int in test_units:
        engine_df = test_df[test_df["unit_id"] == u_int].sort_values("cycle").reset_index(drop=True)
        observed_cycles = int(engine_df["cycle"].max())
        true_rul_final = float(rul_series.iloc[u_int - 1])

        # Generate causal predictions across all observed cycles
        X_seq, valid_cycles = pipeline.get_test_engine_full_sequence(u_int)
        with torch.no_grad():
            tensor_seq = torch.tensor(X_seq, dtype=torch.float32, device=device)
            causal_preds = lstm_model(tensor_seq).cpu().numpy().tolist()

        predicted_rul_final = round(float(causal_preds[-1]), 2)
        err = round(predicted_rul_final - true_rul_final, 2)

        # Retrospective evaluation truth: true_rul_final + (observed_cycles - c)
        # Clearly marked as retrospective evaluation aid
        retrospective_true = [
            round(true_rul_final + (observed_cycles - c), 2) for c in valid_cycles
        ]

        # Key sensor trajectories (subsampled/rounded for efficient frontend transfer)
        sensor_series = {}
        for s_col in pipeline.selected_features:
            sensor_series[s_col] = [round(float(v), 4) for v in engine_df[s_col].values]

        engine_data = {
            "unit_id": u_int,
            "subset": "FD001",
            "split": "test",
            "observed_cycles": observed_cycles,
            "true_rul_final": round(true_rul_final, 2),
            "model_id": "model_lstm_v1",
            "predicted_rul_final": predicted_rul_final,
            "prediction_error": err,
            "prediction_status": "ready",
            "source": "NASA C-MAPSS simulator",
            "cycles": [int(c) for c in valid_cycles],
            "predictions_by_cycle": [round(float(p), 2) for p in causal_preds],
            "retrospective_true_rul_by_cycle": retrospective_true,
            "sensors": sensor_series,
        }

        file_name = f"{u_int:03d}.json"
        with open(engines_dir / file_name, "w") as f:
            json.dump(engine_data, f, indent=2, cls=NumpyEncoder)

        all_engine_summaries.append({
            "unit_id": u_int,
            "observed_cycles": observed_cycles,
            "predicted_rul": predicted_rul_final,
            "true_rul": round(true_rul_final, 2),
            "error": err,
            "status": "ready",
        })

    # Export Manifest
    manifest = {
        "dataset": "NASA C-MAPSS FD001",
        "provenance": {
            "source": "NASA Open Data Portal / PHM08 Data Set Repository",
            "citation": "A. Saxena, K. Goebel, D. Simon, and N. Eklund, 'Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation', PHM08, 2008.",
            "operational_conditions": "1 (Sea Level)",
            "fault_modes": "1 (HPC Degradation)",
            "train_engines": 100,
            "test_engines": 100,
            "total_sensor_channels": 21,
            "selected_sensor_channels": len(pipeline.selected_features),
            "selected_features": pipeline.selected_features,
            "dropped_features": pipeline.dropped_features,
            "variance_threshold": 1e-12,
        },
        "splits": {
            "train_engines_count": len(pipeline.train_units),
            "val_engines_count": len(pipeline.val_units),
            "test_engines_count": 100,
            "train_engine_ids": [int(x) for x in pipeline.train_units],
            "val_engine_ids": [int(x) for x in pipeline.val_units],
            "test_engine_ids": list(range(1, 101)),
        },
        "methodology": {
            "window_size": 30,
            "target_construction": "Run-to-failure linear RUL with 125-cycle cap for training",
            "padding_policy": "Causal left-edge repeat padding for short sequence heads",
            "leakage_safeguards": "Strict engine-group holdout; scalers & variance selection fitted exclusively on training engines",
            "loss_function": "Mean Squared Error",
            "official_evaluation": "Exactly one prediction per test engine from last observation window compared with official RUL_FD001.txt",
        },
        "models": results["models"],
        "fleet_index": all_engine_summaries,
        "generated_at": results["timestamp"],
        "app_status": "ready",
    }

    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    print(f"[AeroPulse Export] Saved manifest to {out_path / 'manifest.json'}")

    # Export Run details
    run_export = {
        "run_id": "model_lstm_v1",
        "model_name": "Causal 2-Layer LSTM",
        "dataset": "NASA C-MAPSS FD001",
        "training_history": results["training_history"],
        "metrics": results["models"]["lstm"]["official_test_uncapped"],
        "test_scatter": [
            {
                "unit_id": int(item["unit_id"]),
                "true_rul": item["true_rul"],
                "predicted_rul": item["predicted_rul"],
                "error": item["error"],
            }
            for item in all_engine_summaries
        ],
    }
    with open(runs_dir / "model_lstm_v1.json", "w") as f:
        json.dump(run_export, f, indent=2, cls=NumpyEncoder)

    # Also duplicate into artifacts/ so both backend and frontend have copies
    art_manifest = Path("artifacts") / "manifest.json"
    with open(art_manifest, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    print(f"[AeroPulse Export] Export complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Retrain models before exporting")
    args = parser.parse_args()

    pipeline = CMAPSSDataPipeline(data_dir="data/raw").fit_and_prepare()

    if args.retrain or not (Path("artifacts/metrics/evaluation_summary.json").exists() and Path("artifacts/checkpoints/best_lstm.pt").exists()):
        from backend.train import run_training
        results, pipeline, model = run_training(epochs=35, batch_size=128, lr=1e-3, seed=42)
        export_all_artifacts(pipeline, model, results)
    else:
        print("[AeroPulse Export] Loading existing checkpoint and evaluation summary...")
        with open("artifacts/metrics/evaluation_summary.json", "r") as f:
            results = json.load(f)
        chk = torch.load("artifacts/checkpoints/best_lstm.pt", map_location="cpu", weights_only=False)
        model = TurbofanLSTM(
            input_dim=chk["input_dim"],
            hidden_dim=chk.get("hidden_dim", 64),
            num_layers=chk.get("num_layers", 2),
            dropout=chk.get("dropout", 0.2),
        )
        model.load_state_dict(chk["model_state_dict"])
        export_all_artifacts(pipeline, model, results, device_name="cpu")
