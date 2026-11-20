"""FastAPI application for AeroPulse prognostics research interface.

Conforms strictly to specs/API_AND_EXPORT_SCHEMA.md.
"""

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch

from backend.data_loader import CMAPSSDataPipeline
from backend.models import TurbofanLSTM


class PredictionRequest(BaseModel):
    model_id: str = Field(..., description="ID of trained model, e.g. model_lstm_v1")
    subset: str = Field(default="FD001", description="Dataset subset")
    unit_id: int = Field(..., ge=1, le=100, description="Test engine unit ID (1..100)")
    observation_cycle: int = Field(..., ge=1, description="Observation cycle cutoff")


class PredictionResponse(BaseModel):
    unit_id: int
    subset: str
    split: str
    observed_through_cycle: int
    model_id: str
    predicted_rul_cycles: float
    true_rul_cycles: Optional[float] = None
    prediction_status: str
    source: str = "NASA C-MAPSS simulator"


class EngineSummary(BaseModel):
    unit_id: int
    subset: str
    split: str
    observed_cycles: int
    true_rul_final: Optional[float] = None
    predicted_rul_final: Optional[float] = None
    prediction_error: Optional[float] = None
    prediction_status: str


class HealthResponse(BaseModel):
    status: str
    service_available: bool
    loaded_model_id: Optional[str]
    dataset_state: str
    dataset_counts: Dict[str, int]
    device: str


# Global cache
app_state: Dict[str, Any] = {
    "model": None,
    "model_id": None,
    "pipeline": None,
    "manifest": None,
    "engine_cache": {},
    "device": "cpu",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load manifest and best checkpoint if available
    manifest_path = Path("frontend/public/data/manifest.json")
    if not manifest_path.exists():
        manifest_path = Path("artifacts/manifest.json")

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            app_state["manifest"] = json.load(f)

    checkpoint_path = Path("artifacts/checkpoints/best_lstm.pt")
    if checkpoint_path.exists():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        app_state["device"] = str(device)
        try:
            chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = TurbofanLSTM(
                input_dim=chk["input_dim"],
                hidden_dim=chk.get("hidden_dim", 64),
                num_layers=chk.get("num_layers", 2),
                dropout=chk.get("dropout", 0.2),
            )
            model.load_state_dict(chk["model_state_dict"])
            model.to(device)
            model.eval()
            app_state["model"] = model
            app_state["model_id"] = "model_lstm_v1"
            print("[AeroPulse API] Successfully loaded model_lstm_v1 into memory.")
        except Exception as e:
            print(f"[AeroPulse API] Warning: could not load model checkpoint: {e}")

    try:
        pipeline = CMAPSSDataPipeline(data_dir="data/raw").fit_and_prepare()
        app_state["pipeline"] = pipeline
    except Exception as e:
        print(f"[AeroPulse API] Pipeline initialization notice: {e}")

    yield


app = FastAPI(
    title="AeroPulse Prognostics API",
    description="Research-grade REST API for NASA C-MAPSS engine degradation and RUL inference.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
def get_health():
    model_loaded = app_state["model"] is not None
    pipeline_ready = app_state["pipeline"] is not None

    dataset_state = "ready" if pipeline_ready else "no_data"
    status_str = "ready" if (model_loaded and pipeline_ready) else "model_not_loaded"

    counts = {
        "train_engines": len(app_state["pipeline"].train_units) if pipeline_ready else 0,
        "val_engines": len(app_state["pipeline"].val_units) if pipeline_ready else 0,
        "test_engines": 100 if pipeline_ready else 0,
    }

    return HealthResponse(
        status=status_str,
        service_available=True,
        loaded_model_id=app_state["model_id"],
        dataset_state=dataset_state,
        dataset_counts=counts,
        device=app_state["device"],
    )


@app.get("/api/v1/datasets")
def get_datasets():
    return {
        "subsets": ["FD001"],
        "active_subset": "FD001",
        "description": "NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset",
        "conditions": "1 (Sea Level)",
        "fault_modes": "1 (HPC Degradation)",
        "train_trajectories": 100,
        "test_trajectories": 100,
        "citation": "A. Saxena et al., PHM08, Denver CO, 2008",
        "source": "NASA Open Data Portal",
    }


@app.get("/api/v1/engines", response_model=List[EngineSummary])
def get_engines(subset: str = "FD001", split: str = "test"):
    if subset != "FD001" or split != "test":
        raise HTTPException(status_code=400, detail="Only subset=FD001 and split=test are supported in this baseline")

    if app_state["manifest"] and "fleet_index" in app_state["manifest"]:
        fleet = app_state["manifest"]["fleet_index"]
        return [
            EngineSummary(
                unit_id=item["unit_id"],
                subset="FD001",
                split="test",
                observed_cycles=item["observed_cycles"],
                true_rul_final=item.get("true_rul"),
                predicted_rul_final=item.get("predicted_rul"),
                prediction_error=item.get("error"),
                prediction_status="ready" if app_state["model"] else "model_not_loaded",
            )
            for item in fleet
        ]

    # Fallback to pipeline inspection if manifest not yet written
    if app_state["pipeline"] and app_state["pipeline"].raw_test_df is not None:
        test_df = app_state["pipeline"].raw_test_df
        rul_series = app_state["pipeline"].test_rul_series
        summaries = []
        for u in sorted(test_df["unit_id"].unique()):
            u_df = test_df[test_df["unit_id"] == u]
            max_c = int(u_df["cycle"].max())
            true_r = float(rul_series.iloc[u - 1]) if rul_series is not None else None
            summaries.append(
                EngineSummary(
                    unit_id=int(u),
                    subset="FD001",
                    split="test",
                    observed_cycles=max_c,
                    true_rul_final=true_r,
                    prediction_status="model_not_loaded",
                )
            )
        return summaries

    raise HTTPException(status_code=503, detail="Dataset not ready")


@app.get("/api/v1/engines/{unit_id}")
def get_engine_detail(unit_id: int, subset: str = "FD001", split: str = "test"):
    if unit_id < 1 or unit_id > 100:
        raise HTTPException(status_code=404, detail=f"Engine {unit_id} not found (must be 1..100)")

    # Check precomputed JSON
    engine_file = Path(f"frontend/public/data/engines/FD001/test/{unit_id:03d}.json")
    if engine_file.exists():
        with open(engine_file, "r") as f:
            return json.load(f)

    # Dynamic fallback
    if app_state["pipeline"] and app_state["pipeline"].raw_test_df is not None:
        p = app_state["pipeline"]
        test_df = p.raw_test_df
        u_df = test_df[test_df["unit_id"] == unit_id].sort_values("cycle").reset_index(drop=True)
        if len(u_df) == 0:
            raise HTTPException(status_code=404, detail="Unit not found")
        max_c = int(u_df["cycle"].max())
        true_r = float(p.test_rul_series.iloc[unit_id - 1]) if p.test_rul_series is not None else None
        return {
            "unit_id": unit_id,
            "subset": subset,
            "split": split,
            "observed_cycles": max_c,
            "true_rul_final": true_r,
            "prediction_status": "model_not_loaded",
            "cycles": u_df["cycle"].tolist(),
        }

    raise HTTPException(status_code=503, detail="Engine data not loaded")


@app.get("/api/v1/engines/{unit_id}/history")
def get_engine_history(unit_id: int, model_id: Optional[str] = None):
    detail = get_engine_detail(unit_id)
    return {
        "unit_id": unit_id,
        "model_id": model_id or app_state["model_id"],
        "cycles": detail.get("cycles", []),
        "predictions_by_cycle": detail.get("predictions_by_cycle", []),
        "retrospective_true_rul_by_cycle": detail.get("retrospective_true_rul_by_cycle", []),
    }


@app.get("/api/v1/experiments")
def get_experiments():
    if app_state["manifest"] and "models" in app_state["manifest"]:
        return {
            "dataset": app_state["manifest"]["dataset"],
            "provenance": app_state["manifest"]["provenance"],
            "models": app_state["manifest"]["models"],
            "methodology": app_state["manifest"]["methodology"],
        }
    raise HTTPException(status_code=503, detail="Experiments manifest not generated yet")


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_rul(req: PredictionRequest):
    if req.model_id != "model_lstm_v1":
        raise HTTPException(status_code=400, detail=f"Model '{req.model_id}' not recognized or not loaded")

    if app_state["model"] is None or app_state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Inference engine or data pipeline not loaded")

    pipeline: CMAPSSDataPipeline = app_state["pipeline"]
    model: TurbofanLSTM = app_state["model"]
    device = torch.device(app_state["device"])

    # Load engine rows up to observation_cycle
    test_df = pipeline.raw_test_df
    assert test_df is not None
    engine_df = test_df[test_df["unit_id"] == req.unit_id].sort_values("cycle").reset_index(drop=True)
    max_available = int(engine_df["cycle"].max())

    if req.observation_cycle > max_available:
        raise HTTPException(
            status_code=400,
            detail=f"Requested cycle {req.observation_cycle} exceeds maximum observed cycle {max_available} for engine {req.unit_id}",
        )

    # Slice strictly up to observation cycle (no future lookahead!)
    cutoff_df = engine_df[engine_df["cycle"] <= req.observation_cycle].copy()
    scaled_df = pipeline.transform_df(cutoff_df)
    feats = scaled_df[pipeline.selected_features].values

    # Construct causal window
    if len(feats) < pipeline.window_size:
        pad_len = pipeline.window_size - len(feats)
        padding = np.repeat(feats[:1], pad_len, axis=0)
        window = np.vstack([padding, feats])
    else:
        window = feats[-pipeline.window_size :]

    with torch.no_grad():
        x_tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32, device=device)
        pred_val = float(model(x_tensor).cpu().numpy()[0])

    true_final = float(pipeline.test_rul_series.iloc[req.unit_id - 1]) if pipeline.test_rul_series is not None else None
    retrospective_truth = true_final + (max_available - req.observation_cycle) if true_final is not None else None

    return PredictionResponse(
        unit_id=req.unit_id,
        subset=req.subset,
        split="test",
        observed_through_cycle=req.observation_cycle,
        model_id=req.model_id,
        predicted_rul_cycles=round(pred_val, 2),
        true_rul_cycles=round(retrospective_truth, 2) if retrospective_truth is not None else None,
        prediction_status="ready",
    )
