"""Integration and contract tests for AeroPulse FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest
from backend.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service_available"] is True
    assert data["loaded_model_id"] == "model_lstm_v1"
    assert data["dataset_state"] == "ready"
    assert data["dataset_counts"]["test_engines"] == 100


def test_datasets_endpoint(client: TestClient):
    resp = client.get("/api/v1/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert "FD001" in data["subsets"]
    assert data["train_trajectories"] == 100
    assert data["test_trajectories"] == 100


def test_engines_list_endpoint(client: TestClient):
    resp = client.get("/api/v1/engines?subset=FD001&split=test")
    assert resp.status_code == 200
    engines = resp.json()
    assert len(engines) == 100
    assert engines[0]["unit_id"] == 1
    assert engines[0]["subset"] == "FD001"
    assert engines[0]["observed_cycles"] > 0
    assert engines[0]["prediction_status"] == "ready"


def test_single_engine_detail(client: TestClient):
    resp = client.get("/api/v1/engines/1?subset=FD001&split=test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["unit_id"] == 1
    assert data["observed_cycles"] == 31  # FD001 test engine 1 has 31 observed cycles
    assert "predicted_rul_final" in data
    assert "true_rul_final" in data
    assert "sensors" in data
    assert "predictions_by_cycle" in data


def test_predict_endpoint_valid_and_invalid(client: TestClient):
    # Valid causal inference request
    req = {
        "model_id": "model_lstm_v1",
        "subset": "FD001",
        "unit_id": 1,
        "observation_cycle": 25,
    }
    resp = client.post("/api/v1/predict", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["unit_id"] == 1
    assert data["observed_through_cycle"] == 25
    assert data["predicted_rul_cycles"] >= 0.0
    assert data["prediction_status"] == "ready"

    # Invalid future lookahead beyond observed history
    req_future = {
        "model_id": "model_lstm_v1",
        "subset": "FD001",
        "unit_id": 1,
        "observation_cycle": 9999,  # exceeds observed cycles
    }
    resp_future = client.post("/api/v1/predict", json=req_future)
    assert resp_future.status_code == 400
    assert "exceeds maximum observed cycle" in resp_future.json()["detail"]


def test_experiments_endpoint(client: TestClient):
    resp = client.get("/api/v1/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "lstm" in data["models"]
    assert "gru" in data["models"]
    assert "ridge_tabular" in data["models"]
