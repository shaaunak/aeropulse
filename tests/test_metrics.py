"""Unit tests for AeroPulse evaluation metrics."""

import math
import numpy as np
import pytest
from backend.metrics import compute_rmse, compute_mae, compute_nasa_score, compute_all_metrics


def test_perfect_predictions():
    y_true = np.array([10.0, 50.0, 100.0])
    y_pred = np.array([10.0, 50.0, 100.0])
    assert compute_rmse(y_true, y_pred) == 0.0
    assert compute_mae(y_true, y_pred) == 0.0
    assert compute_nasa_score(y_true, y_pred) == 0.0


def test_nasa_score_asymmetry_sign_penalty():
    """Verify that an overestimate (e > 0) is penalized MORE than an equal-magnitude underestimate (e < 0).

    In aerospace prognostics, claiming an engine has more life than it does is dangerous.
    """
    target = 100.0
    overestimate = 110.0  # e = +10
    underestimate = 90.0  # e = -10

    score_over = compute_nasa_score([target], [overestimate])
    score_under = compute_nasa_score([target], [underestimate])

    expected_over = math.exp(10.0 / 10.0) - 1.0  # exp(1) - 1 ~ 1.7183
    expected_under = math.exp(10.0 / 13.0) - 1.0  # exp(0.7692) - 1 ~ 1.1582

    assert math.isclose(score_over, expected_over, rel_tol=1e-5)
    assert math.isclose(score_under, expected_under, rel_tol=1e-5)
    assert score_over > score_under, "Overestimating RUL must strictly penalize more than underestimating"


def test_compute_all_metrics_structure():
    y_true = [50.0, 75.0]
    y_pred = [55.0, 70.0]
    metrics = compute_all_metrics(y_true, y_pred)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "nasa_score" in metrics
    assert metrics["sample_count"] == 2
    assert metrics["rmse"] == 5.0
    assert metrics["mae"] == 5.0
