"""AeroPulse evaluation metrics for NASA C-MAPSS prognostics.

Strict mathematical definitions according to official NASA PHM08 benchmark:
- RMSE: Root Mean Squared Error = sqrt(mean((y_pred - y_true)**2))
- MAE: Mean Absolute Error = mean(abs(y_pred - y_true))
- NASA Asymmetric Scoring Function:
    e = y_pred - y_true
    s_i = exp(-e / 13) - 1 if e < 0 else exp(e / 10) - 1
    Total Score = sum(s_i)
  Overestimating RUL (e > 0) is penalized more heavily than underestimating (e < 0).
"""

from typing import Union
import numpy as np


def compute_rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Compute Root Mean Squared Error."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) == 0:
        return 0.0
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def compute_mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Compute Mean Absolute Error."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(np.abs(yp - yt)))


def compute_nasa_score(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Compute official NASA PHM08 asymmetric scoring function.

    Formula:
        e = pred - true
        if e < 0 (early prediction / underestimate): exp(-e / 13) - 1
        if e >= 0 (late prediction / overestimate):  exp(e / 10) - 1
    Score = sum over all engines.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) == 0:
        return 0.0
    errors = yp - yt
    # Clip large positive errors to prevent numerical overflow in floating point exp
    # 700 is approximately max float64 exp limit, but for realistic errors e < 100
    clipped_e = np.clip(errors, -200.0, 200.0)
    score_per_engine = np.where(
        clipped_e < 0,
        np.exp(-clipped_e / 13.0) - 1.0,
        np.exp(clipped_e / 10.0) - 1.0,
    )
    return float(np.sum(score_per_engine))


def compute_all_metrics(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> dict:
    """Compute full suite of benchmark metrics."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    rmse = compute_rmse(yt, yp)
    mae = compute_mae(yt, yp)
    nasa_score = compute_nasa_score(yt, yp)
    errors = (yp - yt).tolist()
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "nasa_score": round(nasa_score, 2),
        "sample_count": len(yt),
        "mean_error": round(float(np.mean(yp - yt)), 4) if len(yt) > 0 else 0.0,
        "std_error": round(float(np.std(yp - yt)), 4) if len(yt) > 0 else 0.0,
        "max_overestimate": round(float(np.max(yp - yt)), 4) if len(yt) > 0 else 0.0,
        "max_underestimate": round(float(np.min(yp - yt)), 4) if len(yt) > 0 else 0.0,
    }
