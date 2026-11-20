"""Unit tests for NASA C-MAPSS data loader and pipeline integrity."""

import numpy as np
import pytest
from backend.data_loader import CMAPSSDataPipeline, load_raw_frames, split_engine_ids


def test_raw_data_loading_and_counts():
    train_df, test_df, rul_series = load_raw_frames("data/raw")
    assert train_df.shape[1] == 26
    assert test_df.shape[1] == 26
    assert len(rul_series) == 100
    assert train_df["unit_id"].nunique() == 100
    assert test_df["unit_id"].nunique() == 100
    assert not train_df.isnull().any().any()
    assert not test_df.isnull().any().any()


def test_train_val_split_disjointness():
    train_ids, val_ids = split_engine_ids(total_engines=100, val_ratio=0.20, seed=42)
    assert len(train_ids) == 80
    assert len(val_ids) == 20
    assert set(train_ids).isdisjoint(set(val_ids))
    assert min(train_ids) >= 1
    assert max(val_ids) <= 100


def test_pipeline_leak_free_feature_selection_and_scaling():
    pipeline = CMAPSSDataPipeline(data_dir="data/raw", val_ratio=0.20, seed=42).fit_and_prepare()

    # Verify dropped features are low variance
    assert len(pipeline.selected_features) > 0
    assert len(pipeline.dropped_features) > 0

    # setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19 are constant in FD001
    for const_feat in ["setting_3", "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]:
        assert const_feat in pipeline.dropped_features, f"{const_feat} should be dropped by variance threshold"

    # Active sensors must be kept
    for active_feat in ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_11", "sensor_12"]:
        assert active_feat in pipeline.selected_features, f"{active_feat} should be selected"

    # Scaler was fit on train units only
    artifacts = pipeline.export_artifacts()
    assert len(artifacts.scaler_mean) == len(pipeline.selected_features)
    assert len(artifacts.train_engine_ids) == 80
    assert len(artifacts.val_engine_ids) == 20


def test_window_generation_and_endpoints():
    pipeline = CMAPSSDataPipeline(data_dir="data/raw", window_size=30, rul_cap=125.0).fit_and_prepare()
    (X_tr, y_tr_cap, y_tr_uncap, tr_ends), (X_va, y_va_cap, y_va_uncap, va_ends) = pipeline.get_train_and_val_windows()

    # Shape checks: (N, 30, D)
    num_features = len(pipeline.selected_features)
    assert X_tr.ndim == 3
    assert X_tr.shape[1] == 30
    assert X_tr.shape[2] == num_features
    assert len(y_tr_cap) == len(X_tr)
    assert len(y_tr_uncap) == len(X_tr)
    assert len(tr_ends) == len(X_tr)

    # Capped targets must never exceed 125.0
    assert (y_tr_cap <= 125.0).all()
    # Uncapped targets can exceed 125.0
    assert (y_tr_uncap >= y_tr_cap).all()

    # Endpoints match train engines only
    train_units_in_ends = np.unique(tr_ends[:, 0])
    assert set(train_units_in_ends).issubset(set(pipeline.train_units))

    val_units_in_ends = np.unique(va_ends[:, 0])
    assert set(val_units_in_ends).issubset(set(pipeline.val_units))
    assert set(train_units_in_ends).isdisjoint(set(val_units_in_ends))


def test_official_test_windows_shape_and_alignment():
    pipeline = CMAPSSDataPipeline(data_dir="data/raw", window_size=30).fit_and_prepare()
    X_test_last, y_test_true, test_units = pipeline.get_official_test_windows()

    assert X_test_last.shape[0] == 100
    assert X_test_last.shape[1] == 30
    assert X_test_last.shape[2] == len(pipeline.selected_features)
    assert len(y_test_true) == 100
    assert len(test_units) == 100
    assert list(test_units) == list(range(1, 101))
    assert np.all(np.isfinite(X_test_last))
    assert np.all(np.isfinite(y_test_true))
