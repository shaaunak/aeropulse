"""NASA C-MAPSS FD001 Data Ingestion, Leakage-Free Preprocessing, and Window Generator.

Strict compliance with AeroPulse ML & Data Contract:
- 80/20 train/validation split by engine ID (disjoint groups)
- Preprocessing and feature selection fitted ONLY on training engines
- Causal 30-cycle temporal windowing strictly bounded within each engine
- Optimization target capped at 125 cycles; raw uncapped target preserved
- Exactly one official test prediction per engine at its final observed cycle
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


COLUMN_NAMES = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

DEFAULT_WINDOW_SIZE = 30
RUL_CAP = 125.0
VARIANCE_THRESHOLD = 1e-12


def find_dataset_files(data_dir: Union[str, Path] = "data/raw") -> Dict[str, Path]:
    """Locate NASA C-MAPSS raw text files recursively under data_dir."""
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    files: Dict[str, Path] = {}
    for target in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]:
        matches = list(p.rglob(target))
        if not matches:
            raise FileNotFoundError(f"Could not find required dataset file '{target}' under '{data_dir}'.")
        files[target] = matches[0]

    return files


def load_raw_frames(data_dir: Union[str, Path] = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw data into pandas structures with schema assertions."""
    paths = find_dataset_files(data_dir)

    train_df = pd.read_csv(paths["train_FD001.txt"], sep=r"\s+", header=None)
    test_df = pd.read_csv(paths["test_FD001.txt"], sep=r"\s+", header=None)
    rul_series = pd.read_csv(paths["RUL_FD001.txt"], sep=r"\s+", header=None).squeeze("columns")

    # Drop completely empty trailing columns if any
    train_df = train_df.dropna(how="all", axis=1)
    test_df = test_df.dropna(how="all", axis=1)

    assert train_df.shape[1] == 26, f"Expected 26 columns in train_FD001, got {train_df.shape[1]}"
    assert test_df.shape[1] == 26, f"Expected 26 columns in test_FD001, got {test_df.shape[1]}"
    assert len(rul_series) == 100, f"Expected 100 ground truth labels, got {len(rul_series)}"

    train_df.columns = COLUMN_NAMES
    test_df.columns = COLUMN_NAMES

    # Ensure integer types for unit and cycle
    train_df["unit_id"] = train_df["unit_id"].astype(int)
    train_df["cycle"] = train_df["cycle"].astype(int)
    test_df["unit_id"] = test_df["unit_id"].astype(int)
    test_df["cycle"] = test_df["cycle"].astype(int)

    # Sort deterministically
    train_df = train_df.sort_values(by=["unit_id", "cycle"]).reset_index(drop=True)
    test_df = test_df.sort_values(by=["unit_id", "cycle"]).reset_index(drop=True)

    # Integrity assertions
    assert train_df["unit_id"].nunique() == 100, "Train must contain exactly 100 unique engines"
    assert test_df["unit_id"].nunique() == 100, "Test must contain exactly 100 unique engines"
    assert not train_df[["unit_id", "cycle"]].duplicated().any(), "Duplicate (unit, cycle) in train"
    assert not test_df[["unit_id", "cycle"]].duplicated().any(), "Duplicate (unit, cycle) in test"
    assert (train_df["cycle"] > 0).all(), "All train cycles must be positive"
    assert (test_df["cycle"] > 0).all(), "All test cycles must be positive"
    assert np.all(np.isfinite(train_df.values)), "Train data contains non-finite values"
    assert np.all(np.isfinite(test_df.values)), "Test data contains non-finite values"

    return train_df, test_df, rul_series


def split_engine_ids(total_engines: int = 100, val_ratio: float = 0.20, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Deterministically partition engine IDs into disjoint train and validation sets."""
    rng = np.random.RandomState(seed)
    all_units = np.arange(1, total_engines + 1)
    val_count = int(total_engines * val_ratio)
    val_units = sorted(rng.choice(all_units, size=val_count, replace=False).tolist())
    train_units = sorted([u for u in all_units if u not in val_units])

    # Assert strict disjointness
    assert set(train_units).isdisjoint(set(val_units)), "Train and validation units must be strictly disjoint"
    assert len(train_units) + len(val_units) == total_engines
    return train_units, val_units


def add_train_rul_targets(df: pd.DataFrame, rul_cap: float = RUL_CAP) -> pd.DataFrame:
    """Calculate true run-to-failure RUL for training engines."""
    df_copy = df.copy()
    max_cycles = df_copy.groupby("unit_id")["cycle"].transform("max")
    df_copy["rul_uncapped"] = (max_cycles - df_copy["cycle"]).astype(float)
    df_copy["rul_capped"] = np.minimum(df_copy["rul_uncapped"], rul_cap)
    return df_copy


@dataclass
class PreprocessorArtifacts:
    selected_features: List[str]
    dropped_features: List[str]
    scaler_mean: Dict[str, float]
    scaler_scale: Dict[str, float]
    window_size: int
    rul_cap: float
    train_engine_ids: List[int]
    val_engine_ids: List[int]


class CMAPSSDataPipeline:
    """Leakage-safe data preparation pipeline for NASA C-MAPSS."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/raw",
        window_size: int = DEFAULT_WINDOW_SIZE,
        rul_cap: float = RUL_CAP,
        val_ratio: float = 0.20,
        seed: int = 42,
        variance_threshold: float = VARIANCE_THRESHOLD,
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.rul_cap = rul_cap
        self.val_ratio = val_ratio
        self.seed = seed
        self.variance_threshold = variance_threshold

        self.scaler = StandardScaler()
        self.selected_features: List[str] = []
        self.dropped_features: List[str] = []
        self.train_units: List[int] = []
        self.val_units: List[int] = []

        self.raw_train_df: Optional[pd.DataFrame] = None
        self.raw_test_df: Optional[pd.DataFrame] = None
        self.test_rul_series: Optional[pd.Series] = None

    def fit_and_prepare(self):
        """Execute leakage-safe ingestion, split, feature selection, and scaler fit."""
        train_df, test_df, rul_series = load_raw_frames(self.data_dir)
        self.raw_train_df = train_df
        self.raw_test_df = test_df
        self.test_rul_series = rul_series

        self.train_units, self.val_units = split_engine_ids(
            total_engines=100, val_ratio=self.val_ratio, seed=self.seed
        )

        # Slice train-split rows ONLY for fitting feature selector and scaler
        train_split_rows = train_df[train_df["unit_id"].isin(self.train_units)].copy()

        # Identify candidate feature columns (settings + sensors)
        candidate_cols = [c for c in COLUMN_NAMES if c not in ["unit_id", "cycle"]]

        # Variance check on train-split ONLY
        variances = train_split_rows[candidate_cols].var()
        self.selected_features = sorted(variances[variances > self.variance_threshold].index.tolist())
        self.dropped_features = sorted(variances[variances <= self.variance_threshold].index.tolist())

        # Fit StandardScaler on train-split rows ONLY
        self.scaler.fit(train_split_rows[self.selected_features])

        return self

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to features of a DataFrame."""
        assert self.selected_features, "Pipeline must be fitted before transforming"
        scaled_df = df.copy()
        scaled_vals = self.scaler.transform(df[self.selected_features])
        for idx, col in enumerate(self.selected_features):
            scaled_df[col] = scaled_vals[:, idx]
        return scaled_df

    def create_sliding_windows(
        self,
        df: pd.DataFrame,
        unit_ids: Optional[List[int]] = None,
        include_targets: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate strictly intra-engine causal sliding windows of shape (N, window_size, D).

        Returns:
            X: 3D numpy array of shape (N, window_size, len(selected_features))
            y_capped: 1D numpy array of target RUL (capped) at window endpoint
            y_uncapped: 1D numpy array of raw target RUL at window endpoint
            endpoints: 2D array of (unit_id, cycle) corresponding to each window endpoint
        """
        if unit_ids is not None:
            df_subset = df[df["unit_id"].isin(unit_ids)].copy()
        else:
            df_subset = df.copy()

        scaled_df = self.transform_df(df_subset)

        X_list: List[np.ndarray] = []
        y_capped_list: List[float] = []
        y_uncapped_list: List[float] = []
        endpoints_list: List[Tuple[int, int]] = []

        feature_matrix = scaled_df[self.selected_features].values
        unit_vals = scaled_df["unit_id"].values
        cycle_vals = scaled_df["cycle"].values

        has_targets = include_targets and ("rul_capped" in scaled_df.columns)
        if has_targets:
            y_cap_vals = scaled_df["rul_capped"].values
            y_uncap_vals = scaled_df["rul_uncapped"].values

        for u in np.unique(unit_vals):
            mask = unit_vals == u
            u_feats = feature_matrix[mask]
            u_cycles = cycle_vals[mask]
            if has_targets:
                u_yc = y_cap_vals[mask]
                u_yu = y_uncap_vals[mask]

            length = len(u_feats)
            if length < self.window_size:
                # Left-edge repeat padding for short trajectories
                pad_len = self.window_size - length
                padding = np.repeat(u_feats[:1], pad_len, axis=0)
                padded_feats = np.vstack([padding, u_feats])
                X_list.append(padded_feats)
                endpoints_list.append((u, int(u_cycles[-1])))
                if has_targets:
                    y_capped_list.append(float(u_yc[-1]))
                    y_uncapped_list.append(float(u_yu[-1]))
            else:
                for i in range(length - self.window_size + 1):
                    window = u_feats[i : i + self.window_size]
                    X_list.append(window)
                    end_idx = i + self.window_size - 1
                    endpoints_list.append((u, int(u_cycles[end_idx])))
                    if has_targets:
                        y_capped_list.append(float(u_yc[end_idx]))
                        y_uncapped_list.append(float(u_yu[end_idx]))

        X = np.array(X_list, dtype=np.float32)
        endpoints = np.array(endpoints_list, dtype=np.int32)
        y_c = np.array(y_capped_list, dtype=np.float32) if has_targets else np.array([])
        y_u = np.array(y_uncapped_list, dtype=np.float32) if has_targets else np.array([])
        return X, y_c, y_u, endpoints

    def get_train_and_val_windows(self):
        """Prepare train and validation windows from raw train data with verified RUL targets."""
        train_with_rul = add_train_rul_targets(self.raw_train_df, rul_cap=self.rul_cap)
        X_train, y_train_cap, y_train_uncap, train_ends = self.create_sliding_windows(
            train_with_rul, unit_ids=self.train_units, include_targets=True
        )
        X_val, y_val_cap, y_val_uncap, val_ends = self.create_sliding_windows(
            train_with_rul, unit_ids=self.val_units, include_targets=True
        )
        return (
            (X_train, y_train_cap, y_train_uncap, train_ends),
            (X_val, y_val_cap, y_val_uncap, val_ends),
        )

    def get_official_test_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract exactly ONE final observation window for each of the 100 test engines.

        Returns:
            X_test_last: shape (100, window_size, D) ordered by unit_id (1..100)
            y_test_true: shape (100,) from official RUL_FD001.txt
            test_units: shape (100,) unit IDs
        """
        assert self.raw_test_df is not None and self.test_rul_series is not None
        scaled_test = self.transform_df(self.raw_test_df)
        feature_matrix = scaled_test[self.selected_features].values
        unit_vals = scaled_test["unit_id"].values

        X_test_list: List[np.ndarray] = []
        ordered_units = sorted(np.unique(unit_vals).tolist())
        assert len(ordered_units) == 100, f"Expected 100 test engines, found {len(ordered_units)}"

        for u in ordered_units:
            mask = unit_vals == u
            u_feats = feature_matrix[mask]
            length = len(u_feats)
            if length < self.window_size:
                pad_len = self.window_size - length
                padding = np.repeat(u_feats[:1], pad_len, axis=0)
                window = np.vstack([padding, u_feats])
            else:
                window = u_feats[-self.window_size :]
            X_test_list.append(window)

        X_test_last = np.array(X_test_list, dtype=np.float32)
        y_test_true = np.array(self.test_rul_series.values, dtype=np.float32)
        test_units = np.array(ordered_units, dtype=np.int32)
        return X_test_last, y_test_true, test_units

    def get_test_engine_full_sequence(self, unit_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get causal windows across all cycles for a single test engine for scrubber playback."""
        assert self.raw_test_df is not None
        engine_df = self.raw_test_df[self.raw_test_df["unit_id"] == unit_id].copy()
        if len(engine_df) == 0:
            raise ValueError(f"Engine {unit_id} not found in test dataset")

        scaled_df = self.transform_df(engine_df)
        feats = scaled_df[self.selected_features].values
        cycles = scaled_df["cycle"].values

        X_windows: List[np.ndarray] = []
        valid_cycles: List[int] = []

        for idx in range(len(feats)):
            if idx + 1 < self.window_size:
                # Left-padded window
                pad_len = self.window_size - (idx + 1)
                padding = np.repeat(feats[:1], pad_len, axis=0)
                window = np.vstack([padding, feats[: idx + 1]])
            else:
                window = feats[idx + 1 - self.window_size : idx + 1]
            X_windows.append(window)
            valid_cycles.append(int(cycles[idx]))

        return np.array(X_windows, dtype=np.float32), np.array(valid_cycles, dtype=np.int32)

    def export_artifacts(self) -> PreprocessorArtifacts:
        """Export serialized parameters for reproducibility and manifest generation."""
        scaler_mean = {col: float(m) for col, m in zip(self.selected_features, self.scaler.mean_)}
        scaler_scale = {col: float(s) for col, s in zip(self.selected_features, self.scaler.scale_)}
        return PreprocessorArtifacts(
            selected_features=self.selected_features,
            dropped_features=self.dropped_features,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            window_size=self.window_size,
            rul_cap=self.rul_cap,
            train_engine_ids=self.train_units,
            val_engine_ids=self.val_units,
        )
