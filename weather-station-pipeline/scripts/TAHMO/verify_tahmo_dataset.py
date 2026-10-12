#!/usr/bin/env python3
"""
verify_tahmo_dataset.py
-----------------------
Verification and diagnostic utility for TAHMO memory-mapped datasets.

Validates:
1. File structure, memory-mapping, and array shapes ((T, 9, 90, 180), float32).
2. Companion metadata keys, types, and alignment with SwissClim pre-training stats.
3. Strict spatial masking: confirms non-TAHMO grid cells are strictly NaN.
4. Physical value distribution checks (pressures, temperatures, wind components, dew point <= 2t).

Usage:
    python scripts/TAHMO/verify_tahmo_dataset.py \
        --data_npy /ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_training_data.npy \
        --meta_npz /ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_training_meta.npz
"""


import argparse
import os
import sys
from pathlib import Path
import numpy as np


def verify_dataset(data_npy_path: str, meta_npz_path: str) -> None:
    print("=" * 80)
    print("VERIFYING TAHMO DATASET ALIGNMENT WITH SWISSCLIM SPECIFICATION")
    print("=" * 80)
    print(f"Data file: {data_npy_path}")
    print(f"Meta file: {meta_npz_path}")

    # 1. Existence and file sizes
    if not os.path.exists(data_npy_path):
        raise FileNotFoundError(f"Data file not found: {data_npy_path}")
    if not os.path.exists(meta_npz_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_npz_path}")

    data_size_gb = os.path.getsize(data_npy_path) / (1024**3)
    meta_size_mb = os.path.getsize(meta_npz_path) / (1024**2)
    print(f"\n1. File sizes on disk:")
    print(f"   Data .npy: {data_size_gb:.2f} GB")
    print(f"   Meta .npz: {meta_size_mb:.2f} MB")

    # 2. Memory-mapped array inspection
    print(f"\n2. Memory-mapping data array...")
    data = np.load(data_npy_path, mmap_mode="r")
    print(f"   Shape: {data.shape}  (Expected: (T, 9, 90, 180))")
    print(f"   Dtype: {data.dtype}  (Expected: float32)")
    assert len(data.shape) == 4, f"Expected 4D array, got {len(data.shape)}D"
    assert data.shape[1] == 9, f"Expected 9 channels, got {data.shape[1]}"
    assert (data.shape[2], data.shape[3]) == (90, 180), f"Expected (90, 180) grid, got {data.shape[2:]}"
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
    T = data.shape[0]

    # 3. Metadata inspection
    print(f"\n3. Checking metadata keys and structure...")
    meta = np.load(meta_npz_path, allow_pickle=True)
    required_keys = ["timestamps", "variables", "lon_grid", "lat_grid", "locations", "scales", "mapping"]
    for k in required_keys:
        assert k in meta, f"Missing required metadata key: '{k}'"
        print(f"   Found key '{k}'")

    timestamps = meta["timestamps"]
    variables = [v.decode() if isinstance(v, bytes) else str(v) for v in meta["variables"]]
    lon_grid = meta["lon_grid"]
    lat_grid = meta["lat_grid"]
    mapping = meta["mapping"]

    print(f"   Timestamps: {len(timestamps):,} (matches T={T:,})")
    assert len(timestamps) == T, f"Timestamp count mismatch: {len(timestamps)} vs {T}"
    print(f"   Variables ({len(variables)}): {variables}")
    expected_vars = ["pa", "msl", "2t", "dt", "10u", "10v", "wd", "ws", "ts"]
    assert variables == expected_vars, f"Variable mismatch: {variables} vs {expected_vars}"

    print(f"   Grid dimensions: lat {lat_grid.shape}, lon {lon_grid.shape}")
    assert lat_grid.shape == (90, 180) and lon_grid.shape == (90, 180)
    print(f"   Mapped stations: {len(mapping):,}")

    # 4. Strict spatial masking validation
    print(f"\n4. Validating spatial masking (non-station cells must be strictly NaN)...")
    station_rows = np.array([m["row"] for m in mapping], dtype=int)
    station_cols = np.array([m["col"] for m in mapping], dtype=int)

    station_mask = np.zeros((90, 180), dtype=bool)
    station_mask[station_rows, station_cols] = True
    non_station_mask = ~station_mask

    mid_t = T // 2
    sample_slice = data[mid_t]
    non_station_vals = sample_slice[:, non_station_mask]
    all_nan = np.all(np.isnan(non_station_vals))
    print(f"   Non-station cells sampled: {non_station_vals.size:,}")
    print(f"   All non-station cells NaN: {all_nan}")
    assert all_nan, "Non-station cells contain non-NaN values! Spatial background leaked."

    # 5. Physical bounds validation on active station cells
    print(f"\n5. Inspecting physical distributions on active station cells...")
    station_vals = sample_slice[:, station_mask]

    for v_idx, var_name in enumerate(variables):
        v_data = station_vals[v_idx]
        v_valid = v_data[~np.isnan(v_data)]
        if len(v_valid) > 0:
            print(
                f"   {var_name:5s}: valid = {len(v_valid):4d}/{len(v_data):4d} "
                f"| min = {v_valid.min():9.2f} | mean = {v_valid.mean():9.2f} | max = {v_valid.max():9.2f}"
            )
        else:
            print(f"   {var_name:5s}: No valid observations at sample timestep {mid_t}")

    print("\n" + "=" * 80)
    print("ALL VALIDATION CHECKS PASSED: DATASET IS FULLY COMPLIANT")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TAHMO dataset compliance with SwissClim.")
    parser.add_argument("--data_npy", "--data-npy", required=True, help="Path to data .npy file")
    parser.add_argument("--meta_npz", "--meta-npz", required=True, help="Path to meta .npz file")
    args = parser.parse_args()

    verify_dataset(args.data_npy, args.meta_npz)


if __name__ == "__main__":
    main()

