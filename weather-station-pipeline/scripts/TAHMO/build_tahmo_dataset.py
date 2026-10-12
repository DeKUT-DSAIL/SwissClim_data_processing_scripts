#!/usr/bin/env python3
"""
build_tahmo_dataset.py
----------------------
Builds ML-ready station datasets from TAHMO observations (prefers clean intermediate
tahmo_qc_fused.zarr, falls back to tahmo_lazy.zarr).

Supports two execution modes:
1. Yearly NPZ Cubes (SwissClim Standard -- Recommended):
   Builds yearly compressed NPZ cubes (weather_data_{YYYY}.npz) matching the schema
   of scripts/02_build_station_data_yearly.py. Allows direct downstream reuse of
   scripts/03_merge_yearly_npz.py, scripts/04_convert_npz_to_hybrid.py, and
   scripts/07_split_dataset_holdout.py.

2. Monolithic Split Memory-Mapped Arrays (--split train|holdout|both):
   Directly creates continuous multi-year binary .npy arrays and .npz metadata companions.

Adopts the official SwissClim processing standards:
- Streams observations or collocated slots block by block.
- Derives 9 meteorological variables: [pa, msl, 2t, dt, 10u, 10v, wd, ws, ts].
- Temporal snapping to hourly timeline with +/- 15 minute rejection threshold and ts tracking.
- Placement onto the Augmented 12.8k (90, 180) spatial grid with zero collisions.
- Embedding of official SwissClim normalization parameters from weather_data_norm_stats.json.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# Ensure local script directory is on sys.path for tahmo_derivations import
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import tahmo_derivations as td

GRID_H = 90
GRID_W = 180
TARGET_VARS = ["pa", "msl", "2t", "dt", "10u", "10v", "wd", "ws", "ts"]

STAT_VAR_MAPPING = {
    "air_pressure": "pa",
    "air_pressure_at_sea_level": "msl",
    "air_temperature": "2t",
    "dew_point_temperature": "dt",
    "u_wind": "10u",
    "v_wind": "10v",
    "wind_from_direction": "wd",
    "wind_speed": "ws",
    "time_shift": "ts",
}


def load_normalization_stats(stats_json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and align SwissClim normalization locations and scales for TARGET_VARS."""
    with open(stats_json_path, "r") as f:
        stats = json.load(f)

    long_vars = stats["variables"]
    loc_dict = stats["locations"]
    scl_dict = stats["scales"]

    short_loc = {}
    short_scl = {}
    for lv in long_vars:
        sv = STAT_VAR_MAPPING.get(lv, lv)
        short_loc[sv] = float(loc_dict[lv])
        short_scl[sv] = float(scl_dict[lv])

    locations = np.array([short_loc[v] for v in TARGET_VARS], dtype=np.float32)
    scales = np.array([short_scl[v] for v in TARGET_VARS], dtype=np.float32)
    return locations, scales


def _populate_station_cube(
    cube: np.ndarray,
    ds: xr.Dataset,
    t_slice_start: int,
    t_slice_end: int,
    active_sids: List[str],
    zarr_stations: List[str],
    station_mapping: Dict,
    station_metadata: Dict,
    valid_time_snap: np.ndarray,
    time_diff_min: np.ndarray,
    nearest_h_idx: np.ndarray,
    batch_stations: int = 50,
) -> None:
    """Core inner loop to stream stations, reconcile, derive 9 vars, snap to hour, and fill cube."""
    n_batches = (len(active_sids) + batch_stations - 1) // batch_stations
    t_start_proc = time.time()

    for b_idx in range(n_batches):
        b_sids = active_sids[b_idx * batch_stations : (b_idx + 1) * batch_stations]
        b_zarr_indices = [zarr_stations.index(sid) for sid in b_sids]

        sub_ds = ds.isel(time=slice(t_slice_start, t_slice_end), station=b_zarr_indices)

        te_raw = sub_ds.te.values
        ap_raw = sub_ds.ap.values
        rh_raw = sub_ds.rh.values
        ws_raw = sub_ds.ws.values
        wd_raw = sub_ds.wd.values

        has_slot = (te_raw.ndim == 3)

        for i_st, sid in enumerate(b_sids):
            mapping_info = station_mapping[sid]
            r = mapping_info["row"]
            c = mapping_info["col"]
            elev = float(station_metadata.get(sid, {}).get("elevation", 0.0))

            if has_slot:
                te_rec = td.reconcile_collocated_scalars(te_raw[:, i_st, :], "te")
                ap_rec = td.reconcile_collocated_scalars(ap_raw[:, i_st, :], "ap")
                rh_rec = td.reconcile_collocated_scalars(rh_raw[:, i_st, :], "rh")
                ws_rec, wd_rec, u_rec, v_rec = td.reconcile_collocated_wind(
                    ws_raw[:, i_st, :], wd_raw[:, i_st, :]
                )
            else:
                te_rec = td.apply_range_gates(te_raw[:, i_st], "te")
                ap_rec = td.apply_range_gates(ap_raw[:, i_st], "ap")
                rh_rec = td.apply_range_gates(rh_raw[:, i_st], "rh")
                ws_rec = td.apply_range_gates(ws_raw[:, i_st], "ws")
                wd_rec = td.apply_range_gates(wd_raw[:, i_st], "wd")
                with np.errstate(all="ignore"):
                    rad = np.radians(wd_rec)
                    u_rec = -ws_rec * np.sin(rad)
                    v_rec = -ws_rec * np.cos(rad)
                    calm = ws_rec < 0.3
                    u_rec[calm] = 0.0
                    v_rec[calm] = 0.0
                    wd_rec[calm] = np.nan

            derived = td.compute_derived_9_vars(
                ap_kpa=ap_rec,
                te_c=te_rec,
                rh_pct=rh_rec,
                ws_ms=ws_rec,
                wd_deg=wd_rec,
                u_ms=u_rec,
                v_ms=v_rec,
                elevation_m=np.full_like(ap_rec, elev),
                ts_min=time_diff_min,
            )

            # Snap to nearest hour: smallest absolute offset wins
            st_valid = valid_time_snap & (~np.isnan(derived["2t"]))
            if not np.any(st_valid):
                continue

            valid_indices = np.where(st_valid)[0]
            v_h_indices = nearest_h_idx[valid_indices]
            v_diffs = np.abs(time_diff_min[valid_indices])

            sort_order = np.argsort(v_diffs)
            valid_indices = valid_indices[sort_order]
            v_h_indices = v_h_indices[sort_order]

            _, unique_hour_first = np.unique(v_h_indices, return_index=True)
            chosen_obs_indices = valid_indices[unique_hour_first]
            chosen_hours = v_h_indices[unique_hour_first]

            for v_idx, var_name in enumerate(TARGET_VARS):
                var_vals = derived[var_name][chosen_obs_indices]
                cube[chosen_hours, v_idx, r, c] = var_vals

        if hasattr(cube, "flush"):
            cube.flush()
        print(f"   Batch {b_idx + 1:2d}/{n_batches:2d} completed ({len(b_sids)} stations written).")

    elapsed_proc = time.time() - t_start_proc
    print(f"\n   Finished processing all stations in {elapsed_proc:.1f}s.")


def build_tahmo_yearly(
    zarr_path: str,
    grid_npz_path: str,
    mapping_json_path: str,
    stats_json_path: str,
    metadata_json_path: str,
    out_prefix: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    batch_stations: int = 50,
    time_threshold_minutes: int = 15,
    use_memmap: bool = False,
    overwrite: bool = False,
) -> None:
    """Build yearly NPZ cubes (weather_data_{YYYY}.npz) matching SwissClim schema."""
    print("=" * 80)
    print(f"BUILDING YEARLY TAHMO DATASET NPZ FILES: {start_year} -> {end_year}")
    print("=" * 80)
    print(f"Zarr store:     {zarr_path}")
    print(f"Grid NPZ:       {grid_npz_path}")
    print(f"Mapping JSON:   {mapping_json_path}")
    print(f"Out prefix:     {out_prefix}")
    print(f"Batch size:     {batch_stations} stations")
    print(f"Snap threshold: +/-{time_threshold_minutes} minutes")

    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Station Grid & Mapping
    grid_data = np.load(grid_npz_path)
    lon_grid = grid_data["lon"].astype(np.float32)
    lat_grid = grid_data["lat"].astype(np.float32)

    with open(mapping_json_path, "r") as f:
        station_mapping = json.load(f)

    with open(metadata_json_path, "r") as f:
        station_metadata = json.load(f)

    mapping_records = []
    for sid, info in sorted(station_mapping.items(), key=lambda x: x[1]["index"]):
        mapping_records.append({
            "station_id": sid,
            "index": int(info["index"]),
            "row": int(info["row"]),
            "col": int(info["col"]),
            "lat_orig": float(info.get("lat_orig", np.nan)),
            "lon_orig": float(info.get("lon_orig", np.nan)),
        })
    mapping_df = pd.DataFrame(mapping_records)

    # 2. Normalization Stats
    locations, scales = load_normalization_stats(stats_json_path)

    # 3. Open Zarr store once
    print(f"\nOpening Zarr store: {zarr_path} ...")
    ds = xr.open_zarr(zarr_path, consolidated=False)
    zarr_stations = ds.station.values.astype(str).tolist()
    zarr_times_raw = ds.time.values

    # Dynamically determine reference time from Zarr units attribute if present
    time_units = str(ds.time.attrs.get("units", "minutes since 2015-01-01 00:00:00"))
    if "since" in time_units:
        ref_time_str = time_units.split("since")[-1].strip().replace(" ", "T")
        ref_time = np.datetime64(ref_time_str, "m")
    else:
        ref_time = np.datetime64("2015-01-01T00:00:00", "m")

    if np.issubdtype(zarr_times_raw.dtype, np.datetime64):
        zarr_times_m = zarr_times_raw.astype("datetime64[m]")
    else:
        zarr_times_m = ref_time + zarr_times_raw.astype("timedelta64[m]")

    active_sids = [sid for sid in zarr_stations if sid in station_mapping]
    print(f"   Active mapped stations: {len(active_sids):,}")

    # Determine global bounds of available data
    zarr_min_time_m = zarr_times_m.min()
    zarr_max_time_m = zarr_times_m.max()

    # Apply explicit start_date and end_date from config if specified
    if start_date:
        cfg_start_m = np.datetime64(start_date[:16], "m")
        zarr_min_time_m = max(zarr_min_time_m, cfg_start_m)
    if end_date:
        cfg_end_m = np.datetime64(end_date[:16], "m")
        zarr_max_time_m = min(zarr_max_time_m, cfg_end_m)

    print(f"   Effective data timeline: {zarr_min_time_m} -> {zarr_max_time_m}")

    # If start_year or end_year not explicitly provided in config, derive from data
    if start_year is None:
        start_year = int(str(zarr_min_time_m)[:4])
    if end_year is None:
        end_year = int(str(zarr_max_time_m)[:4])

    print("=" * 80)
    print(f"BUILDING YEARLY TAHMO DATASET NPZ FILES: {start_year} -> {end_year}")
    print("=" * 80)

    for year in range(start_year, end_year + 1):
        year_out_file = f"{out_prefix}_{year}.npz"
        if os.path.exists(year_out_file) and (os.path.getsize(year_out_file) > 1024) and not overwrite:
            print(f"\n   ✓ Year {year} already exists at {year_out_file} ({os.path.getsize(year_out_file)/(1024*1024):.1f} MB). Skipping.")
            continue

        print("\n" + "-" * 80)
        print(f"PROCESSING YEAR {year} -> {year_out_file}")
        print("-" * 80)

        # Nominal year bounds
        nominal_start_m = np.datetime64(f"{year}-01-01T00:00:00", "m")
        nominal_end_m = np.datetime64(f"{year}-12-31T23:00:00", "m")

        # Clamp to actual Zarr observation window so we never generate empty/polluted hours
        actual_start_m = max(nominal_start_m, zarr_min_time_m)
        actual_end_m = min(nominal_end_m, zarr_max_time_m)

        if actual_start_m > actual_end_m:
            print(f"   ⚠️  Year {year} is completely outside available data ({zarr_min_time_m} to {zarr_max_time_m}). Skipping.")
            continue

        # Build hourly timestamps strictly within available data
        start_dt = actual_start_m.astype("datetime64[h]")
        end_dt = actual_end_m.astype("datetime64[h]") + np.timedelta64(1, "h")
        timestamps_h = np.arange(start_dt, end_dt, np.timedelta64(1, "h"), dtype="datetime64[h]")
        timestamps_ns = timestamps_h.astype("datetime64[ns]")
        T = len(timestamps_h)
        print(f"   Timesteps for {year}: T = {T:,} hours ({start_dt} to {actual_end_m.astype('datetime64[h]')})")

        start_buf = actual_start_m - np.timedelta64(30, "m")
        end_buf = actual_end_m + np.timedelta64(30, "m")
        t_mask = (zarr_times_m >= start_buf) & (zarr_times_m <= end_buf)
        t_indices = np.where(t_mask)[0]

        if len(t_indices) == 0:
            print(f"   ⚠️  No Zarr observations found for year {year} ({start_buf} to {end_buf}). Skipping.")
            continue

        t_slice_start = int(t_indices[0])
        t_slice_end = int(t_indices[-1]) + 1
        times_sliced_m = zarr_times_m[t_slice_start:t_slice_end]
        print(f"   Zarr time slice: index {t_slice_start:,} to {t_slice_end:,} ({len(times_sliced_m):,} 5-min intervals)")

        target_start_m = start_dt.astype("datetime64[m]")
        rel_m = ((times_sliced_m - target_start_m) / np.timedelta64(1, "m")).astype(np.int64)
        nearest_h_idx = np.round(rel_m / 60.0).astype(np.int32)
        time_diff_min = (rel_m - nearest_h_idx * 60).astype(np.float32)
        valid_time_snap = (np.abs(time_diff_min) <= time_threshold_minutes) & (nearest_h_idx >= 0) & (nearest_h_idx < T)

        temp_npy = None
        if use_memmap:
            temp_npy = f"{year_out_file}.tmp.npy"
            cube = np.memmap(temp_npy, dtype=np.float32, mode="w+", shape=(T, len(TARGET_VARS), GRID_H, GRID_W))
            cube[:] = np.nan
        else:
            cube = np.full((T, len(TARGET_VARS), GRID_H, GRID_W), np.nan, dtype=np.float32)

        _populate_station_cube(
            cube=cube,
            ds=ds,
            t_slice_start=t_slice_start,
            t_slice_end=t_slice_end,
            active_sids=active_sids,
            zarr_stations=zarr_stations,
            station_mapping=station_mapping,
            station_metadata=station_metadata,
            valid_time_snap=valid_time_snap,
            time_diff_min=time_diff_min,
            nearest_h_idx=nearest_h_idx,
            batch_stations=batch_stations,
        )

        print(f"   Saving compressed yearly NPZ to {year_out_file}...")
        np.savez_compressed(
            year_out_file,
            data=np.asarray(cube),
            timestamps=timestamps_ns,
            variables=np.array(TARGET_VARS, dtype=object),
            mapping=mapping_df.to_records(index=False),
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            locations=locations,
            scales=scales,
        )
        print(f"   ✅ Saved {year_out_file} (shape: {cube.shape})")

        del cube
        gc.collect()
        if temp_npy and os.path.exists(temp_npy):
            os.remove(temp_npy)

    print("\n" + "=" * 80)
    print("✅ ALL YEARLY TAHMO NPZ FILES BUILT SUCCESSFULLY")
    print("=" * 80)


def build_tahmo_split(
    zarr_path: str,
    grid_npz_path: str,
    mapping_json_path: str,
    stats_json_path: str,
    metadata_json_path: str,
    start_time_str: str,
    end_time_str: str,
    output_data_npy: str,
    output_meta_npz: str,
    batch_stations: int = 50,
    time_threshold_minutes: int = 15,
) -> None:
    """Build monolithic multi-year memory-mapped array for train or holdout split."""
    print("=" * 80)
    print(f"BUILDING MONOLITHIC TAHMO SPLIT: {Path(output_data_npy).name}")
    print("=" * 80)
    print(f"Zarr store:     {zarr_path}")
    print(f"Time range:     {start_time_str} -> {end_time_str}")
    print(f"Grid NPZ:       {grid_npz_path}")
    print(f"Mapping JSON:   {mapping_json_path}")
    print(f"Output data:    {output_data_npy}")
    print(f"Output meta:    {output_meta_npz}")
    print(f"Batch size:     {batch_stations} stations")
    print(f"Snap threshold: +/-{time_threshold_minutes} minutes")

    out_dir = Path(output_data_npy).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Station Grid & Mapping
    print("\n1. Loading grid and station mapping...")
    grid_data = np.load(grid_npz_path)
    lon_grid = grid_data["lon"].astype(np.float32)
    lat_grid = grid_data["lat"].astype(np.float32)

    with open(mapping_json_path, "r") as f:
        station_mapping = json.load(f)
    print(f"   Loaded {len(station_mapping):,} mapped TAHMO stations.")

    with open(metadata_json_path, "r") as f:
        station_metadata = json.load(f)

    mapping_records = []
    for sid, info in sorted(station_mapping.items(), key=lambda x: x[1]["index"]):
        mapping_records.append({
            "station_id": sid,
            "index": int(info["index"]),
            "row": int(info["row"]),
            "col": int(info["col"]),
            "lat_orig": float(info.get("lat_orig", np.nan)),
            "lon_orig": float(info.get("lon_orig", np.nan)),
        })
    mapping_df = pd.DataFrame(mapping_records)

    # 2. Load Normalization Statistics
    print("\n2. Loading SwissClim normalization parameters...")
    locations, scales = load_normalization_stats(stats_json_path)
    for v, loc, scl in zip(TARGET_VARS, locations, scales):
        print(f"   {v:5s}: center = {loc:10.2f}, scale = {scl:10.2f}")

    # 3. Build Hourly Timeline
    print("\n3. Constructing hourly timeline...")
    start_dt = np.datetime64(start_time_str, "h")
    end_dt = np.datetime64(end_time_str, "h") + np.timedelta64(1, "h")
    timestamps_h = np.arange(start_dt, end_dt, np.timedelta64(1, "h"), dtype="datetime64[h]")
    timestamps_ns = timestamps_h.astype("datetime64[ns]")
    T = len(timestamps_h)
    print(f"   Total timesteps T = {T:,} hours")
    print(f"   First timestep:    {timestamps_h[0]}")
    print(f"   Last timestep:     {timestamps_h[-1]}")

    # 4. Open Zarr store and index stations
    print("\n4. Opening Zarr store and aligning stations...")
    ds = xr.open_zarr(zarr_path, consolidated=False)
    zarr_stations = ds.station.values.astype(str).tolist()
    zarr_times_raw = ds.time.values

    ref_time = np.datetime64("2015-01-01T00:00:00", "m")
    if np.issubdtype(zarr_times_raw.dtype, np.datetime64):
        zarr_times_m = zarr_times_raw.astype("datetime64[m]")
    else:
        zarr_times_m = ref_time + zarr_times_raw.astype("timedelta64[m]")

    start_buf = np.datetime64(start_time_str, "m") - np.timedelta64(30, "m")
    end_buf = np.datetime64(end_time_str, "m") + np.timedelta64(30, "m")
    t_mask = (zarr_times_m >= start_buf) & (zarr_times_m <= end_buf)
    t_indices = np.where(t_mask)[0]

    if len(t_indices) == 0:
        raise ValueError(f"No Zarr observations found in window {start_buf} to {end_buf}")

    t_slice_start = int(t_indices[0])
    t_slice_end = int(t_indices[-1]) + 1
    times_sliced_m = zarr_times_m[t_slice_start:t_slice_end]
    print(f"   Zarr time slice: index {t_slice_start:,} to {t_slice_end:,} ({len(times_sliced_m):,} 5-min intervals)")

    # Snapping calculation
    target_start_m = np.datetime64(start_time_str, "m")
    rel_m = ((times_sliced_m - target_start_m) / np.timedelta64(1, "m")).astype(np.int64)
    nearest_h_idx = np.round(rel_m / 60.0).astype(np.int32)
    time_diff_min = (rel_m - nearest_h_idx * 60).astype(np.float32)
    valid_time_snap = (np.abs(time_diff_min) <= time_threshold_minutes) & (nearest_h_idx >= 0) & (nearest_h_idx < T)

    # 5. Initialize Memory-Mapped Output Cube
    print("\n5. Initializing memory-mapped data array on disk...")
    expected_bytes = T * len(TARGET_VARS) * GRID_H * GRID_W * 4
    print(f"   Target shape: ({T}, {len(TARGET_VARS)}, {GRID_H}, {GRID_W}) float32")
    print(f"   Target size:  {expected_bytes / (1024**3):.2f} GB")

    mmap_data = np.memmap(
        output_data_npy,
        dtype=np.float32,
        mode="w+",
        shape=(T, len(TARGET_VARS), GRID_H, GRID_W),
    )
    print("   Filling array canvas with NaN (spatial background)...")
    mmap_data[:] = np.nan
    mmap_data.flush()
    print("   Canvas ready.")

    # 6. Stream and Process Stations in Batches
    active_sids = [sid for sid in zarr_stations if sid in station_mapping]
    print(f"\n6. Streaming and processing {len(active_sids):,} stations in batches of {batch_stations}...")

    _populate_station_cube(
        cube=mmap_data,
        ds=ds,
        t_slice_start=t_slice_start,
        t_slice_end=t_slice_end,
        active_sids=active_sids,
        zarr_stations=zarr_stations,
        station_mapping=station_mapping,
        station_metadata=station_metadata,
        valid_time_snap=valid_time_snap,
        time_diff_min=time_diff_min,
        nearest_h_idx=nearest_h_idx,
        batch_stations=batch_stations,
    )

    # 7. Save Companion Metadata NPZ
    print(f"\n7. Saving companion metadata to {output_meta_npz}...")
    np.savez_compressed(
        output_meta_npz,
        timestamps=timestamps_ns,
        variables=np.array(TARGET_VARS, dtype=object),
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        locations=locations,
        scales=scales,
        mapping=mapping_df.to_records(index=False),
    )
    print(f"   Metadata successfully saved: {output_meta_npz}")
    print("\n" + "=" * 80)
    print(f"SUCCESS: {Path(output_data_npy).name} BUILT AND READY FOR ESFM")
    print("=" * 80)


REPO_DIR = current_dir.parent.parent


def resolve_path(p: Optional[str], default_rel: str) -> str:
    target = p if p is not None else default_rel
    if os.path.exists(target):
        return target
    candidate = str((REPO_DIR / target).resolve())
    if os.path.exists(candidate):
        return candidate
    return target


def resolve_default_zarr() -> str:
    candidates = [
        "/ritom/scratch/cscs/tligawa/ai-4-weather/data/tahmo_qc_fused.zarr",
        "/ritom/scratch/cscs/tligawa/ai-4-weather/data/tahmo_lazy.zarr",
        "/users/tligawa/ritom_scratch/ai-4-weather/data/tahmo_qc_fused.zarr",
        "/users/tligawa/ritom_scratch/ai-4-weather/data/tahmo_lazy.zarr",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TAHMO yearly NPZ or memory-mapped split datasets for ESFM."
    )
    parser.add_argument("--config", default=None, help="Path to config/tahmo_config.json")
    parser.add_argument("--mode", choices=["yearly", "split"], default=None,
                        help="Execution mode: 'yearly' (recommended) or 'split'")
    parser.add_argument("--yearly", action="store_true", help="Build yearly NPZ cubes")
    parser.add_argument("--split", choices=["train", "holdout", "both"], default=None,
                        help="Build monolithic split NPY files")
    parser.add_argument("--start_year", "--start-year", type=int, default=None)
    parser.add_argument("--end_year", "--end-year", type=int, default=None)
    parser.add_argument("--out_prefix", "--out-prefix", type=str, default=None)
    parser.add_argument(
        "--zarr_path", "--zarr-path",
        default=None,
        help="Path to tahmo_qc_fused.zarr or tahmo_lazy.zarr store",
    )
    parser.add_argument("--grid_npz", default=None, help="Path to augmented_station_grid_90x180.npz")
    parser.add_argument("--mapping_json", default=None, help="Path to tahmo_station_mapping_90x180.json")
    parser.add_argument("--stats_json", default=None, help="Path to weather_data_norm_stats.json")
    parser.add_argument("--metadata_json", default=None, help="Path to tahmo_african_metadata.json")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
    parser.add_argument("--time_threshold", type=int, default=15)
    parser.add_argument("--use_memmap", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing yearly NPZ files")

    args = parser.parse_args()

    full_cfg = {}
    cfg_file = resolve_path(args.config, "config/tahmo_config.json")
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                full_cfg = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {cfg_file}: {e}")

    # Determine mode:
    # 1. Explicit --split flag takes precedence for monolithic split mode
    # 2. Explicit --yearly or --mode yearly flag selects yearly mode
    # 3. Default: if "build_yearly" in config, run yearly mode; else run split mode
    yearly_cfg = full_cfg.get("build_yearly", {})
    split_cfg = full_cfg.get("build_dataset", {})

    if args.split is not None:
        mode = "split"
    elif args.yearly or (args.mode == "yearly") or (args.start_year is not None):
        mode = "yearly"
    elif args.mode == "split":
        mode = "split"
    elif "build_yearly" in full_cfg:
        mode = "yearly"
    else:
        mode = "split"

    if mode == "yearly":
        zarr_path = args.zarr_path or yearly_cfg.get("zarr_path") or resolve_default_zarr()
        grid_npz = resolve_path(args.grid_npz or yearly_cfg.get("grid_npz"), "data/augmented_station_grid_90x180.npz")
        mapping_json = resolve_path(args.mapping_json or yearly_cfg.get("mapping_json"), "data/tahmo_station_mapping_90x180.json")
        stats_json = resolve_path(args.stats_json or yearly_cfg.get("stats_json"), "data/weather_data_norm_stats.json")
        metadata_json = resolve_path(args.metadata_json or yearly_cfg.get("metadata_json"), "data/tahmo_african_metadata.json")
        batch_size = args.batch_size or yearly_cfg.get("batch_size") or 200
        out_prefix = args.out_prefix or yearly_cfg.get("out_prefix") or "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/weather_data"
        start_year = args.start_year if args.start_year is not None else yearly_cfg.get("start_year")
        end_year = args.end_year if args.end_year is not None else yearly_cfg.get("end_year")
        start_date = yearly_cfg.get("start_date")
        end_date = yearly_cfg.get("end_date")
        time_thresh = args.time_threshold or yearly_cfg.get("time_threshold") or 15
        use_memmap = args.use_memmap or yearly_cfg.get("use_memmap", False)

        build_tahmo_yearly(
            zarr_path=zarr_path,
            grid_npz_path=grid_npz,
            mapping_json_path=mapping_json,
            stats_json_path=stats_json,
            metadata_json_path=metadata_json,
            out_prefix=out_prefix,
            start_year=int(start_year) if start_year is not None else None,
            end_year=int(end_year) if end_year is not None else None,
            start_date=start_date,
            end_date=end_date,
            batch_stations=int(batch_size),
            time_threshold_minutes=int(time_thresh),
            use_memmap=use_memmap,
            overwrite=args.overwrite,
        )

    else:  # monolithic split mode
        zarr_path = args.zarr_path or split_cfg.get("zarr_path") or resolve_default_zarr()
        grid_npz = resolve_path(args.grid_npz or split_cfg.get("grid_npz"), "data/augmented_station_grid_90x180.npz")
        mapping_json = resolve_path(args.mapping_json or split_cfg.get("mapping_json"), "data/tahmo_station_mapping_90x180.json")
        stats_json = resolve_path(args.stats_json or split_cfg.get("stats_json"), "data/weather_data_norm_stats.json")
        metadata_json = resolve_path(args.metadata_json or split_cfg.get("metadata_json"), "data/tahmo_african_metadata.json")
        batch_size = args.batch_size or split_cfg.get("batch_stations") or 50
        time_thresh = args.time_threshold or split_cfg.get("time_threshold_minutes") or 15

        splits_to_build = []
        if args.split == "train":
            splits_to_build = ["train"]
        elif args.split == "holdout":
            splits_to_build = ["holdout"]
        else:
            splits_to_build = ["train", "holdout"]

        train_cfg = split_cfg.get("train", {
            "start_date": "2015-01-01T00:00:00",
            "end_date": "2021-12-31T23:59:59",
            "output_data_npy": "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_training_data.npy",
            "output_meta_npz": "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_training_meta.npz",
        })

        holdout_cfg = split_cfg.get("holdout", {
            "start_date": "2022-01-01T00:00:00",
            "end_date": "2026-08-31T23:59:59",
            "output_data_npy": "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_holdout_data.npy",
            "output_meta_npz": "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_holdout_meta.npz",
        })

        if "train" in splits_to_build:
            build_tahmo_split(
                zarr_path=zarr_path,
                grid_npz_path=grid_npz,
                mapping_json_path=mapping_json,
                stats_json_path=stats_json,
                metadata_json_path=metadata_json,
                start_time_str=train_cfg["start_date"],
                end_time_str=train_cfg["end_date"],
                output_data_npy=train_cfg["output_data_npy"],
                output_meta_npz=train_cfg["output_meta_npz"],
                batch_stations=int(batch_size),
                time_threshold_minutes=int(time_thresh),
            )

        if "holdout" in splits_to_build:
            build_tahmo_split(
                zarr_path=zarr_path,
                grid_npz_path=grid_npz,
                mapping_json_path=mapping_json,
                stats_json_path=stats_json,
                metadata_json_path=metadata_json,
                start_time_str=holdout_cfg["start_date"],
                end_time_str=holdout_cfg["end_date"],
                output_data_npy=holdout_cfg["output_data_npy"],
                output_meta_npz=holdout_cfg["output_meta_npz"],
                batch_stations=int(batch_size),
                time_threshold_minutes=int(time_thresh),
            )


if __name__ == "__main__":
    main()
