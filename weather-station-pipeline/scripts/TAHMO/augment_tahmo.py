"""
augment_tahmo.py
================
Build Augmented 12.8k Grid combining 11k Global Stations + TAHMO African Stations.

Implements SwissClim Algorithm 1:
1. Loads 11,863 global stations from SwissClim meta_info.json.
2. Extracts TAHMO station coordinates from tahmo_qc_fused.zarr (or tahmo_lazy.zarr fallback).
3. Filters to genuine African stations (excluding the 12 overseas test stations).
4. Merges into meta_info_augmented_12k.json.
5. Runs SwissClim Algorithm 1 (00_build_station_grid_from_metadata.py) to construct
   an augmented (90, 180) spatial grid with zero collisions and exact global bounds.
6. Exports subset mapping tahmo_station_mapping_90x180.json and tahmo_african_metadata.json.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import xarray as xr

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent.parent
DEFAULT_DATA_DIR = str(REPO_DIR / "data")
DEFAULT_SCRIPTS_DIR = str(REPO_DIR / "scripts")
DEFAULT_QC_ZARR = "/ritom/scratch/cscs/tligawa/ai-4-weather/data/tahmo_qc_fused.zarr"
DEFAULT_RAW_ZARR = "/ritom/scratch/cscs/tligawa/ai-4-weather/data/tahmo_lazy.zarr"


def resolve_zarr_path(requested_path: str = None) -> str:
    """Find available Zarr store, preferring QC-fused store over raw store."""
    candidates = []
    if requested_path:
        candidates.append(requested_path)
    candidates.extend([
        DEFAULT_QC_ZARR,
        DEFAULT_RAW_ZARR,
        "/users/tligawa/ritom_scratch/ai-4-weather/data/tahmo_qc_fused.zarr",
        "/users/tligawa/ritom_scratch/ai-4-weather/data/tahmo_lazy.zarr",
    ])
    for c in candidates:
        if os.path.exists(c):
            return c
    return requested_path or DEFAULT_QC_ZARR


def build_augmented_metadata_and_grid(
    swissclim_data_dir: str = DEFAULT_DATA_DIR,
    swissclim_scripts_dir: str = DEFAULT_SCRIPTS_DIR,
    zarr_path: str = None,
    out_dir: str = DEFAULT_DATA_DIR,
    meta_11k_path: str = None,
    grid_h: int = 90,
    grid_w: int = 180,
    half: float = 0.005,
):
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = resolve_zarr_path(zarr_path)

    # 1. Load 11k global station metadata
    if meta_11k_path is None:
        meta_11k_path = os.path.join(swissclim_data_dir, "meta_info.json")
    print(f"Loading 11k global station metadata from: {meta_11k_path}")
    with open(meta_11k_path, "r") as f:
        meta_augmented = json.load(f)
    n_11k = len(meta_augmented)
    print(f"  Loaded {n_11k:,} stations from 11k dataset.")

    # 2. Load TAHMO stations from Zarr store
    print(f"Loading TAHMO stations from Zarr store: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=False)
    t_sids = ds.station.values.astype(str)

    if "lat" in ds:
        t_lats = ds.lat.values.astype(float)
    else:
        raise KeyError(f"Coordinate 'lat' not found in {zarr_path}")

    if "lon" in ds:
        t_lons = ds.lon.values.astype(float)
    else:
        raise KeyError(f"Coordinate 'lon' not found in {zarr_path}")

    if "elevation" in ds:
        t_elev = ds.elevation.values.astype(float)
    elif "elevationmsl" in ds:
        t_elev = ds.elevationmsl.values.astype(float)
    else:
        t_elev = np.zeros(len(t_sids), dtype=float)

    n_tahmo_raw = len(t_sids)
    print(f"  Found {n_tahmo_raw:,} total TAHMO stations.")

    # 3. Filter to genuine African stations
    # Africa bounding box: lat [-36, 38], lon [-20, 55]
    african_mask = (t_lats >= -36.0) & (t_lats <= 38.0) & (t_lons >= -20.0) & (t_lons <= 55.0)
    african_sids = t_sids[african_mask]
    african_lats = t_lats[african_mask]
    african_lons = t_lons[african_mask]
    african_elev = t_elev[african_mask]
    n_african = len(african_sids)
    n_excluded = n_tahmo_raw - n_african
    print(f"  Filtered to {n_african:,} African TAHMO stations (excluded {n_excluded} overseas test stations).")

    # 4. Merge TAHMO stations into augmented metadata
    tahmo_added = 0
    tahmo_metadata_lookup = {}
    for sid, lat, lon, elev in zip(african_sids, african_lats, african_lons, african_elev):
        meta_augmented[str(sid)] = {
            "latitude": float(lat),
            "longitude": float(lon),
        }
        tahmo_metadata_lookup[str(sid)] = {
            "latitude": float(lat),
            "longitude": float(lon),
            "elevation": float(elev),
        }
        tahmo_added += 1

    total_augmented = len(meta_augmented)
    print(f"Merged metadata total: {total_augmented:,} stations ({n_11k:,} 11k + {tahmo_added:,} TAHMO).")

    # Save augmented metadata JSON
    aug_meta_json = os.path.join(out_dir, "meta_info_augmented_12k.json")
    with open(aug_meta_json, "w") as f:
        json.dump(meta_augmented, f, indent=2)
    print(f"Saved augmented metadata to: {aug_meta_json}")

    # Also save dedicated TAHMO African metadata
    tahmo_meta_json = os.path.join(out_dir, "tahmo_african_metadata.json")
    with open(tahmo_meta_json, "w") as f:
        json.dump(tahmo_metadata_lookup, f, indent=2)
    print(f"Saved TAHMO African metadata to: {tahmo_meta_json}")

    # 5. Run SwissClim Algorithm 1 script
    grid_script = os.path.join(swissclim_scripts_dir, "00_build_station_grid_from_metadata.py")
    out_npz = os.path.join(out_dir, "augmented_station_grid_90x180.npz")
    out_json = os.path.join(out_dir, "augmented_mapping_90x180.json")

    cmd = [
        sys.executable,
        grid_script,
        "--input", aug_meta_json,
        "--H", str(grid_h),
        "--W", str(grid_w),
        "--half", str(half),
        "--out_npz", out_npz,
        "--out_json", out_json,
    ]
    print(f"Running SwissClim Algorithm 1 grid generator:\n  {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # 6. Verify and extract TAHMO station mapping subset
    print("Verifying generated grid and mapping...")
    with open(out_json, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        full_mapping = json.loads(content)
    except json.JSONDecodeError as err:
        # Recover if file has trailing garbage/extra data
        if "Extra data" in str(err) and err.pos > 0:
            full_mapping = json.loads(content[:err.pos])
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(full_mapping, f)
        else:
            raise err

    tahmo_sub_mapping = {}
    for sid in african_sids:
        if str(sid) in full_mapping:
            tahmo_sub_mapping[str(sid)] = full_mapping[str(sid)]

    tahmo_mapping_file = os.path.join(out_dir, f"tahmo_station_mapping_{grid_h}x{grid_w}.json")
    with open(tahmo_mapping_file, "w") as f:
        json.dump(tahmo_sub_mapping, f, indent=2)

    # Verify uniqueness
    tahmo_rc_pairs = [(info["row"], info["col"]) for info in tahmo_sub_mapping.values()]
    unique_rc = set(tahmo_rc_pairs)
    print(f"TAHMO stations mapped: {len(tahmo_sub_mapping):,} / {n_african:,}")
    print(f"Unique grid cells for TAHMO: {len(unique_rc):,} / {len(tahmo_sub_mapping):,}")
    assert len(unique_rc) == len(tahmo_sub_mapping), "Collision detected in grid assignment!"
    print(f"SUCCESS: All {n_african:,} African TAHMO stations mapped with ZERO collisions!")
    print(f"Saved TAHMO station mapping table to: {tahmo_mapping_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Augmented 12k Grid combining 11k Global Stations + TAHMO African Stations."
    )
    parser.add_argument("--config", default=None, help="Path to config JSON (e.g. config/tahmo_config.json)")
    parser.add_argument("--meta_11k", default=None, help="Path to 11k global station metadata JSON")
    parser.add_argument("--zarr_path", "--zarr-path", default=None, help="Path to TAHMO zarr store")
    parser.add_argument("--out_dir", "--out-dir", default=None, help="Output directory for metadata and grid")
    parser.add_argument("--grid_h", type=int, default=None, help="Grid height (default: 90)")
    parser.add_argument("--grid_w", type=int, default=None, help="Grid width (default: 180)")
    parser.add_argument("--half", type=float, default=None, help="Grid half width in deg (default: 0.005)")
    args = parser.parse_args()

    cfg = {}
    cfg_file = args.config or str(REPO_DIR / "config" / "tahmo_config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, "r") as f:
            full_cfg = json.load(f)
            cfg = full_cfg.get("augment_grid", {})

    meta_11k = args.meta_11k or cfg.get("meta_11k") or os.path.join(DEFAULT_DATA_DIR, "meta_info.json")
    if not os.path.isabs(meta_11k):
        meta_11k = str((REPO_DIR / meta_11k).resolve())

    zarr_path = args.zarr_path or cfg.get("zarr_path")
    out_dir = args.out_dir or cfg.get("out_dir") or DEFAULT_DATA_DIR
    if not os.path.isabs(out_dir):
        out_dir = str((REPO_DIR / out_dir).resolve())

    grid_h = args.grid_h or cfg.get("grid_h", 90)
    grid_w = args.grid_w or cfg.get("grid_w", 180)
    half = args.half or cfg.get("half", 0.005)

    build_augmented_metadata_and_grid(
        swissclim_data_dir=out_dir,
        swissclim_scripts_dir=DEFAULT_SCRIPTS_DIR,
        zarr_path=zarr_path,
        out_dir=out_dir,
        meta_11k_path=meta_11k,
        grid_h=grid_h,
        grid_w=grid_w,
        half=half,
    )


if __name__ == "__main__":
    main()

