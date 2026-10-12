#!/usr/bin/env python3
"""
build_station_grid_from_metadata.py

Input:
  metadata.json  (dict: station_id -> {"latitude": <float>, "longitude": <float>})

Output:
  - NPZ: lon/lat grids + bounds + station mask/index/id grids
  - JSON: station_id -> row/col + orig/mapped coords + bounds + deltas

Constraints satisfied (matrix case):
  - lat in [-90, 90], lon in [0, 360)
  - lat strictly decreasing along every column: lat[1:,:] - lat[:-1,:] < 0
  - lon strictly increasing along every row: lon[:,1:] - lon[:,:-1] > 0

Bounds satisfied:
  - lat_max > lat_min, lon_max > lon_min (strict, everywhere)
  - abs(lat_*) <= 90
  - 0 <= lon_* <= 360

Station fidelity:
  - Station cells use real metadata:
      lat: unchanged (up to float32 representation)
      lon: wrapped into [0,360) and only tiny float32 nextafter nudges if needed
        to maintain strict row-wise ordering.

Locality:
  - Stations are sorted by latitude (north->south) then split into H consecutive bands (rows).
  - Within each row-band, stations are sorted by lon and assigned strictly increasing columns
    close to preferred col ~ lon/360*(W-1).

Usage:
  python build_station_grid_from_metadata.py --input metadata.json --H 180 --W 360 --half 0.005 \
      --out_npz final_results_station_grid.npz --out_json final_results_mapping.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def wrap_lon_360(lon: float) -> float:
    return (lon + 360.0) % 360.0


def circular_lon_diff_deg(a: float, b: float) -> float:
    d = (a - b) % 360.0
    return float(min(d, 360.0 - d))


def assign_cols_monotone(lons_sorted: List[float], W: int) -> List[int]:
    """Assign strictly increasing integer columns close to lon-proportional preferred cols."""
    n = len(lons_sorted)
    if n == 0:
        return []
    pref = np.rint(np.array(lons_sorted, dtype=np.float64) / 360.0 * (W - 1)).astype(int)
    pref = np.clip(pref, 0, W - 1)

    cols = np.empty(n, dtype=int)
    last = -1
    for i in range(n):
        min_allowed = last + 1
        max_allowed = (W - 1) - (n - 1 - i)
        c = int(pref[i])
        c = max(c, min_allowed)
        c = min(c, max_allowed)
        cols[i] = c
        last = c
    return cols.tolist()


def fill_row_from_station_points(
    W: int,
    cols: List[int],
    lons: List[float],
    lo: float = 0.0,
    hi: float = 360.0,
) -> np.ndarray:
    """
    Build a strictly increasing lon row of length W.
    Station lon values are used as fixed points; other positions are piecewise-linearly filled.
    Strictness is enforced with float32 nextafter.
    """
    hi_excl = float(np.nextafter(np.float32(360.0), np.float32(0.0)))

    if len(cols) == 0:
        base = np.linspace(lo, hi_excl, W, endpoint=True, dtype=np.float32)
        for c in range(1, W):
            if base[c] <= base[c - 1]:
                base[c] = np.nextafter(base[c - 1], np.float32(np.inf))
        return base

    order = np.argsort(cols)
    cols = np.array(cols, dtype=int)[order]
    lons = np.array(lons, dtype=np.float32)[order]
    lons = np.clip(lons, lo, hi_excl).astype(np.float32)

    # Ensure station lons strictly increasing (tiny nudges only if duplicates)
    for i in range(1, len(lons)):
        if lons[i] <= lons[i - 1]:
            lons[i] = np.nextafter(lons[i - 1], np.float32(np.inf))

    # Endpoints for interpolation
    pts_c = [0] + cols.tolist() + [W - 1]
    left_v = max(lo, float(lons[0]) - (cols[0] + 1) * 0.01)
    right_v = min(hi_excl, float(lons[-1]) + (W - 1 - cols[-1] + 1) * 0.01)
    pts_v = [left_v] + lons.tolist() + [right_v]
    pts_v = np.array(pts_v, dtype=np.float32)

    for i in range(1, len(pts_v)):
        if pts_v[i] <= pts_v[i - 1]:
            pts_v[i] = np.nextafter(pts_v[i - 1], np.float32(np.inf))
    if not (pts_v[-1] < 360.0):
        pts_v[-1] = np.nextafter(np.float32(360.0), np.float32(0.0))

    row = np.empty(W, dtype=np.float32)
    for (c0, c1, v0, v1) in zip(pts_c[:-1], pts_c[1:], pts_v[:-1], pts_v[1:]):
        seg = np.linspace(v0, v1, c1 - c0 + 1, dtype=np.float32)
        row[c0 : c1 + 1] = seg

    # Final strict enforcement
    for c in range(1, W):
        if row[c] <= row[c - 1]:
            row[c] = np.nextafter(row[c - 1], np.float32(np.inf))
    if not (row[-1] < 360.0):
        row[-1] = np.nextafter(np.float32(360.0), np.float32(0.0))
        for c in range(W - 2, -1, -1):
            if row[c] >= row[c + 1]:
                row[c] = np.nextafter(row[c + 1], np.float32(0.0))

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="metadata.json")
    ap.add_argument("--H", type=int, default=180)
    ap.add_argument("--W", type=int, default=360)
    ap.add_argument("--half", type=float, default=0.005)
    ap.add_argument("--out_npz", default="final_results_station_grid.npz")
    ap.add_argument("--out_json", default="final_results_mapping.json")
    args = ap.parse_args()

    H, W = int(args.H), int(args.W)
    half = float(args.half)

    meta: Dict[str, Any] = json.loads(Path(args.input).read_text())

    # station list: (sid, lat, lon_orig, lon360)
    stations: List[Tuple[str, float, float, float]] = []
    for sid, coord in meta.items():
        lat = float(coord["latitude"])
        lon = float(coord["longitude"])
        lon360 = wrap_lon_360(lon)
        stations.append((sid, lat, lon, lon360))

    # Sort by latitude (north->south) then lon
    stations.sort(key=lambda x: (-x[1], x[3]))
    N = len(stations)
    if N > H * W:
        raise ValueError(f"N={N} exceeds H*W={H*W}.")

    # Split into H latitude bands
    rows = np.array_split(np.array(stations, dtype=object), H)

    station_mask = np.zeros((H, W), dtype=np.uint8)
    station_index_grid = np.full((H, W), -1, dtype=np.int32)
    station_id_grid = np.full((H, W), "", dtype=object)

    mapping: Dict[str, Any] = {}
    row_station_cols: List[List[int]] = [[] for _ in range(H)]
    row_station_lons: List[List[float]] = [[] for _ in range(H)]
    row_station_lats: List[List[float]] = [[] for _ in range(H)]

    station_idx = 0
    for r in range(H):
        row_st = list(rows[r])
        if len(row_st) == 0:
            continue
        row_st.sort(key=lambda x: float(x[3]))  # lon360
        lons = [float(x[3]) for x in row_st]
        cols = assign_cols_monotone(lons, W)

        for (sid, lat0, lon0, lon360), c in zip(row_st, cols):
            station_mask[r, c] = 1
            station_index_grid[r, c] = station_idx
            station_id_grid[r, c] = sid

            mapping[sid] = {
                "index": int(station_idx),
                "row": int(r),
                "col": int(c),
                "lat_orig": float(lat0),
                "lon_orig": float(lon0),
                "lon_360_orig": float(lon360),
            }
            row_station_cols[r].append(int(c))
            row_station_lons[r].append(float(lon360))
            row_station_lats[r].append(float(lat0))
            station_idx += 1

    # lon grid from station points (row-wise)
    lon = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        lon[r] = fill_row_from_station_points(W, row_station_cols[r], row_station_lons[r], lo=0.0, hi=360.0)

    # lat grid: per-row reference (strictly decreasing), then overwrite station lats
    row_ref = np.zeros(H, dtype=np.float32)
    for r in range(H):
        if len(row_station_lats[r]) > 0:
            row_ref[r] = np.float32(np.max(row_station_lats[r]))
        else:
            row_ref[r] = np.float32(90.0) if r == 0 else np.nextafter(row_ref[r - 1], np.float32(-np.inf))
    for r in range(1, H):
        if row_ref[r] >= row_ref[r - 1]:
            row_ref[r] = np.nextafter(row_ref[r - 1], np.float32(-np.inf))
    row_ref = np.clip(row_ref, -90.0, 90.0).astype(np.float32)

    lat = np.repeat(row_ref[:, None], W, axis=1).astype(np.float32)

    # overwrite station lats with true station lat
    for sid, info in mapping.items():
        r, c = info["row"], info["col"]
        lat[r, c] = np.float32(info["lat_orig"])

    # enforce strict decreasing down columns (tiny nudges only if ties)
    for c in range(W):
        for r in range(1, H):
            if lat[r, c] >= lat[r - 1, c]:
                lat[r, c] = np.nextafter(lat[r - 1, c], np.float32(-np.inf))
    lat = np.clip(lat, -90.0, 90.0).astype(np.float32)

    # bounds
    lat_min = np.clip(lat - half, -90.0, 90.0).astype(np.float32)
    lat_max = np.clip(lat + half, -90.0, 90.0).astype(np.float32)
    lon_min = np.clip(lon - half, 0.0, 360.0).astype(np.float32)
    lon_max = np.clip(lon + half, 0.0, 360.0).astype(np.float32)
    lat_max = np.maximum(lat_max, np.nextafter(lat_min, np.float32(np.inf))).astype(np.float32)
    lon_max = np.maximum(lon_max, np.nextafter(lon_min, np.float32(np.inf))).astype(np.float32)

    # validations (your requirements)
    assert np.all(lat <= 90.0) and np.all(lat >= -90.0)
    assert np.all(lon >= 0.0) and np.all(lon < 360.0)
    assert np.all(lat[1:, :] - lat[:-1, :] < 0)
    assert np.all(lon[:, 1:] - lon[:, :-1] > 0)

    assert (lat_max > lat_min).all()
    assert (lon_max > lon_min).all()
    assert (np.abs(lat_max) <= 90.0).all() and (np.abs(lat_min) <= 90.0).all()
    assert (lon_max <= 360.0).all() and (lon_min <= 360.0).all()
    assert (lon_max >= 0.0).all() and (lon_min >= 0.0).all()

    # per-station deltas & finalize mapping
    dlon_max = 0.0
    dlat_max = 0.0
    for sid, info in mapping.items():
        r, c = info["row"], info["col"]
        info["lon_mapped"] = float(lon[r, c])
        info["lat_mapped"] = float(lat[r, c])
        info["dlon_circ"] = circular_lon_diff_deg(info["lon_mapped"], info["lon_360_orig"])
        info["dlat"] = abs(info["lat_mapped"] - info["lat_orig"])
        info["lon_min"] = float(lon_min[r, c]); info["lon_max"] = float(lon_max[r, c])
        info["lat_min"] = float(lat_min[r, c]); info["lat_max"] = float(lat_max[r, c])
        dlon_max = max(dlon_max, info["dlon_circ"])
        dlat_max = max(dlat_max, info["dlat"])

    np.savez_compressed(
        args.out_npz,
        lon=lon.astype(np.float32),
        lat=lat.astype(np.float32),
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        station_mask=station_mask,
        station_index_grid=station_index_grid,
        station_id_grid=station_id_grid,
        station_ids=np.array([s[0] for s in stations], dtype=object),
        half=np.array([half], dtype=np.float32),
    )
    out_json_path = Path(args.out_json)
    temp_json_path = out_json_path.with_suffix(".tmp")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    temp_json_path.replace(out_json_path)

    print("✅ Done")
    print(f"Stations: {N}, grid: {H}x{W}, empty cells: {H*W-N}")
    print(f"Max station Δlon (circular): {dlon_max}")
    print(f"Max station Δlat: {dlat_max}")


if __name__ == "__main__":
    main()
