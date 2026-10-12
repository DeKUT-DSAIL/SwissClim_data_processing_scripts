#!/usr/bin/env python3
"""
merge_yearly_npz.py
-------------------
Merge multiple yearly NPZ files (e.g. weather_data_2001.npz, weather_data_2002.npz, ...)
into a single NPZ file by concatenating along the time dimension.

Assumptions:
    - Each input NPZ has keys:
        data:       [T_year, V, H, W]
        timestamps: [T_year] datetime64[ns]
        variables:  [V]
        mapping:    station mapping table (structured array)
        lon_grid:   [H, W]
        lat_grid:   [H, W]
        locations:  [V]
        scales:     [V]
    - All non-time metadata (variables, mapping, lon_grid, lat_grid, locations, scales)
      are identical across NPZ files.

The script:
    1. Finds all NPZs matching a pattern (e.g. "weather_data_*.npz")
    2. Sorts them by year (parsed from filename like weather_data_YYYY.npz)
    3. First pass:
        - Checks consistency of metadata
        - Computes total number of timesteps T_total
    4. Second pass:
        - Allocates a memmap for data [T_total, V, H, W]
        - Allocates timestamps [T_total]
        - Sequentially copies each file's data/timestamps into the big arrays
    5. Saves a single compressed NPZ with the same structure.

Usage:
    python merge_yearly_npz.py \
        --input_pattern "weather_data_*.npz" \
        --out_file weather_data_all.npz
"""

import argparse
import glob
import os
import re
import tempfile

import numpy as np


# ----------------------------------------------------------------------
# Helper: extract year from filenames like "weather_data_2001.npz"
# ----------------------------------------------------------------------
def extract_year(path):
    """
    Try to extract a 4-digit year from filename. Assumes pattern like *_YYYY.npz
    or weather_data_YYYY.npz.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    # Find last 4 consecutive digits
    m = re.findall(r"(\d{4})", name)
    if not m:
        raise ValueError(f"Cannot extract year from filename: {path}")
    return int(m[-1])


# ----------------------------------------------------------------------
# Main merging function
# ----------------------------------------------------------------------
def merge_yearly_npz(input_pattern, out_file):
    print("=" * 80)
    print("MERGING YEARLY NPZ FILES")
    print("=" * 80)
    print(f"\nInput pattern: {input_pattern}")
    print(f"Output file:   {out_file}")

    # ------------------------------------------------------------------
    # 1. Discover files and sort by year
    # ------------------------------------------------------------------
    files = sorted(glob.glob(input_pattern))
    if not files:
        raise FileNotFoundError(f"No NPZ files matching pattern: {input_pattern}")

    # Exclude output file if it happens to match the input pattern (e.g. weather_data_all.npz)
    out_file_abs = os.path.abspath(out_file)
    files = [f for f in files if os.path.abspath(f) != out_file_abs]

    files_with_year = []
    for f in files:
        try:
            y = extract_year(f)
            files_with_year.append((f, y))
        except ValueError:
            print(f"  Skipping non-yearly NPZ file: {f}")

    if not files_with_year:
        raise ValueError(f"No valid yearly NPZ files found matching pattern: {input_pattern}")

    files_with_year.sort(key=lambda x: x[1])

    print("\nFiles to merge (in order):")
    for f, y in files_with_year:
        print(f"  {y}: {f}")

    # ------------------------------------------------------------------
    # 2. First pass: check metadata, compute total T
    # ------------------------------------------------------------------
    print("\nFirst pass: checking metadata and computing total length...")

    base_meta = {}
    total_T = 0
    per_file_T = []

    for idx, (f, y) in enumerate(files_with_year):
        print(f"  Reading header from: {f}")
        with np.load(f, allow_pickle=True) as npz:
            data = npz["data"]  # shape [T, V, H, W]
            timestamps = npz["timestamps"]
            variables = npz["variables"]
            mapping = npz["mapping"]
            lon_grid = npz["lon_grid"]
            lat_grid = npz["lat_grid"]
            locations = npz["locations"]
            scales = npz["scales"]

        T, V, H, W = data.shape
        print(f"    shape(data) = {data.shape}, timestamps = {len(timestamps)}")

        if T != len(timestamps):
            raise ValueError(f"{f}: data.shape[0] != len(timestamps)")

        per_file_T.append(T)
        total_T += T

        if idx == 0:
            # Save base metadata to check consistency
            base_meta["variables"] = variables
            base_meta["mapping"] = mapping
            base_meta["lon_grid"] = lon_grid
            base_meta["lat_grid"] = lat_grid
            base_meta["locations"] = locations
            base_meta["scales"] = scales
            base_meta["V"] = V
            base_meta["H"] = H
            base_meta["W"] = W
        else:
            # Check shape consistency
            if V != base_meta["V"] or H != base_meta["H"] or W != base_meta["W"]:
                raise ValueError(
                    f"{f}: data shape mismatch. "
                    f"Expected (*, {base_meta['V']}, {base_meta['H']}, {base_meta['W']}), "
                    f"got (*, {V}, {H}, {W})"
                )

            # Check variables consistency
            if not np.array_equal(variables, base_meta["variables"]):
                raise ValueError(f"{f}: variables do not match the first file")

            # Check mapping consistency
            if not np.array_equal(mapping, base_meta["mapping"]):
                raise ValueError(f"{f}: mapping does not match the first file")

            # Check grid and stats consistency (optional but safer)
            if not np.array_equal(lon_grid, base_meta["lon_grid"]):
                raise ValueError(f"{f}: lon_grid does not match the first file")
            if not np.array_equal(lat_grid, base_meta["lat_grid"]):
                raise ValueError(f"{f}: lat_grid does not match the first file")
            if not np.array_equal(locations, base_meta["locations"]):
                raise ValueError(f"{f}: locations do not match the first file")
            if not np.array_equal(scales, base_meta["scales"]):
                raise ValueError(f"{f}: scales do not match the first file")

    print(f"\nTotal timesteps across all files: {total_T}")
    print(
        f"Data shape will be: ({total_T}, {base_meta['V']}, "
        f"{base_meta['H']}, {base_meta['W']})"
    )

    # ------------------------------------------------------------------
    # 3. Allocate big arrays (use memmap for data)
    # ------------------------------------------------------------------
    print("\nAllocating big arrays...")

    out_dir = os.path.dirname(out_file)
    tmp_data_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat", dir=out_dir if out_dir else None)
    tmp_data_file.close()
    print(f"  Using memmap for data at: {tmp_data_file.name}")

    big_data = np.memmap(
        tmp_data_file.name,
        dtype=data.dtype,
        mode="w+",
        shape=(total_T, base_meta["V"], base_meta["H"], base_meta["W"]),
    )
    big_data[:] = np.nan

    big_timestamps = np.empty(total_T, dtype=timestamps.dtype)

    # ------------------------------------------------------------------
    # 4. Second pass: copy data & timestamps
    # ------------------------------------------------------------------
    print("\nSecond pass: copying data and timestamps...")

    offset = 0
    for (f, y), T in zip(files_with_year, per_file_T):
        print(f"  Copying from {f} (year {y}, T={T}) at offset {offset}")
        with np.load(f, allow_pickle=True, mmap_mode="r") as npz:
            d = npz["data"]         # [T, V, H, W]
            ts = npz["timestamps"]  # [T]
            big_data[offset:offset + T] = d
            big_timestamps[offset:offset + T] = ts

        offset += T

    big_data.flush()

    # ------------------------------------------------------------------
    # 5. Save merged NPZ
    # ------------------------------------------------------------------
    print(f"\nSaving merged NPZ to {out_file}...")

    np.savez_compressed(
        out_file,
        data=big_data,
        timestamps=big_timestamps,
        variables=base_meta["variables"],
        mapping=base_meta["mapping"],
        lon_grid=base_meta["lon_grid"],
        lat_grid=base_meta["lat_grid"],
        locations=base_meta["locations"],
        scales=base_meta["scales"],
    )

    # Clean up memmap file
    data_shape = big_data.shape
    del big_data
    if os.path.exists(tmp_data_file.name):
        os.unlink(tmp_data_file.name)

    print("\n✅ DONE!")
    print(f"  Merged data shape: {data_shape}")
    print(f"  Total timesteps:   {total_T}")
    print(f"  Output file:       {out_file}")
    print("=" * 80)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge yearly station NPZ files into one big NPZ."
    )
    parser.add_argument(
        "--input_pattern",
        required=True,
        help='Glob pattern for input files, e.g. "weather_data_*.npz"',
    )
    parser.add_argument(
        "--out_file",
        required=True,
        help="Output merged NPZ file, e.g. weather_data_all.npz",
    )

    args = parser.parse_args()
    merge_yearly_npz(args.input_pattern, args.out_file)

