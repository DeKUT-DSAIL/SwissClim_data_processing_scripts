#!/usr/bin/env python3
"""
create_holdout_mask.py
----------------------
Create a binary mask for hold-out stations that can be applied to the data cube.

The mask is a boolean array of shape [90, 180] where:
- True (1) = hold-out station
- False (0) = training station

Usage:
    python create_holdout_mask.py \
        --holdout_json final_holdout_mapping_90x180.json \
        --out_npz holdout_mask.npz

Example usage in code:
    import numpy as np

    # Load mask
    mask_data = np.load('holdout_mask.npz')
    holdout_mask = mask_data['holdout_mask']  # Shape: [90, 180]

    # Load data
    data = np.load('weather_data_2023.npz')
    cube = data['data']  # Shape: [T, V, 90, 180]

    # Extract hold-out data only
    holdout_data = cube[:, :, holdout_mask]  # Shape: [T, V, n_holdout_stations]

    # Extract training data only
    training_mask = ~holdout_mask
    training_data = cube[:, :, training_mask]  # Shape: [T, V, n_training_stations]

    # Or keep spatial structure and mask with NaN
    holdout_cube = cube.copy()
    holdout_cube[:, :, ~holdout_mask] = np.nan  # Set training stations to NaN
"""

import argparse
import json
import os
import numpy as np


def create_holdout_mask(holdout_json, mapping_json=None, grid_shape=(90, 180), out_npz=None):
    """
    Create a binary mask for hold-out stations.

    Args:
        holdout_json: Path to hold-out mapping JSON
        grid_shape: Shape of the grid (default: 90x180)
        out_npz: Output NPZ file path

    Returns:
        holdout_mask: Boolean array of shape [90, 180]
    """
    print("=" * 80)
    print("CREATING HOLD-OUT MASK")
    print("=" * 80)
    print(f"\nHold-out JSON: {holdout_json}")
    print(f"Grid shape:    {grid_shape}")
    if out_npz:
        print(f"Output NPZ:    {out_npz}")

    # Load hold-out mapping
    print("\nLoading hold-out stations...")
    with open(holdout_json, 'r') as f:
        holdout_mapping = json.load(f)

    print(f"  Found {len(holdout_mapping)} hold-out stations")

    # Create mask
    print("\nCreating mask...")
    holdout_mask = np.zeros(grid_shape, dtype=bool)

    # Mark hold-out stations
    for station_id, info in holdout_mapping.items():
        row = info['row']
        col = info['col']
        holdout_mask[row, col] = True

    n_holdout = np.sum(holdout_mask)
    total_grid_cells = grid_shape[0] * grid_shape[1]

    if mapping_json and os.path.exists(mapping_json):
        with open(mapping_json, 'r') as f:
            all_mapping = json.load(f)
        n_total_stations = len(all_mapping)
        n_training = n_total_stations - n_holdout
        n_empty_cells = total_grid_cells - n_total_stations

        print(f"  Total active stations:   {n_total_stations}")
        print(f"  Hold-out stations:       {n_holdout}")
        print(f"  True training stations:  {n_training}")
        print(f"  Empty background cells:  {n_empty_cells}")
        print(f"  Total grid canvas cells: {total_grid_cells}")
        print(f"  Hold-out station ratio:  {n_holdout / n_total_stations * 100:.2f}%")
    else:
        n_non_holdout_cells = total_grid_cells - n_holdout
        n_training = n_non_holdout_cells
        print(f"  Hold-out stations:       {n_holdout}")
        print(f"  Non-holdout grid cells:  {n_non_holdout_cells} (includes empty canvas cells)")
        print(f"  Total grid canvas cells: {total_grid_cells}")
        print(f"  Hold-out canvas ratio:   {n_holdout / total_grid_cells * 100:.2f}%")

    # Save mask
    if out_npz:
        print(f"\nSaving mask to {out_npz}...")
        np.savez_compressed(
            out_npz,
            holdout_mask=holdout_mask,
            training_mask=~holdout_mask,
            grid_shape=np.array(grid_shape),
            n_holdout=n_holdout,
            n_training=n_training,
        )
        print(f"  ✓ Saved")

    # Print usage examples
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)

    print("""
# Load the mask
import numpy as np
mask_data = np.load('holdout_mask.npz')
holdout_mask = mask_data['holdout_mask']  # Shape: [90, 180], dtype: bool

# Load your data cube
data = np.load('weather_data_2023.npz')
cube = data['data']  # Shape: [T, V, 90, 180]

# ========================================
# METHOD 1: Extract data while preserving time/variable dimensions
# ========================================

# Get hold-out data only (flattened spatial dimension)
holdout_data = cube[:, :, holdout_mask]
# Shape: [T, V, n_holdout] where n_holdout = """ + str(n_holdout) + """

# Get training data only (flattened spatial dimension)
training_mask = mask_data['training_mask']
training_data = cube[:, :, training_mask]
# Shape: [T, V, n_training] where n_training = """ + str(n_training) + """

# ========================================
# METHOD 2: Mask in-place (keep spatial structure)
# ========================================

# Create hold-out cube (set training stations to NaN)
holdout_cube = cube.copy()
holdout_cube[:, :, ~holdout_mask] = np.nan
# Shape: [T, V, 90, 180] - same as original, but training stations are NaN

# Create training cube (set hold-out stations to NaN)
training_cube = cube.copy()
training_cube[:, :, holdout_mask] = np.nan
# Shape: [T, V, 90, 180] - same as original, but hold-out stations are NaN

# ========================================
# METHOD 3: Get station indices for indexing
# ========================================

# Get row, col indices of hold-out stations
holdout_rows, holdout_cols = np.where(holdout_mask)
# holdout_rows, holdout_cols: 1D arrays of length """ + str(n_holdout) + """

# Extract data for specific stations
for i in range(len(holdout_rows)):
    row, col = holdout_rows[i], holdout_cols[i]
    station_timeseries = cube[:, :, row, col]  # Shape: [T, V]
    # Process station data...

# ========================================
# METHOD 4: Calculate metrics separately
# ========================================

# Calculate loss on hold-out set only
predictions = model_predict(...)  # Shape: [T, V, 90, 180]
targets = cube

holdout_predictions = predictions[:, :, holdout_mask]
holdout_targets = targets[:, :, holdout_mask]

holdout_mse = np.mean((holdout_predictions - holdout_targets) ** 2)
print(f"Hold-out MSE: {holdout_mse}")

# Calculate loss on training set only
training_predictions = predictions[:, :, ~holdout_mask]
training_targets = targets[:, :, ~holdout_mask]

training_mse = np.mean((training_predictions - training_targets) ** 2)
print(f"Training MSE: {training_mse}")
""")

    print("=" * 80)
    print("✅ DONE!")
    print("=" * 80)

    return holdout_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create binary mask for hold-out stations."
    )
    parser.add_argument("--holdout_json", required=True,
                        help="Hold-out mapping JSON file")
    parser.add_argument("--out_npz", required=True,
                        help="Output NPZ file for the mask")
    parser.add_argument("--mapping_json", default=None,
                        help="Optional full station mapping JSON to compute true training station counts")
    parser.add_argument("--grid_height", type=int, default=90,
                        help="Grid height (default: 90)")
    parser.add_argument("--grid_width", type=int, default=180,
                        help="Grid width (default: 180)")

    args = parser.parse_args()

    create_holdout_mask(
        holdout_json=args.holdout_json,
        mapping_json=args.mapping_json,
        grid_shape=(args.grid_height, args.grid_width),
        out_npz=args.out_npz,
    )
