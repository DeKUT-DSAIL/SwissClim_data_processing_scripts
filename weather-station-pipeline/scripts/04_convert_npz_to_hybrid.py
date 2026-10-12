#!/usr/bin/env python3
"""
Convert NPZ file to hybrid format for efficient memory mapping.

Creates:
  1. A single large .npy file for the data array (memory-mappable)
  2. A small .npz file for metadata (timestamps, variables, etc.)

This allows the large data array to be properly memory-mapped while
keeping metadata easily accessible.
"""

import numpy as np
import os

def convert_to_hybrid_format(input_npz: str, output_data_npy: str, output_meta_npz: str):
    """
    Convert NPZ to hybrid NPY+NPZ format.

    Args:
        input_npz: Path to input .npz file
        output_data_npy: Path for output data .npy file
        output_meta_npz: Path for output metadata .npz file
    """
    print("=" * 80)
    print("Converting to Hybrid Format (NPY + NPZ)")
    print("=" * 80)

    print(f"\nInput:  {input_npz}")
    print(f"Output: {output_data_npy} (data array)")
    print(f"        {output_meta_npz} (metadata)")

    # Load input NPZ
    print("\nLoading input NPZ...")
    npz = np.load(input_npz, allow_pickle=True, mmap_mode='r')

    print(f"Found {len(npz.files)} arrays in NPZ:")
    for key in npz.files:
        arr = npz[key]
        if hasattr(arr, 'shape') and hasattr(arr, 'dtype'):
            size_mb = arr.nbytes / (1024**2) if hasattr(arr, 'nbytes') else 0
            print(f"  {key:15s}: shape={str(arr.shape):30s} dtype={str(arr.dtype):10s} size={size_mb:10.1f} MB")
        else:
            print(f"  {key:15s}: {type(arr)}")

    # Extract the large data array
    print(f"\n1. Extracting 'data' array to {output_data_npy}...")
    data = npz["data"]
    print(f"   Shape: {data.shape}")
    print(f"   Dtype: {data.dtype}")
    print(f"   Size:  {data.nbytes / (1024**3):.2f} GB")

    print("   Saving (this will take a while)...")
    np.save(output_data_npy, data, allow_pickle=False)

    saved_size_gb = os.path.getsize(output_data_npy) / (1024**3)
    print(f"   ✅ Saved: {saved_size_gb:.2f} GB")

    # Save metadata to small NPZ
    print(f"\n2. Saving metadata to {output_meta_npz}...")
    metadata_keys = [k for k in npz.files if k != "data"]

    metadata_dict = {}
    for key in metadata_keys:
        metadata_dict[key] = npz[key]
        print(f"   Including: {key}")

    np.savez(output_meta_npz, **metadata_dict)

    meta_size_mb = os.path.getsize(output_meta_npz) / (1024**2)
    print(f"   ✅ Saved: {meta_size_mb:.2f} MB")

    npz.close()

    print("\n" + "=" * 80)
    print("✅ Conversion Complete!")
    print("=" * 80)
    print(f"\nData file:     {output_data_npy} ({saved_size_gb:.2f} GB)")
    print(f"Metadata file: {output_meta_npz} ({meta_size_mb:.2f} MB)")
    print(f"\nTotal size:    {saved_size_gb:.2f} GB + {meta_size_mb:.2f} MB")
    print("\nThe data.npy file can now be efficiently memory-mapped!")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Convert NPZ file to hybrid NPY+NPZ format.")
    parser.add_argument("--input_npz", default="weather_data_all.npz", help="Input NPZ file")
    parser.add_argument("--output_data_npy", default="stations_11k_var9_data.npy", help="Output data NPY file")
    parser.add_argument("--output_meta_npz", default="stations_11k_var9_meta.npz", help="Output metadata NPZ file")
    parser.add_argument("--force", "-y", action="store_true", help="Force overwrite without confirmation")
    args = parser.parse_args()

    input_npz = args.input_npz
    output_data_npy = args.output_data_npy
    output_meta_npz = args.output_meta_npz

    # Check input exists
    if not os.path.exists(input_npz):
        print(f"❌ Error: Input file not found: {input_npz}")
        sys.exit(1)

    # Check if outputs exist
    if os.path.exists(output_data_npy) or os.path.exists(output_meta_npz):
        if not args.force and sys.stdin.isatty():
            print(f"\n⚠️  Warning: Output files already exist:")
            if os.path.exists(output_data_npy):
                print(f"   {output_data_npy}")
            if os.path.exists(output_meta_npz):
                print(f"   {output_meta_npz}")
            response = input("\nOverwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        else:
            print("Note: Overwriting existing output files (--force enabled or non-interactive environment).")

    # Run conversion
    convert_to_hybrid_format(input_npz, output_data_npy, output_meta_npz)
