#!/usr/bin/env python3
"""
select_holdout_stations.py
--------------------------
Select hold-out stations based on:
1. High data coverage (for reliable validation)
2. Geographic distribution (representative sampling)

Uses spatial stratification to ensure even geographic distribution.

Usage:
    python select_holdout_stations.py \
        --coverage_csv station_coverage_2023.csv \
        --mapping_json final_results_mapping_90x180.json \
        --n_holdout 500 \
        --min_coverage 70.0 \
        --out_json final_holdout_mapping_90x180.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans


def uniform_geographic_sampling(df, n_samples, min_coverage, random_state=42, n_regions=None):
    """
    Select stations with UNIFORM geographic distribution.

    Strategy:
    1. Filter stations by minimum coverage threshold
    2. Divide the region into geographic zones using K-means clustering
    3. Sample EQUALLY from each region (not proportionally)
    4. This ensures even spatial distribution across the domain

    Returns:
        DataFrame with selected stations
    """
    print("\n" + "=" * 80)
    print("UNIFORM GEOGRAPHIC SAMPLING")
    print("=" * 80)

    # Step 1: Filter by coverage
    print(f"\n1. Filtering stations by coverage >= {min_coverage}%")
    df_filtered = df[df['all_var_valid_mean_pct'] >= min_coverage].copy()
    print(f"   Stations meeting threshold: {len(df_filtered):,} / {len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")

    if len(df_filtered) < n_samples:
        print(f"\n⚠️  WARNING: Only {len(df_filtered)} stations meet coverage threshold.")
        print(f"   Requested {n_samples} samples. Lowering coverage threshold...")

        df_sorted = df.sort_values('all_var_valid_mean_pct', ascending=False)
        new_threshold = df_sorted.iloc[n_samples - 1]['all_var_valid_mean_pct']
        print(f"   New threshold: {new_threshold:.1f}%")
        df_filtered = df[df['all_var_valid_mean_pct'] >= new_threshold].copy()
        print(f"   Stations after adjustment: {len(df_filtered):,}")

    # Step 2: Determine number of regions
    if n_regions is None or n_regions <= 0:
        # Default adaptive formula
        n_regions = max(5, min(n_samples // 2, len(df_filtered) // 10))
    n_regions = min(n_regions, n_samples, len(df_filtered))

    print(f"\n2. Clustering into {n_regions} geographic regions using K-means")

    # Use lat/lon for clustering
    coords = df_filtered[['lat_orig', 'lon_orig']].values

    # Handle missing coordinates
    valid_coords_mask = ~np.isnan(coords).any(axis=1)
    if not valid_coords_mask.all():
        print(f"   WARNING: {(~valid_coords_mask).sum()} stations have missing coordinates, excluding them")
        df_filtered = df_filtered[valid_coords_mask].copy()
        coords = coords[valid_coords_mask]

    # Normalize coordinates for better clustering (3D sphere representation)
    lat_rad = np.deg2rad(coords[:, 0])
    lon_rad = np.deg2rad(coords[:, 1])

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    coords_3d = np.column_stack([x, y, z])

    # K-means clustering
    kmeans = KMeans(n_clusters=n_regions, random_state=random_state, n_init=10)
    df_filtered['region'] = kmeans.fit_predict(coords_3d)

    # Step 3: Sample EQUALLY from each region
    print(f"\n3. Sampling UNIFORMLY from {n_regions} regions to get {n_samples} stations")

    # Count stations per region
    region_counts = df_filtered['region'].value_counts().sort_index()
    print(f"   Stations per region: min={region_counts.min()}, max={region_counts.max()}, mean={region_counts.mean():.1f}")

    # Calculate stations per region (uniform distribution)
    stations_per_region = n_samples // n_regions
    remaining = n_samples % n_regions

    print(f"   Target: {stations_per_region} stations per region (+{remaining} for largest regions)")

    # Sample uniformly from each region
    selected_stations = []
    regions_with_samples = []

    for region_id in region_counts.index:
        region_df = df_filtered[df_filtered['region'] == region_id].copy()

        # How many to sample from this region
        n_from_region = stations_per_region
        if len(regions_with_samples) < remaining:
            n_from_region += 1

        # Can't sample more than available
        n_from_region = min(n_from_region, len(region_df))

        if n_from_region > 0:
            # Sort by coverage and take top n (prioritize quality within region)
            region_df = region_df.sort_values('all_var_valid_mean_pct', ascending=False)
            selected = region_df.head(n_from_region)
            selected_stations.append(selected)
            regions_with_samples.append(region_id)

    result_df = pd.concat(selected_stations, ignore_index=True)

    # If we don't have enough due to small regions, sample more from larger regions
    if len(result_df) < n_samples:
        print(f"   Need {n_samples - len(result_df)} more stations...")
        # Get stations we haven't selected yet
        selected_ids = set(result_df['station_id'])
        remaining_df = df_filtered[~df_filtered['station_id'].isin(selected_ids)].copy()

        # Sort by coverage and take top
        remaining_df = remaining_df.sort_values('all_var_valid_mean_pct', ascending=False)
        extra = remaining_df.head(n_samples - len(result_df))
        result_df = pd.concat([result_df, extra], ignore_index=True)

    print(f"\n✅ Selected {len(result_df)} stations")
    print(f"   Coverage range: {result_df['all_var_valid_mean_pct'].min():.1f}% - {result_df['all_var_valid_mean_pct'].max():.1f}%")
    print(f"   Mean coverage: {result_df['all_var_valid_mean_pct'].mean():.1f}%")
    print(f"   Median coverage: {result_df['all_var_valid_mean_pct'].median():.1f}%")

    # Show distribution across regions
    region_dist = result_df['region'].value_counts().sort_index()
    print(f"   Stations per region: min={region_dist.min()}, max={region_dist.max()}, mean={region_dist.mean():.1f}, std={region_dist.std():.2f}")

    return result_df


def random_sampling(df, n_samples, min_coverage, random_state=42):
    """
    Select stations using simple random sampling.

    Strategy:
    1. Filter stations by minimum coverage threshold
    2. Randomly sample n_samples stations from the filtered set

    Returns:
        DataFrame with selected stations
    """
    print("\n" + "=" * 80)
    print("RANDOM SAMPLING")
    print("=" * 80)

    # Step 1: Filter by coverage
    print(f"\n1. Filtering stations by coverage >= {min_coverage}%")
    df_filtered = df[df['all_var_valid_mean_pct'] >= min_coverage].copy()
    print(f"   Stations meeting threshold: {len(df_filtered):,} / {len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")

    if len(df_filtered) < n_samples:
        print(f"\n⚠️  WARNING: Only {len(df_filtered)} stations meet coverage threshold.")
        print(f"   Requested {n_samples} samples. Lowering coverage threshold...")

        # Find the coverage threshold that gives us at least n_samples stations
        df_sorted = df.sort_values('all_var_valid_mean_pct', ascending=False)
        new_threshold = df_sorted.iloc[n_samples - 1]['all_var_valid_mean_pct']
        print(f"   New threshold: {new_threshold:.1f}%")
        df_filtered = df[df['all_var_valid_mean_pct'] >= new_threshold].copy()
        print(f"   Stations after adjustment: {len(df_filtered):,}")

    # Step 2: Random sampling
    print(f"\n2. Randomly sampling {n_samples} stations")
    result_df = df_filtered.sample(n=n_samples, random_state=random_state)

    print(f"\n✅ Selected {len(result_df)} stations")
    print(f"   Coverage range: {result_df['all_var_valid_mean_pct'].min():.1f}% - {result_df['all_var_valid_mean_pct'].max():.1f}%")
    print(f"   Mean coverage: {result_df['all_var_valid_mean_pct'].mean():.1f}%")
    print(f"   Median coverage: {result_df['all_var_valid_mean_pct'].median():.1f}%")

    return result_df


def geographic_stratified_sampling(df, n_samples, min_coverage, random_state=42):
    """
    Select stations using geographic stratification to ensure even distribution.

    Strategy:
    1. Filter stations by minimum coverage threshold
    2. Divide the world into geographic regions using K-means clustering
    3. Sample proportionally from each region (weighted by region size)
    4. Within each region, sample stations with highest coverage

    Returns:
        DataFrame with selected stations
    """
    print("\n" + "=" * 80)
    print("GEOGRAPHIC STRATIFIED SAMPLING")
    print("=" * 80)

    # Step 1: Filter by coverage
    print(f"\n1. Filtering stations by coverage >= {min_coverage}%")
    df_filtered = df[df['all_var_valid_mean_pct'] >= min_coverage].copy()
    print(f"   Stations meeting threshold: {len(df_filtered):,} / {len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")

    if len(df_filtered) < n_samples:
        print(f"\n⚠️  WARNING: Only {len(df_filtered)} stations meet coverage threshold.")
        print(f"   Requested {n_samples} samples. Lowering coverage threshold...")

        # Find the coverage threshold that gives us at least n_samples stations
        df_sorted = df.sort_values('all_var_valid_mean_pct', ascending=False)
        new_threshold = df_sorted.iloc[n_samples - 1]['all_var_valid_mean_pct']
        print(f"   New threshold: {new_threshold:.1f}%")
        df_filtered = df[df['all_var_valid_mean_pct'] >= new_threshold].copy()
        print(f"   Stations after adjustment: {len(df_filtered):,}")

    # Step 2: Cluster stations into geographic regions
    # Use more clusters than samples to ensure fine-grained distribution
    n_regions = min(n_samples, len(df_filtered) // 5)  # ~5 stations per region target
    n_regions = max(n_regions, 20)  # At least 20 regions for good distribution

    print(f"\n2. Clustering into {n_regions} geographic regions using K-means")

    # Use lat/lon for clustering
    coords = df_filtered[['lat_orig', 'lon_orig']].values

    # Handle missing coordinates
    valid_coords_mask = ~np.isnan(coords).any(axis=1)
    if not valid_coords_mask.all():
        print(f"   WARNING: {(~valid_coords_mask).sum()} stations have missing coordinates, excluding them")
        df_filtered = df_filtered[valid_coords_mask].copy()
        coords = coords[valid_coords_mask]

    # Normalize coordinates for better clustering
    # Use cosine transformation for longitude to handle wraparound
    lat_rad = np.deg2rad(coords[:, 0])
    lon_rad = np.deg2rad(coords[:, 1])

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    coords_3d = np.column_stack([x, y, z])

    # K-means clustering
    kmeans = KMeans(n_clusters=n_regions, random_state=random_state, n_init=10)
    df_filtered['region'] = kmeans.fit_predict(coords_3d)

    # Step 3: Sample from each region
    print(f"\n3. Sampling {n_samples} stations from {n_regions} regions")

    # Count stations per region
    region_counts = df_filtered['region'].value_counts().sort_index()
    print(f"   Stations per region: min={region_counts.min()}, max={region_counts.max()}, mean={region_counts.mean():.1f}")

    # Sample proportionally from each region, but ensure we get enough from each
    samples_per_region = {}
    remaining_samples = n_samples

    # First pass: allocate proportionally
    for region_id, count in region_counts.items():
        n_from_region = max(1, int(n_samples * count / len(df_filtered)))
        n_from_region = min(n_from_region, count)  # Can't sample more than available
        samples_per_region[region_id] = n_from_region
        remaining_samples -= n_from_region

    # Second pass: distribute remaining samples to largest regions
    while remaining_samples > 0:
        for region_id in region_counts.index:
            if remaining_samples <= 0:
                break
            if samples_per_region[region_id] < region_counts[region_id]:
                samples_per_region[region_id] += 1
                remaining_samples -= 1

    print(f"   Sampling distribution across regions:")
    print(f"     Min per region: {min(samples_per_region.values())}")
    print(f"     Max per region: {max(samples_per_region.values())}")
    print(f"     Mean per region: {np.mean(list(samples_per_region.values())):.1f}")

    # Sample from each region (prioritize highest coverage within each region)
    selected_stations = []
    for region_id, n_from_region in samples_per_region.items():
        region_df = df_filtered[df_filtered['region'] == region_id].copy()
        # Sort by coverage and take top n
        region_df = region_df.sort_values('all_var_valid_mean_pct', ascending=False)
        selected = region_df.head(n_from_region)
        selected_stations.append(selected)

    result_df = pd.concat(selected_stations, ignore_index=True)

    print(f"\n✅ Selected {len(result_df)} stations")
    print(f"   Coverage range: {result_df['all_var_valid_mean_pct'].min():.1f}% - {result_df['all_var_valid_mean_pct'].max():.1f}%")
    print(f"   Mean coverage: {result_df['all_var_valid_mean_pct'].mean():.1f}%")
    print(f"   Median coverage: {result_df['all_var_valid_mean_pct'].median():.1f}%")

    return result_df


def main(coverage_csv, mapping_json, n_holdout, min_coverage, out_json, random_state=42, sampling_method='stratified', n_regions=None):
    print("=" * 80)
    print("SELECTING HOLD-OUT STATIONS")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Coverage CSV:     {coverage_csv}")
    print(f"  Mapping JSON:     {mapping_json}")
    print(f"  N hold-out:       {n_holdout}")
    print(f"  Min coverage:     {min_coverage}%")
    print(f"  Output JSON:      {out_json}")
    print(f"  Random state:     {random_state}")
    print(f"  Sampling method:  {sampling_method}")
    if n_regions:
        print(f"  N regions:        {n_regions}")

    # Load coverage data
    import os
    if coverage_csv and os.path.exists(coverage_csv):
        print(f"\nLoading coverage data from {coverage_csv}...")
        coverage_df = pd.read_csv(coverage_csv)
        print(f"  Loaded {len(coverage_df)} stations")
    else:
        print(f"\nNo coverage CSV provided or found. Deriving candidate station list from mapping JSON: {mapping_json}...")
        with open(mapping_json, 'r') as f:
            mapping_raw = json.load(f)
        records = []
        for sid, d in mapping_raw.items():
            records.append({
                'station_id': sid,
                'row': d.get('row', -1),
                'col': d.get('col', -1),
                'lat_orig': d.get('lat_orig', np.nan),
                'lon_orig': d.get('lon_orig', np.nan),
                'all_var_valid_mean_pct': 100.0,
            })
        coverage_df = pd.DataFrame(records)
        print(f"  Loaded {len(coverage_df)} stations from mapping JSON (defaulting coverage to 100%)")

    # Select hold-out stations
    if sampling_method == 'random':
        holdout_df = random_sampling(
            coverage_df,
            n_samples=n_holdout,
            min_coverage=min_coverage,
            random_state=random_state
        )
    elif sampling_method == 'uniform':
        holdout_df = uniform_geographic_sampling(
            coverage_df,
            n_samples=n_holdout,
            min_coverage=min_coverage,
            random_state=random_state,
            n_regions=n_regions,
        )
    else:
        holdout_df = geographic_stratified_sampling(
            coverage_df,
            n_samples=n_holdout,
            min_coverage=min_coverage,
            random_state=random_state
        )

    # Load original mapping to preserve full metadata
    print("\nLoading original mapping...")
    with open(mapping_json, 'r') as f:
        mapping_data = json.load(f)

    # Create hold-out mapping
    print("\nCreating hold-out mapping...")
    holdout_mapping = {}

    for _, row in holdout_df.iterrows():
        station_id = row['station_id']
        if station_id in mapping_data:
            holdout_mapping[station_id] = mapping_data[station_id]

    print(f"  Hold-out mapping contains {len(holdout_mapping)} stations")

    # Save hold-out mapping atomically
    print(f"\nSaving hold-out mapping to {out_json}...")
    import tempfile
    import os
    tmp_path = f"{out_json}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(holdout_mapping, f, indent=2)
    os.replace(tmp_path, out_json)

    # Print statistics
    print("\n" + "=" * 80)
    print("HOLD-OUT STATION STATISTICS")
    print("=" * 80)

    print(f"\nTotal hold-out stations: {len(holdout_mapping)}")
    print(f"\nCoverage statistics:")
    print(f"  Mean:   {holdout_df['all_var_valid_mean_pct'].mean():.2f}%")
    print(f"  Median: {holdout_df['all_var_valid_mean_pct'].median():.2f}%")
    print(f"  Min:    {holdout_df['all_var_valid_mean_pct'].min():.2f}%")
    print(f"  Max:    {holdout_df['all_var_valid_mean_pct'].max():.2f}%")
    print(f"  Std:    {holdout_df['all_var_valid_mean_pct'].std():.2f}%")

    print(f"\nGeographic distribution:")
    print(f"  Latitude range:  {holdout_df['lat_orig'].min():.2f}° to {holdout_df['lat_orig'].max():.2f}°")
    print(f"  Longitude range: {holdout_df['lon_orig'].min():.2f}° to {holdout_df['lon_orig'].max():.2f}°")

    # Hemisphere distribution
    n_north = (holdout_df['lat_orig'] > 0).sum()
    n_south = (holdout_df['lat_orig'] <= 0).sum()
    print(f"\nHemisphere distribution:")
    print(f"  Northern: {n_north} ({n_north/len(holdout_df)*100:.1f}%)")
    print(f"  Southern: {n_south} ({n_south/len(holdout_df)*100:.1f}%)")

    # Enrich holdout_df with mapping metadata if row or col are missing
    if 'row' not in holdout_df.columns or 'col' not in holdout_df.columns:
        holdout_df['row'] = [mapping_data.get(sid, {}).get('row', -1) for sid in holdout_df['station_id']]
        holdout_df['col'] = [mapping_data.get(sid, {}).get('col', -1) for sid in holdout_df['station_id']]

    # Save hold-out station list for reference
    holdout_list_csv = out_json.replace('.json', '_list.csv')
    cols_to_save = [c for c in ['station_id', 'row', 'col', 'lat_orig', 'lon_orig', 'all_var_valid_mean_pct'] if c in holdout_df.columns]
    holdout_df[cols_to_save].to_csv(holdout_list_csv, index=False)
    print(f"\n✅ Saved hold-out station list to {holdout_list_csv}")

    print("\n" + "=" * 80)
    print("✅ DONE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select hold-out stations with high coverage and geographic distribution."
    )
    parser.add_argument("--coverage_csv", required=False, default=None,
                        help="Station coverage CSV file (optional; defaults to mapping_json if omitted)")
    parser.add_argument("--mapping_json", required=True,
                        help="Original station mapping JSON file")
    parser.add_argument("--n_holdout", type=int, default=500,
                        help="Number of hold-out stations to select")
    parser.add_argument("--min_coverage", type=float, default=70.0,
                        help="Minimum coverage percentage threshold")
    parser.add_argument("--out_json", required=True,
                        help="Output JSON file for hold-out mapping")
    parser.add_argument("--sampling_method", type=str, default='stratified',
                        choices=['random', 'stratified', 'uniform'],
                        help="Sampling method: 'random' for simple random sampling, 'stratified' for proportional geographic stratification, 'uniform' for equal sampling across geographic regions")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")
    parser.add_argument("--n_regions", type=int, default=None,
                        help="Number of geographic clusters/regions for K-Means (optional; defaults to adaptive count)")

    args = parser.parse_args()

    main(
        coverage_csv=args.coverage_csv,
        mapping_json=args.mapping_json,
        n_holdout=args.n_holdout,
        min_coverage=args.min_coverage,
        out_json=args.out_json,
        random_state=args.random_state,
        sampling_method=args.sampling_method,
        n_regions=args.n_regions,
    )
