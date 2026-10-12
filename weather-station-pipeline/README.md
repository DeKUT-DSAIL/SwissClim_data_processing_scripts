# Weather Station Data Processing Pipeline

A comprehensive pipeline for processing global weather station data into machine learning-ready datasets with proper train/holdout splits and geographic distribution.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This pipeline processes raw weather station NetCDF data into a clean, normalized dataset suitable for machine learning models. It handles:

- **Data Download**: Automated downloading of weather station observations
- **Temporal Aggregation**: Building hourly grids for each year
- **Data Merging**: Combining multi-year data into single dataset
- **Normalization**: Global statistics-based normalization
- **Train/Holdout Split**: Geographic stratification for proper validation
- **Efficient Storage**: Hybrid NPY+NPZ format for memory-mapped access

## ✨ Features

- **Geographic Stratification**: Ensures even global distribution of holdout stations
- **Scalable Processing**: Handles 25+ years of hourly data (~220k timesteps)
- **Memory Efficient**: Memory-mapped data access for large datasets (>100GB)
- **Reproducible**: Fixed random seeds and documented normalization stats
- **Flexible**: Configurable coverage thresholds, grid resolutions, and sampling methods

## 🚀 Installation

### Requirements

- Python 3.8+
- 100GB+ free disk space
- 16GB+ RAM (32GB+ recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SwissClim_data_processing_scripts.git
cd SwissClim_data_processing_scripts/weather-station-pipeline

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

Run the complete pipeline with default settings:

```bash
# 0. (Optional) Build station grid mapping from metadata
# Only needed if you want to regenerate the mapping or use different grid resolution
python scripts/00_build_station_grid_from_metadata.py \
    --input data/meta_info.json \
    --H 90 --W 180 \
    --out_npz data/final_results_station_grid_90x180.npz \
    --out_json data/final_results_mapping_90x180.json

# 1. Download raw data (requires credentials)
python scripts/01_download_data.py

# 2. Build yearly data cubes
python scripts/02_build_station_data_yearly.py \
    --nc_folder station_data/raw_data \
    --mapping_json final_results_mapping_90x180.json \
    --stats_json weather_data_norm_stats.json \
    --grid_npz final_results_station_grid_90x180.npz \
    --out_prefix weather_data \
    --workers 48

# 3. Merge all years into single file
python scripts/03_merge_yearly_npz.py \
    --input_pattern "weather_data_*.npz" \
    --output_file weather_data_all.npz

# 4. Convert to hybrid format (NPY + NPZ)
python scripts/04_convert_npz_to_hybrid.py

# 5. Select holdout stations (1000 stations, 40% min coverage, uniform geographic)
python scripts/05_select_holdout_stations.py \
    --coverage_csv station_coverage_2023.csv \
    --mapping_json final_results_mapping_90x180.json \
    --n_holdout 1000 \
    --min_coverage 40.0 \
    --out_json final_holdout_mapping_90x180.json \
    --sampling_method uniform

# 6. Create holdout mask
python scripts/06_create_holdout_mask.py \
    --holdout_json final_holdout_mapping_90x180.json \
    --out_npz holdout_mask.npz

# 7. Split dataset into training and holdout
python scripts/07_split_dataset_holdout.py \
    --data_npy stations_11k_var9_data.npy \
    --meta_npz stations_11k_var9_meta.npz \
    --holdout_mask holdout_mask.npz \
    --out_dir ESFM_Dataset
```

## 📊 Pipeline Steps

### Step 0: Build Station Grid Mapping (Optional/Prerequisite)

Generates grid mapping from raw station metadata. **This step is optional** - the repository already includes pre-generated mapping files in `data/`.

**Script**: `00_build_station_grid_from_metadata.py`

**Purpose**:
- Converts station coordinates to 2D grid positions
- Creates strictly monotonic lat/lon grids
- Generates station-to-grid mapping

**Inputs**:
- `data/meta_info.json` - Station metadata with lat/lon coordinates

**Outputs**:
- `final_results_mapping_90x180.json` - Station-to-grid mapping
- `final_results_station_grid_90x180.npz` - Grid coordinates and masks

**When to run**:
- Never needed if using provided metadata
- Only if regenerating with different grid resolution (e.g., 180×360)
- Only if updating station list

**Configuration**:
```bash
--input        # Input metadata JSON (default: data/meta_info.json)
--H            # Grid height (default: 90)
--W            # Grid width (default: 180)
--half         # Grid cell half-width in degrees (default: 0.005)
--out_npz      # Output grid NPZ file
--out_json     # Output mapping JSON file
```

---

### Step 1: Data Download

Downloads raw weather station observations from the data source.

**Script**: `01_download_data.py`

**Inputs**:
- Data source credentials
- Date range

**Outputs**:
- `station_data/raw_data/*.nc` - NetCDF files with station observations

---

### Step 2: Build Yearly Data

Processes raw NetCDF files into yearly data cubes with hourly resolution.

**Script**: `02_build_station_data_yearly.py`

**Key Features**:
- Hourly temporal resolution
- Nearest-hour snapping with configurable time threshold
- U/V wind component calculation
- Time shift tracking
- Skip existing years automatically

**Inputs**:
- Raw NetCDF files (`station_*.nc`)
- Station mapping JSON
- Global normalization stats JSON
- Grid coordinates NPZ

**Outputs**:
- `weather_data_YYYY.npz` for each year
- Shape: `[T_year, V, 90, 180]`

**Configuration**:
```bash
--nc_folder        # Directory with NetCDF files
--time_threshold   # Max minutes from nearest hour (default: 15)
--batch_size       # Files per batch (default: 200)
--workers          # Parallel workers (default: 8)
--use_memmap       # Use memory-mapped arrays
--start_year       # Optional: first year to process
--end_year         # Optional: last year to process
```

---

### Step 3: Merge Yearly Data

Combines all yearly NPZ files into a single multi-year dataset.

**Script**: `03_merge_yearly_npz.py`

**Inputs**:
- `weather_data_*.npz` - Yearly data files

**Outputs**:
- `weather_data_all.npz` - Combined dataset
- Shape: `[T_total, V, 90, 180]`

**Features**:
- Validates consistent shapes and variables
- Sorts years chronologically
- Concatenates timestamps
- Preserves normalization stats

---

### Step 4: Convert to Hybrid Format

Converts compressed NPZ to hybrid NPY+NPZ format for efficient memory mapping.

**Script**: `04_convert_npz_to_hybrid.py`

**Inputs**:
- `weather_data_all.npz`

**Outputs**:
- `stations_11k_var9_data.npy` - Large data array (memory-mappable)
- `stations_11k_var9_meta.npz` - Metadata (timestamps, variables, etc.)

**Benefits**:
- Fast memory-mapped access
- No decompression overhead
- Efficient for training large models

---

### Step 5: Select Holdout Stations

Selects validation stations with uniform geographic distribution.

**Script**: `05_select_holdout_stations.py`

**Sampling Methods**:
- **`uniform`** (recommended): Equal sampling from geographic regions
- **`stratified`**: Proportional sampling by region size
- **`random`**: Simple random sampling

**Inputs**:
- `station_coverage_2023.csv` - Coverage statistics per station
- `final_results_mapping_90x180.json` - Station mapping

**Outputs**:
- `final_holdout_mapping_90x180.json` - Holdout station mapping
- `final_holdout_mapping_90x180_list.csv` - Human-readable list

**Configuration**:
```bash
--n_holdout        # Number of holdout stations (default: 500)
--min_coverage     # Minimum coverage % (default: 70.0)
--sampling_method  # uniform/stratified/random (default: stratified)
--random_state     # Random seed (default: 42)
```

**Recommended Settings**:
```bash
# For best geographic distribution:
--n_holdout 1000 --min_coverage 40.0 --sampling_method uniform
```

---

### Step 6: Create Holdout Mask

Generates binary masks for efficient train/holdout separation.

**Script**: `06_create_holdout_mask.py`

**Inputs**:
- `final_holdout_mapping_90x180.json`

**Outputs**:
- `holdout_mask.npz` containing:
  - `holdout_mask`: Boolean [90, 180] - True for holdout stations
  - `training_mask`: Boolean [90, 180] - True for training stations
  - `n_holdout`, `n_training`: Station counts
  - `grid_shape`: Grid dimensions

---

### Step 7: Split Dataset

Splits data into separate training and holdout files with masked stations.

**Script**: `07_split_dataset_holdout.py`

**Inputs**:
- `stations_11k_var9_data.npy`
- `stations_11k_var9_meta.npz`
- `holdout_mask.npz`

**Outputs**:
- `ESFM_Dataset/stations_11k_data_training.npy` - Training data (holdout stations = NaN)
- `ESFM_Dataset/stations_11k_meta_training.npz` - Training metadata
- `ESFM_Dataset/stations_11k_data_holdout.npy` - Holdout data (training stations = NaN)
- `ESFM_Dataset/stations_11k_meta_holdout.npz` - Holdout metadata

**Features**:
- Preserves spatial grid structure
- Masks invalid stations with NaN
- Includes masks in metadata for easy filtering

---

## 📁 Included Metadata

The repository includes essential metadata files in the `data/` directory:

| File | Size | Description |
|------|------|-------------|
| `meta_info.json` | 1.3 MB | Source station coordinates (~11,800 stations) |
| `final_results_mapping_90x180.json` | 4.3 MB | Station-to-grid mapping for 90×180 grid |
| `final_results_station_grid_90x180.npz` | 409 KB | Grid coordinates and station masks |
| `weather_data_norm_stats.json` | 1.4 KB | Global normalization statistics |
| `final_holdout_mapping_90x180.json` | 429 KB | Validation station mapping (1000 stations) |

**Total size**: ~7 MB

These files are version-controlled because they are:
- Essential for reproducibility
- Small enough for git
- Time-consuming to regenerate
- Shared across all pipeline runs

See `data/README.md` for detailed documentation of each file.

---

## 📁 Data Format

### Variables

The dataset includes 9 weather variables:

| Variable | Short Name | Unit | Description |
|----------|------------|------|-------------|
| Air Pressure | `pa` | Pa | Surface air pressure |
| Sea Level Pressure | `msl` | Pa | Pressure at mean sea level |
| Air Temperature | `2t` | K | Temperature at 2m height |
| Dew Point Temperature | `dt` | K | Dew point at 2m height |
| U Wind Component | `10u` | m/s | Eastward wind at 10m |
| V Wind Component | `10v` | m/s | Northward wind at 10m |
| Wind Direction | `wd` | degrees | Wind from direction |
| Wind Speed | `ws` | m/s | Wind speed at 10m |
| Time Shift | `ts` | minutes | Observation time offset |

### Data Shape

- **Temporal**: ~219,168 hours (25 years of hourly data)
- **Spatial**: 90 × 180 grid (2° × 2° resolution)
- **Variables**: 9 weather variables
- **Total shape**: `[219168, 9, 90, 180]`

### Normalization

Data is normalized using global statistics (locations and scales):

```python
normalized_value = (raw_value - location) / scale
```

Statistics are stored in metadata for denormalization:
```python
meta = np.load('stations_11k_var9_meta.npz')
locations = meta['locations']  # [9] - per variable
scales = meta['scales']         # [9] - per variable
```

## ⚙️ Configuration

### Coverage Threshold Selection

| Threshold | Stations Available | Use Case |
|-----------|-------------------|----------|
| 70% | ~3,200 | High-quality validation |
| 50% | ~4,100 | Balanced quality/coverage |
| 40% | ~4,800 | Maximum geographic spread |
| 30% | ~5,500 | Including sparse regions |

### Sampling Method Selection

| Method | Best For | Description |
|--------|----------|-------------|
| `uniform` | Global models | Equal stations per geographic region |
| `stratified` | Regional models | Proportional to station density |
| `random` | Quick testing | Simple random selection |

## 📤 Output Structure

```
ESFM_Dataset/
├── stations_11k_data_training.npy      # Training data [219168, 9, 90, 180]
├── stations_11k_meta_training.npz      # Training metadata
├── stations_11k_data_holdout.npy       # Holdout data [219168, 9, 90, 180]
└── stations_11k_meta_holdout.npz       # Holdout metadata

holdout_mask.npz                         # Train/holdout masks
final_holdout_mapping_90x180.json       # Holdout station list
coverage_plots/
└── holdout_distribution.png            # Visualization
```

## 💻 Usage Examples

### Loading Training Data

```python
import numpy as np

# Load training data (memory-mapped)
train_data = np.load('ESFM_Dataset/stations_11k_data_training.npy', mmap_mode='r')
train_meta = np.load('ESFM_Dataset/stations_11k_meta_training.npz', allow_pickle=True)

# Access metadata
timestamps = train_meta['timestamps']
variables = train_meta['variables']
training_mask = train_meta['training_mask']

# Extract only valid training stations
valid_train = train_data[:, :, training_mask]  # [T, V, 15200]
```

### Loading Holdout Data

```python
# Load holdout data
holdout_data = np.load('ESFM_Dataset/stations_11k_data_holdout.npy', mmap_mode='r')
holdout_meta = np.load('ESFM_Dataset/stations_11k_meta_holdout.npz', allow_pickle=True)

# Extract valid holdout stations
holdout_mask = holdout_meta['holdout_mask']
valid_holdout = holdout_data[:, :, holdout_mask]  # [T, V, 1000]
```

### Denormalization

```python
# Load normalization stats
locations = train_meta['locations']  # [9]
scales = train_meta['scales']        # [9]

# Denormalize data
denormalized = train_data * scales[np.newaxis, :, np.newaxis, np.newaxis] + \
               locations[np.newaxis, :, np.newaxis, np.newaxis]
```

### Working with Specific Variables

```python
# Get variable indices
variables = train_meta['variables']
temp_idx = np.where(variables == '2t')[0][0]
wind_u_idx = np.where(variables == '10u')[0][0]

# Extract temperature time series for a specific location
temperature = train_data[:, temp_idx, 45, 90]  # [T]

# Extract wind components
u_wind = train_data[:, wind_u_idx, :, :]  # [T, 90, 180]
```

## 📈 Statistics

### Dataset Coverage

- **Total Stations**: 11,863
- **Training Stations**: 15,200 grid cells (93.83%)
- **Holdout Stations**: 1,000 grid cells (6.17%)
- **Time Period**: 2000-2024 (25 years)
- **Temporal Resolution**: Hourly
- **Spatial Resolution**: 2° × 2° (90 × 180 grid)

### Holdout Station Distribution (Uniform Sampling)

- **Mean Coverage**: 82.2%
- **Coverage Range**: 40.1% - 99.0%
- **Geographic Regions**: 478
- **Stations per Region**: ~2 (std=1.05)
- **Northern Hemisphere**: 75.6%
- **Southern Hemisphere**: 24.4%
- **Latitude Range**: -80.37° to 78.25°

## 🔧 Troubleshooting

### Out of Memory

If you encounter memory issues:
- Use `--use_memmap` flag in step 2
- Reduce `--batch_size` in step 2
- Process years in smaller ranges using `--start_year` and `--end_year`

### Disk Space

Full pipeline requires ~240GB disk space:
- Raw data: ~20GB
- Yearly NPZ: ~15GB
- Merged NPZ: ~17GB
- Hybrid format: ~120GB
- Train/holdout splits: ~240GB

### Performance

To speed up processing:
- Increase `--workers` (max: CPU cores)
- Increase `--batch_size` (if memory permits)
- Use SSD storage for temporary files

---

## TAHMO African Weather Station Pipeline

For regional fine-tuning of ESFM with African weather stations (TAHMO):

### Overview
- **Raw Data**: 5-minute multi-sensor Zarr store (`tahmo_lazy.zarr`).
- **Spatial Gridding**: Augmented 12.8k grid (`augmented_station_grid_90x180.npz`) combining 11k global + 925 African stations.
- **Quality Control**: 4-stage SwissClim QC (physical range gating, collocated multi-sensor fusion, +/-15 min temporal snapping, barometric MSL and Magnus-Tetens dew point derivations).

### Pipeline Execution via `run_pipeline.sh` (Recommended)

You can run the end-to-end TAHMO pipeline using the unified `./run_pipeline.sh` script:

```bash
# Run the complete TAHMO pipeline (Augment -> Build Train & Holdout -> Verify)
./run_pipeline.sh --tahmo

# Build only training split
./run_pipeline.sh --tahmo --split train

# Build only holdout split
./run_pipeline.sh --tahmo --split holdout

# Skip grid augmentation if already generated in data/
./run_pipeline.sh --tahmo --skip-augment

# Custom batch size or config
./run_pipeline.sh --tahmo --batch-size 100 --config config/tahmo_config.json
```

Or submit as a batch job on CSCS Alps/Daint via SLURM:
```bash
sbatch submit_tahmo_pipeline.sbatch
```

### Manual Individual Step Execution
```bash
# 1. Augment spatial grid and generate mapping (Algorithm 1)
python3 scripts/TAHMO/02_00_augment_tahmo.py \
    --meta_11k data/meta_info.json \
    --zarr_path /users/tligawa/ritom_scratch/ai-4-weather/data/tahmo_lazy.zarr \
    --out_dir data/

# 2. Build training & validation dataset
python3 scripts/TAHMO/02_build_tahmo_dataset.py --split train

# 3. Build holdout test dataset
python3 scripts/TAHMO/02_build_tahmo_dataset.py --split holdout

# 4. Verify dataset compliance and physical distributions
python3 scripts/TAHMO/verify_tahmo_dataset.py \
    --data_npy /ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_training_data.npy \
    --meta_npz /ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_training_meta.npz
```


