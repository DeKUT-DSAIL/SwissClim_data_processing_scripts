#!/bin/bash
################################################################################
# Weather Station Data Processing Pipeline
#
# This script runs the complete data processing pipeline for:
#   1. Global 11k Weather Stations (NetCDF -> Hourly NPZ -> Train/Holdout Splits)
#   2. TAHMO African Weather Stations (Zarr -> Augmented Grid -> ML Datasets)
#
# Usage:
#   # Run standard Global pipeline
#   ./run_pipeline.sh [GLOBAL_OPTIONS]
#
#   # Run TAHMO African pipeline (host python environment)
#   ./run_pipeline.sh --tahmo [TAHMO_OPTIONS]
#
#   # Run TAHMO African pipeline inside CSCS container
#   ./run_pipeline.sh --tahmo --container [TAHMO_OPTIONS]
#
# Dataset Selection:
#   --dataset <global|tahmo>      Select dataset mode (default: global)
#   --tahmo                       Shorthand for --dataset tahmo
#   --global                      Shorthand for --dataset global
#
# Container Options (CSCS Alps / Pyxis):
#   --container                   Execute python steps inside the container
#   --container-image <path>      Path to container .sqsh file
#                                 (default: /ritom/scratch/cscs/tligawa/container-images/torch_physics_nemo_img.sqsh)
#   --container-env <path>        Path to Pyxis container environment .toml file
#                                 (default: /ritom/scratch/cscs/tligawa/container-images/torch_physics_nemo.toml)
#
# TAHMO Pipeline Options:
#   --split <train|holdout|both>  Split to generate (default: both)
#   --skip-qc                     Skip collocated sensor QC (Step 1)
#   --skip-augment                Skip grid augmentation (Step 2)
#   --skip-verify                 Skip post-build verification (Step 4)
#   --dry-run                     Run only 1 test block in QC
#   --config <path>               Path to config JSON (default: config/tahmo_config.json)
#   --zarr-path <path>            Path to raw tahmo_lazy.zarr store
#   --batch-size <N>              Batch size for station processing (default: 50)
#
# Global Pipeline Options:
#   --skip-download               Skip raw NetCDF data download step
#   --start-year YYYY             Process only years >= YYYY
#   --end-year YYYY               Process only years <= YYYY
#   --workers N                   Number of parallel workers (default: 48)
#   --batch-size N                Number of files per batch (default: 200)
#   --n-holdout N                 Number of holdout stations (default: 1000)
#   --min-coverage N              Minimum coverage percentage (default: 40.0)
#   --sampling-method <method>    Sampling method: uniform, stratified, random (default: uniform)
#
# General Options:
#   --help                        Show this help message
################################################################################

set -e  # Exit on error

# Ensure execution from pipeline directory
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
cd "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

# Mode
DATASET="global"

# Container configuration
USE_CONTAINER=false
CONTAINER_TOML="/ritom/scratch/cscs/tligawa/container-images/torch_physics_nemo.toml"
CONTAINER_IMAGE="/ritom/scratch/cscs/tligawa/container-images/torch_physics_nemo_img.sqsh"
if [ ! -f "$CONTAINER_IMAGE" ]; then
  CONTAINER_IMAGE="/ritom/scratch/cscs/tligawa/container-images/torch_physicsnemo_img.sqsh"
fi

# Global default values
SKIP_DOWNLOAD=True
START_YEAR=""
END_YEAR=""
WORKERS=48
GLOBAL_BATCH_SIZE=200
N_HOLDOUT=1000
MIN_COVERAGE=40.0
SAMPLING_METHOD="uniform"

# TAHMO default values
TAHMO_SPLIT="both"
TAHMO_MODE="split"
SKIP_QC=false
SKIP_AUGMENT=false
SKIP_VERIFY=false
DRY_RUN=false
TAHMO_CONFIG="config/tahmo_config.json"
TAHMO_ZARR_PATH=""
TAHMO_BATCH_SIZE=200

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    # Mode selection
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --tahmo)
      DATASET="tahmo"
      shift
      ;;
    --global)
      DATASET="global"
      shift
      ;;

    # Container options
    --container)
      USE_CONTAINER=true
      shift
      ;;
    --container-image)
      USE_CONTAINER=true
      CONTAINER_IMAGE="$2"
      shift 2
      ;;
    --container-env)
      USE_CONTAINER=true
      CONTAINER_TOML="$2"
      shift 2
      ;;

    # TAHMO options
    --yearly)
      TAHMO_MODE="yearly"
      shift
      ;;
    --split)
      TAHMO_SPLIT="$2"
      TAHMO_MODE="split"
      shift 2
      ;;
    --skip-qc)
      SKIP_QC=true
      shift
      ;;
    --skip-augment)
      SKIP_AUGMENT=true
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --config)
      TAHMO_CONFIG="$2"
      shift 2
      ;;
    --zarr-path)
      TAHMO_ZARR_PATH="$2"
      shift 2
      ;;

    # Global options
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    --start-year)
      START_YEAR="--start_year $2"
      shift 2
      ;;
    --end-year)
      END_YEAR="--end_year $2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --batch-size)
      TAHMO_BATCH_SIZE="$2"
      GLOBAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --n-holdout)
      N_HOLDOUT="$2"
      shift 2
      ;;
    --min-coverage)
      MIN_COVERAGE="$2"
      shift 2
      ;;
    --sampling-method)
      SAMPLING_METHOD="$2"
      shift 2
      ;;

    # General
    --help)
      head -n 55 "$SCRIPT_PATH" | tail -n +2 | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
  echo ""
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}STEP $1: $2${NC}"
  echo -e "${BLUE}========================================${NC}"
}

print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
  echo -e "${RED}✗ $1${NC}"
}

# Disable container re-nesting if already inside an active container or SLURM step
if [ "$USE_CONTAINER" = true ]; then
  if [ -f /.enroot ] || [ -f /.dockerenv ] || [ -n "$ENROOT_PID" ] || [ -n "$SLURM_STEP_ID" ] || ! which srun >/dev/null 2>&1; then
    USE_CONTAINER=false
  fi
fi

# Python binary detection (prefer modern python 3.11/3.10/3.9/3.8 over system python 3.6 on host)
HOST_PYTHON=""
for py in python3.11 python3.10 python3.9 python3.8 python3; do
  if command -v "$py" >/dev/null 2>&1; then
    HOST_PYTHON="$py"
    break
  fi
done

# Python launcher: handles either native python or srun container execution
run_python() {
  if [ "$USE_CONTAINER" = true ] || [ "$USE_CONTAINER" = "True" ]; then
    if [ -n "$CONTAINER_TOML" ] && [ -f "$CONTAINER_TOML" ]; then
      srun --container-workdir="$SCRIPT_DIR" --environment="$CONTAINER_TOML" python3 "$@"
    else
      srun --container-image="$CONTAINER_IMAGE" \
           --container-workdir="$SCRIPT_DIR" \
           --container-mounts="/users:/users,/ritom:/ritom,/capstor:/capstor,/iopsstor:/iopsstor" \
           --container-writable \
           python3 "$@"
    fi
  else
    $HOST_PYTHON "$@"
  fi
}

# Check Python or Container availability
if [ "$USE_CONTAINER" = false ]; then
  if [ -z "$HOST_PYTHON" ]; then
    print_error "Python 3 is required but not found in current environment"
    exit 1
  fi
else
  which srun >/dev/null 2>&1 || {
    print_error "'srun' command is required for container execution on CSCS"
    exit 1
  }
fi

################################################################################
# TAHMO Pipeline Execution
################################################################################
if [ "$DATASET" = "tahmo" ]; then
  echo ""
  echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║  TAHMO African Weather Station Processing Pipeline         ║${NC}"
  echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo "Configuration:"
  echo "  Split target:    $TAHMO_SPLIT"
  echo "  Config file:     $TAHMO_CONFIG"
  echo "  Batch size:      $TAHMO_BATCH_SIZE stations"
  if [ -n "$TAHMO_ZARR_PATH" ]; then
    echo "  Zarr store:      $TAHMO_ZARR_PATH"
  fi
  echo "  Container:       $USE_CONTAINER"
  if [ "$USE_CONTAINER" = true ]; then
    echo "  Container img:   $CONTAINER_IMAGE"
    if [ -f "$CONTAINER_TOML" ]; then
      echo "  Container toml:  $CONTAINER_TOML"
    fi
  fi
  echo "  Python binary:   $(which python3) ($(python3 --version 2>&1))"
  echo "  Skip collocated QC: $SKIP_QC"
  echo "  Skip augment:    $SKIP_AUGMENT"
  echo "  Skip verify:     $SKIP_VERIFY"
  echo "  Dry run:         $DRY_RUN"
  echo ""

  # Step 1: Collocated Sensor QC and Reconciliation
  if [ "$SKIP_QC" = false ]; then
    print_step 1 "Running Collocated Multi-Sensor Quality Control & Reconciliation"
    QC_ARGS=("--config" "$TAHMO_CONFIG")
    if [ -n "$TAHMO_ZARR_PATH" ]; then
      QC_ARGS+=("--input_zarr" "$TAHMO_ZARR_PATH")
    fi
    if [ "$DRY_RUN" = true ]; then
      QC_ARGS+=("--dry-run")
    fi
    run_python scripts/TAHMO/collocated_qc.py "${QC_ARGS[@]}"
    print_success "Collocated sensor QC complete: intermediate clean Zarr store generated"
  else
    print_warning "Skipping collocated QC (using existing tahmo_qc_fused.zarr)"
  fi

  # Step 2: Augment Spatial Grid & Station Mapping
  if [ "$SKIP_AUGMENT" = false ]; then
    print_step 2 "Augmenting Spatial Grid with TAHMO African Stations (Algorithm 1)"
    AUG_ARGS=("--config" "$TAHMO_CONFIG")
    if [ -n "$TAHMO_ZARR_PATH" ]; then
      AUG_ARGS+=("--zarr_path" "$TAHMO_ZARR_PATH")
    fi
    run_python scripts/TAHMO/augment_tahmo.py "${AUG_ARGS[@]}"
    print_success "Augmented 12.8k grid and zero-collision TAHMO mapping generated"
  else
    print_warning "Skipping grid augmentation (using existing data/augmented_station_grid_90x180.npz)"
  fi

  # Step 3: Build TAHMO Dataset
  if [ "$TAHMO_MODE" = "yearly" ]; then
    print_step 3 "Building Yearly TAHMO NPZ Data Cubes (SwissClim Standard)"
    BUILD_ARGS=(
      "--config" "$TAHMO_CONFIG"
      "--mode" "yearly"
      "--batch_size" "$TAHMO_BATCH_SIZE"
    )
    if [ -n "$TAHMO_ZARR_PATH" ]; then
      BUILD_ARGS+=("--zarr_path" "$TAHMO_ZARR_PATH")
    fi
    if [ -n "$START_YEAR" ]; then
      BUILD_ARGS+=($START_YEAR)
    fi
    if [ -n "$END_YEAR" ]; then
      BUILD_ARGS+=($END_YEAR)
    fi
    run_python scripts/TAHMO/build_tahmo_dataset.py "${BUILD_ARGS[@]}"
    print_success "Yearly TAHMO NPZ cubes built"

    # Step 3b: Merge yearly NPZ files
    print_step "3b" "Merging Yearly TAHMO NPZ files"
    MERGE_IN=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('merge_yearly',{}).get('input_pattern','/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/weather_data_[0-9][0-9][0-9][0-9].npz'))" 2>/dev/null || echo "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/weather_data_[0-9][0-9][0-9][0-9].npz")
    MERGE_OUT=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('merge_yearly',{}).get('output_file','/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/weather_data_all.npz'))" 2>/dev/null || echo "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/weather_data_all.npz")
    run_python scripts/03_merge_yearly_npz.py --input_pattern "$MERGE_IN" --out_file "$MERGE_OUT"
    print_success "Yearly TAHMO NPZ data merged into $MERGE_OUT"

    # Step 3c: Convert to hybrid format
    print_step "3c" "Converting Merged NPZ to Hybrid Format (NPY + NPZ)"
    HYBRID_NPY=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('convert_hybrid',{}).get('output_data_npy','/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_stations_var9_data.npy'))" 2>/dev/null || echo "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_stations_var9_data.npy")
    HYBRID_NPZ=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('convert_hybrid',{}).get('output_meta_npz','/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_stations_var9_meta.npz'))" 2>/dev/null || echo "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_stations_var9_meta.npz")
    run_python scripts/04_convert_npz_to_hybrid.py --input_npz "$MERGE_OUT" --output_data_npy "$HYBRID_NPY" --output_meta_npz "$HYBRID_NPZ" --force
    print_success "Hybrid format conversion complete ($HYBRID_NPY)"

    # Step 3d: Select holdout stations
    print_step "3d" "Selecting TAHMO Holdout Stations (Geographic Uniform Sampling)"
    HOLDOUT_MAPPING=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('select_holdout',{}).get('out_json','data/tahmo_holdout_mapping_90x180.json'))" 2>/dev/null || echo "data/tahmo_holdout_mapping_90x180.json")
    N_HOLDOUT=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('select_holdout',{}).get('n_holdout', 150))" 2>/dev/null || echo "150")
    N_REGIONS=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('select_holdout',{}).get('n_regions', ''))" 2>/dev/null || echo "")
    SAMPLING_METHOD=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('select_holdout',{}).get('sampling_method', 'uniform'))" 2>/dev/null || echo "uniform")

    HOLDOUT_ARGS=(
      --mapping_json "data/tahmo_station_mapping_90x180.json"
      --n_holdout "$N_HOLDOUT"
      --out_json "$HOLDOUT_MAPPING"
      --sampling_method "$SAMPLING_METHOD"
    )
    if [ -n "$N_REGIONS" ]; then
      HOLDOUT_ARGS+=(--n_regions "$N_REGIONS")
    fi

    run_python scripts/05_select_holdout_stations.py "${HOLDOUT_ARGS[@]}"
    print_success "Selected $N_HOLDOUT holdout stations across Africa"

    # Step 3e: Create holdout mask
    print_step "3e" "Creating Holdout Mask (90x180)"
    HOLDOUT_MASK=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('create_mask',{}).get('out_npz','/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/holdout_mask.npz'))" 2>/dev/null || echo "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/holdout_mask.npz")
    run_python scripts/06_create_holdout_mask.py \
      --holdout_json "$HOLDOUT_MAPPING" \
      --mapping_json "data/tahmo_station_mapping_90x180.json" \
      --out_npz "$HOLDOUT_MASK"
    print_success "Created holdout mask ($HOLDOUT_MASK)"

    # Step 3f: Split dataset into training and holdout (.npy + .npz)
    print_step "3f" "Splitting TAHMO Dataset into Training & Holdout Datasets"
    SPLIT_OUT_DIR=$(run_python -c "import json; c=json.load(open('$TAHMO_CONFIG')); print(c.get('split_dataset',{}).get('out_dir','/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO'))" 2>/dev/null || echo "/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO")
    run_python scripts/07_split_dataset_holdout.py \
      --data_npy "$HYBRID_NPY" \
      --meta_npz "$HYBRID_NPZ" \
      --holdout_mask "$HOLDOUT_MASK" \
      --out_dir "$SPLIT_OUT_DIR" \
      --out_prefix tahmo
    print_success "Dataset split complete: $SPLIT_OUT_DIR/tahmo_training_data.npy & $SPLIT_OUT_DIR/tahmo_holdout_data.npy"

  else
    print_step 3 "Building Memory-Mapped ML Datasets (Split: $TAHMO_SPLIT)"
    BUILD_ARGS=(
      "--config" "$TAHMO_CONFIG"
      "--split" "$TAHMO_SPLIT"
      "--batch_size" "$TAHMO_BATCH_SIZE"
    )
    if [ -n "$TAHMO_ZARR_PATH" ]; then
      BUILD_ARGS+=("--zarr_path" "$TAHMO_ZARR_PATH")
    fi
    run_python scripts/TAHMO/build_tahmo_dataset.py "${BUILD_ARGS[@]}"
    print_success "TAHMO dataset building complete"
  fi

  # Step 4: Verification
  if [ "$SKIP_VERIFY" = false ]; then
    print_step 4 "Verifying Dataset Compliance and Physical Distributions"

    # Extract output paths from config if possible, fallback to standard defaults
    TRAIN_NPY="/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_training_data.npy"
    TRAIN_NPZ="/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_training_meta.npz"
    HOLDOUT_NPY="/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_holdout_data.npy"
    HOLDOUT_NPZ="/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO/tahmo_holdout_meta.npz"

    if [ "$TAHMO_SPLIT" = "train" ] || [ "$TAHMO_SPLIT" = "both" ]; then
      if [ -f "$TRAIN_NPY" ] && [ -f "$TRAIN_NPZ" ]; then
        echo "Verifying Training split..."
        run_python scripts/TAHMO/verify_tahmo_dataset.py --data_npy "$TRAIN_NPY" --meta_npz "$TRAIN_NPZ"
      fi
    fi

    if [ "$TAHMO_SPLIT" = "holdout" ] || [ "$TAHMO_SPLIT" = "both" ]; then
      if [ -f "$HOLDOUT_NPY" ] && [ -f "$HOLDOUT_NPZ" ]; then
        echo "Verifying Holdout split..."
        run_python scripts/TAHMO/verify_tahmo_dataset.py --data_npy "$HOLDOUT_NPY" --meta_npz "$HOLDOUT_NPZ"
      fi
    fi

    print_success "Verification passed: All checks aligned with SwissClim standard"
  else
    print_warning "Skipping dataset verification"
  fi

  echo ""
  echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${GREEN}║  TAHMO Pipeline Completed Successfully!                    ║${NC}"
  echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo "Generated Datasets:"
  if [ "$TAHMO_SPLIT" = "train" ] || [ "$TAHMO_SPLIT" = "both" ]; then
    echo "  - Training Data: $TRAIN_NPY"
    echo "  - Training Meta: $TRAIN_NPZ"
  fi
  if [ "$TAHMO_SPLIT" = "holdout" ] || [ "$TAHMO_SPLIT" = "both" ]; then
    echo "  - Holdout Data:  $HOLDOUT_NPY"
    echo "  - Holdout Meta:  $HOLDOUT_NPZ"
  fi
  echo ""
  echo "Next steps:"
  echo "  1. Train or fine-tune ESFM on TAHMO dataset"
  echo "  2. Run validation against TAHMO holdout observations"
  echo ""
  exit 0
fi

################################################################################
# Global Pipeline Execution
################################################################################
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Global Weather Station Data Processing Pipeline           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Download data
if [ "$SKIP_DOWNLOAD" = false ]; then
  print_step 1 "Downloading raw data"
  run_python scripts/01_download_data.py
  print_success "Data download complete"
else
  print_warning "Skipping data download (using existing data)"
fi

# Step 2: Build yearly data
print_step 2 "Building yearly data cubes"
run_python scripts/02_build_station_data_yearly.py \
  --nc_folder station_data/raw_data \
  --mapping_json final_results_mapping_90x180.json \
  --stats_json weather_data_norm_stats.json \
  --grid_npz final_results_station_grid_90x180.npz \
  --out_prefix weather_data \
  --workers $WORKERS \
  --batch_size $GLOBAL_BATCH_SIZE \
  $START_YEAR $END_YEAR
print_success "Yearly data cubes built"

# Step 3: Merge yearly data
print_step 3 "Merging yearly data"
run_python scripts/03_merge_yearly_npz.py \
  --input_pattern "weather_data_*.npz" \
  --output_file weather_data_all.npz
print_success "Data merged into single file"

# Step 4: Convert to hybrid format
print_step 4 "Converting to hybrid format (NPY + NPZ)"
run_python scripts/04_convert_npz_to_hybrid.py
print_success "Converted to hybrid format"

# Step 5: Select holdout stations
print_step 5 "Selecting holdout stations"
run_python scripts/05_select_holdout_stations.py \
  --coverage_csv station_coverage_2023.csv \
  --mapping_json final_results_mapping_90x180.json \
  --n_holdout $N_HOLDOUT \
  --min_coverage $MIN_COVERAGE \
  --out_json final_holdout_mapping_90x180.json \
  --sampling_method $SAMPLING_METHOD
print_success "Holdout stations selected ($N_HOLDOUT stations)"

# Step 6: Create holdout mask
print_step 6 "Creating holdout mask"
run_python scripts/06_create_holdout_mask.py \
  --holdout_json final_holdout_mapping_90x180.json \
  --out_npz holdout_mask.npz
print_success "Holdout mask created"

# Step 7: Split dataset
print_step 7 "Splitting dataset into training and holdout"
run_python scripts/07_split_dataset_holdout.py \
  --data_npy stations_11k_var9_data.npy \
  --meta_npz stations_11k_var9_meta.npz \
  --holdout_mask holdout_mask.npz \
  --out_dir ESFM_Dataset
print_success "Dataset split complete"

# Optional: Plot holdout distribution
if [ -f "scripts/plot_holdout_distribution.py" ]; then
  print_step "BONUS" "Plotting holdout distribution"
  run_python scripts/plot_holdout_distribution.py
  print_success "Visualization saved to coverage_plots/holdout_distribution.png"
fi

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Pipeline Complete!                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Output files:"
echo "  - ESFM_Dataset/stations_11k_data_training.npy"
echo "  - ESFM_Dataset/stations_11k_meta_training.npz"
echo "  - ESFM_Dataset/stations_11k_data_holdout.npy"
echo "  - ESFM_Dataset/stations_11k_meta_holdout.npz"
echo ""
echo "Next steps:"
echo "  1. Review coverage_plots/holdout_distribution.png"
echo "  2. Load data using examples in docs/usage_examples.py"
echo "  3. Start training your model!"
echo ""
