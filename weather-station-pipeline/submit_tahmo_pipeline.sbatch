#!/bin/bash
#SBATCH --job-name=tahmo_holdout_select
#SBATCH --account=g164
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --output=tahmo_holdout_select_%j.out
#SBATCH --error=tahmo_holdout_select_%j.err

echo "Starting TAHMO pipeline on CSCS inside torch_physics_nemo container..."
echo "Running on node: $(hostname)"
echo "Date: $(date)"

# Robust directory resolution: works whether run from /users, $HOME, or /ritom scratch via symlink
CANDIDATE_DIRS=(
  "/users/tligawa/SwissClim_data_processing_scripts/weather-station-pipeline"
  "/ritom/scratch/cscs/tligawa/home/SwissClim_data_processing_scripts/weather-station-pipeline"
  "$HOME/SwissClim_data_processing_scripts/weather-station-pipeline"
  "$(pwd)/weather-station-pipeline"
  "$(pwd)"
)

PIPELINE_DIR=""
for d in "${CANDIDATE_DIRS[@]}"; do
  if [ -f "$d/run_pipeline.sh" ]; then
    PIPELINE_DIR="$d"
    break
  fi
done

if [ -z "$PIPELINE_DIR" ]; then
  echo "Error: Could not locate run_pipeline.sh in any candidate path!" >&2
  exit 1
fi

echo "Resolved PIPELINE_DIR: $PIPELINE_DIR"
cd "$PIPELINE_DIR"

CONTAINER_TOML="/ritom/scratch/cscs/tligawa/container-images/torch_physics_nemo.toml"

# Prevent host virtualenv leakage into container
unset VIRTUAL_ENV
unset PYTHONPATH

# Execute TAHMO pipeline inside the container via srun (CSCS recommended pattern)
# - --ntasks=1: Ensures only a single pipeline instance runs (prevents duplicate tasks)
# - --container-workdir: Overrides default toml workdir to execute directly inside the pipeline directory
# - Absolute script path ensures execve locates the script inside the container filesystem
srun --ntasks=1 \
     --cpus-per-task=288 \
     --container-workdir="$PIPELINE_DIR" \
     --environment="$CONTAINER_TOML" \
     "$PIPELINE_DIR/run_pipeline.sh" --tahmo --yearly

echo "Pipeline finished at $(date)"
