#!/bin/bash
exec "$(dirname "$0")/weather-station-pipeline/run_pipeline.sh" "$@"

