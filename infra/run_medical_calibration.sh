#!/bin/bash
# run_medical_calibration.sh - Extract zip and run calibration sweep for one dataset
#
# Usage:
#   ./infra/run_medical_calibration.sh <dataset_name>
#
# Example:
#   ./infra/run_medical_calibration.sh aptos2019
#
# Expects data/medical_calibration/<dataset_name>.zip on the remote.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET="$1"
ZIP_PATH="data/medical_calibration/${DATASET}.zip"
EXTRACT_DIR="data/medical_calibration/${DATASET}_extracted"

if [[ ! -f "$ZIP_PATH" ]]; then
    echo "ERROR: $ZIP_PATH not found"
    exit 1
fi

# Extract if not already done
if [[ ! -d "$EXTRACT_DIR" ]]; then
    echo "Extracting $ZIP_PATH..."
    mkdir -p "$EXTRACT_DIR"
    python3 -c "import zipfile; zipfile.ZipFile('$ZIP_PATH').extractall('$EXTRACT_DIR')"
    echo "Extracted to $EXTRACT_DIR"
fi

# Find the data directory (contains train/val/test subdirs)
TRAIN_DIR=$(find "$EXTRACT_DIR" -type d -name "train" | head -1)
if [[ -z "$TRAIN_DIR" ]]; then
    echo "ERROR: No train/ directory found in $EXTRACT_DIR"
    find "$EXTRACT_DIR" -maxdepth 3 -type d
    exit 1
fi
DATA_DIR=$(dirname "$TRAIN_DIR")
CHECKPOINT="$EXTRACT_DIR/checkpoint-best.pth"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "Dataset: $DATASET"
echo "Data: $DATA_DIR"
echo "Checkpoint: $CHECKPOINT"

# Run calibration sweep
python -m jl.double_descent.medical_calibration.calibrate \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_DIR" \
    --output-path "./output/medical_calibration/${DATASET}" \
    --sweep
