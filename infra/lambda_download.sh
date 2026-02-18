#!/bin/bash
# lambda_download.sh - Download training logs and metrics from Lambda Labs instance
#
# Downloads training log and metrics (not the model), then auto-generates training plots.
#
# Usage:
#   ./lambda_download.sh <instance_ip>
#   ./lambda_download.sh <instance_ip> <local_output_dir>

set -euo pipefail

SSH_KEY_PATH="${LAMBDA_SSH_KEY_PATH:?LAMBDA_SSH_KEY_PATH not set}"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <instance_ip> [local_output_dir]"
    exit 1
fi

INSTANCE_IP="$1"
LOCAL_OUTPUT="${2:-./data/lambda_output}"

log_info "Downloading artifacts from $INSTANCE_IP to $LOCAL_OUTPUT..."

mkdir -p "$LOCAL_OUTPUT"

# Copy metrics file
scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no \
    "ubuntu@$INSTANCE_IP:~/variance-min-classification/output/metrics.jsonl" \
    "$LOCAL_OUTPUT/" 2>/dev/null || log_info "No metrics file yet"

# Copy validation metrics file
scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no \
    "ubuntu@$INSTANCE_IP:~/variance-min-classification/output/val_metrics.jsonl" \
    "$LOCAL_OUTPUT/" 2>/dev/null || log_info "No validation metrics file yet"

# Copy training log
scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no \
    "ubuntu@$INSTANCE_IP:~/variance-min-classification/training.log" \
    "$LOCAL_OUTPUT/" 2>/dev/null || log_info "No training log yet"

log_info "Downloaded to $LOCAL_OUTPUT:"
ls -la "$LOCAL_OUTPUT"

# Auto-generate training plots if metrics file exists
METRICS_FILE="$LOCAL_OUTPUT/metrics.jsonl"
if [[ -f "$METRICS_FILE" ]]; then
    log_info "Generating training plots..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    DATA_DIR="$PROJECT_ROOT/data"
    mkdir -p "$DATA_DIR"

    # Activate venv and run plotting script
    source "$PROJECT_ROOT/venv/bin/activate" 2>/dev/null || true
    python -m jl.reward_model.plot_metrics "$METRICS_FILE" --output-dir "$DATA_DIR"

    log_info "Plots saved to $DATA_DIR"
else
    log_info "No metrics.jsonl found - skipping plot generation"
fi
