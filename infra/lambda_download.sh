#!/bin/bash
# lambda_download.sh - Download training logs and metrics from Lambda Labs instance
#
# Downloads training log and metrics (not the model).
#
# Usage:
#   ./lambda_download.sh <instance_ip> [options]
#
# Options:
#   --output-dir <dir>    Local directory for downloads (default: ./data/lambda_output)
#
# Examples:
#   ./lambda_download.sh 192.222.54.255
#   ./lambda_download.sh 192.222.54.255 --output-dir ./my_output

set -euo pipefail

SSH_KEY_PATH="${LAMBDA_SSH_KEY_PATH:?LAMBDA_SSH_KEY_PATH not set}"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <instance_ip> [--output-dir <dir>]"
    exit 1
fi

INSTANCE_IP="$1"
shift

# Parse optional arguments
LOCAL_OUTPUT="./data/lambda_output"
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            LOCAL_OUTPUT="$2"
            shift 2
            ;;
        *)
            # Legacy: treat first positional arg as output dir
            LOCAL_OUTPUT="$1"
            shift
            ;;
    esac
done

log_info "Downloading artifacts from $INSTANCE_IP to $LOCAL_OUTPUT..."

mkdir -p "$LOCAL_OUTPUT"

# Copy metrics file(s) - handles reward_model, double_descent, and transformer_dd
scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no \
    "ubuntu@$INSTANCE_IP:~/variance-min-classification/output/metrics.jsonl" \
    "$LOCAL_OUTPUT/" 2>/dev/null || log_info "No metrics.jsonl file"

scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no \
    "ubuntu@$INSTANCE_IP:~/variance-min-classification/output/metrics_k*.jsonl" \
    "$LOCAL_OUTPUT/" 2>/dev/null || log_info "No metrics_k*.jsonl files"

scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no \
    "ubuntu@$INSTANCE_IP:~/variance-min-classification/output/metrics_d*.jsonl" \
    "$LOCAL_OUTPUT/" 2>/dev/null || log_info "No metrics_d*.jsonl files"

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

log_info "To plot results, run the appropriate plot module:"
log_info "  Reward model:    python -m jl.reward_model.plot_metrics <metrics.jsonl> --output-dir ./data"
log_info "  ResNet18 DD:     python -m jl.double_descent.resnet18.plot <output_dir> --output-dir ./data"
log_info "  Transformer DD:  python -m jl.double_descent.transformer.plot <metrics_d*.jsonl or dir> --output-dir ./data"
