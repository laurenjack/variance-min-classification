#!/bin/bash
# lambda_upload.sh - Upload local training artifacts to a Lambda Labs instance
#
# Mirrors the local data path to the remote output path:
#   local:  ./data/{experiment_type}/{timestamp}/
#   remote: ~/variance-min-classification/output/{experiment_type}/{timestamp}/
#
# Usage:
#   ./infra/lambda_upload.sh <instance_ip> <local_folder>
#
# Examples:
#   ./infra/lambda_upload.sh 192.222.54.255 ./data/transformer_variance/03-01-1010

set -euo pipefail

SSH_KEY_PATH="${LAMBDA_SSH_KEY_PATH:?LAMBDA_SSH_KEY_PATH not set}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <instance_ip> <local_folder>"
    echo ""
    echo "Example: $0 192.222.54.255 ./data/transformer_variance/03-01-1010"
    exit 1
fi

INSTANCE_IP="$1"
LOCAL_FOLDER="$2"

if [[ ! -d "$LOCAL_FOLDER" ]]; then
    log_error "Local folder not found: $LOCAL_FOLDER"
    exit 1
fi

# Extract experiment_type/timestamp from the folder path.
# Expects path ending in .../data/{experiment_type}/{timestamp} or
# .../data/{experiment_type}/{timestamp}/
# Strip trailing slash
LOCAL_FOLDER="${LOCAL_FOLDER%/}"

TIMESTAMP=$(basename "$LOCAL_FOLDER")
EXPERIMENT_TYPE=$(basename "$(dirname "$LOCAL_FOLDER")")

REMOTE_DIR="~/variance-min-classification/output/$EXPERIMENT_TYPE/$TIMESTAMP"

log_info "Uploading: $LOCAL_FOLDER"
log_info "  -> $EXPERIMENT_TYPE/$TIMESTAMP"
log_info "  -> remote: $REMOTE_DIR"

# Create remote directory
ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" \
    "mkdir -p $REMOTE_DIR"

# Upload all files
scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no -r \
    "$LOCAL_FOLDER"/* \
    "ubuntu@$INSTANCE_IP:$REMOTE_DIR/"

log_info "Upload complete."

# Show what was uploaded
FILE_COUNT=$(ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" \
    "ls $REMOTE_DIR/ | wc -l")
log_info "Files on remote: $FILE_COUNT"
