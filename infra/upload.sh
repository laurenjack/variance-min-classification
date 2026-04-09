#!/bin/bash
# upload.sh - Upload local training artifacts to a remote instance via SSH
#
# Mirrors the local data path to the remote output path:
#   local:  ./data/{experiment_type}/{timestamp}/
#   remote: /workspace/variance-min-classification/output/{experiment_type}/{timestamp}/
#
# Usage:
#   ./infra/upload.sh <instance_ip> <local_folder> [--user <user>] [--port <port>]
#
# Examples:
#   ./infra/upload.sh 192.222.54.255 ./data/transformer_variance/03-01-1010
#   ./infra/upload.sh 192.222.54.255 ./data/transformer_variance/03-01-1010 --user root --port 22005  # RunPod

set -euo pipefail

SSH_KEY_PATH="${SSH_KEY_PATH:?SSH_KEY_PATH not set}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <instance_ip> <local_folder> [--user <user>] [--port <port>]"
    echo ""
    echo "Example: $0 192.222.54.255 ./data/transformer_variance/03-01-1010"
    echo "Example: $0 192.222.54.255 ./data/transformer_variance/03-01-1010 --user root --port 22005  # RunPod"
    exit 1
fi

INSTANCE_IP="$1"
LOCAL_FOLDER="$2"
shift 2

if [[ ! -d "$LOCAL_FOLDER" ]]; then
    log_error "Local folder not found: $LOCAL_FOLDER"
    exit 1
fi

# Parse optional arguments
SSH_USER="root"
SSH_PORT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            SSH_USER="$2"
            shift 2
            ;;
        --port)
            SSH_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Build SSH options
SSH_OPTS="-o StrictHostKeyChecking=no"
if [[ -n "$SSH_PORT" ]]; then
    SSH_OPTS="$SSH_OPTS -p $SSH_PORT"
fi

# Build SCP options (uses -P for port, not -p)
SCP_OPTS="-o StrictHostKeyChecking=no"
if [[ -n "$SSH_PORT" ]]; then
    SCP_OPTS="$SCP_OPTS -P $SSH_PORT"
fi

# Extract experiment_type/timestamp from the folder path.
# Expects path ending in .../data/{experiment_type}/{timestamp} or
# .../data/{experiment_type}/{timestamp}/
# Strip trailing slash
LOCAL_FOLDER="${LOCAL_FOLDER%/}"

TIMESTAMP=$(basename "$LOCAL_FOLDER")
EXPERIMENT_TYPE=$(basename "$(dirname "$LOCAL_FOLDER")")

REMOTE_DIR="/workspace/variance-min-classification/output/$EXPERIMENT_TYPE/$TIMESTAMP"

log_info "SSH user: $SSH_USER"
if [[ -n "$SSH_PORT" ]]; then
    log_info "SSH port: $SSH_PORT"
fi

log_info "Uploading: $LOCAL_FOLDER"
log_info "  -> $EXPERIMENT_TYPE/$TIMESTAMP"
log_info "  -> remote: $REMOTE_DIR"

# Create remote directory
ssh -i "$SSH_KEY_PATH" $SSH_OPTS "$SSH_USER@$INSTANCE_IP" \
    "mkdir -p $REMOTE_DIR"

# Upload all files
scp -i "$SSH_KEY_PATH" $SCP_OPTS -r \
    "$LOCAL_FOLDER"/* \
    "$SSH_USER@$INSTANCE_IP:$REMOTE_DIR/"

log_info "Upload complete."

# Show what was uploaded
FILE_COUNT=$(ssh -i "$SSH_KEY_PATH" $SSH_OPTS "$SSH_USER@$INSTANCE_IP" \
    "ls $REMOTE_DIR/ | wc -l")
log_info "Files on remote: $FILE_COUNT"
