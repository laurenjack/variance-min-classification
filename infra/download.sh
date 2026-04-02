#!/bin/bash
# download.sh - Download training logs and metrics from a remote instance via SSH
#
# Downloads training artifacts with structured paths:
#   remote: ~/output/{experiment_type}/{timestamp}/
#   local:  ./data/{experiment_type}/{timestamp}/
#
# Auto-runs appropriate plot scripts after download.
#
# Usage:
#   ./infra/download.sh <instance_ip> [--user <user>] [--port <port>]
#
# Examples:
#   ./infra/download.sh 192.222.54.255
#   ./infra/download.sh 192.222.54.255 --user root --port 22005  # RunPod

set -euo pipefail

SSH_KEY_PATH="${SSH_KEY_PATH:?SSH_KEY_PATH not set}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <instance_ip> [--user <user>] [--port <port>]"
    exit 1
fi

INSTANCE_IP="$1"
shift

# Parse optional arguments
SSH_USER="ubuntu"
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

LOCAL_DATA="./data"

log_info "SSH user: $SSH_USER"
if [[ -n "$SSH_PORT" ]]; then
    log_info "SSH port: $SSH_PORT"
fi

log_info "Discovering experiment folders on $INSTANCE_IP..."

# List remote output structure
REMOTE_STRUCTURE=$(ssh -i "$SSH_KEY_PATH" $SSH_OPTS "$SSH_USER@$INSTANCE_IP" \
    'find ~/variance-min-classification/output -mindepth 2 -maxdepth 2 -type d 2>/dev/null || echo ""')

if [[ -z "$REMOTE_STRUCTURE" ]]; then
    log_warn "No structured output folders found. Checking for legacy flat output..."

    # Fallback: check for flat metrics files
    ssh -i "$SSH_KEY_PATH" $SSH_OPTS "$SSH_USER@$INSTANCE_IP" \
        'ls ~/variance-min-classification/output/*.jsonl 2>/dev/null || echo "No metrics files found"'
    exit 1
fi

log_info "Found experiment folders:"
echo "$REMOTE_STRUCTURE" | sed 's|.*/output/||'

# Download each experiment type's folders
for EXPERIMENT_TYPE in resnet18 resnet18_variance transformer transformer_variance transformer_m2m100_variance reward_model; do
    # Find timestamp folders for this experiment type
    FOLDERS=$(echo "$REMOTE_STRUCTURE" | grep "/$EXPERIMENT_TYPE/" || true)

    if [[ -z "$FOLDERS" ]]; then
        continue
    fi

    log_info "Downloading $EXPERIMENT_TYPE experiments..."

    for REMOTE_FOLDER in $FOLDERS; do
        # Extract timestamp from path (e.g., ~/variance-min-classification/output/transformer/03-01-1010)
        TIMESTAMP=$(basename "$REMOTE_FOLDER")
        LOCAL_FOLDER="$LOCAL_DATA/$EXPERIMENT_TYPE/$TIMESTAMP"

        mkdir -p "$LOCAL_FOLDER"

        log_info "  -> $EXPERIMENT_TYPE/$TIMESTAMP"

        # Download all files from this folder
        scp -i "$SSH_KEY_PATH" $SCP_OPTS -r \
            "$SSH_USER@$INSTANCE_IP:$REMOTE_FOLDER/*" \
            "$LOCAL_FOLDER/" 2>/dev/null || log_warn "    No files in $EXPERIMENT_TYPE/$TIMESTAMP"
    done
done

# Also download training log to the most recent folder
log_info "Downloading training.log..."
LATEST_FOLDER=$(echo "$REMOTE_STRUCTURE" | sort | tail -1)
if [[ -n "$LATEST_FOLDER" ]]; then
    EXPERIMENT_TYPE=$(echo "$LATEST_FOLDER" | sed 's|.*/output/\([^/]*\)/.*|\1|')
    TIMESTAMP=$(basename "$LATEST_FOLDER")
    LOCAL_FOLDER="$LOCAL_DATA/$EXPERIMENT_TYPE/$TIMESTAMP"

    scp -i "$SSH_KEY_PATH" $SCP_OPTS \
        "$SSH_USER@$INSTANCE_IP:~/variance-min-classification/training.log" \
        "$LOCAL_FOLDER/" 2>/dev/null || log_warn "No training.log found"
fi

log_info "Download complete. Contents:"
log_info "  Metrics files:"
find "$LOCAL_DATA" -mindepth 2 -maxdepth 3 -type f -name "*.jsonl" | head -10
log_info "  Model files:"
find "$LOCAL_DATA" -mindepth 2 -maxdepth 3 -type f -name "*.pt" | head -10

# Run plot scripts
log_info "Generating plots..."

# Activate venv if available
if [[ -f "./venv/bin/activate" ]]; then
    source ./venv/bin/activate
fi

# Plot each downloaded experiment
for EXPERIMENT_TYPE in resnet18 resnet18_variance transformer transformer_variance transformer_m2m100_variance reward_model; do
    EXPERIMENT_DIR="$LOCAL_DATA/$EXPERIMENT_TYPE"

    if [[ ! -d "$EXPERIMENT_DIR" ]]; then
        continue
    fi

    # Find timestamp folders
    for TIMESTAMP_DIR in "$EXPERIMENT_DIR"/*/; do
        if [[ ! -d "$TIMESTAMP_DIR" ]]; then
            continue
        fi

        TIMESTAMP=$(basename "$TIMESTAMP_DIR")
        log_info "Plotting $EXPERIMENT_TYPE/$TIMESTAMP..."

        case $EXPERIMENT_TYPE in
            resnet18)
                if [[ -f "$TIMESTAMP_DIR/evaluation.jsonl" ]]; then
                    python -m jl.double_descent.resnet18.plot_evaluation \
                        "$TIMESTAMP_DIR/evaluation.jsonl" --output-dir "$TIMESTAMP_DIR" || \
                        log_warn "Failed to plot resnet18"
                else
                    log_info "No evaluation.jsonl in $TIMESTAMP_DIR"
                fi
                ;;
            resnet18_variance)
                if [[ -f "$TIMESTAMP_DIR/evaluation.jsonl" ]]; then
                    python -m jl.double_descent.resnet18.plot_variance_evaluation \
                        "$TIMESTAMP_DIR/evaluation.jsonl" --output-dir "$TIMESTAMP_DIR" || \
                        log_warn "Failed to plot resnet18_variance"
                else
                    log_info "No evaluation.jsonl in $TIMESTAMP_DIR (run variance_evaluation first)"
                fi
                if [[ -f "$TIMESTAMP_DIR/temperature-scaled/evaluation.jsonl" ]]; then
                    python -m jl.double_descent.resnet18.plot_variance_evaluation \
                        "$TIMESTAMP_DIR/temperature-scaled/evaluation.jsonl" \
                        --output-dir "$TIMESTAMP_DIR/temperature-scaled" \
                        --temperature-scaled || \
                        log_warn "Failed to plot resnet18_variance (temperature-scaled)"
                fi
                ;;
            transformer)
                if [[ -f "$TIMESTAMP_DIR/evaluation.jsonl" ]]; then
                    python -m jl.double_descent.transformer.plot_evaluation \
                        "$TIMESTAMP_DIR/evaluation.jsonl" --output-dir "$TIMESTAMP_DIR" || \
                        log_warn "Failed to plot transformer"
                else
                    log_info "No evaluation.jsonl in $TIMESTAMP_DIR"
                fi
                ;;
            transformer_variance)
                if [[ -f "$TIMESTAMP_DIR/evaluation.jsonl" ]]; then
                    python -m jl.double_descent.transformer.plot_variance_evaluation \
                        "$TIMESTAMP_DIR/evaluation.jsonl" --output-dir "$TIMESTAMP_DIR" || \
                        log_warn "Failed to plot transformer_variance"
                else
                    log_info "No evaluation.jsonl in $TIMESTAMP_DIR (run variance_evaluation first)"
                fi
                if [[ -f "$TIMESTAMP_DIR/temperature-scaled/evaluation.jsonl" ]]; then
                    python -m jl.double_descent.transformer.plot_variance_evaluation \
                        "$TIMESTAMP_DIR/temperature-scaled/evaluation.jsonl" \
                        --output-dir "$TIMESTAMP_DIR/temperature-scaled" \
                        --temperature-scaled || \
                        log_warn "Failed to plot transformer_variance (temperature-scaled)"
                fi
                ;;
            transformer_m2m100_variance)
                if [[ -f "$TIMESTAMP_DIR/reference/evaluation.jsonl" ]]; then
                    python -m jl.double_descent.transformer.plot_variance_evaluation \
                        "$TIMESTAMP_DIR/reference/evaluation.jsonl" --output-dir "$TIMESTAMP_DIR/reference" || \
                        log_warn "Failed to plot transformer_m2m100_variance"
                else
                    log_info "No reference/evaluation.jsonl in $TIMESTAMP_DIR (run variance_evaluation --reference-logits first)"
                fi
                ;;
            reward_model)
                if [[ -f "$TIMESTAMP_DIR/metrics.jsonl" ]]; then
                    python -m jl.reward_model.plot_metrics "$TIMESTAMP_DIR/metrics.jsonl" \
                        --output-dir "$TIMESTAMP_DIR" || \
                        log_warn "Failed to plot reward_model"
                fi
                ;;
        esac
    done
done

log_info "Done! Results in $LOCAL_DATA/"
