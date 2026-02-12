#!/bin/bash
# lambda_train.sh - Run reward model training on a Lambda Labs instance
#
# Prerequisites:
#   - Instance running (use lambda_launch.sh first)
#   - HF_TOKEN environment variable set (for gated models like Llama)
#   - LAMBDA_SSH_KEY_PATH environment variable set
#
# Usage:
#   ./lambda_train.sh <instance_ip> [--background] [--learning-rate <lr>] [--warmup-steps <steps>]
#
# Examples:
#   ./lambda_train.sh 192.222.54.255
#   ./lambda_train.sh 192.222.54.255 --background
#   ./lambda_train.sh 192.222.54.255 --learning-rate 3e-5
#   ./lambda_train.sh 192.222.54.255 --background --learning-rate 3e-4
#   ./lambda_train.sh 192.222.54.255 --learning-rate 3e-5 --warmup-steps 50

set -euo pipefail

# Configuration
SSH_KEY_PATH="${LAMBDA_SSH_KEY_PATH:?LAMBDA_SSH_KEY_PATH not set}"
REPO_URL="https://github.com/laurenjack/variance-min-classification.git"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <instance_ip> [--background] [--learning-rate <lr>] [--warmup-steps <steps>]"
    exit 1
fi

INSTANCE_IP="$1"
shift

# Parse optional arguments
BACKGROUND=""
LEARNING_RATE=""
WARMUP_STEPS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --background)
            BACKGROUND="true"
            shift
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --warmup-steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Build flags for Python command
EXTRA_FLAGS=""
if [[ -n "$LEARNING_RATE" ]]; then
    EXTRA_FLAGS="--learning-rate $LEARNING_RATE"
    log_info "Using learning rate: $LEARNING_RATE"
fi
if [[ -n "$WARMUP_STEPS" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --warmup-steps $WARMUP_STEPS"
    log_info "Using warmup steps: $WARMUP_STEPS"
fi

# Check prerequisites
if [[ -z "${HF_TOKEN:-}" ]]; then
    log_warn "HF_TOKEN not set - gated models (like Llama) will fail"
fi

if [[ ! -f "$SSH_KEY_PATH" ]]; then
    log_error "SSH private key not found at $SSH_KEY_PATH"
    exit 1
fi

chmod 600 "$SSH_KEY_PATH" 2>/dev/null || true

log_info "Connecting to $INSTANCE_IP..."

# Build the remote script
# Note: We pass HF_TOKEN from local env to remote
REMOTE_SCRIPT="
set -ex

# Set HF token for gated models
export HF_TOKEN='${HF_TOKEN:-}'

echo '=== Setting up environment ==='
cd ~

# Clone the repo (or pull if exists)
if [[ -d 'variance-min-classification' ]]; then
    cd variance-min-classification
    git pull
else
    git clone $REPO_URL
    cd variance-min-classification
fi

# Create virtual environment if needed
if [[ ! -d 'venv' ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-gpu.txt

echo '=== Starting training ==='
mkdir -p output data

python -m jl.reward_model.main \\
    --train-path ./data/tokenized \\
    --output-path ./output $EXTRA_FLAGS

echo '=== Training complete ==='
ls -la output/
"

REMOTE_SCRIPT_BG="
set -ex

# Set HF token for gated models
export HF_TOKEN='${HF_TOKEN:-}'

echo '=== Setting up environment ==='
cd ~

# Clone the repo (or pull if exists)
if [[ -d 'variance-min-classification' ]]; then
    cd variance-min-classification
    git pull
else
    git clone $REPO_URL
    cd variance-min-classification
fi

# Create virtual environment if needed
if [[ ! -d 'venv' ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-gpu.txt

echo '=== Starting training in background ==='
mkdir -p output data

nohup python -m jl.reward_model.main \\
    --train-path ./data/tokenized \\
    --output-path ./output $EXTRA_FLAGS \\
    > training.log 2>&1 &

echo \"Training started in background. PID: \\\$!\"
echo \"Monitor with: ssh -i $SSH_KEY_PATH ubuntu@$INSTANCE_IP 'tail -f ~/variance-min-classification/training.log'\"
"

if [[ "$BACKGROUND" == "true" ]]; then
    log_info "Starting training in background mode..."
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" bash <<< "$REMOTE_SCRIPT_BG"
    log_info "Training started in background on $INSTANCE_IP"
    log_info "Check progress: ssh -i $SSH_KEY_PATH ubuntu@$INSTANCE_IP 'tail -f ~/variance-min-classification/training.log'"
else
    log_info "Starting training (foreground mode)..."
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" bash <<< "$REMOTE_SCRIPT"
fi
