#!/bin/bash
# train.sh - Run training on a remote GPU instance via SSH
#
# Prerequisites:
#   - Instance running with SSH access
#   - HF_TOKEN environment variable set (for gated models like Llama)
#   - SSH_KEY_PATH environment variable set (path to SSH private key)
#
# Usage:
#   ./infra/train.sh <instance_ip> [--user <user>] [--port <port>] [--background] [--module <module>] [--learning-rate <lr>] [--warmup-steps <steps>] [--variance]
#
# Examples:
#   ./infra/train.sh 192.222.54.255
#   ./infra/train.sh 192.222.54.255 --background
#   ./infra/train.sh 192.222.54.255 --user root --port 22005  # RunPod
#   ./infra/train.sh 192.222.54.255 --learning-rate 3e-5
#   ./infra/train.sh 192.222.54.255 --module jl.double_descent.resnet18.resnet18_main
#   ./infra/train.sh 192.222.54.255 --module jl.double_descent.transformer.transformer_main --variance

set -euo pipefail

# Configuration
SSH_KEY_PATH="${SSH_KEY_PATH:?SSH_KEY_PATH not set}"
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
    echo "Usage: $0 <instance_ip> [--user <user>] [--port <port>] [--background] [--module <module>] [--learning-rate <lr>] [--warmup-steps <steps>]"
    exit 1
fi

INSTANCE_IP="$1"
shift

# Parse optional arguments
BACKGROUND=""
LEARNING_RATE=""
WARMUP_STEPS=""
K_START=""
COSINE_DECAY_EPOCH=""
VARIANCE=""
MODULE="jl.reward_model.reward_main"
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
        --k-start)
            K_START="$2"
            shift 2
            ;;
        --cosine-decay-epoch)
            COSINE_DECAY_EPOCH="$2"
            shift 2
            ;;
        --variance)
            VARIANCE="true"
            shift
            ;;
        --m2m100-variance)
            M2M100_VARIANCE="true"
            shift
            ;;
        --module)
            MODULE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown argument: $1"
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

log_info "SSH user: $SSH_USER"
if [[ -n "$SSH_PORT" ]]; then
    log_info "SSH port: $SSH_PORT"
fi

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
if [[ -n "$K_START" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --k-start $K_START"
    log_info "Using k-start: $K_START"
fi
if [[ -n "$COSINE_DECAY_EPOCH" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --cosine-decay-epoch $COSINE_DECAY_EPOCH"
    log_info "Using cosine decay from epoch: $COSINE_DECAY_EPOCH"
fi
if [[ -n "$VARIANCE" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --variance"
    log_info "Using variance mode"
fi
if [[ -n "${M2M100_VARIANCE:-}" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --m2m100-variance"
    log_info "Using M2M100 variance mode"
fi
log_info "Using module: $MODULE"

# Derive experiment type from module and generate timestamp
TIMESTAMP=$(date +"%m-%d-%H%M")
if [[ "$MODULE" == "jl.reward_model.reward_main" ]]; then
    EXPERIMENT_TYPE="reward_model"
elif [[ "$MODULE" == "jl.double_descent.resnet18.resnet18_main" ]]; then
    if [[ -n "$VARIANCE" ]]; then
        EXPERIMENT_TYPE="resnet18_variance"
    else
        EXPERIMENT_TYPE="resnet18"
    fi
elif [[ "$MODULE" == "jl.double_descent.transformer.transformer_main" ]]; then
    if [[ -n "${M2M100_VARIANCE:-}" ]]; then
        EXPERIMENT_TYPE="transformer_m2m100_variance"
    elif [[ -n "$VARIANCE" ]]; then
        EXPERIMENT_TYPE="transformer_variance"
    else
        EXPERIMENT_TYPE="transformer"
    fi
else
    EXPERIMENT_TYPE="other"
fi

OUTPUT_PATH="./output/$EXPERIMENT_TYPE/$TIMESTAMP"
log_info "Experiment type: $EXPERIMENT_TYPE"
log_info "Output path: $OUTPUT_PATH"

# Build the python command based on module
if [[ "$MODULE" == "jl.reward_model.reward_main" ]]; then
    PYTHON_CMD="python -m $MODULE --train-path ./data/tokenized --output-path $OUTPUT_PATH $EXTRA_FLAGS"
elif [[ "$MODULE" == "jl.double_descent.resnet18.resnet18_main" ]]; then
    PYTHON_CMD="python -m $MODULE --output-path $OUTPUT_PATH --data-path ./data $EXTRA_FLAGS"
elif [[ "$MODULE" == "jl.double_descent.transformer.transformer_main" ]]; then
    if [[ -n "${M2M100_VARIANCE:-}" ]]; then
        PYTHON_CMD="python -m $MODULE --output-path $OUTPUT_PATH --data-path ./data/iwslt14.m2m100.de-en $EXTRA_FLAGS"
    else
        PYTHON_CMD="python -m $MODULE --output-path $OUTPUT_PATH --data-path ./data/iwslt14.tokenized.de-en $EXTRA_FLAGS"
    fi
else
    # Generic module - just pass output-path
    PYTHON_CMD="python -m $MODULE --output-path $OUTPUT_PATH $EXTRA_FLAGS"
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

log_info "Connecting to $SSH_USER@$INSTANCE_IP..."

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
pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

echo '=== Starting training ==='
mkdir -p output data

# Run IWSLT preprocessing if using transformer module and data doesn't exist
if [[ $MODULE == jl.double_descent.transformer.transformer_main ]]; then
    if [[ -n '${M2M100_VARIANCE:-}' ]] && [[ ! -f data/iwslt14.m2m100.de-en/vocab_mapping.json ]]; then
        echo '=== Running M2M100 IWSLT preprocessing ==='
        pip install sentencepiece
        python -m jl.double_descent.transformer.prepare_m2m100_data
    elif [[ ! -f data/iwslt14.tokenized.de-en/train.de ]]; then
        echo '=== Running IWSLT preprocessing ==='
        ./infra/prepare_iwslt14.sh
    fi
fi

$PYTHON_CMD

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
pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

echo '=== Starting training in background ==='
mkdir -p output data

# Run IWSLT preprocessing if using transformer module and data doesn't exist
if [[ $MODULE == jl.double_descent.transformer.transformer_main ]]; then
    if [[ -n '${M2M100_VARIANCE:-}' ]] && [[ ! -f data/iwslt14.m2m100.de-en/vocab_mapping.json ]]; then
        echo '=== Running M2M100 IWSLT preprocessing ==='
        pip install sentencepiece
        python -m jl.double_descent.transformer.prepare_m2m100_data
    elif [[ ! -f data/iwslt14.tokenized.de-en/train.de ]]; then
        echo '=== Running IWSLT preprocessing ==='
        ./infra/prepare_iwslt14.sh
    fi
fi

nohup $PYTHON_CMD > training.log 2>&1 &

echo \"Training started in background. PID: \\\$!\"
echo \"Monitor with: ssh -i $SSH_KEY_PATH ${SSH_PORT:+-p $SSH_PORT} $SSH_USER@$INSTANCE_IP 'tail -f ~/variance-min-classification/training.log'\"
"

if [[ "$BACKGROUND" == "true" ]]; then
    log_info "Starting training in background mode..."
    ssh -i "$SSH_KEY_PATH" $SSH_OPTS "$SSH_USER@$INSTANCE_IP" bash <<< "$REMOTE_SCRIPT_BG"
    log_info "Training started in background on $INSTANCE_IP"
    log_info "Check progress: ssh -i $SSH_KEY_PATH ${SSH_PORT:+-p $SSH_PORT} $SSH_USER@$INSTANCE_IP 'tail -f ~/variance-min-classification/training.log'"
else
    log_info "Starting training (foreground mode)..."
    ssh -i "$SSH_KEY_PATH" $SSH_OPTS "$SSH_USER@$INSTANCE_IP" bash <<< "$REMOTE_SCRIPT"
fi
