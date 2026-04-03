#!/bin/bash
# setup_remote.sh - Set up a remote GPU instance for training
#
# Clones/pulls repo, creates venv, installs deps, writes env vars.
# After running this, the remote is ready to execute any training command.
#
# Prerequisites:
#   - SSH_KEY_PATH environment variable set
#   - HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY set (optional, written to remote .env)
#
# Usage:
#   ./infra/setup_remote.sh <instance_ip> [--user <user>] [--port <port>]
#
# Examples:
#   ./infra/setup_remote.sh 192.222.54.255
#   ./infra/setup_remote.sh 192.222.54.255 --user root --port 22005  # RunPod

set -euo pipefail

SSH_KEY_PATH="${SSH_KEY_PATH:?SSH_KEY_PATH not set}"
REPO_URL="https://github.com/laurenjack/variance-min-classification.git"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

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

if [[ ! -f "$SSH_KEY_PATH" ]]; then
    log_error "SSH private key not found at $SSH_KEY_PATH"
    exit 1
fi

chmod 600 "$SSH_KEY_PATH" 2>/dev/null || true

log_info "Setting up $SSH_USER@$INSTANCE_IP..."

# Build remote .env content from local env vars
REMOTE_ENV=""
[[ -n "${HF_TOKEN:-}" ]] && REMOTE_ENV="${REMOTE_ENV}export HF_TOKEN='${HF_TOKEN}'\n"
[[ -n "${KAGGLE_USERNAME:-}" ]] && REMOTE_ENV="${REMOTE_ENV}export KAGGLE_USERNAME='${KAGGLE_USERNAME}'\n"
[[ -n "${KAGGLE_KEY:-}" ]] && REMOTE_ENV="${REMOTE_ENV}export KAGGLE_KEY='${KAGGLE_KEY}'\n"

ssh -i "$SSH_KEY_PATH" $SSH_OPTS "$SSH_USER@$INSTANCE_IP" bash <<SETUP_EOF
set -ex

cd ~

# Clone or pull repo
if [[ -d 'variance-min-classification' ]]; then
    cd variance-min-classification
    git pull
else
    git clone $REPO_URL
    cd variance-min-classification
fi

# Write .env file with API tokens
cat > .env << 'ENV_EOF'
$(echo -e "$REMOTE_ENV")
ENV_EOF

# Create venv if needed
if [[ ! -d 'venv' ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

# Create data/output dirs
mkdir -p output data

echo '=== Setup complete ==='
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
SETUP_EOF

log_info "Remote setup complete."
log_info "SSH in with: ssh -i $SSH_KEY_PATH ${SSH_PORT:+-p $SSH_PORT} $SSH_USER@$INSTANCE_IP"
log_info "Then: cd ~/variance-min-classification && source venv/bin/activate && source .env"
