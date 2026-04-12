#!/bin/bash
# setup.sh - Set up this machine for training
#
# Run this on a fresh GPU instance (e.g. RunPod) after cloning the repo.
# Creates venv, installs deps (PyTorch CUDA + all extras), writes .env.
#
# Prerequisites:
#   Environment variables (optional, written to .env):
#     HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY
#
# Usage:
#   cd /workspace/variance-min-classification
#   ./infra/setup.sh

set -ex

# Write .env file with API tokens
cat > .env << 'EOF'
export HF_HOME='/workspace/.cache/huggingface'
EOF
[[ -n "${HF_TOKEN:-}" ]] && echo "export HF_TOKEN='${HF_TOKEN}'" >> .env
[[ -n "${KAGGLE_USERNAME:-}" ]] && echo "export KAGGLE_USERNAME='${KAGGLE_USERNAME}'" >> .env
[[ -n "${KAGGLE_KEY:-}" ]] && echo "export KAGGLE_KEY='${KAGGLE_KEY}'" >> .env

# Create venv if needed
if [[ ! -d 'venv' ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
# Use ./infra/setup.sh --llm to also install reward model deps (transformers, flash attention, etc.)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if [[ "${1:-}" == "--llm" ]]; then
    pip install -e ".[gpu,llm]"
    echo "Installed GPU + LLM dependencies"
else
    pip install -e ".[gpu]"
    echo "Installed GPU dependencies (use --llm for reward model training)"
fi

# Create data/output dirs
mkdir -p output data

echo '=== Setup complete ==='
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
