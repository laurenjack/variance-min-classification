---
name: setup_remote
description: Set up a remote RunPod GPU instance for running experiments — clone repo, install deps, prepare data.
---

# Setup Remote

Sets up a remote RunPod GPU instance.

## Inputs

- `instance_ip` (required): IP of the GPU instance
- `port` (required): SSH port (varies per RunPod instance — check the dashboard)

## Prerequisites

- `source .env` first to load `SSH_KEY_PATH`, `HF_TOKEN`, etc.
- A running RunPod instance with SSH access.
- **Push local changes first** with the `push` skill so the remote clones/pulls the latest code.

## Process

1. **Load env**: `source .env`.

2. **Clone or update repo on remote**:
   ```bash
   ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -p <port> root@<instance_ip> \
     'if [[ -d /root/variance-min-classification ]]; then \
        cd /root/variance-min-classification && git pull; \
      else \
        cd /root && git clone https://github.com/laurenjack/variance-min-classification.git; \
      fi'
   ```

3. **Run setup script** (default = GPU only; add `--llm` only for reward model training):
   ```bash
   ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -p <port> root@<instance_ip> \
     'cd /root/variance-min-classification && ./scripts/setup.sh'
   ```

   For reward model training:
   ```bash
   ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -p <port> root@<instance_ip> \
     'cd /root/variance-min-classification && HF_TOKEN='"'"'$HF_TOKEN'"'"' ./scripts/setup.sh --llm'
   ```

   This creates the venv, installs torch + GPU deps, writes API tokens to `.env`, creates `data/` and `output/`.

4. **Data prep** — ask which experiment:
   - **IWSLT14 (transformer DD)**:
     ```bash
     ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -p <port> root@<instance_ip> \
       'cd /root/variance-min-classification && source venv/bin/activate && ./scripts/prepare_iwslt14.sh'
     ```
   - **CIFAR-10/100 (ResNet18 DD)**: no prep — auto-downloads.

5. **Confirm**:
   ```bash
   ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -p <port> root@<instance_ip> \
     'cd /root/variance-min-classification && source venv/bin/activate && \
      python -c "import torch; print(f\"GPUs: {torch.cuda.device_count()}\")"'
   ```

## After Setup

```bash
ssh -i $SSH_KEY_PATH -p <port> root@<instance_ip>
cd /root/variance-min-classification && source venv/bin/activate && source .env
```
