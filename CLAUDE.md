# CLAUDE.md

Do not commit unless asked to.

## Environment

Activate the Python virtual environment before running any code:

```bash
source venv/bin/activate
```

When running tests or code, always start the venv as part of the same command:
```bash
source venv/bin/activate && python -m pytest tests/
```

When you make a change to a test or code module, run ALL tests affected by that change.

## Dependencies

Dependencies are managed via `pyproject.toml` with optional extras:

```bash
# Core only (CPU, no torch) — plotting, analysis
pip install -e .

# GPU experiments (double descent, etc.) — needs torch+torchvision installed separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[gpu]"

# LLM reward model training — adds transformers, flash attention, etc.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[gpu,llm]"
```

Note: torch and torchvision must be installed separately before the extras because CPU vs CUDA builds require different `--index-url` flags that pyproject.toml cannot express. On remote GPU instances, use `./scripts/setup.sh` which handles this automatically.

The `[llm]` extra is only needed for reward model training (`jl.reward_model`). All other experiments (ResNet18 double descent, Transformer double descent) only need `[gpu]`. The seq2seq transformer experiments use a custom small transformer — they do not use HuggingFace transformers.

## Project Structure

This project explores variance-minimizing classification models with various architectures and regularization strategies.

- `jl/config.py` - `Config` class holding all model/training hyperparameters with validation
- `jl/models.py` - Model architectures: `Resnet`, `MLP`, `MultiLinear`, `KPolynomial`, `SimpleMLP`
- `jl/model_creator.py` - Factory functions that instantiate models from a `Config`
- `jl/single_runner.py` - `train_once()` training loop (handles optimizers, schedulers, scaled regularization, weight tracking)
- `jl/ten_runner.py` - Multi-run training (`train_multi`) for ensemble experiments
- `jl/feature_experiments/` - Experiment scripts and specialized modules (dropout, scaled regularization, optimizers)
- `jl/posterior_minimizer/` - Posterior/variance analysis and weight tracking utilities
- `jl/reward_model/` - RLHF reward model training (HuggingFace transformers)

## Running Experiments

```bash
source venv/bin/activate
python -m jl.feature_experiments.run_single_feature
python -m jl.feature_experiments.run_feature_combinations
```

## Reward Model Training

### Local
```bash
source venv/bin/activate
python -m jl.reward_model.reward_main --train-path ./data/tokenized --output-path ./output
```

Optional: specify learning rate or warmup steps for hyperparameter experiments:
```bash
python -m jl.reward_model.reward_main --train-path ./data/tokenized --output-path ./output --learning-rate 3e-5
python -m jl.reward_model.reward_main --train-path ./data/tokenized --output-path ./output --warmup-steps 50
```

The learning rate schedule uses quadratic warmup (`lr = target_lr * (step/warmup_steps)²`) followed by cosine decay.

If the training data doesn't exist at `--train-path`, it will be downloaded and tokenized automatically.

Training outputs:
- `output/final_model.pt` - trained model weights
- `output/metrics.jsonl` - structured metrics (step, epoch, loss, accuracy, lr) for graphing

### Remote GPU Training (RunPod)

Training runs on RunPod GPU instances. SSH in and run commands directly.

**Prerequisites:**

1. Copy `.env.example` to `.env` and fill in your values:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. Required environment variables (set in `.env`):
   - `SSH_KEY_PATH` - Local path to SSH private key
   - `HF_TOKEN` - HuggingFace token (required for gated models/datasets)

**First-time setup** (or use `/setup_remote`):
```bash
source .env
ssh -i $SSH_KEY_PATH -p <port> root@<ip>

# On the remote box:
cd /root
git clone https://github.com/laurenjack/jl-research.git
cd jl-research
./scripts/setup.sh           # creates venv, installs torch + GPU deps
./scripts/setup.sh --llm     # also installs transformers, flash attention (reward model only)
./scripts/prepare_iwslt14.sh  # only needed for transformer experiments
```

**Running training:**
```bash
# SSH in
ssh -i $SSH_KEY_PATH -p <port> root@<ip>
cd /root/jl-research && source venv/bin/activate && source .env

# Run in background
nohup python -m jl.double_descent.resnet18.resnet18_main \
    --output-path ./output/resnet18/$(date +%m-%d-%H%M) --data-path ./data \
    > training.log 2>&1 &
```

**Monitoring:**
```bash
ssh -i $SSH_KEY_PATH -p <port> root@<ip> 'tail -f /root/jl-research/training.log'
```

**Downloading results** (or use `/download`):
```bash
scp -i $SSH_KEY_PATH -P <port> -r root@<ip>:/root/jl-research/output/<experiment>/<timestamp>/ ./data/<experiment>/<timestamp>/
```

**Helper scripts on the remote box:**

| Script | Purpose |
|--------|---------|
| `scripts/setup.sh` | Create venv, install deps, write .env |
| `scripts/prepare_iwslt14.sh` | Download and BPE-tokenize IWSLT'14 data |

**Notes:**
- SSH port varies per instance — check the RunPod dashboard

**NVIDIA MPS (required for multi-process / `--models-per-gpu > 1` / `--track-shadows` + clean-only side-by-side):**

Start the MPS control daemon on the remote BEFORE launching any multi-process training. Without it, processes on the same GPU time-slice instead of running concurrently — which silently makes those runs much slower or memory-blocked. Verify and start with:

```bash
ssh -i $SSH_KEY_PATH -p <port> root@<ip> "pgrep -f 'nvidia-cuda-mps-control -d' >/dev/null || nvidia-cuda-mps-control -d"
```

And pass the pipe directory to the training command so its workers attach to MPS:

```bash
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps nohup python -m jl.double_descent.transformer.transformer_main ...
```

This applies to: transformer `transformer_main`/`clean_only_main` with multiple d_models, resnet `resnet18_main`/`clean_only_main` with `--models-per-gpu > 1`, and any side-by-side launch of shadow + clean on the same GPU(s).

**Plotting metrics:**
```bash
source venv/bin/activate
python -m jl.reward_model.plot_metrics ./output/metrics.jsonl --output-dir ./data
```

## Double Descent Experiments (ResNet18)

Training ResNet18 models with varying width to reproduce double descent curves.

```bash
source venv/bin/activate
python -m jl.double_descent.resnet18.resnet18_main --output-path ./output --data-path ./data
```

**Plotting:**

```bash
source venv/bin/activate

# plot_evaluation: Final evaluation metrics vs k (auto-generated by trainer)
# Single figure with 2 subplots (error, loss)
python -m jl.double_descent.resnet18.plot_evaluation ./output/evaluation.jsonl --output-dir ./data

# plot_single_k: Epoch-wise training curves for a single k
# Single figure with 2 subplots (error vs epoch, loss vs epoch)
python -m jl.double_descent.resnet18.plot_single_k ./output --k 18 --output-dir ./data
```

## Transformer Double Descent Experiments

Training Transformers on IWSLT'14 de-en translation with varying d_model to reproduce Deep Double Descent Figure 3.

**Requirements:** Exactly 8 GPUs. The script trains 24 models automatically:
- 24 d_model values: 8, 16, 24, ..., 192
- 36K training samples
- 3 batches total, each training 8 models in parallel

```bash
source venv/bin/activate
python -m jl.double_descent.transformer.transformer_main --output-path ./output --data-path ./data/iwslt14.tokenized.de-en
```

Data preprocessing is automatic on remote instances (runs `./scripts/prepare_iwslt14.sh` if data not present).

**Output:** 24 metrics files: `metrics_d{8-192}_36k.jsonl`

**Plotting:**

```bash
source venv/bin/activate

# plot_evaluation: Final evaluation metrics vs d_model (auto-generated by trainer)
# Single figure with 2 subplots (loss + test BLEU, test accuracy)
python -m jl.double_descent.transformer.plot_evaluation ./output/evaluation.jsonl --output-dir ./data

# plot_single_d_model: Step-wise training curves for a single d_model and sample size
# Single figure with 2 subplots (loss vs step, accuracy vs step)
python -m jl.double_descent.transformer.plot_single_d_model ./output --d-model 128 --samples 36k --output-dir ./data
```

