# variance-min-classification

Reproducing double descent experiments — see the accompanying blog post [From Double Descent to Scaling Laws](https://jacklaurenson.ai/blog/from-double-descent-to-scaling-laws/) for context and results.

## What's in here

- **ResNet18 on CIFAR-10** with 15% label noise — a reproduction of the Nakkiran et al. (2019) deep double descent result, plus a bias-variance decomposition of the test loss.
- **Transformer seq2seq on IWSLT'14 de-en** — same double descent phenomenon in the original Vaswani et al. transformer architecture, trained on 36K sequence pairs.

The repo also contains earlier variance-minimization experiments (`jl/feature_experiments/`, `jl/posterior_minimizer/`) and an RLHF reward-model training pipeline (`jl/reward_model/`).

## Running it

```bash
source venv/bin/activate

# ResNet18 double descent
python -m jl.double_descent.resnet18.resnet18_main --output-path ./output --data-path ./data

# Transformer double descent (requires 8 GPUs)
python -m jl.double_descent.transformer.transformer_main --output-path ./output --data-path ./data/iwslt14.tokenized.de-en
```

See [`CLAUDE.md`](./CLAUDE.md) for full setup, plotting commands, and the reward-model pipeline.

## GPU infrastructure

Training runs on [RunPod](https://www.runpod.io) GPU instances. First-time setup is automated:

```bash
./scripts/setup.sh           # creates venv, installs torch + GPU deps
./scripts/prepare_iwslt14.sh # downloads + tokenizes IWSLT'14 (transformer experiments only)
```

### NVIDIA MPS

The experiments train many small models in parallel on the same GPU (`--models-per-gpu > 1`, or 8 transformers per batch on 8 GPUs). Without the NVIDIA Multi-Process Service (MPS), processes on the same GPU **time-slice** rather than run concurrently — silently slowing every run. Start the MPS control daemon on the remote before launching training:

```bash
ssh -i $SSH_KEY_PATH -p <port> root@<ip> \
  "pgrep -f 'nvidia-cuda-mps-control -d' >/dev/null || nvidia-cuda-mps-control -d"
```

And pass the pipe directory to the training command so its workers attach to MPS:

```bash
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps nohup python -m jl.double_descent.transformer.transformer_main ...
```
