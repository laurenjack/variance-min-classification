# jl-research

A repo for my personal ML research projects, my recent focus has been studying how we went [from double descent to the scaling laws](https://jacklaurenson.ai/blog/from-double-descent-to-scaling-laws/).

## What's in here

- **Novel bias-variance decomposition of the log loss** — trains many ResNet18 / Transformer models on disjoint training splits and decomposes the test cross-entropy as `CE(p, q̄) = H(p) + KL(p ‖ q̄) + Jensen Gap` — entropy, bias, and variance terms, all non-negative. For the IWSLT'14 transformer, an M2M100-12B oracle supplies the true distribution `p` so the full distributional decomposition is recovered rather than just the label-only Jensen Gap (`jl/double_descent/{resnet18,transformer}/variance_evaluation.py`).
- **Pre-trained LLMs are low variance** — Bradley-Terry reward-model training on top of Llama-family pre-trained LLMs (Anthropic HH-RLHF, HelpSteer2-Preference). Training past the interpolation threshold continues to improve the held-out win rate — the regime the decomposition above predicts when the underlying pre-trained model is already low variance (`jl/reward_model/`).
- **Older variance minimization work** — earlier synthetic-problem experiments (SingleFeatures, Kaleidoscope) and posterior / weight-tracking analyses that originally motivated this line of work, studying how architecture and regularization shift model variance directly (`jl/feature_experiments/`, `jl/posterior_minimizer/`).

## Running it

See [`CLAUDE.md`](./CLAUDE.md) for full setup.

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
