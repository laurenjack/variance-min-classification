# jl-research

A mono repo for my ML research projects, my recent focus has been studying how we went [from double descent to the scaling laws](https://jacklaurenson.ai/blog/from-double-descent-to-scaling-laws/).

## What's in here

- **Novel bias-variance decomposition of the log loss**
- **Pre-trained LLMs are low variance**
- **Reproduction of Nakkrin et. al, 2019, with bias-variance decomposition**
- **Older variance minimization work**

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
