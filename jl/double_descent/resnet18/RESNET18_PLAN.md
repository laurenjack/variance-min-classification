# Plan: Reproduce Deep Double Descent Figure 1

## Goal
Reproduce the ResNet18 double-descent curve from Nakkiran et al. (2019) Figure 1:
- ResNet18 with varying width parameter k (1-64)
- CIFAR-10 with 15% label noise
- Train/test error AND loss vs model width (to show error and loss can be at odds)

## Approach

**Requires exactly 8 GPUs.** Two experiment modes:

### Default Mode (Double Descent)
Trains 8 models per batch with k values incrementing by 4:
- Default k-start=4: trains k=4, 8, 12, 16, 20, 24, 28, 32

### Variance Mode (Bias-Variance Decomposition)
Trains `num_splits` independently-initialised models per width k, each on a
disjoint chunk of CIFAR-10 train, to decompose test cross-entropy into bias
and variance terms vs k. See [Variance Mode](#variance-mode) below.

---

## Phase 1: Package Structure

```
jl/double_descent/resnet18/
├── __init__.py
├── resnet18_config.py     # DDConfig dataclass
├── resnet18_data.py       # CIFAR-10 with label noise
├── resnet18_main.py       # Entry point (requires 10 GPUs)
├── resnet18k.py           # Standard PreActResNet18
├── trainer.py             # Single-model training function (calls evaluation.py)
├── evaluation.py          # Final metrics for main runs
├── plot_evaluation.py     # Main runs: error/loss vs k
├── plot_single_k.py       # Training curves for single k
└── RESNET18_PLAN.md       # This plan file
```

---

## Phase 2: Implementation Details

### 2.1 Config (`resnet18_config.py`)

```python
@dataclass
class DDConfig:
    # Width parameter
    k_start: int = 4  # Starting k value, trains k, k+4, k+8, ..., k+4*(N-1)

    # Training
    epochs: int = 800
    batch_size: int = 128
    learning_rate: float = 0.001
    optimizer: str = "adam_w"
    cosine_decay_epoch: int = 100  # Cosine decay LR from this epoch

    # Data
    label_noise: float = 0.15
    data_augmentation: bool = True

    # Logging
    log_interval: int = 1  # Every epoch
```

### 2.2 ResNet18 (`resnet18k.py`)

Standard Pre-activation ResNet18 with width parameter k:
- Pre-activation: BN → ReLU → Conv
- 4 stages with widths [k, 2k, 4k, 8k]
- Strides [1, 2, 2, 2]
- Final: avg_pool → linear(8k → 10)

No masking needed - each model is built with its actual k value.

### 2.3 Data Loading (`resnet18_data.py`)

```python
def load_cifar10_with_noise(noise_prob: float, data_dir: str):
    """
    Load CIFAR-10 with label noise.

    - Download if needed (should be pre-downloaded before spawning processes)
    - Apply noise_prob corruption to training labels (fixed, sampled once)
    - Data augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip
    - Return train_loader, test_loader
    """
```

Each of the 8 processes loads data independently from the pre-downloaded files.

### 2.4 Multi-GPU Training (`resnet18_main.py`)

Requires exactly 10 GPUs. Trains 10 models per batch with hardcoded k values.

```python
# Hardcoded experiment parameters
REQUIRED_GPUS = 10
K_VALUES = [2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
NUM_BATCHES = 2
```

### 2.5 Single Model Trainer (`trainer.py`)

```python
def train_single_model(gpu_id: int, k: int, config: DDConfig, output_path: str, data_path: str):
    """Train a single ResNet18 with width k on the specified GPU."""
    device = torch.device(f"cuda:{gpu_id}")

    # Load data (each process loads independently)
    train_loader, test_loader = load_cifar10_with_noise(
        config.label_noise, config.batch_size, data_path
    )

    # Create model
    model = make_resnet18k(k=k, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Standard training loop
    for epoch in range(config.epochs):
        # Train
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

        # Evaluate and log
        train_error, train_loss = evaluate(model, train_loader, device)
        test_error, test_loss = evaluate(model, test_loader, device)
        log_metrics(output_path, k, epoch, train_error, test_error, train_loss, test_loss)
```

### 2.6 Output Structure

**Default mode** outputs to `output/resnet18/MM-DD-HHmm/`:

```
output/resnet18/03-01-1010/
├── metrics_k4.jsonl
├── metrics_k8.jsonl
├── ...
├── metrics_k32.jsonl
├── model_k4.pt
├── model_k8.pt
├── ...
└── model_k32.pt
```

**Metrics format (JSONL):**
```json
{"epoch": 1, "k": 4, "train_error": 0.85, "test_error": 0.87, "train_loss": 2.31, "test_loss": 2.35}
```

**Four metrics per epoch:**
- `train_error`: 1 - train accuracy (classification error rate)
- `test_error`: 1 - test accuracy (classification error rate)
- `train_loss`: Cross-entropy loss on training set
- `test_loss`: Cross-entropy loss on test set

---

## Phase 3: Remote Training

### 3.1 Running Training

Manually provision an 8-GPU instance (8x V100, 8x A100, or 8x H100), then:

```bash
# SSH into the remote instance, then:
cd /root/variance-min-classification && source venv/bin/activate && source .env
python -m jl.double_descent.resnet18.resnet18_main \
    --output-path ./output/resnet18/$(date +%m-%d-%H%M) --data-path ./data
```

### 3.2 Downloading Results

Use `/download` or scp results manually. Then run plot scripts locally.

---

## Phase 4: Testing Strategy

### 4.1 Local Smoke Test (if 8 GPUs available)
- k = 72,80,88,96,104,112,120,128 (default, incrementing by 8)
- 10 epochs
- Verify parallel training works, metrics files created

### 4.2 8-GPU Initial Run
- k = 72,80,88,96,104,112,120,128
- 800 epochs
- Monitor GPU utilization, training speed

### 4.3 Full Run
- Default run: k-start=4, produces k=4,8,12,16,20,24,28,32
- 800 epochs
- Produces metrics_k4.jsonl through metrics_k32.jsonl

---

## Temperature Scaling

Temperature scaling is integrated into `evaluation.py`. When `--val-split` is used, the evaluation automatically fits a scalar temperature T on the 5K val set (L-BFGS, lr=1.0) and evaluates on the 10K test set. The following fields are added to each `evaluation.jsonl` row:

- `temperature`: fitted T value
- `ts_loss`: test cross-entropy after dividing logits by T
- `ts_error`: test error after temperature scaling

The shared utility lives in `jl/double_descent/temperature_scaling.py`.

---

## Shadow Tracking (`--track-shadows`)

Decomposes the AdamW trajectory into two ground-truth buckets so we can
project per-bucket cumulative weight updates onto the trained model and
test the "support-vector" reading of double descent on CIFAR-10 with
15% label noise.

- **Bucket 0 (clean)**: training images whose noisy label equals the
  original CIFAR label (~85% of the 45K train pool).
- **Bucket 1 (mislabeled)**: training images whose noisy label differs
  from the original (~15%). Built from the ground-truth mislabel mask
  — we know which images we flipped.

### How it works

Per step:
1. One forward pass to get logits and per-sample CE losses.
2. Two backward passes (one per bucket): `L_b = (per_sample_loss *
   mask_b).sum() / batch_size`, gradients computed via
   `torch.autograd.grad`.
3. AdamW update split per bucket using a per-bucket first moment `m_b`
   and a shared second moment `v` (same trick the transformer
   `bucket_shadow_trainer` uses). The per-bucket increment
   `−lr · m̂_b / (√v̂ + ε)` accumulates into `shadow_b`.
4. Weight decay is W's self-shrinkage and is NOT attributed to any
   bucket.

Roughly 3× per-step cost vs. the standard trainer.

### Running

```bash
python -m jl.double_descent.resnet18.resnet18_main \
    --output-path ./output/resnet18_shadows/$(date +%m-%d-%H%M) \
    --data-path ./data \
    --track-shadows
```

Incompatible with `--variance`. Implies `--val-split` (the shadow trainer
always carves a val set for the early-stop checkpoint).

### Outputs (per width k)

```
output/resnet18_shadows/MM-DD-HHmm/
├── metrics_k{K}.jsonl            # per-epoch incl. shadow_norms + shadow_shares
├── model_k{K}.pt                  # final model
├── bucket_shadows_k{K}.pt         # {shadows[2], param_names, bucket sizes}
├── bucket_shares_k{K}.json        # final L2 norms + shares + best_val_*
└── early_stop/
    └── model_k{K}.pt              # best-val checkpoint
```

### Post-hoc projection

Use `jl.double_descent.transformer.extract_projection_shares` (script is
model-agnostic — operates on saved shadow files of any shape):

```bash
python -m jl.double_descent.transformer.extract_projection_shares \
    ./output/resnet18_shadows/MM-DD-HHmm \
    --reference final_weight \
    --model-dir ./output/resnet18_shadows/MM-DD-HHmm
```

Computes `c_b = ⟨Δ_b, W_T⟩ / ‖W_T‖²` per bucket (the scalar projection
coefficient onto `W_T`); `Σ_b c_b` reports what fraction of `W_T`'s
magnitude was constructed by gradients along `W_T`'s direction.

---

## Variance Mode

Trains `num_splits` ResNet18 models per width k on disjoint subsets of CIFAR-10
train. Used to decompose test cross-entropy into a bias term and a variance
term (Jensen Gap) across model width.

### Setup
- 50K CIFAR-10 train shuffled with a fixed seed, then split into `num_splits`
  disjoint chunks (e.g. 4 × 12.5K).
- 15% label noise applied independently within each chunk (different seed per
  split, so corruption is not shared across models).
- Test set: standard CIFAR-10 test (10K, clean), shared across splits — same
  distribution as the splits since CIFAR's test set is i.i.d. with train.
- Each (k, split_id) pair trains one model through the modern trainer (BF16
  autocast, GPU-resident data pipeline, ES checkpointing on val loss).

### Decomposition

For a single test sample with true label y and `N = num_splits` model
distributions `q_j` over classes:

```
loss_j         = -log q_j(y)                          (per-model NLL)
q_bar(y)       = (1/N) · Σ_j q_j(y)                   (ensemble probability)
bias           = -log q_bar(y)
jensen_gap     = log q_bar(y) − (1/N) · Σ_j log q_j(y)   (≥ 0, the variance term)

mean_test_loss = bias + jensen_gap
```

### Running

```bash
python -m jl.double_descent.resnet18.resnet18_main \
    --variance --num-splits 4 \
    --output-path ./output/resnet18_variance/$(date +%m-%d-%H%M) \
    --data-path ./data
```

With `--num-splits 4` and 8 GPUs, each batch trains 2 k-values × 4 splits = 8
models. The default k sweep is reused — same `K_VALUES` list as default mode.

### Output Structure

```
output/resnet18_variance/MM-DD-HHmm/
├── metrics_k{K}_split{S}.jsonl
├── model_k{K}_split{S}.pt
├── evaluation.jsonl                  # FINAL bias/variance per k
└── early_stop/
    ├── model_k{K}_split{S}.pt        # Best-val checkpoint per (k, split)
    └── evaluation.jsonl              # ES bias/variance per k
```

### Evaluation + Plotting

```bash
# Compute mean_test_loss + Jensen Gap from saved checkpoints
python -m jl.double_descent.resnet18.variance_evaluation \
    --model-path ./data/resnet18_variance/MM-DD-HHmm \
    --data-path ./data

# Plot bias-variance decomposition
python -m jl.double_descent.resnet18.plot_variance_evaluation \
    ./data/resnet18_variance/MM-DD-HHmm/evaluation.jsonl \
    --output-dir ./data/resnet18_variance/MM-DD-HHmm
```

`variance_evaluation.py` runs on the FINAL checkpoints by default. Pass
`--early-stop` to evaluate the `early_stop/` checkpoints instead; output is
written to `early_stop/evaluation.jsonl`. Run the plot script twice (once per
evaluation.jsonl) to compare FINAL vs ES bias-variance.

---

## Phase 5: Plotting

Plots can be generated locally after downloading results, or via `/download` which runs them automatically.

### `plot_evaluation.py` - Final metrics across k values (main runs)
Plots evaluation metrics for main (non-variance) runs:
- Single figure with 2 subplots:
  - Top: Train/Test error vs k
  - Bottom: Train/Test loss vs k

Generated automatically by trainer. Can also run manually:
```bash
python -m jl.double_descent.resnet18.plot_evaluation ./data/resnet18/03-01-1010/evaluation.jsonl
```

Output: `data/resnet18/03-01-1010/resnet18_evaluation.png`

### `plot_single_k.py` - Training curves for single k
Plots epoch-wise training for a specific k value:
- Single figure with 2 subplots:
  - Top: Train/Test error vs epoch
  - Bottom: Train/Test loss vs epoch

```bash
python -m jl.double_descent.resnet18.plot_single_k ./data/resnet18/03-01-1010 --k 72
```

---

## Implementation Order

1. Simplify `resnet18_config.py` - remove width_min/width_max
2. Simplify `resnet18k.py` - remove masking, use standard BatchNorm
3. Remove `masked_batchnorm.py` - no longer needed
4. Update `resnet18_data.py` - add download-only function
5. Rewrite `trainer.py` - single model training function
6. Rewrite `resnet18_main.py` - multiprocessing with 8-GPU check
7. Update `plot.py` - load multiple metrics files
8. Test on 8-GPU instance

---

## Key Differences from Paper

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Trials | 5 | 1 |
| Training | Sequential per k | 8 models parallel on 8 GPUs |
| BatchNorm | Standard | Standard |
| Epochs | 4000 | 800 |
| Width range | 1-64 | Default k=4,8,...,32 (increment by 4) |
| k increment | 1 | 4 |

---

## Hardware Requirements

- **Exactly 8 GPUs required** - script fails fast otherwise
- Recommended: 8x V100 (16GB), 8x A100, or 8x H100 on Lambda Labs or RunPod
- Each GPU trains one model independently
- Memory per GPU: ~2-4GB for k≤64, larger models need more

