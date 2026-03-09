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

### Variance Mode (`--variance` flag)
Trains 64 models to analyze variance across different training data:
- 16 k values: 4, 8, 12, 16, ..., 64 (increment by 4)
- 4 disjoint training splits (12.5K samples each from 50K)
- 15% label noise applied independently to each split
- 8 batches total (2 k values × 4 splits per batch)

---

## Phase 1: Package Structure

```
jl/double_descent/resnet18/
├── __init__.py
├── resnet18_config.py     # DDConfig dataclass
├── resnet18_data.py       # CIFAR-10 with label noise + disjoint splits
├── resnet18_main.py       # Entry point (requires 8 GPUs, --variance flag)
├── resnet18k.py           # Standard PreActResNet18
├── trainer.py             # Single-model training function
├── plot_vary_k.py         # Final metrics plot across k values
├── plot_single_k.py       # Training curves for single k
├── evaluate.py            # Variance evaluation (Jensen Gap with Bessel's correction)
├── plot_evaluation.py     # Bias-variance decomposition plot
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

Requires exactly 8 GPUs. Two modes:

**Default mode:** Trains 8 models with k incrementing by 4.

**Variance mode (`--variance`):** Trains 2 k values × 4 splits = 8 models per batch.

```python
# Hardcoded experiment parameters
REQUIRED_GPUS = 8
K_INCREMENT = 4

# Variance experiment parameters
VARIANCE_K_VALUES = list(range(4, 68, 4))  # [4, 8, 12, ..., 64] - 16 values
VARIANCE_NUM_SPLITS = 4  # 4 disjoint training sets of 12.5K each

def main():
    # Require exactly 8 GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus != REQUIRED_GPUS:
        raise RuntimeError(f"Requires exactly 8 GPUs, found {num_gpus}")

    if args.variance:
        run_variance(args, config)  # 8 batches: 2 k × 4 splits
    else:
        run_default(args, config)   # 1 batch: 8 k values
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

**Variance mode** outputs to `output/resnet18_variance/MM-DD-HHmm/`:

```
output/resnet18_variance/03-01-1010/
├── metrics_k4_split0.jsonl
├── metrics_k4_split1.jsonl
├── metrics_k4_split2.jsonl
├── metrics_k4_split3.jsonl
├── ...
├── metrics_k64_split3.jsonl
├── model_k4_split0.pt
├── ...
├── model_k64_split3.pt
└── evaluation.jsonl  # After running evaluate.py
```

**Metrics format (JSONL):**
```json
{"epoch": 1, "k": 4, "train_error": 0.85, "test_error": 0.87, "train_loss": 2.31, "test_loss": 2.35}
{"epoch": 1, "k": 4, "split_id": 0, "train_error": 0.85, ...}  # Variance mode
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
# Default mode: k-start=4, trains k=4,8,12,16,20,24,28,32 on 8 GPUs
./infra/train.sh <ip> --module jl.double_descent.resnet18.resnet18_main

# Specify a different starting k (increments by 4 per GPU)
./infra/train.sh <ip> --module jl.double_descent.resnet18.resnet18_main --k-start 36

# Variance mode: 16 k values × 4 splits = 64 models
./infra/train.sh <ip> --module jl.double_descent.resnet18.resnet18_main --variance
```

### 3.2 Downloading Results

```bash
./infra/download.sh <ip>
```

Downloads all metrics and model files to `data/resnet18/MM-DD-HHmm/` and auto-generates plots.

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

### 4.4 Variance Run
- Variance mode: 16 k values × 4 splits = 64 models
- 8 batches (2 k per batch × 4 splits)
- 800 epochs per model
- Produces metrics_k4_split0.jsonl through metrics_k64_split3.jsonl

---

## Variance Experiment Details

### Data Split Strategy

The 50K training samples are shuffled with a fixed seed, then partitioned into 4 disjoint 12.5K chunks:
- Split 0: indices 0-12499
- Split 1: indices 12500-24999
- Split 2: indices 25000-37499
- Split 3: indices 37500-49999

15% label noise is applied independently to each split after partitioning.

### Execution Order

| Batch | k values | Splits |
|-------|----------|--------|
| 1     | 4, 8     | 0, 1, 2, 3 each |
| 2     | 12, 16   | 0, 1, 2, 3 each |
| 3     | 20, 24   | 0, 1, 2, 3 each |
| 4     | 28, 32   | 0, 1, 2, 3 each |
| 5     | 36, 40   | 0, 1, 2, 3 each |
| 6     | 44, 48   | 0, 1, 2, 3 each |
| 7     | 52, 56   | 0, 1, 2, 3 each |
| 8     | 60, 64   | 0, 1, 2, 3 each |

### Evaluation

After training, run evaluation on a GPU instance:

```bash
python -m jl.double_descent.resnet18.evaluate \
    --model-path ./output/resnet18_variance/03-01-1010 \
    --data-path ./data
```

This computes for each k:
- **Mean test loss**: Cross-entropy loss averaged across the 4 training splits
- **Jensen Gap**: E[log(q_bar[y] / q_j[y])] with Bessel's correction (n-1=3)
- **Implied Bias**: test_loss - jensen_gap

Output: `evaluation.jsonl` alongside the model files.

### Temperature Scaling

Optional post-hoc calibration to demonstrate that simple calibration cannot recover first-descent loss:

```bash
python -m jl.double_descent.resnet18.evaluate \
    --model-path ./output/resnet18_variance/03-01-1010 \
    --data-path ./data --temperature-scaling
```

This fits a scalar temperature T per k value using L-BFGS on one randomly chosen model's test NLL, then recomputes the full bias-variance decomposition with `softmax(logits/T)` across all models.

Output: `temperature-scaled/evaluation.jsonl` in the model directory (same schema plus a `"temperature"` field).

### Expected Calibration Error (ECE)

Compute ECE (M=20 equal-width bins) using only split 0 models on the test set:

```bash
python -m jl.double_descent.resnet18.evaluate \
    --model-path ./output/resnet18_variance/03-01-1010 \
    --data-path ./data --ece
```

One model per k, parallelized across all available GPUs. For each prediction the confidence is max(softmax), binned into 20 equal-width intervals.

Output: `ece.jsonl` alongside the model files (one line per k with `{"k": N, "ece": X, "num_samples": Y}`).

### Plotting

```bash
python -m jl.double_descent.resnet18.plot_evaluation \
    ./output/resnet18_variance/03-01-1010/evaluation.jsonl \
    --output-dir ./data
```

Produces `bias_variance.png` showing test loss, Jensen Gap, and implied bias vs k.

For temperature-scaled results:

```bash
python -m jl.double_descent.resnet18.plot_evaluation \
    ./output/resnet18_variance/03-01-1010/temperature-scaled/evaluation.jsonl \
    --output-dir ./data --temperature-scaled
```

Produces `bias_variance.png` with "(Temperature Scaled)" in the title.

For ECE results:

```bash
python -m jl.double_descent.resnet18.plot_ece \
    ./output/resnet18_variance/03-01-1010/ece.jsonl \
    --output-dir ./data
```

Produces `ece.png` showing ECE vs k.

---

## Phase 5: Plotting

Plots are auto-generated by `download.sh` and saved alongside metrics in the run folder.

### `plot_vary_k.py` - Final metrics across k values
Plots final-epoch results for a range of k values (reproduces Figure 1):
- Single figure with 2 subplots:
  - Top: Train/Test error vs k
  - Bottom: Train/Test loss vs k

```bash
# Output saved to same directory as metrics
python -m jl.double_descent.resnet18.plot_vary_k ./data/resnet18/03-01-1010 --min-k 72 --max-k 128
```

Output: `data/resnet18/03-01-1010/resnet18_vary_k_72_to_128.png`

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
| Trials | 5 | 1 (default) or 4 (variance) |
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
