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
├── trainer.py             # Single-model training function (calls evaluation.py)
├── evaluation.py          # Final metrics + ECE for main runs
├── plot_evaluation.py     # Main runs: error/loss/ECE vs k
├── plot_single_k.py       # Training curves for single k
├── variance_evaluation.py # Variance mode: Jensen Gap decomposition
├── plot_variance_evaluation.py  # Bias-variance decomposition plot
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

After training, run variance evaluation on a GPU instance:

```bash
python -m jl.double_descent.resnet18.variance_evaluation \
    --model-path ./output/resnet18_variance/03-01-1010 \
    --data-path ./data
```

This computes for each k:
- **Mean test loss**: Cross-entropy loss averaged across the 4 training splits
- **Jensen Gap**: E[log(q_bar[y] / q_j[y])]
- **Entropy + Bias**: test_loss - jensen_gap

Output: `evaluation.jsonl` alongside the model files.

### Plotting

```bash
python -m jl.double_descent.resnet18.plot_variance_evaluation \
    ./output/resnet18_variance/03-01-1010/evaluation.jsonl \
    --output-dir ./data
```

Produces `bias_variance.png` showing test loss, Jensen Gap, and implied bias vs k.

---

## Phase 5: Plotting

Plots are auto-generated by `download.sh` and saved alongside metrics in the run folder.

### `plot_evaluation.py` - Final metrics across k values (main runs)
Plots evaluation metrics for main (non-variance) runs:
- Single figure with 2 subplots:
  - Top: Train/Test error vs k
  - Bottom: Train/Test loss vs k, with ECE on right y-axis

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

## Phase 6: Final-Layer Fine-Tuning

Fine-tunes only the final linear layer (`model.linear`) of each trained model using L-BFGS or SGD with L2 regularization.

### Running Fine-Tuning

```bash
# L-BFGS (default)
python -m jl.double_descent.resnet18.fine_tune \
    --model-path ./output/resnet18/03-01-1010 \
    --data-path ./data \
    --l2-lambda 1e-5 --max-steps 100

# SGD
python -m jl.double_descent.resnet18.fine_tune \
    --model-path ./output/resnet18/03-01-1010 \
    --data-path ./data \
    --sgd --l2-lambda 1e-5 --sgd-epochs 100 --sgd-lr 0.01
```

- Discovers all `model_k*.pt` files (excludes variance/split models)
- Extracts features from frozen backbone (no data augmentation, BatchNorm in eval mode)
- Fine-tunes a standalone copy of the final linear layer
- Parallelizes across available GPUs (one model per GPU)
- Saves only the fine-tuned layer weights (not full model)

### Output

L-BFGS writes to `fine_tuned/lambda_*/`, SGD writes to `fine_tuned/sgd_lambda_*/`:

```
output/resnet18/03-01-1010/fine_tuned/lambda_1e-05/
├── layer_k4.pt                    # Final layer state_dict only
├── layer_k8.pt
├── ...
└── fine_tune_metadata.jsonl       # {k, final_loss, final_grad_norm, steps, l2_lambda}
```

### Evaluation (shared with Transformer, requires GPU)

Computes original and fine-tuned test loss, test error, and ECE, parallelized across all available GPUs.
Pass the layer directory directly:

```bash
python -m jl.double_descent.fine_tune_evaluation --fine-tune \
    --resnet-path ./output/resnet18/03-01-1010 \
    --resnet-layer-dir ./output/resnet18/03-01-1010/fine_tuned/lambda_1e-03 \
    --data-path ./data
```

Output: `fine_tune_evaluation.jsonl` in the layer directory, with schema:
```json
{"k": 4, "original_loss": 1.23, "fine_tuned_loss": 1.10, "original_error": 0.15, "fine_tuned_error": 0.14, "original_ece": 0.08, "fine_tuned_ece": 0.03}
```

### Plotting (shared with Transformer, no GPU required)

```bash
python -m jl.double_descent.plot_fine_tune \
    --resnet-ft-eval ./data/resnet18/03-01-1010/fine_tuned/lambda_1e-03/fine_tune_evaluation.jsonl \
    --resnet-ts-eval ./data/resnet18/03-01-1010/temperature_scaled/temperature_scaled_evaluation.jsonl \
    --output-dir ./data
```

Produces `fine_tune_comparison.png` with original vs fine-tuned vs temp-scaled test loss, error, and ECE.

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
