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
├── evaluation.py          # Final metrics + ECE for main runs
├── plot_evaluation.py     # Main runs: error/loss/ECE vs k
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
cd /workspace/variance-min-classification && source venv/bin/activate && source .env
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
- `ts_ece`: ECE after temperature scaling

The shared utility lives in `jl/double_descent/temperature_scaling.py`.

---

## Phase 5: Plotting

Plots can be generated locally after downloading results, or via `/download` which runs them automatically.

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

