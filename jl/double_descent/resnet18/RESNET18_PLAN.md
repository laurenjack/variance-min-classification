# Plan: Reproduce Deep Double Descent Figure 1

## Goal
Reproduce the ResNet18 double-descent curve from Nakkiran et al. (2019) Figure 1:
- ResNet18 with varying width parameter k (1-64)
- CIFAR-10 with 15% label noise
- Train/test error AND loss vs model width (to show error and loss can be at odds)

## Approach
Train 8 models in parallel on an 8-GPU instance using `torch.multiprocessing`. Each GPU trains one model with a different k value, incrementing by 2. Default k=18. On 8 GPUs with default k-start=18, trains k=18,20,22,24,26,28,30,32.

---

## Phase 1: Package Structure

```
jl/double_descent/resnet18/
├── __init__.py
├── resnet18_config.py   # DDConfig dataclass
├── resnet18_data.py     # CIFAR-10 with label noise
├── resnet18_main.py     # Entry point (spawns 8 processes)
├── resnet18k.py         # Standard PreActResNet18
├── trainer.py           # Single-model training function
├── plot.py              # Visualization (5 plots)
└── RESNET18_PLAN.md     # This plan file
```

---

## Phase 2: Implementation Details

### 2.1 Config (`resnet18_config.py`)

```python
@dataclass
class DDConfig:
    # Training
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 0.0001
    optimizer: str = "adam"

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

```python
def main():
    # Parse args: --k-start, --output-path, --data-path, etc.

    # Check GPU count
    num_gpus = torch.cuda.device_count()
    if num_gpus < 8:
        raise RuntimeError(f"Requires 8 GPUs, found {num_gpus}")

    # Download data once before spawning
    download_cifar10(args.data_path)

    # Spawn 8 training processes
    mp.set_start_method('spawn')
    processes = []
    for gpu_id in range(8):
        k = args.k_start + 2 * gpu_id  # Increment by 2
        p = mp.Process(
            target=train_single_model,
            args=(gpu_id, k, config, args.output_path, args.data_path)
        )
        p.start()
        processes.append(p)

    # Wait for all to complete
    for p in processes:
        p.join()
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

### 2.6 Metrics Output

Each k value gets its own file: `output/metrics_k{k}.jsonl`

```json
{"epoch": 1, "k": 5, "train_error": 0.85, "test_error": 0.87, "train_loss": 2.31, "test_loss": 2.35}
{"epoch": 2, "k": 5, "train_error": 0.82, "test_error": 0.84, "train_loss": 2.12, "test_loss": 2.18}
...
```

**Four metrics per epoch:**
- `train_error`: 1 - train accuracy (classification error rate)
- `test_error`: 1 - test accuracy (classification error rate)
- `train_loss`: Cross-entropy loss on training set
- `test_loss`: Cross-entropy loss on test set

---

## Phase 3: Lambda Integration

### 3.1 Running Training

Manually provision an 8-GPU instance (8x V100, 8x A100, or 8x H100), then:

```bash
# Default: k-start=18, trains k=18,20,22,24,26,28,30,32 on 8 GPUs
./infra/lambda_train.sh <ip> --module jl.double_descent.resnet18.resnet18_main

# Or specify a different starting k (increments by 2 per GPU)
./infra/lambda_train.sh <ip> --module jl.double_descent.resnet18.resnet18_main --k-start 34
```

### 3.2 Downloading Results

```bash
./infra/lambda_download.sh <ip> --plot-module jl.double_descent.resnet18.plot
```

The plot module will combine all `metrics_k*.jsonl` files to generate plots.

---

## Phase 4: Testing Strategy

### 4.1 Local Smoke Test (if 8 GPUs available)
- k = 18,20,22,24,26,28,30,32 (default, incrementing by 2)
- 10 epochs
- Verify parallel training works, metrics files created

### 4.2 8-GPU Initial Run
- k = 18,20,22,24,26,28,30,32
- 500 epochs
- Monitor GPU utilization, training speed

### 4.3 Full Run
- Default run: k-start=18, produces k=18,20,22,24,26,28,30,32
- 500 epochs
- Produces metrics_k18.jsonl through metrics_k32.jsonl

---

## Phase 5: Plotting

Two plot scripts in `jl/double_descent/resnet18/`:

### `plot_vary_k.py` - Final metrics across k values
Plots final-epoch results for a range of k values (reproduces Figure 1):
- Single figure with 2 subplots:
  - Top: Train/Test error vs k
  - Bottom: Train/Test loss vs k

```bash
python -m jl.double_descent.resnet18.plot_vary_k ./output --min-k 18 --max-k 32 --output-dir ./data
```

### `plot_single_k.py` - Training curves for single k
Plots epoch-wise training for a specific k value:
- Single figure with 2 subplots:
  - Top: Train/Test error vs epoch
  - Bottom: Train/Test loss vs epoch

```bash
python -m jl.double_descent.resnet18.plot_single_k ./output --k 18 --output-dir ./data
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
| Epochs | 4000 | 500 |
| Width range | 1-64 | Default k=18,20,...,32 (increment by 2) |

---

## Hardware Requirements

- **8 GPUs required** - script will fail with clear error if fewer available
- Recommended: 8x V100 (16GB), 8x A100, or 8x H100 on Lambda Labs
- Each GPU trains one model independently
- Memory per GPU: ~2-4GB for k≤64 (ResNet18 is small)
