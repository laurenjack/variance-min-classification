# Plan: Reproduce Deep Double Descent Figure 1

## Goal
Reproduce the ResNet18 double-descent curve from Nakkiran et al. (2019) Figure 1:
- ResNet18 with varying width parameter k (1-64)
- CIFAR-10 with 15% label noise
- Train/test error vs model width

## Approach
Use channel masking with vmap to train all width configurations in parallel on H100.

---

## Phase 1: Package Structure

Create `jl/double_descent/` mirroring `jl/reward_model/`:

```
jl/double_descent/
├── __init__.py
├── config.py          # DDConfig dataclass
├── resnet18k.py       # PreActResNet18 with channel masking
├── masked_batchnorm.py # MaskedBatchNorm2d implementation
├── data.py            # CIFAR-10 with label noise
├── trainer.py         # Parallel training with vmap
├── main.py            # Entry point
└── deep_double_descent.pdf  # (already exists)
```

---

## Phase 2: Implementation Details

### 2.1 Config (`config.py`)

```python
@dataclass
class DDConfig:
    # Model
    width_min: int = 1
    width_max: int = 64  # Start with 16 for testing

    # Training (from paper)
    epochs: int = 4000
    batch_size: int = 128
    learning_rate: float = 0.0001
    optimizer: str = "adam"

    # Data
    label_noise: float = 0.15
    data_augmentation: bool = True

    # Logging
    log_interval: int = 1  # Every epoch
```

### 2.2 Masked BatchNorm (`masked_batchnorm.py`)

Implement `MaskedBatchNorm2d`:
- Compute running mean/var only over active channels (mask == 1)
- Zero output for inactive channels
- Handle the per-channel γ, β parameters correctly

### 2.3 ResNet18 with Masking (`resnet18k.py`)

Copy `PreActResNet` from `double-descent/models/resnet18k.py` and modify:

1. Replace `BatchNorm2d` with `MaskedBatchNorm2d`
2. Add `width_mask` parameter to forward pass (shape: `[k_max]`)
3. After each conv layer, apply channel mask: `out = out * mask.view(1, -1, 1, 1)`
4. Shortcut convolutions also need masking

Key architecture (from paper):
- Pre-activation ResNet18: BN → ReLU → Conv
- 4 blocks with widths [k, 2k, 4k, 8k]
- Strides [1, 2, 2, 2]
- Final: avg_pool → linear(8k → 10)

### 2.4 Data Loading (`data.py`)

```python
def load_cifar10_with_noise(noise_prob: float, data_dir: str):
    """
    Load CIFAR-10 with label noise.

    - Download if needed
    - Apply noise_prob corruption to training labels (fixed, sampled once)
    - Data augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip
    - Return train_loader, test_loader
    """
```

### 2.5 Parallel Trainer (`trainer.py`)

Adapt from `multi_runner.py`:

1. **Build models**: Create `width_max - width_min + 1` copies of ResNet18(k=k_max)
2. **Stack parameters**: Use `stack_module_state`
3. **Create width masks**:
   ```python
   # For k in [1, 2, ..., 16], mask first k channels of each layer
   # Layer 1: mask[:k], Layer 2: mask[:2k], etc.
   ```
4. **Vectorized forward/backward**: Use `vmap` over the width dimension
5. **Training loop**:
   - For each epoch:
     - For each batch:
       - Forward all widths in parallel
       - Compute cross-entropy loss per width
       - Backward all widths in parallel
       - Update parameters
     - Evaluate train/test error for all widths
     - Log metrics to JSONL

### 2.6 Metrics Output

Store in `output/metrics.jsonl`:
```json
{"epoch": 1, "k": 1, "train_error": 0.85, "test_error": 0.87, "train_loss": 2.31, "test_loss": 2.35}
{"epoch": 1, "k": 2, "train_error": 0.82, "test_error": 0.84, "train_loss": 2.12, "test_loss": 2.18}
...
{"epoch": 4000, "k": 64, "train_error": 0.0, "test_error": 0.28, "train_loss": 0.001, "test_loss": 0.95}
```

**Four metrics per (epoch, k):**
- `train_error`: 1 - train accuracy (classification error rate)
- `test_error`: 1 - test accuracy (classification error rate)
- `train_loss`: Cross-entropy loss on training set
- `test_loss`: Cross-entropy loss on test set

This captures the key phenomenon: loss and error can be at odds, especially near the interpolation threshold where train_loss → 0 but test_loss may spike.

---

## Phase 3: Lambda Integration

### 3.1 Modify `lambda_train.sh`

Add `--module` argument:
```bash
./infra/lambda_train.sh <ip> --module jl.double_descent.main
```

Changes:
- Parse `--module` argument (default: `jl.reward_model.reward_main`)
- Pass module to remote script

### 3.2 Update CLAUDE.md

Document the double descent training command.

---

## Phase 4: Testing Strategy

### 4.1 Local Smoke Test
- k = 1-4 only
- 10 epochs
- Verify shapes, masking, metrics logging

### 4.2 H100 Initial Run
- k = 1-16
- 4000 epochs
- Monitor memory usage, step times

### 4.3 Full Run (if 1-16 works)
- k = 1-64
- 4000 epochs

---

## Phase 5: Plotting

Create `jl/double_descent/plot.py`:
- Load metrics.jsonl
- **Plot 1**: Test/train error vs k (final epoch) - reproduces Figure 1
- **Plot 2**: Test/train loss vs k (final epoch) - shows loss/error divergence
- **Plot 3**: Test error vs (k, epoch) heatmap - reproduces Figure 2
- **Plot 4**: Test loss vs (k, epoch) heatmap - shows loss dynamics
- Save PNGs to `data/`

---

## Implementation Order

1. `config.py` - Simple dataclass
2. `masked_batchnorm.py` - Core masking primitive
3. `resnet18k.py` - Adapt from double-descent repo
4. `data.py` - CIFAR-10 with noise
5. `trainer.py` - Parallel training (most complex)
6. `main.py` - Entry point with arg parsing
7. Modify `lambda_train.sh`
8. Local smoke test
9. `plot.py` - Visualization
10. H100 run with k=1-16

---

## Key Differences from Paper

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Trials | 5 | 1 |
| Training | Sequential per k | Parallel via vmap |
| BatchNorm | Standard | Masked |
| Width range | 1-64 | 1-16 initially, then 1-64 |

---

## Open Questions / Risks

1. **MaskedBatchNorm correctness**: Need to verify running stats are computed correctly over active channels only
2. **Memory**: If 16 widths don't fit, we'll split into 2 passes of 8
3. **Training dynamics**: Masked approach may have slightly different training dynamics than truly separate models - we should verify final test error matches paper's ~28% for k=64
