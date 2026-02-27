# Plan: Reproduce Deep Double Descent Figure 3

## Goal
Reproduce the Transformer double-descent curve from Nakkiran et al. (2019) Figure 3:
- 6-layer encoder-decoder Transformer with varying embedding dimension d_model (8-192)
- IWSLT'14 German-to-English translation
- Train/test loss, accuracy, AND BLEU vs model width

---

## Approach
Train N models in parallel on N GPUs using `torch.multiprocessing`. Each GPU trains one model with a different d_model value, incrementing by 8. On 8 GPUs with default d_model_start=8, trains d_model=8,16,24,32,40,48,56,64. On 1 GPU with d_model_start=8, trains d_model=8 only.

To cover d_model=8-192 (24 values), run 3 times with `--d-model-start 8`, `--d-model-start 72`, `--d-model-start 136`.

---

## Phase 1: Data Preprocessing (One-Time, Local)

### 1.1 Preprocessing Script (`infra/prepare_iwslt14.sh`)

A shell script that:
1. Clones Moses tokenizer (Perl scripts from GitHub - no pip install)
2. Installs subword-nmt via pip (for BPE)
3. Downloads IWSLT'14 de-en raw data
4. Applies Moses tokenization
5. Lowercases text
6. Learns joint BPE vocabulary (10K merge operations)
7. Applies BPE to train/valid/test splits
8. Outputs to `data/iwslt14.tokenized.de-en/`

Based on fairseq's `prepare-iwslt14.sh`. **Note**: Does NOT require fairseq as a dependency - only uses Moses (Perl) and subword-nmt (Python).

### 1.2 Output Structure

```
data/iwslt14.tokenized.de-en/
├── train.de          # BPE-tokenized German (160K sentences)
├── train.en          # BPE-tokenized English
├── valid.de          # ~7K sentences
├── valid.en
├── test.de           # ~7K sentences
├── test.en
└── code              # BPE merge rules
```

**Size**: ~15-20MB total
**Time**: ~5 minutes on CPU

---

## Phase 2: Package Structure

```
jl/transformer_dd/
├── __init__.py
├── transformer_config.py   # TDDConfig dataclass
├── transformer_data.py     # IWSLT data loading with max-tokens batching
├── transformer_main.py     # Entry point (spawns N processes)
├── transformer_model.py    # Encoder-decoder Transformer
├── trainer.py              # Single-model training function
├── plot.py                 # Visualization (3 plots)
└── bleu.py                 # BLEU score computation
```

---

## Phase 3: Implementation Details

### 3.1 Config (`transformer_config.py`)

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class TDDConfig:
    """Configuration for Transformer double descent training."""

    # Model architecture (from paper)
    d_model_start: int = 8          # Starting embedding dim, trains d_model, d_model+8, ..., d_model+8*(N-1)
    n_layers: int = 6               # Encoder and decoder layers
    n_heads: int = 8                # Attention heads (always 8, even for small d_model)
    d_ff_multiplier: int = 4        # d_ff = d_ff_multiplier * d_model
    # Note: dropout hardcoded to 0.0 in model (per paper)

    # Training (from paper: 80K steps)
    max_steps: int = 80000          # Gradient steps
    max_tokens: int = 4096          # Tokens per batch (max-tokens batching)
    warmup_steps: int = 4000        # LR warmup steps
    optimizer: str = "adam_w"       # AdamW with Vaswani params (beta1=0.9, beta2=0.98, eps=1e-9)

    # Regularization
    label_smoothing: Optional[float] = 0.1  # None to disable

    # Data
    train_samples: int = 4000       # Subsample training set (paper uses 4K and 18K)
    subsample_seed: int = 42        # Fixed seed for reproducibility

    # Logging
    log_interval: int = 100         # Log train metrics every N steps
    eval_interval: int = 100        # Evaluate on valid set every N steps
```

### 3.2 Transformer Model (`transformer_model.py`)

Standard encoder-decoder Transformer following Vaswani et al. (2017):

```python
class TransformerModel(nn.Module):
    """
    Encoder-decoder Transformer for translation.

    Architecture:
    - Shared source/target vocabulary (joint BPE)
    - Learned positional embeddings
    - 6 encoder layers, 6 decoder layers
    - 8 attention heads (head_dim = d_model // 8)
    - d_ff = 4 * d_model
    - Pre-norm (LayerNorm before attention/FFN)
    - Dropout = 0.0 (hardcoded per paper)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff_multiplier: int = 4,
        max_seq_len: int = 512,
    ):
        # Note: dropout hardcoded to 0.0 (per paper)
        ...
```

Key implementation notes:
- Use PyTorch's `nn.MultiheadAttention` for attention layers
- Pre-LayerNorm architecture (more stable training)
- Shared embedding for encoder/decoder (joint BPE vocab)
- Causal masking in decoder self-attention

### 3.3 Data Loading (`transformer_data.py`)

```python
def load_iwslt14(
    data_dir: str,
    train_samples: int = 4000,
    subsample_seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, Vocab]:
    """
    Load IWSLT'14 de-en with subsampling.

    Returns:
        train_dataset: Subsampled to train_samples
        valid_dataset: Full validation set (~7K)
        test_dataset: Full test set (~7K)
        vocab: Shared vocabulary built from BPE tokens
    """

class MaxTokensBatchSampler:
    """
    Batch sampler that groups sentences to fit max_tokens per batch.

    - Sorts sentences by length
    - Greedily fills batches up to max_tokens
    - Shuffles batch order (not within-batch order)
    """
```

### 3.4 Multi-GPU Training (`transformer_main.py`)

```python
"""
Entry point for Transformer Double Descent training.

This script trains N models in parallel on N GPUs using torch.multiprocessing.
Each GPU trains one model with d_model, d_model+8, d_model+16, ..., d_model+8*(N-1).

Usage:
    # Train models starting at d_model=8 (one model per available GPU)
    python -m jl.transformer_dd.transformer_main --output-path ./output --d-model-start 8

    # For quick smoke test:
    python -m jl.transformer_dd.transformer_main --output-path ./output --d-model-start 64 --max-steps 100

    # On 8 GPUs with d-model-start=8, trains d_model=8,16,24,32,40,48,56,64
    # On 1 GPU with d-model-start=8, trains d_model=8
"""

def main():
    args = parse_args()
    config = TDDConfig()

    # Override config with command line arguments
    if args.d_model_start is not None:
        config.d_model_start = args.d_model_start
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    # ... etc for other args

    # Check GPU count - require at least 1 GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError(
            "This script requires at least 1 GPU, but none were found. "
            "Please run on a machine with CUDA-capable GPUs."
        )

    # Compute d_model values for this run (one d_model per GPU, incrementing by 8)
    d_model_values = [config.d_model_start + 8 * i for i in range(num_gpus)]

    logger.info("Transformer Double Descent Training")
    logger.info(f"d_model values: {d_model_values}")
    logger.info(f"Max steps: {config.max_steps}, Max tokens/batch: {config.max_tokens}")
    logger.info(f"GPUs available: {num_gpus}")

    # Spawn training processes (one per GPU)
    mp.set_start_method('spawn', force=True)
    processes = []
    for gpu_id in range(num_gpus):
        d_model = d_model_values[gpu_id]
        p = mp.Process(
            target=train_single_model,
            args=(gpu_id, d_model, config, args.output_path, args.data_path)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started process for d_model={d_model} on GPU {gpu_id}")

    # Wait for all to complete
    for i, p in enumerate(processes):
        p.join()
        logger.info(f"Process for d_model={d_model_values[i]} completed")
```

### 3.5 Single Model Trainer (`trainer.py`)

```python
def train_single_model(
    gpu_id: int,
    d_model: int,
    config: TDDConfig,
    output_path: str,
    data_path: str,
) -> None:
    """Train a single Transformer with embedding dimension d_model."""
    device = torch.device(f"cuda:{gpu_id}")

    # Load data
    train_dataset, valid_dataset, test_dataset, vocab = load_iwslt14(
        data_path, config.train_samples, config.subsample_seed
    )

    # Create model
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
    ).to(device)

    # Optimizer with Vaswani LR schedule (betas and eps hardcoded per paper)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0,  # Scaled by scheduler
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    def lr_lambda(step):
        # Vaswani formula: d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * config.warmup_steps ** -1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function with optional label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.label_smoothing or 0.0,
    )

    # Training loop (80K steps)
    step = 0
    while step < config.max_steps:
        for src, tgt in train_loader:
            # Forward pass
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.view(-1, len(vocab)), tgt[:, 1:].view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1

            # Log metrics
            if step % config.log_interval == 0:
                log_metrics(...)

            # Evaluate on validation set
            if step % config.eval_interval == 0:
                evaluate(model, valid_dataset, ...)

            if step >= config.max_steps:
                break

    # Final evaluation with BLEU
    test_loss, test_acc = evaluate(model, test_dataset, device)
    test_bleu = compute_bleu(model, test_dataset, vocab, device)
    train_bleu = compute_bleu(model, train_dataset, vocab, device)

    log_final_metrics(output_path, d_model, test_loss, test_acc, test_bleu, train_bleu)
```

### 3.6 BLEU Computation (`bleu.py`)

```python
def compute_bleu(
    model: nn.Module,
    dataset: Dataset,
    vocab: Vocab,
    device: torch.device,
    max_len: int = 128,
) -> float:
    """
    Compute corpus-level BLEU score using greedy decoding.

    - Generates translations autoregressively
    - Uses greedy decoding (argmax at each step)
    - Computes BLEU using sacrebleu library
    """
```

### 3.7 Metrics Output

Each d_model value gets its own file: `output/metrics_d{d_model}.jsonl`

```json
{"step": 100, "d_model": 64, "train_loss": 8.5, "train_acc": 0.05, "valid_loss": 8.7, "valid_acc": 0.04, "lr": 0.0001}
{"step": 200, "d_model": 64, "train_loss": 7.2, "train_acc": 0.12, "valid_loss": 7.5, "valid_acc": 0.10, "lr": 0.0002}
...
{"step": 80000, "d_model": 64, "train_loss": 2.1, "train_acc": 0.65, "valid_loss": 3.2, "valid_acc": 0.48, "lr": 0.00003, "train_bleu": 45.2, "test_bleu": 28.5, "test_loss": 3.3, "test_acc": 0.47}
```

**Metrics tracked:**
- `train_loss`: Cross-entropy loss on training batch
- `train_acc`: Token-level accuracy on training batch
- `valid_loss`: Loss on full validation set
- `valid_acc`: Token-level accuracy on validation set
- `lr`: Current learning rate
- `train_bleu`: BLEU on training set (final only)
- `test_bleu`: BLEU on test set (final only)
- `test_loss`: Loss on test set (final only)
- `test_acc`: Accuracy on test set (final only)

---

## Phase 4: Lambda Integration

### 4.1 Running Training

```bash
# Provision 8-GPU instance, then run 3 times to cover d_model=8-192:

# Run 1: d_model = 8,16,24,32,40,48,56,64 (8 GPUs in parallel)
./infra/lambda_train.sh <ip> --module jl.transformer_dd.transformer_main --d-model-start 8

# Run 2: d_model = 72,80,88,96,104,112,120,128
./infra/lambda_train.sh <ip> --module jl.transformer_dd.transformer_main --d-model-start 72

# Run 3: d_model = 136,144,152,160,168,176,184,192
./infra/lambda_train.sh <ip> --module jl.transformer_dd.transformer_main --d-model-start 136
```

### 4.2 Downloading Results

```bash
./infra/lambda_download.sh <ip> --plot-module jl.transformer_dd.plot
```

The plot module will combine all `metrics_d*.jsonl` files to generate plots.

---

## Phase 5: Plotting

Create `jl/transformer_dd/plot.py`:

### Final Metrics Plots (all d_model values)

When given a directory of metrics files:

**Plot 1: Loss vs d_model**
- X-axis: Transformer embedding dimension (d_model)
- Y-axis: Cross-entropy loss (per-token perplexity)
- Lines: Train loss, Test loss
- Save: `data/transformer_dd_loss.png`

**Plot 2: Accuracy vs d_model**
- X-axis: Transformer embedding dimension (d_model)
- Y-axis: Token-level accuracy
- Lines: Train accuracy, Test accuracy
- Save: `data/transformer_dd_accuracy.png`

**Plot 3: BLEU vs d_model**
- X-axis: Transformer embedding dimension (d_model)
- Y-axis: BLEU score
- Lines: Train BLEU, Test BLEU
- Save: `data/transformer_dd_bleu.png`

### Step-wise Training Curves (single d_model)

When given a single metrics file (e.g., `metrics_d64.jsonl`) or `--d-model 64`:

**Single figure with 2 subplots:**
- Top: Train loss & Valid loss vs Step
- Bottom: Train accuracy & Valid accuracy vs Step
- Save: `data/transformer_dd_training_d64.png`

### Usage

```bash
# All plots for all d_model values (final metrics: loss, accuracy, BLEU)
python -m jl.transformer_dd.plot ./output --output-dir ./data

# Step-wise training curves for a specific d_model
python -m jl.transformer_dd.plot ./output/metrics_d64.jsonl --output-dir ./data
# OR
python -m jl.transformer_dd.plot ./output --d-model 64 --output-dir ./data
```

---

## Phase 6: Testing Strategy

### 6.1 Local Preprocessing Test
```bash
# Run preprocessing script
./infra/prepare_iwslt14.sh

# Verify output
ls -la data/iwslt14.tokenized.de-en/
wc -l data/iwslt14.tokenized.de-en/train.de  # Should be ~160K
```

### 6.2 Local Smoke Test (1 GPU)
```bash
# On 1 GPU, trains only d_model=64
python -m jl.transformer_dd.transformer_main \
    --output-path ./output \
    --d-model-start 64 \
    --max-steps 100 \
    --train-samples 1000
```
- Verify: model trains, metrics file created, no errors
- Output: `output/metrics_d64.jsonl`

### 6.3 Multi-GPU Test (if available)
```bash
# On 8 GPUs with d-model-start=8, trains d_model=8,16,24,32,40,48,56,64
python -m jl.transformer_dd.transformer_main \
    --output-path ./output \
    --d-model-start 8 \
    --max-steps 1000
```
- Monitor GPU utilization, training speed
- Output: `output/metrics_d8.jsonl` through `output/metrics_d64.jsonl`

### 6.4 Full Run (8 GPUs)
- 3 runs: d-model-start = 8, 72, 136
- 80K steps each
- Produces metrics_d8.jsonl through metrics_d192.jsonl (24 files total)

---

## Implementation Order

1. Create `infra/prepare_iwslt14.sh` - preprocessing script
2. Run preprocessing, verify output in `data/`
3. Create `jl/transformer_dd/transformer_config.py` - config dataclass
4. Create `jl/transformer_dd/transformer_data.py` - data loading + max-tokens batching
5. Create `jl/transformer_dd/transformer_model.py` - Transformer architecture
6. Create `jl/transformer_dd/bleu.py` - BLEU computation
7. Create `jl/transformer_dd/trainer.py` - single model training
8. Create `jl/transformer_dd/transformer_main.py` - multi-GPU entry point
9. Create `jl/transformer_dd/plot.py` - visualization
10. Local smoke test
11. Full run on 8-GPU instance

---

## Key Differences from Paper

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Optimizer | Adam | AdamW (same hyperparams) |
| Trials | Unknown | 1 |
| Sample sizes | 4K and 18K | 4K (configurable) |
| Framework | fairseq | Custom PyTorch |
| BLEU frequency | Unknown | End of training only |

---

## Hardware Requirements

- **Preprocessing**: CPU only, ~5 minutes
- **Training**: At least 1 GPU required (script fails with clear error if none found)
  - 1 GPU: trains single d_model value (d_model_start only)
  - N GPUs: trains N d_model values in parallel (d_model_start, d_model_start+8, ..., d_model_start+8*(N-1))
- **Memory**: ~2-8GB per GPU depending on d_model
- **Recommended**: 8x V100/A100/H100 on Lambda Labs for full runs
- **Local testing**: 1 GPU sufficient for smoke tests

---

## Dependencies

### Training (add to requirements.txt)
- `sacrebleu` - BLEU score computation

### Preprocessing only (installed by `prepare_iwslt14.sh`)
- `subword-nmt` - BPE tokenization (pip install)
- Moses tokenizer - Perl scripts (cloned from GitHub, not pip)

**Note**: No fairseq dependency required. The preprocessing script is self-contained and based on fairseq's approach but doesn't import fairseq.
