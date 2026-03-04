# Plan: Reproduce Deep Double Descent Figure 3

## Goal
Reproduce the Transformer double-descent curve from Nakkiran et al. (2019) Figure 3:
- 6-layer encoder-decoder Transformer with varying embedding dimension d_model (8-384)
- IWSLT'14 German-to-English translation
- Train/test loss, accuracy, AND BLEU vs model width
- **Both 4K and 18K training samples** (overlaid on same plot)

---

## Approach

**Three experiment modes**, all requiring exactly 8 GPUs:

### Default Mode (Double Descent)
Trains 48 models to reproduce the double descent curve:
- 24 d_model values: 8, 16, 24, ..., 192
- 2 sample sizes: 18K first, then 4K
- 6 batches total (3 per sample size), each training 8 models in parallel

### Long Double Descent Mode (`--long-double-descent` flag)
Extends the curve to larger models:
- 24 d_model values: 384, 376, 368, ..., 200 (largest first to detect OOM early)
- 18K samples only
- 3 batches total, 8 models in parallel
- Combined with default mode, produces full 8-384 range plot

### Variance Mode (`--variance` flag)
Trains 48 models to analyze variance across different training data:
- 6 d_model values: 32, 64, 96, 128, 160, 192
- 8 disjoint 18K training splits (from shuffled 160K base data)
- 6 batches total (one per d_model), training all 8 splits in parallel

No command-line arguments for d_model or train_samples - these are hardcoded.

---

## Phase 1: Data Preprocessing (Automatic)

### 1.1 Preprocessing Script (`infra/prepare_iwslt14.sh`)

A shell script that:
1. Downloads IWSLT'14 de-en from HuggingFace (`bbaaaa/iwslt14-de-en`, parquet branch)
2. Lowercases text
3. Learns joint BPE vocabulary (10K merge operations) using subword-nmt
4. Applies BPE to train/valid/test splits
5. Outputs to `data/iwslt14.tokenized.de-en/`

**Note**: Original wit3.fbk.eu URL is no longer available (404). Using community-uploaded HuggingFace dataset.

### 1.2 Automatic Preprocessing in Training

The `train.sh` script automatically checks for preprocessed data and runs `prepare_iwslt14.sh` if needed. This happens on the remote instance, so no manual data upload is required.

The training script (`transformer_main.py`) will fail with a clear error message if data files are missing (as a safety check).

### 1.3 Output Structure

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
jl/double_descent/transformer/
├── __init__.py
├── transformer_config.py     # TDDConfig dataclass (no d_model/train_samples)
├── transformer_data.py       # IWSLT data loading with max-tokens batching
├── transformer_main.py       # Entry point (hardcoded experiment, requires 8 GPUs)
├── transformer_model.py      # Encoder-decoder Transformer
├── trainer.py                # Single-model training function
├── plot_vary_d_model.py      # Final metrics plot (overlays 4K and 18K)
├── plot_single_d_model.py    # Step-wise training curves for single model
├── bleu.py                   # BLEU score computation
└── TRANSFORMER_PLAN.md       # This plan file
```

---

## Phase 3: Implementation Details

### 3.1 Config (`transformer_config.py`)

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class TDDConfig:
    """Configuration for Transformer double descent training.

    Note: d_model and train_samples are NOT configurable - they are
    hardcoded in transformer_main.py for the full experiment.
    """

    # Model architecture (from paper)
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
    train_samples: int,
    subsample_seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, Vocab]:
    """
    Load IWSLT'14 de-en with subsampling.

    Args:
        data_dir: Directory containing preprocessed data.
        train_samples: Number of training samples (4000 or 18000).
        subsample_seed: Seed for reproducible subsampling.

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

Runs the FULL experiment automatically:
- Requires exactly 8 GPUs (fails fast otherwise)
- Trains all 48 models: 24 d_model values x 2 sample sizes
- 6 batches total, 8 models per batch

Usage:
    python -m jl.double_descent.transformer.transformer_main \
        --output-path ./output \
        --data-path ./data/iwslt14.tokenized.de-en
"""

# Hardcoded experiment parameters
TRAIN_SAMPLES = [18000, 4000]  # 18K first, then 4K
D_MODEL_VALUES = list(range(8, 200, 8))  # [8, 16, 24, ..., 192] - 24 values
REQUIRED_GPUS = 8


def main():
    args = parse_args()
    config = TDDConfig()

    # Require exactly 8 GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus != REQUIRED_GPUS:
        raise RuntimeError(
            f"This experiment requires exactly {REQUIRED_GPUS} GPUs, "
            f"but found {num_gpus}. Please run on an 8-GPU instance."
        )

    logger.info("Transformer Double Descent - Full Experiment")
    logger.info(f"d_model values: {D_MODEL_VALUES}")
    logger.info(f"Train samples: {TRAIN_SAMPLES}")
    logger.info(f"Total models: {len(D_MODEL_VALUES) * len(TRAIN_SAMPLES)} = 48")
    logger.info(f"Batches: {len(D_MODEL_VALUES) // REQUIRED_GPUS * len(TRAIN_SAMPLES)} = 6")

    # Outer loop: sample sizes (18K first, then 4K)
    for train_samples in TRAIN_SAMPLES:
        samples_k = train_samples // 1000
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {samples_k}K sample runs")
        logger.info(f"{'='*60}")

        # Inner loop: d_model batches (3 batches of 8)
        for batch_idx in range(0, len(D_MODEL_VALUES), REQUIRED_GPUS):
            batch_d_models = D_MODEL_VALUES[batch_idx:batch_idx + REQUIRED_GPUS]
            batch_num = batch_idx // REQUIRED_GPUS + 1

            logger.info(f"\n[{samples_k}K] Batch {batch_num}/3: d_model = {batch_d_models}")

            # Spawn 8 training processes
            mp.set_start_method('spawn', force=True)
            processes = []
            for gpu_id, d_model in enumerate(batch_d_models):
                p = mp.Process(
                    target=train_single_model,
                    args=(gpu_id, d_model, train_samples, config,
                          args.output_path, args.data_path)
                )
                p.start()
                processes.append(p)

            # Wait for batch to complete
            for p in processes:
                p.join()

            logger.info(f"[{samples_k}K] Batch {batch_num}/3 complete")

    logger.info("\n" + "="*60)
    logger.info("Full experiment complete!")
    logger.info(f"Metrics files: {args.output_path}/metrics_d*_*k.jsonl")
```

### 3.5 Single Model Trainer (`trainer.py`)

```python
def train_single_model(
    gpu_id: int,
    d_model: int,
    train_samples: int,
    config: TDDConfig,
    output_path: str,
    data_path: str,
) -> None:
    """Train a single Transformer with embedding dimension d_model.

    Args:
        gpu_id: GPU device ID (0-7).
        d_model: Embedding dimension.
        train_samples: Number of training samples (4000 or 18000).
        config: Training configuration.
        output_path: Directory to save metrics.
        data_path: Directory containing preprocessed IWSLT data.
    """
    device = torch.device(f"cuda:{gpu_id}")
    samples_k = train_samples // 1000

    # Load data with specified sample count
    train_dataset, valid_dataset, test_dataset, vocab = load_iwslt14(
        data_path, train_samples, config.subsample_seed
    )

    # ... training loop ...

    # Log to file with sample size in name
    log_metrics(output_path, d_model, samples_k, metrics)
```

### 3.6 Output Structure

Training outputs are organized by timestamp in `output/transformer/MM-DD-HHmm/`:

```
output/transformer/03-01-1010/
├── metrics_d8_18k.jsonl
├── metrics_d8_4k.jsonl
├── ...
├── metrics_d192_18k.jsonl
├── metrics_d192_4k.jsonl
├── model_d8_18k.pt
├── model_d8_4k.pt
├── ...
├── model_d192_18k.pt
└── model_d192_4k.pt
```

**Metrics format (JSONL):**
```json
{"step": 100, "d_model": 64, "train_samples": 18000, "train_loss": 8.5, "train_acc": 0.05, "valid_loss": 8.7, "valid_acc": 0.04, "lr": 0.0001}
{"step": 200, "d_model": 64, "train_samples": 18000, "train_loss": 7.2, "train_acc": 0.12, "valid_loss": 7.5, "valid_acc": 0.10, "lr": 0.0002}
...
{"step": 80000, "d_model": 64, "train_samples": 18000, "train_loss": 2.1, "train_acc": 0.65, "valid_loss": 3.2, "valid_acc": 0.48, "lr": 0.00003, "train_bleu": 45.2, "test_bleu": 28.5, "test_loss": 3.3, "test_acc": 0.47}
```

**Total output files:** 96 (48 metrics + 48 models)

---

## Phase 4: Remote Training

### 4.1 Running Training

**Default double descent (d_model 8-192):**
```bash
./infra/train.sh <ip> --module jl.double_descent.transformer.transformer_main
```

The script automatically:
1. Checks for exactly 8 GPUs (fails fast otherwise)
2. Runs 18K samples: batches for d_model 8-64, 72-128, 136-192
3. Runs 4K samples: batches for d_model 8-64, 72-128, 136-192
4. Produces 48 metrics files total

**Long double descent (d_model 384-200, extends the curve):**
```bash
./infra/train.sh <ip> --module jl.double_descent.transformer.transformer_main --long-double-descent
```

The script:
1. Runs d_model 384, 376, ..., 200 (largest first to detect OOM)
2. 18K samples only, 24 models, 3 batches
3. Produces 24 additional metrics files (same naming convention)

**Note:** Run long-double-descent AFTER the default experiment, using the same output folder. The plot function will auto-discover all files and produce a combined 8-384 range plot.

### 4.2 Downloading Results

```bash
./infra/download.sh <ip>
```

Downloads all metrics and model files to `data/transformer/MM-DD-HHmm/` and auto-generates plots.

---

## Phase 5: Plotting

Plots are auto-generated by `download.sh` and saved alongside metrics in the run folder.

### `plot_vary_d_model.py` - Final metrics across d_model values (MAIN PLOT)

Auto-loads ALL metrics files and overlays 4K and 18K curves:
- Single figure with 3 subplots:
  - Top: Test loss vs d_model (4K and 18K overlaid)
  - Middle: Test accuracy vs d_model (4K and 18K overlaid)
  - Bottom: Test BLEU vs d_model (4K and 18K overlaid)

```bash
# Output saved to same directory as metrics
python -m jl.double_descent.transformer.plot_vary_d_model ./data/transformer/03-01-1010
```

Output: `data/transformer/03-01-1010/transformer_double_descent.png`

### `plot_single_d_model.py` - Training curves for single d_model

Plots step-wise training for a specific d_model and sample size:
- Single figure with 2 subplots:
  - Top: Train/Valid loss vs step
  - Bottom: Train/Valid accuracy vs step

```bash
python -m jl.double_descent.transformer.plot_single_d_model ./data/transformer/03-01-1010 --d-model 64 --samples 18k
```

---

## Phase 6: Testing Strategy

### 6.1 Local Preprocessing Test
```bash
# Run preprocessing script (required before any training)
./infra/prepare_iwslt14.sh

# Verify output
ls -la data/iwslt14.tokenized.de-en/
wc -l data/iwslt14.tokenized.de-en/train.de  # Should be ~170K
```

**Note**: If data files are missing, `transformer_main.py` will fail with a clear error listing the missing files.

### 6.2 Full Run (8 GPUs required)

**Default double descent:**
```bash
python -m jl.double_descent.transformer.transformer_main \
    --output-path ./output \
    --data-path ./data/iwslt14.tokenized.de-en
```

**Expected runtime:** ~6-8 hours total (48 models × ~8 minutes each)

**Output:** 48 files: `metrics_d{8-192}_{4k,18k}.jsonl`

**Long double descent (extends curve to larger models):**
```bash
python -m jl.double_descent.transformer.transformer_main \
    --long-double-descent \
    --output-path ./output \
    --data-path ./data/iwslt14.tokenized.de-en
```

**Expected runtime:** ~3-4 hours (24 models × ~8 minutes each, larger models take longer)

**Output:** 24 files: `metrics_d{200-384}_18k.jsonl`

---

## Execution Order

| Batch | Sample Size | d_model Values |
|-------|-------------|----------------|
| 1     | 18K         | 8, 16, 24, 32, 40, 48, 56, 64 |
| 2     | 18K         | 72, 80, 88, 96, 104, 112, 120, 128 |
| 3     | 18K         | 136, 144, 152, 160, 168, 176, 184, 192 |
| 4     | 4K          | 8, 16, 24, 32, 40, 48, 56, 64 |
| 5     | 4K          | 72, 80, 88, 96, 104, 112, 120, 128 |
| 6     | 4K          | 136, 144, 152, 160, 168, 176, 184, 192 |

### Long Double Descent Execution Order (`--long-double-descent`)

| Batch | Sample Size | d_model Values |
|-------|-------------|----------------|
| 1     | 18K         | 384, 376, 368, 360, 352, 344, 336, 328 |
| 2     | 18K         | 320, 312, 304, 296, 288, 280, 272, 264 |
| 3     | 18K         | 256, 248, 240, 232, 224, 216, 208, 200 |

**Note:** Runs largest models first to detect OOM early.

---

## Variance Experiment (--variance flag)

A second experiment mode that trains multiple models per d_model value to analyze variance across different training data splits.

### Overview

- **6 d_model values**: 32, 64, 96, 128, 160, 192 (increments of 32)
- **8 disjoint training splits**: Each split is 18K samples from the 160K base dataset
- **48 total models**: 6 d_model × 8 splits
- **6 batches**: One per d_model, training 8 models in parallel (one per GPU)

### Data Split Strategy

The 160K training samples are shuffled with a fixed seed, then partitioned into 8 disjoint 18K chunks:
- Split 0: indices 0-17999
- Split 1: indices 18000-35999
- ...
- Split 7: indices 126000-143999

### Running the Variance Experiment

```bash
# On Lambda Labs 8-GPU instance
./infra/train.sh <ip> --module jl.double_descent.transformer.transformer_main --variance
```

### Output Structure

```
output/transformer_variance/03-01-1010/
├── metrics_d32_split0.jsonl
├── metrics_d32_split1.jsonl
├── ...
├── metrics_d32_split7.jsonl
├── metrics_d64_split0.jsonl
├── ...
├── metrics_d192_split7.jsonl
├── model_d32_split0.pt
├── model_d32_split1.pt
├── ...
└── model_d192_split7.pt
```

**Total output files:** 96 (48 metrics + 48 models)

### Execution Order

| Batch | d_model | Splits |
|-------|---------|--------|
| 1     | 32      | 0, 1, 2, 3, 4, 5, 6, 7 |
| 2     | 64      | 0, 1, 2, 3, 4, 5, 6, 7 |
| 3     | 96      | 0, 1, 2, 3, 4, 5, 6, 7 |
| 4     | 128     | 0, 1, 2, 3, 4, 5, 6, 7 |
| 5     | 160     | 0, 1, 2, 3, 4, 5, 6, 7 |
| 6     | 192     | 0, 1, 2, 3, 4, 5, 6, 7 |

### Analyzing the Variance

We examine whether double descent is driven primarily through a reduction in bias or variance. Given the bias-variance decomposition:

E[CE(p, q)] = CE(p, E[q]) + E[log(E[q][y] / q[y])]

Where the first term is bias and the second term (Jensen Gap) is variance.

#### Metrics

For each d_model:
- **Mean test loss**: Cross-entropy loss averaged across the 8 training splits
- **Jensen Gap**: E[log(q_bar[y] / q_j[y])] - the variance term
- **Implied Bias**: test_loss - jensen_gap
- **Mean confidence**: Average probability assigned to the predicted (most likely) token
- **Mean log confidence**: Average log probability of the predicted token

#### Plots

Three separate figures, d_model on x-axis:
1. **Bias-variance decomposition**: test loss, Jensen Gap, implied bias (3 lines)
2. **Mean confidence** vs d_model
3. **Mean log confidence** vs d_model

#### Computation

For each d_model:
1. Run all 8 split-models on the test set, collect per-token softmax distributions q_1, ..., q_8
2. Compute q_bar = (1/8) * sum_j(q_j) — the mean distribution at each token position
3. For each model j, compute Jensen Gap: log(q_bar[y] / q_j[y]) per token
4. Compute max probability (confidence) and log of max probability per token
5. Average all metrics over the 8 models and all test tokens

#### Implementation

Four new files:

1. **`infra/upload.sh`** — Upload a local model run folder to the remote instance
   - Takes an IP and a local folder path as arguments
   - Uploads to the mirrored remote path (e.g., local `./data/transformer_variance/03-01-1010/` → remote `~/variance-min-classification/output/transformer_variance/03-01-1010/`)

2. **`.claude/commands/evaluate.md`** — Agent command that orchestrates evaluation on a remote GPU instance
   - Takes a remote IP
   - SSHs to remote and checks if training output exists under `~/variance-min-classification/output/transformer_variance/`
   - If not present, uses `upload.sh` to upload the local data
   - Ensures preprocessed IWSLT data exists on remote (runs `prepare_iwslt14.sh` if missing)
   - Ensures code is up to date on remote (git pull + venv setup)
   - Runs `evaluate.py` on the remote instance
   - Downloads evaluation results (`evaluation.jsonl`) via scp
   - Runs `plot_evaluation.py` locally to produce the plot

3. **`jl/double_descent/transformer/evaluate.py`** — Python module that runs on the remote GPU
   - Discovers all variance model files (`model_d*_split*.pt`) in the output directory
   - For each d_model, loads all 8 split-models
   - Model architecture inferred from filename (`d_model` from name, all other params from `TDDConfig` defaults)
   - Runs forward pass on the full test set, collects per-token softmax distributions
   - Computes mean test loss, Jensen Gap, mean confidence, and mean log confidence per d_model
   - Outputs `evaluation.jsonl` alongside the model files: one line per d_model with `{"d_model": N, "mean_test_loss": X, "mean_jensen_gap": Y, "mean_confidence": Z, "mean_log_confidence": W}`
   - Usage: `python -m jl.double_descent.transformer.evaluate --model-path ./output/transformer_variance/03-01-1010 --data-path ./data/iwslt14.tokenized.de-en`

4. **`jl/double_descent/transformer/plot_evaluation.py`** — Plotting script that runs locally
   - Reads the evaluation JSONL output
   - Produces three separate figures: bias-variance decomposition (test loss + Jensen Gap + implied bias), mean confidence, and mean log confidence vs d_model
   - Usage: `python -m jl.double_descent.transformer.plot_evaluation ./data/transformer_variance/03-01-1010/evaluation.jsonl --output-dir ./data`


## Key Differences from Paper

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Optimizer | Adam | AdamW (same hyperparams) |
| Trials | Unknown | 1 (double descent) or 8 (variance) |
| Sample sizes | 4K and 18K | Both (overlaid on same plot) |
| Framework | fairseq | Custom PyTorch |
| BLEU frequency | Unknown | End of training only |

---

## Hardware Requirements

- **Preprocessing**: CPU only, ~5 minutes
- **Training**: **Exactly 8 GPUs required** (script fails fast otherwise)
- **Memory**: ~2-8GB per GPU depending on d_model
- **Instance type**: 8x Nvidia GPUs from Lambda Labs
- **Estimated time**: 6-8 hours for full experiment

---

## Dependencies

### Training (add to requirements.txt)
- `sacrebleu` - BLEU score computation

### Preprocessing only (installed by `prepare_iwslt14.sh`)
- `subword-nmt` - BPE tokenization (pip install)
- Moses tokenizer - Perl scripts (cloned from GitHub, not pip)

**Note**: No fairseq dependency required. The preprocessing script is self-contained and based on fairseq's approach but doesn't import fairseq.
