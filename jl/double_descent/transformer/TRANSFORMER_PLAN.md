# Plan: Reproduce Deep Double Descent Figure 3

## Goal
Reproduce the Transformer double-descent curve from Nakkiran et al. (2019) Figure 3:
- 6-layer encoder-decoder Transformer with varying embedding dimension d_model (8-192)
- IWSLT'14 German-to-English translation
- Train/test cross-entropy loss, test accuracy, and test BLEU vs model width
- **36K training samples**

---

## Approach

**Two experiment modes**, all requiring exactly 8 GPUs:

### Default Mode (Double Descent)
Trains 24 models to reproduce the double descent curve:
- 24 d_model values: 8, 16, 24, ..., 192
- 36K training samples
- 3 batches total, each training 8 models in parallel

### Variance Mode (`--variance` flag)
Trains 96 models to analyze variance across different training data:
- 24 d_model values: 16, 32, 48, ..., 384
- 4 disjoint 36K training splits (from shuffled 160K base data)
- 12 batches total (2 d_model per batch × 4 splits = 8 GPUs per batch)

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
├── trainer.py                # Single-model training function (calls evaluation.py)
├── bleu.py                   # BLEU score computation
├── evaluation.py             # Final metrics + ECE for main runs
├── plot_evaluation.py        # Main runs: loss/ECE/accuracy/BLEU vs d_model
├── plot_single_d_model.py    # Step-wise training curves for single model
├── variance_evaluation.py    # Variance mode: Jensen Gap with Bessel's correction
├── plot_variance_evaluation.py  # Bias-variance decomposition plot
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

    # Training (80K steps)
    max_steps: int = 80000          # Gradient steps
    max_tokens: int = 4096          # Tokens per batch (max-tokens batching)
    warmup_steps: int = 4000        # LR warmup steps
    learning_rate: float = 3e-4     # Peak learning rate after warmup
    optimizer: str = "adam_w"       # AdamW with beta1=0.9, beta2=0.98, eps=1e-9

    # LR Schedule: Linear warmup for warmup_steps, then cosine decay to 0

    # Regularization
    label_smoothing: Optional[float] = None  # Disabled

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
- Trains 24 models: 24 d_model values x 1 sample size (36K)
- 3 batches total, 8 models per batch

Usage:
    python -m jl.double_descent.transformer.transformer_main \
        --output-path ./output \
        --data-path ./data/iwslt14.tokenized.de-en
"""

# Hardcoded experiment parameters
TRAIN_SAMPLES = [36000]  # 36K samples
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

    train_samples = TRAIN_SAMPLES[0]
    samples_k = train_samples // 1000

    logger.info("Transformer Double Descent - Full Experiment")
    logger.info(f"d_model values: {D_MODEL_VALUES}")
    logger.info(f"Train samples: {train_samples}")
    logger.info(f"Total models: {len(D_MODEL_VALUES)}")
    logger.info(f"Batches: {len(D_MODEL_VALUES) // REQUIRED_GPUS}")

    # Loop over d_model batches (3 batches of 8)
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
    logger.info(f"Metrics files: {args.output_path}/metrics_d*_{samples_k}k.jsonl")
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
├── metrics_d8_36k.jsonl
├── metrics_d16_36k.jsonl
├── ...
├── metrics_d192_36k.jsonl
├── model_d8_36k.pt
├── model_d16_36k.pt
├── ...
└── model_d192_36k.pt
```

**Metrics format (JSONL):**
```json
{"step": 100, "d_model": 64, "train_samples": 36000, "train_loss": 8.5, "train_acc": 0.05, "valid_loss": 8.7, "valid_acc": 0.04, "lr": 0.0001}
{"step": 200, "d_model": 64, "train_samples": 36000, "train_loss": 7.2, "train_acc": 0.12, "valid_loss": 7.5, "valid_acc": 0.10, "lr": 0.0002}
...
{"step": 80000, "d_model": 64, "train_samples": 36000, "train_loss": 2.1, "train_acc": 0.65, "valid_loss": 3.2, "valid_acc": 0.48, "lr": 0.00003, "train_bleu": 45.2, "test_bleu": 28.5, "test_loss": 3.3, "test_acc": 0.47}
```

**Total output files:** 48 (24 metrics + 24 models)

---

## Phase 4: Remote Training

### 4.1 Running Training

**Default double descent (d_model 8-192):**
```bash
./infra/train.sh <ip> --module jl.double_descent.transformer.transformer_main
```

The script automatically:
1. Checks for exactly 8 GPUs (fails fast otherwise)
2. Runs 36K samples: 3 batches for d_model 8-64, 72-128, 136-192
3. Produces 24 metrics files total

### 4.2 Downloading Results

```bash
./infra/download.sh <ip>
```

Downloads all metrics and model files to `data/transformer/MM-DD-HHmm/` and auto-generates plots.

---

## Phase 5: Plotting

Plots are auto-generated by `download.sh` and saved alongside metrics in the run folder.

### `plot_evaluation.py` - Final metrics across d_model values (MAIN PLOT)

Plots evaluation metrics for main (non-variance) runs:
- Single figure with 3 subplots:
  - Top: Train and test cross-entropy loss vs d_model, ECE on right y-axis
  - Middle: Test accuracy vs d_model
  - Bottom: Test BLEU vs d_model

Generated automatically by trainer. Can also run manually:
```bash
python -m jl.double_descent.transformer.plot_evaluation ./data/transformer/03-01-1010/evaluation.jsonl
```

Output: `data/transformer/03-01-1010/transformer_evaluation.png`

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

**Expected runtime:** ~3-4 hours total (24 models in 3 batches)

**Output:** 24 files: `metrics_d{8-192}_36k.jsonl`

---

## Execution Order

| Batch | Sample Size | d_model Values |
|-------|-------------|----------------|
| 1     | 36K         | 8, 16, 24, 32, 40, 48, 56, 64 |
| 2     | 36K         | 72, 80, 88, 96, 104, 112, 120, 128 |
| 3     | 36K         | 136, 144, 152, 160, 168, 176, 184, 192 |

---

## Variance Experiment (--variance flag)

A second experiment mode that trains multiple models per d_model value to analyze variance across different training data splits.

### Overview

- **24 d_model values**: 16, 32, 48, ..., 384 (increments of 16)
- **4 disjoint training splits**: Each split is 36K samples from the 160K base dataset
- **96 total models**: 24 d_model × 4 splits
- **12 batches**: 2 d_model values per batch, each with 4 splits = 8 GPUs per batch

### Data Split Strategy

The 160K training samples are shuffled with a fixed seed, then partitioned into 4 disjoint 36K chunks:
- Split 0: indices 0-35999
- Split 1: indices 36000-71999
- Split 2: indices 72000-107999
- Split 3: indices 108000-143999

### Running the Variance Experiment

```bash
# On Lambda Labs 8-GPU instance
./infra/train.sh <ip> --module jl.double_descent.transformer.transformer_main --variance
```

### Output Structure

```
output/transformer_variance/03-01-1010/
├── metrics_d16_split0.jsonl
├── metrics_d16_split1.jsonl
├── metrics_d16_split2.jsonl
├── metrics_d16_split3.jsonl
├── metrics_d32_split0.jsonl
├── ...
├── metrics_d384_split3.jsonl
├── model_d16_split0.pt
├── ...
└── model_d384_split3.pt
```

**Total output files:** 192 (96 metrics + 96 models)

### Execution Order

| Batch | d_model values | Splits | GPUs |
|-------|---------------|--------|------|
| 1     | 16, 32        | 0-3 each | 0-3: d16, 4-7: d32 |
| 2     | 48, 64        | 0-3 each | 0-3: d48, 4-7: d64 |
| 3     | 80, 96        | 0-3 each | 0-3: d80, 4-7: d96 |
| 4     | 112, 128      | 0-3 each | 0-3: d112, 4-7: d128 |
| 5     | 144, 160      | 0-3 each | 0-3: d144, 4-7: d160 |
| 6     | 176, 192      | 0-3 each | 0-3: d176, 4-7: d192 |
| 7     | 208, 224      | 0-3 each | 0-3: d208, 4-7: d224 |
| 8     | 240, 256      | 0-3 each | 0-3: d240, 4-7: d256 |
| 9     | 272, 288      | 0-3 each | 0-3: d272, 4-7: d288 |
| 10    | 304, 320      | 0-3 each | 0-3: d304, 4-7: d320 |
| 11    | 336, 352      | 0-3 each | 0-3: d336, 4-7: d352 |
| 12    | 368, 384      | 0-3 each | 0-3: d368, 4-7: d384 |

### Analyzing the Variance

We examine whether double descent is driven primarily through a reduction in bias or variance. Given the bias-variance decomposition:

E[CE(p, q)] = CE(p, E[q]) + E[log(E[q][y] / q[y])]

Where the first term is bias and the second term (Jensen Gap) is variance.

#### Metrics

For each d_model:
- **Mean test loss**: Cross-entropy loss averaged across the 4 training splits
- **Jensen Gap**: E[log(q_bar[y] / q_j[y])] - the variance term
- **Implied Bias**: test_loss - jensen_gap

#### Plot

Single figure with d_model on x-axis showing three lines:
- Mean test loss
- Jensen Gap (variance)
- Implied bias

#### Computation

For each d_model:
1. Run all 4 split-models on the test set, collect per-token softmax distributions q_1, ..., q_4
2. Compute q_bar = (1/4) * sum_j(q_j) — the mean distribution at each token position
3. For each model j, compute Jensen Gap: log(q_bar[y] / q_j[y]) per token
4. Average all metrics over the 4 models and all test tokens

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

3. **`jl/double_descent/transformer/variance_evaluation.py`** — Python module that runs on the remote GPU
   - Discovers all variance model files (`model_d*_split*.pt`) in the output directory
   - For each d_model, loads all 4 split-models
   - Model architecture inferred from filename (`d_model` from name, all other params from `TDDConfig` defaults)
   - Runs forward pass on the full test set, collects per-token softmax distributions
   - Computes mean test loss and Jensen Gap per d_model
   - Outputs `evaluation.jsonl` alongside the model files: one line per d_model with `{"d_model": N, "mean_test_loss": X, "mean_jensen_gap": Y}`
   - Usage: `python -m jl.double_descent.transformer.variance_evaluation --model-path ./output/transformer_variance/03-01-1010 --data-path ./data/iwslt14.tokenized.de-en`

4. **`jl/double_descent/transformer/plot_variance_evaluation.py`** — Plotting script that runs locally
   - Reads the evaluation JSONL output
   - Produces a single figure: bias-variance decomposition (test loss, Jensen Gap, implied bias) vs d_model
   - Usage: `python -m jl.double_descent.transformer.plot_variance_evaluation ./data/transformer_variance/03-01-1010/evaluation.jsonl --output-dir ./data`

#### Temperature Scaling

Optional post-hoc calibration to demonstrate that simple calibration cannot recover first-descent loss:

```bash
python -m jl.double_descent.transformer.variance_evaluation \
    --model-path ./output/transformer_variance/03-01-1010 \
    --data-path ./data/iwslt14.tokenized.de-en --temperature-scaling
```

This fits a scalar temperature T per d_model using L-BFGS on one randomly chosen model's test NLL (batch-wise to avoid storing full logits), then recomputes the full bias-variance decomposition with `softmax(logits/T)` across all models. Forward passes are parallelized across 8 GPUs (one model per GPU).

Output: `temperature-scaled/evaluation.jsonl` in the model directory (same schema plus a `"temperature"` field).

Plotting temperature-scaled results:

```bash
python -m jl.double_descent.transformer.plot_variance_evaluation \
    ./data/transformer_variance/03-01-1010/temperature-scaled/evaluation.jsonl \
    --output-dir ./data --temperature-scaled
```


## Phase 7: Final-Layer Fine-Tuning

Fine-tunes only the output projection layer (`model.output_proj`) of each trained model using L-BFGS with L2 regularization. The output projection is first untied from the embedding (copied into a standalone `nn.Linear`), then fine-tuned independently.

### Running Fine-Tuning

```bash
python -m jl.double_descent.transformer.fine_tune \
    --model-path ./output/transformer/03-01-1010 \
    --data-path ./data/iwslt14.tokenized.de-en \
    --l2-lambda 1e-5 --max-steps 100
```

- Discovers all `model_d*_*k.pt` files (excludes variance/split models)
- Unties output projection from embedding (copies weights into standalone layer)
- Extracts decoder features using forward hook on `decoder_norm`
- Excludes padding tokens from features/targets
- Fine-tunes the untied output projection with L-BFGS + L2
- Parallelizes across available GPUs (one model per GPU)
- No label smoothing during fine-tuning (decomposition requires standard CE)

### Output

```
output/transformer/03-01-1010/fine_tuned/
├── layer_d8_36k.pt                # Output projection state_dict only
├── layer_d16_36k.pt
├── ...
└── fine_tune_metadata.jsonl       # {d_model, train_samples, final_loss, final_grad_norm, steps, l2_lambda}
```

### Plotting (shared with ResNet18)

```bash
python -m jl.double_descent.plot_fine_tune \
    --resnet-path ./data/resnet18/03-01-1010 \
    --transformer-path ./data/transformer/03-01-1010 \
    --output-dir ./data
```

Produces `fine_tune_comparison.png` with side-by-side original vs fine-tuned test loss.

---

## Phase 8: M2M100 Reference Distribution Experiment (`--m2m100-variance` flag)

A new experiment mode that replaces the BPE tokenizer with M2M100's SentencePiece tokenizer and uses M2M100-12B as a reference distribution p(x) for a full distributional bias-variance decomposition.

### Motivation

The existing variance experiment (Phase 6) only evaluates the Jensen Gap at the ground truth token y, which is effectively a point estimate of p(x). This underestimates the true variance because it ignores the full shape of the predicted distributions. Using M2M100-12B as a reference distribution p(x) enables the complete decomposition:

```
E[CE(p, q)] = H(p) + KL(p || q_bar) + Jensen Gap
              ^^^^   ^^^^^^^^^^^^^^   ^^^^^^^^^^
             entropy      bias         variance
```

where H(p) is the entropy of the reference distribution, KL(p || q_bar) is the bias (divergence between reference and mean prediction), and the Jensen Gap is the variance (disagreement among individual predictions).

### Tokenizer Change

M2M100's SentencePiece tokenizer replaces the BPE 10K tokenizer. Only ~18,144 of M2M100's 128K vocabulary tokens actually appear in the IWSLT training data. These are remapped to "compact IDs" (0..~18K) to keep embedding tables small. M2M100's logits are renormalized over this compact subset.

Compact vocab special tokens: PAD=0, BOS=1, EOS=2, UNK=3. The full mapping is stored in `data/iwslt14.m2m100.de-en/vocab_mapping.json`.

### New Files

| File | Purpose |
|------|---------|
| `prepare_m2m100_data.py` | Downloads IWSLT14, tokenizes with M2M100 tokenizer (no lowercasing), saves compact IDs and vocab mapping to `data/iwslt14.m2m100.de-en/` |
| `extract_m2m100_reference.py` | One-time script on H100: loads M2M100-12B (FP16, ~24GB), teacher-forces test set, extracts top-500 renormalized logits per position + per-position entropy. Output: `reference_logits.pt` (~280MB) |

### Modified Files

| File | Change |
|------|--------|
| `transformer_data.py` | New classes: `M2M100Vocab`, `M2M100TranslationDataset`, `load_m2m100_iwslt14_variance_split()` |
| `trainer.py` | `m2m100: bool` parameter on `train_single_model()` |
| `transformer_main.py` | `--m2m100-variance` flag; pilot: d_model {112, 128} x 4 splits of 36K = 8 models = 1 batch |
| `variance_evaluation.py` | `--reference-logits` flag enables distributional decomposition (entropy, bias, variance) |
| `plot_variance_evaluation.py` | Auto-detects distributional format, plots entropy (constant line), bias, variance, test loss |
| `infra/train.sh` | `--m2m100-variance` flag |
| `infra/download.sh` | `transformer_m2m100_variance` experiment type |

### Running the Experiment

```bash
# 1. Preprocess data (CPU, ~5min)
python -m jl.double_descent.transformer.prepare_m2m100_data

# 2. Train variance models (8 GPUs required)
./infra/train.sh <ip> --module jl.double_descent.transformer.transformer_main --m2m100-variance

# 3. Extract M2M100 reference logits (one-time, single H100)
python -m jl.double_descent.transformer.extract_m2m100_reference \
    --data-path ./data/iwslt14.m2m100.de-en \
    --output-path ./data/iwslt14.m2m100.de-en/reference_logits.pt \
    --batch-size 128 --top-k 500

# 4. Evaluate with distributional decomposition
python -m jl.double_descent.transformer.variance_evaluation \
    --model-path ./output/transformer_m2m100_variance/<timestamp> \
    --data-path ./data/iwslt14.m2m100.de-en \
    --reference-logits ./data/iwslt14.m2m100.de-en/reference_logits.pt

# 5. Plot
python -m jl.double_descent.transformer.plot_variance_evaluation \
    ./data/transformer_m2m100_variance/<timestamp>/reference/evaluation.jsonl
```

### Output Format

Distributional `evaluation.jsonl` includes entropy, bias, and variance fields:

```json
{"d_model": 112, "mean_test_loss": 3.14, "entropy": 2.01, "bias": 0.85, "variance": 0.28}
{"d_model": 128, "mean_test_loss": 2.98, "entropy": 2.01, "bias": 0.72, "variance": 0.25}
```

The plot auto-detects this format (presence of `entropy` field) and renders:
- Entropy as a constant horizontal line
- Bias vs d_model
- Variance vs d_model
- Mean test loss vs d_model

### Pilot Scope

Currently hardcoded to d_model {112, 128} with 4 splits of 36K samples. This fits in a single batch (2 d_model x 4 splits = 8 GPUs). To expand the experiment, modify `M2M100_D_MODEL_VALUES` and related constants in `transformer_main.py`.

### Output Structure

```
output/transformer_m2m100_variance/<timestamp>/
├── metrics_d112_split0.jsonl
├── metrics_d112_split1.jsonl
├── metrics_d112_split2.jsonl
├── metrics_d112_split3.jsonl
├── metrics_d128_split0.jsonl
├── metrics_d128_split1.jsonl
├── metrics_d128_split2.jsonl
├── metrics_d128_split3.jsonl
├── model_d112_split0.pt
├── ...
└── model_d128_split3.pt
```

### Data Output Structure

```
data/iwslt14.m2m100.de-en/
├── train.de.pt              # Compact token IDs
├── train.en.pt
├── valid.de.pt
├── valid.en.pt
├── test.de.pt
├── test.en.pt
├── vocab_mapping.json       # M2M100 token ID → compact ID mapping
└── reference_logits.pt      # Top-500 renormalized logits from M2M100-12B (~280MB)
```

---

## Key Differences from Paper

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Optimizer | Adam | AdamW (beta1=0.9, beta2=0.98, eps=1e-9) |
| LR Schedule | Vaswani (inverse sqrt) | Warmup + cosine decay to 0 |
| Learning Rate | Scaled by d_model | Fixed 3e-4 peak |
| Label Smoothing | 0.1 | Disabled |
| Trials | Unknown | 1 (double descent) or 4 (variance) |
| Sample sizes | 4K and 18K | 36K (default), 36K per split (variance) |
| Framework | fairseq | Custom PyTorch |
| BLEU frequency | Unknown | End of training only |

---

## Hardware Requirements

- **Preprocessing**: CPU only, ~5 minutes
- **Training**: **Exactly 8 GPUs required** (script fails fast otherwise)
- **Memory**: ~2-8GB per GPU depending on d_model
- **Instance type**: 8x Nvidia GPUs from Lambda Labs
- **Estimated time**: ~3-4 hours for full experiment (24 models in 3 batches)

---

## Dependencies

### Training (add to requirements.txt)
- `sacrebleu` - BLEU score computation

### Preprocessing only (installed by `prepare_iwslt14.sh`)
- `subword-nmt` - BPE tokenization (pip install)
- Moses tokenizer - Perl scripts (cloned from GitHub, not pip)

**Note**: No fairseq dependency required. The preprocessing script is self-contained and based on fairseq's approach but doesn't import fairseq.
