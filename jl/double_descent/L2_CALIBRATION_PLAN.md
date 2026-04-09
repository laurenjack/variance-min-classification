# L2 Calibrate Sweep Plan

## Goal

Sweep over L2 lambda values for a single ResNet18 model (specified by `k`),
calibrating the final linear layer. Selects best lambda by validation ECE
and reports test metrics. Now lives in `calibration/calibrate_resnet18k.py`.

## Key Decisions

- **ResNet only** (no transformer for now)
- **SGD only** (not L-BFGS)
- **No model saving** — just report results to file and stdout
- **Test set split**: CIFAR-10 test set (10K) randomly split in half with a fixed
  seed. First 5K = validation (for lambda selection), second 5K = test (for
  final reporting).
- **Multi-GPU**: parallelizes across lambda values (batches of `num_gpus` at a
  time), same pattern as existing `l2_calibrate.py` which parallelizes across k
  values. Single k, but multiple lambdas run concurrently.

## Lambda Values

Same range as medical calibration:
```python
[1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]
```

## SGD Defaults

- `lr`: 0.1
- `epochs`: 100
- `momentum`: 0.9

These are configurable via CLI args.

## Metrics

Per-lambda, evaluated on both val and test halves:
- NLL (cross-entropy loss)
- Accuracy
- ECE (expected calibration error)
- Brier score (multi-class)

## Changes

### 1. `l2_calibrate_lib.py` — add `compute_brier_score`

Reuse the Brier score formula from `calibration/evaluate.py`:
```python
def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()
```

### 2. `calibration/calibrate_resnet18k.py` (was `resnet18/l2_calibrate_sweep.py`)

**CLI args:**
- `--model-path` (required) — directory containing `model_k*.pt` files
- `--data-path` (default: `./data`) — CIFAR-10 data directory
- `--k` (required) — which model width to sweep
- `--sgd-epochs` (default: 100)
- `--sgd-lr` (default: 0.1)
- `--seed` (default: 42) — seed for test set split

**Flow:**
1. Main process (CPU): load model for given `k`, freeze backbone
2. Main process (CPU): split CIFAR-10 test set into val/test halves (fixed seed)
3. Main process (CPU): extract features from train, val, and test sets (on GPU 0)
4. Save original layer state dict for workers to copy
5. Spawn workers in batches of `num_gpus`, each worker gets one lambda:
   a. Worker receives: `gpu_id`, `l2_lambda`, pre-extracted features (via
      shared memory / reloaded on each GPU), original layer weights
   b. Copy original weights into fresh `nn.Linear`
   c. Run `sgd_l2_calibrate_final_layer` on training features
   d. Evaluate on val features → compute NLL, accuracy, ECE, Brier
   e. Evaluate on test features → compute same metrics
   f. Write result to shared mp.Manager dict
6. Main process: collect all results, select best lambda by **val ECE** (lowest)
7. Print summary table (all lambdas with val+test metrics, mark best)
8. Write results to `<model-path>/l2_calibrate_sweep/sweep_k{k}.jsonl`
   - One JSON line per lambda: `{l2_lambda, val_nll, val_accuracy, val_ece, val_brier, test_nll, test_accuracy, test_ece, test_brier}`

**Note on feature sharing:** Features are extracted once on GPU 0 in the main
process, saved to CPU tensors. Each spawned worker loads them onto its assigned
GPU. This avoids redundant forward passes — only the SGD calibration + eval
runs per-GPU.

**Evaluation helper** (inline in the file):
```python
def evaluate_calibrated(linear, features, labels, device) -> dict:
    """Compute NLL, accuracy, ECE, Brier on given features/labels."""
```
Uses `compute_ece` from `resnet18/evaluation.py` and `compute_brier_score` from
`l2_calibrate_lib.py`.

**Summary table** (printed to stdout, same style as medical calibration):
```
Lambda      Val ECE    Val NLL    Val Acc    Test ECE   Test NLL   Test Acc   Test Brier
1e-04        0.0312     0.8234     0.8520     0.0298     0.8190     0.8540     0.2100
...                                                                            <-- best
```

## Output Structure

```
<model-path>/
  l2_calibrate_sweep/
    sweep_k4.jsonl
    sweep_k8.jsonl
    ...
```

## Experiment Results (k=64, long_double_descent)

### Full-model vs final-layer calibration

Tested full-model SGD calibration (all parameters + weight decay) against
final-layer L2 calibration. Full-model was consistently worse:

| Mode | Best Lambda | Test ECE | Test Acc | Test NLL |
|------|------------|----------|----------|----------|
| Final-layer (SGD, lr=0.1) | 5e-3 | 0.0184 | 86.6% | 0.467 |
| Full-model (SGD, lr=0.1) | 1e-5 | 0.0930 | 85.8% | 0.728 |
| Full-model (SGD, lr=0.03) | 5e-5 | 0.0789 | 86.8% | 0.578 |

**Conclusion:** Full-model calibration destroys learned backbone features even
with tiny weight decay. Final-layer calibration is strictly better — it
preserves the backbone and only adjusts the decision boundary. Removed `--full`
mode from the sweep script. Also switched to L-BFGS (matching l2_calibrate.py)
since we only calibrate a single linear layer.

## Usage

```bash
source venv/bin/activate
python -m jl.double_descent.calibration.calibrate_resnet18k \
    --model-path ./output/resnet18/03-01-1010 \
    --data-path ./data \
    --k 12
```

---

## Transformer Calibration Sweep (IWSLT'14)

### Goal

Sweep over L2 lambda values for a single Transformer model (specified by
`--d-model`), calibrating the output projection layer. Baselines fit on val,
L2 calibration fit on train, best lambda selected by val metric, final
evaluation on test.

### Key Decisions

- **Single d_model** — `--d-model` is required, no default
- **L-BFGS only** — same as other calibration sweeps
- **Untied output projection** — the Transformer ties `output_proj.weight`
  to `embedding.weight`. L2 calibration creates an untied copy, same as
  `transformer/l2_calibrate.py`
- **Baselines**: uncalibrated, temperature scaling, vector scaling. No
  histogram binning (useless for NLL on large vocab), no Dirichlet L2
  (K×K matrix infeasible with ~10K vocab)
- **Multi-GPU parallelism** — L-BFGS fitting is sequential (fast), but BLEU
  evaluation requires autoregressive decoding so we parallelize across
  lambdas: each GPU loads the full model, swaps in a calibrated head, and
  runs greedy decoding on the test set

### Metrics

- **NLL** (token-level cross-entropy, primary calibration metric)
- **Token-level accuracy**
- **Token-level ECE** (top-1 confidence vs correctness, 20 bins)
- **BLEU** (corpus-level via sacrebleu, greedy decoding — task quality)

No Brier score (uninformative with ~10K vocab), no AUROC/AUPR.

### Data Split

Uses the existing IWSLT'14 train/valid/test splits:
- **Train**: subsampled training set (same `train_samples` as original training) — used for L2 calibration feature extraction
- **Val**: IWSLT'14 valid split (~7K sentences) — baselines fit here, lambda selection
- **Test**: IWSLT'14 test split (~7K sentences) — final evaluation

### Implementation: `calibration/calibrate_transformer.py`

**CLI args:**
- `--model-path` (required) — directory containing `model_d*_*k.pt` files
- `--data-path` (required) — IWSLT'14 preprocessed data directory
- `--d-model` (required) — which model dimension to calibrate
- `--output-path` (optional) — output directory
- `--max-steps` (default: 100) — L-BFGS steps per lambda
- `--sweep-metric` (default: ece) — `ece`, `nll`, or `bleu`
- `--train-samples` (default: 36000) — must match original training

**Flow (hybrid: sweep.py + multi-GPU BLEU pass):**

1. Load model for given `d_model`, freeze, untie output projection
2. Extract decoder features for train/val/test using forward hook on
   `decoder_norm` (same approach as `transformer/l2_calibrate.py`)
3. **Phase 1 — sweep.py (modified):** Run baselines (temperature, vector
   scaling) + L2 lambda sweep. Computes token-level NLL, accuracy, ECE on
   val and test. `sweep.py` needs a `skip_baselines` option to skip
   histogram binning and Dirichlet L2.
4. **Phase 2 — multi-GPU BLEU pass:** For each lambda (+ uncalibrated +
   temperature + vector scaling), spawn a worker per GPU. Each worker:
   a. Loads full model on its GPU
   b. Unties output projection, swaps in calibrated weights
   c. Runs `compute_bleu()` on the test set (greedy decoding)
   d. Returns BLEU score
5. Combine Phase 1 + Phase 2 results. Select best lambda by val metric.
6. Print summary table, save results.

**Swapping calibrated weights for BLEU:**
```python
# Untie output_proj from embedding
model.output_proj = nn.Linear(d_model, vocab_size, bias=False).to(device)
# Load calibrated weights
model.output_proj.load_state_dict(calibrated_state)
```

### Lambda Values

Same as default: `[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 1.0, 2.0, 3.0, 5.0, 10.0]`

### Usage

```bash
source venv/bin/activate
python -m jl.double_descent.calibration.calibrate_transformer \
    --model-path ./output/transformer/03-01-1010 \
    --data-path ./data/iwslt14.tokenized.de-en \
    --d-model 128
```
