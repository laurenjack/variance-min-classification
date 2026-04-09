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
python -m jl.double_descent.calibration.calibrate_resnet \
    --model-path ./output/resnet18/03-01-1010 \
    --data-path ./data \
    --k 12
```
