# RETFound + APTOS-2019 Calibration Experiment Plan

## Goal

Use the **pre-fine-tuned RETFound checkpoint** on APTOS-2019 to evaluate calibration approaches:

1. **Uncalibrated** — direct model outputs
2. **Temperature scaling** — fit scalar T on validation set
3. **Final-layer fine-tuning** — L-BFGS + L2 on classifier head using training set

## Assets

### Pre-fine-tuned model
- **Source:** `rmaphoh/RETFound` BENCHMARK.md → Google Drive
- **File:** `data/medical_calibration/checkpoint-best.pth` (3.4GB, epoch 27/50)
- **Architecture:** timm `vit_large_patch16_224`, global_pool="avg", 307M params
- **Head:** `nn.Linear(1024, 5)` — 5-class DR severity grading
- **Checkpoint keys:** `model`, `optimizer`, `epoch`, `scaler`, `args`

### Data (pre-split by paper authors)
- **Source:** `rmaphoh/RETFound` BENCHMARK.md → Google Drive
- **Location:** `data/medical_calibration/APTOS2019/`
- **Preprocessing:** AutoMorph, images already resized
- **Format:** ImageFolder with class subdirectories

| Split | Count | Purpose |
|-------|-------|---------|
| Train | 2,048 (56%) | Final-layer fine-tuning (our method) |
| Val | 514 (14%) | Temperature scaling fitting |
| Test | 1,100 (30%) | Final evaluation of all methods |

Class distribution per split:

| Class | Name | Train | Val | Test |
|-------|------|-------|-----|------|
| 0 | No DR | 1,010 | 253 | 542 |
| 1 | Mild NPDR | 207 | 52 | 111 |
| 2 | Moderate NPDR | 559 | 140 | 300 |
| 3 | Severe NPDR | 108 | 27 | 58 |
| 4 | Proliferative DR | 164 | 42 | 89 |

---

## Package Structure

```
jl/double_descent/medical_calibration/
├── __init__.py
├── config.py              # MedCalConfig dataclass
├── calibrate.py           # Main entry: load model, fit calibrators, evaluate
├── data.py                # Data loading utilities (unused for pre-split data)
├── model.py               # Model building utilities (unused with pre-finetuned ckpt)
├── train.py               # Fine-tuning from scratch (unused with pre-finetuned ckpt)
├── main.py                # Fine-tuning entry point (unused with pre-finetuned ckpt)
└── plot.py                # Reliability diagrams (TODO)
```

**Note:** `data.py`, `model.py`, `train.py`, `main.py` support fine-tuning from scratch
if needed, but are not used when working with the pre-fine-tuned checkpoint.

---

## Experimental Protocol

### Phase 1 — Setup
- Download pre-fine-tuned checkpoint and pre-split data from Google Drive
- Place in `data/medical_calibration/`
- Verify checkpoint loads (5-class head present, epoch 27)

### Phase 2 — Calibration (single script: `calibrate.py`)

```bash
python -m jl.double_descent.medical_calibration.calibrate \
    --checkpoint ./data/medical_calibration/checkpoint-best.pth \
    --data-path ./data/medical_calibration/APTOS2019 \
    --output-path ./output/medical_calibration \
    --l2-lambda 1e-3 --max-steps 100
```

Steps within `calibrate.py`:
1. Load fine-tuned RETFound model
2. Run uncalibrated inference on test set → save logits
3. Fit temperature T on **validation** logits via L-BFGS → evaluate on test
4. Extract training features via `model.forward_features()` → [2048, 1024]
5. Copy `model.head` into standalone `nn.Linear(1024, 5)`
6. Call `fine_tune_lib.fine_tune_final_layer()` with L-BFGS + L2 on **training** features
7. Evaluate fine-tuned head on test features
8. Save `calibration_results.json` with all metrics

### Phase 3 — Analysis
- Print results table and delta table
- Generate reliability diagrams (TODO: `plot.py`)

---

## Metrics

| Metric | Purpose |
|--------|---------|
| NLL / cross-entropy | Calibration — proper scoring rule |
| Accuracy | Classification performance |
| ECE (20 bins) | Calibration — binned confidence vs accuracy |
| Brier score | Calibration — combined discrimination + calibration |
| AUROC (macro, one-vs-rest) | Discrimination — does ranking change? |
| AUPR (macro) | Discrimination under class imbalance |

Paper reference: RETFound reports AUROC 0.944 (95% CI: 0.941–0.946) on APTOS-2019.

---

## Success Criteria

A calibration method is promising if, relative to the uncalibrated model, it:

- Reduces **ECE**
- Improves **NLL**
- Improves **Brier score**
- Preserves or minimally changes **AUROC**
- Preserves or minimally changes **AUPR**

Temperature scaling is the minimum baseline to beat.

---

## Outputs

### Saved artifacts
- `calibration_results.json` — all metrics for all three methods
- `test_logits.pt` — uncalibrated test logits + labels
- `calibrated_head.pt` — fine-tuned head weights

### Tables (printed by calibrate.py)
1. **Main results table** — Method × {NLL, Acc, ECE, Brier, AUROC, AUPR}
2. **Delta table** — Method × {ΔNLL, ΔAcc, ΔECE, ΔBrier, ΔAUROC, ΔAUPR}

### Figures (TODO)
- Reliability diagram per method (3 panels)

---

## Hardware

- Single GPU sufficient (inference + L-BFGS on 2K features)
- Can run on A40, A100, or even smaller GPUs
- RunPod/Lambda via `infra/setup_remote.sh`

---

## Key Design Decisions

- **Temperature scaling fits on val** — standard practice, same data used for model selection
- **Final-layer fine-tuning fits on train** — our method has more capacity (1024×5 + 5 params), so we use the larger training set. This is analogous to how we fine-tune in the double descent experiments
- **Feature extraction uses `model.forward_features()`** — timm's built-in method returns pooled features before the head layer, model-agnostic
- **Shared optimization via `fine_tune_lib.py`** — same L-BFGS code used for ResNet and Transformer calibration
