# RETFound Calibration Experiment

## Goal

Evaluate post-hoc calibration approaches on **pre-trained RETFound** checkpoints across 7 ophthalmology datasets:

1. **Uncalibrated** — direct model outputs
2. **Temperature scaling** — fit scalar T on validation set via L-BFGS
3. **L2 calibration** — L-BFGS + L2 on classifier head using training set

## Results Summary

L2 calibration beats temperature scaling on 5 of 7 datasets, with the largest gains on datasets with more classes (JSIEC, 39 classes) and more training data (APTOS2019, 2K images).

| Dataset | Classes | Train | Val | Test | L2 ΔNLL | L2 ΔECE | L2 ΔAcc |
|---------|---------|-------|-----|------|---------|---------|---------|
| **APTOS2019** | 5 | 2,048 | 514 | 1,100 | **-0.048** | **-0.033** | **+1.7%** |
| MESSIDOR2 | 5 | 972 | 246 | 526 | +0.019 | -0.010 | -1.0% |
| IDRID | 5 | 329 | 84 | 104 | +0.097 | +0.085 | +3.9% |
| **PAPILA** | 3 | 311 | 79 | 98 | **-0.032** | +0.015 | **+5.1%** |
| **Glaucoma** | 3 | 861 | 218 | 465 | **-0.009** | **-0.013** | **+0.2%** |
| **JSIEC** | 39 | 534 | 150 | 318 | **-0.105** | **-0.077** | **+1.9%** |
| **Retina** | 4 | 336 | 84 | 181 | **-0.031** | +0.000 | **+2.2%** |

Deltas are vs uncalibrated. Bold = L2 calibration improves over both uncalibrated and temp scaling.

---

## Assets

### Pre-trained checkpoints and data
- **Source:** `rmaphoh/RETFound` [BENCHMARK.md](https://github.com/rmaphoh/RETFound/blob/main/BENCHMARK.md) → Google Drive
- **Architecture:** timm `vit_large_patch16_224`, global_pool="avg", 307M params
- **All CFP (color fundus photography)** — same base model, different trained heads
- **Data:** pre-split by paper authors into train/val/test ImageFolder format
- **Preprocessing:** AutoMorph, images already resized

### Local storage

Data and checkpoints stored as zips (checkpoint stripped of optimizer state, ~1.2GB each):

```
data/medical_calibration/
├── aptos2019.zip          # DR grading, 5 classes, 3,662 images
├── messidor2.zip          # DR grading, 5 classes, 1,744 images
├── idrid.zip              # DR grading, 5 classes, 517 images
├── papila.zip             # Glaucoma, 3 classes, 488 images
├── glaucoma_fundus.zip    # Glaucoma, 3 classes, 1,544 images
├── jsiec.zip              # Multi-disease, 39 classes, 1,002 images
├── retina_cataract.zip    # Cataract grading, 4 classes, 601 images
└── results/               # ECE-sweep calibration_results.json per dataset
    └── results_nll/       # NLL-sweep calibration_results.json per dataset
```

### Google Drive download links

Data splits:
- APTOS2019: `162YPf4OhMVxj9TrQH0GnJv0n7z7gJWpj`
- MESSIDOR2: `1vOLBUK9xdzNV8eVkRjVdNrRwhPfaOmda`
- IDRID: `1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3`
- PAPILA: `1JltYs7WRWEU0yyki1CQw5-10HEbqCMBE`
- Glaucoma_fundus: `18vSazOYDsUGdZ64gGkTg3E6jiNtcrUrI`
- JSIEC: `1q0GFQb-dYwzIx8AwlaFZenUJItix4s8z`
- Retina: `1vdmjMRDoUm9yk83HMArLiPcLDk_dm92Q`

Checkpoint folders (each contains `checkpoint-best.pth`):
- APTOS2019: `16kL5V-1U7ACc-68PSHjAq6vyXRJvUoq3`
- MESSIDOR2: `1OTBRAHNbaytpwzwMHw9SWrltJouEEuxF`
- IDRID: `18Ml-B7nhejK4rnNG8upjqIARSlMP5kUc`
- PAPILA: `1cHOX6C4NQVi9B6n-7Bxxg7b4-wdI4c73`
- Glaucoma_fundus: `10JbanmVxjyX6mghXbxGnGVX1p9nwqsja`
- JSIEC: `1eosdBXsONUy49cwDO80AOzDHkHiPNJvv`
- Retina: `1n7mXxN-ZUKauOrAlBAiF2E_36F6f0wZD`

Use `download_one.py <dataset_name>` to download, strip optimizer, and zip.

---

## Package Structure

```
jl/double_descent/medical_calibration/
├── __init__.py
├── config.py              # MedCalConfig dataclass
├── calibrate.py           # Load model, fit calibrators, evaluate, sweep
└── MEDICAL_CALIBRATION_PLAN.md
```

---

## Running

### Single dataset, single lambda

```bash
python -m jl.double_descent.medical_calibration.calibrate \
    --checkpoint ./data/medical_calibration/aptos2019_extracted/checkpoint-best.pth \
    --data-path ./data/medical_calibration/aptos2019_extracted/APTOS2019 \
    --output-path ./output/medical_calibration/aptos2019 \
    --l2-lambda 0.3
```

### Lambda sweep (select by val ECE)

```bash
python -m jl.double_descent.medical_calibration.calibrate \
    --checkpoint <checkpoint> --data-path <data> \
    --output-path <output> --sweep
```

### Lambda sweep (select by val NLL)

```bash
python -m jl.double_descent.medical_calibration.calibrate \
    --checkpoint <checkpoint> --data-path <data> \
    --output-path <output> --sweep --sweep-metric nll
```

### Run on remote via helper script

```bash
# Extract zip and run sweep (on remote instance)
bash infra/run_medical_calibration.sh <dataset_name>
```

Expects `data/medical_calibration/<dataset_name>.zip` on the remote. Extracts, finds train/ dir and checkpoint, runs sweep.

---

## How calibrate.py works

1. Load trained RETFound model (auto-detects num_classes from checkpoint)
2. Extract features once for all splits via `model.forward_features()` + `model.forward_head(x, pre_logits=True)` → [N, 1024]
3. Collect test logits for uncalibrated evaluation
4. **Temperature scaling:** fit scalar T on **validation** logits via L-BFGS, evaluate on test
5. **L2 calibration:** copy `model.head` into standalone `nn.Linear`, run `l2_calibrate_lib.l2_calibrate_final_layer()` with L-BFGS + L2 on **training** features
6. With `--sweep`: try 14 lambda values, select best by val metric (ECE or NLL), report test metrics for the winner
7. Save `calibration_results.json`, `test_logits.pt`, `calibrated_head.pt`

Sweep lambda values: `[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 1, 2, 3, 5, 10]`

---

## Metrics

| Metric | Purpose |
|--------|---------|
| NLL / cross-entropy | Calibration — proper scoring rule |
| Accuracy | Classification performance |
| ECE (20 bins) | Calibration — binned confidence vs accuracy |
| Brier score | Calibration + discrimination |
| AUROC (macro, one-vs-rest) | Discrimination |
| AUPR (macro) | Discrimination under class imbalance |

---

## Key Design Decisions

- **Temperature scaling fits on val** — standard practice, same data used for model selection
- **L2 calibration fits on train** — our method has more capacity (classes×1024 params), so we use the larger training set
- **Lambda selected on val** — proper train/val/test protocol, no test leakage
- **Feature extraction uses timm's `forward_head(pre_logits=True)`** — applies pool + fc_norm, returns [B, 1024]
- **Shared optimization via `l2_calibrate_lib.py`** — same L-BFGS code used for ResNet and Transformer calibration
- **Auto-detects num_classes from checkpoint** — works across all 7 datasets without config changes

---

## Hardware

- Single GPU sufficient (inference + L-BFGS on small feature matrices)
- Each dataset sweep takes ~1-2 minutes
- RunPod/Lambda via `infra/setup_remote.sh`

---

## Observations

- L2 calibration works best with **more classes** (JSIEC: 39 classes, biggest improvement) and **more training data** (APTOS: 2K images)
- On tiny datasets (IDRID: 329 train, 104 test) neither method helps much
- Selected lambdas vary by dataset (0.01 to 3.0) — the sweep is important
- ECE-selection and NLL-selection give similar results; ECE-selection slightly better overall
- L-BFGS converges by step ~11 for all datasets — 30 steps is more than sufficient
- The base model's strength matters: L2 calibration can't fix bad features (MESSIDOR2: AUROC 0.883)
- **SGD vs L-BFGS:** Full-batch SGD (momentum=0.9) was tested as an alternative optimizer. At lr=0.1 it diverges badly (NLL +1 to +10). At lr=0.01 it nearly matches L-BFGS on most datasets but remains worse on JSIEC (39 classes): ΔNLL -0.065 vs -0.105, ΔECE -0.049 vs -0.077. L-BFGS is the right tool — the problem is small and convex (1024×C weights), and L-BFGS converges to grad_norm ~1e-6 in ~11 steps vs SGD still at ~0.04 after 100 epochs.
