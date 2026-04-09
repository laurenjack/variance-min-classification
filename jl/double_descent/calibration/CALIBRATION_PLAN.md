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
jl/double_descent/calibration/
├── __init__.py
├── baselines.py             # Temp scaling, vector scaling, histogram binning, Dirichlet L2
├── evaluate.py              # evaluate_logits, evaluate_probs, compute_ece, compute_brier_score
├── sweep.py                 # run_calibration_sweep() — shared core logic
├── config.py                # MedCalConfig dataclass
├── calibrate_retfound.py    # CLI: RETFound ophthalmology models
├── calibrate_resnet.py      # CLI: ResNet18k on CIFAR-10 (double descent)
├── calibrate_cifar.py       # CLI: ResNet-110 on CIFAR-10/100 (Guo et al. recipe)
├── calibrate_imagenet.py    # CLI: ResNet-152 / ViT-B/16 on ImageNet (pre-trained from timm)
├── resnet110.py             # ResNet-110 model definition (CIFAR variant, 6n+2 layers)
└── CALIBRATION_PLAN.md
```

---

## Running

### RETFound (ophthalmology datasets)

```bash
python -m jl.double_descent.calibration.calibrate_retfound \
    --checkpoint <checkpoint> --data-path <data> \
    --output-path <output>
```

### ResNet18k (CIFAR-10)

```bash
python -m jl.double_descent.calibration.calibrate_resnet \
    --model-path ./data/resnet18/long_double_descent \
    --data-path ./data --k 64
```

### ResNet-110 (CIFAR-10 / CIFAR-100, Guo et al. recipe)

Training and calibration are separate steps (same pattern as ResNet18k):

```bash
# 1. Train ResNet-110 on remote (~20 min single GPU)
python -m jl.double_descent.calibration.train_resnet110 \
    --dataset cifar100 --data-path ./data --output-path ./output

# 2. Download to local
./infra/download.sh $IP

# 3. Calibrate (local or remote)
python -m jl.double_descent.calibration.calibrate_cifar \
    --dataset cifar100 --model-path ./data/resnet110/04-09-1200 \
    --data-path ./data --output-path ./output/cifar100_calibration
```

The `--model-path` directory contains `resnet110_cifar10.pt` and/or `resnet110_cifar100.pt`.
The script selects the checkpoint matching `--dataset`.

### ImageNet (ResNet-152 / ViT-B/16)

```bash
# ResNet-152 (pre-trained from timm, no training)
python -m jl.double_descent.calibration.calibrate_imagenet \
    --model resnet152 --data-path ./data/imagenet --output-path ./output/imagenet

# ViT-B/16 (pre-trained from timm, no training)
python -m jl.double_descent.calibration.calibrate_imagenet \
    --model vit_base_patch16_224 --data-path ./data/imagenet --output-path ./output/imagenet
```

ImageNet data can be stored locally or streamed from HuggingFace (`ILSVRC/imagenet-1k`,
requires HF token and terms acceptance). Pass `--streaming` to use HuggingFace streaming
instead of local ImageFolder.

### RETFound with NLL selection

```bash
python -m jl.double_descent.calibration.calibrate_retfound \
    --checkpoint <checkpoint> --data-path <data> \
    --output-path <output> --sweep-metric nll
```

### Run on remote via helper script

```bash
# Extract zip and run sweep (on remote instance)
bash infra/run_medical_calibration.sh <dataset_name>
```

Expects `data/medical_calibration/<dataset_name>.zip` on the remote. Extracts, finds train/ dir and checkpoint, runs sweep.

---

## How sweep.py works

1. Dataset-specific script loads model, extracts features for train/val/test
2. `run_calibration_sweep()` receives features + original head state
3. Computes val/test logits from original head
4. **Baselines (fit on val):** temperature scaling, vector scaling, histogram binning, Dirichlet L2
5. **L2 calibration (fit on train):** sweep lambda values, select best by val metric (ECE or NLL), report on test
6. Save `calibration_results.json`, `sweep_results.json`, `test_logits.pt`, `calibrated_head.pt`

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

## CIFAR-10/100 + ResNet-110 (Guo et al. recipe)

### Goal

Replicate the canonical calibration benchmark from Guo et al. 2017 ("On Calibration of
Modern Neural Networks"): ResNet-110 on CIFAR-10 and CIFAR-100. This is the standard
experimental setup used by most calibration papers (Kull et al. 2019, Kumar et al. 2019,
Mukhoti et al. 2020, Nixon et al. 2019).

### Model

- **ResNet-110**: 6×18+2 = 110 layers, ~1.7M params, 3 groups of basic blocks (16→32→64 channels)
- **Train from scratch** — no pre-trained checkpoint available on HuggingFace
- Training recipe (He et al. 2015 / Guo et al. 2017): 200 epochs, SGD momentum 0.9,
  weight decay 1e-4, batch size 128, LR 0.1 dropped by 10x at epochs 100 and 150
- Standard data augmentation: random crop (32×32 with 4px padding), random horizontal flip
- Expected accuracy: ~93-94% (CIFAR-10), ~74-76% (CIFAR-100)
- Training time: ~20 min on a single H100/A100

### Dataset

- **CIFAR-10**: 50K train / 10K test, 10 classes, 32×32 RGB
- **CIFAR-100**: 50K train / 10K test, 100 classes (500 images/class), 32×32 RGB
- Auto-downloads via `torchvision.datasets.CIFAR10` / `CIFAR100`

### Val/test split (Guo et al. protocol)

Guo et al. hold out validation from the **training set**, leaving the test set untouched:

- **Train**: 45K (from 50K training set) — used for model training + L2 calibration features
- **Val**: 5K (held out from training set) — used for calibration fitting (baselines) + lambda selection
- **Test**: 10K (original test set, untouched) — used for final evaluation

This differs from our existing `calibrate_resnet.py` which splits the 10K test set in half.

### Implementation: `calibrate_cifar.py`

1. Train ResNet-110 on 45K training subset (or load existing checkpoint)
2. Extract features from penultimate layer (64-dim after global avg pool)
3. Run `sweep.py` with 45K train features / 5K val features / 10K test features

---

## ImageNet + ResNet-152 + ViT-B/16

### Goal

Evaluate L2 calibration on ImageNet-1K with two canonical architectures: ResNet-152
(Guo et al. 2017 standard) and ViT-B/16 (modern standard from Minderer et al. 2021).

### Models

Both loaded pre-trained from **timm** — no training needed:

- **ResNet-152**: `timm.create_model("resnet152", pretrained=True)` — 60.2M params,
  2048-dim features before head. Standard ImageNet calibration model (Guo et al., Kull et al.)
- **ViT-B/16**: `timm.create_model("vit_base_patch16_224", pretrained=True)` — 86.6M params,
  768-dim features before head. Modern calibration benchmark (Minderer et al. 2021)

### Dataset

- **ImageNet-1K (ILSVRC 2012)**: 1.28M train, 50K val, 1000 classes
- Source: HuggingFace `ILSVRC/imagenet-1k` (gated, requires HF token + terms acceptance)
- Download recommended (~150GB) — avoids streaming latency for repeated experiments
- Also supports `torchvision.datasets.ImageNet` if already on disk

### Val/test split (Guo et al. protocol)

ImageNet has no public test labels, so Guo et al. split the 50K validation set:

- **Train**: 1.28M (full training set) — L2 calibration uses all training features by default.
  Optional `--train-subset N` to subsample (e.g. 50K) for faster iteration
- **Val**: 25K (first half of 50K val, fixed seed) — calibration fitting + lambda selection
- **Test**: 25K (second half of 50K val, fixed seed) — final evaluation

### Implementation: `calibrate_imagenet.py`

1. Load pre-trained model from timm (ResNet-152 or ViT-B/16)
2. Extract features on train subset + val + test splits
3. Run `sweep.py` with extracted features


---

## Key Design Decisions

- **All baselines fit on val** — temperature scaling, vector scaling, histogram binning, Dirichlet L2
- **L2 calibration fits on train** — our method has more capacity (classes×d params), so we use the larger training set
- **Lambda selected on val** — proper train/val/test protocol, no test leakage
- **Shared sweep logic in `sweep.py`** — same code for RETFound, ResNet18, and future models
- **Feature extraction is model-specific** — `calibrate_retfound.py` uses timm's `forward_head(pre_logits=True)`, `calibrate_resnet.py` uses ResNet18k's penultimate layer
- **Shared optimization via `l2_calibrate_lib.py`** — same L-BFGS code used for ResNet and Transformer calibration
- **Auto-detects num_classes from checkpoint** — works across all datasets without config changes
- **Multi-GPU feature extraction** — `DataParallel` or sharded `DataLoader` across available GPUs; works transparently on single GPU too

---

## Hardware

- Works on single GPU; auto-parallelizes feature extraction across all available GPUs
- Multi-GPU speeds up feature extraction (the bottleneck for large datasets like ImageNet)
- L-BFGS sweep itself is CPU-bound on small feature matrices — fast regardless of GPU count
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
- **Magnitude-only variant:** Tested learning one scalar α_c per class (C params) instead of the full weight matrix (1024×C params). Logit_c = α_c * (W_c @ x + b_c) with L2 penalty λ * Σ_c α_c² * (||W_c||² + b_c²). Results: better ECE than full L2 on JSIEC (-0.092 vs -0.077) and APTOS (-0.038 vs -0.033), but consistently worse NLL. Less harmful on weak datasets (IDRID, MESSIDOR2). Removed from code — full L2 is better overall. See commit `27daec0` for implementation.
