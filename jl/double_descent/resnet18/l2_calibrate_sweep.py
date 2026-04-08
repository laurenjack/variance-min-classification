"""Sweep L2 lambda values for ResNet18 final-layer calibration (L-BFGS).

Loads a single ResNet18 model (by k), extracts features once, then sweeps
over lambda values in parallel across GPUs. Selects best lambda by validation
ECE and reports test metrics.

The CIFAR-10 test set (10K) is split in half with a fixed seed:
first half = validation (for lambda selection), second half = test.

Usage:
    python -m jl.double_descent.resnet18.l2_calibrate_sweep \
        --model-path ./output/resnet18/03-01-1010 \
        --data-path ./data \
        --k 12
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from jl.double_descent.calibration_baselines import (
    apply_dirichlet_l2,
    apply_histogram_binning,
    apply_vector_scaling,
    fit_dirichlet_l2,
    fit_histogram_binning,
    fit_vector_scaling,
)
from jl.double_descent.l2_calibrate_lib import compute_brier_score, l2_calibrate_final_layer
from jl.double_descent.resnet18.evaluation import compute_ece, discover_models
from jl.double_descent.resnet18.l2_calibrate import extract_features
from jl.double_descent.resnet18.resnet18_config import DDConfig
from jl.double_descent.resnet18.resnet18_data import load_cifar10_with_noise
from jl.double_descent.resnet18.resnet18k import make_resnet18k

logger = logging.getLogger(__name__)

LAMBDAS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]


def evaluate_calibrated(
    linear: nn.Linear, features: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> Dict[str, float]:
    """Compute NLL, accuracy, ECE, Brier on given features/labels."""
    linear.eval()
    with torch.no_grad():
        logits = linear(features.to(device)).cpu()
    return evaluate_logits(logits, labels)


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute NLL, accuracy, ECE, Brier from logits."""
    probs = F.softmax(logits, dim=-1)
    max_probs, predictions = probs.max(dim=-1)
    correct = predictions == labels

    nll = F.cross_entropy(logits, labels).item()
    accuracy = correct.float().mean().item()
    ece = compute_ece(max_probs, correct)
    brier = compute_brier_score(probs, labels)

    return {
        "nll": round(nll, 6),
        "accuracy": round(accuracy, 6),
        "ece": round(ece, 6),
        "brier": round(brier, 6),
    }


def evaluate_probs(probs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute NLL, accuracy, ECE, Brier from probabilities."""
    probs = probs.clamp(min=1e-8)
    max_probs, predictions = probs.max(dim=-1)
    correct = predictions == labels

    nll = -torch.log(probs[torch.arange(len(labels)), labels]).mean().item()
    accuracy = correct.float().mean().item()
    ece = compute_ece(max_probs, correct)
    brier = compute_brier_score(probs, labels)

    return {
        "nll": round(nll, 6),
        "accuracy": round(accuracy, 6),
        "ece": round(ece, 6),
        "brier": round(brier, 6),
    }


def sweep_worker(
    gpu_id: int,
    l2_lambda: float,
    k: int,
    original_layer_state: dict,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    max_steps: int,
    result_dict: dict,
) -> None:
    """Run L-BFGS L2 calibration for one lambda value on one GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"[k={k}, lambda={l2_lambda:.0e}] Starting on GPU {gpu_id}")

    # Copy original weights into fresh linear layer
    in_features = 8 * k
    linear = nn.Linear(in_features, 10, bias=True).to(device)
    linear.load_state_dict(original_layer_state)

    # Run L-BFGS calibration
    l2_calibrate_final_layer(
        features=train_features,
        targets=train_labels,
        linear_layer=linear,
        l2_lambda=l2_lambda,
        max_steps=max_steps,
        device=device,
    )

    # Evaluate on val and test
    val_metrics = evaluate_calibrated(linear, val_features, val_labels, device)
    test_metrics = evaluate_calibrated(linear, test_features, test_labels, device)

    result_dict[l2_lambda] = {
        "l2_lambda": l2_lambda,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    logger.info(
        f"[k={k}, lambda={l2_lambda:.0e}] Done: "
        f"val_ece={val_metrics['ece']:.4f}, test_ece={test_metrics['ece']:.4f}"
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Sweep L2 lambda for ResNet18 final-layer calibration (L-BFGS)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_k*.pt files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Directory containing CIFAR-10 data",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="ResNet18 width parameter k",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of L-BFGS steps per lambda (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for test set val/test split (default: 42)",
    )
    args = parser.parse_args()

    # Discover and validate model
    models = discover_models(args.model_path)
    if args.k not in models:
        available = sorted(models.keys())
        raise ValueError(f"No model found for k={args.k}. Available: {available}")

    model_path = str(models[args.k])
    logger.info(f"Using model: {model_path}")

    # Pre-download CIFAR-10 before spawning workers
    import torchvision
    torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True)
    torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True)

    # Compute val/test split indices
    n_test = 10000  # CIFAR-10 test set size
    perm = torch.randperm(n_test, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[: n_test // 2]
    test_idx = perm[n_test // 2 :]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and freeze
    model = make_resnet18k(k=args.k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    original_layer_state = model.linear.state_dict()

    # Extract training features
    config = DDConfig()
    train_loader, _ = load_cifar10_with_noise(
        noise_prob=config.label_noise,
        batch_size=config.batch_size,
        data_augmentation=False,
        data_dir=args.data_path,
    )

    logger.info("Extracting training features...")
    all_train_features = []
    all_train_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            feats = extract_features(model, images)
            all_train_features.append(feats.cpu())
            all_train_labels.append(labels)

    train_features = torch.cat(all_train_features, dim=0)
    train_labels = torch.cat(all_train_labels, dim=0)
    logger.info(f"Training features: {train_features.shape}")

    # Extract test features and split into val/test
    _, test_loader = load_cifar10_with_noise(
        noise_prob=0.0,
        batch_size=config.batch_size,
        data_augmentation=False,
        data_dir=args.data_path,
    )

    logger.info("Extracting test features...")
    all_test_features = []
    all_test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feats = extract_features(model, images)
            all_test_features.append(feats.cpu())
            all_test_labels.append(labels)

    full_test_features = torch.cat(all_test_features, dim=0)
    full_test_labels = torch.cat(all_test_labels, dim=0)

    val_features = full_test_features[val_idx]
    val_labels = full_test_labels[val_idx]
    test_features = full_test_features[test_idx]
    test_labels = full_test_labels[test_idx]
    logger.info(f"Val: {val_features.shape[0]} samples, Test: {test_features.shape[0]} samples")

    # Compute val/test logits from original model for baselines
    with torch.no_grad():
        linear_layer = model.linear
        val_logits = linear_layer(val_features.to(device)).cpu()
        test_logits = linear_layer(test_features.to(device)).cpu()

    # Evaluate uncalibrated
    uncal_test = evaluate_logits(test_logits, test_labels)
    logger.info(f"Uncalibrated: {uncal_test}")

    # === Calibration baselines (fit on val) ===
    logger.info("=== Fitting calibration baselines on val ===")

    # Temperature scaling
    temperature = nn.Parameter(torch.ones(1))
    ts_opt = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    def ts_closure():
        ts_opt.zero_grad()
        loss = F.cross_entropy(val_logits / temperature, val_labels)
        loss.backward()
        return loss
    ts_opt.step(ts_closure)
    T = temperature.item()
    ts_test = evaluate_logits(test_logits / T, test_labels)
    logger.info(f"Temperature scaling (T={T:.4f}): {ts_test}")

    # Vector scaling
    vs_weights, vs_biases = fit_vector_scaling(val_logits, val_labels)
    vs_test = evaluate_logits(apply_vector_scaling(test_logits, vs_weights, vs_biases), test_labels)
    logger.info(f"Vector scaling: {vs_test}")

    # Histogram binning
    bin_bounds, bin_accs = fit_histogram_binning(val_logits, val_labels)
    hb_test = evaluate_probs(apply_histogram_binning(test_logits, bin_bounds, bin_accs), test_labels)
    logger.info(f"Histogram binning: {hb_test}")

    # Dirichlet L2
    dir_W, dir_b = fit_dirichlet_l2(val_logits, val_labels)
    dir_test = evaluate_logits(apply_dirichlet_l2(test_logits, dir_W, dir_b), test_labels)
    logger.info(f"Dirichlet L2: {dir_test}")

    baselines = {
        "uncalibrated": uncal_test,
        "temperature_scaled": {**ts_test, "temperature": round(T, 6)},
        "vector_scaled": vs_test,
        "histogram_binning": hb_test,
        "dirichlet_l2": dir_test,
    }

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Run sweep across GPUs
    num_gpus = max(torch.cuda.device_count(), 1)
    logger.info(f"Sweeping {len(LAMBDAS)} lambda values across {num_gpus} GPUs")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    for batch_start in range(0, len(LAMBDAS), num_gpus):
        batch = LAMBDAS[batch_start : batch_start + num_gpus]
        batch_num = batch_start // num_gpus + 1
        total_batches = (len(LAMBDAS) + num_gpus - 1) // num_gpus
        logger.info(
            f"Batch {batch_num}/{total_batches}: "
            f"lambda = {[f'{l:.0e}' for l in batch]}"
        )

        processes = []
        for gpu_id, l2_lambda in enumerate(batch):
            p = mp.Process(
                target=sweep_worker,
                args=(
                    gpu_id % num_gpus,
                    l2_lambda,
                    args.k,
                    original_layer_state,
                    train_features,
                    train_labels,
                    val_features,
                    val_labels,
                    test_features,
                    test_labels,
                    args.max_steps,
                    result_dict,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Worker exited with code {p.exitcode}")

        logger.info(f"Batch {batch_num}/{total_batches} complete")

    # Collect results in lambda order
    results = [result_dict[lam] for lam in LAMBDAS if lam in result_dict]

    # Select best by val ECE
    best = min(results, key=lambda r: r["val_ece"])
    best_lambda = best["l2_lambda"]
    logger.info(f"Best lambda={best_lambda:.0e} (val ECE={best['val_ece']:.4f})")

    # Print baselines
    print(f"\n=== Calibration comparison for k={args.k} ===")
    print("=" * 60)
    print(f"{'Method':<20} {'Test NLL':>10} {'Test Acc':>10} {'Test ECE':>10} {'Test Brier':>10}")
    print("-" * 60)
    for method, metrics in baselines.items():
        print(
            f"{method:<20} {metrics['nll']:>10.4f} {metrics['accuracy']:>10.4f}"
            f" {metrics['ece']:>10.4f} {metrics['brier']:>10.4f}"
        )
    # Print best L2 calibration result
    print(
        f"{'l2_calibrated':<20} {best['test_nll']:>10.4f} {best['test_accuracy']:>10.4f}"
        f" {best['test_ece']:>10.4f} {best['test_brier']:>10.4f}  (λ={best_lambda:.0e})"
    )
    print("=" * 60)

    # Print deltas vs uncalibrated
    base = baselines["uncalibrated"]
    print(f"\n{'Method':<20} {'ΔNLL':>10} {'ΔAcc':>10} {'ΔECE':>10} {'ΔBrier':>10}")
    print("-" * 60)
    for method in ["temperature_scaled", "vector_scaled", "histogram_binning", "dirichlet_l2"]:
        m = baselines[method]
        print(
            f"{method:<20} {m['nll'] - base['nll']:>+10.4f} {m['accuracy'] - base['accuracy']:>+10.4f}"
            f" {m['ece'] - base['ece']:>+10.4f} {m['brier'] - base['brier']:>+10.4f}"
        )
    print(
        f"{'l2_calibrated':<20} {best['test_nll'] - base['nll']:>+10.4f} {best['test_accuracy'] - base['accuracy']:>+10.4f}"
        f" {best['test_ece'] - base['ece']:>+10.4f} {best['test_brier'] - base['brier']:>+10.4f}"
    )
    print("=" * 60)

    # Print L2 sweep table
    print(f"\nL2 Calibrate Sweep for k={args.k} (L-BFGS, max_steps={args.max_steps})")
    print(f"Selecting by val ECE")
    print("=" * 90)
    print(
        f"{'Lambda':<12} {'Val ECE':>10} {'Val NLL':>10} {'Val Acc':>10} {'Val Brier':>10}"
        f" {'Test ECE':>10} {'Test NLL':>10} {'Test Acc':>10}"
    )
    print("-" * 90)
    for r in results:
        marker = " <-- best" if r["l2_lambda"] == best_lambda else ""
        print(
            f"{r['l2_lambda']:<12.0e} {r['val_ece']:>10.4f} {r['val_nll']:>10.4f}"
            f" {r['val_accuracy']:>10.4f} {r['val_brier']:>10.4f}"
            f" {r['test_ece']:>10.4f} {r['test_nll']:>10.4f}"
            f" {r['test_accuracy']:>10.4f}{marker}"
        )
    print("=" * 90)

    # Write results
    output_dir = Path(args.model_path) / "l2_calibrate_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sweep_k{args.k}.jsonl"

    all_results = {
        "baselines": baselines,
        "l2_sweep": [dict(r) for r in results],
        "best_l2": {"l2_lambda": best_lambda, **{k: best[k] for k in best if k != "l2_lambda"}},
    }
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
