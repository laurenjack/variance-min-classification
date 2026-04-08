"""Sweep L2 lambda values for ResNet18 calibration (SGD).

Two modes:
  Default: Final-layer only. Extracts features once, sweeps lambdas on the
           final linear layer.
  --full:  Full model. Trains all parameters with SGD + weight decay.

Loads a single ResNet18 model (by k), sweeps over lambda values in parallel
across GPUs. Selects best lambda by validation ECE and reports test metrics.

The CIFAR-10 test set (10K) is split in half with a fixed seed:
first half = validation (for lambda selection), second half = test.

Usage:
    # Final-layer sweep
    python -m jl.double_descent.resnet18.l2_calibrate_sweep \
        --model-path ./output/resnet18/03-01-1010 \
        --data-path ./data \
        --k 12

    # Full-model sweep
    python -m jl.double_descent.resnet18.l2_calibrate_sweep \
        --model-path ./output/resnet18/03-01-1010 \
        --data-path ./data \
        --k 12 --full
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from jl.double_descent.l2_calibrate_lib import compute_brier_score, sgd_l2_calibrate_final_layer
from jl.double_descent.resnet18.evaluation import compute_ece, discover_models
from jl.double_descent.resnet18.l2_calibrate import extract_features
from jl.double_descent.resnet18.resnet18_config import DDConfig
from jl.double_descent.resnet18.resnet18_data import NoisyCIFAR10, load_cifar10_with_noise
from jl.double_descent.resnet18.resnet18k import make_resnet18k

logger = logging.getLogger(__name__)

LAMBDAS = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]


def evaluate_calibrated(
    linear: nn.Linear, features: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> Dict[str, float]:
    """Compute NLL, accuracy, ECE, Brier on given features/labels."""
    linear.eval()
    with torch.no_grad():
        logits = linear(features.to(device)).cpu()

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


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Compute NLL, accuracy, ECE, Brier on a full model with a data loader."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

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


def full_sweep_worker(
    gpu_id: int,
    l2_lambda: float,
    k: int,
    model_path: str,
    data_path: str,
    val_indices: torch.Tensor,
    test_indices: torch.Tensor,
    sgd_epochs: int,
    sgd_lr: float,
    result_dict: dict,
) -> None:
    """Run full-model SGD L2 calibration for one lambda value on one GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"[k={k}, lambda={l2_lambda:.0e}, FULL] Starting on GPU {gpu_id}")

    config = DDConfig()

    # Load model
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.train()

    # Load training data (no augmentation for consistency)
    train_loader, _ = load_cifar10_with_noise(
        noise_prob=config.label_noise,
        batch_size=config.batch_size,
        data_augmentation=False,
        data_dir=data_path,
    )

    # SGD with weight decay = 2 * l2_lambda (same convention as l2_calibrate_lib)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=sgd_lr,
        momentum=0.9,
        weight_decay=2 * l2_lambda,
    )

    # Train
    for epoch in range(sgd_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 10 == 0 or epoch == sgd_epochs - 1:
            avg_loss = epoch_loss / num_batches
            logger.info(f"[k={k}, lambda={l2_lambda:.0e}] Epoch {epoch + 1}/{sgd_epochs}: loss={avg_loss:.6f}")

    # Build val/test loaders from split indices
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    test_dataset = NoisyCIFAR10(
        root=data_path,
        train=False,
        noise_prob=0.0,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    val_loader = DataLoader(
        Subset(test_dataset, val_indices.tolist()),
        batch_size=config.batch_size, shuffle=False, num_workers=4,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices.tolist()),
        batch_size=config.batch_size, shuffle=False, num_workers=4,
    )

    # Evaluate
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    result_dict[l2_lambda] = {
        "l2_lambda": l2_lambda,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    logger.info(
        f"[k={k}, lambda={l2_lambda:.0e}, FULL] Done: "
        f"val_ece={val_metrics['ece']:.4f}, test_ece={test_metrics['ece']:.4f}"
    )


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
    sgd_epochs: int,
    sgd_lr: float,
    result_dict: dict,
) -> None:
    """Run SGD L2 calibration for one lambda value on one GPU."""
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

    # Run SGD calibration
    sgd_l2_calibrate_final_layer(
        features=train_features,
        targets=train_labels,
        linear_layer=linear,
        l2_lambda=l2_lambda,
        epochs=sgd_epochs,
        lr=sgd_lr,
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
        description="Sweep L2 lambda for ResNet18 final-layer calibration (SGD)"
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
        "--sgd-epochs",
        type=int,
        default=100,
        help="Number of SGD epochs per lambda (default: 100)",
    )
    parser.add_argument(
        "--sgd-lr",
        type=float,
        default=0.1,
        help="SGD learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for test set val/test split (default: 42)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full-model L2 calibration (all parameters, not just final layer)",
    )
    args = parser.parse_args()

    # Discover and validate model
    models = discover_models(args.model_path)
    if args.k not in models:
        available = sorted(models.keys())
        raise ValueError(f"No model found for k={args.k}. Available: {available}")

    model_path = str(models[args.k])
    mode_str = "FULL" if args.full else "final-layer"
    logger.info(f"Using model: {model_path} (mode: {mode_str})")

    # Pre-download CIFAR-10 before spawning workers
    import torchvision
    torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True)
    torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True)

    # Compute val/test split indices (shared by both modes)
    n_test = 10000  # CIFAR-10 test set size
    perm = torch.randperm(n_test, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[: n_test // 2]
    test_idx = perm[n_test // 2 :]

    num_gpus = max(torch.cuda.device_count(), 1)
    logger.info(f"Sweeping {len(LAMBDAS)} lambda values across {num_gpus} GPUs")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    if args.full:
        # Full-model mode: each worker loads model + data independently
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
                    target=full_sweep_worker,
                    args=(
                        gpu_id % num_gpus,
                        l2_lambda,
                        args.k,
                        model_path,
                        args.data_path,
                        val_idx,
                        test_idx,
                        args.sgd_epochs,
                        args.sgd_lr,
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

    else:
        # Final-layer mode: extract features once, sweep on linear layer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = make_resnet18k(k=args.k, num_classes=10).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        original_layer_state = model.linear.state_dict()

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

        del model
        torch.cuda.empty_cache()

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
                        args.sgd_epochs,
                        args.sgd_lr,
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

    # Print summary table
    mode_label = "full-model" if args.full else "final-layer"
    print(f"\nL2 Calibrate Sweep for k={args.k} ({mode_label}, SGD, lr={args.sgd_lr}, epochs={args.sgd_epochs})")
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
    filename = f"sweep_k{args.k}_full.jsonl" if args.full else f"sweep_k{args.k}.jsonl"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
