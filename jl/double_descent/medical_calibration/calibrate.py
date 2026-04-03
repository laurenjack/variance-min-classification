"""Calibration methods for RETFound on APTOS-2019.

Loads the fine-tuned checkpoint, applies two calibration approaches:
1. Temperature scaling (fit T on validation set)
2. Final-layer fine-tuning with L-BFGS (fit on training set)

Then evaluates all three modes (uncalibrated, temp-scaled, fine-tuned)
on the test set.

Usage:
    python -m jl.double_descent.medical_calibration.calibrate \
        --checkpoint ./data/medical_calibration/checkpoint-best.pth \
        --data-path ./data/medical_calibration/APTOS2019 \
        --output-path ./output/medical_calibration
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.fine_tune_lib import fine_tune_final_layer
from jl.double_descent.medical_calibration.config import MedCalConfig

logger = logging.getLogger(__name__)


# --- Model loading ---


def load_retfound_model(checkpoint_path: str, config: MedCalConfig, device: torch.device) -> nn.Module:
    """Load fine-tuned RETFound checkpoint into a timm ViT-Large.

    Detects num_classes from checkpoint head.weight shape.
    """
    import timm

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint["model"]["head.weight"].shape[0]
    config.num_classes = num_classes

    model = timm.create_model(
        "vit_large_patch16_224",
        num_classes=num_classes,
        drop_path_rate=0.0,  # No drop path at inference
        global_pool="avg" if config.global_pool else "token",
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}), "
        f"{num_classes} classes, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )
    return model


# --- Feature extraction ---


def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from the frozen backbone (before the head layer).

    Uses timm's forward_features() which returns the pooled representation
    before the classifier head.

    Returns:
        (features [N, 1024], labels [N])
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            x = model.forward_features(images)
            # forward_head with pre_logits=True applies pool + fc_norm
            features = model.forward_head(x, pre_logits=True)
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


# --- Temperature scaling ---


def fit_temperature(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """Fit scalar temperature T on validation set via L-BFGS."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = temperature.item()
    logger.info(f"Fitted temperature: T={T:.4f}")
    return T


# --- Evaluation metrics ---


def compute_ece(
    confidences: torch.Tensor, correct: torch.Tensor, num_bins: int = 20
) -> float:
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = mask | (confidences == lo)
        n_bin = mask.sum().item()
        if n_bin > 0:
            avg_confidence = confidences[mask].mean().item()
            avg_accuracy = correct[mask].float().mean().item()
            ece += (n_bin / n) * abs(avg_accuracy - avg_confidence)
    return ece


def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Multi-class Brier score."""
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()


def evaluate_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """Compute all metrics from logits and labels.

    Returns:
        Dict with nll, accuracy, ece, brier, auroc, aupr.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    probs = F.softmax(logits, dim=-1)
    max_probs, predictions = probs.max(dim=1)
    correct = (predictions == labels)

    nll = F.cross_entropy(logits, labels).item()
    accuracy = correct.float().mean().item()
    ece = compute_ece(max_probs, correct)
    brier = compute_brier_score(probs, labels)

    probs_np = probs.numpy()
    labels_np = labels.numpy()

    try:
        auroc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0

    try:
        # One-hot for average_precision_score
        labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).numpy()
        aupr = average_precision_score(labels_one_hot, probs_np, average="macro")
    except ValueError:
        aupr = 0.0

    return {
        "nll": round(nll, 6),
        "accuracy": round(accuracy, 6),
        "ece": round(ece, 6),
        "brier": round(brier, 6),
        "auroc": round(auroc, 6),
        "aupr": round(aupr, 6),
    }


# --- Data loading ---


def build_loader(
    data_dir: str, split: str, config: MedCalConfig
) -> DataLoader:
    """Build a DataLoader for a split using torchvision ImageFolder."""
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ImageFolder(
        root=str(Path(data_dir) / split),
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    logger.info(f"Loaded {split}: {len(dataset)} images, {len(dataset.classes)} classes")
    logger.info(f"  Class mapping: {dataset.class_to_idx}")
    return loader


# --- Main ---


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Calibrate RETFound on APTOS-2019"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned RETFound checkpoint (.pth)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to APTOS2019 directory with train/val/test subdirs",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=1e-1,
        help="L2 regularization for final-layer fine-tuning",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="L-BFGS steps for final-layer fine-tuning",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep over lambda values, select best by val ECE, report on test",
    )
    args = parser.parse_args()

    config = MedCalConfig()
    config.l2_lambda = args.l2_lambda
    config.lbfgs_max_steps = args.max_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_retfound_model(args.checkpoint, config, device)

    # Load data splits
    train_loader = build_loader(args.data_path, "train", config)
    val_loader = build_loader(args.data_path, "val", config)
    test_loader = build_loader(args.data_path, "test", config)

    # Extract features once (shared by all modes)
    logger.info("Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    logger.info(
        f"Features: train={train_features.shape}, val={val_features.shape}, "
        f"test={test_features.shape}"
    )

    # Collect test logits for uncalibrated + temp scaling
    model.eval()
    test_logits_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            test_logits_list.append(logits.cpu())
    test_logits = torch.cat(test_logits_list)

    # === Uncalibrated ===
    logger.info("=== Uncalibrated evaluation ===")
    uncalibrated_metrics = evaluate_logits(test_logits, test_labels)
    logger.info(f"Uncalibrated: {uncalibrated_metrics}")

    torch.save(
        {"logits": test_logits, "labels": test_labels},
        out_dir / "test_logits.pt",
    )

    # === Temperature scaling (fit on validation) ===
    logger.info("=== Temperature scaling ===")
    T = fit_temperature(model, val_loader, device)
    ts_metrics = evaluate_logits(test_logits / T, test_labels)
    logger.info(f"Temperature-scaled (T={T:.4f}): {ts_metrics}")

    # === Final-layer fine-tuning ===
    if args.sweep:
        logger.info("=== Lambda sweep (selecting by val ECE) ===")
        lambdas = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 1.0, 2.0, 3.0, 5.0, 10.0]
        original_head_state = model.head.state_dict()

        sweep_results = []
        for lam in lambdas:
            linear = nn.Linear(train_features.shape[1], config.num_classes).to(device)
            linear.load_state_dict(original_head_state)

            fine_tune_final_layer(
                features=train_features,
                targets=train_labels,
                linear_layer=linear,
                l2_lambda=lam,
                max_steps=config.lbfgs_max_steps,
                device=device,
            )

            # Evaluate on val
            linear.eval()
            with torch.no_grad():
                val_logits = linear(val_features.to(device)).cpu()
            val_metrics = evaluate_logits(val_logits, val_labels)

            sweep_results.append((lam, val_metrics, linear.state_dict()))
            logger.info(f"  λ={lam:.0e}: val_ece={val_metrics['ece']:.4f}, val_nll={val_metrics['nll']:.4f}")

        # Select best by val ECE
        best_idx = min(range(len(sweep_results)), key=lambda i: sweep_results[i][1]["ece"])
        best_lambda, best_val_metrics, best_state = sweep_results[best_idx]
        logger.info(f"Best λ={best_lambda:.0e} (val ECE={best_val_metrics['ece']:.4f})")

        # Print sweep table
        print("\n" + "=" * 50)
        print(f"{'Lambda':<12} {'Val ECE':>10} {'Val NLL':>10} {'Val Acc':>10}")
        print("-" * 50)
        for lam, val_m, _ in sweep_results:
            marker = " <-- best" if lam == best_lambda else ""
            print(f"{lam:<12.0e} {val_m['ece']:>10.4f} {val_m['nll']:>10.4f} {val_m['accuracy']:>10.4f}{marker}")
        print("=" * 50)

        # Evaluate best on test
        linear = nn.Linear(train_features.shape[1], config.num_classes).to(device)
        linear.load_state_dict(best_state)
        linear.eval()
        with torch.no_grad():
            ft_logits = linear(test_features.to(device)).cpu()
        ft_metrics = evaluate_logits(ft_logits, test_labels)

        torch.save(best_state, out_dir / "calibrated_head.pt")
        config.l2_lambda = best_lambda

        # Save sweep details
        sweep_path = out_dir / "sweep_results.json"
        with open(sweep_path, "w") as f:
            json.dump(
                [{"l2_lambda": lam, **vm} for lam, vm, _ in sweep_results],
                f, indent=2,
            )

    else:
        logger.info("=== Final-layer fine-tuning ===")
        linear = nn.Linear(train_features.shape[1], config.num_classes).to(device)
        linear.load_state_dict(model.head.state_dict())

        metadata = fine_tune_final_layer(
            features=train_features,
            targets=train_labels,
            linear_layer=linear,
            l2_lambda=config.l2_lambda,
            max_steps=config.lbfgs_max_steps,
            device=device,
        )
        logger.info(f"Fine-tune metadata: {metadata}")

        torch.save(linear.state_dict(), out_dir / "calibrated_head.pt")

        linear.eval()
        with torch.no_grad():
            ft_logits = linear(test_features.to(device)).cpu()
        ft_metrics = evaluate_logits(ft_logits, test_labels)

    logger.info(f"Fine-tuned: {ft_metrics}")

    # === Save results ===
    results = {
        "uncalibrated": uncalibrated_metrics,
        "temperature_scaled": {**ts_metrics, "temperature": round(T, 6)},
        "fine_tuned": {**ft_metrics, "l2_lambda": config.l2_lambda, "max_steps": config.lbfgs_max_steps},
    }

    results_path = out_dir / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'NLL':>8} {'Acc':>8} {'ECE':>8} {'Brier':>8} {'AUROC':>8} {'AUPR':>8}")
    print("-" * 70)
    for method, metrics in results.items():
        print(
            f"{method:<20} {metrics['nll']:>8.4f} {metrics['accuracy']:>8.4f} "
            f"{metrics['ece']:>8.4f} {metrics['brier']:>8.4f} "
            f"{metrics['auroc']:>8.4f} {metrics['aupr']:>8.4f}"
        )
    print("=" * 70)

    # Print deltas
    print(f"\n{'Method':<20} {'ΔNLL':>8} {'ΔAcc':>8} {'ΔECE':>8} {'ΔBrier':>8} {'ΔAUROC':>8} {'ΔAUPR':>8}")
    print("-" * 70)
    base = uncalibrated_metrics
    for method in ["temperature_scaled", "fine_tuned"]:
        m = results[method]
        print(
            f"{method:<20} {m['nll'] - base['nll']:>+8.4f} {m['accuracy'] - base['accuracy']:>+8.4f} "
            f"{m['ece'] - base['ece']:>+8.4f} {m['brier'] - base['brier']:>+8.4f} "
            f"{m['auroc'] - base['auroc']:>+8.4f} {m['aupr'] - base['aupr']:>+8.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
